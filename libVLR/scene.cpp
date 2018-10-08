#include "scene.h"

namespace VLR {
    static std::string readTxtFile(const std::string& filepath) {
        std::ifstream ifs;
        ifs.open(filepath, std::ios::in);
        if (ifs.fail())
            return "";

        std::stringstream sstream;
        sstream << ifs.rdbuf();

        return std::string(sstream.str());
    };



    // ----------------------------------------------------------------
    // Shallow Hierarchy

    void SHGroup::addChild(SHTransform* transform) {
        TransformStatus status;
        SHGeometryGroup* descendant;
        status.hasGeometryDescendant = transform->hasGeometryDescendant(&descendant);
        m_transforms[transform] = status;
        if (status.hasGeometryDescendant) {
            optix::Transform optixTransform = transform->getOptiXObject();
            optixTransform->setChild(descendant->getOptiXObject());

            RTobject trChild;
            rtTransformGetChild(optixTransform->get(), &trChild);
            VLRAssert(trChild, "Transform must have a child.");

            m_optixGroup->addChild(optixTransform);
            m_optixAcceleration->markDirty();
            ++m_numValidTransforms;
        }
    }

    void SHGroup::removeChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        const TransformStatus status = m_transforms.at(transform);
        m_transforms.erase(transform);
        if (status.hasGeometryDescendant) {
            m_optixGroup->removeChild(transform->getOptiXObject());
            m_optixAcceleration->markDirty();
            --m_numValidTransforms;
        }
    }

    void SHGroup::updateChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);
        SHGeometryGroup* descendant;
        optix::Transform optixTransform = transform->getOptiXObject();
        if (status.hasGeometryDescendant) {
            if (!transform->hasGeometryDescendant()) {
                m_optixGroup->removeChild(optixTransform);
                m_optixAcceleration->markDirty();
                status.hasGeometryDescendant = false;
                --m_numValidTransforms;
            }
        }
        else {
            if (transform->hasGeometryDescendant(&descendant)) {
                optixTransform->setChild(descendant->getOptiXObject());

                RTobject trChild;
                rtTransformGetChild(optixTransform->get(), &trChild);
                VLRAssert(trChild, "Transform must have a child.");

                m_optixGroup->addChild(optixTransform);
                m_optixAcceleration->markDirty();
                status.hasGeometryDescendant = true;
                ++m_numValidTransforms;
            }
        }
    }

    void SHGroup::addChild(SHGeometryGroup* geomGroup) {
        m_geomGroups.insert(geomGroup);

        optix::GeometryGroup optixGeomGroup = geomGroup->getOptiXObject();

        m_optixGroup->addChild(optixGeomGroup);
        m_optixAcceleration->markDirty();
    }

    void SHGroup::removeChild(SHGeometryGroup* geomGroup) {
        m_geomGroups.erase(geomGroup);

        optix::GeometryGroup optixGeomGroup = geomGroup->getOptiXObject();

        m_optixGroup->removeChild(optixGeomGroup);
        m_optixAcceleration->markDirty();
    }

    void SHGroup::printOptiXHierarchy() {
        std::stack<RTobject> stackRTObjects;
        std::stack<RTobjecttype> stackRTObjectTypes;

        std::set<RTgroup> groupList;
        std::set<RTtransform> transformList;
        std::set<RTgeometrygroup> geometryGroupList;
        std::set<RTgeometryinstance> geometryInstanceList;

        stackRTObjects.push(m_optixGroup->get());
        stackRTObjectTypes.push(RT_OBJECTTYPE_GROUP);
        while (!stackRTObjects.empty()) {
            RTobject object = stackRTObjects.top();
            RTobjecttype objType = stackRTObjectTypes.top();
            stackRTObjects.pop();
            stackRTObjectTypes.pop();

            VLRDebugPrintf("0x%p: ", object);

            switch (objType) {
            case RT_OBJECTTYPE_GROUP: {
                auto group = (RTgroup)object;
                VLRDebugPrintf("Group\n");

                groupList.insert(group);

                uint32_t numChildren;
                rtGroupGetChildCount(group, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTobject childObject = nullptr;
                    RTobjecttype childObjType;
                    rtGroupGetChild(group, i, &childObject);
                    rtGroupGetChildType(group, i, &childObjType);

                    VLRDebugPrintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(childObjType);
                }

                break;
            }
            case RT_OBJECTTYPE_TRANSFORM: {
                auto transform = (RTtransform)object;
                VLRDebugPrintf("Transform\n");

                transformList.insert(transform);

                RTobject childObject = nullptr;
                RTobjecttype childObjType;
                rtTransformGetChild(transform, &childObject);
                rtTransformGetChildType(transform, &childObjType);

                VLRDebugPrintf("- 0x%p\n", childObject);

                stackRTObjects.push(childObject);
                stackRTObjectTypes.push(childObjType);

                break;
            }
            case RT_OBJECTTYPE_SELECTOR: {
                VLRAssert_NotImplemented();
                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_GROUP: {
                auto geometryGroup = (RTgeometrygroup)object;
                VLRDebugPrintf("GeometryGroup\n");

                geometryGroupList.insert(geometryGroup);

                uint32_t numChildren;
                rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTgeometryinstance childObject = nullptr;
                    rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                    VLRDebugPrintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(RT_OBJECTTYPE_GEOMETRY_INSTANCE);
                }

                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_INSTANCE: {
                auto geometryInstance = (RTgeometryinstance)object;
                VLRDebugPrintf("GeometryInstance\n");

                geometryInstanceList.insert(geometryInstance);

                break;
            }
            default:
                VLRDebugPrintf("\n");
                VLRAssert_ShouldNotBeCalled();
                break;
            }

            VLRDebugPrintf("\n");
        }



        VLRDebugPrintf("Groups:\n");
        for (auto group : groupList) {
            VLRDebugPrintf("  0x%p:\n", group);
            uint32_t numChildren;
            rtGroupGetChildCount(group, &numChildren);
            RTacceleration acceleration;
            rtGroupGetAcceleration(group, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            VLRDebugPrintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTobject childObject = nullptr;
                rtGroupGetChild(group, i, &childObject);

                VLRDebugPrintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        VLRDebugPrintf("Transforms:\n");
        for (auto transform : transformList) {
            VLRDebugPrintf("  0x%p:\n", transform);
            RTobject childObject = nullptr;
            rtTransformGetChild(transform, &childObject);
            float mat[16];
            float invMat[16];
            rtTransformGetMatrix(transform, true, mat, invMat);
            VLRDebugPrintf("    Matrix\n");
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
            VLRDebugPrintf("    Inverse Matrix\n");
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[0], invMat[4], invMat[8], invMat[12]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[1], invMat[5], invMat[9], invMat[13]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[2], invMat[6], invMat[10], invMat[14]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[3], invMat[7], invMat[11], invMat[15]);

            VLRDebugPrintf("  - 0x%p\n", childObject);
        }

        VLRDebugPrintf("GeometryGroups:\n");
        for (auto geometryGroup : geometryGroupList) {
            VLRDebugPrintf("  0x%p:\n", geometryGroup);
            uint32_t numChildren;
            rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
            RTacceleration acceleration;
            rtGeometryGroupGetAcceleration(geometryGroup, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            VLRDebugPrintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTgeometryinstance childObject = nullptr;
                rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                VLRDebugPrintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        VLRDebugPrintf("GeometryInstances:\n");
        for (auto geometryInstance : geometryInstanceList) {
            VLRDebugPrintf("  0x%p:\n", geometryInstance);
        }
    }



    void SHTransform::resolveTransform() {
        int32_t stackIdx = 0;
        const SHTransform* stack[5];
        std::fill_n(stack, lengthof(stack), nullptr);
        const SHTransform* nextSHTr = m_childIsTransform ? m_childTransform : nullptr;
        while (nextSHTr) {
            stack[stackIdx++] = nextSHTr;
            nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
        }

        StaticTransform res;
        std::string concatenatedName = "";
        --stackIdx;
        while (stackIdx >= 0) {
            const SHTransform* shtr = stack[stackIdx--];
            res = shtr->m_transform * res;
            concatenatedName = "-" + shtr->getName() + concatenatedName;
        }
        res = m_transform * res;
        concatenatedName = m_name + concatenatedName;

        float mat[16], invMat[16];
        res.getArrays(mat, invMat);
        m_optixTransform->setMatrix(true, mat, invMat);

        //if (true/*m_parent*/) {
        //    VLRDebugPrintf("%s:\n", concatenatedName.c_str());
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
        //    VLRDebugPrintf("\n");
        //}
    }

    void SHTransform::setTransform(const StaticTransform &transform) {
        m_transform = transform;
        resolveTransform();
    }

    void SHTransform::update() {
        resolveTransform();
    }

    bool SHTransform::isStatic() const {
        // TODO: implement
        return true;
    }

    StaticTransform SHTransform::getStaticTransform() const {
        if (isStatic()) {
            float mat[16], invMat[16];
            m_optixTransform->getMatrix(true, mat, invMat);
            return StaticTransform(Matrix4x4(mat));
        }
        else {
            return StaticTransform();
        }
    }

    void SHTransform::setChild(SHGeometryGroup* geomGroup) {
        VLRAssert(!m_childIsTransform, "Transform which doesn't have a child transform can have a geometry group as a child.");
        m_childGeometryGroup = geomGroup;
    }

    bool SHTransform::hasGeometryDescendant(SHGeometryGroup** descendant) const {
        if (descendant)
            *descendant = nullptr;

        const SHTransform* nextSHTr = this;
        while (nextSHTr) {
            if (!nextSHTr->m_childIsTransform && nextSHTr->m_childGeometryGroup != nullptr) {
                if (descendant)
                    *descendant = nextSHTr->m_childGeometryGroup;
                return true;
            }
            else {
                nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
            }
        }

        return false;
    }



    void SHGeometryGroup::addGeometryInstance(const SHGeometryInstance* instance) {
        m_instances.insert(instance);
        m_optixGeometryGroup->addChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGeometryGroup::removeGeometryInstance(const SHGeometryInstance* instance) {
        m_instances.erase(instance);
        m_optixGeometryGroup->removeChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    // static
    void SurfaceNode::initialize(Context &context) {
        TriangleMeshSurfaceNode::initialize(context);
        InfiniteSphereSurfaceNode::initialize(context);
    }

    // static
    void SurfaceNode::finalize(Context &context) {
        InfiniteSphereSurfaceNode::finalize(context);
        TriangleMeshSurfaceNode::finalize(context);
    }

    void SurfaceNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);
    }

    void SurfaceNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);
    }



    std::map<uint32_t, TriangleMeshSurfaceNode::OptiXProgramSet> TriangleMeshSurfaceNode::OptiXProgramSets;

    // static
    void TriangleMeshSurfaceNode::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/triangle_intersection.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.programIntersectTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::intersectTriangle");
        programSet.programCalcBBoxForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::calcBBoxForTriangle");

        programSet.callableProgramDecodeHitPointForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::decodeHitPointForTriangle");
        programSet.callableProgramDecodeTexCoordForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::decodeTexCoordForTriangle");

        programSet.callableProgramSampleTriangleMesh = optixContext->createProgramFromPTXString(ptx, "VLR::sampleTriangleMesh");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TriangleMeshSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleTriangleMesh->destroy();

        programSet.callableProgramDecodeTexCoordForTriangle->destroy();
        programSet.callableProgramDecodeHitPointForTriangle->destroy();

        programSet.programCalcBBoxForTriangle->destroy();
        programSet.programIntersectTriangle->destroy();

        OptiXProgramSets.erase(context.getID());
    }

    TriangleMeshSurfaceNode::TriangleMeshSurfaceNode(Context &context, const std::string &name) : SurfaceNode(context, name) {
    }

    TriangleMeshSurfaceNode::~TriangleMeshSurfaceNode() {
        for (auto it = m_shGeometryInstances.crbegin(); it != m_shGeometryInstances.crend(); ++it)
            delete *it;
        m_shGeometryInstances.clear();

        for (auto it = m_optixGeometries.begin(); it != m_optixGeometries.end(); ++it) {
            OptiXGeometry &geom = *it;
            geom.primDist.finalize(m_context);
            geom.optixIndexBuffer->destroy();
            geom.optixGeometry->destroy();
        }
        m_optixVertexBuffer->destroy();
    }

    void TriangleMeshSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
    }

    void TriangleMeshSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 削除した親に対してジオメトリインスタンスの削除を行わせる。
        std::set<SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryRemoved, delta);
    }

    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        optix::Context optixContext = m_context.getOptiXContext();
        m_optixVertexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_vertices.size());
        m_optixVertexBuffer->setElementSize(sizeof(Shared::Vertex));
        {
            auto dstVertices = (Shared::Vertex*)m_optixVertexBuffer->map();
            static_assert(sizeof(Vertex) == sizeof(Shared::Vertex), "These two types must match in size (and structure).");
            std::copy_n((Shared::Vertex*)m_vertices.data(), m_vertices.size(), dstVertices);
            m_optixVertexBuffer->unmap();
        }

        // TODO: 頂点情報更新時の処理。(IndexBufferとの整合性など)
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(std::vector<uint32_t> &&indices, SurfaceMaterial* material, Float4Texture* texNormalAlpha) {
        optix::Context optixContext = m_context.getOptiXContext();
        const OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        OptiXGeometry geom;
        CompensatedSum<float> sumImportances(0.0f);
        {
            geom.indices = std::move(indices);
            uint32_t numTriangles = (uint32_t)geom.indices.size() / 3;

            geom.optixGeometry = optixContext->createGeometry();
            geom.optixGeometry->setPrimitiveCount(numTriangles);
            geom.optixGeometry->setIntersectionProgram(progSet.programIntersectTriangle);
            geom.optixGeometry->setBoundingBoxProgram(progSet.programCalcBBoxForTriangle);

            geom.optixIndexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, numTriangles);
            geom.optixIndexBuffer->setElementSize(sizeof(Shared::Triangle));

            std::vector<float> areas;
            areas.resize(numTriangles);
            {
                auto dstTriangles = (Shared::Triangle*)geom.optixIndexBuffer->map();
                for (auto i = 0; i < numTriangles; ++i) {
                    uint32_t i0 = geom.indices[3 * i + 0];
                    uint32_t i1 = geom.indices[3 * i + 1];
                    uint32_t i2 = geom.indices[3 * i + 2];

                    dstTriangles[i] = Shared::Triangle{ i0, i1, i2 };

                    const Vertex (&v)[3] = { m_vertices[i0], m_vertices[i1], m_vertices[i2] };
                    areas[i] = std::max<float>(0.0f, 0.5f * cross(v[1].position - v[0].position, v[2].position - v[0].position).length());
                    sumImportances += areas[i];
                }
                geom.optixIndexBuffer->unmap();
            }

            if (material->isEmitting())
                geom.primDist.initialize(m_context, areas.data(), areas.size());
        }
        m_optixGeometries.push_back(geom);

        m_materials.push_back(material);
        m_texNormalAlphas.push_back(texNormalAlpha);

        Shared::SurfaceLightDescriptor lightDesc;
        lightDesc.body.asMeshLight.vertexBuffer = m_optixVertexBuffer->getId();
        lightDesc.body.asMeshLight.triangleBuffer = geom.optixIndexBuffer->getId();
        geom.primDist.getInternalType(&lightDesc.body.asMeshLight.primDistribution);
        lightDesc.body.asMeshLight.materialIndex = material->getMaterialIndex();
        lightDesc.sampleFunc = progSet.callableProgramSampleTriangleMesh->getId();
        lightDesc.importance = material->isEmitting() ? 1.0f : 0.0f; // TODO:

        SHGeometryInstance* geomInst = new SHGeometryInstance(m_context, lightDesc);
        {
            optix::GeometryInstance optixGeomInst = geomInst->getOptiXObject();
            optixGeomInst->setGeometry(geom.optixGeometry);
            optixGeomInst->setMaterialCount(1);
            optixGeomInst["VLR::pv_vertexBuffer"]->set(m_optixVertexBuffer);
            optixGeomInst["VLR::pv_triangleBuffer"]->set(geom.optixIndexBuffer);
            optixGeomInst["VLR::pv_sumImportances"]->setFloat(sumImportances.result);
            optixGeomInst["VLR::pv_progDecodeTexCoord"]->set(progSet.callableProgramDecodeTexCoordForTriangle);
            optixGeomInst["VLR::pv_progDecodeHitPoint"]->set(progSet.callableProgramDecodeHitPointForTriangle);
            if (texNormalAlpha) {
                optixGeomInst->setMaterial(0, m_context.getOptiXMaterialWithAlpha());
                optixGeomInst["VLR::pv_texNormalAlpha"]->set(texNormalAlpha->getOptiXObject());
                optixGeomInst["VLR::pv_progFetchAlpha"]->set(m_context.getOptiXCallableProgramFetchAlpha());
                optixGeomInst["VLR::pv_progFetchNormal"]->set(m_context.getOptiXCallableProgramFetchNormal());
            }
            else {
                optixGeomInst->setMaterial(0, m_context.getOptiXMaterialDefault());
                optixGeomInst["VLR::pv_progFetchAlpha"]->set(m_context.getOptiXCallableProgramNullFetchAlpha());
                optixGeomInst["VLR::pv_progFetchNormal"]->set(m_context.getOptiXCallableProgramNullFetchNormal());
            }
            uint32_t matIndex = material->getMaterialIndex();
            optixGeomInst["VLR::pv_materialIndex"]->setUserData(sizeof(matIndex), &matIndex);
            optixGeomInst["VLR::pv_importance"]->setFloat(lightDesc.importance);
        }
        m_shGeometryInstances.push_back(geomInst);

        // JP: 親にジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        delta.insert(geomInst);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
        }
    }



    std::map<uint32_t, InfiniteSphereSurfaceNode::OptiXProgramSet> InfiniteSphereSurfaceNode::OptiXProgramSets;

    // static
    void InfiniteSphereSurfaceNode::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/infinite_sphere_intersection.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.programIntersectInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::intersectInfiniteSphere");
        programSet.programCalcBBoxForInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::calcBBoxForInfiniteSphere");

        programSet.callableProgramDecodeHitPointForInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::decodeHitPointForInfiniteSphere");
        programSet.callableProgramDecodeTexCoordForInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::decodeTexCoordForInfiniteSphere");

        programSet.callableProgramSampleInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::sampleInfiniteSphere");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void InfiniteSphereSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleInfiniteSphere->destroy();

        programSet.callableProgramDecodeTexCoordForInfiniteSphere->destroy();
        programSet.callableProgramDecodeHitPointForInfiniteSphere->destroy();

        programSet.programCalcBBoxForInfiniteSphere->destroy();
        programSet.programIntersectInfiniteSphere->destroy();

        OptiXProgramSets.erase(context.getID());
    }

    InfiniteSphereSurfaceNode::InfiniteSphereSurfaceNode(Context &context, const std::string &name, SurfaceMaterial* material) : 
        SurfaceNode(context, name), m_material(material) {
        optix::Context optixContext = m_context.getOptiXContext();
        const OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        m_optixGeometry = optixContext->createGeometry();
        m_optixGeometry->setPrimitiveCount(1);
        m_optixGeometry->setIntersectionProgram(progSet.programIntersectInfiniteSphere);
        m_optixGeometry->setBoundingBoxProgram(progSet.programCalcBBoxForInfiniteSphere);

        Shared::SurfaceLightDescriptor lightDesc;
        lightDesc.body.asEnvironmentLight.materialIndex = m_material->getMaterialIndex();
        //geom.primDist.getInternalType(&lightDesc.body.asEnvironmentLight.importanceMap);
        lightDesc.sampleFunc = progSet.callableProgramSampleInfiniteSphere->getId();
        lightDesc.importance = material->isEmitting() ? 1.0f : 0.0f; // TODO:

        m_shGeometryInstance = new SHGeometryInstance(m_context, lightDesc);
        {
            optix::GeometryInstance optixGeomInst = m_shGeometryInstance->getOptiXObject();
            optixGeomInst->setGeometry(m_optixGeometry);
            optixGeomInst->setMaterialCount(1);
            optixGeomInst->setMaterial(0, m_context.getOptiXMaterialDefault());
            optixGeomInst["VLR::pv_progDecodeTexCoord"]->set(progSet.callableProgramDecodeTexCoordForInfiniteSphere);
            optixGeomInst["VLR::pv_progDecodeHitPoint"]->set(progSet.callableProgramDecodeHitPointForInfiniteSphere);
            optixGeomInst["VLR::pv_progFetchAlpha"]->set(m_context.getOptiXCallableProgramNullFetchAlpha());
            optixGeomInst["VLR::pv_progFetchNormal"]->set(m_context.getOptiXCallableProgramNullFetchNormal());
            uint32_t matIndex = material->getMaterialIndex();
            optixGeomInst["VLR::pv_materialIndex"]->setUserData(sizeof(matIndex), &matIndex);
            optixGeomInst["VLR::pv_importance"]->setFloat(lightDesc.importance);
        }
    }

    InfiniteSphereSurfaceNode::~InfiniteSphereSurfaceNode() {
        delete m_shGeometryInstance;

        m_optixGeometry->destroy();
    }

    void InfiniteSphereSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        delta.insert(m_shGeometryInstance);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
    }

    void InfiniteSphereSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 削除した親に対してジオメトリインスタンスの削除を行わせる。
        std::set<SHGeometryInstance*> delta;
        delta.insert(m_shGeometryInstance);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryRemoved, delta);
    }



    ParentNode::ParentNode(Context &context, const std::string &name, const Transform* localToWorld) :
        Node(context, name), m_localToWorld(localToWorld), m_shGeomGroup(context) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: 
        if (m_localToWorld->isStatic()) {
            StaticTransform* tr = (StaticTransform*)m_localToWorld;
            m_shTransforms[nullptr] = new SHTransform(name, m_context, *tr, nullptr);
        }
        else {
            VLRAssert_NotImplemented();
        }
    }

    ParentNode::~ParentNode() {
        for (auto it = m_shTransforms.crbegin(); it != m_shTransforms.crend(); ++it)
            delete it->second;
        m_shTransforms.clear();
    }

    void ParentNode::setName(const std::string &name) {
        Node::setName(name);
        for (auto it = m_shTransforms.begin(); it != m_shTransforms.end(); ++it)
            it->second->setName(name);
    }

    void ParentNode::addChild(InternalNode* child) {
        m_children.insert(child);
        child->addParent(this);
    }

    void ParentNode::addChild(SurfaceNode* child) {
        m_children.insert(child);
        child->addParent(this);
    }

    void ParentNode::removeChild(InternalNode* child) {
        m_children.erase(child);
        child->removeParent(this);
    }

    void ParentNode::removeChild(SurfaceNode* child) {
        m_children.erase(child);
        child->removeParent(this);
    }

    void ParentNode::setTransform(const Transform* localToWorld) {
        m_localToWorld = localToWorld;

        // JP: 管理中のSHTransformを更新する。
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                StaticTransform* tr = (StaticTransform*)m_localToWorld;
                SHTransform* shtr = it->second;
                shtr->setTransform(*tr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
    }



    void InternalNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) {
        switch (eventType) {
        case UpdateEvent::TransformAdded: {
            // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
            //     子のSHTransformをキーとして辞書に保存する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_localToWorld->isStatic()) {
                    StaticTransform* tr = (StaticTransform*)m_localToWorld;
                    SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                    m_shTransforms[*it] = shtr;
                    delta.insert(shtr);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 親に自分が保持するSHTransformが増えたことを通知(増分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            break;
        }
        case UpdateEvent::TransformRemoved: {
            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを削除。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shTransforms.erase(*it);
                delta.insert(shtr);
            }

            // JP: 親に自分が保持するSHTransformが減ったことを通知(減分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            for (auto it = delta.cbegin(); it != delta.cend(); ++it)
                delete *it;

            break;
        }
        case UpdateEvent::TransformUpdated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
                delta.insert(shtr);
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 親に自分が保持するSHTransformが更新されたことを通知(更新分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            break;
        }
        case UpdateEvent::GeometryAdded:
        case UpdateEvent::GeometryRemoved: {
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                delta.insert(shtr);
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 親に自分が保持するSHTransformが更新されたことを通知(更新分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    void InternalNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) {
        switch (eventType) {
        case UpdateEvent::GeometryAdded: {
            // JP: このInternalNodeが管理するGeometryGroupにGeometryInstanceを追加する。
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                m_shGeomGroup.addGeometryInstance(*it);
                geomInstDelta.push_back(TransformAndGeometryInstance{ selfTransform, *it });
            }

            // JP: このInternalNodeのTransformにGeometryGroupをセットする。
            //     このInternalNodeのTransformを末尾に持つTransformに変更があったことを親に知らせる。
            if (m_shGeomGroup.getNumInstances() > 0) {
                selfTransform->setChild(&m_shGeomGroup);

                std::set<SHTransform*> delta;
                delta.insert(selfTransform);
                for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                    ParentNode* parent = *it;
                    parent->childUpdateEvent(eventType, delta, geomInstDelta);
                }
            }

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: 
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                m_shGeomGroup.removeGeometryInstance(*it);
                geomInstDelta.push_back(TransformAndGeometryInstance{ selfTransform, *it });
            }

            if (m_shGeomGroup.getNumInstances() == 0) {
                selfTransform->setChild(nullptr);

                std::set<SHTransform*> delta;
                delta.insert(selfTransform);
                for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                    ParentNode* parent = *it;
                    parent->childUpdateEvent(eventType, delta, geomInstDelta);
                }
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    InternalNode::InternalNode(Context &context, const std::string &name, const Transform* localToWorld) :
        ParentNode(context, name, localToWorld) {
    }

    void InternalNode::setTransform(const Transform* localToWorld) {
        ParentNode::setTransform(localToWorld);

        // JP: 親に変形情報が更新されたことを通知する。
        std::set<SHTransform*> delta;
        std::vector<TransformAndGeometryInstance> geomInstDelta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            delta.insert(it->second);

            SHGeometryGroup* shGeomGroup = nullptr;
            if (it->second->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumInstances(); ++i) {
                    geomInstDelta.push_back(TransformAndGeometryInstance{ it->second, shGeomGroup->getGeometryInstanceAt(i) });
                }
            }
        }
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(UpdateEvent::TransformUpdated, delta, geomInstDelta);
        }
    }

    void InternalNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);

        // JP: 追加した親に対して変形情報の追加を行わせる。
        std::set<SHTransform*> delta;
        std::vector<TransformAndGeometryInstance> geomInstDelta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            delta.insert(it->second);

            SHGeometryGroup* shGeomGroup = nullptr;
            if (it->second->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumInstances(); ++i) {
                    geomInstDelta.push_back(TransformAndGeometryInstance{ it->second, shGeomGroup->getGeometryInstanceAt(i) });
                }
            }
        }
        parent->childUpdateEvent(UpdateEvent::TransformAdded, delta, geomInstDelta);
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 削除した親に対して変形情報の削除を行わせる。
        std::set<SHTransform*> delta;
        std::vector<TransformAndGeometryInstance> geomInstDelta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            delta.insert(it->second);

            SHGeometryGroup* shGeomGroup = nullptr;
            if (it->second->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumInstances(); ++i) {
                    geomInstDelta.push_back(TransformAndGeometryInstance{ it->second, shGeomGroup->getGeometryInstanceAt(i) });
                }
            }
        }
        parent->childUpdateEvent(UpdateEvent::TransformRemoved, delta, geomInstDelta);
    }



    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) {
        switch (eventType) {
        case UpdateEvent::TransformAdded: {
            // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
            //     子のSHTransformをキーとして辞書に保存する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_localToWorld->isStatic()) {
                    StaticTransform* tr = (StaticTransform*)m_localToWorld;
                    SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                    m_shTransforms[*it] = shtr;
                    delta.insert(shtr);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    VLRDebugPrintf("Surface light cannot be instanced.");
                    VLRAssert_ShouldNotBeCalled();
                }
                else {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[it->geomInstance];
                    it->geomInstance->getSurfaceLightDescriptor(&lightDesc);
                    if (it->transform->isStatic()) {
                        StaticTransform tr = it->transform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.asMeshLight.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
            }
            m_surfaceLightsAreSetup = false;

            // JP: SHGroupにもSHTransformを追加する。
            for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
                SHTransform* shtr = *it;
                m_shGroup.addChild(shtr);
            }

            break;
        }
        case UpdateEvent::TransformRemoved: {
            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    m_surfaceLights.erase(it->geomInstance);
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            // JP: 子InternalNodeが持つSHTransformがつながっているSHTransformを削除。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shTransforms.erase(*it);
                delta.insert(shtr);
            }

            // JP: SHGroupからもSHTransformを削除する。
            for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
                SHTransform* shtr = *it;
                m_shGroup.removeChild(shtr);
            }

            for (auto it = delta.cbegin(); it != delta.cend(); ++it)
                delete *it;

            break;
        }
        case UpdateEvent::TransformUpdated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights.at(it->geomInstance);
                    if (it->transform->isStatic()) {
                        StaticTransform tr = it->transform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.asMeshLight.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        case UpdateEvent::GeometryAdded: {
            // JP: SHGroupに対してSHTransformの末尾のジオメトリ状態に変化があったことを通知する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shGroup.updateChild(shtr);
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    VLRDebugPrintf("Surface light cannot be instanced.");
                    VLRAssert_ShouldNotBeCalled();
                }
                else {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[it->geomInstance];
                    it->geomInstance->getSurfaceLightDescriptor(&lightDesc);
                    if (it->transform->isStatic()) {
                        StaticTransform tr = it->transform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.asMeshLight.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: SHGroupに対してSHTransformの末尾のジオメトリ状態に変化があったことを通知する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shGroup.updateChild(shtr);
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    m_surfaceLights.erase(it->geomInstance);
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) {
        switch (eventType) {
        case UpdateEvent::GeometryAdded: {
            // JP: 
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.addGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() > 0) {
                selfTransform->setChild(&m_shGeomGroup);
                m_shGroup.updateChild(selfTransform);
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_surfaceLights.count(*it)) {
                    VLRDebugPrintf("Surface light cannot be instanced.");
                    VLRAssert_ShouldNotBeCalled();
                }
                else {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[*it];
                    (*it)->getSurfaceLightDescriptor(&lightDesc);
                    if (selfTransform->isStatic()) {
                        StaticTransform tr = selfTransform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.asMeshLight.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: 
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.removeGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() == 0) {
                selfTransform->setChild(nullptr);
                m_shGroup.updateChild(selfTransform);
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_surfaceLights.count(*it)) {
                    m_surfaceLights.erase(*it);
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    RootNode::RootNode(Context &context, const Transform* localToWorld) :
        ParentNode(context, "Root", localToWorld), m_shGroup(context), m_surfaceLightsAreSetup(false) {
        SHTransform* shtr = m_shTransforms[0];
        m_shGroup.addChild(shtr);

        // DELETE ME
        //float value[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        //m_envTex = new ConstantFloat4Texture(context, value);
        //m_envMat = new MatteSurfaceMaterial(context, m_envTex);
        //m_envSphere = new InfiniteSphereSurfaceNode(context, "Env", m_envMat);

        //addChild(m_envSphere);
    }

    RootNode::~RootNode() {
        if (m_surfaceLightsAreSetup) {
            m_surfaceLightImpDist.finalize(m_context);

            m_optixSurfaceLightDescriptorBuffer->destroy();

            m_surfaceLightsAreSetup = false;
        }

        // DELETE ME
        //removeChild(m_envSphere);

        //delete m_envSphere;
        //delete m_envMat;
        //delete m_envTex;
    }

    void RootNode::set() {
        optix::Context optixContext = m_context.getOptiXContext();

        optixContext["VLR::pv_topGroup"]->set(m_shGroup.getOptiXObject());

        if (!m_surfaceLightsAreSetup) {
            if (m_optixSurfaceLightDescriptorBuffer)
                m_optixSurfaceLightDescriptorBuffer->destroy();
            m_optixSurfaceLightDescriptorBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_surfaceLights.size());
            m_optixSurfaceLightDescriptorBuffer->setElementSize(sizeof(Shared::SurfaceLightDescriptor));

            std::vector<float> importances;
            importances.resize(m_surfaceLights.size());

            {
                Shared::SurfaceLightDescriptor* descs = (Shared::SurfaceLightDescriptor*)m_optixSurfaceLightDescriptorBuffer->map();
                for (auto it = m_surfaceLights.cbegin(); it != m_surfaceLights.cend(); ++it) {
                    uint32_t index = std::distance(m_surfaceLights.cbegin(), it);
                    const Shared::SurfaceLightDescriptor &lightDesc = it->second;
                    descs[index] = lightDesc;
                    importances[index] = lightDesc.importance;
                }
                m_optixSurfaceLightDescriptorBuffer->unmap();
            }

            m_surfaceLightImpDist.finalize(m_context);
            m_surfaceLightImpDist.initialize(m_context, importances.data(), importances.size());

            m_surfaceLightsAreSetup = true;
        }

        Shared::DiscreteDistribution1D lightImpDist;
        m_surfaceLightImpDist.getInternalType(&lightImpDist);
        optixContext["VLR::pv_lightImpDist"]->setUserData(sizeof(lightImpDist), &lightImpDist);

        optixContext["VLR::pv_surfaceLightDescriptorBuffer"]->set(m_optixSurfaceLightDescriptorBuffer);
    }



    Scene::Scene(Context &context, const Transform* localToWorld) : 
    Object(context), m_rootNode(context, localToWorld), m_matEnv(nullptr) {
        std::string ptx = readTxtFile("resources/ptxes/infinite_sphere_intersection.ptx");

        optix::Context optixContext = context.getOptiXContext();

        m_callableProgramSampleInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::sampleInfiniteSphere");
    }

    Scene::~Scene() {
        m_callableProgramSampleInfiniteSphere->destroy();
    }

    void Scene::setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv) {
        m_matEnv = matEnv;
    }

    void Scene::set() {
        m_rootNode.set();

        optix::Context optixContext = m_context.getOptiXContext();

        Shared::SurfaceLightDescriptor envLight;
        envLight.importance = 0.0f;
        if (m_matEnv) {
            m_matEnv->getImportanceMap().getInternalType(&envLight.body.asEnvironmentLight.importanceMap);
            envLight.body.asEnvironmentLight.materialIndex = m_matEnv->getMaterialIndex();
            envLight.importance = 1.0f;
            envLight.sampleFunc = m_callableProgramSampleInfiniteSphere->getId();
        }

        optixContext["VLR::pv_envLightDescriptor"]->setUserData(sizeof(envLight), &envLight);
    }



    // static
    void Camera::initialize(Context &context) {
        PerspectiveCamera::initialize(context);
        EquirectangularCamera::initialize(context);
    }

    // static
    void Camera::finalize(Context &context) {
        EquirectangularCamera::finalize(context);
        PerspectiveCamera::finalize(context);
    }



    std::map<uint32_t, PerspectiveCamera::OptiXProgramSet> PerspectiveCamera::OptiXProgramSets;

    // static
    void PerspectiveCamera::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/cameras.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.callableProgramSampleLensPosition = optixContext->createProgramFromPTXString(ptx, "VLR::PerspectiveCamera_sampleLensPosition");
        programSet.callableProgramSampleIDF = optixContext->createProgramFromPTXString(ptx, "VLR::PerspectiveCamera_sampleIDF");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void PerspectiveCamera::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleIDF->destroy();
        programSet.callableProgramSampleLensPosition->destroy();

        OptiXProgramSets.erase(context.getID());
    }

    PerspectiveCamera::PerspectiveCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                                         float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) :
        Camera(context), m_data(sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist) {
        m_data.position = position;
        m_data.orientation = orientation;
    }

    void PerspectiveCamera::set() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_perspectiveCamera"]->setUserData(sizeof(Shared::PerspectiveCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }



    std::map<uint32_t, EquirectangularCamera::OptiXProgramSet> EquirectangularCamera::OptiXProgramSets;

    // static
    void EquirectangularCamera::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/cameras.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.callableProgramSampleLensPosition = optixContext->createProgramFromPTXString(ptx, "VLR::EquirectangularCamera_sampleLensPosition");
        programSet.callableProgramSampleIDF = optixContext->createProgramFromPTXString(ptx, "VLR::EquirectangularCamera_sampleIDF");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EquirectangularCamera::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleIDF->destroy();
        programSet.callableProgramSampleLensPosition->destroy();

        OptiXProgramSets.erase(context.getID());
    }

    EquirectangularCamera::EquirectangularCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                                                 float sensitivity, float phiAngle, float thetaAngle) :
        Camera(context), m_data(sensitivity, phiAngle, thetaAngle) {
        m_data.position = position;
        m_data.orientation = orientation;
    }

    void EquirectangularCamera::set() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_equirectangularCamera"]->setUserData(sizeof(Shared::EquirectangularCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }
}
