#include "scene.h"

namespace VLR {
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
                status.hasGeometryDescendant = true;
                ++m_numValidTransforms;
            }
        }
        m_optixAcceleration->markDirty();
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

            vlrprintf("0x%p: ", object);

            switch (objType) {
            case RT_OBJECTTYPE_GROUP: {
                auto group = (RTgroup)object;
                vlrprintf("Group\n");

                groupList.insert(group);

                uint32_t numChildren;
                rtGroupGetChildCount(group, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTobject childObject = nullptr;
                    RTobjecttype childObjType;
                    rtGroupGetChild(group, i, &childObject);
                    rtGroupGetChildType(group, i, &childObjType);

                    vlrprintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(childObjType);
                }

                break;
            }
            case RT_OBJECTTYPE_TRANSFORM: {
                auto transform = (RTtransform)object;
                vlrprintf("Transform\n");

                transformList.insert(transform);

                RTobject childObject = nullptr;
                RTobjecttype childObjType;
                rtTransformGetChild(transform, &childObject);
                rtTransformGetChildType(transform, &childObjType);

                vlrprintf("- 0x%p\n", childObject);

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
                vlrprintf("GeometryGroup\n");

                geometryGroupList.insert(geometryGroup);

                uint32_t numChildren;
                rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTgeometryinstance childObject = nullptr;
                    rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                    vlrprintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(RT_OBJECTTYPE_GEOMETRY_INSTANCE);
                }

                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_INSTANCE: {
                auto geometryInstance = (RTgeometryinstance)object;
                vlrprintf("GeometryInstance\n");

                RTgeometry geometry = nullptr;
                RTgeometrytriangles geometryTriangles = nullptr;
                rtGeometryInstanceGetGeometry(geometryInstance, &geometry);
                rtGeometryInstanceGetGeometryTriangles(geometryInstance, &geometryTriangles);
                VLRAssert((geometry != nullptr) ^ (geometryTriangles != nullptr), "Only one Geometry or GeometryTriangles node can be attached to a GeometryInstance at once.");
                uint32_t numPrims;
                if (geometry) {
                    rtGeometryGetPrimitiveCount(geometry, &numPrims);
                    vlrprintf("- Geometry 0x%p: %u [primitives]\n", geometry, numPrims);
                }
                if (geometryTriangles) {
                    rtGeometryTrianglesGetPrimitiveCount(geometryTriangles, &numPrims);
                    vlrprintf("- GeometryTriangles 0x%p: %u [primitives]\n", geometryTriangles, numPrims);
                }

                geometryInstanceList.insert(geometryInstance);

                break;
            }
            default:
                vlrprintf("\n");
                VLRAssert_ShouldNotBeCalled();
                break;
            }

            vlrprintf("\n");
        }



        vlrprintf("Groups:\n");
        for (auto group : groupList) {
            vlrprintf("  0x%p:\n", group);
            uint32_t numChildren;
            rtGroupGetChildCount(group, &numChildren);
            RTacceleration acceleration;
            rtGroupGetAcceleration(group, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            vlrprintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTobject childObject = nullptr;
                rtGroupGetChild(group, i, &childObject);

                vlrprintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        vlrprintf("Transforms:\n");
        for (auto transform : transformList) {
            vlrprintf("  0x%p:\n", transform);
            RTobject childObject = nullptr;
            rtTransformGetChild(transform, &childObject);
            float mat[16];
            float invMat[16];
            rtTransformGetMatrix(transform, true, mat, invMat);
            vlrprintf("    Matrix\n");
            vlrprintf("      %g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
            vlrprintf("      %g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
            vlrprintf("      %g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
            vlrprintf("      %g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
            vlrprintf("    Inverse Matrix\n");
            vlrprintf("      %g, %g, %g, %g\n", invMat[0], invMat[4], invMat[8], invMat[12]);
            vlrprintf("      %g, %g, %g, %g\n", invMat[1], invMat[5], invMat[9], invMat[13]);
            vlrprintf("      %g, %g, %g, %g\n", invMat[2], invMat[6], invMat[10], invMat[14]);
            vlrprintf("      %g, %g, %g, %g\n", invMat[3], invMat[7], invMat[11], invMat[15]);

            vlrprintf("  - 0x%p\n", childObject);
        }

        vlrprintf("GeometryGroups:\n");
        for (auto geometryGroup : geometryGroupList) {
            vlrprintf("  0x%p:\n", geometryGroup);
            uint32_t numChildren;
            rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
            RTacceleration acceleration;
            rtGeometryGroupGetAcceleration(geometryGroup, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            vlrprintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTgeometryinstance childObject = nullptr;
                rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                vlrprintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        vlrprintf("GeometryInstances:\n");
        for (auto geometryInstance : geometryInstanceList) {
            vlrprintf("  0x%p:\n", geometryInstance);
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
        //    vlrDevPrintf("%s:\n", concatenatedName.c_str());
        //    vlrDevPrintf("%g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
        //    vlrDevPrintf("%g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
        //    vlrDevPrintf("%g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
        //    vlrDevPrintf("%g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
        //    vlrDevPrintf("\n");
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

    void SHGeometryGroup::updateGeometryInstance(const SHGeometryInstance* instance) {
        VLRAssert(m_instances.count(instance), "There is no instance which matches the given instance.");
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
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/triangle_intersection.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        if (context.RTXEnabled()) {
            programSet.programCalcAttributeForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::calcAttributeForTriangle");
        }
        else {
            programSet.programIntersectTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::intersectTriangle");
            programSet.programCalcBBoxForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::calcBBoxForTriangle");
        }

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

        if (context.RTXEnabled()) {
            programSet.programCalcAttributeForTriangle->destroy();
        }
        else {
            programSet.programCalcBBoxForTriangle->destroy();
            programSet.programIntersectTriangle->destroy();
        }

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
            if (m_context.RTXEnabled())
                geom.optixGeometryTriangles->destroy();
            else
                geom.optixGeometry->destroy();
        }
        m_optixVertexBuffer->destroy();
    }

    void TriangleMeshSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);

        parent->geometryAddEvent(delta);
    }

    void TriangleMeshSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの削除を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);

        parent->geometryRemoveEvent(delta);
    }

    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        optix::Context optixContext = m_context.getOptiXContext();
        m_optixVertexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_vertices.size());
        m_optixVertexBuffer->setElementSize(sizeof(Vertex));
        {
            auto dstVertices = (Vertex*)m_optixVertexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy_n((Vertex*)m_vertices.data(), m_vertices.size(), dstVertices);
            m_optixVertexBuffer->unmap();
        }

        // TODO: 頂点情報更新時の処理。(IndexBufferとの整合性など)
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterial* material, 
                                                   const ShaderNodePlug &nodeNormal, const ShaderNodePlug &nodeAlpha, TangentType tangentType) {
        optix::Context optixContext = m_context.getOptiXContext();
        const OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        OptiXGeometry geom;
        CompensatedSum<float> sumImportances(0.0f);
        {
            geom.indices = std::move(indices);
            uint32_t numTriangles = (uint32_t)geom.indices.size() / 3;

            if (m_context.RTXEnabled()) {
                geom.optixGeometryTriangles = optixContext->createGeometryTriangles();
                geom.optixGeometryTriangles->setAttributeProgram(progSet.programCalcAttributeForTriangle);
            }
            else {
                geom.optixGeometry = optixContext->createGeometry();
                geom.optixGeometry->setIntersectionProgram(progSet.programIntersectTriangle);
                geom.optixGeometry->setBoundingBoxProgram(progSet.programCalcBBoxForTriangle);
            }

            geom.optixIndexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, numTriangles);
            geom.optixIndexBuffer->setElementSize(sizeof(Shared::Triangle));

            std::vector<float> areas;
            areas.resize(numTriangles);
            {
                auto dstTriangles = (Shared::Triangle*)geom.optixIndexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
                for (auto i = 0; i < numTriangles; ++i) {
                    uint32_t i0 = geom.indices[3 * i + 0];
                    uint32_t i1 = geom.indices[3 * i + 1];
                    uint32_t i2 = geom.indices[3 * i + 2];

                    dstTriangles[i] = Shared::Triangle{ i0, i1, i2 };

                    const Vertex (&v)[3] = { m_vertices[i0], m_vertices[i1], m_vertices[i2] };
                    areas[i] = std::fmax(0.0f, 0.5f * cross(v[1].position - v[0].position, v[2].position - v[0].position).length());
                    sumImportances += areas[i];
                }
                geom.optixIndexBuffer->unmap();
            }

            if (m_context.RTXEnabled()) {
                geom.optixGeometryTriangles->setPrimitiveCount(numTriangles);
                // TODO: share the same index buffer with different offsets.
                geom.optixGeometryTriangles->setTriangleIndices(geom.optixIndexBuffer, 0, sizeof(Shared::Triangle), RT_FORMAT_UNSIGNED_INT3);
                geom.optixGeometryTriangles->setVertices(m_vertices.size(), m_optixVertexBuffer, 0, sizeof(Vertex), RT_FORMAT_FLOAT3);
                geom.optixGeometryTriangles->setBuildFlags(RTgeometrybuildflags(0));
            }
            else {
                geom.optixGeometry->setPrimitiveCount(numTriangles);
            }

            if (material->isEmitting())
                geom.primDist.initialize(m_context, areas.data(), areas.size());
        }
        m_optixGeometries.push_back(geom);

        bool nodeNormalIsValid;
        bool nodeAlphaIsValid;

        m_materials.push_back(material);
        if (nodeNormal.node) {
            if (Shared::NodeTypeInfo<Normal3D>::ConversionIsDefinedFrom(nodeNormal.getType())) {
                m_nodeNormals.push_back(nodeNormal);
                nodeNormalIsValid = true;
            }
            else {
                vlrprintf("%s: Invalid plug type for normal is passed.\n", m_name.c_str());
                m_nodeNormals.push_back(ShaderNodePlug());
                nodeNormalIsValid = false;
            }
        }
        else {
            m_nodeNormals.push_back(ShaderNodePlug());
            nodeNormalIsValid = false;
        }
        if (nodeAlpha.node) {
            if (Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(nodeAlpha.getType())) {
                m_nodeAlphas.push_back(nodeAlpha);
                nodeAlphaIsValid = true;
            }
            else {
                vlrprintf("%s: Invalid plug type for alpha is passed.\n", m_name.c_str());
                m_nodeAlphas.push_back(ShaderNodePlug());
                nodeAlphaIsValid = false;
            }
        }
        else {
            m_nodeAlphas.push_back(ShaderNodePlug());
            nodeAlphaIsValid = false;
        }

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
            if (m_context.RTXEnabled())
                optixGeomInst->setGeometryTriangles(geom.optixGeometryTriangles);
            else
                optixGeomInst->setGeometry(geom.optixGeometry);

            optixGeomInst["VLR::pv_vertexBuffer"]->set(m_optixVertexBuffer);
            optixGeomInst["VLR::pv_triangleBuffer"]->set(geom.optixIndexBuffer);
            optixGeomInst["VLR::pv_sumImportances"]->setFloat(sumImportances.result);

            optixGeomInst["VLR::pv_progDecodeTexCoord"]->set(progSet.callableProgramDecodeTexCoordForTriangle);
            optixGeomInst["VLR::pv_progDecodeHitPoint"]->set(progSet.callableProgramDecodeHitPointForTriangle);

            optixGeomInst["VLR::pv_tangentType"]->setUserData(sizeof(tangentType), &tangentType);

            Shared::ShaderNodePlug sNodeNormal = Shared::ShaderNodePlug::Invalid();
            if (nodeNormalIsValid)
                sNodeNormal = nodeNormal.getSharedType();
            optixGeomInst["VLR::pv_nodeNormal"]->setUserData(sizeof(sNodeNormal), &sNodeNormal);

            optixGeomInst->setMaterialCount(1);
            Shared::ShaderNodePlug sNodeAlpha = Shared::ShaderNodePlug::Invalid();
            if (nodeAlphaIsValid) {
                optixGeomInst->setMaterial(0, m_context.getOptiXMaterialWithAlpha());
                sNodeAlpha = nodeAlpha.getSharedType();
            }
            else {
                optixGeomInst->setMaterial(0, m_context.getOptiXMaterialDefault());
            }
            optixGeomInst["VLR::pv_nodeAlpha"]->setUserData(sizeof(sNodeAlpha), &sNodeAlpha);

            uint32_t matIndex = material->getMaterialIndex();
            optixGeomInst["VLR::pv_materialIndex"]->setUserData(sizeof(matIndex), &matIndex);
            optixGeomInst["VLR::pv_importance"]->setFloat(lightDesc.importance);
        }
        m_shGeometryInstances.push_back(geomInst);

        // JP: 親にジオメトリインスタンスの追加を行わせる。
        std::set<const SHGeometryInstance*> delta;
        delta.insert(geomInst);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryAddEvent(delta);
        }
    }



    std::map<uint32_t, InfiniteSphereSurfaceNode::OptiXProgramSet> InfiniteSphereSurfaceNode::OptiXProgramSets;

    // static
    void InfiniteSphereSurfaceNode::initialize(Context &context) {
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/infinite_sphere_intersection.ptx");

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
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        delta.insert(m_shGeometryInstance);

        parent->geometryAddEvent(delta);
    }

    void InfiniteSphereSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの削除を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        delta.insert(m_shGeometryInstance);

        parent->geometryRemoveEvent(delta);
    }



    ParentNode::ParentNode(Context &context, const std::string &name, const Transform* localToWorld) :
        Node(context, name), m_serialChildID(0), m_localToWorld(localToWorld), m_shGeomGroup(context) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: Create a SHTransform having Transform of this node.
        if (m_localToWorld->isStatic()) {
            auto tr = (const StaticTransform*)m_localToWorld;
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

    void ParentNode::createConcatanatedTransforms(const std::set<SHTransform*>& childDelta, std::set<SHTransform*>* delta) {
        // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
        //     子のSHTransformをキーとして辞書に保存する。
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                StaticTransform* tr = (StaticTransform*)m_localToWorld;
                SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                m_shTransforms[*it] = shtr;
                if (delta)
                    delta->insert(shtr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
    }

    void ParentNode::removeConcatanatedTransforms(const std::set<SHTransform*>& childDelta, std::set<SHTransform*>* delta) {
        // JP: 
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            SHTransform* shtr = m_shTransforms.at(*it);
            m_shTransforms.erase(*it);
            if (delta)
                delta->insert(shtr);
        }
    }

    void ParentNode::updateConcatanatedTransforms(const std::set<SHTransform*>& childDelta, std::set<SHTransform*>* delta) {
        // JP: 
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            SHTransform* shtr = m_shTransforms.at(*it);
            shtr->update();
            if (delta)
                delta->insert(shtr);
        }
    }

    void ParentNode::addToGeometryGroup(const std::set<const SHGeometryInstance*>& childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup.addGeometryInstance(*it);

        SHTransform* selfTransform = m_shTransforms.at(nullptr);
        selfTransform->setChild(m_shGeomGroup.getNumInstances() > 0 ? &m_shGeomGroup : nullptr);
    }

    void ParentNode::removeFromGeometryGroup(const std::set<const SHGeometryInstance*>& childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup.removeGeometryInstance(*it);

        SHTransform* selfTransform = m_shTransforms.at(nullptr);
        selfTransform->setChild(m_shGeomGroup.getNumInstances() > 0 ? &m_shGeomGroup : nullptr);
    }

    void ParentNode::updateGeometryGroup(const std::set<const SHGeometryInstance*>& childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup.updateGeometryInstance(*it);

        SHTransform* selfTransform = m_shTransforms.at(nullptr);
        selfTransform->setChild(m_shGeomGroup.getNumInstances() > 0 ? &m_shGeomGroup : nullptr);
    }

    void ParentNode::setTransform(const Transform* localToWorld) {
        m_localToWorld = localToWorld;

        // JP: 管理中のSHTransformを更新する。
        // EN: updpate SHTransform under control.
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                auto tr = (StaticTransform*)m_localToWorld;
                SHTransform* shtr = it->second;
                shtr->setTransform(*tr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
    }

    void ParentNode::addToChildMap(Node* child) {
        if (m_childToSerialIDMap.count(child) > 0)
            return;
        m_childToSerialIDMap[child] = m_serialChildID;
        m_serialIDToChlidMap[m_serialChildID] = child;
        ++m_serialChildID;
    }

    void ParentNode::removeFromChildMap(Node* child) {
        if (m_childToSerialIDMap.count(child) == 0)
            return;
        uint32_t serialID = m_childToSerialIDMap.at(child);
        m_childToSerialIDMap.erase(child);
        m_serialIDToChlidMap.erase(serialID);
    }

    void ParentNode::addChild(InternalNode* child) {
        addToChildMap(child);
        child->addParent(this);
    }

    void ParentNode::removeChild(InternalNode* child) {
        removeFromChildMap(child);
        child->removeParent(this);
    }

    void ParentNode::addChild(SurfaceNode* child) {
        addToChildMap((Node*)child);
        child->addParent(this);
    }

    void ParentNode::removeChild(SurfaceNode* child) {
        removeFromChildMap((Node*)child);
        child->removeParent(this);
    }

    uint32_t ParentNode::getNumChildren() const {
        return (uint32_t)m_childToSerialIDMap.size();
    }

    void ParentNode::getChildren(Node** children) const {
        uint32_t i = 0;
        for (auto it = m_serialIDToChlidMap.cbegin(); it != m_serialIDToChlidMap.cend(); ++it)
            children[i++] = it->second;
    }

    Node* ParentNode::getChildAt(uint32_t index) const {
        if (index >= m_serialIDToChlidMap.size())
            return nullptr;

        auto it = m_serialIDToChlidMap.cbegin();
        std::advance(it, index); // want to make this operation O(log(n)).

        return it->second;
    }



    InternalNode::InternalNode(Context &context, const std::string &name, const Transform* localToWorld) :
        ParentNode(context, name, localToWorld) {
    }

    void InternalNode::transformAddEvent(const std::set<SHTransform*>& childDelta) {
        std::set<SHTransform*> delta;
        createConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 親に自分が保持するSHTransformが増えたことを通知(増分を通知)。
        // EN: 
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            auto parent = *it;
            parent->transformAddEvent(delta);
        }
    }

    void InternalNode::transformRemoveEvent(const std::set<SHTransform*>& childDelta) {
        std::set<SHTransform*> delta;
        removeConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 親に自分が保持するSHTransformが減ったことを通知(減分を通知)。
        // EN: 
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            auto parent = *it;
            parent->transformRemoveEvent(delta);
        }

        for (auto it = delta.cbegin(); it != delta.cend(); ++it)
            delete *it;
    }

    void InternalNode::transformUpdateEvent(const std::set<SHTransform*>& childDelta) {
        std::set<SHTransform*> delta;
        updateConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 親に自分が保持するSHTransformが更新されたことを通知(更新分を通知)。
        // EN: 
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            auto parent = *it;
            parent->transformUpdateEvent(delta);
        }
    }

    void InternalNode::geometryAddEvent(const std::set<const SHGeometryInstance*>& childDelta) {
        addToGeometryGroup(childDelta);

        geometryAddEvent(nullptr, childDelta);
    }

    void InternalNode::geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryAddEvent(transform, geomInstDelta);
        }
    }

    void InternalNode::geometryRemoveEvent(const std::set<const SHGeometryInstance*>& childDelta) {
        removeFromGeometryGroup(childDelta);

        geometryRemoveEvent(nullptr, childDelta);
    }

    void InternalNode::geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryRemoveEvent(transform, geomInstDelta);
        }
    }

    void InternalNode::setTransform(const Transform* localToWorld) {
        ParentNode::setTransform(localToWorld);

        // JP: 親に変形情報が更新されたことを通知する。
        // EN: 
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->transformUpdateEvent(delta);
        }
    }

    void InternalNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);

        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);

        // JP: 追加した親に対して「自身のSHTransform + 管理中の下位との連結SHTransform」の追加を行わせる。
        // EN: 
        parent->transformAddEvent(delta);

        // JP: 子孫が持つSHGeometryInstanceの追加を親に伝える。
        // EN: 
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            std::set<const SHGeometryInstance*> geomInstDelta;

            SHTransform* shtr = it->second;
            SHGeometryGroup* geomGroup;
            if (shtr->hasGeometryDescendant(&geomGroup)) {
                for (int i = 0; i < geomGroup->getNumInstances(); ++i)
                    geomInstDelta.insert(geomGroup->getGeometryInstanceAt(i));

                parent->geometryAddEvent(shtr, geomInstDelta);
            }
        }
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 子孫が持つSHGeometryInstanceの削除を親に伝える。
        // EN: 
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            std::set<const SHGeometryInstance*> geomInstDelta;

            SHTransform* shtr = it->second;
            SHGeometryGroup* geomGroup;
            if (shtr->hasGeometryDescendant(&geomGroup)) {
                for (int i = 0; i < geomGroup->getNumInstances(); ++i)
                    geomInstDelta.insert(geomGroup->getGeometryInstanceAt(i));

                parent->geometryRemoveEvent(shtr, geomInstDelta);
            }
        }

        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);

        // JP: 追加した親に対して「自身のSHTransform + 管理中の下位との連結SHTransform」の削除を行わせる。
        // EN: 
        parent->transformRemoveEvent(delta);
    }



    RootNode::RootNode(Context &context, const Transform* localToWorld) :
        ParentNode(context, "Root", localToWorld), m_shGroup(context), m_surfaceLightsAreSetup(false) {
        SHTransform* shtr = m_shTransforms[0];
        m_shGroup.addChild(shtr);
    }

    RootNode::~RootNode() {
        if (m_surfaceLightsAreSetup) {
            m_surfaceLightImpDist.finalize(m_context);

            m_optixSurfaceLightDescriptorBuffer->destroy();

            m_surfaceLightsAreSetup = false;
        }
    }

    void RootNode::transformAddEvent(const std::set<SHTransform*>& childDelta) {
        std::set<SHTransform*> delta;
        createConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: SHGroupにもSHTransformを追加する。
        // EN: 
        for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
            SHTransform* shtr = *it;
            m_shGroup.addChild(shtr);
        }
    }

    void RootNode::transformRemoveEvent(const std::set<SHTransform*>& childDelta) {
        std::set<SHTransform*> delta;
        removeConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: SHGroupからSHTransformを削除する。
        // EN: 
        for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
            SHTransform* shtr = *it;
            m_shGroup.removeChild(shtr);
        }

        for (auto it = delta.cbegin(); it != delta.cend(); ++it)
            delete *it;
    }

    void RootNode::transformUpdateEvent(const std::set<SHTransform*>& childDelta) {
        std::set<SHTransform*> delta;
        updateConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: SHTransformを更新する。
        // EN: 
        for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
            SHTransform* shtr = *it;
            m_shGroup.updateChild(shtr);
        }
    }

    void RootNode::geometryAddEvent(const std::set<const SHGeometryInstance*>& childDelta) {
        addToGeometryGroup(childDelta);

        geometryAddEvent(nullptr, childDelta);
    }

    void RootNode::geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        m_shGroup.updateChild(transform);

        for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
            auto key = TransformAndGeometryInstance{ transform, *it };
            if (m_surfaceLights.count(key)) {
                VLRAssert_ShouldNotBeCalled();
            }
            else {
                Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[key]; // new entry
                (*it)->getSurfaceLightDescriptor(&lightDesc);
                if (transform->isStatic()) {
                    StaticTransform tr = transform->getStaticTransform();
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
    }

    void RootNode::geometryRemoveEvent(const std::set<const SHGeometryInstance*>& childDelta) {
        removeFromGeometryGroup(childDelta);

        geometryRemoveEvent(nullptr, childDelta);
    }

    void RootNode::geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        m_shGroup.updateChild(transform);

        for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
            auto key = TransformAndGeometryInstance{ transform, *it };
            if (m_surfaceLights.count(key)) {
                m_surfaceLights.erase(key);
            }
            else {
                VLRAssert_ShouldNotBeCalled();
            }
        }
        m_surfaceLightsAreSetup = false;
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
                Shared::SurfaceLightDescriptor* descs = (Shared::SurfaceLightDescriptor*)m_optixSurfaceLightDescriptorBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
                for (auto it = m_surfaceLights.cbegin(); it != m_surfaceLights.cend(); ++it) {
                    uint32_t index = std::distance(m_surfaceLights.cbegin(), it);
                    const Shared::SurfaceLightDescriptor &lightDesc = it->second;
                    if (lightDesc.importance > 0)
                        vlrDevPrintf("Light %u: %g\n", index, lightDesc.importance);
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
    Object(context), m_rootNode(context, localToWorld), m_matEnv(nullptr), m_envRotationPhi(0) {
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/infinite_sphere_intersection.ptx");

        optix::Context optixContext = context.getOptiXContext();

        m_callableProgramSampleInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::sampleInfiniteSphere");
    }

    Scene::~Scene() {
        m_callableProgramSampleInfiniteSphere->destroy();
    }

    void Scene::setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv) {
        m_matEnv = matEnv;
    }

    void Scene::setEnvironmentRotation(float rotationPhi) {
        m_envRotationPhi = rotationPhi;
    }

    void Scene::set() {
        m_rootNode.set();

        optix::Context optixContext = m_context.getOptiXContext();

        Shared::SurfaceLightDescriptor envLight;
        envLight.importance = 0.0f;
        if (m_matEnv) {
            m_matEnv->getImportanceMap().getInternalType(&envLight.body.asEnvironmentLight.importanceMap);
            envLight.body.asEnvironmentLight.materialIndex = m_matEnv->getMaterialIndex();
            envLight.body.asEnvironmentLight.rotationPhi = m_envRotationPhi;
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
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/cameras.ptx");

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

    PerspectiveCamera::PerspectiveCamera(Context &context) :
        Camera(context) {
        m_data.position = Point3D(0, 0, 0);
        m_data.orientation = Quaternion::Identity();
        m_data.aspect = 1.0f;
        m_data.fovY = 45 * M_PI / 180;
        m_data.lensRadius = 1.0f;
        m_data.sensitivity = 1.0f;
        m_data.objPlaneDistance = 1.0f;
        m_data.setImagePlaneArea();
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
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/cameras.ptx");

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

    EquirectangularCamera::EquirectangularCamera(Context &context) :
        Camera(context) {
        m_data.position = Point3D(0, 0, 0);
        m_data.orientation = Quaternion::Identity();
        m_data.phiAngle = 2 * M_PI;
        m_data.thetaAngle = M_PI;
        m_data.sensitivity = 1.0f;
    }

    void EquirectangularCamera::set() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_equirectangularCamera"]->setUserData(sizeof(Shared::EquirectangularCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }
}
