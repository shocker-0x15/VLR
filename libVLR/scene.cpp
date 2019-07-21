#include "scene.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Shallow Hierarchy

    void SHGroup::destroyOptiXDescendants(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);
        SHGeometryGroup* descendant;
        transform->hasGeometryDescendant(&descendant);

        for (int i = descendant->getNumInstances() - 1; i >= 0; --i) {
            const SHGeometryInstance* inst = descendant->getGeometryInstanceAt(i);
            VLRAssert(status.geomInstances.count(inst), "SHGeometryInstance doesn't exist.");
            optix::GeometryInstance optixInst = status.geomInstances.at(inst);

            status.geomGroup->removeChild(optixInst);
            status.geomInstances.erase(inst);

            uint32_t geomInstIndex;
            optixInst["VLR::pv_geomInstIndex"]->getUserData(sizeof(geomInstIndex), &geomInstIndex);
            m_geometryInstanceDescriptorBuffer.release(geomInstIndex);

            optixInst->destroy();
        }
        status.geomGroup->destroy();
        status.geomGroup = nullptr;

        m_optixGroup->removeChild(status.transform);
        status.transform->destroy();
        status.transform = nullptr;

        m_surfaceLightsAreSetup = false;

        m_optixAcceleration->markDirty();
    }

    void SHGroup::addChild(SHTransform* transform) {
        m_transforms[transform] = std::move(TransformStatus());
    }

    void SHGroup::removeChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        const TransformStatus &status = m_transforms.at(transform);
        if (status.hasGeometryDescendant) {
            destroyOptiXDescendants(transform);

            --m_numValidTransforms;
        }
        m_transforms.erase(transform);
    }

    void SHGroup::updateChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);

        if (status.transform) {
            StaticTransform tr = transform->getStaticTransform();
            float mat[16], invMat[16];
            tr.getArrays(mat, invMat);
            status.transform->setMatrix(true, mat, invMat);
        }

        for (auto it = status.geomInstances.cbegin(); it != status.geomInstances.cend(); ++it) {
            optix::GeometryInstance optixInst = it->second;

            uint32_t geomInstIndex;
            optixInst["VLR::pv_geomInstIndex"]->getUserData(sizeof(geomInstIndex), &geomInstIndex);

            Shared::GeometryInstanceDescriptor geomInstDesc;
            m_geometryInstanceDescriptorBuffer.get(geomInstIndex, &geomInstDesc);
            if (transform->isStatic()) {
                StaticTransform tr = transform->getStaticTransform();
                float mat[16], invMat[16];
                tr.getArrays(mat, invMat);
                geomInstDesc.body.asTriMesh.transform = Shared::StaticTransform(Matrix4x4(mat));
            }
            else {
                VLRAssert_NotImplemented();
            }
            m_geometryInstanceDescriptorBuffer.update(geomInstIndex, geomInstDesc);
        }

        m_surfaceLightsAreSetup = false;

        m_optixAcceleration->markDirty();
    }

    void SHGroup::addGeometryInstances(SHTransform* transform, std::set<const SHGeometryInstance*> geomInsts) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);

        status.hasGeometryDescendant = true;
        ++m_numValidTransforms;

        SHGeometryGroup* descendant;
        transform->hasGeometryDescendant(&descendant);

        optix::Context optixContext = m_context.getOptiXContext();

        if (!status.transform) {
            status.transform = optixContext->createTransform();
            StaticTransform tr = transform->getStaticTransform();
            float mat[16], invMat[16];
            tr.getArrays(mat, invMat);
            status.transform->setMatrix(true, mat, invMat);

            m_optixGroup->addChild(status.transform);
        }

        if (!status.geomGroup) {
            status.geomGroup = optixContext->createGeometryGroup();
            status.geomGroup->setAcceleration(descendant->getAcceleration());

            status.transform->setChild(status.geomGroup);
        }

        for (auto it = geomInsts.cbegin(); it != geomInsts.cend(); ++it) {
            const SHGeometryInstance* inst = *it;
            VLRAssert(descendant->has(inst), "Invalid child.");
            optix::GeometryInstance optixInst = inst->createGeometryInstance(m_context);

            uint32_t geomInstIndex = m_geometryInstanceDescriptorBuffer.allocate();
            optixInst["VLR::pv_geomInstIndex"]->setUserData(sizeof(geomInstIndex), &geomInstIndex);

            Shared::GeometryInstanceDescriptor geomInstDesc;
            inst->createGeometryInstanceDescriptor(&geomInstDesc);
            if (transform->isStatic()) {
                StaticTransform tr = transform->getStaticTransform();
                float mat[16], invMat[16];
                tr.getArrays(mat, invMat);
                geomInstDesc.body.asTriMesh.transform = Shared::StaticTransform(Matrix4x4(mat));
            }
            else {
                VLRAssert_NotImplemented();
            }
            m_geometryInstanceDescriptorBuffer.update(geomInstIndex, geomInstDesc);

            status.geomInstances[inst] = optixInst;
            status.geomGroup->addChild(optixInst);
        }

        m_surfaceLightsAreSetup = false;

        m_optixAcceleration->markDirty();
    }

    void SHGroup::removeGeometryInstances(SHTransform* transform, std::set<const SHGeometryInstance*> geomInsts) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);

        SHGeometryGroup* descendant;
        transform->hasGeometryDescendant(&descendant);

        for (auto it = geomInsts.cbegin(); it != geomInsts.cend(); ++it) {
            const SHGeometryInstance* inst = *it;
            VLRAssert(descendant->has(inst), "Invalid child.");
            optix::GeometryInstance optixInst = status.geomInstances.at(inst);

            status.geomGroup->removeChild(optixInst);
            status.geomInstances.erase(inst);

            uint32_t geomInstIndex;
            optixInst["VLR::pv_geomInstIndex"]->getUserData(sizeof(geomInstIndex), &geomInstIndex);
            m_geometryInstanceDescriptorBuffer.release(geomInstIndex);

            optixInst->destroy();
        }

        if (status.geomInstances.size() == 0) {
            status.geomGroup->destroy();
            status.geomGroup = nullptr;

            m_optixGroup->removeChild(status.transform);
            status.transform->destroy();
            status.transform = nullptr;

            status.hasGeometryDescendant = false;
            --m_numValidTransforms;
        }

        m_surfaceLightsAreSetup = false;

        m_optixAcceleration->markDirty();
    }

    void SHGroup::setup() {
        optix::Context optixContext = m_context.getOptiXContext();

        optixContext["VLR::pv_topGroup"]->set(m_optixGroup);
        optixContext["VLR::pv_geometryInstanceDescriptorBuffer"]->set(m_geometryInstanceDescriptorBuffer.optixBuffer);

        if (!m_surfaceLightsAreSetup) {
            std::vector<float> importances;
            importances.resize(m_geometryInstanceDescriptorBuffer.maxNumElements, 0.0f);

            {
                auto descs = (Shared::GeometryInstanceDescriptor*)m_geometryInstanceDescriptorBuffer.optixBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
                for (int i = 0; i < m_geometryInstanceDescriptorBuffer.maxNumElements; ++i) {
                    if (!m_geometryInstanceDescriptorBuffer.slotFinder.getUsage(i))
                        continue;

                    const Shared::GeometryInstanceDescriptor &desc = descs[i];
                    if (desc.importance > 0)
                        vlrDevPrintf("Light %u: %g\n", i, desc.importance);
                    importances[i] = desc.importance;
                }
                m_geometryInstanceDescriptorBuffer.optixBuffer->unmap();
            }

            m_surfaceLightImpDist.finalize(m_context);
            m_surfaceLightImpDist.initialize(m_context, importances.data(), importances.size());

            m_surfaceLightsAreSetup = true;
        }

        Shared::DiscreteDistribution1D lightImpDist;
        m_surfaceLightImpDist.getInternalType(&lightImpDist);
        optixContext["VLR::pv_lightImpDist"]->setUserData(sizeof(lightImpDist), &lightImpDist);
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



    StaticTransform SHTransform::resolveTransform() const {
        int32_t stackIdx = 0;
        const SHTransform* stack[10];
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

        return res;

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
    }

    void SHTransform::update() {
    }

    bool SHTransform::isStatic() const {
        // TODO: implement
        return true;
    }

    StaticTransform SHTransform::getStaticTransform() const {
        if (isStatic()) {
            return resolveTransform();
        }
        else {
            VLRAssert_ShouldNotBeCalled();
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
        m_optixAcceleration->markDirty();
    }

    void SHGeometryGroup::removeGeometryInstance(const SHGeometryInstance* instance) {
        m_instances.erase(instance);
        m_optixAcceleration->markDirty();
    }

    void SHGeometryGroup::updateGeometryInstance(const SHGeometryInstance* instance) {
        VLRAssert(m_instances.count(instance), "There is no instance which matches the given instance.");
        m_optixAcceleration->markDirty();
    }



    optix::GeometryInstance SHGeometryInstance::createGeometryInstance(Context &context) const {
        optix::Context optixContext = context.getOptiXContext();

        optix::GeometryInstance geomInst = optixContext->createGeometryInstance();

        geomInst["VLR::pv_progDecodeHitPoint"]->set(m_progDecodeHitPoint);

        geomInst->setMaterialCount(1);
        geomInst->setMaterial(0, m_material);
        geomInst["VLR::pv_materialIndex"]->setUserData(sizeof(m_materialIndex), &m_materialIndex);
        geomInst["VLR::pv_importance"]->setFloat(m_importance);

        Shared::ShaderNodePlug sNodeNormal = m_nodeNormal.getSharedType();
        geomInst["VLR::pv_nodeNormal"]->setUserData(sizeof(sNodeNormal), &sNodeNormal);

        Shared::ShaderNodePlug sNodeTangent = m_nodeTangent.getSharedType();
        geomInst["VLR::pv_nodeTangent"]->setUserData(sizeof(sNodeTangent), &sNodeTangent);

        Shared::ShaderNodePlug sNodeAlpha = m_nodeAlpha.getSharedType();
        geomInst["VLR::pv_nodeAlpha"]->setUserData(sizeof(sNodeAlpha), &sNodeAlpha);

        if (m_isTriMesh) {
            if (context.RTXEnabled())
                geomInst->setGeometryTriangles(m_geometryTriangles);
            else
                geomInst->setGeometry(m_geometry);

            geomInst["VLR::pv_vertexBuffer"]->set(m_triMeshProp.vertexBuffer);
            geomInst["VLR::pv_triangleBuffer"]->set(m_triMeshProp.triangleBuffer);
            geomInst["VLR::pv_sumImportances"]->setFloat(m_triMeshProp.sumImportances);
        }
        else {
            geomInst->setGeometry(m_geometry);
            VLRAssert_NotImplemented();
        }

        return geomInst;
    }

    void SHGeometryInstance::createGeometryInstanceDescriptor(Shared::GeometryInstanceDescriptor* desc) const {
        desc->materialIndex = m_materialIndex;
        desc->importance = m_importance;
        desc->sampleFunc = m_progSample;

        if (m_isTriMesh) {
            desc->body.asTriMesh.vertexBuffer = m_triMeshProp.vertexBuffer->getId();
            desc->body.asTriMesh.triangleBuffer = m_triMeshProp.triangleBuffer->getId();
            m_triMeshProp.primDist.getInternalType(&desc->body.asTriMesh.primDistribution);
            desc->body.asTriMesh.transform = Shared::StaticTransform(Matrix4x4::Identity());
        }
        else {
            VLRAssert_NotImplemented();
        }
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

        programSet.callableProgramSampleTriangleMesh = optixContext->createProgramFromPTXString(ptx, "VLR::sampleTriangleMesh");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TriangleMeshSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleTriangleMesh->destroy();

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
                                                   const ShaderNodePlug &nodeNormal, const ShaderNodePlug& nodeTangent, const ShaderNodePlug &nodeAlpha) {
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

        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;

        m_materials.push_back(material);
        if (nodeNormal.node) {
            if (Shared::NodeTypeInfo<Normal3D>::ConversionIsDefinedFrom(nodeNormal.getType()))
                plugNormal = nodeNormal;
            else
                vlrprintf("%s: Invalid plug type for normal is passed.\n", m_name.c_str());
        }
        m_nodeNormals.push_back(plugNormal);
        if (nodeTangent.node) {
            if (Shared::NodeTypeInfo<Vector3D>::ConversionIsDefinedFrom(nodeTangent.getType()))
                plugTangent = nodeTangent;
            else
                vlrprintf("%s: Invalid plug type for tangent is passed.\n", m_name.c_str());
        }
        m_nodeTangents.push_back(plugTangent);
        if (nodeAlpha.node) {
            if (Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(nodeAlpha.getType()))
                plugAlpha = nodeAlpha;
            else
                vlrprintf("%s: Invalid plug type for alpha is passed.\n", m_name.c_str());
        }
        m_nodeAlphas.push_back(plugAlpha);

        optix::Program progDecodeHitPoint = progSet.callableProgramDecodeHitPointForTriangle;
        int32_t progSample = progSet.callableProgramSampleTriangleMesh->getId();

        optix::Material optixMaterial = plugAlpha.isValid() ? m_context.getOptiXMaterialWithAlpha() : m_context.getOptiXMaterialDefault();
        uint32_t materialIndex = material->getMaterialIndex();
        float importance = material->isEmitting() ? 1.0f : 0.0f; // TODO

        SHGeometryInstance* geomInst;
        if (m_context.RTXEnabled()) {
            geomInst = new SHGeometryInstance(geom.optixGeometryTriangles,
                                              progDecodeHitPoint, progSample,
                                              optixMaterial, materialIndex, importance,
                                              plugNormal, plugTangent, plugAlpha,
                                              m_optixVertexBuffer, geom.optixIndexBuffer,
                                              geom.primDist, sumImportances.result);
        }
        else {
            geomInst = new SHGeometryInstance(geom.optixGeometry,
                                              progDecodeHitPoint, progSample,
                                              optixMaterial, materialIndex, importance,
                                              plugNormal, plugTangent, plugAlpha,
                                              m_optixVertexBuffer, geom.optixIndexBuffer,
                                              geom.primDist, sumImportances.result);
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

        programSet.callableProgramSampleInfiniteSphere = optixContext->createProgramFromPTXString(ptx, "VLR::sampleInfiniteSphere");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void InfiniteSphereSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleInfiniteSphere->destroy();

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

        m_shGeometryInstance = new SHGeometryInstance(m_optixGeometry,
                                                      progSet.callableProgramDecodeHitPointForInfiniteSphere,
                                                      progSet.callableProgramSampleInfiniteSphere,
                                                      m_context.getOptiXMaterialDefault(), material->getMaterialIndex(),
                                                      material->isEmitting() ? 1.0f : 0.0f);
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

    // TODO: 関数に分ける必要性が感じられない？
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

    void ParentNode::geometryAddEvent(const std::set<const SHGeometryInstance*>& childDelta) {
        addToGeometryGroup(childDelta);

        geometryAddEvent(nullptr, childDelta);
    }

    void ParentNode::geometryRemoveEvent(const std::set<const SHGeometryInstance*>& childDelta) {
        geometryRemoveEvent(nullptr, childDelta);

        removeFromGeometryGroup(childDelta);
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

    void InternalNode::geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryAddEvent(transform, geomInstDelta);
        }
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
        ParentNode(context, "Root", localToWorld), m_shGroup(context) {
        SHTransform* shtr = m_shTransforms[0];
        m_shGroup.addChild(shtr);
    }

    RootNode::~RootNode() {
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

    void RootNode::geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        m_shGroup.addGeometryInstances(transform, geomInstDelta);
    }

    void RootNode::geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        m_shGroup.removeGeometryInstances(transform, geomInstDelta);
    }

    void RootNode::setup() {
        m_shGroup.setup();
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

    void Scene::setup() {
        m_rootNode.setup();

        optix::Context optixContext = m_context.getOptiXContext();

        Shared::GeometryInstanceDescriptor envLight;
        envLight.importance = 0.0f;
        if (m_matEnv) {
            m_matEnv->getImportanceMap().getInternalType(&envLight.body.asInfSphere.importanceMap);
            envLight.body.asInfSphere.rotationPhi = m_envRotationPhi;
            envLight.materialIndex = m_matEnv->getMaterialIndex();
            envLight.importance = 1.0f;
            envLight.sampleFunc = m_callableProgramSampleInfiniteSphere->getId();
        }

        optixContext["VLR::pv_envLightDescriptor"]->setUserData(sizeof(envLight), &envLight);
    }



    std::string Camera::s_cameras_ptx;

    // static
    void Camera::commonInitializeProcedure(Context& context, const char* identifiers[2], OptiXProgramSet* programSet) {
        const std::string& ptx = s_cameras_ptx;

        optix::Context optixContext = context.getOptiXContext();

        programSet->callableProgramSampleLensPosition = optixContext->createProgramFromPTXString(ptx, identifiers[0]);
        programSet->callableProgramSampleIDF = optixContext->createProgramFromPTXString(ptx, identifiers[1]);
    }

    // static
    void Camera::commonFinalizeProcedure(Context& context, OptiXProgramSet& programSet) {
        if (programSet.callableProgramSampleLensPosition) {
            programSet.callableProgramSampleIDF->destroy();
            programSet.callableProgramSampleLensPosition->destroy();
        }
    }
    
    // static
    void Camera::initialize(Context &context) {
        s_cameras_ptx = readTxtFile(getExecutableDirectory() / "ptxes/cameras.ptx");

        PerspectiveCamera::initialize(context);
        EquirectangularCamera::initialize(context);
    }

    // static
    void Camera::finalize(Context &context) {
        EquirectangularCamera::finalize(context);
        PerspectiveCamera::finalize(context);
    }



    std::vector<ParameterInfo> PerspectiveCamera::ParameterInfos;
    
    std::map<uint32_t, Camera::OptiXProgramSet> PerspectiveCamera::OptiXProgramSets;

    // static
    void PerspectiveCamera::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("position", VLRParameterFormFlag_ImmediateValue, ParameterPoint3D),
            ParameterInfo("orientation", VLRParameterFormFlag_ImmediateValue, ParameterQuaternion),
            ParameterInfo("aspect", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("sensitivity", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("fovy", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("lens radius", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("op distance", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            "VLR::PerspectiveCamera_sampleLensPosition",
            "VLR::PerspectiveCamera_sampleIDF"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void PerspectiveCamera::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
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

    bool PerspectiveCamera::get(const char* paramName, Point3D* value) const {
        if (testParamName(paramName, "position")) {
            *value = m_data.position;
        }
        else {
            return false;
        }

        return true;
    }

    bool PerspectiveCamera::get(const char* paramName, Quaternion* value) const {
        if (testParamName(paramName, "orientation")) {
            *value = m_data.orientation;
        }
        else {
            return false;
        }

        return true;
    }

    bool PerspectiveCamera::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "aspect")) {
            if (length != 1)
                return false;

            values[0] = m_data.aspect;
        }
        else if (testParamName(paramName, "sensitivity")) {
            if (length != 1)
                return false;

            values[0] = m_data.sensitivity;
        }
        else if (testParamName(paramName, "fovy")) {
            if (length != 1)
                return false;

            values[0] = m_data.fovY;
        }
        else if (testParamName(paramName, "lens radius")) {
            if (length != 1)
                return false;

            values[0] = m_data.lensRadius;
        }
        else if (testParamName(paramName, "op distance")) {
            if (length != 1)
                return false;

            values[0] = m_data.objPlaneDistance;
        }
        else {
            return false;
        }

        return true;
    }

    bool PerspectiveCamera::set(const char* paramName, const Point3D& value) {
        if (testParamName(paramName, "position")) {
            m_data.position = value;
        }
        else {
            return false;
        }

        return true;
    }

    bool PerspectiveCamera::set(const char* paramName, const Quaternion& value) {
        if (testParamName(paramName, "orientation")) {
            m_data.orientation = value;
        }
        else {
            return false;
        }

        return true;
    }

    bool PerspectiveCamera::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "aspect")) {
            if (length != 1)
                return false;

            m_data.aspect = std::max(0.001f, values[0]);
        }
        else if (testParamName(paramName, "sensitivity")) {
            if (length != 1)
                return false;

            m_data.sensitivity = std::max(0.0f, std::isfinite(values[0]) ? values[0] : 1.0f);
        }
        else if (testParamName(paramName, "fovy")) {
            if (length != 1)
                return false;

            m_data.fovY = VLR::clamp<float>(values[0], 0.0001f, M_PI * 0.999f);
        }
        else if (testParamName(paramName, "lens radius")) {
            if (length != 1)
                return false;

            m_data.lensRadius = std::max(0.0f, values[0]);
        }
        else if (testParamName(paramName, "op distance")) {
            if (length != 1)
                return false;

            m_data.objPlaneDistance = std::max(0.0f, values[0]);
        }
        else {
            return false;
        }
        m_data.setImagePlaneArea();

        return true;
    }

    void PerspectiveCamera::setup() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_perspectiveCamera"]->setUserData(sizeof(Shared::PerspectiveCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }



    std::vector<ParameterInfo> EquirectangularCamera::ParameterInfos;
    
    std::map<uint32_t, Camera::OptiXProgramSet> EquirectangularCamera::OptiXProgramSets;

    // static
    void EquirectangularCamera::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("position", VLRParameterFormFlag_ImmediateValue, ParameterPoint3D),
            ParameterInfo("orientation", VLRParameterFormFlag_ImmediateValue, ParameterQuaternion),
            ParameterInfo("sensitivity", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("h angle", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("v angle", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            "VLR::EquirectangularCamera_sampleLensPosition",
            "VLR::EquirectangularCamera_sampleIDF"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EquirectangularCamera::finalize(Context &context) {
        OptiXProgramSet& programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
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

    bool EquirectangularCamera::get(const char* paramName, Point3D* value) const {
        if (testParamName(paramName, "position")) {
            *value = m_data.position;
        }
        else {
            return false;
        }

        return true;
    }

    bool EquirectangularCamera::get(const char* paramName, Quaternion* value) const {
        if (testParamName(paramName, "orientation")) {
            *value = m_data.orientation;
        }
        else {
            return false;
        }

        return true;
    }

    bool EquirectangularCamera::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "sensitivity")) {
            if (length != 1)
                return false;

            values[0] = m_data.sensitivity;
        }
        else if (testParamName(paramName, "h angle")) {
            if (length != 1)
                return false;

            values[0] = m_data.phiAngle;
        }
        else if (testParamName(paramName, "v angle")) {
            if (length != 1)
                return false;

            values[0] = m_data.thetaAngle;
        }
        else {
            return false;
        }

        return true;
    }

    bool EquirectangularCamera::set(const char* paramName, const Point3D& value) {
        if (testParamName(paramName, "position")) {
            m_data.position = value;
        }
        else {
            return false;
        }

        return true;
    }

    bool EquirectangularCamera::set(const char* paramName, const Quaternion& value) {
        if (testParamName(paramName, "orientation")) {
            m_data.orientation = value;
        }
        else {
            return false;
        }

        return true;
    }

    bool EquirectangularCamera::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "sensitivity")) {
            if (length != 1)
                return false;

            m_data.sensitivity = std::max(0.0f, std::isfinite(values[0]) ? values[0] : 1.0f);
        }
        else if (testParamName(paramName, "h angle")) {
            if (length != 1)
                return false;

            m_data.phiAngle = VLR::clamp<float>(values[0], 0.01f, 2 * M_PI);
        }
        else if (testParamName(paramName, "v angle")) {
            if (length != 1)
                return false;

            m_data.thetaAngle = VLR::clamp<float>(values[0], 0.01f, M_PI);
        }
        else {
            return false;
        }

        return true;
    }

    void EquirectangularCamera::setup() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_equirectangularCamera"]->setUserData(sizeof(Shared::EquirectangularCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }
}
