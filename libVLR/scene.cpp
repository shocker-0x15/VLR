#include "scene.h"

namespace vlr {
    // ----------------------------------------------------------------
    // Shallow Hierarchy

    void SHGeometryGroup::addChild(const SHGeometryInstance* geomInst) {
        m_shGeomInsts.push_back(geomInst);
        m_optixGas.addChild(geomInst->optixGeomInst);
    }

    void SHGeometryGroup::removeChild(const SHGeometryInstance* geomInst) {
        auto idx = std::find(m_shGeomInsts.cbegin(), m_shGeomInsts.cend(), geomInst);
        VLRAssert(idx != m_shGeomInsts.cend(), "SHGeometryInstance %p is not a child of SHGeometryGroup %p.", geomInst, this);
        m_shGeomInsts.erase(idx);
        m_optixGas.removeChildAt(m_optixGas.findChildIndex(geomInst->optixGeomInst));
    }

    void SHGeometryGroup::updateChild(const SHGeometryInstance* geomInst) {
        (void)geomInst;
        m_optixGas.markDirty();
    }

    void SHGeometryGroup::getGeometryInstanceIndices(uint32_t* indices) const {
        for (int i = 0; i < m_shGeomInsts.size(); ++i)
            indices[i] = m_shGeomInsts[i]->geomInstIndex;
    }

    void SHGeometryGroup::getGeometryInstanceImportanceValues(float* values) const {
        for (int i = 0; i < m_shGeomInsts.size(); ++i)
            values[i] = m_shGeomInsts[i]->data.importance;
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

    void SHTransform::setChild(const SHGeometryGroup* childGeomGroup) {
        VLRAssert(!m_childIsTransform, "Transform which doesn't have a child transform can have a geometry group as a child.");
        m_childGeomGroup = childGeomGroup;
    }

    bool SHTransform::hasGeometryDescendant(const SHGeometryGroup** descendant) const {
        if (descendant)
            *descendant = nullptr;

        const SHTransform* nextSHTr = this;
        while (nextSHTr) {
            if (!nextSHTr->m_childIsTransform && nextSHTr->m_childGeomGroup != nullptr) {
                if (descendant)
                    *descendant = nextSHTr->m_childGeomGroup;
                return true;
            }
            else {
                nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
            }
        }

        return false;
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



    std::map<uint32_t, TriangleMeshSurfaceNode::OptiXProgramSet> TriangleMeshSurfaceNode::s_optiXProgramSets;

    // static
    void TriangleMeshSurfaceNode::initialize(Context &context) {
        optixu::Pipeline optixPipeline = context.getOptixPipeline();

        OptiXProgramSet programSet;
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/triangle_intersection.ptx");
        programSet.optixModule = optixPipeline.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
            VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));
        programSet.dcDecodeHitPointForTriangle = context.createDirectCallableProgram(
            programSet.optixModule, RT_DC_NAME_STR("decodeHitPointForTriangle"));
        programSet.dcSampleTriangleMesh = context.createDirectCallableProgram(
            programSet.optixModule, RT_DC_NAME_STR("sampleTriangleMesh"));

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TriangleMeshSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = s_optiXProgramSets.at(context.getID());
        context.destroyDirectCallableProgram(programSet.dcSampleTriangleMesh);
        context.destroyDirectCallableProgram(programSet.dcDecodeHitPointForTriangle);
        s_optiXProgramSets.erase(context.getID());
    }

    TriangleMeshSurfaceNode::TriangleMeshSurfaceNode(Context &context, const std::string &name) :
        SurfaceNode(context, name) {}

    TriangleMeshSurfaceNode::~TriangleMeshSurfaceNode() {
        for (auto it = m_materialGroups.rbegin(); it != m_materialGroups.rend(); ++it) {
            MaterialGroup &matGroup = *it;
            m_context.releaseGeometryInstance(matGroup.shGeomInst->geomInstIndex);
            matGroup.shGeomInst->optixGeomInst.destroy();
            delete matGroup.shGeomInst;
            matGroup.primDist.finalize(m_context);
            matGroup.optixIndexBuffer.finalize();
        }
        m_optixVertexBuffer.finalize();
    }

    void TriangleMeshSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_materialGroups.cbegin(); it != m_materialGroups.cend(); ++it)
            delta.insert(it->shGeomInst);

        parent->geometryAddEvent(delta);
    }

    void TriangleMeshSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの削除を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_materialGroups.cbegin(); it != m_materialGroups.cend(); ++it)
            delta.insert(it->shGeomInst);

        parent->geometryRemoveEvent(delta);
    }

    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        CUcontext cuContext = m_context.getCUcontext();
        m_optixVertexBuffer.initialize(cuContext, g_bufferType, m_vertices);

        // TODO: 頂点情報更新時の処理。(IndexBufferとの整合性など)
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterial* material, 
                                                   const ShaderNodePlug &nodeNormal, const ShaderNodePlug& nodeTangent, const ShaderNodePlug &nodeAlpha) {
        optixu::Scene optixScene = m_context.getOptiXScene();
        CUcontext cuContext = m_context.getCUcontext();
        CUstream cuStream = 0;
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());

        MaterialGroup matGroup;
        CompensatedSum<float> sumImportances(0.0f);
        {
            matGroup.indices = std::move(indices);
            uint32_t numTriangles = (uint32_t)matGroup.indices.size() / 3;

            matGroup.optixIndexBuffer.initialize(cuContext, g_bufferType, numTriangles);

            std::vector<float> areas;
            areas.resize(numTriangles);
            {
                auto dstTriangles = matGroup.optixIndexBuffer.map(cuStream, cudau::BufferMapFlag::WriteOnlyDiscard);
                for (auto i = 0; i < numTriangles; ++i) {
                    uint32_t i0 = matGroup.indices[3 * i + 0];
                    uint32_t i1 = matGroup.indices[3 * i + 1];
                    uint32_t i2 = matGroup.indices[3 * i + 2];

                    dstTriangles[i] = shared::Triangle{ i0, i1, i2 };

                    const Vertex (&v)[3] = { m_vertices[i0], m_vertices[i1], m_vertices[i2] };
                    areas[i] = std::fmax(0.0f, 0.5f * cross(v[1].position - v[0].position,
                                                            v[2].position - v[0].position).length());
                    sumImportances += areas[i];
                }
                matGroup.optixIndexBuffer.unmap();
            }

            if (material->isEmitting())
                matGroup.primDist.initialize(m_context, areas.data(), areas.size());
        }

        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;

        matGroup.material = material;
        if (nodeNormal.node) {
            if (shared::NodeTypeInfo<Normal3D>::ConversionIsDefinedFrom(nodeNormal.getType()))
                plugNormal = nodeNormal;
            else
                vlrprintf("%s: Invalid plug type for normal is passed.\n", m_name.c_str());
        }
        matGroup.nodeNormal = plugNormal;
        if (nodeTangent.node) {
            if (shared::NodeTypeInfo<Vector3D>::ConversionIsDefinedFrom(nodeTangent.getType()))
                plugTangent = nodeTangent;
            else
                vlrprintf("%s: Invalid plug type for tangent is passed.\n", m_name.c_str());
        }
        matGroup.nodeTangent = plugTangent;
        if (nodeAlpha.node) {
            if (shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(nodeAlpha.getType()))
                plugAlpha = nodeAlpha;
            else
                vlrprintf("%s: Invalid plug type for alpha is passed.\n", m_name.c_str());
        }
        matGroup.nodeAlpha = plugAlpha;

        optixu::Material optixMaterial = plugAlpha.isValid() ?
            m_context.getOptiXMaterialWithAlpha() :
            m_context.getOptiXMaterialDefault();

        SHGeometryInstance* shGeomInst = new SHGeometryInstance;

        shGeomInst->data.asTriMesh.vertexBuffer = m_optixVertexBuffer.getDevicePointer();
        shGeomInst->data.asTriMesh.triangleBuffer = matGroup.optixIndexBuffer.getDevicePointer();
        matGroup.primDist.getInternalType(&shGeomInst->data.asTriMesh.primDistribution);
        shGeomInst->data.progSample = progSet.dcSampleTriangleMesh;
        shGeomInst->data.progDecodeHitPoint = progSet.dcDecodeHitPointForTriangle;
        shGeomInst->data.nodeNormal = matGroup.nodeNormal.getSharedType();
        shGeomInst->data.nodeTangent = matGroup.nodeTangent.getSharedType();
        shGeomInst->data.nodeAlpha = matGroup.nodeAlpha.getSharedType();
        shGeomInst->data.materialIndex = material->getMaterialIndex();
        shGeomInst->data.importance = material->isEmitting() ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。

        shGeomInst->optixGeomInst = optixScene.createGeometryInstance();
        shGeomInst->optixGeomInst.setVertexBuffer(m_optixVertexBuffer);
        shGeomInst->optixGeomInst.setTriangleBuffer(matGroup.optixIndexBuffer);
        shGeomInst->optixGeomInst.setNumMaterials(1, optixu::BufferView());
        shGeomInst->optixGeomInst.setMaterial(0, 0, optixMaterial);
        shGeomInst->optixGeomInst.setUserData(shGeomInst->data);
        shGeomInst->optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);

        shGeomInst->geomInstIndex = m_context.allocateGeometryInstance();
        m_context.updateGeometryInstance(shGeomInst->geomInstIndex, shGeomInst->data);

        matGroup.shGeomInst = shGeomInst;

        m_materialGroups.push_back(std::move(matGroup));

        // JP: 親にジオメトリインスタンスの追加を行わせる。
        std::set<const SHGeometryInstance*> delta;
        delta.insert(shGeomInst);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryAddEvent(delta);
        }
    }



    std::map<uint32_t, InfiniteSphereSurfaceNode::OptiXProgramSet> InfiniteSphereSurfaceNode::s_optiXProgramSets;

    // static
    void InfiniteSphereSurfaceNode::initialize(Context &context) {
        optixu::Pipeline optixPipeline = context.getOptixPipeline();

        OptiXProgramSet programSet;
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/infinite_sphere_intersection.ptx");
        programSet.optixModule = optixPipeline.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
            VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));
        programSet.dcDecodeHitPointForInfiniteSphere = context.createDirectCallableProgram(
            programSet.optixModule, RT_DC_NAME_STR("decodeHitPointForInfiniteSphere"));
        programSet.dcSampleInfiniteSphere = context.createDirectCallableProgram(
            programSet.optixModule, RT_DC_NAME_STR("sampleInfiniteSphere"));

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void InfiniteSphereSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = s_optiXProgramSets.at(context.getID());
        context.destroyDirectCallableProgram(programSet.dcSampleInfiniteSphere);
        context.destroyDirectCallableProgram(programSet.dcDecodeHitPointForInfiniteSphere);
        s_optiXProgramSets.erase(context.getID());
    }

    InfiniteSphereSurfaceNode::InfiniteSphereSurfaceNode(Context &context, const std::string &name, EnvironmentEmitterSurfaceMaterial* material) :
        SurfaceNode(context, name), m_material(material) {
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        optixu::Scene optixScene = m_context.getOptiXScene();

        m_shGeomInst = new SHGeometryInstance();

        m_material->getImportanceMap().getInternalType(&m_shGeomInst->data.asInfSphere.importanceMap);
        m_shGeomInst->data.progSample = progSet.dcSampleInfiniteSphere;
        m_shGeomInst->data.progDecodeHitPoint = progSet.dcDecodeHitPointForInfiniteSphere;
        m_shGeomInst->data.materialIndex = material->getMaterialIndex();
        m_shGeomInst->data.importance = material->isEmitting() ? 1.0f : 0.0f; // TODO

        m_shGeomInst->optixGeomInst = optixScene.createGeometryInstance();
        m_shGeomInst->optixGeomInst.setNumMaterials(1, optixu::BufferView());
        m_shGeomInst->optixGeomInst.setMaterial(0, 0, m_context.getOptiXMaterialDefault());
        m_shGeomInst->optixGeomInst.setUserData(m_shGeomInst->data);
        m_shGeomInst->optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
    }

    InfiniteSphereSurfaceNode::~InfiniteSphereSurfaceNode() {
        m_shGeomInst->optixGeomInst.destroy();
        delete m_shGeomInst;
    }

    void InfiniteSphereSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        delta.insert(m_shGeomInst);

        parent->geometryAddEvent(delta);
    }

    void InfiniteSphereSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの削除を行わせる。
        // EN: 
        std::set<const SHGeometryInstance*> delta;
        delta.insert(m_shGeomInst);

        parent->geometryRemoveEvent(delta);
    }    



    ParentNode::ParentNode(Context &context, const std::string &name, const Transform* localToWorld) :
        Node(context, name), m_serialChildID(0), m_localToWorld(localToWorld) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: Create a SHTransform having Transform of this node.
        if (m_localToWorld->isStatic()) {
            auto tr = (const StaticTransform*)m_localToWorld;
            m_shTransforms[nullptr] = new SHTransform(name, m_context, *tr, nullptr);
        }
        else {
            VLRAssert_NotImplemented();
        }

        optixu::Scene optixScene = m_context.getOptiXScene();
        optixu::GeometryAccelerationStructure optixGas = optixScene.createGeometryAccelerationStructure();
        optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
        optixGas.setNumMaterialSets(1);
        optixGas.setNumRayTypes(0, shared::RayType::NumTypes);
        m_shGeomGroup = new SHGeometryGroup(optixGas);
        m_shTransforms.at(nullptr)->setChild(m_shGeomGroup);
    }

    ParentNode::~ParentNode() {
        delete m_shGeomGroup;

        for (auto it = m_shTransforms.crbegin(); it != m_shTransforms.crend(); ++it)
            delete it->second;
        m_shTransforms.clear();
    }

    void ParentNode::setName(const std::string &name) {
        Node::setName(name);
        for (auto it = m_shTransforms.begin(); it != m_shTransforms.end(); ++it)
            it->second->setName(name);
    }

    void ParentNode::createConcatanatedTransforms(const std::set<SHTransform*> &childDelta, std::set<SHTransform*>* delta) {
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

    void ParentNode::removeConcatanatedTransforms(const std::set<SHTransform*> &childDelta, std::set<SHTransform*>* delta) {
        // JP: 
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            SHTransform* shtr = m_shTransforms.at(*it);
            m_shTransforms.erase(*it);
            if (delta)
                delta->insert(shtr);
        }
    }

    void ParentNode::updateConcatanatedTransforms(const std::set<SHTransform*> &childDelta, std::set<SHTransform*>* delta) {
        // JP: 
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            SHTransform* shtr = m_shTransforms.at(*it);
            shtr->update();
            if (delta)
                delta->insert(shtr);
        }
    }

    void ParentNode::addToGeometryGroup(const std::set<const SHGeometryInstance*> &childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup->addChild(*it);
    }

    void ParentNode::removeFromGeometryGroup(const std::set<const SHGeometryInstance*> &childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup->removeChild(*it);
    }

    void ParentNode::updateGeometryGroup(const std::set<const SHGeometryInstance*> &childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup->updateChild(*it);
    }

    void ParentNode::geometryAddEvent(const std::set<const SHGeometryInstance*> &childDelta) {
        addToGeometryGroup(childDelta);

        geometryAddEvent(nullptr);
    }

    void ParentNode::geometryRemoveEvent(const std::set<const SHGeometryInstance*> &childDelta) {
        geometryRemoveEvent(nullptr);

        removeFromGeometryGroup(childDelta);
    }

    void ParentNode::geometryUpdateEvent(const std::set<const SHGeometryInstance*> &childDelta) {
        updateGeometryGroup(childDelta);

        geometryUpdateEvent(nullptr);
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

    void ParentNode::prepareSetup(size_t* asScratchSize) {
        *asScratchSize = 0;
        size_t tempAsScratchSize;
        m_shGeomGroup->prepareSetup(&tempAsScratchSize);
        *asScratchSize = std::max(tempAsScratchSize, *asScratchSize);
        for (auto it = m_childToSerialIDMap.cbegin(); it != m_childToSerialIDMap.cend(); ++it) {
            it->first->prepareSetup(&tempAsScratchSize);
            *asScratchSize = std::max(tempAsScratchSize, *asScratchSize);
        }
    }

    void ParentNode::setup(CUstream cuStream, const cudau::Buffer &asScratchMem, shared::PipelineLaunchParameters* launchParams) {
        for (auto it = m_childToSerialIDMap.cbegin(); it != m_childToSerialIDMap.cend(); ++it)
            it->first->setup(cuStream, asScratchMem, launchParams);
        m_shGeomGroup->setup(cuStream, asScratchMem);
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

    void InternalNode::geometryAddEvent(const SHTransform* childTransform) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryAddEvent(transform);
        }
    }

    void InternalNode::geometryRemoveEvent(const SHTransform* childTransform) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryRemoveEvent(transform);
        }
    }

    void InternalNode::geometryUpdateEvent(const SHTransform* childTransform) {
        SHTransform* transform = m_shTransforms.at(childTransform);

        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryUpdateEvent(transform);
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
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            parent->geometryAddEvent(it->second);
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 子孫が持つSHGeometryInstanceの削除を親に伝える。
        // EN: 
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            parent->geometryRemoveEvent(it->second);

        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);

        // JP: 追加した親に対して「自身のSHTransform + 管理中の下位との連結SHTransform」の削除を行わせる。
        // EN: 
        parent->transformRemoveEvent(delta);
    }



    RootNode::RootNode(Context &context, const Transform* localToWorld) :
        ParentNode(context, "Root", localToWorld) {
        optixu::Scene optixScene = m_context.getOptiXScene();

        {
            SHTransform* shtr = m_shTransforms.at(nullptr);

            StaticTransform xfm = shtr->getStaticTransform();
            float mat[16], invMat[16];
            xfm.getArrays(mat, invMat);

            Instance &inst = m_instances[shtr];
            inst.instIndex = m_context.allocateInstance();
            inst.optixInst = optixScene.createInstance();
            inst.optixInst.setID(inst.instIndex);
            inst.data.transform = shared::StaticTransform(Matrix4x4(mat), Matrix4x4(invMat));
            inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            inst.data.geomInstIndices = nullptr;
            m_context.updateInstance(inst.instIndex, inst.data);
        }

        m_optixIas = optixScene.createInstanceAccelerationStructure();
        m_optixIas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    }

    RootNode::~RootNode() {
        m_optixInstanceBuffer.finalize();
        m_optixIasMem.finalize();
        m_optixIas.destroy();

        const SHTransform* shtr = m_shTransforms.at(nullptr);

        Instance &inst = m_instances.at(shtr);
        inst.lightGeomInstDistribution.finalize(m_context);
        inst.geomInstIndices.finalize();
        inst.optixInst.destroy();
        m_context.releaseInstance(inst.instIndex);
        m_instances.erase(shtr);
    }

    void RootNode::transformAddEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> concatDelta;
        createConcatanatedTransforms(childDelta, &concatDelta);
        VLRAssert(childDelta.size() == concatDelta.size(), "The number of elements must match.");

        optixu::Scene optixScene = m_context.getOptiXScene();

        // JP: 
        // EN: 
        for (auto it = concatDelta.cbegin(); it != concatDelta.cend(); ++it) {
            const SHTransform* shtr = *it;

            StaticTransform xfm = shtr->getStaticTransform();
            float mat[16], invMat[16];
            xfm.getArrays(mat, invMat);

            float tMat[12];
            tMat[ 0] = mat[ 0]; tMat[ 1] = mat[ 4]; tMat[ 2] = mat[ 8]; tMat[ 3] = mat[12];
            tMat[ 4] = mat[ 1]; tMat[ 5] = mat[ 5]; tMat[ 6] = mat[ 9]; tMat[ 7] = mat[13];
            tMat[ 8] = mat[ 2]; tMat[ 9] = mat[ 6]; tMat[10] = mat[10]; tMat[11] = mat[14];

            Instance &inst = m_instances[shtr];
            inst = {};
            inst.instIndex = m_context.allocateInstance();
            inst.optixInst = optixScene.createInstance();
            inst.optixInst.setID(inst.instIndex);
            inst.optixInst.setTransform(tMat);
            inst.data.transform = shared::StaticTransform(Matrix4x4(mat), Matrix4x4(invMat));
            inst.data.geomInstIndices = nullptr;
            inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            inst.data.importance = 0.0f;
            m_context.updateInstance(inst.instIndex, inst.data);
        }
    }

    void RootNode::transformRemoveEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> concatDelta;
        removeConcatanatedTransforms(childDelta, &concatDelta);
        VLRAssert(childDelta.size() == concatDelta.size(), "The number of elements must match.");

        // JP: SHGroupからSHTransformを削除する。
        // EN: 
        for (auto it = concatDelta.cbegin(); it != concatDelta.cend(); ++it) {
            SHTransform* shtr = *it;

            Instance &inst = m_instances.at(shtr);
            inst.optixInst.destroy();
            m_context.releaseInstance(inst.instIndex);
            m_instances.erase(shtr);
        }

        for (auto it = concatDelta.cbegin(); it != concatDelta.cend(); ++it)
            delete *it;
    }

    void RootNode::transformUpdateEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> delta;
        updateConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: SHTransformを更新する。
        // EN: 
        for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
            SHTransform* shtr = *it;

            StaticTransform xfm = shtr->getStaticTransform();
            float mat[16], invMat[16];
            xfm.getArrays(mat, invMat);

            float tMat[12];
            tMat[0] = mat[0]; tMat[1] = mat[4]; tMat[2] = mat[8]; tMat[3] = mat[12];
            tMat[4] = mat[1]; tMat[5] = mat[5]; tMat[6] = mat[9]; tMat[7] = mat[13];
            tMat[8] = mat[2]; tMat[9] = mat[6]; tMat[10] = mat[10]; tMat[11] = mat[14];
            
            Instance &inst = m_instances.at(shtr);
            inst.optixInst.setTransform(tMat);
        }

        // TODO: 子のいないTransformの更新に対してもdirtyにしてしまう。
        m_optixIas.markDirty();
    }

    void RootNode::geometryAddEvent(const SHTransform* childTransform) {
        CUcontext cuContext = m_context.getCUcontext();

        SHTransform* transform = m_shTransforms.at(childTransform);
        Instance &inst = m_instances.at(transform);

        const SHGeometryGroup* geomGroup;
        if (transform->hasGeometryDescendant(&geomGroup)) {
            uint32_t numGeomInsts = geomGroup->getNumChildren();
            if (numGeomInsts > 0) {
                std::vector<uint32_t> geomInstIndices(numGeomInsts);
                std::vector<float> geomInstImportanceValues(numGeomInsts);
                geomGroup->getGeometryInstanceIndices(geomInstIndices.data());
                geomGroup->getGeometryInstanceImportanceValues(geomInstImportanceValues.data());

                float sumImportances = 0.0f;
                for (float importance : geomInstImportanceValues)
                    sumImportances += importance;

                inst.geomInstIndices.finalize();
                inst.geomInstIndices.initialize(cuContext, g_bufferType, numGeomInsts);
                inst.geomInstIndices.write(geomInstIndices.data(), numGeomInsts);
                inst.lightGeomInstDistribution.finalize(m_context);
                inst.lightGeomInstDistribution.initialize(m_context, geomInstImportanceValues.data(), numGeomInsts);

                inst.data.geomInstIndices = inst.geomInstIndices.getDevicePointer();
                inst.lightGeomInstDistribution.getInternalType(&inst.data.lightGeomInstDistribution);
                inst.data.importance = sumImportances > 0.0f ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
                m_context.updateInstance(inst.instIndex, inst.data);

                inst.optixInst.setChild(geomGroup->getOptixGas());
                //optixu::GeometryInstance geomInst = geomGroup->getOptixGas().getChild(0);
                //optixu::BufferView triBuffer = geomInst.getTriangleBuffer();
                //uint32_t numPrims = triBuffer.numElements();
                //if (numPrims == 2178)
                //    inst.optixInst.setVisibilityMask(0b0001);
                //else if (numPrims == 2)
                //    inst.optixInst.setVisibilityMask(0b0010);
                //else if (numPrims == 3936)
                //    inst.optixInst.setVisibilityMask(0b0100);
                //else if (numPrims == 57152)
                //    inst.optixInst.setVisibilityMask(0b1000);
                m_optixIas.addChild(inst.optixInst);
            }
        }
    }

    void RootNode::geometryRemoveEvent(const SHTransform* childTransform) {
        CUcontext cuContext = m_context.getCUcontext();

        SHTransform* transform = m_shTransforms.at(childTransform);
        Instance &inst = m_instances.at(transform);

        const SHGeometryGroup* geomGroup;
        if (transform->hasGeometryDescendant(&geomGroup)) {
            uint32_t numGeomInsts = geomGroup->getNumChildren();
            if (numGeomInsts > 0) {
                std::vector<uint32_t> geomInstIndices(numGeomInsts);
                std::vector<float> geomInstImportanceValues(numGeomInsts);
                geomGroup->getGeometryInstanceIndices(geomInstIndices.data());
                geomGroup->getGeometryInstanceImportanceValues(geomInstImportanceValues.data());

                float sumImportances = 0.0f;
                for (float importance : geomInstImportanceValues)
                    sumImportances += importance;

                inst.geomInstIndices.finalize();
                inst.geomInstIndices.initialize(cuContext, g_bufferType, numGeomInsts);
                inst.geomInstIndices.write(geomInstIndices.data(), numGeomInsts);
                inst.lightGeomInstDistribution.finalize(m_context);
                inst.lightGeomInstDistribution.initialize(m_context, geomInstImportanceValues.data(), numGeomInsts);

                inst.data.geomInstIndices = inst.geomInstIndices.getDevicePointer();
                inst.lightGeomInstDistribution.getInternalType(&inst.data.lightGeomInstDistribution);
                inst.data.importance = sumImportances > 0.0f ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
                m_context.updateInstance(inst.instIndex, inst.data);
            }
            else {
                m_optixIas.removeChildAt(m_optixIas.findChildIndex(inst.optixInst));

                inst.geomInstIndices.finalize();
                inst.lightGeomInstDistribution.finalize(m_context);

                inst.data.geomInstIndices = nullptr;
                inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
                inst.data.importance = 0.0f;
                m_context.updateInstance(inst.instIndex, inst.data);
            }
        }
        else {
            m_optixIas.removeChildAt(m_optixIas.findChildIndex(inst.optixInst));

            inst.geomInstIndices.finalize();
            inst.lightGeomInstDistribution.finalize(m_context);

            inst.data.importance = 0.0f;
            inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            inst.data.geomInstIndices = nullptr;
            m_context.updateInstance(inst.instIndex, inst.data);
        }
    }

    void RootNode::geometryUpdateEvent(const SHTransform* childTransform) {
        CUcontext cuContext = m_context.getCUcontext();

        SHTransform* transform = m_shTransforms.at(childTransform);
        Instance &inst = m_instances.at(transform);

        const SHGeometryGroup* geomGroup;
        if (transform->hasGeometryDescendant(&geomGroup)) {
            uint32_t numGeomInsts = geomGroup->getNumChildren();
            std::vector<uint32_t> geomInstIndices(numGeomInsts);
            std::vector<float> geomInstImportanceValues(numGeomInsts);
            geomGroup->getGeometryInstanceIndices(geomInstIndices.data());
            geomGroup->getGeometryInstanceImportanceValues(geomInstImportanceValues.data());

            float sumImportances = 0.0f;
            for (float importance : geomInstImportanceValues)
                sumImportances += importance;

            inst.geomInstIndices.finalize();
            inst.geomInstIndices.initialize(cuContext, g_bufferType, numGeomInsts);
            inst.geomInstIndices.write(geomInstIndices.data(), numGeomInsts);
            inst.lightGeomInstDistribution.finalize(m_context);
            inst.lightGeomInstDistribution.initialize(m_context, geomInstImportanceValues.data(), numGeomInsts);

            inst.data.geomInstIndices = inst.geomInstIndices.getDevicePointer();
            inst.lightGeomInstDistribution.getInternalType(&inst.data.lightGeomInstDistribution);
            inst.data.importance = sumImportances > 0.0f ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
            m_context.updateInstance(inst.instIndex, inst.data);
        }
    }

    void RootNode::prepareSetup(size_t* asScratchSize) {
        *asScratchSize = 0;
        size_t tempAsScratchSize;
        ParentNode::prepareSetup(&tempAsScratchSize);
        *asScratchSize = std::max(tempAsScratchSize, *asScratchSize);

        OptixAccelBufferSizes asSizes;
        m_optixIas.prepareForBuild(&asSizes);
        if (!m_optixIasMem.isInitialized() || m_optixIasMem.sizeInBytes() < asSizes.outputSizeInBytes) {
            m_optixInstanceBuffer.finalize();
            m_optixIasMem.finalize();
            CUcontext cuContext = m_optixIas.getContext().getCUcontext();
            m_optixIasMem.initialize(cuContext, g_bufferType, asSizes.outputSizeInBytes, 1);
            m_optixInstanceBuffer.initialize(cuContext, g_bufferType, std::max(m_optixIas.getNumChildren(), 1u));
        }
        *asScratchSize = std::max(std::max(asSizes.tempSizeInBytes, asSizes.tempUpdateSizeInBytes), *asScratchSize);
    }

    void RootNode::setup(CUstream cuStream, const cudau::Buffer &asScratchMem, shared::PipelineLaunchParameters* launchParams) {
        ParentNode::setup(cuStream, asScratchMem, launchParams);

        launchParams->topGroup = m_optixIas.rebuild(cuStream, m_optixInstanceBuffer, m_optixIasMem, asScratchMem);
    }

    void RootNode::getInstanceIndices(uint32_t* indices) const {
        int i = 0;
        for (auto &it : m_instances)
            indices[i++] = it.second.instIndex;
    }

    void RootNode::getInstanceImportanceValues(float* importances) const {
        int i = 0;
        for (auto &it : m_instances)
            importances[i++] = it.second.data.importance;
    }



    std::map<uint32_t, Scene::OptiXProgramSet> Scene::s_optiXProgramSets;

    // static
    void Scene::initialize(Context &context) {
        optixu::Pipeline optixPipeline = context.getOptixPipeline();

        OptiXProgramSet programSet;
        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/infinite_sphere_intersection.ptx");
        programSet.optixModule = optixPipeline.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
            VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));
        programSet.dcSampleInfiniteSphere = context.createDirectCallableProgram(
            programSet.optixModule, RT_DC_NAME_STR("sampleInfiniteSphere"));

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Scene::finalize(Context &context) {
        OptiXProgramSet &programSet = s_optiXProgramSets.at(context.getID());
        context.destroyDirectCallableProgram(programSet.dcSampleInfiniteSphere);
        s_optiXProgramSets.erase(context.getID());
    }
    
    Scene::Scene(Context &context, const Transform* localToWorld) : 
    Object(context), m_rootNode(context, localToWorld), m_matEnv(nullptr) {
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        CUcontext cuContext = m_context.getCUcontext();

        m_geomInstIndex = m_context.allocateGeometryInstance();
        m_geomInstance.progSample = progSet.dcSampleInfiniteSphere;
        m_geomInstance.progDecodeHitPoint = 0xFFFFFFFF;
        m_geomInstance.nodeNormal = shared::ShaderNodePlug::Invalid();
        m_geomInstance.nodeTangent = shared::ShaderNodePlug::Invalid();
        m_geomInstance.nodeAlpha = shared::ShaderNodePlug::Invalid();
        m_geomInstance.materialIndex = 0;
        m_geomInstance.importance = 0.0f;
        m_context.updateGeometryInstance(m_geomInstIndex, m_geomInstance);

        std::vector<uint32_t> geomInstIndices;
        std::vector<float> geomInstImportanceValues;
        geomInstIndices.push_back(m_geomInstIndex);
        geomInstImportanceValues.push_back(1.0f);
        m_instIndex = m_context.allocateInstance();
        m_geomInstIndices.initialize(cuContext, g_bufferType, geomInstIndices);
        m_lightGeomInstDistribution.initialize(m_context, geomInstImportanceValues.data(), geomInstImportanceValues.size());

        m_instance.geomInstIndices = m_geomInstIndices.getDevicePointer();
        m_lightGeomInstDistribution.getInternalType(&m_instance.lightGeomInstDistribution);
        m_instance.rotationPhi = 0.0f;
        m_context.updateInstance(m_instIndex, m_instance);
    }

    Scene::~Scene() {
        m_lightInstDist.finalize(m_context);
        m_lightInstIndices.finalize();

        m_geomInstIndices.finalize();
        m_context.releaseInstance(m_instIndex);
        m_context.releaseGeometryInstance(m_geomInstIndex);
    }

    void Scene::setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv) {
        m_matEnv = matEnv;
        m_instance.importance = matEnv ? 1.0f : 0.0f; // TODO: どこでImportance Map取得する？
        m_context.updateInstance(m_instIndex, m_instance);
    }

    void Scene::setEnvironmentRotation(float rotationPhi) {
        m_instance.rotationPhi = rotationPhi;
        m_context.updateInstance(m_instIndex, m_instance);
    }

    void Scene::prepareSetup(size_t* asScratchSize) {
        m_rootNode.prepareSetup(asScratchSize);
    }

    void Scene::setup(CUstream cuStream, const cudau::Buffer &asScratchMem, shared::PipelineLaunchParameters* launchParams) {
        CUcontext cuContext = m_context.getCUcontext();

        m_rootNode.setup(cuStream, asScratchMem, launchParams);

        uint32_t numInsts = m_rootNode.getNumInstances();
        std::vector<uint32_t> instIndices(numInsts + 1);
        std::vector<float> lightImportances(numInsts + 1);
        instIndices[0] = m_instIndex;
        lightImportances[0] = m_instance.importance;
        if (m_matEnv) {
            m_matEnv->getImportanceMap().getInternalType(&m_geomInstance.asInfSphere.importanceMap);
            m_geomInstance.materialIndex = m_matEnv->getMaterialIndex();
            m_geomInstance.importance = 1.0f;
        }
        else {
            m_geomInstance.materialIndex = 0xFFFFFFFF;
            m_geomInstance.importance = 0.0f;
        }
        m_context.updateGeometryInstance(m_geomInstIndex, m_geomInstance);

        m_rootNode.getInstanceIndices(instIndices.data() + 1);
        m_rootNode.getInstanceImportanceValues(lightImportances.data() + 1);
        m_lightInstIndices.finalize();
        m_lightInstIndices.initialize(cuContext, g_bufferType, instIndices);
        launchParams->instIndices = m_lightInstIndices.getDevicePointer();

        m_lightInstDist.finalize(m_context);
        m_lightInstDist.initialize(m_context, lightImportances.data(), lightImportances.size());
        m_lightInstDist.getInternalType(&launchParams->lightInstDist);

        launchParams->envLightInstIndex = m_instIndex;
    }



    std::map<uint32_t, optixu::Module> Camera::s_optixModules;

    // static
    void Camera::commonInitializeProcedure(Context& context, const char* identifiers[2], OptiXProgramSet* programSet) {
        optixu::Module optixModule = s_optixModules.at(context.getID());

        programSet->dcSampleLensPosition = context.createDirectCallableProgram(optixModule, identifiers[0]);
        programSet->dcSampleIDF = context.createDirectCallableProgram(optixModule, identifiers[1]);
    }

    // static
    void Camera::commonFinalizeProcedure(Context& context, OptiXProgramSet& programSet) {
        if (programSet.dcSampleLensPosition) {
            context.destroyDirectCallableProgram(programSet.dcSampleIDF);
            context.destroyDirectCallableProgram(programSet.dcSampleLensPosition);
        }
    }
    
    // static
    void Camera::initialize(Context &context) {
        optixu::Pipeline optixPipeline = context.getOptixPipeline();

        std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/cameras.ptx");
        s_optixModules[context.getID()] = optixPipeline.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
            VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        PerspectiveCamera::initialize(context);
        EquirectangularCamera::initialize(context);
    }

    // static
    void Camera::finalize(Context &context) {
        EquirectangularCamera::finalize(context);
        PerspectiveCamera::finalize(context);

        s_optixModules[context.getID()].destroy();
    }



    std::vector<ParameterInfo> PerspectiveCamera::ParameterInfos;
    
    std::map<uint32_t, Camera::OptiXProgramSet> PerspectiveCamera::s_optiXProgramSets;

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
            RT_DC_NAME_STR("PerspectiveCamera_sampleLensPosition"),
            RT_DC_NAME_STR("PerspectiveCamera_sampleIDF")
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void PerspectiveCamera::finalize(Context &context) {
        OptiXProgramSet &programSet = s_optiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        s_optiXProgramSets.erase(context.getID());
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

            m_data.fovY = vlr::clamp<float>(values[0], 0.0001f, M_PI * 0.999f);
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

    void PerspectiveCamera::setup(shared::PipelineLaunchParameters* launchParams) const {
        OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        launchParams->perspectiveCamera = m_data;
        launchParams->progSampleLensPosition = progSet.dcSampleLensPosition;
        launchParams->progSampleIDF = progSet.dcSampleIDF;
    }



    std::vector<ParameterInfo> EquirectangularCamera::ParameterInfos;
    
    std::map<uint32_t, Camera::OptiXProgramSet> EquirectangularCamera::s_optiXProgramSets;

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
            RT_DC_NAME_STR("EquirectangularCamera_sampleLensPosition"),
            RT_DC_NAME_STR("EquirectangularCamera_sampleIDF")
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EquirectangularCamera::finalize(Context &context) {
        OptiXProgramSet& programSet = s_optiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        s_optiXProgramSets.erase(context.getID());
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

            m_data.phiAngle = vlr::clamp<float>(values[0], 0.01f, 2 * M_PI);
        }
        else if (testParamName(paramName, "v angle")) {
            if (length != 1)
                return false;

            m_data.thetaAngle = vlr::clamp<float>(values[0], 0.01f, M_PI);
        }
        else {
            return false;
        }

        return true;
    }

    void EquirectangularCamera::setup(shared::PipelineLaunchParameters* launchParams) const {
        OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        launchParams->equirectangularCamera = m_data;
        launchParams->progSampleLensPosition = progSet.dcSampleLensPosition;
        launchParams->progSampleIDF = progSet.dcSampleIDF;
    }
}
