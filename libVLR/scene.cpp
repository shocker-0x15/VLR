#include "scene.h"

namespace vlr {
    // ----------------------------------------------------------------
    // Shallow Hierarchy

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



    std::unordered_map<uint32_t, TriangleMeshSurfaceNode::OptiXProgramSet> TriangleMeshSurfaceNode::s_optiXProgramSets;

    // static
    void TriangleMeshSurfaceNode::initialize(Context &context) {
        OptiXProgramSet programSet;
        programSet.dcDecodeHitPointForTriangle = context.createDirectCallableProgram(
            OptiXModule_Triangle, RT_DC_NAME_STR("decodeHitPointForTriangle"));
        programSet.dcSampleTriangleMesh = context.createDirectCallableProgram(
            OptiXModule_Triangle, RT_DC_NAME_STR("sampleTriangleMesh"));

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
            delete matGroup.shGeomInst;
            matGroup.primDist.finalize(m_context);
            matGroup.optixIndexBuffer.finalize();
        }
        m_optixVertexBuffer.finalize();
    }

    void TriangleMeshSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_materialGroups.cbegin(); it != m_materialGroups.cend(); ++it)
            delta.insert(it->shGeomInst);

        // JP: 追加した親にジオメトリインスタンスの追加を行わせる。
        // EN: Let the new parent add geometry instances.
        parent->addGeometryInstance(delta);
    }

    void TriangleMeshSurfaceNode::removeParent(ParentNode* parent) {
        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_materialGroups.cbegin(); it != m_materialGroups.cend(); ++it)
            delta.insert(it->shGeomInst);

        // JP: 削除する親にジオメトリインスタンスの削除を行わせる。
        // EN: Let the parent being removed remove geometry instances.
        parent->removeGeometryInstance(delta);

        SurfaceNode::removeParent(parent);
    }

    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        CUcontext cuContext = m_context.getCUcontext();
        m_optixVertexBuffer.initialize(cuContext, g_bufferType, m_vertices);

        // TODO: 頂点情報更新時の処理。(IndexBufferとの整合性など)
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(
        std::vector<uint32_t> &&indices, const SurfaceMaterial* material, 
        const ShaderNodePlug &nodeNormal, const ShaderNodePlug& nodeTangent, const ShaderNodePlug &nodeAlpha) {
        CUcontext cuContext = m_context.getCUcontext();

        MaterialGroup matGroup;
        CompensatedSum<float> sumImportances(0.0f);
        {
            matGroup.indices = std::move(indices);
            uint32_t numTriangles = static_cast<uint32_t>(matGroup.indices.size()) / 3;

            matGroup.optixIndexBuffer.initialize(cuContext, g_bufferType, numTriangles);

            std::vector<float> areas;
            areas.resize(numTriangles);
            {
                auto dstTriangles = matGroup.optixIndexBuffer.map(0, cudau::BufferMapFlag::WriteOnlyDiscard);
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
                matGroup.optixIndexBuffer.unmap(0);
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

        SHGeometryInstance* shGeomInst = new SHGeometryInstance;
        shGeomInst->surfNode = this;
        shGeomInst->userData = static_cast<uint32_t>(m_materialGroups.size());

        matGroup.shGeomInst = shGeomInst;
        m_materialGroups.push_back(std::move(matGroup));

        // JP: 親達にジオメトリインスタンスの追加を行わせる。
        // EN: Let parents add geometry instances.
        std::set<const SHGeometryInstance*> delta;
        delta.insert(shGeomInst);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->addGeometryInstance(delta);
        }
    }

    void TriangleMeshSurfaceNode::setupData(
        uint32_t userData,
        optixu::GeometryInstance* optixGeomInst, shared::GeometryInstance* geomInst) const {
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        const MaterialGroup &matGroup = m_materialGroups[userData];

        geomInst->asTriMesh.vertexBuffer = m_optixVertexBuffer.getDevicePointer();
        geomInst->asTriMesh.triangleBuffer = matGroup.optixIndexBuffer.getDevicePointer();
        matGroup.primDist.getInternalType(&geomInst->asTriMesh.primDistribution);
        geomInst->progSample = progSet.dcSampleTriangleMesh;
        geomInst->progDecodeHitPoint = progSet.dcDecodeHitPointForTriangle;
        geomInst->nodeNormal = matGroup.nodeNormal.getSharedType();
        geomInst->nodeTangent = matGroup.nodeTangent.getSharedType();
        geomInst->nodeAlpha = matGroup.nodeAlpha.getSharedType();
        geomInst->materialIndex = matGroup.material->getMaterialIndex();
        geomInst->importance = matGroup.material->isEmitting() ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。

        optixu::Material optixMaterial = matGroup.nodeAlpha.isValid() ?
            m_context.getOptiXMaterialWithAlpha() :
            m_context.getOptiXMaterialDefault();
        optixGeomInst->setVertexBuffer(m_optixVertexBuffer);
        optixGeomInst->setTriangleBuffer(matGroup.optixIndexBuffer);
        optixGeomInst->setNumMaterials(1, optixu::BufferView());
        optixGeomInst->setMaterial(0, 0, optixMaterial);
        optixGeomInst->setUserData(*geomInst);
        optixGeomInst->setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
    }



    std::unordered_map<uint32_t, InfiniteSphereSurfaceNode::OptiXProgramSet> InfiniteSphereSurfaceNode::s_optiXProgramSets;

    // static
    void InfiniteSphereSurfaceNode::initialize(Context &context) {
        OptiXProgramSet programSet;
        programSet.dcDecodeHitPointForInfiniteSphere = context.createDirectCallableProgram(
            OptiXModule_InfiniteSphere, RT_DC_NAME_STR("decodeHitPointForInfiniteSphere"));
        programSet.dcSampleInfiniteSphere = context.createDirectCallableProgram(
            OptiXModule_InfiniteSphere, RT_DC_NAME_STR("sampleInfiniteSphere"));

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
        m_shGeomInst = new SHGeometryInstance;
        m_shGeomInst->surfNode = this;
        m_shGeomInst->userData = 0;
    }

    InfiniteSphereSurfaceNode::~InfiniteSphereSurfaceNode() {
        delete m_shGeomInst;
    }

    void InfiniteSphereSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        std::set<const SHGeometryInstance*> delta;
        delta.insert(m_shGeomInst);

        // JP: 追加した親にジオメトリインスタンスの追加を行わせる。
        // EN: Let the new parent add geometry instances.
        parent->addGeometryInstance(delta);
    }

    void InfiniteSphereSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        std::set<const SHGeometryInstance*> delta;
        delta.insert(m_shGeomInst);

        // JP: 削除する親にジオメトリインスタンスの削除を行わせる。
        // EN: Let the parent being removed remove geometry instances.
        parent->removeGeometryInstance(delta);
    }    

    void InfiniteSphereSurfaceNode::setupData(
        uint32_t userData,
        optixu::GeometryInstance* optixGeomInst, shared::GeometryInstance* geomInst) const {
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());

        m_material->getImportanceMap().getInternalType(&geomInst->asInfSphere.importanceMap);
        geomInst->progSample = progSet.dcSampleInfiniteSphere;
        geomInst->progDecodeHitPoint = progSet.dcDecodeHitPointForInfiniteSphere;
        geomInst->materialIndex = m_material->getMaterialIndex();
        geomInst->importance = m_material->isEmitting() ? 1.0f : 0.0f; // TODO

        optixGeomInst->setNumMaterials(1, optixu::BufferView());
        optixGeomInst->setMaterial(0, 0, m_context.getOptiXMaterialDefault());
        optixGeomInst->setUserData(*geomInst);
        optixGeomInst->setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
    }



    ParentNode::ParentNode(Context &context, const std::string &name, const Transform* localToWorld) :
        Node(context, name), m_localToWorld(localToWorld) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: Create a SHTransform having Transform of this node.
        if (m_localToWorld->isStatic()) {
            auto tr = dynamic_cast<const StaticTransform*>(m_localToWorld);
            m_shTransforms[nullptr] = new SHTransform(name, *tr, nullptr);
        }
        else {
            VLRAssert_NotImplemented();
        }

        m_shGeomGroup = new SHGeometryGroup();
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
        // EN: Create a SHTransform concatenating own transform and a child internal node's transform.
        //     Store it to the dictionary with the child SHTransform as a key.
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                auto tr = dynamic_cast<const StaticTransform*>(m_localToWorld);
                SHTransform* shtr = new SHTransform(m_name, *tr, *it);
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
        // EN: 
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            SHTransform* shtr = m_shTransforms.at(*it);
            m_shTransforms.erase(*it);
            if (delta)
                delta->insert(shtr);
        }
    }

    void ParentNode::updateConcatanatedTransforms(const std::set<SHTransform*> &childDelta, std::set<SHTransform*>* delta) {
        // JP: 
        // EN: 
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
            SHTransform* shtr = m_shTransforms.at(*it);
            shtr->update();
            if (delta)
                delta->insert(shtr);
        }
    }

    void ParentNode::addGeometryInstance(const std::set<const SHGeometryInstance*> &childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup->addChild(*it);

        // JP: このノード(nullptrがこのノードに対応)にジオメトリインスタンスが追加されたことを示す
        //     イベントを発生させる。
        // EN: Trigger an event that indicates geometry instances has been added to this node
        //     (nullptr means this node).
        geometryAddEvent(nullptr, childDelta);
    }

    void ParentNode::removeGeometryInstance(const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: このノード(nullptrがこのノードに対応)からジオメトリインスタンスが削除されたことを示す
        //     イベントを発生させる。
        // EN: Trigger an event that indicates geometry instances has been removed from this node
        //     (nullptr means this node).
        geometryRemoveEvent(nullptr, childDelta);

        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup->removeChild(*it);
    }

    void ParentNode::updateGeometryInstance(const std::set<const SHGeometryInstance*> &childDelta) {
        for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
            m_shGeomGroup->updateChild(*it);

        // JP: このノード(nullptrがこのノードに対応)のジオメトリインスタンスが更新されたことを示す
        //     イベントを発生させる。
        // EN: Trigger an event that indicates geometry instances of this node has been updated
        //     (nullptr means this node).
        geometryUpdateEvent(nullptr, childDelta);
    }

    void ParentNode::setTransform(const Transform* localToWorld) {
        m_localToWorld = localToWorld;

        // JP: 管理中のSHTransformを更新する。
        // EN: update SHTransforms held by this node.
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                auto tr = dynamic_cast<const StaticTransform*>(m_localToWorld);
                SHTransform* shtr = it->second;
                shtr->setTransform(*tr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
    }

    void ParentNode::addChild(InternalNode* child) {
        if (m_children.count(child) > 0)
            return;
        m_children.insert(child);
        m_orderedChildren.push_back(child);
        child->addParent(this);
    }

    void ParentNode::removeChild(InternalNode* child) {
        if (m_children.count(child) == 0)
            return;
        m_children.erase(child);
        auto idx = std::find(m_orderedChildren.cbegin(), m_orderedChildren.cend(), child);
        m_orderedChildren.erase(idx);
        child->removeParent(this);
    }

    void ParentNode::addChild(SurfaceNode* child) {
        if (m_children.count(child) > 0)
            return;
        m_children.insert(child);
        m_orderedChildren.push_back(child);
        child->addParent(this);
    }

    void ParentNode::removeChild(SurfaceNode* child) {
        if (m_children.count(child) == 0)
            return;
        m_children.erase(child);
        auto idx = std::find(m_orderedChildren.cbegin(), m_orderedChildren.cend(), child);
        m_orderedChildren.erase(idx);
        child->removeParent(this);
    }

    uint32_t ParentNode::getNumChildren() const {
        return static_cast<uint32_t>(m_orderedChildren.size());
    }

    void ParentNode::getChildren(Node** children) const {
        uint32_t i = 0;
        for (auto it : m_orderedChildren)
            children[i++] = it;
    }

    Node* ParentNode::getChildAt(uint32_t index) const {
        return m_orderedChildren[index];
    }



    InternalNode::InternalNode(Context &context, const std::string &name, const Transform* localToWorld) :
        ParentNode(context, name, localToWorld) {
    }

    void InternalNode::transformAddEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> delta;
        createConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 親に自分が保持するSHTransformが増えたことを通知(増分を通知)。
        // EN: Notify parents that SHTransforms held by this node has increased (Notify the delta).
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            auto parent = *it;
            parent->transformAddEvent(delta);
        }
    }

    void InternalNode::transformRemoveEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> delta;
        removeConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 親に自分が保持するSHTransformが増えたことを通知(増分を通知)。
        // EN: Notify parents that SHTransforms held by this node has decreased (Notify the delta).
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            auto parent = *it;
            parent->transformRemoveEvent(delta);
        }

        for (auto it = delta.cbegin(); it != delta.cend(); ++it)
            delete *it;
    }

    void InternalNode::transformUpdateEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> delta;
        updateConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 親に自分が保持するSHTransformが更新されたことを通知(変化分を通知)。
        // EN: Notify parents that SHTransforms held by this node has been updated (Notify the delta).
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            auto parent = *it;
            parent->transformUpdateEvent(delta);
        }
    }

    void InternalNode::geometryAddEvent(const SHTransform* childTransform,
                                        const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: 親達にchildTransformに対応するノードにジオメトリインスタンスが追加されたことを伝える。
        //     childTransformがnullptrの場合、このノードを示す。
        // EN: Tell parents that geometry instances has been added to a node corresponding to childTransform.
        //     It indicates this node when childTransform is nullptr.
        SHTransform* transform = m_shTransforms.at(childTransform);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryAddEvent(transform, childDelta);
        }
    }

    void InternalNode::geometryRemoveEvent(const SHTransform* childTransform,
                                           const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: 親達にchildTransformに対応するノードからジオメトリインスタンスが削除されたことを伝える。
        //     childTransformがnullptrの場合、このノードを示す。
        // EN: Tell parents that geometry instances has been removed from a node corresponding to childTransform.
        //     It indicates this node when childTransform is nullptr.
        SHTransform* transform = m_shTransforms.at(childTransform);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryRemoveEvent(transform, childDelta);
        }
    }

    void InternalNode::geometryUpdateEvent(const SHTransform* childTransform,
                                           const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: 親達にchildTransformに対応するノードのジオメトリインスタンスが更新されたことを伝える。
        //     childTransformがnullptrの場合、このノードを示す。
        // EN: Tell parents that geometry instances of a node corresponding to childTransform has been updated.
        //     It indicates this node when childTransform is nullptr.
        SHTransform* transform = m_shTransforms.at(childTransform);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->geometryUpdateEvent(transform, childDelta);
        }
    }

    void InternalNode::setTransform(const Transform* localToWorld) {
        ParentNode::setTransform(localToWorld);

        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);

        // JP: 親達に変形情報が更新されたことを通知する。
        // EN: Notify parents that the transforms has been updated.
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

        // JP: 追加した親に「親自身のTransformとこのノードが管理するSHTransformの連結SHTransform」の追加を行わせる。
        // EN: Let the new parent add
        //     "Concatenated SHTransforms consisting of the parent's transform and
        //     SHTransforms managed by this node".
        parent->transformAddEvent(delta);

        // JP: 親達にこのノード以下に含まれる全てのジオメトリインスタンスを伝える。
        // EN: Tell parents about all the geometry instances contained under this node.
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            SHTransform* shtr = it->second;
            std::set<const SHGeometryInstance*> children;
            const SHGeometryGroup* shGeomGroup;
            if (shtr->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumChildren(); ++i)
                    children.insert(shGeomGroup->childAt(i));
            }
            if (children.size() > 0)
                parent->geometryAddEvent(shtr, children);
        }
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 親達にこのノード以下に含まれる全てのジオメトリインスタンスを伝える。
        // EN: Tell parents about all the geometry instances contained under this node.
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            SHTransform* shtr = it->second;
            std::set<const SHGeometryInstance*> children;
            const SHGeometryGroup* shGeomGroup;
            if (shtr->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumChildren(); ++i)
                    children.insert(shGeomGroup->childAt(i));
            }
            parent->geometryRemoveEvent(shtr, children);
        }

        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);

        // JP: 削除される親に「親自身のTransformとこのノードが管理するSHTransformの連結SHTransform」の削除を行わせる。
        // EN: Let the parent being removed remove
        //     "Concatenated SHTransforms consisting of the parent's transform and
        //     SHTransforms managed by this node".
        parent->transformRemoveEvent(delta);
    }



    std::unordered_map<uint32_t, Scene::OptiXProgramSet> Scene::s_optiXProgramSets;

    // static
    void Scene::initialize(Context &context) {
        OptiXProgramSet programSet;
        programSet.dcSampleInfiniteSphere = context.createDirectCallableProgram(
            OptiXModule_InfiniteSphere, RT_DC_NAME_STR("sampleInfiniteSphere"));

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Scene::finalize(Context &context) {
        OptiXProgramSet &programSet = s_optiXProgramSets.at(context.getID());
        context.destroyDirectCallableProgram(programSet.dcSampleInfiniteSphere);
        s_optiXProgramSets.erase(context.getID());
    }
    
    Scene::Scene(Context &context, const Transform* localToWorld) :
        ParentNode(context, "Root", localToWorld),
        m_iasIsDirty(true),
        m_matEnv(nullptr) {
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        CUcontext cuContext = m_context.getCUcontext();

        m_geomInstBuffer.initialize(cuContext, 65536);
        m_instBuffer.initialize(cuContext, 65536);

        m_optixScene = m_context.getOptiXContext().createScene();

        // Environmental Light
        {
            m_envGeomInstIndex = m_geomInstBuffer.allocate();

            m_envGeomInstance.progSample = progSet.dcSampleInfiniteSphere;
            m_envGeomInstance.progDecodeHitPoint = 0xFFFFFFFF;
            m_envGeomInstance.nodeNormal = shared::ShaderNodePlug::Invalid();
            m_envGeomInstance.nodeTangent = shared::ShaderNodePlug::Invalid();
            m_envGeomInstance.nodeAlpha = shared::ShaderNodePlug::Invalid();
            m_envGeomInstance.materialIndex = 0;
            m_envGeomInstance.importance = 0.0f;

            m_envInstIndex = m_instBuffer.allocate();

            std::vector<uint32_t> envGeomInstIndices;
            std::vector<float> envGeomInstImportanceValues;
            envGeomInstIndices.push_back(m_envGeomInstIndex);
            envGeomInstImportanceValues.push_back(1.0f);
            m_envGeomInstIndices.initialize(cuContext, g_bufferType, envGeomInstIndices);
            m_envLightGeomInstDistribution.initialize(
                m_context, envGeomInstImportanceValues.data(), envGeomInstImportanceValues.size());

            m_envInstance.geomInstIndices = m_envGeomInstIndices.getDevicePointer();
            m_envLightGeomInstDistribution.getInternalType(&m_envInstance.lightGeomInstDistribution);
            m_envInstance.rotationPhi = 0.0f;
        }

        // JP: ルートノード直属のインスタンスの初期化。
        // EN: Initialize the instance which directly belongs to the root node.
        {
            SHTransform* shtr = m_shTransforms.at(nullptr);

            StaticTransform xfm = shtr->getStaticTransform();
            float mat[16], invMat[16];
            xfm.getArrays(mat, invMat);

            Instance &inst = m_instances[shtr];
            inst.instIndex = m_instBuffer.allocate();
            inst.optixInst = m_optixScene.createInstance();
            inst.optixInst.setID(inst.instIndex);
            inst.data.transform = shared::StaticTransform(Matrix4x4(mat), Matrix4x4(invMat));
            inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            inst.data.geomInstIndices = nullptr;
        }

        m_ias = m_optixScene.createInstanceAccelerationStructure();
        m_ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    }

    Scene::~Scene() {
        m_lightInstDist.finalize(m_context);
        m_lightInstIndices.finalize();

        m_instanceBuffer.finalize();
        m_iasMem.finalize();
        m_ias.destroy();

        const SHTransform* shtr = m_shTransforms.at(nullptr);

        {
            Instance &inst = m_instances.at(shtr);
            inst.lightGeomInstDistribution.finalize(m_context);
            inst.geomInstIndices.finalize();
            inst.optixInst.destroy();
            m_instBuffer.release(inst.instIndex);
            m_instances.erase(shtr);
        }

        {
            m_envLightGeomInstDistribution.finalize(m_context);
            m_envGeomInstIndices.finalize();
            m_instBuffer.release(m_envInstIndex);
            m_geomInstBuffer.release(m_envGeomInstIndex);
        }

        m_optixScene.destroy();

        m_instBuffer.finalize();
        m_geomInstBuffer.finalize();
    }

    void Scene::transformAddEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> concatDelta;
        createConcatanatedTransforms(childDelta, &concatDelta);
        VLRAssert(childDelta.size() == concatDelta.size(), "The number of elements must match.");

        // JP: 追加されたトランスフォームパスに対応するインスタンスを生成する。
        // EN: Create instances corresponding to added transform paths.
        for (auto it = concatDelta.cbegin(); it != concatDelta.cend(); ++it) {
            const SHTransform* shtr = *it;
            Instance &inst = m_instances[shtr];
            inst = {};
            inst.instIndex = m_instBuffer.allocate();
            inst.optixInst = m_optixScene.createInstance();
            inst.optixInst.setID(inst.instIndex);
            m_dirtyInstances.insert(shtr);
        }

        m_iasIsDirty = true;
    }

    void Scene::transformRemoveEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> concatDelta;
        removeConcatanatedTransforms(childDelta, &concatDelta);
        VLRAssert(childDelta.size() == concatDelta.size(), "The number of elements must match.");

        // JP: 削除されるトランスフォームパスに対応するインスタンスを削除する。
        // EN: Destory instances corresponding to transform paths being removed.
        for (auto it = concatDelta.cbegin(); it != concatDelta.cend(); ++it) {
            SHTransform* shtr = *it;

            Instance &inst = m_instances.at(shtr);
            inst.optixInst.destroy();
            m_instBuffer.release(inst.instIndex);
            m_instances.erase(shtr);
        }

        for (auto it = concatDelta.cbegin(); it != concatDelta.cend(); ++it)
            delete *it;

        m_iasIsDirty = true;
    }

    void Scene::transformUpdateEvent(const std::set<SHTransform*> &childDelta) {
        std::set<SHTransform*> delta;
        updateConcatanatedTransforms(childDelta, &delta);
        VLRAssert(childDelta.size() == delta.size(), "The number of elements must match.");

        // JP: 更新されたトランスフォームパスに対応するインスタンスをdirtyとしてマークする。
        // EN: Mark instances corresponding to updated transform paths as dirty.
        for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
            SHTransform* shtr = *it;
            m_dirtyInstances.insert(shtr);
        }

        m_iasIsDirty = true;
    }

    void Scene::geometryAddEvent(const SHTransform* childTransform,
                                 const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: 追加されたジオメトリインスタンスをOptiX上に生成する。
        // EN: Create the added geometry instances on OptiX.
        for (const SHGeometryInstance* shGeomInst : childDelta) {
            GeometryInstance &geomInst = m_geometryInstances[shGeomInst];
            geomInst = {};
            geomInst.geomInstIndex = m_geomInstBuffer.allocate();
            geomInst.optixGeomInst = m_optixScene.createGeometryInstance();
            m_dirtyGeometryInstances.insert(shGeomInst);
        }

        // JP: トランスフォームパスに含まれるジオメトリグループに対応するGASにジオメトリインスタンスを追加する。
        // EN: Add geometry instances to the GAS corresponding to a geometry group
        //     contained in the transform path.
        const SHGeometryGroup* shGeomGroup;
        childTransform->hasGeometryDescendant(&shGeomGroup);
        bool isNewGeomGroup = m_geometryASes.count(shGeomGroup) == 0;
        if (isNewGeomGroup) {
            GeometryAS &gas = m_geometryASes[shGeomGroup];
            gas.optixGas = m_optixScene.createGeometryAccelerationStructure();
            gas.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
            gas.optixGas.setNumMaterialSets(1);
            gas.optixGas.setNumRayTypes(0, shared::MaxNumRayTypes);
        }
        GeometryAS &gas = m_geometryASes.at(shGeomGroup);
        for (const SHGeometryInstance* shGeomInst : childDelta) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            gas.optixGas.addChild(geomInst.optixGeomInst);
        }
        m_dirtyGeometryASes.insert(shGeomGroup);

        // JP: トランスフォームパスに対応するインスタンスにGASをセット、IASにインスタンスを追加する。
        // EN: Set the GAS to the instance corresponding to the transform path, then add the instance to the IAS.
        const SHTransform* shtr = m_shTransforms.at(childTransform);
        const Instance &inst = m_instances.at(shtr);
        inst.optixInst.setChild(m_geometryASes.at(shGeomGroup).optixGas);
        if (isNewGeomGroup)
            m_ias.addChild(inst.optixInst);
        m_dirtyInstances.insert(shtr);

        m_iasIsDirty = true;
    }

    void Scene::geometryRemoveEvent(const SHTransform* childTransform,
                                    const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: 削除されるジオメトリインスタンスをOptiX上からも削除する。
        // EN: Remove the geometry instances being removed also from OptiX.
        for (const SHGeometryInstance* shGeomInst : childDelta) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            geomInst.optixGeomInst.destroy();
            m_geomInstBuffer.release(geomInst.geomInstIndex);
            m_geometryInstances.erase(shGeomInst);
        }

        // JP: トランスフォームパスに含まれるジオメトリグループに対応するGASからジオメトリインスタンスを削除する。
        // EN: Remove geometry instances from the GAS corresponding to a geometry group
        //     contained in the transform path.
        const SHGeometryGroup* shGeomGroup = nullptr;
        childTransform->hasGeometryDescendant(&shGeomGroup);
        bool isEmptyGeomGroup =
            (shGeomGroup->getNumChildren() - static_cast<uint32_t>(childDelta.size())) == 0;
        GeometryAS &gas = m_geometryASes.at(shGeomGroup);
        if (isEmptyGeomGroup) {
            gas.optixGasMem.finalize();
            gas.optixGas.destroy();
            m_geometryASes.erase(shGeomGroup);
        }
        else {
            for (const SHGeometryInstance* shGeomInst : childDelta) {
                GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
                gas.optixGas.removeChildAt(gas.optixGas.findChildIndex(geomInst.optixGeomInst));
            }
        }
        m_dirtyGeometryASes.insert(shGeomGroup);

        // JP: トランスフォームパスに対応するインスタンスをIASから削除する。
        // EN: Remove the instance corresponding to the transform path from the IAS.
        const SHTransform* shtr = m_shTransforms.at(childTransform);
        const Instance &inst = m_instances.at(shtr);
        if (isEmptyGeomGroup)
            m_ias.removeChildAt(m_ias.findChildIndex(inst.optixInst));
        m_dirtyInstances.insert(shtr);

        m_iasIsDirty = true;
    }

    void Scene::geometryUpdateEvent(const SHTransform* childTransform,
                                    const std::set<const SHGeometryInstance*> &childDelta) {
        // JP: 更新されるジオメトリインスタンスをdirtyとしてマークする。
        // EN: Mark the geometry instances being updated as dirty.
        for (const SHGeometryInstance* shGeomInst : childDelta)
            m_dirtyGeometryInstances.insert(shGeomInst);

        // JP: トランスフォームパスに含まれるジオメトリグループに対応するGASをdirtyとしてマークする。
        // EN: Mark the GAS corresponding to a geometry group contained in the transform path as dirty.
        const SHGeometryGroup* shGeomGroup;
        childTransform->hasGeometryDescendant(&shGeomGroup);
        m_dirtyGeometryASes.insert(shGeomGroup);

        // JP: トランスフォームパスに対応するインスタンスをdirtyとしてマークする。
        // EN: Mark the instance corresponding to the transform path as dirty.
        const SHTransform* shtr = m_shTransforms.at(childTransform);
        m_dirtyInstances.insert(shtr);

        m_iasIsDirty = true;
    }

    void Scene::prepareSetup(size_t* asScratchSize, optixu::Scene* optixScene) {
        CUcontext cuContext = m_context.getCUcontext();
        *asScratchSize = 0;

        // JP: GPUに送るジオメトリインスタンスのデータをセットアップする。
        // EN: Setup the geometry instance data sent to a GPU.
        for (const SHGeometryInstance* shGeomInst : m_dirtyGeometryInstances) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            shGeomInst->surfNode->setupData(shGeomInst->userData, &geomInst.optixGeomInst, &geomInst.data);
        }

        // JP: GASのメモリを初期化していない、もしくはサイズが足りない場合にのみ確保を行う。
        //     既に必要なサイズ以上を確保している場合は何もしない。
        // EN: GAS memory allocation is done only when it is not initialized or the size is insufficient.
        for (const SHGeometryGroup* shGeomGroup : m_dirtyGeometryASes) {
            GeometryAS &gas = m_geometryASes.at(shGeomGroup);
            OptixAccelBufferSizes asSizes;
            gas.optixGas.prepareForBuild(&asSizes);
            if (asSizes.outputSizeInBytes > 0 &&
                (!gas.optixGasMem.isInitialized() ||
                 gas.optixGasMem.sizeInBytes() < asSizes.outputSizeInBytes)) {
                gas.optixGasMem.finalize();
                gas.optixGasMem.initialize(cuContext, g_bufferType, asSizes.outputSizeInBytes, 1);
            }
            *asScratchSize = std::max<size_t>({
                *asScratchSize, asSizes.tempSizeInBytes, asSizes.tempUpdateSizeInBytes });
        }

        // JP: IASのメモリを初期化していない、もしくはサイズが足りない場合にのみ確保を行う。
        //     既に必要なサイズ以上を確保している場合は何もしない。
        // EN: IAS memory allocation is done only when it is not initialized or the size is insufficient.
        if (m_iasIsDirty) {
            OptixAccelBufferSizes asSizes;
            m_ias.prepareForBuild(&asSizes);
            if (!m_iasMem.isInitialized() || m_iasMem.sizeInBytes() < asSizes.outputSizeInBytes) {
                m_iasMem.finalize();
                m_iasMem.initialize(cuContext, g_bufferType, asSizes.outputSizeInBytes, 1);
                m_instanceBuffer.finalize();
                m_instanceBuffer.initialize(cuContext, g_bufferType, std::max(m_ias.getNumChildren(), 1u));
            }
            *asScratchSize = std::max<size_t>({
                *asScratchSize, asSizes.tempSizeInBytes, asSizes.tempUpdateSizeInBytes });
        }

        *optixScene = m_optixScene;
    }

    void Scene::setup(
        CUstream stream,
        const cudau::Buffer &asScratchMem, shared::PipelineLaunchParameters* launchParams) {
        CUcontext cuContext = m_context.getCUcontext();

        launchParams->geomInstBuffer = m_geomInstBuffer.optixBuffer.getDevicePointer();
        launchParams->instBuffer = m_instBuffer.optixBuffer.getDevicePointer();

        // JP: ジオメトリインスタンスのデータをGPUに転送する。
        // EN: Transfer the geometry instance data to a GPU.
        for (const SHGeometryInstance* shGeomInst : m_dirtyGeometryInstances) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            m_geomInstBuffer.update(geomInst.geomInstIndex, geomInst.data, stream);
        }
        m_dirtyGeometryInstances.clear();

        // JP: インスタンスのデータをセットアップしてGPUに転送する。
        // EN: Setup the instance data, then transfer them to a GPU.
        bool instancesAreUpdated = m_dirtyInstances.size() > 0;
        for (const SHTransform* shtr : m_dirtyInstances) {
            StaticTransform xfm = shtr->getStaticTransform();
            float mat[16], invMat[16];
            xfm.getArrays(mat, invMat);

            float tMat[12];
            tMat[0] = mat[0]; tMat[1] = mat[4]; tMat[2] = mat[8]; tMat[3] = mat[12];
            tMat[4] = mat[1]; tMat[5] = mat[5]; tMat[6] = mat[9]; tMat[7] = mat[13];
            tMat[8] = mat[2]; tMat[9] = mat[6]; tMat[10] = mat[10]; tMat[11] = mat[14];

            Instance &inst = m_instances.at(shtr);
            inst.optixInst.setTransform(tMat);
            inst.data.transform = shared::StaticTransform(Matrix4x4(mat), Matrix4x4(invMat));
            inst.data.geomInstIndices = nullptr;
            inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            inst.data.importance = 0.0f;

            // JP: インスタンスに含まれるジオメトリインスタンスの重要度分布をセットアップする。
            // EN: Setup importance distribution of geometry instances contained in the instance.
            float sumImportances = 0.0f;
            const SHGeometryGroup* shGeomGroup;
            uint32_t numGeomInsts = 0;
            if (shtr->hasGeometryDescendant(&shGeomGroup))
                numGeomInsts = shGeomGroup->getNumChildren();
            if (numGeomInsts > 0) {
                std::vector<uint32_t> geomInstIndices;
                std::vector<float> geomInstImportanceValues;
                for (int i = 0; i < numGeomInsts; ++i) {
                    const SHGeometryInstance* shGeomInst = shGeomGroup->childAt(i);
                    GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
                    geomInstIndices.push_back(geomInst.geomInstIndex);
                    geomInstImportanceValues.push_back(geomInst.data.importance);
                }

                for (float importance : geomInstImportanceValues)
                    sumImportances += importance;

                inst.geomInstIndices.finalize();
                inst.geomInstIndices.initialize(cuContext, g_bufferType, numGeomInsts);
                inst.geomInstIndices.write(geomInstIndices.data(), numGeomInsts, stream);
                inst.lightGeomInstDistribution.finalize(m_context);
                inst.lightGeomInstDistribution.initialize(m_context, geomInstImportanceValues.data(), numGeomInsts);

                inst.data.geomInstIndices = inst.geomInstIndices.getDevicePointer();
                inst.lightGeomInstDistribution.getInternalType(&inst.data.lightGeomInstDistribution);
                inst.data.importance = sumImportances > 0.0f ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
            }

            m_instBuffer.update(inst.instIndex, inst.data, stream);
        }
        m_dirtyInstances.clear();

        // JP: dirtyとしてマークされているGASをビルドする。
        // EN: Build GASes marked as dirty.
        for (const SHGeometryGroup* shGeomGroup : m_dirtyGeometryASes) {
            GeometryAS &gas = m_geometryASes.at(shGeomGroup);
            gas.optixGas.rebuild(stream, gas.optixGasMem, asScratchMem);
        }
        m_dirtyGeometryASes.clear();

        // JP: IASがdirtyな場合はビルドを行う。
        // EN: Build the IAS when marked as dirty.
        if (m_iasIsDirty)
            m_ias.rebuild(stream, m_instanceBuffer, m_iasMem, asScratchMem);
        m_iasIsDirty = false;

        launchParams->topGroup = m_ias.getHandle();

        // JP: 環境光源に対応するジオメトリインスタンスとインスタンスをセットアップしてGPUに転送する。
        // EN: Setup the geometry instance and instance for the environmental light, then transfer to a GPU.
        if (m_matEnv) {
            m_matEnv->getImportanceMap().getInternalType(&m_envGeomInstance.asInfSphere.importanceMap);
            m_envGeomInstance.materialIndex = m_matEnv->getMaterialIndex();
            m_envGeomInstance.importance = 1.0f;
        }
        else {
            m_envGeomInstance.materialIndex = 0xFFFFFFFF;
            m_envGeomInstance.importance = 0.0f;
        }
        m_geomInstBuffer.update(m_envGeomInstIndex, m_envGeomInstance, stream);
        m_instBuffer.update(m_envInstIndex, m_envInstance, stream);

        // JP: シーンのインスタンスの重要度分布をセットアップしてGPUに転送する。
        // EN: Setup importance distribution of instances of the scene, then transfer to a GPU.
        if (instancesAreUpdated) {
            std::vector<uint32_t> instIndices;
            instIndices.push_back(m_envInstIndex);
            for (auto &it : m_instances)
                instIndices.push_back(it.second.instIndex);
            m_lightInstIndices.finalize();
            m_lightInstIndices.initialize(cuContext, g_bufferType, instIndices);
            launchParams->instIndices = m_lightInstIndices.getDevicePointer();

            std::vector<float> lightImportances;
            lightImportances.push_back(m_envInstance.importance);
            for (auto &it : m_instances)
                lightImportances.push_back(it.second.data.importance);
            m_lightInstDist.finalize(m_context);
            m_lightInstDist.initialize(m_context, lightImportances.data(), lightImportances.size());
            m_lightInstDist.getInternalType(&launchParams->lightInstDist);
        }

        launchParams->envLightInstIndex = m_envInstIndex;
    }

    void Scene::setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv) {
        m_matEnv = matEnv;
        m_envInstance.importance = matEnv ? 1.0f : 0.0f; // TODO: どこでImportance Map取得する？
    }

    void Scene::setEnvironmentRotation(float rotationPhi) {
        m_envInstance.rotationPhi = rotationPhi;
    }



    // static
    void Camera::commonInitializeProcedure(Context& context, const char* identifiers[2], OptiXProgramSet* programSet) {
        programSet->dcSampleLensPosition = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[0]);
        programSet->dcSampleIDF = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[1]);
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
        PerspectiveCamera::initialize(context);
        EquirectangularCamera::initialize(context);
    }

    // static
    void Camera::finalize(Context &context) {
        EquirectangularCamera::finalize(context);
        PerspectiveCamera::finalize(context);
    }



    std::vector<ParameterInfo> PerspectiveCamera::ParameterInfos;
    
    std::unordered_map<uint32_t, Camera::OptiXProgramSet> PerspectiveCamera::s_optiXProgramSets;

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
    
    std::unordered_map<uint32_t, Camera::OptiXProgramSet> EquirectangularCamera::s_optiXProgramSets;

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
