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

    const SHGeometryGroup* SHTransform::getGeometryDescendant() const {
        const SHTransform* nextSHTr = this;
        const SHGeometryGroup* geomGroup = nullptr;
        while (nextSHTr) {
            if (nextSHTr->m_childIsTransform) {
                nextSHTr = nextSHTr->m_childTransform;
            }
            else {
                geomGroup = nextSHTr->m_childGeomGroup;
                nextSHTr = nullptr;
            }
        }

        return geomGroup;
    }

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    // static
    void SurfaceNode::initialize(Context &context) {
        TriangleMeshSurfaceNode::initialize(context);
        PointSurfaceNode::initialize(context);
        InfiniteSphereSurfaceNode::initialize(context);
    }

    // static
    void SurfaceNode::finalize(Context &context) {
        InfiniteSphereSurfaceNode::finalize(context);
        PointSurfaceNode::finalize(context);
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
            BoundingBox3D aabb;
            auto dstTriangles = matGroup.optixIndexBuffer.map(0, cudau::BufferMapFlag::WriteOnlyDiscard);
            for (auto i = 0; i < static_cast<int>(numTriangles); ++i) {
                uint32_t i0 = matGroup.indices[3 * i + 0];
                uint32_t i1 = matGroup.indices[3 * i + 1];
                uint32_t i2 = matGroup.indices[3 * i + 2];

                dstTriangles[i] = shared::Triangle{ i0, i1, i2 };

                const Vertex (&v)[3] = { m_vertices[i0], m_vertices[i1], m_vertices[i2] };
                areas[i] = 0.5f * cross(v[1].position - v[0].position,
                                        v[2].position - v[0].position).length();
                sumImportances += areas[i];

                aabb.unify(v[0].position).unify(v[1].position).unify(v[2].position);
            }
            matGroup.optixIndexBuffer.unmap(0);
            matGroup.aabb = aabb;

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
        geomInst->asTriMesh.aabb = matGroup.aabb;
        geomInst->progSample = progSet.dcSampleTriangleMesh;
        geomInst->progDecodeHitPoint = progSet.dcDecodeHitPointForTriangle;
        geomInst->nodeNormal = matGroup.nodeNormal.getSharedType();
        geomInst->nodeTangent = matGroup.nodeTangent.getSharedType();
        geomInst->nodeAlpha = matGroup.nodeAlpha.getSharedType();
        geomInst->materialIndex = matGroup.material->getMaterialIndex();
        geomInst->importance = matGroup.material->isEmitting() ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
        geomInst->geomType = shared::GeometryType_TriangleMesh;
        geomInst->isActive = true;

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



    std::unordered_map<uint32_t, PointSurfaceNode::OptiXProgramSet> PointSurfaceNode::s_optiXProgramSets;

    // static
    void PointSurfaceNode::initialize(Context &context) {
        OptiXProgramSet programSet;
        programSet.dcSamplePoint = context.createDirectCallableProgram(
            OptiXModule_Point, RT_DC_NAME_STR("samplePoint"));

        s_optiXProgramSets[context.getID()] = programSet;
    }

    // static
    void PointSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = s_optiXProgramSets.at(context.getID());
        context.destroyDirectCallableProgram(programSet.dcSamplePoint);
        s_optiXProgramSets.erase(context.getID());
    }

    PointSurfaceNode::PointSurfaceNode(Context &context, const std::string &name) :
        SurfaceNode(context, name) {}

    PointSurfaceNode::~PointSurfaceNode() {
        for (auto it = m_materialGroups.rbegin(); it != m_materialGroups.rend(); ++it) {
            MaterialGroup &matGroup = *it;
            delete matGroup.shGeomInst;
            matGroup.primDist.finalize(m_context);
            matGroup.optixIndexBuffer.finalize();
        }
        m_optixVertexBuffer.finalize();
    }

    void PointSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_materialGroups.cbegin(); it != m_materialGroups.cend(); ++it)
            delta.insert(it->shGeomInst);

        // JP: 追加した親にジオメトリインスタンスの追加を行わせる。
        // EN: Let the new parent add geometry instances.
        parent->addGeometryInstance(delta);
    }

    void PointSurfaceNode::removeParent(ParentNode* parent) {
        std::set<const SHGeometryInstance*> delta;
        for (auto it = m_materialGroups.cbegin(); it != m_materialGroups.cend(); ++it)
            delta.insert(it->shGeomInst);

        // JP: 削除する親にジオメトリインスタンスの削除を行わせる。
        // EN: Let the parent being removed remove geometry instances.
        parent->removeGeometryInstance(delta);

        SurfaceNode::removeParent(parent);
    }

    void PointSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        CUcontext cuContext = m_context.getCUcontext();
        m_optixVertexBuffer.initialize(cuContext, g_bufferType, m_vertices);

        // TODO: 頂点情報更新時の処理。(IndexBufferとの整合性など)
    }

    void PointSurfaceNode::addMaterialGroup(
        std::vector<uint32_t> &&indices, const SurfaceMaterial* material) {
        CUcontext cuContext = m_context.getCUcontext();

        MaterialGroup matGroup;
        CompensatedSum<float> sumImportances(0.0f);
        {
            matGroup.indices = std::move(indices);
            uint32_t numPoints = static_cast<uint32_t>(matGroup.indices.size());

            matGroup.optixIndexBuffer.initialize(cuContext, g_bufferType, numPoints);

            std::vector<float> importances;
            importances.resize(numPoints);
            {
                auto dstPoints = matGroup.optixIndexBuffer.map(0, cudau::BufferMapFlag::WriteOnlyDiscard);
                for (auto i = 0; i < static_cast<int>(numPoints); ++i) {
                    dstPoints[i] = matGroup.indices[i];
                    importances[i] = 1.0f;
                    sumImportances += importances[i];
                }
                matGroup.optixIndexBuffer.unmap(0);
            }

            if (material->isEmitting())
                matGroup.primDist.initialize(m_context, importances.data(), importances.size());
        }

        matGroup.material = material;

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

    void PointSurfaceNode::setupData(
        uint32_t userData,
        optixu::GeometryInstance* optixGeomInst, shared::GeometryInstance* geomInst) const {
        const OptiXProgramSet &progSet = s_optiXProgramSets.at(m_context.getID());
        const MaterialGroup &matGroup = m_materialGroups[userData];

        geomInst->asPoints.vertexBuffer = m_optixVertexBuffer.getDevicePointer();
        geomInst->asPoints.indexBuffer = matGroup.optixIndexBuffer.getDevicePointer();
        matGroup.primDist.getInternalType(&geomInst->asPoints.primDistribution);
        geomInst->progSample = progSet.dcSamplePoint;
        geomInst->progDecodeHitPoint = 0;
        geomInst->nodeNormal = shared::ShaderNodePlug::Invalid();
        geomInst->nodeTangent = shared::ShaderNodePlug::Invalid();
        geomInst->nodeAlpha = shared::ShaderNodePlug::Invalid();
        geomInst->materialIndex = matGroup.material->getMaterialIndex();
        geomInst->importance = matGroup.material->isEmitting() ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
        geomInst->geomType = shared::GeometryType_Points;
        geomInst->isActive = true;
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
        geomInst->nodeNormal = shared::ShaderNodePlug::Invalid();
        geomInst->nodeTangent = shared::ShaderNodePlug::Invalid();
        geomInst->nodeAlpha = shared::ShaderNodePlug::Invalid();
        geomInst->materialIndex = m_material->getMaterialIndex();
        geomInst->importance = m_material->isEmitting() ? 1.0f : 0.0f; // TODO
        geomInst->geomType = shared::GeometryType_InfiniteSphere;
        geomInst->isActive = true;
    }



    ParentNode::ParentNode(Context &context, const std::string &name, const Transform* localToWorld) :
        Node(context, name), m_localToWorld(localToWorld) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: Create a SHTransform having Transform of this node.
        m_shGeomGroup = new SHGeometryGroup();
        if (m_localToWorld->isStatic()) {
            auto tr = dynamic_cast<const StaticTransform*>(m_localToWorld);
            SHTransform* shtr = new SHTransform(name, *tr, m_shGeomGroup);
            m_shTransforms[nullptr] = shtr;
        }
        else {
            VLRAssert_NotImplemented();
        }
    }

    ParentNode::~ParentNode() {
        for (auto it = m_shTransforms.crbegin(); it != m_shTransforms.crend(); ++it)
            delete it->second;
        m_shTransforms.clear();

        delete m_shGeomGroup;
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

    InternalNode::~InternalNode() {
        for (vlr::Node* child : m_orderedChildren)
            child->removeParent(this);
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
            const SHGeometryGroup* shGeomGroup = shtr->getGeometryDescendant();
            for (int i = 0; i < static_cast<int>(shGeomGroup->getNumChildren()); ++i)
                children.insert(shGeomGroup->childAt(i));
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
            const SHGeometryGroup* shGeomGroup = shtr->getGeometryDescendant();
            for (int i = 0; i < static_cast<int>(shGeomGroup->getNumChildren()); ++i)
                children.insert(shGeomGroup->childAt(i));
            if (children.size() > 0)
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



    // static
    void Scene::initialize(Context &context) {
    }

    // static
    void Scene::finalize(Context &context) {
    }
    
    Scene::Scene(Context &context, const Transform* localToWorld) :
        ParentNode(context, "Root", localToWorld),
        m_iasIsDirty(true),
        m_matEnv(nullptr), m_envNode(nullptr), m_envIsDirty(false) {
        CUcontext cuContext = m_context.getCUcontext();

        shared::GeometryInstance initialGeomInst = {};
        initialGeomInst.isActive = false;
        shared::Instance initialInst = {};
        initialInst.isActive = false;
        m_geomInstBuffer.initialize(cuContext, 65536, initialGeomInst);
        m_instBuffer.initialize(cuContext, 65536, initialInst);

        m_optixScene = m_context.getOptiXContext().createScene();

        static_assert(sizeof(BoundingBox3D) == sizeof(BoundingBox3DAsOrderedInt),
                      "sizeof(BoundingBox3D) and sizeof(BoundingBox3DAsOrderedInt) should match.");
        CUDADRV_CHECK(cuMemAlloc(&m_sceneBounds, sizeof(shared::SceneBounds)));

        {
            m_envGeomInst = {};
            m_envGeomInst.geomInstIndex = m_geomInstBuffer.allocate();
            m_envGeomInst.referenceCount = 1;
            m_envGeomInst.data.geomInstIndex = m_envGeomInst.geomInstIndex;
            m_envGeomInst.data.isActive = false;

            m_envInst = {};
            m_envInst.instIndex = m_instBuffer.allocate();
            m_envInst.optixGeomInstIndices.initialize(cuContext, g_bufferType, 1, m_envGeomInst.geomInstIndex);
            m_envInst.data.rotationPhi = 0.0f;
            m_envInst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            m_envInst.data.geomInstIndices = m_envInst.optixGeomInstIndices.getDevicePointer();
            m_envInst.data.isActive = false;

            m_envIsDirty = true;
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
            inst.optixInst.setVisibilityMask(shared::VisibilityGroup_Everything);
            inst.data.transform = shared::StaticTransform(Matrix4x4(mat), Matrix4x4(invMat));
            inst.data.lightGeomInstDistribution = shared::DiscreteDistribution1D();
            inst.data.geomInstIndices = nullptr;
            inst.data.isActive = false;
        }

        m_ias = m_optixScene.createInstanceAccelerationStructure();
        m_ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    }

    Scene::~Scene() {
        for (vlr::Node* child : m_orderedChildren)
            child->removeParent(this);

        if (m_envNode)
            delete m_envNode;

        m_computeInstAabbs_instIndices.finalize();
        m_computeInstAabbs_itemOffsets.finalize();

        m_lightInstDist.finalize(m_context);
        m_lightInstIndices.finalize();

        m_instanceBuffer.finalize();
        m_iasMem.finalize();
        m_ias.destroy();

        const SHTransform* shtr = m_shTransforms.at(nullptr);

        {
            Instance &inst = m_instances.at(shtr);
            inst.lightGeomInstDistribution.finalize(m_context);
            inst.optixGeomInstIndices.finalize();
            inst.optixInst.destroy();
            m_instBuffer.release(inst.instIndex);
            m_instances.erase(shtr);
        }

        {
            m_envInst.lightGeomInstDistribution.finalize(m_context);
            m_envInst.optixGeomInstIndices.finalize();
            m_instBuffer.release(m_envInst.instIndex);

            m_geomInstBuffer.release(m_envGeomInst.geomInstIndex);
        }

        cuMemFree(m_sceneBounds);

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
            inst.optixInst.setVisibilityMask(shared::VisibilityGroup_Everything);
            inst.data.importance = 0.0f;
            inst.data.isActive = false;
        }
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
            uint32_t instIdxInIas = m_ias.findChildIndex(inst.optixInst);
            if (instIdxInIas != 0xFFFFFFFF)
                m_ias.removeChildAt(instIdxInIas);
            uint32_t instIndex = inst.instIndex;
            inst.optixInst.destroy();
            m_instBuffer.release(instIndex);
            m_instances.erase(shtr);
            m_dirtyInstances.erase(shtr);
            m_removedInstanceIndices.insert(instIndex);
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
            if (m_geometryInstances.count(shGeomInst) > 0) {
                GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
                ++geomInst.referenceCount;
                continue;
            }
            GeometryInstance &geomInst = m_geometryInstances[shGeomInst];
            geomInst = {};
            geomInst.geomInstIndex = m_geomInstBuffer.allocate();
            if (shGeomInst->surfNode->isIntersectable())
                geomInst.optixGeomInst = m_optixScene.createGeometryInstance();
            geomInst.referenceCount = 1;
            m_dirtyGeometryInstances.insert(shGeomInst);
        }

        // JP: トランスフォームパスに含まれるジオメトリグループに対応するGASにジオメトリインスタンスを追加する。
        // EN: Add geometry instances to the GAS corresponding to a geometry group
        //     contained in the transform path.
        const SHTransform* shtr = m_shTransforms.at(childTransform);
        const SHGeometryGroup* shGeomGroup = shtr->getGeometryDescendant();
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
            if (shGeomInst->surfNode->isIntersectable() &&
                gas.optixGas.findChildIndex(geomInst.optixGeomInst) == 0xFFFFFFFF)
                gas.optixGas.addChild(geomInst.optixGeomInst);
        }
        m_dirtyGeometryASes.insert(shGeomGroup);

        // JP: トランスフォームパスに対応するインスタンスにGASをセット、IASにインスタンスを追加する。
        // EN: Set the GAS to the instance corresponding to the transform path, then add the instance to the IAS.
        const Instance &inst = m_instances.at(shtr);
        inst.optixInst.setChild(m_geometryASes.at(shGeomGroup).optixGas);
        if (m_ias.findChildIndex(inst.optixInst) == 0xFFFFFFFF)
            m_ias.addChild(inst.optixInst);
        m_dirtyInstances.insert(shtr);

        m_iasIsDirty = true;
    }

    void Scene::geometryRemoveEvent(const SHTransform* childTransform,
                                    const std::set<const SHGeometryInstance*> &childDelta) {
        const SHTransform* shtr = m_shTransforms.at(childTransform);
        const SHGeometryGroup* shGeomGroup = shtr->getGeometryDescendant();
        GeometryAS &gas = m_geometryASes.at(shGeomGroup);

        // JP: トランスフォームパスに含まれるジオメトリグループに対応するGASからジオメトリインスタンスを削除する。
        //     削除されるジオメトリインスタンスをOptiX上からも削除する。
        // EN: Remove geometry instances from the GAS corresponding to a geometry group
        //     contained in the transform path.
        //     Remove the geometry instances being removed also from OptiX.
        uint32_t numRemovedGeomInsts = 0;
        for (const SHGeometryInstance* shGeomInst : childDelta) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            if (--geomInst.referenceCount > 0)
                continue;
            if (shGeomInst->surfNode->isIntersectable())
                gas.optixGas.removeChildAt(gas.optixGas.findChildIndex(geomInst.optixGeomInst));
            ++numRemovedGeomInsts;

            geomInst.optixGeomInst.destroy();
            uint32_t geomInstIndex = geomInst.geomInstIndex;
            m_geomInstBuffer.release(geomInstIndex);
            m_geometryInstances.erase(shGeomInst);
            m_dirtyGeometryInstances.erase(shGeomInst);
            m_removedGeometryInstanceIndices.insert(geomInstIndex);
        }

        // JP: 空になったGASは削除する。
        // EN: Remove the GAS if empty.
        bool isEmptyGeomGroup = gas.optixGas.getNumChildren() == 0;
        if (isEmptyGeomGroup) {
            gas.optixGasMem.finalize();
            gas.optixGas.destroy();
            m_geometryASes.erase(shGeomGroup);
        }
        else {
            if (numRemovedGeomInsts > 0)
                m_dirtyGeometryASes.insert(shGeomGroup);
        }

        // JP: トランスフォームパスに対応するインスタンスをIASから削除する。
        // EN: Remove the instance corresponding to the transform path from the IAS.
        if (numRemovedGeomInsts > 0) {
            if (isEmptyGeomGroup) {
                Instance &inst = m_instances.at(shtr);
                uint32_t instIdxInIas = m_ias.findChildIndex(inst.optixInst);
                if (instIdxInIas != 0xFFFFFFFF)
                    m_ias.removeChildAt(instIdxInIas);
            }
            m_dirtyInstances.insert(shtr);
        }

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
        const SHTransform* shtr = m_shTransforms.at(childTransform);
        const SHGeometryGroup* shGeomGroup = shtr->getGeometryDescendant();
        m_dirtyGeometryASes.insert(shGeomGroup);

        // JP: トランスフォームパスに対応するインスタンスをdirtyとしてマークする。
        // EN: Mark the instance corresponding to the transform path as dirty.
        m_dirtyInstances.insert(shtr);

        m_iasIsDirty = true;
    }

    void Scene::prepareSetup(size_t* asScratchSize, optixu::Scene* optixScene) {
        CUcontext cuContext = m_context.getCUcontext();
        *asScratchSize = 0;

        if (m_envIsDirty) {
            if (m_envNode)
                m_envNode->setupData(0, nullptr, &m_envGeomInst.data);
        }

        // JP: GPUに送るジオメトリインスタンスのデータをセットアップする。
        // EN: Setup the geometry instance data sent to a GPU.
        for (const SHGeometryInstance* shGeomInst : m_dirtyGeometryInstances) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            shGeomInst->surfNode->setupData(shGeomInst->userData, &geomInst.optixGeomInst, &geomInst.data);
            geomInst.data.geomInstIndex = geomInst.geomInstIndex;
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
                gas.optixGasMem.initialize(
                    cuContext, g_bufferType, static_cast<uint32_t>(asSizes.outputSizeInBytes), 1);
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
                m_iasMem.initialize(
                    cuContext, g_bufferType, static_cast<uint32_t>(asSizes.outputSizeInBytes), 1);
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

        // JP: ジオメトリインスタンスのデータをGPUに転送する。
        // EN: Transfer the geometry instance data to a GPU.
        for (const SHGeometryInstance* shGeomInst : m_dirtyGeometryInstances) {
            GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
            m_geomInstBuffer.update(geomInst.geomInstIndex, geomInst.data, stream);
        }
        m_dirtyGeometryInstances.clear();
        for (uint32_t geomInstIndex : m_removedGeometryInstanceIndices) {
            shared::GeometryInstance initialGeomInst = {};
            initialGeomInst.isActive = false;
            m_geomInstBuffer.update(geomInstIndex, initialGeomInst, stream);
        }
        m_removedGeometryInstanceIndices.clear();

        // JP: インスタンスのデータをセットアップしてGPUに転送する。
        // EN: Setup the instance data, then transfer them to a GPU.
        bool instancesAreUpdated = m_dirtyInstances.size() > 0;
        std::vector<uint32_t> computeInstAabbs_instIndices;
        std::vector<uint32_t> computeInstAabbs_itemOffsets;
        uint32_t computeInstAabbs_itemOffset = 0;
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
            BoundingBox3DAsOrderedInt initAabb;
            inst.data.childAabb = reinterpret_cast<BoundingBox3D &>(initAabb);
            inst.data.aabbIsDirty = true;
            inst.data.importance = 0.0f;

            // JP: インスタンスに含まれるジオメトリインスタンスの重要度分布をセットアップする。
            // EN: Setup importance distribution of geometry instances contained in the instance.
            float sumImportances = 0.0f;
            const SHGeometryGroup* shGeomGroup = shtr->getGeometryDescendant();
            uint32_t numGeomInsts = shGeomGroup->getNumChildren();
            inst.data.isActive = numGeomInsts > 0;
            if (numGeomInsts > 0) {
                std::vector<uint32_t> geomInstIndices;
                std::vector<float> geomInstImportanceValues;
                for (int i = 0; i < static_cast<int>(numGeomInsts); ++i) {
                    const SHGeometryInstance* shGeomInst = shGeomGroup->childAt(i);
                    GeometryInstance &geomInst = m_geometryInstances.at(shGeomInst);
                    geomInstIndices.push_back(geomInst.geomInstIndex);
                    geomInstImportanceValues.push_back(geomInst.data.importance);

                    computeInstAabbs_instIndices.push_back(inst.instIndex);
                    computeInstAabbs_itemOffsets.push_back(computeInstAabbs_itemOffset);
                }
                computeInstAabbs_itemOffset += numGeomInsts;

                for (float importance : geomInstImportanceValues)
                    sumImportances += importance;

                inst.optixGeomInstIndices.finalize();
                inst.optixGeomInstIndices.initialize(
                    cuContext, g_bufferType, geomInstIndices, stream);
                inst.lightGeomInstDistribution.finalize(m_context);
                inst.lightGeomInstDistribution.initialize(
                    m_context, geomInstImportanceValues.data(), numGeomInsts);

                inst.data.geomInstIndices = inst.optixGeomInstIndices.getDevicePointer();
                inst.lightGeomInstDistribution.getInternalType(&inst.data.lightGeomInstDistribution);
                inst.data.importance = sumImportances > 0.0f ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
            }

            m_instBuffer.update(inst.instIndex, inst.data, stream);
        }
        m_dirtyInstances.clear();
        for (uint32_t instIndex : m_removedInstanceIndices) {
            shared::Instance initialInst = {};
            initialInst.isActive = false;
            m_instBuffer.update(instIndex, initialInst, stream);
        }
        m_removedInstanceIndices.clear();

        // JP: インスタンスAABBを計算する。
        // EN: Compute instance AABBs.
        if (computeInstAabbs_itemOffset > 0) {
            if (m_computeInstAabbs_instIndices.numElements() < computeInstAabbs_itemOffset) {
                m_computeInstAabbs_instIndices.finalize();
                m_computeInstAabbs_itemOffsets.finalize();
                m_computeInstAabbs_instIndices.initialize(cuContext, g_bufferType, computeInstAabbs_itemOffset);
                m_computeInstAabbs_itemOffsets.initialize(cuContext, g_bufferType, computeInstAabbs_itemOffset);
            }

            m_computeInstAabbs_instIndices.write(computeInstAabbs_instIndices, stream);
            m_computeInstAabbs_itemOffsets.write(computeInstAabbs_itemOffsets, stream);

            m_context.computeInstanceAABBs(
                stream,
                m_computeInstAabbs_instIndices.getDevicePointer(), m_computeInstAabbs_itemOffsets.getDevicePointer(),
                m_instBuffer.optixBuffer.getDevicePointer(), m_geomInstBuffer.optixBuffer.getDevicePointer(),
                computeInstAabbs_itemOffset);
            m_context.finalizeInstanceAABBs(
                stream,
                m_instBuffer.optixBuffer.getDevicePointer(), m_instBuffer.optixBuffer.numElements());
        }

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

        // JP: 環境光源に対応するジオメトリインスタンスとインスタンスをセットアップしてGPUに転送する。
        // EN: Setup the geometry instance and instance for the environmental light, then transfer to a GPU.
        if (m_envIsDirty) {
            if (m_envNode) {
                float importance = m_envGeomInst.data.importance;
                m_envInst.lightGeomInstDistribution.finalize(m_context);
                m_envInst.lightGeomInstDistribution.initialize(m_context, &importance, 1);
                m_envInst.lightGeomInstDistribution.getInternalType(&m_envInst.data.lightGeomInstDistribution);
                m_envInst.data.importance = importance > 0.0f ? 1.0f : 0.0f; // TODO: 面積やEmitterの特性の考慮。
                m_envInst.data.isActive = true;
            }
            else {
                m_envGeomInst.data.importance = 0.0f;
                m_envGeomInst.data.isActive = false;
                m_envInst.data.importance = 0.0f;
                m_envInst.data.isActive = false;
            }
            m_geomInstBuffer.update(m_envGeomInst.geomInstIndex, m_envGeomInst.data, stream);
            m_instBuffer.update(m_envInst.instIndex, m_envInst.data, stream);
            instancesAreUpdated = true;
        }
        m_envIsDirty = false;

        // JP: シーンのインスタンスの重要度分布をセットアップしてGPUに転送する。
        // EN: Setup importance distribution of instances of the scene, then transfer to a GPU.
        if (instancesAreUpdated) {
            shared::SceneBounds initSceneBounds;
            initSceneBounds.aabbAsInt = BoundingBox3DAsOrderedInt();
            initSceneBounds.center = Point3D(0.0f);
            initSceneBounds.worldRadius = 0.0f;
            initSceneBounds.worldDiscArea = 0.0f;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(m_sceneBounds, &initSceneBounds, sizeof(initSceneBounds), stream));
            m_context.computeSceneAABB(
                stream, m_instBuffer.optixBuffer.getDevicePointer(), m_instBuffer.optixBuffer.numElements(),
                reinterpret_cast<shared::SceneBounds*>(m_sceneBounds));
            m_context.finalizeSceneBounds(
                stream, reinterpret_cast<shared::SceneBounds*>(m_sceneBounds));

            //CUDADRV_CHECK(cuStreamSynchronize(stream));
            //shared::SceneBounds sceneBounds;
            //CUDADRV_CHECK(cuMemcpyDtoH(&sceneBounds, m_sceneBounds, sizeof(sceneBounds)));
            //printf("Scene AABB: (%g, %g, %g) - (%g, %g, %g)\n"
            //       "      Center: (%g, %g, %g)\n"
            //       "      DiscArea: %g\n",
            //       sceneBounds.aabb.minP.x, sceneBounds.aabb.minP.y, sceneBounds.aabb.minP.z,
            //       sceneBounds.aabb.maxP.x, sceneBounds.aabb.maxP.y, sceneBounds.aabb.maxP.z,
            //       sceneBounds.center.x, sceneBounds.center.y, sceneBounds.center.z,
            //       sceneBounds.worldDiscArea);

            std::vector<uint32_t> instIndices;
            instIndices.push_back(m_envInst.instIndex);
            for (auto &it : m_instances)
                instIndices.push_back(it.second.instIndex);
            m_lightInstIndices.finalize();
            m_lightInstIndices.initialize(cuContext, g_bufferType, instIndices, stream);

            std::vector<float> lightImportances;
            lightImportances.push_back(m_envInst.data.importance);
            for (auto &it : m_instances)
                lightImportances.push_back(it.second.data.importance);
            m_lightInstDist.finalize(m_context);
            m_lightInstDist.initialize(m_context, lightImportances.data(), lightImportances.size());
        }

        launchParams->geomInstBuffer = m_geomInstBuffer.optixBuffer.getDevicePointer();
        launchParams->instBuffer = m_instBuffer.optixBuffer.getDevicePointer();
        launchParams->topGroup = m_ias.getHandle();
        launchParams->sceneBounds = reinterpret_cast<shared::SceneBounds*>(m_sceneBounds);
        launchParams->instIndices = m_lightInstIndices.getDevicePointer();
        m_lightInstDist.getInternalType(&launchParams->lightInstDist);
        launchParams->envLightInstIndex = m_envInst.instIndex;
    }

    void Scene::setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv) {
        m_matEnv = matEnv;
        if (m_envNode)
            delete m_envNode;
        m_envNode = new InfiniteSphereSurfaceNode(m_context, "Environment", matEnv);
        m_envIsDirty = true;
    }

    void Scene::setEnvironmentRotation(float rotationPhi) {
        m_envInst.data.rotationPhi = rotationPhi;
        m_envIsDirty = true;
    }



    // static
    void Camera::commonInitializeProcedure(
        Context& context, const char* identifiers[NumCameraCallableNames], OptiXProgramSet* programSet) {
        programSet->dcSampleLensPosition = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_sample]);
        programSet->dcTestLensIntersection = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_testIntersection]);

        programSet->dcSetupIDF = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_setupIDF]);

        programSet->dcIDFSampleInternal = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_IDF_sampleInternal]);
        programSet->dcIDFEvaluateSpatialImportanceInternal = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_IDF_evaluateSpatialImportanceInternal]);
        programSet->dcIDFEvaluateDirectionalImportanceInternal = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_IDF_evaluateDirectionalImportanceInternal]);
        programSet->dcIDFEvaluatePDFInternal = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_IDF_evaluatePDFInternal]);
        programSet->dcIDFBackProjectDirectionInternal = context.createDirectCallableProgram(
            OptiXModule_Camera, identifiers[CameraCallableName_IDF_backProjectDirection]);

        shared::IDFProcedureSet idfProcSet;
        {
            idfProcSet.progSampleInternal = programSet->dcIDFSampleInternal;
            idfProcSet.progEvaluateSpatialImportanceInternal = programSet->dcIDFEvaluateSpatialImportanceInternal;
            idfProcSet.progEvaluateDirectionalImportanceInternal = programSet->dcIDFEvaluateDirectionalImportanceInternal;
            idfProcSet.progEvaluatePDFInternal = programSet->dcIDFEvaluatePDFInternal;
            idfProcSet.progBackProjectDirectionInternal = programSet->dcIDFBackProjectDirectionInternal;
        }
        programSet->idfProcedureSetIndex = context.allocateIDFProcedureSet();
        context.updateIDFProcedureSet(programSet->idfProcedureSetIndex, idfProcSet, 0);
    }

    // static
    void Camera::commonFinalizeProcedure(Context& context, OptiXProgramSet& programSet) {
        context.releaseIDFProcedureSet(programSet.idfProcedureSetIndex);

        context.destroyDirectCallableProgram(programSet.dcIDFBackProjectDirectionInternal);
        context.destroyDirectCallableProgram(programSet.dcIDFEvaluatePDFInternal);
        context.destroyDirectCallableProgram(programSet.dcIDFEvaluateDirectionalImportanceInternal);
        context.destroyDirectCallableProgram(programSet.dcIDFEvaluateSpatialImportanceInternal);
        context.destroyDirectCallableProgram(programSet.dcIDFSampleInternal);

        context.destroyDirectCallableProgram(programSet.dcSetupIDF);

        context.destroyDirectCallableProgram(programSet.dcTestLensIntersection);
        context.destroyDirectCallableProgram(programSet.dcSampleLensPosition);
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
            RT_DC_NAME_STR("PerspectiveCamera_sample"),
            RT_DC_NAME_STR("PerspectiveCamera_testIntersection"),
            RT_DC_NAME_STR("PerspectiveCamera_setupIDF"),
            RT_DC_NAME_STR("PerspectiveCameraIDF_sampleInternal"),
            RT_DC_NAME_STR("PerspectiveCameraIDF_evaluateSpatialImportanceInternal"),
            RT_DC_NAME_STR("PerspectiveCameraIDF_evaluateDirectionalImportanceInternal"),
            RT_DC_NAME_STR("PerspectiveCameraIDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("PerspectiveCameraIDF_backProjectDirection"),
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
        launchParams->progSampleLensPosition = progSet.dcSampleLensPosition;
        launchParams->progTestLensIntersection = progSet.dcTestLensIntersection;
        launchParams->cameraDescriptor.idfProcedureSetIndex = progSet.idfProcedureSetIndex;
        launchParams->cameraDescriptor.progSetupIDF = progSet.dcSetupIDF;
        std::memcpy(launchParams->cameraDescriptor.data, &m_data, sizeof(m_data));
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
            RT_DC_NAME_STR("EquirectangularCamera_sample"),
            RT_DC_NAME_STR("EquirectangularCamera_testIntersection"),
            RT_DC_NAME_STR("EquirectangularCamera_setupIDF"),
            RT_DC_NAME_STR("EquirectangularCameraIDF_sampleInternal"),
            RT_DC_NAME_STR("EquirectangularCameraIDF_evaluateSpatialImportanceInternal"),
            RT_DC_NAME_STR("EquirectangularCameraIDF_evaluateDirectionalImportanceInternal"),
            RT_DC_NAME_STR("EquirectangularCameraIDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("EquirectangularCameraIDF_backProjectDirection"),
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
        launchParams->progSampleLensPosition = progSet.dcSampleLensPosition;
        launchParams->progTestLensIntersection = progSet.dcTestLensIntersection;
        launchParams->cameraDescriptor.idfProcedureSetIndex = progSet.idfProcedureSetIndex;
        launchParams->cameraDescriptor.progSetupIDF = progSet.dcSetupIDF;
        std::memcpy(launchParams->cameraDescriptor.data, &m_data, sizeof(m_data));
    }
}
