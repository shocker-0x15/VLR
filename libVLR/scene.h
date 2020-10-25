#pragma once

#include "materials.h"

namespace VLR {
    class Transform : public TypeAwareClass {
    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        virtual ~Transform() {}

        virtual bool isStatic() const = 0;
    };



    class StaticTransform : public Transform {
        Matrix4x4 m_matrix;
        Matrix4x4 m_invMatrix;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        StaticTransform(const Matrix4x4 &m = Matrix4x4::Identity()) : m_matrix(m), m_invMatrix(invert(m)) {}

        bool isStatic() const override { return true; }

        StaticTransform operator*(const Matrix4x4 &m) const { return StaticTransform(m_matrix * m); }
        StaticTransform operator*(const StaticTransform &t) const { return StaticTransform(m_matrix * t.m_matrix); }
        bool operator==(const StaticTransform &t) const { return m_matrix == t.m_matrix; }
        bool operator!=(const StaticTransform &t) const { return m_matrix != t.m_matrix; }

        void getArrays(float mat[16], float invMat[16]) const {
            m_matrix.getArray(mat);
            m_invMatrix.getArray(invMat);
        }
    };



    // ----------------------------------------------------------------
    // Shallow Hierarchy
    // JP: レンダリング時のパフォーマンスを考えるとシーン階層は浅いほうが良い。
    //     ユーザーには任意の深いシーングラフを許可する裏で同時に浅いシーングラフを生成する。
    // EN: Shallower scene hierarchy is preferrable for rendering performance.
    //     Generate shallow scene graph while allowing the user build arbitrarily deep scene graph.

    class SHGeometryGroup;
    struct SHGeometryInstance;

    class SHGeometryGroup {
        optixu::GeometryAccelerationStructure m_optixGas;
        std::vector<const SHGeometryInstance*> m_shGeomInsts;

    public:
        SHGeometryGroup(const optixu::GeometryAccelerationStructure &optixGas) :
            m_optixGas(optixGas) {}
        ~SHGeometryGroup() {
            m_optixGas.destroy();
        }

        void addChild(const SHGeometryInstance* geomInst);
        void removeChild(const SHGeometryInstance* geomInst);
        void updateChild(const SHGeometryInstance* geomInst);

        uint32_t getNumChildren() const { return m_shGeomInsts.size(); }
        void getGeometryInstanceIndices(uint32_t* indices) const;
        void getGeometryInstanceImportanceValues(float* values) const;
    };

    struct SHGeometryInstance {
        uint32_t geomInstIndex;
        optixu::GeometryInstance optixGeomInst;
        Shared::GeometryInstance data;
    };

    class SHTransform {
        std::string m_name;

        StaticTransform m_transform;
        union {
            const SHTransform* m_childTransform;
            const SHGeometryGroup* m_childGeomGroup;
        };
        bool m_childIsTransform;

        StaticTransform resolveTransform() const;

    public:
        SHTransform(const std::string &name, Context &context, const StaticTransform &transform, const SHTransform* childTransform) :
            m_name(name), m_transform(transform), m_childTransform(childTransform), m_childIsTransform(childTransform != nullptr) {}
        ~SHTransform() {}

        const std::string &getName() const { return m_name; }
        void setName(const std::string &name) {
            m_name = name;
        }

        void setTransform(const StaticTransform &transform);
        void update();
        bool isStatic() const;
        StaticTransform getStaticTransform() const;

        void setChild(const SHGeometryGroup* childGeomGroup);
        bool hasGeometryDescendant(const SHGeometryGroup** descendant = nullptr) const;
    };

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    class Node;
    class ParentNode;
    class RootNode;
    class InternalNode;



    class Node : public Object {
    protected:
        std::string m_name;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        Node(Context &context, const std::string &name) :
            Object(context), m_name(name) {}
        virtual ~Node() {}

        virtual void setName(const std::string &name) {
            m_name = name;
        }

        const std::string &getName() const {
            return m_name;
        }
    };



    class SurfaceNode : public Node {
    protected:
        std::set<ParentNode*> m_parents;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        SurfaceNode(Context &context, const std::string &name) : Node(context, name) {}
        virtual ~SurfaceNode() {}

        virtual void addParent(ParentNode* parent);
        virtual void removeParent(ParentNode* parent);
    };



    class TriangleMeshSurfaceNode : public SurfaceNode {
        struct OptiXProgramSet {
            optixu::Module optixModule;
            CallableProgram dcDecodeHitPointForTriangle;
            CallableProgram dcSampleTriangleMesh;
        };

        static std::map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        struct MaterialGroup {
            std::vector<uint32_t> indices;
            cudau::TypedBuffer<Shared::Triangle> optixIndexBuffer;
            DiscreteDistribution1D primDist;
            const SurfaceMaterial* material;
            ShaderNodePlug nodeNormal;
            ShaderNodePlug nodeTangent;
            ShaderNodePlug nodeAlpha;
            SHGeometryInstance* shGeomInst;

            MaterialGroup() {}
            MaterialGroup(MaterialGroup &&v) {
                indices = std::move(v.indices);
                optixIndexBuffer = std::move(v.optixIndexBuffer);
                primDist = std::move(v.primDist);
                material = v.material;
                nodeNormal = v.nodeNormal;
                nodeTangent = v.nodeTangent;
                nodeAlpha = v.nodeAlpha;
                shGeomInst = v.shGeomInst;
            }
            MaterialGroup &operator=(MaterialGroup &&v) {
                indices = std::move(v.indices);
                optixIndexBuffer = std::move(v.optixIndexBuffer);
                primDist = std::move(v.primDist);
                material = v.material;
                nodeNormal = v.nodeNormal;
                nodeTangent = v.nodeTangent;
                nodeAlpha = v.nodeAlpha;
                shGeomInst = v.shGeomInst;
                return *this;
            }
        };

        std::vector<Vertex> m_vertices;
        cudau::TypedBuffer<Vertex> m_optixVertexBuffer;
        std::vector<MaterialGroup> m_materialGroups;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        TriangleMeshSurfaceNode(Context &context, const std::string &name);
        ~TriangleMeshSurfaceNode();

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;

        void setVertices(std::vector<Vertex> &&vertices);
        void addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterial* material, 
                              const ShaderNodePlug &nodeNormal, const ShaderNodePlug& nodeTangent, const ShaderNodePlug &nodeAlpha);
    };



    class InfiniteSphereSurfaceNode : public SurfaceNode {
        struct OptiXProgramSet {
            optixu::Module optixModule;
            CallableProgram dcDecodeHitPointForInfiniteSphere;
            CallableProgram dcSampleInfiniteSphere;
        };

        static std::map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        SurfaceMaterial* m_material;
        SHGeometryInstance m_shGeomInst;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        InfiniteSphereSurfaceNode(Context &context, const std::string &name, SurfaceMaterial* material);
        ~InfiniteSphereSurfaceNode();

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;
    };



    // JP: ParentNodeは対応するひとつのSHTransformとSHGeometryGroupを持つ。
    //     また自分の子孫が持つ連なったSHTransformも管理する。
    //     SHGeometryGroupにはParentNodeに「直接」所属するSurfaceNodeのジオメトリが登録される。
    // EN: ParentNode has a single SHTransform corresponding the node itself and a single SHGeometryGroup.
    //     And it manages chaining SHTransforms that children have.
    //     SHGeometryGroup holds geometries of SurfaceNode's which *directly* belong to the ParentNode.
    class ParentNode : public Node {
        void addToChildMap(Node* child);
        void removeFromChildMap(Node* child);

    protected:
        uint32_t m_serialChildID;
        std::map<Node*, uint32_t> m_childToSerialIDMap;
        std::map<uint32_t, Node*> m_serialIDToChlidMap;
        const Transform* m_localToWorld;

        // key: child SHTransform
        // JP: 自分自身のトランスフォームのみを含むSHTransformはnullptrをキーとする。
        // EN: SHTransform containing only the self transform uses nullptr as the key.
        std::map<const SHTransform*, SHTransform*> m_shTransforms;

        SHGeometryGroup* m_shGeomGroup;

        void createConcatanatedTransforms(const std::set<SHTransform*>& childDelta, std::set<SHTransform*>* delta);
        void removeConcatanatedTransforms(const std::set<SHTransform*>& childDelta, std::set<SHTransform*>* delta);
        void updateConcatanatedTransforms(const std::set<SHTransform*>& childDelta, std::set<SHTransform*>* delta);

        void addToGeometryGroup(const std::set<const SHGeometryInstance*> &childDelta);
        void removeFromGeometryGroup(const std::set<const SHGeometryInstance*> &childDelta);
        void updateGeometryGroup(const std::set<const SHGeometryInstance*> &childDelta);

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        ParentNode(Context &context, const std::string &name, const Transform* localToWorld);
        virtual ~ParentNode();

        void setName(const std::string &name) override;

        virtual void transformAddEvent(const std::set<SHTransform*> &childDelta) = 0;
        virtual void transformRemoveEvent(const std::set<SHTransform*> &childDelta) = 0;
        virtual void transformUpdateEvent(const std::set<SHTransform*> &childDelta) = 0;

        void geometryAddEvent(const std::set<const SHGeometryInstance*> &childDelta);
        virtual void geometryAddEvent(const SHTransform* childTransform) = 0;
        void geometryRemoveEvent(const std::set<const SHGeometryInstance*> &childDelta);
        virtual void geometryRemoveEvent(const SHTransform* childTransform) = 0;
        void geometryUpdateEvent(const std::set<const SHGeometryInstance*> &childDelta);
        virtual void geometryUpdateEvent(const SHTransform* childTransform) = 0;

        virtual void setTransform(const Transform* localToWorld);
        const Transform* getTransform() const {
            return m_localToWorld;
        }

        void addChild(InternalNode* child);
        void removeChild(InternalNode* child);
        void addChild(SurfaceNode* child);
        void removeChild(SurfaceNode* child);
        uint32_t getNumChildren() const;
        void getChildren(Node** children) const;
        Node* getChildAt(uint32_t index) const;
    };



    class InternalNode : public ParentNode {
        std::set<ParentNode*> m_parents;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        InternalNode(Context &context, const std::string &name, const Transform* localToWorld);

        void transformAddEvent(const std::set<SHTransform*> &childDelta) override;
        void transformRemoveEvent(const std::set<SHTransform*> &childDelta) override;
        void transformUpdateEvent(const std::set<SHTransform*> &childDelta) override;

        void geometryAddEvent(const SHTransform* childTransform) override;
        void geometryRemoveEvent(const SHTransform* childTransform) override;
        void geometryUpdateEvent(const SHTransform* childTransform) override;

        void setTransform(const Transform* localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        struct Instance {
            optixu::Instance optixInst;
            uint32_t instIndex;
            cudau::TypedBuffer<uint32_t> geomInstIndices;
            DiscreteDistribution1D lightGeomInstDistribution;
            Shared::Instance data;
        };
        optixu::InstanceAccelerationStructure m_optixIas;
        std::map<const SHTransform*, Instance> m_instances;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        RootNode(Context &context, const Transform* localToWorld);
        ~RootNode();

        void transformAddEvent(const std::set<SHTransform*> &childDelta) override;
        void transformRemoveEvent(const std::set<SHTransform*> &childDelta) override;
        void transformUpdateEvent(const std::set<SHTransform*> &childDelta) override;

        void geometryAddEvent(const SHTransform* childTransform) override;
        void geometryRemoveEvent(const SHTransform* childTransform) override;
        void geometryUpdateEvent(const SHTransform* childTransform) override;

        void setup(Shared::PipelineLaunchParameters* launchParams);
    };



    class Scene : public Object {
        RootNode m_rootNode;
        optixu::Module m_optixModule;
        CallableProgram m_dcSampleInfiniteSphere;
        EnvironmentEmitterSurfaceMaterial* m_matEnv;
        float m_envRotationPhi;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        Scene(Context &context, const Transform* localToWorld);
        ~Scene();

        void setTransform(const Transform* localToWorld) {
            m_rootNode.setTransform(localToWorld);
        }

        void addChild(InternalNode* child) {
            m_rootNode.addChild(child);
        }
        void addChild(SurfaceNode* child) {
            m_rootNode.addChild(child);
        }
        void removeChild(InternalNode* child) {
            m_rootNode.removeChild(child);
        }
        void removeChild(SurfaceNode* child) {
            m_rootNode.removeChild(child);
        }
        uint32_t getNumChildren() const {
            return m_rootNode.getNumChildren();
        }
        void getChildren(Node** children) const {
            m_rootNode.getChildren(children);
        }
        Node* getChildAt(uint32_t index) const {
            return m_rootNode.getChildAt(index);
        }

        // TODO: 内部実装をInfiniteSphereSurfaceNode + EnvironmentEmitterMaterialを使ったものに変えられないかを考える。
        void setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv);
        void setEnvironmentRotation(float rotationPhi);

        void setup(Shared::PipelineLaunchParameters* launchParams);
    };



    class Camera : public Queryable {
    protected:
        struct OptiXProgramSet {
            CallableProgram dcSampleLensPosition;
            CallableProgram dcSampleIDF;
        };

        static std::map<uint32_t, optixu::Module> s_optixModules;
        static void commonInitializeProcedure(Context& context, const char* identifiers[2], OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context& context, OptiXProgramSet& programSet);

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Camera(Context &context) : 
            Queryable(context) {}
        virtual ~Camera() {}

        virtual void setup(Shared::PipelineLaunchParameters* launchParams) const = 0;
    };



    class PerspectiveCamera : public Camera {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        Shared::PerspectiveCamera m_data;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        PerspectiveCamera(Context &context);

        bool get(const char* paramName, Point3D* value) const override;
        bool get(const char* paramName, Quaternion* value) const override;
        bool get(const char* paramName, float* values, uint32_t length) const override;

        bool set(const char* paramName, const Point3D &value) override;
        bool set(const char* paramName, const Quaternion &value) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;

        void setup(Shared::PipelineLaunchParameters* launchParams) const override;
    };



    class EquirectangularCamera : public Camera {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        Shared::EquirectangularCamera m_data;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        EquirectangularCamera(Context &context);

        bool get(const char* paramName, Point3D* value) const override;
        bool get(const char* paramName, Quaternion* value) const override;
        bool get(const char* paramName, float* values, uint32_t length) const override;

        bool set(const char* paramName, const Point3D& value) override;
        bool set(const char* paramName, const Quaternion& value) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;

        void setup(Shared::PipelineLaunchParameters* launchParams) const override;
    };
}
