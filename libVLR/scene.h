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

    class SHGroup;
    class SHTransform;
    class SHGeometryGroup;
    class SHGeometryInstance;

    class SHGroup {
        optix::Group m_optixGroup;
        optix::Acceleration m_optixAcceleration;
        struct TransformStatus {
            bool hasGeometryDescendant;
        };
        std::map<const SHTransform*, TransformStatus> m_transforms;
        uint32_t m_numValidTransforms;
        std::set<const SHGeometryGroup*> m_geomGroups;

    public:
        SHGroup(Context &context) : m_numValidTransforms(0) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixGroup = optixContext->createGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGroup->setAcceleration(m_optixAcceleration);
        }
        ~SHGroup() {
            m_optixAcceleration->destroy();
            m_optixGroup->destroy();
        }

        void addChild(SHTransform* transform);
        void removeChild(SHTransform* transform);
        void updateChild(SHTransform* transform);

        void addChild(SHGeometryGroup* geomGroup);
        void removeChild(SHGeometryGroup* geomGroup);

        const optix::Group &getOptiXObject() const {
            return m_optixGroup;
        }

        void printOptiXHierarchy();
    };

    class SHTransform {
        std::string m_name;
        optix::Transform m_optixTransform;

        StaticTransform m_transform;
        union {
            const SHTransform* m_childTransform;
            SHGeometryGroup* m_childGeometryGroup;
        };
        bool m_childIsTransform;

        void resolveTransform();

    public:
        SHTransform(const std::string &name, Context &context, const StaticTransform &transform, const SHTransform* childTransform) :
            m_name(name), m_transform(transform), m_childTransform(childTransform), m_childIsTransform(childTransform != nullptr) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixTransform = optixContext->createTransform();

            resolveTransform();
        }
        ~SHTransform() {
            m_optixTransform->destroy();
        }

        const std::string &getName() const { return m_name; }
        void setName(const std::string &name) {
            m_name = name;
        }

        void setTransform(const StaticTransform &transform);
        void update();
        bool isStatic() const;
        StaticTransform getStaticTransform() const;

        void setChild(SHGeometryGroup* geomGroup);
        bool hasGeometryDescendant(SHGeometryGroup** descendant = nullptr) const;

        const optix::Transform &getOptiXObject() const {
            return m_optixTransform;
        }
    };

    class SHGeometryGroup {
        optix::GeometryGroup m_optixGeometryGroup;
        optix::Acceleration m_optixAcceleration;
        std::set<const SHGeometryInstance*> m_instances;

    public:
        SHGeometryGroup(Context &context) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixGeometryGroup = optixContext->createGeometryGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGeometryGroup->setAcceleration(m_optixAcceleration);
        }
        ~SHGeometryGroup() {
            m_optixAcceleration->destroy();
            m_optixGeometryGroup->destroy();
        }

        void addGeometryInstance(const SHGeometryInstance* instance);
        void removeGeometryInstance(const SHGeometryInstance* instance);
        void updateGeometryInstance(const SHGeometryInstance* instance);
        const SHGeometryInstance* getGeometryInstanceAt(uint32_t index) const {
            auto it = m_instances.cbegin();
            std::advance(it, index);
            return *it;
        }
        uint32_t getNumInstances() const {
            return (uint32_t)m_instances.size();
        }

        const optix::GeometryGroup &getOptiXObject() const {
            return m_optixGeometryGroup;
        }
    };

    class SHGeometryInstance {
        optix::GeometryInstance m_optixGeometryInstance;
        Shared::SurfaceLightDescriptor m_surfaceLightDescriptor;

    public:
        SHGeometryInstance(Context &context, const Shared::SurfaceLightDescriptor &lightDesc) : m_surfaceLightDescriptor(lightDesc) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixGeometryInstance = optixContext->createGeometryInstance();
        }
        ~SHGeometryInstance() {
            m_optixGeometryInstance->destroy();
        }

        void getSurfaceLightDescriptor(Shared::SurfaceLightDescriptor* lightDesc) const {
            *lightDesc = m_surfaceLightDescriptor;
        }

        const optix::GeometryInstance &getOptiXObject() const {
            return m_optixGeometryInstance;
        }
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
            optix::Program programCalcAttributeForTriangle; // Attribute Program
            optix::Program programIntersectTriangle; // Intersection Program
            optix::Program programCalcBBoxForTriangle; // Bounding Box Program
            optix::Program callableProgramDecodeHitPointForTriangle;
            optix::Program callableProgramDecodeTexCoordForTriangle;
            optix::Program callableProgramSampleTriangleMesh;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        struct OptiXGeometry {
            std::vector<uint32_t> indices;
            optix::Buffer optixIndexBuffer;
            optix::GeometryTriangles optixGeometryTriangles;
            optix::Geometry optixGeometry;
            DiscreteDistribution1D primDist;
        };

        std::vector<Vertex> m_vertices;
        optix::Buffer m_optixVertexBuffer;
        std::vector<OptiXGeometry> m_optixGeometries;
        std::vector<const SurfaceMaterial*> m_materials;
        std::vector<ShaderNodePlug> m_nodeNormals;
        std::vector<ShaderNodePlug> m_nodeAlphas;
        std::vector<SHGeometryInstance*> m_shGeometryInstances;

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
                              const ShaderNodePlug &nodeNormal, const ShaderNodePlug &alpha, TangentType tangentType);
    };



    class InfiniteSphereSurfaceNode : public SurfaceNode {
        struct OptiXProgramSet {
            optix::Program programIntersectInfiniteSphere; // Intersection Program
            optix::Program programCalcBBoxForInfiniteSphere; // Bounding Box Program
            optix::Program callableProgramDecodeHitPointForInfiniteSphere;
            optix::Program callableProgramDecodeTexCoordForInfiniteSphere;
            optix::Program callableProgramSampleInfiniteSphere;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        optix::Geometry m_optixGeometry;
        SurfaceMaterial* m_material;
        SHGeometryInstance* m_shGeometryInstance;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        InfiniteSphereSurfaceNode(Context &context, const std::string &name, SurfaceMaterial* material);
        ~InfiniteSphereSurfaceNode();

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;
    };



    struct TransformAndGeometryInstance {
        const SHTransform* transform;
        const SHGeometryInstance* geomInstance;

        bool operator<(const TransformAndGeometryInstance& v) const {
            if (transform < v.transform) {
                return true;
            }
            else if (transform == v.transform) {
                if (geomInstance < v.geomInstance)
                    return true;
            }
            return false;
        }
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

        SHGeometryGroup m_shGeomGroup;

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

        virtual void transformAddEvent(const std::set<SHTransform*>& childDelta) = 0;
        virtual void transformRemoveEvent(const std::set<SHTransform*>& childDelta) = 0;
        virtual void transformUpdateEvent(const std::set<SHTransform*>& childDelta) = 0;

        virtual void geometryAddEvent(const std::set<const SHGeometryInstance*> &childDelta) = 0;
        virtual void geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) = 0;
        virtual void geometryRemoveEvent(const std::set<const SHGeometryInstance*> &childDelta) = 0;
        virtual void geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) = 0;

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

        void transformAddEvent(const std::set<SHTransform*>& childDelta) override;
        void transformRemoveEvent(const std::set<SHTransform*>& childDelta) override;
        void transformUpdateEvent(const std::set<SHTransform*>& childDelta) override;

        void geometryAddEvent(const std::set<const SHGeometryInstance*>& childDelta) override;
        void geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;
        void geometryRemoveEvent(const std::set<const SHGeometryInstance*>& childDelta) override;
        void geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;

        void setTransform(const Transform* localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;
        std::map<TransformAndGeometryInstance, Shared::SurfaceLightDescriptor> m_surfaceLights;
        optix::Buffer m_optixSurfaceLightDescriptorBuffer;
        DiscreteDistribution1D m_surfaceLightImpDist;
        bool m_surfaceLightsAreSetup;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        RootNode(Context &context, const Transform* localToWorld);
        ~RootNode();

        void transformAddEvent(const std::set<SHTransform*>& childDelta) override;
        void transformRemoveEvent(const std::set<SHTransform*>& childDelta) override;
        void transformUpdateEvent(const std::set<SHTransform*>& childDelta) override;

        void geometryAddEvent(const std::set<const SHGeometryInstance*>& childDelta) override;
        void geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;
        void geometryRemoveEvent(const std::set<const SHGeometryInstance*>& childDelta) override;
        void geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;

        void set();
    };



    class Scene : public Object {
        RootNode m_rootNode;
        optix::Program m_callableProgramSampleInfiniteSphere;
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

        void setup();
    };



    class Camera : public Queryable {
    protected:
        struct OptiXProgramSet {
            optix::Program callableProgramSampleLensPosition;
            optix::Program callableProgramSampleIDF;
        };

        static std::string s_cameras_ptx;
        static void commonInitializeProcedure(Context& context, const char* identifiers[2], OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context& context, OptiXProgramSet& programSet);

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Camera(Context &context) : 
            Queryable(context) {}
        virtual ~Camera() {}

        virtual void setup() const = 0;
    };



    class PerspectiveCamera : public Camera {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        void setup() const override;
    };



    class EquirectangularCamera : public Camera {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        void setup() const override;
    };
}
