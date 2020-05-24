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
        Context &m_context;
        //optix::Group m_optixGroup;
        //optix::Acceleration m_optixAcceleration;
        struct TransformStatus {
            bool hasGeometryDescendant;
            //optix::Transform transform;
            //optix::GeometryGroup geomGroup;
            //std::map<const SHGeometryInstance*, optix::GeometryInstance> geomInstances;

            TransformStatus() : hasGeometryDescendant(false) {}
            TransformStatus(TransformStatus &&v) {
                hasGeometryDescendant = v.hasGeometryDescendant;
                //transform = v.transform;
                //geomGroup = v.geomGroup;
                //geomInstances = std::move(v.geomInstances);
            }
            TransformStatus &operator=(TransformStatus &&v) {
                hasGeometryDescendant = v.hasGeometryDescendant;
                //transform = v.transform;
                //geomGroup = v.geomGroup;
                //geomInstances = std::move(v.geomInstances);
                return *this;
            }
        };
        std::map<const SHTransform*, TransformStatus> m_transforms;
        uint32_t m_numValidTransforms;

        SlotBuffer<Shared::GeometryInstanceDescriptor> m_geometryInstanceDescriptorBuffer;
        DiscreteDistribution1D m_surfaceLightImpDist;
        bool m_surfaceLightsAreSetup;

        void createOptiXDescendants(SHTransform* transform);
        void destroyOptiXDescendants(SHTransform* transform);

    public:
        SHGroup(Context &context) : m_context(context), m_numValidTransforms(0), m_surfaceLightsAreSetup(false) {
            //optix::Context optixContext = m_context.getOptiXContext();
            //m_optixGroup = optixContext->createGroup();
            //m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            //m_optixGroup->setAcceleration(m_optixAcceleration);

            //m_geometryInstanceDescriptorBuffer.initialize(optixContext, 65536, nullptr);
        }
        ~SHGroup() {
            //if (m_surfaceLightsAreSetup)
            //    m_surfaceLightImpDist.finalize(m_context);

            //m_geometryInstanceDescriptorBuffer.finalize();

            //m_optixAcceleration->destroy();
            //m_optixGroup->destroy();
        }

        void addChild(SHTransform* transform);
        void removeChild(SHTransform* transform);
        void updateChild(SHTransform* transform);

        void addGeometryInstances(SHTransform* transform, std::set<const SHGeometryInstance*> geomInsts);
        void removeGeometryInstances(SHTransform* transform, std::set<const SHGeometryInstance*> geomInsts);

        void setup();

        void printOptiXHierarchy();
    };

    class SHTransform {
        std::string m_name;

        StaticTransform m_transform;
        union {
            const SHTransform* m_childTransform;
            SHGeometryGroup* m_childGeometryGroup;
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

        void setChild(SHGeometryGroup* geomGroup);
        bool hasGeometryDescendant(SHGeometryGroup** descendant = nullptr) const;
    };

    class SHGeometryGroup {
        //optix::Acceleration m_optixAcceleration;
        std::set<const SHGeometryInstance*> m_instances;

    public:
        SHGeometryGroup(Context &context) {
            //optix::Context optixContext = context.getOptiXContext();
            //m_optixAcceleration = optixContext->createAcceleration("Trbvh");
        }
        ~SHGeometryGroup() {
            //m_optixAcceleration->destroy();
        }

        void addGeometryInstance(const SHGeometryInstance* instance);
        void removeGeometryInstance(const SHGeometryInstance* instance);
        void updateGeometryInstance(const SHGeometryInstance* instance);
        bool has(const SHGeometryInstance* instance) const {
            return m_instances.count(instance) > 0;
        }
        const SHGeometryInstance* getGeometryInstanceAt(uint32_t index) const {
            auto it = m_instances.cbegin();
            std::advance(it, index);
            return *it;
        }
        uint32_t getNumInstances() const {
            return (uint32_t)m_instances.size();
        }

        //optix::Acceleration getAcceleration() const {
        //    return m_optixAcceleration;
        //}
    };

    class SHGeometryInstance {
        struct TriangleMeshProperty {
            //optix::Buffer vertexBuffer;
            //optix::Buffer triangleBuffer;
            DiscreteDistribution1D primDist;
            float sumImportances;
        };
        struct InfiniteSphereProperty {
        };

        //optix::Geometry m_geometry;
        //optix::GeometryTriangles m_geometryTriangles;
        //optix::Program m_progDecodeHitPoint;
        int32_t m_progSample;
        //optix::Material m_material;
        uint32_t m_materialIndex;
        float m_importance;
        ShaderNodePlug m_nodeNormal;
        ShaderNodePlug m_nodeTangent;
        ShaderNodePlug m_nodeAlpha;
        TriangleMeshProperty m_triMeshProp;
        InfiniteSphereProperty m_infSphereProp;
        bool m_isTriMesh;

    public:
        //SHGeometryInstance(const optix::Geometry &geometry, const optix::Program &progDecodeHitPoint, int32_t progSample,
        //                   const optix::Material &material, uint32_t materialIndex, float importance,
        //                   const ShaderNodePlug &nodeNormal, const ShaderNodePlug &nodeTangent, const ShaderNodePlug &nodeAlpha,
        //                   const optix::Buffer &vertexBuffer, const optix::Buffer &triangleBuffer,
        //                   const DiscreteDistribution1D &primDist, float sumImportances) :
        //m_geometry(geometry), m_progDecodeHitPoint(progDecodeHitPoint), m_progSample(progSample),
        //m_material(material), m_materialIndex(materialIndex), m_importance(importance),
        //m_nodeNormal(nodeNormal), m_nodeTangent(nodeTangent), m_nodeAlpha(nodeAlpha) {
        //    m_triMeshProp.vertexBuffer = vertexBuffer;
        //    m_triMeshProp.triangleBuffer = triangleBuffer;
        //    m_triMeshProp.primDist = primDist;
        //    m_triMeshProp.sumImportances = sumImportances;
        //    m_isTriMesh = true;
        //}
        //SHGeometryInstance(const optix::GeometryTriangles &geometryTriangles, const optix::Program &progDecodeHitPoint, int32_t progSample,
        //                   const optix::Material &material, uint32_t materialIndex, float importance,
        //                   const ShaderNodePlug &nodeNormal, const ShaderNodePlug &nodeTangent, const ShaderNodePlug &nodeAlpha,
        //                   const optix::Buffer &vertexBuffer, const optix::Buffer &triangleBuffer,
        //                   const DiscreteDistribution1D &primDist, float sumImportances) :
        //    m_geometryTriangles(geometryTriangles), m_progDecodeHitPoint(progDecodeHitPoint), m_progSample(progSample),
        //    m_material(material), m_materialIndex(materialIndex), m_importance(importance),
        //    m_nodeNormal(nodeNormal), m_nodeTangent(nodeTangent), m_nodeAlpha(nodeAlpha) {
        //    m_triMeshProp.vertexBuffer = vertexBuffer;
        //    m_triMeshProp.triangleBuffer = triangleBuffer;
        //    m_triMeshProp.primDist = primDist;
        //    m_triMeshProp.sumImportances = sumImportances;
        //    m_isTriMesh = true;
        //}
        //SHGeometryInstance(const optix::Geometry &geometry, const optix::Program &progDecodeHitPoint, int32_t progSample,
        //                   const optix::Material &material, uint32_t materialIndex, float importance) :
        //    m_geometry(geometry), m_progDecodeHitPoint(progDecodeHitPoint), m_progSample(progSample),
        //    m_material(material), m_materialIndex(materialIndex), m_importance(importance) {
        //    m_isTriMesh = false;
        //}
        ~SHGeometryInstance() {}

        //optix::GeometryInstance createGeometryInstance(Context &context) const;
        void createGeometryInstanceDescriptor(Shared::GeometryInstanceDescriptor* desc) const;
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
            //optixu::Program programCalcAttributeForTriangle; // Attribute Program
            //optixu::Program programIntersectTriangle; // Intersection Program
            //optixu::Program programCalcBBoxForTriangle; // Bounding Box Program
            optixu::ProgramGroup callableProgramDecodeHitPointForTriangle;
            optixu::ProgramGroup callableProgramSampleTriangleMesh;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        struct OptiXGeometry {
            std::vector<uint32_t> indices;
            cudau::Buffer optixIndexBuffer;
            //optix::GeometryTriangles optixGeometryTriangles;
            //optix::Geometry optixGeometry;
            DiscreteDistribution1D primDist;
        };

        std::vector<Vertex> m_vertices;
        cudau::Buffer m_optixVertexBuffer;
        std::vector<OptiXGeometry> m_optixGeometries;
        std::vector<const SurfaceMaterial*> m_materials;
        std::vector<ShaderNodePlug> m_nodeNormals;
        std::vector<ShaderNodePlug> m_nodeTangents;
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
                              const ShaderNodePlug &nodeNormal, const ShaderNodePlug& nodeTangent, const ShaderNodePlug &nodeAlpha);
    };



    class InfiniteSphereSurfaceNode : public SurfaceNode {
        struct OptiXProgramSet {
            //optixu::Program programIntersectInfiniteSphere; // Intersection Program
            //optixu::Program programCalcBBoxForInfiniteSphere; // Bounding Box Program
            optixu::ProgramGroup callableProgramDecodeHitPointForInfiniteSphere;
            optixu::ProgramGroup callableProgramSampleInfiniteSphere;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        //optix::Geometry m_optixGeometry;
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

        void geometryAddEvent(const std::set<const SHGeometryInstance*> &childDelta);
        virtual void geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) = 0;
        void geometryRemoveEvent(const std::set<const SHGeometryInstance*> &childDelta);
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

        void geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;
        void geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;

        void setTransform(const Transform* localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        RootNode(Context &context, const Transform* localToWorld);
        ~RootNode();

        void transformAddEvent(const std::set<SHTransform*>& childDelta) override;
        void transformRemoveEvent(const std::set<SHTransform*>& childDelta) override;
        void transformUpdateEvent(const std::set<SHTransform*>& childDelta) override;

        void geometryAddEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;
        void geometryRemoveEvent(const SHTransform* childTransform, const std::set<const SHGeometryInstance*>& geomInstDelta) override;

        void setup();
    };



    class Scene : public Object {
        RootNode m_rootNode;
        optixu::ProgramGroup m_callableProgramSampleInfiniteSphere;
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
            optixu::ProgramGroup callableProgramSampleLensPosition;
            optixu::ProgramGroup callableProgramSampleIDF;
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
