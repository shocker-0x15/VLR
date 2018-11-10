#pragma once

#include "materials.h"

namespace VLR {
    class Transform : public TypeAwareClass {
    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        virtual ~Transform() {}

        virtual bool isStatic() const = 0;
    };



    class StaticTransform : public Transform {
        Matrix4x4 m_matrix;
        Matrix4x4 m_invMatrix;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

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
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

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
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SurfaceNode(Context &context, const std::string &name) : Node(context, name) {}
        virtual ~SurfaceNode() {}

        virtual void addParent(ParentNode* parent);
        virtual void removeParent(ParentNode* parent);
    };



    class TriangleMeshSurfaceNode : public SurfaceNode {
        struct OptiXProgramSet {
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
            optix::Geometry optixGeometry;
            DiscreteDistribution1D primDist;
        };

        std::vector<Vertex> m_vertices;
        optix::Buffer m_optixVertexBuffer;
        std::vector<OptiXGeometry> m_optixGeometries;
        std::vector<const SurfaceMaterial*> m_materials;
        std::vector<ShaderNodeSocketIdentifier> m_nodeNormals;
        std::vector<ShaderNodeSocketIdentifier> m_nodeAlphas;
        std::vector<SHGeometryInstance*> m_shGeometryInstances;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        TriangleMeshSurfaceNode(Context &context, const std::string &name);
        ~TriangleMeshSurfaceNode();

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;

        void setVertices(std::vector<Vertex> &&vertices);
        void addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterial* material, 
                              const ShaderNodeSocketIdentifier &nodeNormal, const ShaderNodeSocketIdentifier &alpha, VLRTangentType tangentType);
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
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

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
    };



    class ParentNode : public Node {
    protected:
        std::set<Node*> m_children;
        const Transform* m_localToWorld;

        // key: child SHTransform
        // SHTransform containing only the self transform uses nullptr as the key.
        std::map<const SHTransform*, SHTransform*> m_shTransforms;

        SHGeometryGroup m_shGeomGroup;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ParentNode(Context &context, const std::string &name, const Transform* localToWorld);
        virtual ~ParentNode();

        void setName(const std::string &name) override;

        enum class UpdateEvent {
            TransformAdded = 0,
            TransformRemoved,
            TransformUpdated,
            GeometryAdded,
            GeometryRemoved,
        };

        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*> &childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) = 0;
        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) = 0;
        virtual void setTransform(const Transform* localToWorld);
        const Transform* getTransform() const {
            return m_localToWorld;
        }

        void addChild(InternalNode* child);
        void addChild(SurfaceNode* child);
        void removeChild(InternalNode* child);
        void removeChild(SurfaceNode* child);
    };



    class InternalNode : public ParentNode {
        std::set<ParentNode*> m_parents;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        InternalNode(Context &context, const std::string &name, const Transform* localToWorld);

        void setTransform(const Transform* localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;
        std::map<const SHGeometryInstance*, Shared::SurfaceLightDescriptor> m_surfaceLights;
        optix::Buffer m_optixSurfaceLightDescriptorBuffer;
        DiscreteDistribution1D m_surfaceLightImpDist;
        bool m_surfaceLightsAreSetup;

        // DELETE ME
        //Float4Texture* m_envTex;
        //SurfaceMaterial* m_envMat;
        //InfiniteSphereSurfaceNode* m_envSphere;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        RootNode(Context &context, const Transform* localToWorld);
        ~RootNode();

        void set();
    };



    class Scene : public Object {
        RootNode m_rootNode;
        optix::Program m_callableProgramSampleInfiniteSphere;
        EnvironmentEmitterSurfaceMaterial* m_matEnv;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

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

        // TODO: 内部実装をInfiniteSphereSurfaceNode + EnvironmentEmitterMaterialを使ったものに変えられないかを考える。
        void setEnvironment(EnvironmentEmitterSurfaceMaterial* matEnv);

        void set();
    };



    class Camera : public Object {
    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Camera(Context &context) : 
            Object(context) {}
        virtual ~Camera() {}

        virtual void set() const = 0;
    };



    class PerspectiveCamera : public Camera {
        struct OptiXProgramSet {
            optix::Program callableProgramSampleLensPosition;
            optix::Program callableProgramSampleIDF;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Shared::PerspectiveCamera m_data;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        PerspectiveCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                          float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist);

        void set() const override;

        void setPosition(const Point3D &position) {
            m_data.position = position;
        }
        void setOrientation(const Quaternion &orientation) {
            m_data.orientation = orientation;
        }
        void setSensitivity(float sensitivity) {
            m_data.sensitivity = sensitivity;
        }
        void setFovY(float fovY) {
            m_data.fovY = fovY;
            m_data.setImagePlaneArea();
        }
        void setLensRadius(float lensRadius) {
            m_data.lensRadius = lensRadius;
        }
        void setObjectPlaneDistance(float distance) {
            m_data.objPlaneDistance = distance;
            m_data.setImagePlaneArea();
        }
    };



    class EquirectangularCamera : public Camera {
        struct OptiXProgramSet {
            optix::Program callableProgramSampleLensPosition;
            optix::Program callableProgramSampleIDF;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Shared::EquirectangularCamera m_data;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EquirectangularCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                              float sensitivity, float phiAngle, float thetaAngle);

        void set() const override;

        void setPosition(const Point3D &position) {
            m_data.position = position;
        }
        void setOrientation(const Quaternion &orientation) {
            m_data.orientation = orientation;
        }
        void setSensitivity(float sensitivity) {
            m_data.sensitivity = sensitivity;
        }
        void setAngles(float phiAngle, float thetaAngle) {
            m_data.phiAngle = phiAngle;
            m_data.thetaAngle = thetaAngle;
        }
    };
}
