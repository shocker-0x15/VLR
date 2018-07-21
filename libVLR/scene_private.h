#pragma once

#include <scene.h>
#include "basic_types_internal.h"
#include "shared.h"

namespace VLR {
    class Context {
        static uint32_t NextID;
        static uint32_t getInstanceID() {
            return NextID++;
        }

        uint32_t m_ID;
        optix::Context m_optixContext;

        optix::Program m_optixCallableProgramNullFetchAlpha;
        optix::Program m_optixCallableProgramNullFetchNormal;
        optix::Program m_optixCallableProgramFetchAlpha;
        optix::Program m_optixCallableProgramFetchNormal;

        optix::Program m_optixProgramStochasticAlphaAnyHit; // -- Any Hit Program
        optix::Program m_optixProgramAlphaAnyHit; // ------------ Any Hit Program
        optix::Program m_optixProgramPathTracingIteration; // --- Closest Hit Program

        optix::Program m_optixProgramPathTracing; // ------------ Ray Generation Program
        optix::Program m_optixProgramPathTracingMiss; // -------- Miss Program
        optix::Program m_optixProgramException; // -------------- Exception Program

        optix::Buffer m_outputBuffer;
        optix::Buffer m_rngBuffer;
        uint32_t m_width;
        uint32_t m_height;

    public:
        Context();
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        void bindOpenGLBuffer(uint32_t bufferID, uint32_t width, uint32_t height);

        void render(Scene &scene, Camera* camera);

        optix::Context &getOptiXContext() {
            return m_optixContext;
        }

        optix::Program &getOptiXCallableProgramNullFetchAlpha() {
            return m_optixCallableProgramNullFetchAlpha;
        }
        optix::Program &getOptiXCallableProgramNullFetchNormal() {
            return m_optixCallableProgramNullFetchNormal;
        }
        optix::Program &getOptiXCallableProgramFetchAlpha() {
            return m_optixCallableProgramFetchAlpha;
        }
        optix::Program &getOptiXCallableProgramFetchNormal() {
            return m_optixCallableProgramFetchNormal;
        }

        optix::Program &getOptiXProgramStochasticAlphaAnyHit() {
            return m_optixProgramStochasticAlphaAnyHit;
        }
        optix::Program &getOptiXProgramAlphaAnyHit() {
            return m_optixProgramAlphaAnyHit;
        }
        optix::Program &getOptiXProgramPathTracingIteration() {
            return m_optixProgramPathTracingIteration;
        }
        optix::Program &getOptiXProgramPathTracingMiss() {
            return m_optixProgramPathTracingMiss;
        }
        optix::Program &getOptiXProgramPathTracing() {
            return m_optixProgramPathTracing;
        }
        optix::Program &getOptiXProgramException() {
            return m_optixProgramException;
        }
    };



    struct ObjectType {
        struct Value {
            uint64_t field0;

            Value() {}
            constexpr Value(uint64_t v);

            bool operator==(const Value &v) const;
            Value operator&(const Value &v) const;
            Value operator|(const Value &v) const;
        } value;

        static const ObjectType E_Context;
        static const ObjectType E_Image2D;
        static const ObjectType E_LinearImage2D;
        static const ObjectType E_FloatTexture;
        static const ObjectType E_Float2Texture;
        static const ObjectType E_Float3Texture;
        static const ObjectType E_Float4Texture;
        static const ObjectType E_ImageFloat4Texture;
        static const ObjectType E_ConstantFloat4Texture;
        static const ObjectType E_SurfaceMaterial;
        static const ObjectType E_MatteSurfaceMaterial;
        static const ObjectType E_UE4SurfaceMaterial;
        static const ObjectType E_Node;
        static const ObjectType E_SurfaceNode;
        static const ObjectType E_TriangleMeshSurfaceNode;
        static const ObjectType E_ParentNode;
        static const ObjectType E_InternalNode;
        static const ObjectType E_RootNode;
        static const ObjectType E_Scene;
        static const ObjectType E_Camera;
        static const ObjectType E_PerspectiveCamera;

        constexpr ObjectType(Value v = (Value)0) : value(v) { }

        bool is(const ObjectType &v) const;
        bool isMemberOf(const ObjectType &v) const;
    };

    class Object {
    protected:
        Context &m_context;

    public:
        Object(Context &context);
        virtual ~Object() {}

        virtual ObjectType getType() const = 0;

        Context &getContext() {
            return m_context;
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
        std::set<const SHGeometryGroup*> m_geometryGroups;

    public:
        SHGroup(Context &context) : m_numValidTransforms(0) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixGroup = optixContext->createGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGroup->setAcceleration(m_optixAcceleration);
        }
        ~SHGroup() {
            m_optixAcceleration->destroy();
            m_optixGroup->destroy();
        }

        void addChild(SHTransform* transform);
        void addChild(SHGeometryGroup* geomGroup);
        void removeChild(SHTransform* transform);
        void removeChild(SHGeometryGroup* geomGroup);
        void updateChild(SHTransform* transform);
        uint32_t getNumValidChildren() {
            return (uint32_t)(m_geometryGroups.size() + m_numValidTransforms);
        }

        optix::Group &getOptiXObject() {
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
            optix::Context &optixContext = context.getOptiXContext();
            m_optixTransform = optixContext->createTransform();

            resolveTransform();
        }
        ~SHTransform() {
            m_optixTransform->destroy();
        }

        const std::string &getName() const { return m_name; }

        void setTransform(const StaticTransform &transform);
        void update();

        void setChild(SHGeometryGroup* geomGroup);
        bool hasGeometryDescendant(SHGeometryGroup** descendant = nullptr) const;

        optix::Transform &getOptiXObject() {
            return m_optixTransform;
        }
    };

    class SHGeometryGroup {
        optix::GeometryGroup m_optixGeometryGroup;
        optix::Acceleration m_optixAcceleration;
        std::set<const SHGeometryInstance*> m_instances;

    public:
        SHGeometryGroup(Context &context) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixGeometryGroup = optixContext->createGeometryGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGeometryGroup->setAcceleration(m_optixAcceleration);
        }
        ~SHGeometryGroup() {
            m_optixAcceleration->destroy();
            m_optixGeometryGroup->destroy();
        }

        void addGeometryInstance(SHGeometryInstance* instance);
        void removeGeometryInstance(SHGeometryInstance* instance);
        uint32_t getNumInstances() const {
            return (uint32_t)m_instances.size();
        }

        optix::GeometryGroup &getOptiXObject() {
            return m_optixGeometryGroup;
        }
    };

    class SHGeometryInstance {
        optix::GeometryInstance m_optixGeometryInstance;

    public:
        SHGeometryInstance(Context &context) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixGeometryInstance = optixContext->createGeometryInstance();
        }
        ~SHGeometryInstance() {
            m_optixGeometryInstance->destroy();
        }

        optix::GeometryInstance &getOptiXObject() {
            return m_optixGeometryInstance;
        }
    };

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    class Node;
    class ParentNode;
    class RootNode;



    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    struct RGBA16Fx4 { uint16_t/*half*/ r, g, b, a; };
    struct RGBA32Fx4 { float r, g, b, a; };
    struct Gray8 { uint8_t v; };

    extern const size_t sizesOfDataFormats[(uint32_t)DataFormat::Num];

    class Image2D : public Object {
        uint32_t m_width, m_height;
        DataFormat m_dataFormat;
        optix::Buffer m_optixDataBuffer;

    public:
        static DataFormat getInternalFormat(DataFormat inputFormat);

        Image2D(Context &context, uint32_t width, uint32_t height, DataFormat dataFormat);
        virtual ~Image2D();

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        uint32_t getStride() const {
            return (uint32_t)sizesOfDataFormats[(uint32_t)m_dataFormat];
        }

        optix::Buffer &getOptiXObject() {
            return m_optixDataBuffer;
        }
    };



    class LinearImage2D : public Image2D {
        std::vector<uint8_t> m_data;

    public:
        LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat);

        virtual ObjectType getType() const override { return ObjectType::E_LinearImage2D; }
    };



    class FloatTexture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        FloatTexture(Context &context);
        virtual ~FloatTexture();

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };



    class Float2Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        Float2Texture(Context &context);
        virtual ~Float2Texture();

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };



    class Float3Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        Float3Texture(Context &context);
        virtual ~Float3Texture();

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };



    class Float4Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        Float4Texture(Context &context);
        virtual ~Float4Texture();

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };



    class ConstantFloat4Texture : public Float4Texture {
        Image2D* m_image;

    public:
        ConstantFloat4Texture(Context &context, const float value[4]);
        ~ConstantFloat4Texture();

        ObjectType getType() const override { return ObjectType::E_ConstantFloat4Texture; }
    };
    
    
    
    class ImageFloat4Texture : public Float4Texture {
        Image2D* m_image;

    public:
        ImageFloat4Texture(Context &context, Image2D* image);

        ObjectType getType() const override { return ObjectType::E_ImageFloat4Texture; }
    };



    class SurfaceMaterial : public Object {
    protected:
        optix::Material m_optixMaterial;

    public:
        static void initialize(Context &context);
        static void finalize(Context &context);

        SurfaceMaterial(Context &context);
        virtual ~SurfaceMaterial() {}

        optix::Material &getOptiXObject() {
            return m_optixMaterial;
        }
    };



    class MatteSurfaceMaterial : public SurfaceMaterial {
        struct OptiXProgramSet {
            optix::Program callableProgramSetup;
            optix::Program callableProgramGetBaseColor;
            optix::Program callableProgramBSDFmatches;
            optix::Program callableProgramSampleBSDFInternal;
            optix::Program callableProgramEvaluateBSDFInternal;
            optix::Program callableProgramEvaluateBSDF_PDFInternal;
            optix::Program callableProgramEvaluateEmittanceInternal;
            optix::Program callableProgramEvaluateEDFInternal;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Float4Texture* m_texAlbedoRoughness;

    public:
        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context, Float4Texture* texAlbedoRoughness);

        ObjectType getType() const override { return ObjectType::E_MatteSurfaceMaterial; }
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        struct OptiXProgramSet {
            optix::Program callableProgramSetup;
            optix::Program callableProgramGetBaseColor;
            optix::Program callableProgramBSDFmatches;
            optix::Program callableProgramSampleBSDFInternal;
            optix::Program callableProgramEvaluateBSDFInternal;
            optix::Program callableProgramEvaluateBSDF_PDFInternal;
            optix::Program callableProgramEvaluateEmittanceInternal;
            optix::Program callableProgramEvaluateEDFInternal;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Float3Texture* m_texBaseColor;
        Float2Texture* m_texRoughnessMetallic;

    public:
        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context, Float3Texture* texBaseColor, Float2Texture* texRoughnessMetallic);

        ObjectType getType() const override { return ObjectType::E_UE4SurfaceMaterial; }
    };



    class Node : public Object {
    protected:
        std::string m_name;
    public:
        Node(Context &context, const std::string &name) :
            Object(context), m_name(name) {}
        virtual ~Node() {}
    };



    class SurfaceNode : public Node {
    protected:
        std::set<ParentNode*> m_parents;

    public:
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
            optix::Program callableProgramNullFetchAlpha;
            optix::Program callableProgramNullFetchNormal;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        struct OptiXGeometry {
            optix::Buffer optixIndexBuffer;
            optix::Geometry optixGeometry;
        };

        std::vector<Vertex> m_vertices;
        optix::Buffer m_optixVertexBuffer;
        std::vector<std::vector<uint32_t>> m_sameMaterialGroups;
        std::vector<OptiXGeometry> m_optixGeometries;
        std::vector<SurfaceMaterial*> m_materials;
        std::vector<SHGeometryInstance*> m_shGeometryInstances;
    public:
        static void initialize(Context &context);
        static void finalize(Context &context);

        TriangleMeshSurfaceNode(Context &context, const std::string &name);
        ~TriangleMeshSurfaceNode();

        ObjectType getType() const override { return ObjectType::E_TriangleMeshSurfaceNode; }

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;

        void setVertices(std::vector<Vertex> &&vertices);
        void addMaterialGroup(std::vector<uint32_t> &&indices, SurfaceMaterial* material);
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
        ParentNode(Context &context, const std::string &name, const Transform* localToWorld);
        virtual ~ParentNode();

        enum class UpdateEvent {
            TransformAdded = 0,
            TransformRemoved,
            TransformUpdated,
            GeometryAdded,
            GeometryRemoved,
        };

        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*> &childDelta) = 0;
        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) = 0;
        virtual void setTransform(const Transform* localToWorld);

        void addChild(InternalNode* child);
        void addChild(SurfaceNode* child);
        void removeChild(InternalNode* child);
        void removeChild(SurfaceNode* child);
    };



    class InternalNode : public ParentNode {
        std::set<ParentNode*> m_parents;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        InternalNode(Context &context, const std::string &name, const Transform* localToWorld);

        ObjectType getType() const override { return ObjectType::E_InternalNode; }

        void setTransform(const Transform* localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        RootNode(Context &context, const Transform* localToWorld);

        ObjectType getType() const override { return ObjectType::E_RootNode; }

        SHGroup &getSHGroup() {
            return m_shGroup;
        }
    };



    class Scene : public Object {
        RootNode m_rootNode;

    public:
        Scene(Context &context, const Transform* localToWorld);

        ObjectType getType() const override { return ObjectType::E_Scene; }

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

        SHGroup &getSHGroup() {
            return m_rootNode.getSHGroup();
        }
    };


    class Camera : public Object {
    public:
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
        static void initialize(Context &context);
        static void finalize(Context &context);

        PerspectiveCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                          float aspect, float fovY, float lensRadius, float imgPDist, float objPDist);

        ObjectType getType() const override { return ObjectType::E_PerspectiveCamera; }

        void set() const override;

        void setPosition(const Point3D &position) {
            m_data.position = position;
        }
        void setOrientation(const Quaternion &orientation) {
            m_data.orientation = orientation;
        }
        void setLensRadius(float lensRadius) {
            m_data.lensRadius = lensRadius;
        }
        void setObjectPlaneDistance(float distance) {
            m_data.setObjectPlaneDistance(distance);
        }
    };
}
