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

        optix::Program m_optixProgramPathTracingMiss; // -------- Miss Program
        optix::Program m_optixProgramPathTracing; // ------------ Ray Generation Program
        optix::Program m_optixProgramException; // -------------- Exception Program

    public:
        Context();

        uint32_t getID() const {
            return m_ID;
        }
        
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

        optix::GeometryInstance &getOptiXObject() {
            return m_optixGeometryInstance;
        }
    };

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    class ParentNode;
    class RootNode;



    class Node {
    protected:
        std::string m_name;
        Context &m_context;
    public:
        Node(const std::string &name, Context &context) : 
            m_name(name), m_context(context) {}
        virtual ~Node() {}
    };



    class ParentNode : public Node {
    protected:
        std::set<NodeRef> m_children;
        TransformRef m_localToWorld;

        // key: child SHTransform
        // SHTransform containing only the self transform uses nullptr as the key.
        std::map<const SHTransform*, SHTransform*> m_shTransforms;

        SHGeometryGroup m_shGeomGroup;

    public:
        ParentNode(const std::string &name, Context &context, const TransformRef &localToWorld);
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
        virtual void setTransform(const TransformRef &localToWorld);

        void addChild(const InternalNodeRef &child);
        void addChild(const SurfaceNodeRef &child);
        void removeChild(const InternalNodeRef &child);
        void removeChild(const SurfaceNodeRef &child);
    };



    class InternalNode : public ParentNode {
        std::set<ParentNode*> m_parents;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        InternalNode(const std::string &name, Context &context, const TransformRef &localToWorld);

        void setTransform(const TransformRef &localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        RootNode(Context &context, const TransformRef &localToWorld);

        SHGroup &getSHGroup() {
            return m_shGroup;
        }
    };



    class SurfaceNode : public Node {
    protected:
        std::set<ParentNode*> m_parents;

    public:
        static void init(Context &context);

        SurfaceNode(const std::string &name, Context &context) : Node(name, context) {}
        virtual ~SurfaceNode() {}

        virtual void addParent(ParentNode* parent);
        virtual void removeParent(ParentNode* parent);
    };



    struct Vertex {
        Point3D position;
        Normal3D normal;
        Vector3D tangent;
        TexCoord2D texCoord;
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
        std::vector<SurfaceMaterialRef> m_materials;
        std::vector<SHGeometryInstance*> m_shGeometryInstances;
    public:
        static void init(Context &context);

        TriangleMeshSurfaceNode(const std::string &name, Context &context);
        ~TriangleMeshSurfaceNode();

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;

        void setVertices(std::vector<Vertex> &&vertices);
        void addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterialRef &material);
    };



    class Image2D {
        uint32_t m_width, m_height;
        DataFormat m_dataFormat;
        optix::Buffer m_optixDataBuffer;

    public:
        static DataFormat getInternalFormat(DataFormat inputFormat);

        Image2D(Context &context, uint32_t width, uint32_t height, DataFormat dataFormat);
        virtual ~Image2D() {}

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
    };



    class FloatTexture {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        FloatTexture(Context &context);
        virtual ~FloatTexture() {}

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };
    
    
    
    class Float2Texture {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        Float2Texture(Context &context);
        virtual ~Float2Texture() {}

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };

    
    
    class Float3Texture {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        Float3Texture(Context &context);
        virtual ~Float3Texture() {}

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };



    class Float4Texture {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        Float4Texture(Context &context);
        virtual ~Float4Texture() {}

        optix::TextureSampler &getOptiXObject() {
            return m_optixTextureSampler;
        }
    };



    class ImageFloat4Texture : public Float4Texture {
        Image2DRef m_image;

    public:
        ImageFloat4Texture(Context &context, const Image2DRef &image);
    };



    class SurfaceMaterial {
    protected:
        optix::Material m_optixMaterial;

    public:
        static void init(Context &context);

        SurfaceMaterial(Context &context);
        virtual ~SurfaceMaterial() {}

        optix::Material &getOptiXObject() {
            return m_optixMaterial;
        }
    };

    
    
    class MatteSurfaceMaterial : public SurfaceMaterial {
        struct OptiXProgramSet {
            optix::Program callableProgramGetBaseColor;
            optix::Program callableProgramBSDFmatches;
            optix::Program callableProgramSampleBSDFInternal;
            optix::Program callableProgramEvaluateBSDFInternal;
            optix::Program callableProgramEvaluateBSDF_PDFInternal;
            optix::Program callableProgramEvaluateEmittance;
            optix::Program callableProgramEvaluateEDFInternal;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Float4TextureRef m_texAlbedoRoughness;

    public:
        static void init(Context &context);

        MatteSurfaceMaterial(Context &context, const Float4TextureRef &texAlbedoRoughness);
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        struct OptiXProgramSet {
            optix::Program callableProgramGetBaseColor;
            optix::Program callableProgramBSDFmatches;
            optix::Program callableProgramSampleBSDFInternal;
            optix::Program callableProgramEvaluateBSDFInternal;
            optix::Program callableProgramEvaluateBSDF_PDFInternal;
            optix::Program callableProgramEvaluateEmittance;
            optix::Program callableProgramEvaluateEDFInternal;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Float3TextureRef m_texBaseColor;
        Float2TextureRef m_texRoughnessMetallic;

    public:
        static void init(Context &context);

        UE4SurfaceMaterial(Context &context, const Float3TextureRef &texBaseColor, const Float2TextureRef &texRoughnessMetallic);
    };
}
