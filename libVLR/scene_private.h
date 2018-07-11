#pragma once

#include <scene.h>
#include "basic_types_internal.h"
#include "shared.h"

namespace VLR {
    class Context {
        optix::Context m_optixContext;

    public:
        Context() {
            m_optixContext = optix::Context::create();
        }

        optix::Context &getOptiXContext() {
            return m_optixContext;
        }
    };



    class Transform {
    public:
        virtual ~Transform() {}

        virtual bool isStatic() const = 0;
    };



    class StaticTransform : public Transform {
        Matrix4x4 m_matrix;
        Matrix4x4 m_invMatrix;
    public:
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
        std::set<const SHTransform*> m_transforms;
        std::set<const SHGeometryGroup*> m_geometryGroups;

    public:
        SHGroup(Context &context) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixGroup = optixContext->createGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGroup->setAcceleration(m_optixAcceleration);
        }

        void addChild(SHTransform* transform);
        void addChild(SHGeometryGroup* geomGroup);
        void removeChild(SHTransform* transform);
        void removeChild(SHGeometryGroup* geomGroup);

        void childUpdateEvent(const SHTransform* transform);
        void childUpdateEvent(const SHGeometryGroup* geomGroup);

        optix::Group &getOptiXObject() {
            return m_optixGroup;
        }
    };

    class SHTransform {
        std::string m_name;
        SHGroup* m_parent;
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
            m_name(name), m_parent(nullptr), m_transform(transform), m_childTransform(childTransform), m_childIsTransform(childTransform != nullptr) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixTransform = optixContext->createTransform();

            resolveTransform();
        }

        const std::string &getName() const { return m_name; }

        void setTransform(const StaticTransform &transform);
        void update();

        void addChild(SHGeometryGroup* geomGroup);

        void setParent(SHGroup* parent) {
            m_parent = parent;
        }

        optix::Transform &getOptiXObject() {
            return m_optixTransform;
        }
    };

    class SHGeometryGroup {
        optix::GeometryGroup m_optixGeometryGroup;
        optix::Acceleration m_optixAcceleration;
        std::vector<const SHGeometryInstance*> m_instances;

    public:
        SHGeometryGroup(Context &context) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixGeometryGroup = optixContext->createGeometryGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGeometryGroup->setAcceleration(m_optixAcceleration);
        }

        void addGeometryInstance(SHGeometryInstance* instance);

        void childUpdateEvent(const SHGeometryInstance* instance);

        optix::GeometryGroup &getOptiXObject() {
            return m_optixGeometryGroup;
        }
    };

    class SHGeometryInstance {
        std::set<SHGeometryGroup*> m_parents;
        optix::GeometryInstance m_optixGeometryInstance;

        void notifyUpdateToParent() const {
            for (auto parent : m_parents) {
                parent->childUpdateEvent(this);
            }
        }

    public:
        SHGeometryInstance(Context &context) {
            optix::Context &optixContext = context.getOptiXContext();
            m_optixGeometryInstance = optixContext->createGeometryInstance();
        }

        void setParent(SHGeometryGroup* parent) {
            m_parents.insert(parent);
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
        Node(const std::string &name, Context &context) : m_name(name), m_context(context) {}
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
        ParentNode(const std::string &name, Context &context, const TransformRef &localToWorld) : Node(name, context), m_localToWorld(localToWorld), m_shGeomGroup(context) {
            // JP: Ž©•ªŽ©g‚ÌTransform‚ðŽ‚Á‚½SHTransform‚ð¶¬B
            // EN: 
            if (m_localToWorld->isStatic()) {
                StaticTransform* tr = (StaticTransform*)m_localToWorld.get();
                m_shTransforms[nullptr] = new SHTransform(name, m_context, *tr, nullptr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
        ~ParentNode() {
            for (auto it = m_shTransforms.crbegin(); it != m_shTransforms.crend(); ++it)
                delete it->second;
            m_shTransforms.clear();
        }

        enum class UpdateEvent {
            Added = 0,
            Removed,
            Updated,
        };

        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) = 0;
        virtual void setTransform(const TransformRef &localToWorld);

        void addChild(const InternalNodeRef &child);
        void addChild(const SurfaceNodeRef &child);
        void removeChild(const InternalNodeRef &child);
        void removeChild(const SurfaceNodeRef &child);
    };



    class InternalNode : public ParentNode {
        std::set<ParentNode*> m_parents;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) override;

    public:
        InternalNode(const std::string &name, Context &context, const TransformRef &localToWorld) : ParentNode(name, context, localToWorld) {}
        ~InternalNode() {}

        void setTransform(const TransformRef &localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) override;

    public:
        RootNode(Context &context, const TransformRef &localToWorld) : ParentNode("Root", context, localToWorld), m_shGroup(context) {
            SHTransform* shtr = m_shTransforms[0];
            m_shGroup.addChild(shtr);
        }
    };



    class SurfaceNode : public Node {
        std::set<ParentNode*> m_parents;

    public:
        SurfaceNode(const std::string &name, Context &context) : Node(name, context) {}
        virtual ~SurfaceNode() {}

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    struct Vertex {
        Point3D position;
        Normal3D normal;
        Vector3D tangent;
        TexCoord2D texCoord;
    };
    
    class TriangleMeshSurfaceNode : public SurfaceNode {
        struct OptiXGeometry {
            optix::Buffer optixIndexBuffer;
            optix::Geometry optixGeometry;
        };

        std::vector<Vertex> m_vertices;
        optix::Buffer m_optixVertexBuffer;
        std::vector<std::vector<uint32_t>> m_sameMaterialGroups;
        std::vector<OptiXGeometry> m_optixGeometries;
        std::vector<SurfaceMaterialRef> m_materials;
        std::vector<SHGeometryInstance> m_shGeometryInstances;
    public:
        TriangleMeshSurfaceNode(const std::string &name, Context &context) : SurfaceNode(name, context) {}

        void setVertices(std::vector<Vertex> &&vertices);

        void addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterialRef &material);
    };



    class Image2D {
        uint32_t m_width, m_height;
        DataFormat m_dataFormat;

    public:
        Image2D(uint32_t width, uint32_t height, DataFormat dataFormat) :
            m_width(width), m_height(height), m_dataFormat(dataFormat) {}
        virtual ~Image2D() {}

        uint32_t getWidth() const { return m_width; }
        uint32_t getHeight() const { return m_height; }
        uint32_t getStride() const { return (uint32_t)sizesOfDataFormats[(uint32_t)m_dataFormat]; }

        static DataFormat getInternalFormat(DataFormat inputFormat);
    };



    class LinearImage2D : public Image2D {
        std::vector<uint8_t> m_data;

    public:
        LinearImage2D(const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat);
    };



    class FloatTexture {
    public:
        virtual ~FloatTexture() {}
    };
    
    
    
    class Float2Texture {
    public:
        virtual ~Float2Texture() {}
    };

    
    
    class Float3Texture {
    public:
        virtual ~Float3Texture() {}
    };



    class Float4Texture {
    public:
        virtual ~Float4Texture() {}
    };



    class ImageFloat4Texture : public Float4Texture {
    public:
        ImageFloat4Texture() {}
    };



    class SurfaceMaterial {
    public:
        virtual ~SurfaceMaterial() {}
    };

    
    
    class MatteSurfaceMaterial : public SurfaceMaterial {
        Float4TextureRef m_albedoRoughnessTex;

    public:
        MatteSurfaceMaterial(const Float4TextureRef &albedoRoughnessTex) : m_albedoRoughnessTex(albedoRoughnessTex) {}
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        Float3TextureRef m_baseColorTex;
        Float2TextureRef m_roughnessMetallicTex;
    public:
        UE4SurfaceMaterial(const Float3TextureRef &baseColorTex, const Float2TextureRef &roughnessMetallicTex) :
            m_baseColorTex(baseColorTex), m_roughnessMetallicTex(roughnessMetallicTex) {}
    };
}
