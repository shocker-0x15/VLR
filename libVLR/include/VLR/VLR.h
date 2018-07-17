#pragma once

#include "scene.h"

#define VLR_ERROR_NO_ERROR        0x80000000
#define VLR_ERROR_INVALID_CONTEXT 0x80000001
#define VLR_ERROR_INVALID_TYPE    0x80000002

extern "C" {
    typedef uint32_t VLRResult;
    typedef void* VLRObject;
    typedef VLR::Context* VLRContext;
    typedef VLR::Transform VLRTransform;
    typedef VLR::StaticTransform VLRStaticTransform;
    typedef VLR::DataFormat VLRDataFormat;
    typedef VLR::Image2D* VLRImage2D;
    typedef VLR::LinearImage2D* VLRLinearImage2D;
    typedef VLR::FloatTexture* VLRFloatTexture;
    typedef VLR::Float2Texture* VLRFloat2Texture;
    typedef VLR::Float3Texture* VLRFloat3Texture;
    typedef VLR::Float4Texture* VLRFloat4Texture;
    typedef VLR::ImageFloat4Texture* VLRImageFloat4Texture;
    typedef VLR::SurfaceMaterial* VLRSurfaceMaterial;
    typedef VLR::MatteSurfaceMaterial* VLRMatteSurfaceMaterial;
    typedef VLR::UE4SurfaceMaterial* VLRUE4SurfaceMaterial;
    typedef VLR::SurfaceNode* VLRSurfaceNode;
    typedef VLR::Vertex VLRVertex;
    typedef VLR::TriangleMeshSurfaceNode* VLRTriangleMeshSurfaceNode;
    typedef VLR::InternalNode* VLRInternalNode;
    typedef VLR::Scene* VLRScene;

    VLR_API VLRResult vlrCreateContext(VLRContext* context);
    VLR_API VLRResult vlrDestroyContext(VLRContext context);

    VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                             uint32_t width, uint32_t height, VLRDataFormat format, uint8_t* linearData);
    VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image);
    VLR_API VLRResult vlrLinearImage2DGetWidth(VLRLinearImage2D image, uint32_t* width);
    VLR_API VLRResult vlrLinearImage2DGetHeight(VLRLinearImage2D image, uint32_t* height);
    VLR_API VLRResult vlrLinearImage2DGetStride(VLRLinearImage2D image, uint32_t* stride);

    VLR_API VLRResult vlrImageFloat4TextureCreate(VLRContext context, VLRImageFloat4Texture* texture,
                                                  VLRImage2D image);
    VLR_API VLRResult vlrImageFloat4TextureDestroy(VLRContext context, VLRImageFloat4Texture texture);

    VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material,
                                                    VLRFloat4Texture texAlbedoRoughness);
    VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material);

    VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material,
                                                  VLRFloat3Texture texBaseColor, VLRFloat2Texture texRoughnessMetallic);
    VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material);

    VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode,
                                                       const char* name);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, VLRVertex* vertices, uint32_t numVertices);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, uint32_t* indices, uint32_t numIndices, VLRSurfaceMaterial material);

    VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                            const char* name, const VLRStaticTransform* transform);
    VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node);
    VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, const VLRStaticTransform* localToWorld);
    VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRObject child);
    VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRObject child);

    VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                     const VLRStaticTransform* transform);
    VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene);
    VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, const VLRStaticTransform* localToWorld);
    VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRObject child);
    VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRObject child);

    // DELETE ME
    VLR_API VLRResult vlrSceneTest(VLRScene scene);
}

namespace VLRCpp {
    class Image2DHolder;
    class LinearImage2DHolder;
    class FloatTextureHolder;
    class Float2TextureHolder;
    class Float3TextureHolder;
    class Float4TextureHolder;
    class ImageFloat4TextureHolder;
    class SurfaceMaterialHolder;
    class MatteSurfaceMaterialHolder;
    class UE4SurfaceMaterialHolder;
    class NodeHolder;
    class SurfaceNodeHolder;
    class TriangleMeshSurfaceNodeHolder;
    class InternalNodeHolder;
    class SceneHolder;



    typedef std::shared_ptr<VLRStaticTransform> StaticTransformRef;
    typedef std::shared_ptr<Image2DHolder> Image2DRef;
    typedef std::shared_ptr<LinearImage2DHolder> LinearImage2DRef;
    typedef std::shared_ptr<FloatTextureHolder> FloatTextureRef;
    typedef std::shared_ptr<Float2TextureHolder> Float2TextureRef;
    typedef std::shared_ptr<Float3TextureHolder> Float3TextureRef;
    typedef std::shared_ptr<Float4TextureHolder> Float4TextureRef;
    typedef std::shared_ptr<ImageFloat4TextureHolder> ImageFloat4TextureRef;
    typedef std::shared_ptr<SurfaceMaterialHolder> SurfaceMaterialRef;
    typedef std::shared_ptr<MatteSurfaceMaterialHolder> MatteSurfaceMaterialRef;
    typedef std::shared_ptr<UE4SurfaceMaterialHolder> UE4SurfaceMaterialRef;
    typedef std::shared_ptr<NodeHolder> NodeRef;
    typedef std::shared_ptr<SurfaceNodeHolder> SurfaceNodeRef;
    typedef std::shared_ptr<TriangleMeshSurfaceNodeHolder> TriangleMeshSurfaceNodeRef;
    typedef std::shared_ptr<InternalNodeHolder> InternalNodeRef;
    typedef std::shared_ptr<SceneHolder> SceneRef;



    class Object {
    protected:
        VLRContext m_rawContext;

    public:
        Object(VLRContext context) : m_rawContext(context) {}
        virtual ~Object() {}

        virtual VLRObject get() const = 0;
    };



    class Image2DHolder : public Object {
    public:
        Image2DHolder(VLRContext context) : Object(context) {}
    };



    class LinearImage2DHolder : public Image2DHolder {
        VLRLinearImage2D m_raw;

    public:
        LinearImage2DHolder(VLRContext context, uint32_t width, uint32_t height, VLRDataFormat format, uint8_t* linearData) :
            Image2DHolder(context) {
            vlrLinearImage2DCreate(context, &m_raw, width, height, format, linearData);
        }
        ~LinearImage2DHolder() {
            vlrLinearImage2DDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }

        uint32_t getWidth() {
            uint32_t width;
            vlrLinearImage2DGetWidth(m_raw, &width);
            return width;
        }
        uint32_t getHeight() {
            uint32_t height;
            vlrLinearImage2DGetHeight(m_raw, &height);
            return height;
        }
        uint32_t getStride() {
            uint32_t stride;
            vlrLinearImage2DGetStride(m_raw, &stride);
            return stride;
        }
    };



    class FloatTextureHolder : public Object {
    public:
        FloatTextureHolder(VLRContext context) : Object(context) {}
    };
    
    
    
    class Float2TextureHolder : public Object {
    public:
        Float2TextureHolder(VLRContext context) : Object(context) {}
    };
    
    
    
    class Float3TextureHolder : public Object {
    public:
        Float3TextureHolder(VLRContext context) : Object(context) {}
    };
    
    
    
    class Float4TextureHolder : public Object {
    public:
        Float4TextureHolder(VLRContext context) : Object(context) {}
    };



    class ImageFloat4TextureHolder : public Float4TextureHolder {
        VLRImageFloat4Texture m_raw;
        Image2DRef m_image;

    public:
        ImageFloat4TextureHolder(VLRContext context, const Image2DRef &image) :
            Float4TextureHolder(context), m_image(image) {
            vlrImageFloat4TextureCreate(context, &m_raw, (VLRImage2D)m_image->get());
        }
        ~ImageFloat4TextureHolder() {
            vlrImageFloat4TextureDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }
    };



    class SurfaceMaterialHolder : public Object {
    public:
        SurfaceMaterialHolder(VLRContext context) : Object(context) {}
    };



    class MatteSurfaceMaterialHolder : public SurfaceMaterialHolder {
        VLRMatteSurfaceMaterial m_raw;
        Float4TextureRef m_texAlbedoRoughness;

    public:
        MatteSurfaceMaterialHolder(VLRContext context, const Float4TextureRef &texAlbedoRoughness) :
            SurfaceMaterialHolder(context), m_texAlbedoRoughness(texAlbedoRoughness) {
            vlrMatteSurfaceMaterialCreate(context, &m_raw, (VLRFloat4Texture)m_texAlbedoRoughness->get());
        }
        ~MatteSurfaceMaterialHolder() {
            vlrMatteSurfaceMaterialDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }
    };



    class UE4SurfaceMaterialHolder : public SurfaceMaterialHolder {
        VLRUE4SurfaceMaterial m_raw;
        Float3TextureRef m_texBaseColor;
        Float2TextureRef m_texRoughnessMetallic;

    public:
        UE4SurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texBaseColor, const Float2TextureRef &texRoughnessMetallic) :
            SurfaceMaterialHolder(context), m_texBaseColor(texBaseColor), m_texRoughnessMetallic(texRoughnessMetallic) {
            vlrUE4SurfaceMaterialCreate(context, &m_raw, (VLRFloat3Texture)m_texBaseColor->get(), (VLRFloat2Texture)m_texRoughnessMetallic->get());
        }
        ~UE4SurfaceMaterialHolder() {
            vlrUE4SurfaceMaterialDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }
    };
    
    
    
    class NodeHolder : public Object {
    public:
        NodeHolder(VLRContext context) : Object(context) {}
    };



    class SurfaceNodeHolder : public NodeHolder {
    public:
        SurfaceNodeHolder(VLRContext context) : NodeHolder(context) {}
    };



    class TriangleMeshSurfaceNodeHolder : public SurfaceNodeHolder {
        VLRTriangleMeshSurfaceNode m_raw;
        std::set<SurfaceMaterialRef> m_materials;

    public:
        TriangleMeshSurfaceNodeHolder(VLRContext context, const char* name) :
            SurfaceNodeHolder(context) {
            vlrTriangleMeshSurfaceNodeCreate(m_rawContext, &m_raw, name);
        }
        ~TriangleMeshSurfaceNodeHolder() {
            vlrTriangleMeshSurfaceNodeDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }

        void setVertices(VLRVertex* vertices, uint32_t numVertices) {
            vlrTriangleMeshSurfaceNodeSetVertices(m_raw, vertices, numVertices);
        }
        void addMaterialGroup(uint32_t* indices, uint32_t numIndices, const SurfaceMaterialRef &material) {
            m_materials.insert(material);
            vlrTriangleMeshSurfaceNodeAddMaterialGroup(m_raw, indices, numIndices, (VLRSurfaceMaterial)material->get());
        }
    };



    class InternalNodeHolder : public NodeHolder {
        VLRInternalNode m_raw;
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;

    public:
        InternalNodeHolder(VLRContext context, const char* name, const StaticTransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            vlrInternalNodeCreate(m_rawContext, &m_raw, name, m_transform.get());
        }
        ~InternalNodeHolder() {
            vlrInternalNodeDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            vlrInternalNodeSetTransform(m_raw, transform.get());
        }
        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            vlrInternalNodeAddChild(m_raw, child->get());
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            vlrInternalNodeRemoveChild(m_raw, child->get());
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            vlrInternalNodeAddChild(m_raw, child->get());
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            vlrInternalNodeRemoveChild(m_raw, child->get());
        }
    };



    class SceneHolder : public Object {
        VLRScene m_raw;
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;

    public:
        SceneHolder(VLRContext context, const StaticTransformRef &transform) :
            Object(context), m_transform(transform) {
            vlrSceneCreate(m_rawContext, &m_raw, m_transform.get());
        }
        ~SceneHolder() {
            vlrSceneDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return m_raw; }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            vlrSceneSetTransform(m_raw, transform.get());
        }
        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            vlrSceneAddChild(m_raw, child->get());
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            vlrSceneRemoveChild(m_raw, child->get());
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            vlrSceneAddChild(m_raw, child->get());
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            vlrSceneRemoveChild(m_raw, child->get());
        }

        void test() const {
            vlrSceneTest(m_raw);
        }
    };



    class Context {
        VLRContext m_rawContext;

    public:
        Context() {
            VLRResult res = vlrCreateContext(&m_rawContext);
        }
        ~Context() {
            VLRResult res = vlrDestroyContext(m_rawContext);
        }

        LinearImage2DRef createLinearImage2D(uint32_t width, uint32_t height, VLRDataFormat format, uint8_t* linearData) const {
            return std::make_shared<LinearImage2DHolder>(m_rawContext, width, height, format, linearData);
        }

        ImageFloat4TextureRef createImageFloat4Texture(const Image2DRef &image) const {
            return std::make_shared<ImageFloat4TextureHolder>(m_rawContext, image);
        }

        MatteSurfaceMaterialRef createMatteSurfaceMaterial(const Float4TextureRef &texAlbedoRoughness) const {
            return std::make_shared<MatteSurfaceMaterialHolder>(m_rawContext, texAlbedoRoughness);
        }

        UE4SurfaceMaterialRef createUE4SurfaceMaterial(const Float3TextureRef &texBaseColor, const Float2TextureRef &texRoughnessMetallic) const {
            return std::make_shared<UE4SurfaceMaterialHolder>(m_rawContext, texBaseColor, texRoughnessMetallic);
        }

        TriangleMeshSurfaceNodeRef createTriangleMeshSurfaceNode(const char* name) const {
            return std::make_shared<TriangleMeshSurfaceNodeHolder>(m_rawContext, name);
        }

        InternalNodeRef createInternalNode(const char* name, const StaticTransformRef &transform) const {
            return std::make_shared<InternalNodeHolder>(m_rawContext, name, transform);
        }

        SceneRef createScene(const StaticTransformRef &transform) const {
            return std::make_shared<SceneHolder>(m_rawContext, transform);
        }
    };
}
