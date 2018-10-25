#pragma once

#include <algorithm>
#include <vector>
#include <memory>

#include "VLR.h"
#include "basic_types.h"

namespace VLRCpp {
#define VLR_DECLARE_HOLDER_AND_REFERENCE(Name)\
    class Name ## Holder;\
    typedef std::shared_ptr<Name ## Holder> Name ## Ref

    VLR_DECLARE_HOLDER_AND_REFERENCE(Image2D);
    VLR_DECLARE_HOLDER_AND_REFERENCE(LinearImage2D);

    VLR_DECLARE_HOLDER_AND_REFERENCE(ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(OffsetAndScaleUVTextureMap2DShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ConstantTextureShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Image2DTextureShaderNode);

    VLR_DECLARE_HOLDER_AND_REFERENCE(FloatTexture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ConstantFloatTexture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float2Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ConstantFloat2Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float3Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ConstantFloat3Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ImageFloat3Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float4Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ConstantFloat4Texture);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ImageFloat4Texture);

    VLR_DECLARE_HOLDER_AND_REFERENCE(SurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(MatteSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(SpecularReflectionSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(SpecularScatteringSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(MicrofacetReflectionSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(MicrofacetScatteringSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(LambertianScatteringSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(UE4SurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(DiffuseEmitterSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(MultiSurfaceMaterial);
    VLR_DECLARE_HOLDER_AND_REFERENCE(EnvironmentEmitterSurfaceMaterial);

    VLR_DECLARE_HOLDER_AND_REFERENCE(Transform);
    VLR_DECLARE_HOLDER_AND_REFERENCE(StaticTransform);

    VLR_DECLARE_HOLDER_AND_REFERENCE(Node);
    VLR_DECLARE_HOLDER_AND_REFERENCE(SurfaceNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(TriangleMeshSurfaceNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(InternalNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Scene);

    VLR_DECLARE_HOLDER_AND_REFERENCE(Camera);
    VLR_DECLARE_HOLDER_AND_REFERENCE(PerspectiveCamera);
    VLR_DECLARE_HOLDER_AND_REFERENCE(EquirectangularCamera);



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
        LinearImage2DHolder(VLRContext context, uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData) :
            Image2DHolder(context) {
            VLRResult res = vlrLinearImage2DCreate(context, &m_raw, width, height, format, applyDegamma, linearData);
        }
        ~LinearImage2DHolder() {
            VLRResult res = vlrLinearImage2DDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        uint32_t getWidth() {
            uint32_t width;
            VLRResult res = vlrLinearImage2DGetWidth(m_raw, &width);
            return width;
        }
        uint32_t getHeight() {
            uint32_t height;
            VLRResult res = vlrLinearImage2DGetHeight(m_raw, &height);
            return height;
        }
        uint32_t getStride() {
            uint32_t stride;
            VLRResult res = vlrLinearImage2DGetStride(m_raw, &stride);
            return stride;
        }
    };



    class ShaderNodeHolder : public Object {
    protected:
        VLRShaderNode m_raw;

    public:
        ShaderNodeHolder(VLRContext context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class OffsetAndScaleUVTextureMap2DShaderNodeHolder : public ShaderNodeHolder {
        float m_offset[2];
        float m_scale[2];

    public:
        OffsetAndScaleUVTextureMap2DShaderNodeHolder(VLRContext context, const float offset[2], const float scale[2]) :
            ShaderNodeHolder(context), m_offset{ offset[0], offset[1] }, m_scale{ scale[0], scale[1] } {
            VLRResult res = vlrOffsetAndScaleUVTextureMap2DShaderNodeCreate(context, (VLROffsetAndScaleUVTextureMap2DShaderNode*)&m_raw, m_offset, m_scale);
        }
        ~OffsetAndScaleUVTextureMap2DShaderNodeHolder() {
            VLRResult res = vlrOffsetAndScaleUVTextureMap2DShaderNodeDestroy(m_rawContext, (VLROffsetAndScaleUVTextureMap2DShaderNode)m_raw);
        }
    };



    class ConstantTextureShaderNodeHolder : public ShaderNodeHolder {
        VLR::RGBSpectrum m_spectrum;
        float m_alpha;

    public:
        ConstantTextureShaderNodeHolder(VLRContext context, const VLR::RGBSpectrum &spectrum, float alpha) :
            ShaderNodeHolder(context), m_spectrum(spectrum), m_alpha(alpha) {
            float _spectrum[] = { spectrum.r, spectrum.g, spectrum.b };
            VLRResult res = vlrConstantTextureShaderNodeCreate(context, (VLRConstantTextureShaderNode*)&m_raw, _spectrum, m_alpha);
        }
        ~ConstantTextureShaderNodeHolder() {
            VLRResult res = vlrConstantTextureShaderNodeDestroy(m_rawContext, (VLRConstantTextureShaderNode)m_raw);
        }
    };



    class Image2DTextureShaderNodeHolder : public ShaderNodeHolder {
        Image2DRef m_image;
        ShaderNodeRef m_nodeTexCoord;

    public:
        Image2DTextureShaderNodeHolder(VLRContext context, const Image2DRef &image, const ShaderNodeRef &nodeTexCoord) :
            ShaderNodeHolder(context), m_image(image), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrImage2DTextureShaderNodeCreate(context, (VLRImage2DTextureShaderNode*)&m_raw, (VLRImage2D)image->get(), (VLRShaderNode)m_nodeTexCoord->get());
        }
        ~Image2DTextureShaderNodeHolder() {
            VLRResult res = vlrImage2DTextureShaderNodeDestroy(m_rawContext, (VLRImage2DTextureShaderNode)m_raw);
        }

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            VLRResult res = vlrImage2DTextureShaderNodeSetFilterMode(m_rawContext, (VLRImage2DTextureShaderNode)m_raw, minification, magnification, mipmapping);
        }
    };



    class FloatTextureHolder : public Object {
    protected:
        VLRFloatTexture m_raw;

    public:
        FloatTextureHolder(VLRContext context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class ConstantFloatTextureHolder : public FloatTextureHolder {
    public:
        ConstantFloatTextureHolder(VLRContext context, const float value) :
            FloatTextureHolder(context) {
            VLRResult res = vlrConstantFloatTextureCreate(context, (VLRConstantFloatTexture*)&m_raw, value);
        }
        ~ConstantFloatTextureHolder() {
            VLRResult res = vlrConstantFloatTextureDestroy(m_rawContext, (VLRConstantFloatTexture)m_raw);
        }
    };



    class Float2TextureHolder : public Object {
    protected:
        VLRFloat2Texture m_raw;

    public:
        Float2TextureHolder(VLRContext context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class ConstantFloat2TextureHolder : public Float2TextureHolder {
    public:
        ConstantFloat2TextureHolder(VLRContext context, const float value[2]) :
            Float2TextureHolder(context) {
            VLRResult res = vlrConstantFloat2TextureCreate(context, (VLRConstantFloat2Texture*)&m_raw, value);
        }
        ~ConstantFloat2TextureHolder() {
            VLRResult res = vlrConstantFloat2TextureDestroy(m_rawContext, (VLRConstantFloat2Texture)m_raw);
        }
    };



    class Float3TextureHolder : public Object {
    protected:
        VLRFloat3Texture m_raw;

    public:
        Float3TextureHolder(VLRContext context) : Object(context) {}

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            VLRResult res = vlrFloat3TextureSetFilterMode(m_rawContext, m_raw, minification, magnification, mipmapping);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class ConstantFloat3TextureHolder : public Float3TextureHolder {
    public:
        ConstantFloat3TextureHolder(VLRContext context, const float value[3]) :
            Float3TextureHolder(context) {
            VLRResult res = vlrConstantFloat3TextureCreate(context, (VLRConstantFloat3Texture*)&m_raw, value);
        }
        ~ConstantFloat3TextureHolder() {
            VLRResult res = vlrConstantFloat3TextureDestroy(m_rawContext, (VLRConstantFloat3Texture)m_raw);
        }
    };



    class ImageFloat3TextureHolder : public Float3TextureHolder {
        Image2DRef m_image;

    public:
        ImageFloat3TextureHolder(VLRContext context, const Image2DRef &image) :
            Float3TextureHolder(context), m_image(image) {
            VLRResult res = vlrImageFloat3TextureCreate(context, (VLRImageFloat3Texture*)&m_raw, (VLRImage2D)m_image->get());
        }
        ~ImageFloat3TextureHolder() {
            VLRResult res = vlrImageFloat3TextureDestroy(m_rawContext, (VLRImageFloat3Texture)m_raw);
        }
    };



    class Float4TextureHolder : public Object {
    protected:
        VLRFloat4Texture m_raw;

    public:
        Float4TextureHolder(VLRContext context) : Object(context) {}

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            VLRResult res = vlrFloat4TextureSetFilterMode(m_rawContext, m_raw, minification, magnification, mipmapping);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class ConstantFloat4TextureHolder : public Float4TextureHolder {
    public:
        ConstantFloat4TextureHolder(VLRContext context, const float value[4]) :
            Float4TextureHolder(context) {
            VLRResult res = vlrConstantFloat4TextureCreate(context, (VLRConstantFloat4Texture*)&m_raw, value);
        }
        ~ConstantFloat4TextureHolder() {
            VLRResult res = vlrConstantFloat4TextureDestroy(m_rawContext, (VLRConstantFloat4Texture)m_raw);
        }
    };



    class ImageFloat4TextureHolder : public Float4TextureHolder {
        Image2DRef m_image;

    public:
        ImageFloat4TextureHolder(VLRContext context, const Image2DRef &image) :
            Float4TextureHolder(context), m_image(image) {
            VLRResult res = vlrImageFloat4TextureCreate(context, (VLRImageFloat4Texture*)&m_raw, (VLRImage2D)m_image->get());
        }
        ~ImageFloat4TextureHolder() {
            VLRResult res = vlrImageFloat4TextureDestroy(m_rawContext, (VLRImageFloat4Texture)m_raw);
        }
    };



    class SurfaceMaterialHolder : public Object {
    protected:
        VLRSurfaceMaterial m_raw;

    public:
        SurfaceMaterialHolder(VLRContext context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class MatteSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeRef m_nodeAlbedo;

    public:
        MatteSurfaceMaterialHolder(VLRContext context, const ShaderNodeRef &nodeAlbedo) :
            SurfaceMaterialHolder(context), m_nodeAlbedo(nodeAlbedo) {
            VLRResult res = vlrMatteSurfaceMaterialCreate(context, (VLRMatteSurfaceMaterial*)&m_raw,
                                                          m_nodeAlbedo ? (VLRShaderNode)m_nodeAlbedo->get() : nullptr);
        }
        ~MatteSurfaceMaterialHolder() {
            VLRResult res = vlrMatteSurfaceMaterialDestroy(m_rawContext, (VLRMatteSurfaceMaterial)m_raw);
        }
    };



    class SpecularReflectionSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texCoeffR;
        Float3TextureRef m_texEta;
        Float3TextureRef m_tex_k;
        ShaderNodeRef m_nodeTexCoord;

    public:
        SpecularReflectionSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texCoeffR, const Float3TextureRef &texEta, const Float3TextureRef &tex_k, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texCoeffR(texCoeffR), m_texEta(texEta), m_tex_k(tex_k), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrSpecularReflectionSurfaceMaterialCreate(context, (VLRSpecularReflectionSurfaceMaterial*)&m_raw,
                                                                       (VLRFloat3Texture)m_texCoeffR->get(), (VLRFloat3Texture)m_texEta->get(), (VLRFloat3Texture)m_tex_k->get(),
                                                                       m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~SpecularReflectionSurfaceMaterialHolder() {
            VLRResult res = vlrSpecularReflectionSurfaceMaterialDestroy(m_rawContext, (VLRSpecularReflectionSurfaceMaterial)m_raw);
        }
    };



    class SpecularScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texCoeff;
        Float3TextureRef m_texEtaExt;
        Float3TextureRef m_texEtaInt;
        ShaderNodeRef m_nodeTexCoord;

    public:
        SpecularScatteringSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texCoeff, const Float3TextureRef &texEtaExt, const Float3TextureRef &texEtaInt, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texCoeff(texCoeff), m_texEtaExt(texEtaExt), m_texEtaInt(texEtaInt), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrSpecularScatteringSurfaceMaterialCreate(context, (VLRSpecularScatteringSurfaceMaterial*)&m_raw,
                                                                       (VLRFloat3Texture)m_texCoeff->get(), (VLRFloat3Texture)m_texEtaExt->get(), (VLRFloat3Texture)m_texEtaInt->get(),
                                                                       m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~SpecularScatteringSurfaceMaterialHolder() {
            VLRResult res = vlrSpecularScatteringSurfaceMaterialDestroy(m_rawContext, (VLRSpecularScatteringSurfaceMaterial)m_raw);
        }
    };



    class MicrofacetReflectionSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texEta;
        Float3TextureRef m_tex_k;
        Float2TextureRef m_texRoughness;
        ShaderNodeRef m_nodeTexCoord;

    public:
        MicrofacetReflectionSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texEta, const Float3TextureRef &tex_k, const Float2TextureRef &texRoughness, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texEta(texEta), m_tex_k(tex_k), m_texRoughness(texRoughness), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrMicrofacetReflectionSurfaceMaterialCreate(context, (VLRMicrofacetReflectionSurfaceMaterial*)&m_raw,
                                                                         (VLRFloat3Texture)m_texEta->get(), (VLRFloat3Texture)m_tex_k->get(), (VLRFloat2Texture)m_texRoughness->get(),
                                                                         m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~MicrofacetReflectionSurfaceMaterialHolder() {
            VLRResult res = vlrMicrofacetReflectionSurfaceMaterialDestroy(m_rawContext, (VLRMicrofacetReflectionSurfaceMaterial)m_raw);
        }
    };



    class MicrofacetScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texCoeff;
        Float3TextureRef m_texEtaExt;
        Float3TextureRef m_texEtaInt;
        Float2TextureRef m_texRoughness;
        ShaderNodeRef m_nodeTexCoord;

    public:
        MicrofacetScatteringSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texCoeff, const Float3TextureRef &texEtaExt, const Float3TextureRef &texEtaInt, const Float2TextureRef &texRoughness, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texCoeff(texCoeff), m_texEtaExt(texEtaExt), m_texEtaInt(texEtaInt), m_texRoughness(texRoughness), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrMicrofacetScatteringSurfaceMaterialCreate(context, (VLRMicrofacetScatteringSurfaceMaterial*)&m_raw,
                                                                         (VLRFloat3Texture)m_texCoeff->get(), (VLRFloat3Texture)m_texEtaExt->get(), (VLRFloat3Texture)m_texEtaInt->get(), (VLRFloat2Texture)m_texRoughness->get(),
                                                                         m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~MicrofacetScatteringSurfaceMaterialHolder() {
            VLRResult res = vlrMicrofacetScatteringSurfaceMaterialDestroy(m_rawContext, (VLRMicrofacetScatteringSurfaceMaterial)m_raw);
        }
    };



    class LambertianScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texCoeff;
        FloatTextureRef m_texF0;
        ShaderNodeRef m_nodeTexCoord;

    public:
        LambertianScatteringSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texCoeff, const FloatTextureRef &texF0, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texCoeff(texCoeff), m_texF0(texF0), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrLambertianScatteringSurfaceMaterialCreate(context, (VLRLambertianScatteringSurfaceMaterial*)&m_raw,
                                                                         (VLRFloat3Texture)m_texCoeff->get(), (VLRFloatTexture)m_texF0->get(),
                                                                         m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~LambertianScatteringSurfaceMaterialHolder() {
            VLRResult res = vlrLambertianScatteringSurfaceMaterialDestroy(m_rawContext, (VLRLambertianScatteringSurfaceMaterial)m_raw);
        }
    };



    class UE4SurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texBaseColor;
        Float3TextureRef m_texOcclusionRoughnessMetallic;
        ShaderNodeRef m_nodeTexCoord;

    public:
        UE4SurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texBaseColor, const Float3TextureRef &texOcclusionRoughnessMetallic, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texBaseColor(texBaseColor), m_texOcclusionRoughnessMetallic(texOcclusionRoughnessMetallic), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrUE4SurfaceMaterialCreate(context, (VLRUE4SurfaceMaterial*)&m_raw,
                                                        (VLRFloat3Texture)m_texBaseColor->get(), (VLRFloat3Texture)m_texOcclusionRoughnessMetallic->get(),
                                                        m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~UE4SurfaceMaterialHolder() {
            VLRResult res = vlrUE4SurfaceMaterialDestroy(m_rawContext, (VLRUE4SurfaceMaterial)m_raw);
        }
    };



    class DiffuseEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texEmittance;
        ShaderNodeRef m_nodeTexCoord;

    public:
        DiffuseEmitterSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texEmittance, const ShaderNodeRef &nodeTexCoord) :
            SurfaceMaterialHolder(context), m_texEmittance(texEmittance), m_nodeTexCoord(nodeTexCoord) {
            VLRResult res = vlrDiffuseEmitterSurfaceMaterialCreate(context, (VLRDiffuseEmitterSurfaceMaterial*)&m_raw,
                                                                   (VLRFloat3Texture)m_texEmittance->get(),
                                                                   m_nodeTexCoord ? (VLRShaderNode)m_nodeTexCoord->get() : nullptr);
        }
        ~DiffuseEmitterSurfaceMaterialHolder() {
            VLRResult res = vlrDiffuseEmitterSurfaceMaterialDestroy(m_rawContext, (VLRDiffuseEmitterSurfaceMaterial)m_raw);
        }
    };



    class MultiSurfaceMaterialHolder : public SurfaceMaterialHolder {
        std::vector<SurfaceMaterialRef> m_materials;

    public:
        MultiSurfaceMaterialHolder(VLRContext context, const SurfaceMaterialRef* materials, uint32_t numMaterials) :
            SurfaceMaterialHolder(context) {
            for (int i = 0; i < numMaterials; ++i)
                m_materials.push_back(materials[i]);

            VLRSurfaceMaterial rawMats[4];
            for (int i = 0; i < numMaterials; ++i)
                rawMats[i] = (VLRSurfaceMaterial)materials[i]->get();
            VLRResult res = vlrMultiSurfaceMaterialCreate(context, (VLRMultiSurfaceMaterial*)&m_raw, rawMats, numMaterials);
        }
        ~MultiSurfaceMaterialHolder() {
            VLRResult res = vlrMultiSurfaceMaterialDestroy(m_rawContext, (VLRMultiSurfaceMaterial)m_raw);
        }
    };



    class EnvironmentEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        Float3TextureRef m_texEmittance;

    public:
        EnvironmentEmitterSurfaceMaterialHolder(VLRContext context, const Float3TextureRef &texEmittance) :
            SurfaceMaterialHolder(context), m_texEmittance(texEmittance) {
            VLRResult res = vlrEnvironmentEmitterSurfaceMaterialCreate(context, (VLREnvironmentEmitterSurfaceMaterial*)&m_raw, (VLRFloat3Texture)m_texEmittance->get());
        }
        ~EnvironmentEmitterSurfaceMaterialHolder() {
            VLRResult res = vlrEnvironmentEmitterSurfaceMaterialDestroy(m_rawContext, (VLREnvironmentEmitterSurfaceMaterial)m_raw);
        }
    };



    class TransformHolder : public Object {
    protected:
        VLRTransform m_raw;
    public:
        TransformHolder(VLRContext context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class StaticTransformHolder : public TransformHolder {
    public:
        StaticTransformHolder(VLRContext context, const float mat[16]) : TransformHolder(context) {
            VLRResult res = vlrStaticTransformCreate(m_rawContext, (VLRStaticTransform*)&m_raw, mat);
        }
        StaticTransformHolder(VLRContext context, const VLR::Matrix4x4 &mat) : TransformHolder(context) {
            float matArray[16];
            mat.getArray(matArray);
            VLRResult res = vlrStaticTransformCreate(m_rawContext, (VLRStaticTransform*)&m_raw, matArray);
        }
        ~StaticTransformHolder() {
            VLRResult res = vlrStaticTransformDestroy(m_rawContext, (VLRStaticTransform)m_raw);
        }
    };



    enum class NodeType {
        TriangleMeshSurfaceNode = 0,
        InternalNode,
    };

    class NodeHolder : public Object {
    public:
        NodeHolder(VLRContext context) : Object(context) {}

        virtual NodeType getNodeType() const = 0;
        virtual void setName(const std::string &name) const = 0;
        virtual const char* getName() const = 0;
    };



    class SurfaceNodeHolder : public NodeHolder {
    public:
        SurfaceNodeHolder(VLRContext context) : NodeHolder(context) {}
    };



    class TriangleMeshSurfaceNodeHolder : public SurfaceNodeHolder {
        VLRTriangleMeshSurfaceNode m_raw;
        std::vector<SurfaceMaterialRef> m_materials;
        std::vector<Float4TextureRef> m_texNormalAlphas;

    public:
        TriangleMeshSurfaceNodeHolder(VLRContext context, const char* name) :
            SurfaceNodeHolder(context) {
            VLRResult res = vlrTriangleMeshSurfaceNodeCreate(m_rawContext, &m_raw, name);
        }
        ~TriangleMeshSurfaceNodeHolder() {
            VLRResult res = vlrTriangleMeshSurfaceNodeDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        NodeType getNodeType() const override { return NodeType::TriangleMeshSurfaceNode; }
        void setName(const std::string &name) const override {
            VLRResult res = vlrTriangleMeshSurfaceNodeSetName(m_raw, name.c_str());
        }
        const char* getName() const override {
            const char* name;
            VLRResult res = vlrTriangleMeshSurfaceNodeGetName(m_raw, &name);
            return name;
        }

        void setVertices(VLR::Vertex* vertices, uint32_t numVertices) {
            VLRResult res = vlrTriangleMeshSurfaceNodeSetVertices(m_raw, (VLRVertex*)vertices, numVertices);
        }
        void addMaterialGroup(uint32_t* indices, uint32_t numIndices,
                              const SurfaceMaterialRef &material, const Float4TextureRef &texNormalAlpha) {
            m_materials.push_back(material);
            m_texNormalAlphas.push_back(texNormalAlpha);
            if (texNormalAlpha) {
                VLRResult res = vlrTriangleMeshSurfaceNodeAddMaterialGroup(m_raw, indices, numIndices,
                    (VLRSurfaceMaterial)material->get(), (VLRFloat4Texture)texNormalAlpha->get());
            }
            else {
                VLRResult res = vlrTriangleMeshSurfaceNodeAddMaterialGroup(m_raw, indices, numIndices,
                    (VLRSurfaceMaterial)material->get(), nullptr);
            }
        }
    };



    class InternalNodeHolder : public NodeHolder {
        VLRInternalNode m_raw;
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;

    public:
        InternalNodeHolder(VLRContext context, const char* name, const StaticTransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            VLRResult res = vlrInternalNodeCreate(m_rawContext, &m_raw, name, (VLRTransform)m_transform->get());
        }
        ~InternalNodeHolder() {
            VLRResult res = vlrInternalNodeDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        NodeType getNodeType() const override { return NodeType::InternalNode; }
        void setName(const std::string &name) const override {
            VLRResult res = vlrInternalNodeSetName(m_raw, name.c_str());
        }
        const char* getName() const override {
            const char* name;
            VLRResult res = vlrInternalNodeGetName(m_raw, &name);
            return name;
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            VLRResult res = vlrInternalNodeSetTransform(m_raw, (VLRTransform)transform->get());
        }
        StaticTransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            VLRResult res = vlrInternalNodeAddChild(m_raw, child->get());
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            VLRResult res = vlrInternalNodeRemoveChild(m_raw, child->get());
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            VLRResult res = vlrInternalNodeAddChild(m_raw, child->get());
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            VLRResult res = vlrInternalNodeRemoveChild(m_raw, child->get());
        }
        uint32_t getNumChildren() const {
            return (uint32_t)m_children.size();
        }
        NodeRef getChildAt(uint32_t index) const {
            auto it = m_children.cbegin();
            std::advance(it, index);
            return *it;
        }
    };



    class SceneHolder : public Object {
        VLRScene m_raw;
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;
        EnvironmentEmitterSurfaceMaterialRef m_matEnv;

    public:
        SceneHolder(VLRContext context, const StaticTransformRef &transform) :
            Object(context), m_transform(transform) {
            VLRResult res = vlrSceneCreate(m_rawContext, &m_raw, (VLRTransform)m_transform->get());
        }
        ~SceneHolder() {
            VLRResult res = vlrSceneDestroy(m_rawContext, m_raw);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            VLRResult res = vlrSceneSetTransform(m_raw, (VLRTransform)transform->get());
        }
        StaticTransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            VLRResult res = vlrSceneAddChild(m_raw, child->get());
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            VLRResult res = vlrSceneRemoveChild(m_raw, child->get());
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            VLRResult res = vlrSceneAddChild(m_raw, child->get());
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            VLRResult res = vlrSceneRemoveChild(m_raw, child->get());
        }
        uint32_t getNumChildren() const {
            return (uint32_t)m_children.size();
        }
        NodeRef getChildAt(uint32_t index) const {
            auto it = m_children.cbegin();
            std::advance(it, index);
            return *it;
        }

        void setEnvironment(const EnvironmentEmitterSurfaceMaterialRef &matEnv) {
            m_matEnv = matEnv;
            VLRResult res = vlrSceneSetEnvironment(m_raw, (VLREnvironmentEmitterSurfaceMaterial)m_matEnv->get());
        }
    };



    class CameraHolder : public Object {
    protected:
        VLRCamera m_raw;

    public:
        CameraHolder(VLRContext context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class PerspectiveCameraHolder : public CameraHolder {
    public:
        PerspectiveCameraHolder(VLRContext context, const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) :
            CameraHolder(context) {
            VLRResult res = vlrPerspectiveCameraCreate(context, (VLRPerspectiveCamera*)&m_raw, (VLRPoint3D*)&position, (VLRQuaternion*)&orientation,
                                                       sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist);
        }
        ~PerspectiveCameraHolder() {
            VLRResult res = vlrPerspectiveCameraDestroy(m_rawContext, (VLRPerspectiveCamera)m_raw);
        }

        void setPosition(const VLR::Point3D &position) {
            VLRResult res = vlrPerspectiveCameraSetPosition((VLRPerspectiveCamera)m_raw, (VLRPoint3D*)&position);
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            VLRResult res = vlrPerspectiveCameraSetOrientation((VLRPerspectiveCamera)m_raw, (VLRQuaternion*)&orientation);
        }
        void setSensitivity(float sensitivity) {
            VLRResult res = vlrPerspectiveCameraSetSensitivity((VLRPerspectiveCamera)m_raw, sensitivity);
        }
        void setFovY(float fovY) {
            VLRResult res = vlrPerspectiveCameraSetFovY((VLRPerspectiveCamera)m_raw, fovY);
        }
        void setLensRadius(float lensRadius) {
            VLRResult res = vlrPerspectiveCameraSetLensRadius((VLRPerspectiveCamera)m_raw, lensRadius);
        }
        void setObjectPlaneDistance(float distance) {
            VLRResult res = vlrPerspectiveCameraSetObjectPlaneDistance((VLRPerspectiveCamera)m_raw, distance);
        }
    };



    class EquirectangularCameraHolder : public CameraHolder {
    public:
        EquirectangularCameraHolder(VLRContext context, const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                    float sensitivity, float phiAngle, float thetaAngle) :
            CameraHolder(context) {
            VLRResult res = vlrEquirectangularCameraCreate(context, (VLREquirectangularCamera*)&m_raw, (VLRPoint3D*)&position, (VLRQuaternion*)&orientation,
                                                           sensitivity, phiAngle, thetaAngle);
        }
        ~EquirectangularCameraHolder() {
            VLRResult res = vlrEquirectangularCameraDestroy(m_rawContext, (VLREquirectangularCamera)m_raw);
        }

        void setPosition(const VLR::Point3D &position) {
            VLRResult res = vlrEquirectangularCameraSetPosition((VLREquirectangularCamera)m_raw, (VLRPoint3D*)&position);
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            VLRResult res = vlrEquirectangularCameraSetOrientation((VLREquirectangularCamera)m_raw, (VLRQuaternion*)&orientation);
        }
        void setSensitivity(float sensitivity) {
            VLRResult res = vlrEquirectangularCameraSetSensitivity((VLREquirectangularCamera)m_raw, sensitivity);
        }
        void setAngles(float phiAngle, float thetaAngle) {
            VLRResult res = vlrEquirectangularCameraSetAngles((VLREquirectangularCamera)m_raw, phiAngle, thetaAngle);
        }
    };



    class Context {
        VLRContext m_rawContext;

    public:
        Context(bool logging, uint32_t stackSize) {
            VLRResult res = vlrCreateContext(&m_rawContext, logging, stackSize);
        }
        ~Context() {
            VLRResult res = vlrDestroyContext(m_rawContext);
        }

        VLRContext get() const {
            return m_rawContext;
        }

        void setDevices(const int32_t* devices, uint32_t numDevices) const {
            VLRResult res = vlrContextSetDevices(m_rawContext, devices, numDevices);
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID) const {
            VLRResult res = vlrContextBindOutputBuffer(m_rawContext, width, height, glBufferID);
        }

        void* mapOutputBuffer() const {
            void* ptr = nullptr;
            VLRResult res = vlrContextMapOutputBuffer(m_rawContext, &ptr);
            return ptr;
        }

        void unmapOutputBuffer() const {
            VLRResult res = vlrContextUnmapOutputBuffer(m_rawContext);
        }

        void render(const SceneRef &scene, const CameraRef &camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            VLRResult res = vlrContextRender(m_rawContext, (VLRScene)scene->get(), (VLRCamera)camera->get(), shrinkCoeff, firstFrame, numAccumFrames);
        }

        LinearImage2DRef createLinearImage2D(uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData) const {
            return std::make_shared<LinearImage2DHolder>(m_rawContext, width, height, format, applyDegamma, linearData);
        }

        OffsetAndScaleUVTextureMap2DShaderNodeRef createOffsetAndScaleUVTextureMap2DShaderNode(const float offset[2], const float scale[2]) const {
            return std::make_shared<OffsetAndScaleUVTextureMap2DShaderNodeHolder>(m_rawContext, offset, scale);
        }

        ConstantTextureShaderNodeRef createConstantTextureShaderNode(const VLR::RGBSpectrum &spectrum, float alpha) const {
            return std::make_shared<ConstantTextureShaderNodeHolder>(m_rawContext, spectrum, alpha);
        }

        Image2DTextureShaderNodeRef createImage2DTextureShaderNode(const Image2DRef &image, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<Image2DTextureShaderNodeHolder>(m_rawContext, image, nodeTexCoord);
        }

        ConstantFloatTextureRef createConstantFloatTexture(const float value) const {
            return std::make_shared<ConstantFloatTextureHolder>(m_rawContext, value);
        }

        ConstantFloat2TextureRef createConstantFloat2Texture(const float value[2]) const {
            return std::make_shared<ConstantFloat2TextureHolder>(m_rawContext, value);
        }

        ConstantFloat3TextureRef createConstantFloat3Texture(const float value[3]) const {
            return std::make_shared<ConstantFloat3TextureHolder>(m_rawContext, value);
        }

        ImageFloat3TextureRef createImageFloat3Texture(const Image2DRef &image) const {
            return std::make_shared<ImageFloat3TextureHolder>(m_rawContext, image);
        }

        ConstantFloat4TextureRef createConstantFloat4Texture(const float value[4]) const {
            return std::make_shared<ConstantFloat4TextureHolder>(m_rawContext, value);
        }

        ImageFloat4TextureRef createImageFloat4Texture(const Image2DRef &image) const {
            return std::make_shared<ImageFloat4TextureHolder>(m_rawContext, image);
        }

        MatteSurfaceMaterialRef createMatteSurfaceMaterial(const ShaderNodeRef &nodeAlbedo) const {
            return std::make_shared<MatteSurfaceMaterialHolder>(m_rawContext, nodeAlbedo);
        }

        SpecularReflectionSurfaceMaterialRef createSpecularReflectionSurfaceMaterial(const Float3TextureRef &texCoeffR, const Float3TextureRef &texEta, const Float3TextureRef &tex_k, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<SpecularReflectionSurfaceMaterialHolder>(m_rawContext, texCoeffR, texEta, tex_k, nodeTexCoord);
        }

        SpecularScatteringSurfaceMaterialRef createSpecularScatteringSurfaceMaterial(const Float3TextureRef &texCoeff, const Float3TextureRef &texEtaExt, const Float3TextureRef &texEtaInt, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<SpecularScatteringSurfaceMaterialHolder>(m_rawContext, texCoeff, texEtaExt, texEtaInt, nodeTexCoord);
        }

        MicrofacetReflectionSurfaceMaterialRef createMicrofacetReflectionSurfaceMaterial(const Float3TextureRef &texEta, const Float3TextureRef &tex_k, const Float2TextureRef &texRoughness, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<MicrofacetReflectionSurfaceMaterialHolder>(m_rawContext, texEta, tex_k, texRoughness, nodeTexCoord);
        }

        MicrofacetScatteringSurfaceMaterialRef createMicrofacetScatteringSurfaceMaterial(const Float3TextureRef &texCoeff, const Float3TextureRef &texEtaExt, const Float3TextureRef &texEtaInt, const Float2TextureRef &texRoughness, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<MicrofacetScatteringSurfaceMaterialHolder>(m_rawContext, texCoeff, texEtaExt, texEtaInt, texRoughness, nodeTexCoord);
        }

        LambertianScatteringSurfaceMaterialRef createLambertianScatteringSurfaceMaterial(const Float3TextureRef &texCoeff, const FloatTextureRef &texF0, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<LambertianScatteringSurfaceMaterialHolder>(m_rawContext, texCoeff, texF0, nodeTexCoord);
        }

        UE4SurfaceMaterialRef createUE4SurfaceMaterial(const Float3TextureRef &texBaseColor, const Float3TextureRef &texOcclusionRoughnessMetallic, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<UE4SurfaceMaterialHolder>(m_rawContext, texBaseColor, texOcclusionRoughnessMetallic, nodeTexCoord);
        }

        DiffuseEmitterSurfaceMaterialRef createDiffuseEmitterSurfaceMaterial(const Float3TextureRef &texEmittance, const ShaderNodeRef &nodeTexCoord) const {
            return std::make_shared<DiffuseEmitterSurfaceMaterialHolder>(m_rawContext, texEmittance, nodeTexCoord);
        }

        MultiSurfaceMaterialRef createMultiSurfaceMaterial(const SurfaceMaterialRef* materials, uint32_t numMaterials) const {
            return std::make_shared<MultiSurfaceMaterialHolder>(m_rawContext, materials, numMaterials);
        }

        EnvironmentEmitterSurfaceMaterialRef createEnvironmentEmitterSurfaceMaterial(const Float3TextureRef &texEmittance) const {
            return std::make_shared<EnvironmentEmitterSurfaceMaterialHolder>(m_rawContext, texEmittance);
        }

        StaticTransformRef createStaticTransform(const float mat[16]) const {
            return std::make_shared<StaticTransformHolder>(m_rawContext, mat);
        }

        StaticTransformRef createStaticTransform(const VLR::Matrix4x4 &mat) const {
            return std::make_shared<StaticTransformHolder>(m_rawContext, mat);
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

        PerspectiveCameraRef createPerspectiveCamera(const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                                     float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) const {
            return std::make_shared<PerspectiveCameraHolder>(m_rawContext, position, orientation, sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist);
        }

        EquirectangularCameraRef createEquirectangularCamera(const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                                             float sensitivity, float phiAngle, float thetaAngle) const {
            return std::make_shared<EquirectangularCameraHolder>(m_rawContext, position, orientation, sensitivity, phiAngle, thetaAngle);
        }
    };
}
