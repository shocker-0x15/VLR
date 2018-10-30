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
    VLR_DECLARE_HOLDER_AND_REFERENCE(FloatShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float2ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float3ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float4ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(OffsetAndScaleUVTextureMap2DShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ConstantTextureShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Image2DTextureShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(EnvironmentTextureShaderNode);

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



    static inline void errorCheck(VLRResult errorCode) {
        if (errorCode != VLR_ERROR_NO_ERROR)
            throw std::runtime_error(vlrGetErrorMessage(errorCode));
    }



    class Object : public std::enable_shared_from_this<Object> {
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
            errorCheck(vlrLinearImage2DCreate(context, &m_raw, width, height, format, applyDegamma, linearData));
        }
        ~LinearImage2DHolder() {
            errorCheck(vlrLinearImage2DDestroy(m_rawContext, m_raw));
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        uint32_t getWidth() {
            uint32_t width;
            errorCheck(vlrLinearImage2DGetWidth(m_raw, &width));
            return width;
        }
        uint32_t getHeight() {
            uint32_t height;
            errorCheck(vlrLinearImage2DGetHeight(m_raw, &height));
            return height;
        }
        uint32_t getStride() {
            uint32_t stride;
            errorCheck(vlrLinearImage2DGetStride(m_raw, &stride));
            return stride;
        }
    };



    struct ShaderNodeSocket {
        ShaderNodeRef node;
        VLRShaderNodeSocketInfo socketInfo;

        ShaderNodeSocket() {}
        ShaderNodeSocket(const ShaderNodeRef &_node, const VLRShaderNodeSocketInfo &_socketInfo) :
            node(_node), socketInfo(_socketInfo) {}

        VLRShaderNode getNode() const;
    };



    class ShaderNodeHolder : public Object {
    protected:
        VLRShaderNode m_raw;

    public:
        ShaderNodeHolder(VLRContext context) : Object(context) {}

        ShaderNodeSocket getSocket(VLRShaderNodeSocketType socketType, uint32_t index) {
            VLRShaderNodeSocketInfo socketInfo;
            errorCheck(vlrShaderNodeGetSocket(m_raw, socketType, index, &socketInfo));
            return ShaderNodeSocket(std::dynamic_pointer_cast<ShaderNodeHolder>(shared_from_this()), socketInfo);
        }

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    VLRShaderNode ShaderNodeSocket::getNode() const {
        if (node)
            return (VLRShaderNode)node->get();
        else
            return nullptr;
    }



    class FloatShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_node0;

    public:
        FloatShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloatShaderNodeCreate(context, (VLRFloatShaderNode*)&m_raw));
        }
        ~FloatShaderNodeHolder() {
            errorCheck(vlrFloatShaderNodeDestroy(m_rawContext, (VLRFloatShaderNode)m_raw));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloatShaderNodeSetNode0((VLRFloatShaderNode)m_raw, m_node0.getNode(), m_node0.socketInfo));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloatShaderNodeSetImmediateValue0((VLRFloatShaderNode)m_raw, value));
        }
    };



    class Float2ShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_node0;
        ShaderNodeSocket m_node1;

    public:
        Float2ShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat2ShaderNodeCreate(context, (VLRFloat2ShaderNode*)&m_raw));
        }
        ~Float2ShaderNodeHolder() {
            errorCheck(vlrFloat2ShaderNodeDestroy(m_rawContext, (VLRFloat2ShaderNode)m_raw));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloat2ShaderNodeSetNode0((VLRFloat2ShaderNode)m_raw, m_node0.getNode(), m_node0.socketInfo));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloat2ShaderNodeSetImmediateValue0((VLRFloat2ShaderNode)m_raw, value));
        }
        void setNode1(const ShaderNodeSocket &node1) {
            m_node1 = node1;
            errorCheck(vlrFloat2ShaderNodeSetNode1((VLRFloat2ShaderNode)m_raw, m_node1.getNode(), m_node1.socketInfo));
        }
        void setImmediateValue1(float value) {
            errorCheck(vlrFloat2ShaderNodeSetImmediateValue1((VLRFloat2ShaderNode)m_raw, value));
        }
    };



    class Float3ShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_node0;
        ShaderNodeSocket m_node1;
        ShaderNodeSocket m_node2;

    public:
        Float3ShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat3ShaderNodeCreate(context, (VLRFloat3ShaderNode*)&m_raw));
        }
        ~Float3ShaderNodeHolder() {
            errorCheck(vlrFloat3ShaderNodeDestroy(m_rawContext, (VLRFloat3ShaderNode)m_raw));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloat3ShaderNodeSetNode0((VLRFloat3ShaderNode)m_raw, m_node0.getNode(), m_node0.socketInfo));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloat3ShaderNodeSetImmediateValue0((VLRFloat3ShaderNode)m_raw, value));
        }
        void setNode1(const ShaderNodeSocket &node1) {
            m_node1 = node1;
            errorCheck(vlrFloat3ShaderNodeSetNode1((VLRFloat3ShaderNode)m_raw, m_node1.getNode(), m_node1.socketInfo));
        }
        void setImmediateValue1(float value) {
            errorCheck(vlrFloat3ShaderNodeSetImmediateValue1((VLRFloat3ShaderNode)m_raw, value));
        }
        void setNode2(const ShaderNodeSocket &node2) {
            m_node2 = node2;
            errorCheck(vlrFloat3ShaderNodeSetNode2((VLRFloat3ShaderNode)m_raw, m_node2.getNode(), m_node2.socketInfo));
        }
        void setImmediateValue2(float value) {
            errorCheck(vlrFloat3ShaderNodeSetImmediateValue2((VLRFloat3ShaderNode)m_raw, value));
        }
    };



    class Float4ShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_node0;
        ShaderNodeSocket m_node1;
        ShaderNodeSocket m_node2;
        ShaderNodeSocket m_node3;

    public:
        Float4ShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat4ShaderNodeCreate(context, (VLRFloat4ShaderNode*)&m_raw));
        }
        ~Float4ShaderNodeHolder() {
            errorCheck(vlrFloat4ShaderNodeDestroy(m_rawContext, (VLRFloat4ShaderNode)m_raw));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloat4ShaderNodeSetNode0((VLRFloat4ShaderNode)m_raw, m_node0.getNode(), m_node0.socketInfo));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue0((VLRFloat4ShaderNode)m_raw, value));
        }
        void setNode1(const ShaderNodeSocket &node1) {
            m_node1 = node1;
            errorCheck(vlrFloat4ShaderNodeSetNode1((VLRFloat4ShaderNode)m_raw, m_node1.getNode(), m_node1.socketInfo));
        }
        void setImmediateValue1(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue1((VLRFloat4ShaderNode)m_raw, value));
        }
        void setNode2(const ShaderNodeSocket &node2) {
            m_node2 = node2;
            errorCheck(vlrFloat4ShaderNodeSetNode2((VLRFloat4ShaderNode)m_raw, m_node2.getNode(), m_node2.socketInfo));
        }
        void setImmediateValue2(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue2((VLRFloat4ShaderNode)m_raw, value));
        }
        void setNode3(const ShaderNodeSocket &node3) {
            m_node3 = node3;
            errorCheck(vlrFloat4ShaderNodeSetNode3((VLRFloat4ShaderNode)m_raw, m_node3.getNode(), m_node3.socketInfo));
        }
        void setImmediateValue3(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue3((VLRFloat4ShaderNode)m_raw, value));
        }
    };



    class OffsetAndScaleUVTextureMap2DShaderNodeHolder : public ShaderNodeHolder {
    public:
        OffsetAndScaleUVTextureMap2DShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrOffsetAndScaleUVTextureMap2DShaderNodeCreate(context, (VLROffsetAndScaleUVTextureMap2DShaderNode*)&m_raw));
        }
        ~OffsetAndScaleUVTextureMap2DShaderNodeHolder() {
            errorCheck(vlrOffsetAndScaleUVTextureMap2DShaderNodeDestroy(m_rawContext, (VLROffsetAndScaleUVTextureMap2DShaderNode)m_raw));
        }

        void setValues(const float offset[2], const float scale[2]) {
            errorCheck(vlrOffsetAndScaleUVTextureMap2DShaderNodeSetValues((VLROffsetAndScaleUVTextureMap2DShaderNode)m_raw, offset, scale));
        }
    };



    class ConstantTextureShaderNodeHolder : public ShaderNodeHolder {
    public:
        ConstantTextureShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrConstantTextureShaderNodeCreate(context, (VLRConstantTextureShaderNode*)&m_raw));
        }
        ~ConstantTextureShaderNodeHolder() {
            errorCheck(vlrConstantTextureShaderNodeDestroy(m_rawContext, (VLRConstantTextureShaderNode)m_raw));
        }

        void setValues(const VLR::RGBSpectrum &spectrum, float alpha) {
            errorCheck(vlrConstantTextureShaderNodeSetValues((VLRConstantTextureShaderNode)m_raw, (float*)&spectrum, alpha));
        }
    };



    class Image2DTextureShaderNodeHolder : public ShaderNodeHolder {
        Image2DRef m_image;
        ShaderNodeSocket m_nodeTexCoord;

    public:
        Image2DTextureShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrImage2DTextureShaderNodeCreate(context, (VLRImage2DTextureShaderNode*)&m_raw));
        }
        ~Image2DTextureShaderNodeHolder() {
            errorCheck(vlrImage2DTextureShaderNodeDestroy(m_rawContext, (VLRImage2DTextureShaderNode)m_raw));
        }

        void setImage(const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrImage2DTextureShaderNodeSetImage((VLRImage2DTextureShaderNode)m_raw, (VLRImage2D)m_image->get()));
        }
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            errorCheck(vlrImage2DTextureShaderNodeSetFilterMode((VLRImage2DTextureShaderNode)m_raw, minification, magnification, mipmapping));
        }
        void setNodeTexCoord(const ShaderNodeSocket &nodeTexCoord) {
            m_nodeTexCoord = nodeTexCoord;
            errorCheck(vlrImage2DTextureShaderNodeSetNodeTexCoord((VLRImage2DTextureShaderNode)m_raw, m_nodeTexCoord.getNode(), m_nodeTexCoord.socketInfo));
        }
    };



    class EnvironmentTextureShaderNodeHolder : public ShaderNodeHolder {
        Image2DRef m_image;
        ShaderNodeSocket m_nodeTexCoord;

    public:
        EnvironmentTextureShaderNodeHolder(VLRContext context) : ShaderNodeHolder(context) {
            errorCheck(vlrEnvironmentTextureShaderNodeCreate(context, (VLREnvironmentTextureShaderNode*)&m_raw));
        }
        ~EnvironmentTextureShaderNodeHolder() {
            errorCheck(vlrEnvironmentTextureShaderNodeDestroy(m_rawContext, (VLREnvironmentTextureShaderNode)m_raw));
        }

        void setImage(const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrEnvironmentTextureShaderNodeSetImage((VLREnvironmentTextureShaderNode)m_raw, (VLRImage2D)m_image->get()));
        }
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            errorCheck(vlrEnvironmentTextureShaderNodeSetFilterMode((VLREnvironmentTextureShaderNode)m_raw, minification, magnification, mipmapping));
        }
        bool setNodeTexCoord(const ShaderNodeSocket &nodeTexCoord) {
            m_nodeTexCoord = nodeTexCoord;
            errorCheck(vlrEnvironmentTextureShaderNodeSetNodeTexCoord((VLREnvironmentTextureShaderNode)m_raw, m_nodeTexCoord.getNode(), m_nodeTexCoord.socketInfo));
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
        ShaderNodeSocket m_nodeAlbedo;

    public:
        MatteSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMatteSurfaceMaterialCreate(context, (VLRMatteSurfaceMaterial*)&m_raw));
        }
        ~MatteSurfaceMaterialHolder() {
            errorCheck(vlrMatteSurfaceMaterialDestroy(m_rawContext, (VLRMatteSurfaceMaterial)m_raw));
        }

        void setNodeAlbedo(const ShaderNodeSocket &node) {
            m_nodeAlbedo = node;
            errorCheck(vlrMatteSurfaceMaterialSetNodeAlbedo((VLRMatteSurfaceMaterial)m_raw, m_nodeAlbedo.getNode(), m_nodeAlbedo.socketInfo));
        }
        void setImmediateValueAlbedo(const VLR::RGBSpectrum &value) {
            errorCheck(vlrMatteSurfaceMaterialSetImmediateValueAlbedo((VLRMatteSurfaceMaterial)m_raw, (float*)&value));
        }
    };



    class SpecularReflectionSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeffR;
        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;

    public:
        SpecularReflectionSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialCreate(context, (VLRSpecularReflectionSurfaceMaterial*)&m_raw));
        }
        ~SpecularReflectionSurfaceMaterialHolder() {
            errorCheck(vlrSpecularReflectionSurfaceMaterialDestroy(m_rawContext, (VLRSpecularReflectionSurfaceMaterial)m_raw));
        }

        void setNodeCoeffR(const ShaderNodeSocket &node) {
            m_nodeCoeffR = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR((VLRSpecularReflectionSurfaceMaterial)m_raw, m_nodeCoeffR.getNode(), m_nodeCoeffR.socketInfo));
        }
        void setImmediateValueCoeffR(const VLR::RGBSpectrum &value) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR((VLRSpecularReflectionSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeEta(const ShaderNodeSocket &node) {
            m_nodeEta = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNodeEta((VLRSpecularReflectionSurfaceMaterial)m_raw, m_nodeEta.getNode(), m_nodeEta.socketInfo));
        }
        void setImmediateValueEta(const VLR::RGBSpectrum &value) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta((VLRSpecularReflectionSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNode_k(const ShaderNodeSocket &node) {
            m_node_k = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNode_k((VLRSpecularReflectionSurfaceMaterial)m_raw, m_node_k.getNode(), m_node_k.socketInfo));
        }
        void setImmediateValue_k(const VLR::RGBSpectrum &value) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k((VLRSpecularReflectionSurfaceMaterial)m_raw, (float*)&value));
        }
    };



    class SpecularScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;

    public:
        SpecularScatteringSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialCreate(context, (VLRSpecularScatteringSurfaceMaterial*)&m_raw));
        }
        ~SpecularScatteringSurfaceMaterialHolder() {
            errorCheck(vlrSpecularScatteringSurfaceMaterialDestroy(m_rawContext, (VLRSpecularScatteringSurfaceMaterial)m_raw));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeCoeff((VLRSpecularScatteringSurfaceMaterial)m_raw, m_nodeCoeff.getNode(), m_nodeCoeff.socketInfo));
        }
        void setImmediateValueCoeff(const VLR::RGBSpectrum &value) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff((VLRSpecularScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeEtaExt(const ShaderNodeSocket &node) {
            m_nodeEtaExt = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt((VLRSpecularScatteringSurfaceMaterial)m_raw, m_nodeEtaExt.getNode(), m_nodeEtaExt.socketInfo));
        }
        void setImmediateValueEtaExt(const VLR::RGBSpectrum &value) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt((VLRSpecularScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeEtaInt(const ShaderNodeSocket &node) {
            m_nodeEtaInt = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt((VLRSpecularScatteringSurfaceMaterial)m_raw, m_nodeEtaInt.getNode(), m_nodeEtaInt.socketInfo));
        }
        void setImmediateValueEtaInt(const VLR::RGBSpectrum &value) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt((VLRSpecularScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
    };



    class MicrofacetReflectionSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;
        ShaderNodeSocket m_nodeRoughness;

    public:
        MicrofacetReflectionSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialCreate(context, (VLRMicrofacetReflectionSurfaceMaterial*)&m_raw));
        }
        ~MicrofacetReflectionSurfaceMaterialHolder() {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialDestroy(m_rawContext, (VLRMicrofacetReflectionSurfaceMaterial)m_raw));
        }

        void setNodeEta(const ShaderNodeSocket &node) {
            m_nodeEta = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNodeEta((VLRMicrofacetReflectionSurfaceMaterial)m_raw, m_nodeEta.getNode(), m_nodeEta.socketInfo));
        }
        void setImmediateValueEta(const VLR::RGBSpectrum &value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta((VLRMicrofacetReflectionSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNode_k(const ShaderNodeSocket &node) {
            m_node_k = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNode_k((VLRMicrofacetReflectionSurfaceMaterial)m_raw, m_node_k.getNode(), m_node_k.socketInfo));
        }
        void setImmediateValue_k(const VLR::RGBSpectrum &value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k((VLRMicrofacetReflectionSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeRoughness(const ShaderNodeSocket &node) {
            m_nodeRoughness = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughness((VLRMicrofacetReflectionSurfaceMaterial)m_raw, m_nodeRoughness.getNode(), m_nodeRoughness.socketInfo));
        }
        void setImmediateValueRoughness(const float value[2]) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness((VLRMicrofacetReflectionSurfaceMaterial)m_raw, value));
        }
    };



    class MicrofacetScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;
        ShaderNodeSocket m_nodeRoughness;

    public:
        MicrofacetScatteringSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialCreate(context, (VLRMicrofacetScatteringSurfaceMaterial*)&m_raw));
        }
        ~MicrofacetScatteringSurfaceMaterialHolder() {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialDestroy(m_rawContext, (VLRMicrofacetScatteringSurfaceMaterial)m_raw));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeCoeff.getNode(), m_nodeCoeff.socketInfo));
        }
        void setImmediateValueCoeff(const VLR::RGBSpectrum &value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff((VLRMicrofacetScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeEtaExt(const ShaderNodeSocket &node) {
            m_nodeEtaExt = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeEtaExt.getNode(), m_nodeEtaExt.socketInfo));
        }
        void setImmediateValueEtaExt(const VLR::RGBSpectrum &value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeEtaInt(const ShaderNodeSocket &node) {
            m_nodeEtaInt = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeEtaInt.getNode(), m_nodeEtaInt.socketInfo));
        }
        void setImmediateValueEtaInt(const VLR::RGBSpectrum &value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeRoughness(const ShaderNodeSocket &node) {
            m_nodeRoughness = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughness((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeRoughness.getNode(), m_nodeRoughness.socketInfo));
        }
        void setImmediateValueRoughness(const float value[2]) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness((VLRMicrofacetScatteringSurfaceMaterial)m_raw, value));
        }
    };



    class LambertianScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeF0;

    public:
        LambertianScatteringSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialCreate(context, (VLRLambertianScatteringSurfaceMaterial*)&m_raw));
        }
        ~LambertianScatteringSurfaceMaterialHolder() {
            errorCheck(vlrLambertianScatteringSurfaceMaterialDestroy(m_rawContext, (VLRLambertianScatteringSurfaceMaterial)m_raw));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetNodeCoeff((VLRLambertianScatteringSurfaceMaterial)m_raw, m_nodeCoeff.getNode(), m_nodeCoeff.socketInfo));
        }
        void setImmediateValueCoeff(const VLR::RGBSpectrum &value) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff((VLRLambertianScatteringSurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeF0(const ShaderNodeSocket &node) {
            m_nodeF0 = node;
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetNodeF0((VLRLambertianScatteringSurfaceMaterial)m_raw, m_nodeF0.getNode(), m_nodeF0.socketInfo));
        }
        void setImmediateValueF0(float value) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetImmediateValueF0((VLRLambertianScatteringSurfaceMaterial)m_raw, value));
        }
    };



    class UE4SurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeBaseColor;
        ShaderNodeSocket m_nodeOcclusionRoughnessMetallic;

    public:
        UE4SurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrUE4SurfaceMaterialCreate(context, (VLRUE4SurfaceMaterial*)&m_raw));
        }
        ~UE4SurfaceMaterialHolder() {
            errorCheck(vlrUE4SurfaceMaterialDestroy(m_rawContext, (VLRUE4SurfaceMaterial)m_raw));
        }

        void setNodeBaseColor(const ShaderNodeSocket &node) {
            m_nodeBaseColor = node;
            errorCheck(vlrUE4SufaceMaterialSetNodeBaseColor((VLRUE4SurfaceMaterial)m_raw, m_nodeBaseColor.getNode(), m_nodeBaseColor.socketInfo));
        }
        void setImmediateValueBaseColor(const VLR::RGBSpectrum &value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueBaseColor((VLRUE4SurfaceMaterial)m_raw, (float*)&value));
        }
        void setNodeOcclusionRoughnessMetallic(const ShaderNodeSocket &node) {
            m_nodeOcclusionRoughnessMetallic = node;
            errorCheck(vlrUE4SufaceMaterialSetNodeOcclusionRoughnessMetallic((VLRUE4SurfaceMaterial)m_raw, m_nodeOcclusionRoughnessMetallic.getNode(), m_nodeOcclusionRoughnessMetallic.socketInfo));
        }
        void setImmediateValueOcclusion(float value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueOcclusion((VLRUE4SurfaceMaterial)m_raw, value));
        }
        void setImmediateValueRoughness(float value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueRoughness((VLRUE4SurfaceMaterial)m_raw, value));
        }
        void setImmediateValueMetallic(float value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueMetallic((VLRUE4SurfaceMaterial)m_raw, value));
        }
    };



    class DiffuseEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEmittance;

    public:
        DiffuseEmitterSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialCreate(context, (VLRDiffuseEmitterSurfaceMaterial*)&m_raw));
        }
        ~DiffuseEmitterSurfaceMaterialHolder() {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialDestroy(m_rawContext, (VLRDiffuseEmitterSurfaceMaterial)m_raw));
        }

        void setNodeEmittance(const ShaderNodeSocket &node) {
            m_nodeEmittance = node;
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance((VLRDiffuseEmitterSurfaceMaterial)m_raw, m_nodeEmittance.getNode(), m_nodeEmittance.socketInfo));
        }
        void setImmediateValueEmittance(const VLR::RGBSpectrum &value) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance((VLRDiffuseEmitterSurfaceMaterial)m_raw, (float*)&value));
        }
    };



    class MultiSurfaceMaterialHolder : public SurfaceMaterialHolder {
        std::vector<SurfaceMaterialRef> m_materials;

    public:
        MultiSurfaceMaterialHolder(VLRContext context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMultiSurfaceMaterialCreate(context, (VLRMultiSurfaceMaterial*)&m_raw));
        }
        ~MultiSurfaceMaterialHolder() {
            errorCheck(vlrMultiSurfaceMaterialDestroy(m_rawContext, (VLRMultiSurfaceMaterial)m_raw));
        }

        void setSubMaterial(uint32_t index, VLRSurfaceMaterial mat) {
            errorCheck(vlrMultiSurfaceMaterialSetSubMaterial((VLRMultiSurfaceMaterial)m_raw, index, mat));
        }
    };



    class EnvironmentEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEmittance;

    public:
        EnvironmentEmitterSurfaceMaterialHolder(VLRContext context) :
            SurfaceMaterialHolder(context) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialCreate(context, (VLREnvironmentEmitterSurfaceMaterial*)&m_raw));
        }
        ~EnvironmentEmitterSurfaceMaterialHolder() {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialDestroy(m_rawContext, (VLREnvironmentEmitterSurfaceMaterial)m_raw));
        }

        void setNodeEmittance(EnvironmentTextureShaderNodeRef node) {
            m_nodeEmittance = node->getSocket(VLRShaderNodeSocketType_RGBSpectrum, 0);
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittance((VLREnvironmentEmitterSurfaceMaterial)m_raw, (VLREnvironmentTextureShaderNode)m_nodeEmittance.getNode()));
        }
        void setImmediateValueEmittance(const VLR::RGBSpectrum &value) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance((VLREnvironmentEmitterSurfaceMaterial)m_raw, (float*)&value));
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
            errorCheck(vlrStaticTransformCreate(m_rawContext, (VLRStaticTransform*)&m_raw, mat));
        }
        StaticTransformHolder(VLRContext context, const VLR::Matrix4x4 &mat) : TransformHolder(context) {
            float matArray[16];
            mat.getArray(matArray);
            errorCheck(vlrStaticTransformCreate(m_rawContext, (VLRStaticTransform*)&m_raw, matArray));
        }
        ~StaticTransformHolder() {
            errorCheck(vlrStaticTransformDestroy(m_rawContext, (VLRStaticTransform)m_raw));
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
        std::vector<ShaderNodeSocket> m_nodeNormals;
        std::vector<ShaderNodeSocket> m_nodeAlphas;

    public:
        TriangleMeshSurfaceNodeHolder(VLRContext context, const char* name) :
            SurfaceNodeHolder(context) {
            errorCheck(vlrTriangleMeshSurfaceNodeCreate(m_rawContext, &m_raw, name));
        }
        ~TriangleMeshSurfaceNodeHolder() {
            errorCheck(vlrTriangleMeshSurfaceNodeDestroy(m_rawContext, m_raw));
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        NodeType getNodeType() const override { return NodeType::TriangleMeshSurfaceNode; }
        void setName(const std::string &name) const override {
            errorCheck(vlrTriangleMeshSurfaceNodeSetName(m_raw, name.c_str()));
        }
        const char* getName() const override {
            const char* name;
            errorCheck(vlrTriangleMeshSurfaceNodeGetName(m_raw, &name));
            return name;
        }

        void setVertices(VLR::Vertex* vertices, uint32_t numVertices) {
            errorCheck(vlrTriangleMeshSurfaceNodeSetVertices(m_raw, (VLRVertex*)vertices, numVertices));
        }
        void addMaterialGroup(uint32_t* indices, uint32_t numIndices,
                              const SurfaceMaterialRef &material, 
                              const ShaderNodeSocket &nodeNormal,
                              const ShaderNodeSocket &nodeAlpha) {
            m_materials.push_back(material);
            m_nodeNormals.push_back(nodeNormal);
            m_nodeAlphas.push_back(nodeAlpha);
            errorCheck(vlrTriangleMeshSurfaceNodeAddMaterialGroup((VLRTriangleMeshSurfaceNode)m_raw, indices, numIndices,
                                                                  (VLRSurfaceMaterial)material->get(),
                                                                  nodeNormal.getNode(), nodeNormal.socketInfo,
                                                                  nodeAlpha.getNode(), nodeAlpha.socketInfo));
        }
    };



    class InternalNodeHolder : public NodeHolder {
        VLRInternalNode m_raw;
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;

    public:
        InternalNodeHolder(VLRContext context, const char* name, const StaticTransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            errorCheck(vlrInternalNodeCreate(m_rawContext, &m_raw, name, (VLRTransform)m_transform->get()));
        }
        ~InternalNodeHolder() {
            errorCheck(vlrInternalNodeDestroy(m_rawContext, m_raw));
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        NodeType getNodeType() const override { return NodeType::InternalNode; }
        void setName(const std::string &name) const override {
            errorCheck(vlrInternalNodeSetName(m_raw, name.c_str()));
        }
        const char* getName() const override {
            const char* name;
            errorCheck(vlrInternalNodeGetName(m_raw, &name));
            return name;
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrInternalNodeSetTransform(m_raw, (VLRTransform)transform->get()));
        }
        StaticTransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrInternalNodeAddChild(m_raw, child->get()));
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrInternalNodeRemoveChild(m_raw, child->get()));
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrInternalNodeAddChild(m_raw, child->get()));
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrInternalNodeRemoveChild(m_raw, child->get()));
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
            errorCheck(vlrSceneCreate(m_rawContext, &m_raw, (VLRTransform)m_transform->get()));
        }
        ~SceneHolder() {
            errorCheck(vlrSceneDestroy(m_rawContext, m_raw));
        }

        VLRObject get() const override { return (VLRObject)m_raw; }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrSceneSetTransform(m_raw, (VLRTransform)transform->get()));
        }
        StaticTransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrSceneAddChild(m_raw, child->get()));
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrSceneRemoveChild(m_raw, child->get()));
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrSceneAddChild(m_raw, child->get()));
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrSceneRemoveChild(m_raw, child->get()));
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
            errorCheck(vlrSceneSetEnvironment(m_raw, (VLREnvironmentEmitterSurfaceMaterial)m_matEnv->get()));
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
            errorCheck(vlrPerspectiveCameraDestroy(m_rawContext, (VLRPerspectiveCamera)m_raw));
        }

        void setPosition(const VLR::Point3D &position) {
            errorCheck(vlrPerspectiveCameraSetPosition((VLRPerspectiveCamera)m_raw, (VLRPoint3D*)&position));
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            errorCheck(vlrPerspectiveCameraSetOrientation((VLRPerspectiveCamera)m_raw, (VLRQuaternion*)&orientation));
        }
        void setSensitivity(float sensitivity) {
            errorCheck(vlrPerspectiveCameraSetSensitivity((VLRPerspectiveCamera)m_raw, sensitivity));
        }
        void setFovY(float fovY) {
            errorCheck(vlrPerspectiveCameraSetFovY((VLRPerspectiveCamera)m_raw, fovY));
        }
        void setLensRadius(float lensRadius) {
            errorCheck(vlrPerspectiveCameraSetLensRadius((VLRPerspectiveCamera)m_raw, lensRadius));
        }
        void setObjectPlaneDistance(float distance) {
            errorCheck(vlrPerspectiveCameraSetObjectPlaneDistance((VLRPerspectiveCamera)m_raw, distance));
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
            errorCheck(vlrEquirectangularCameraDestroy(m_rawContext, (VLREquirectangularCamera)m_raw));
        }

        void setPosition(const VLR::Point3D &position) {
            errorCheck(vlrEquirectangularCameraSetPosition((VLREquirectangularCamera)m_raw, (VLRPoint3D*)&position));
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            errorCheck(vlrEquirectangularCameraSetOrientation((VLREquirectangularCamera)m_raw, (VLRQuaternion*)&orientation));
        }
        void setSensitivity(float sensitivity) {
            errorCheck(vlrEquirectangularCameraSetSensitivity((VLREquirectangularCamera)m_raw, sensitivity));
        }
        void setAngles(float phiAngle, float thetaAngle) {
            errorCheck(vlrEquirectangularCameraSetAngles((VLREquirectangularCamera)m_raw, phiAngle, thetaAngle));
        }
    };



    class Context {
        VLRContext m_rawContext;

    public:
        Context(bool logging, uint32_t stackSize) {
            errorCheck(vlrCreateContext(&m_rawContext, logging, stackSize));
        }
        ~Context() {
            errorCheck(vlrDestroyContext(m_rawContext));
        }

        VLRContext get() const {
            return m_rawContext;
        }

        void setDevices(const int32_t* devices, uint32_t numDevices) const {
            errorCheck(vlrContextSetDevices(m_rawContext, devices, numDevices));
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID) const {
            errorCheck(vlrContextBindOutputBuffer(m_rawContext, width, height, glBufferID));
        }

        void* mapOutputBuffer() const {
            void* ptr = nullptr;
            errorCheck(vlrContextMapOutputBuffer(m_rawContext, &ptr));
            return ptr;
        }

        void unmapOutputBuffer() const {
            errorCheck(vlrContextUnmapOutputBuffer(m_rawContext));
        }

        void render(const SceneRef &scene, const CameraRef &camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            errorCheck(vlrContextRender(m_rawContext, (VLRScene)scene->get(), (VLRCamera)camera->get(), shrinkCoeff, firstFrame, numAccumFrames));
        }



        LinearImage2DRef createLinearImage2D(uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData) const {
            return std::make_shared<LinearImage2DHolder>(m_rawContext, width, height, format, applyDegamma, linearData);
        }



        FloatShaderNodeRef createFloatShaderNode() const {
            return std::make_shared<FloatShaderNodeHolder>(m_rawContext);
        }
        
        Float2ShaderNodeRef createFloat2ShaderNode() const {
            return std::make_shared<Float2ShaderNodeHolder>(m_rawContext);
        }
        
        Float3ShaderNodeRef createFloat3ShaderNode() const {
            return std::make_shared<Float3ShaderNodeHolder>(m_rawContext);
        }
        
        Float4ShaderNodeRef createFloat4ShaderNode() const {
            return std::make_shared<Float4ShaderNodeHolder>(m_rawContext);
        }

        OffsetAndScaleUVTextureMap2DShaderNodeRef createOffsetAndScaleUVTextureMap2DShaderNode() const {
            return std::make_shared<OffsetAndScaleUVTextureMap2DShaderNodeHolder>(m_rawContext);
        }

        ConstantTextureShaderNodeRef createConstantTextureShaderNode() const {
            return std::make_shared<ConstantTextureShaderNodeHolder>(m_rawContext);
        }

        Image2DTextureShaderNodeRef createImage2DTextureShaderNode() const {
            return std::make_shared<Image2DTextureShaderNodeHolder>(m_rawContext);
        }

        EnvironmentTextureShaderNodeRef createEnvironmentTextureShaderNode() const {
            return std::make_shared<EnvironmentTextureShaderNodeHolder>(m_rawContext);
        }



        MatteSurfaceMaterialRef createMatteSurfaceMaterial() const {
            return std::make_shared<MatteSurfaceMaterialHolder>(m_rawContext);
        }

        SpecularReflectionSurfaceMaterialRef createSpecularReflectionSurfaceMaterial() const {
            return std::make_shared<SpecularReflectionSurfaceMaterialHolder>(m_rawContext);
        }

        SpecularScatteringSurfaceMaterialRef createSpecularScatteringSurfaceMaterial() const {
            return std::make_shared<SpecularScatteringSurfaceMaterialHolder>(m_rawContext);
        }

        MicrofacetReflectionSurfaceMaterialRef createMicrofacetReflectionSurfaceMaterial() const {
            return std::make_shared<MicrofacetReflectionSurfaceMaterialHolder>(m_rawContext);
        }

        MicrofacetScatteringSurfaceMaterialRef createMicrofacetScatteringSurfaceMaterial() const {
            return std::make_shared<MicrofacetScatteringSurfaceMaterialHolder>(m_rawContext);
        }

        LambertianScatteringSurfaceMaterialRef createLambertianScatteringSurfaceMaterial() const {
            return std::make_shared<LambertianScatteringSurfaceMaterialHolder>(m_rawContext);
        }

        UE4SurfaceMaterialRef createUE4SurfaceMaterial() const {
            return std::make_shared<UE4SurfaceMaterialHolder>(m_rawContext);
        }

        DiffuseEmitterSurfaceMaterialRef createDiffuseEmitterSurfaceMaterial() const {
            return std::make_shared<DiffuseEmitterSurfaceMaterialHolder>(m_rawContext);
        }

        MultiSurfaceMaterialRef createMultiSurfaceMaterial() const {
            return std::make_shared<MultiSurfaceMaterialHolder>(m_rawContext);
        }

        EnvironmentEmitterSurfaceMaterialRef createEnvironmentEmitterSurfaceMaterial() const {
            return std::make_shared<EnvironmentEmitterSurfaceMaterialHolder>(m_rawContext);
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
