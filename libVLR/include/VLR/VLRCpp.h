#pragma once

#include <algorithm>
#include <vector>
#include <set>
#include <memory>

#include "VLR.h"
#include "basic_types.h"

namespace VLRCpp {
#define VLR_DECLARE_HOLDER_AND_REFERENCE(Name)\
    class Name ## Holder;\
    typedef std::shared_ptr<Name ## Holder> Name ## Ref

    VLR_DECLARE_HOLDER_AND_REFERENCE(Image2D);
    VLR_DECLARE_HOLDER_AND_REFERENCE(LinearImage2D);
    VLR_DECLARE_HOLDER_AND_REFERENCE(BlockCompressedImage2D);

    VLR_DECLARE_HOLDER_AND_REFERENCE(ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(GeometryShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(FloatShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float2ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float3ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float4ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ScaleAndOffsetFloatShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(TripletSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(RegularSampledSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(IrregularSampledSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float3ToSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(ScaleAndOffsetUVTextureMap2DShaderNode);
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
    VLR_DECLARE_HOLDER_AND_REFERENCE(OldStyleSurfaceMaterial);
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

    class Context;
    typedef std::shared_ptr<Context> ContextRef;
    typedef std::shared_ptr<const Context> ContextConstRef;



    static inline void errorCheck(VLRResult errorCode) {
        if (errorCode != VLR_ERROR_NO_ERROR)
            throw std::runtime_error(vlrGetErrorMessage(errorCode));
    }

    static inline VLRContext getRaw(const ContextConstRef &context);



    class Object : public std::enable_shared_from_this<Object> {
    protected:
        ContextConstRef m_context;
        VLRObject m_raw;

    public:
        Object(const ContextConstRef &context) : m_context(context) {}
        virtual ~Object() {}

        VLRObject get() const { return m_raw; }
    };



    class Image2DHolder : public Object {
    public:
        Image2DHolder(const ContextConstRef &context) : Object(context) {}

        uint32_t getWidth() const {
            uint32_t width;
            errorCheck(vlrImage2DGetWidth((VLRImage2D)m_raw, &width));
            return width;
        }
        uint32_t getHeight() const {
            uint32_t height;
            errorCheck(vlrImage2DGetHeight((VLRImage2D)m_raw, &height));
            return height;
        }
        uint32_t getStride() const {
            uint32_t stride;
            errorCheck(vlrImage2DGetStride((VLRImage2D)m_raw, &stride));
            return stride;
        }
        VLRDataFormat getOriginalDataFormat() const {
            VLRDataFormat format;
            errorCheck(vlrImage2DGetOriginalDataFormat((VLRImage2D)m_raw, &format));
            return format;
        }
        bool originalHasAlpha() const {
            bool hasAlpha;
            errorCheck(vlrImage2DOriginalHasAlpha((VLRImage2D)m_raw, &hasAlpha));
            return hasAlpha;
        }
    };



    class LinearImage2DHolder : public Image2DHolder {
    public:
        LinearImage2DHolder(const ContextConstRef &context, const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) :
            Image2DHolder(context) {
            errorCheck(vlrLinearImage2DCreate(getRaw(m_context), (VLRLinearImage2D*)&m_raw, const_cast<uint8_t*>(linearData), width, height, format, spectrumType, colorSpace));
        }
        ~LinearImage2DHolder() {
            errorCheck(vlrLinearImage2DDestroy(getRaw(m_context), (VLRLinearImage2D)m_raw));
        }
    };



    class BlockCompressedImage2DHolder : public Image2DHolder {
    public:
        BlockCompressedImage2DHolder(const ContextConstRef &context, const uint8_t* const* data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height, VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) :
            Image2DHolder(context) {
            errorCheck(vlrBlockCompressedImage2DCreate(getRaw(m_context), (VLRBlockCompressedImage2D*)&m_raw, const_cast<uint8_t**>(data), const_cast<size_t*>(sizes), mipCount, width, height, dataFormat, spectrumType, colorSpace));
        }
        ~BlockCompressedImage2DHolder() {
            errorCheck(vlrBlockCompressedImage2DDestroy(getRaw(m_context), (VLRBlockCompressedImage2D)m_raw));
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
    public:
        ShaderNodeHolder(const ContextConstRef &context) : Object(context) {}

        ShaderNodeSocket getSocket(VLRShaderNodeSocketType socketType, uint32_t option) {
            VLRShaderNodeSocketInfo socketInfo;
            errorCheck(vlrShaderNodeGetSocket((VLRShaderNode)m_raw, socketType, option, &socketInfo));
            return ShaderNodeSocket(std::dynamic_pointer_cast<ShaderNodeHolder>(shared_from_this()), socketInfo);
        }
    };



    inline VLRShaderNode ShaderNodeSocket::getNode() const {
        if (node)
            return (VLRShaderNode)node->get();
        else
            return nullptr;
    }



    class GeometryShaderNodeHolder : public ShaderNodeHolder {
    public:
        GeometryShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrGeometryShaderNodeCreate(getRaw(m_context), (VLRGeometryShaderNode*)&m_raw));
        }
        ~GeometryShaderNodeHolder() {
            errorCheck(vlrGeometryShaderNodeDestroy(getRaw(m_context), (VLRGeometryShaderNode)m_raw));
        }
    };



    class FloatShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_node0;

    public:
        FloatShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloatShaderNodeCreate(getRaw(m_context), (VLRFloatShaderNode*)&m_raw));
        }
        ~FloatShaderNodeHolder() {
            errorCheck(vlrFloatShaderNodeDestroy(getRaw(m_context), (VLRFloatShaderNode)m_raw));
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
        Float2ShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat2ShaderNodeCreate(getRaw(m_context), (VLRFloat2ShaderNode*)&m_raw));
        }
        ~Float2ShaderNodeHolder() {
            errorCheck(vlrFloat2ShaderNodeDestroy(getRaw(m_context), (VLRFloat2ShaderNode)m_raw));
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
        Float3ShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat3ShaderNodeCreate(getRaw(m_context), (VLRFloat3ShaderNode*)&m_raw));
        }
        ~Float3ShaderNodeHolder() {
            errorCheck(vlrFloat3ShaderNodeDestroy(getRaw(m_context), (VLRFloat3ShaderNode)m_raw));
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
        Float4ShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat4ShaderNodeCreate(getRaw(m_context), (VLRFloat4ShaderNode*)&m_raw));
        }
        ~Float4ShaderNodeHolder() {
            errorCheck(vlrFloat4ShaderNodeDestroy(getRaw(m_context), (VLRFloat4ShaderNode)m_raw));
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



    class ScaleAndOffsetFloatShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_nodeValue;
        ShaderNodeSocket m_nodeScale;
        ShaderNodeSocket m_nodeOffset;

    public:
        ScaleAndOffsetFloatShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrScaleAndOffsetFloatShaderNodeCreate(getRaw(m_context), (VLRScaleAndOffsetFloatShaderNode*)&m_raw));
        }
        ~ScaleAndOffsetFloatShaderNodeHolder() {
            errorCheck(vlrScaleAndOffsetFloatShaderNodeDestroy(getRaw(m_context), (VLRScaleAndOffsetFloatShaderNode)m_raw));
        }

        void setNodeValue(const ShaderNodeSocket &node) {
            m_nodeValue = node;
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetNodeValue((VLRScaleAndOffsetFloatShaderNode)m_raw, m_nodeValue.getNode(), m_nodeValue.socketInfo));
        }
        void setNodeScale(const ShaderNodeSocket &node) {
            m_nodeScale = node;
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetNodeScale((VLRScaleAndOffsetFloatShaderNode)m_raw, m_nodeScale.getNode(), m_nodeScale.socketInfo));
        }
        void setNodeOffset(const ShaderNodeSocket &node) {
            m_nodeOffset = node;
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetNodeOffset((VLRScaleAndOffsetFloatShaderNode)m_raw, m_nodeOffset.getNode(), m_nodeOffset.socketInfo));
        }
        void setImmediateValueScale(float value) {
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetImmediateValueScale((VLRScaleAndOffsetFloatShaderNode)m_raw, value));
        }
        void setImmediateValueOffset(float value) {
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetImmediateValueOffset((VLRScaleAndOffsetFloatShaderNode)m_raw, value));
        }
    };



    class TripletSpectrumShaderNodeHolder : public ShaderNodeHolder {
    public:
        TripletSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrTripletSpectrumShaderNodeCreate(getRaw(m_context), (VLRTripletSpectrumShaderNode*)&m_raw));
        }
        ~TripletSpectrumShaderNodeHolder() {
            errorCheck(vlrTripletSpectrumShaderNodeDestroy(getRaw(m_context), (VLRTripletSpectrumShaderNode)m_raw));
        }

        void setImmediateValueSpectrumType(VLRSpectrumType spectrumType) {
            errorCheck(vlrTripletSpectrumShaderNodeSetImmediateValueSpectrumType((VLRTripletSpectrumShaderNode)m_raw, spectrumType));
        }
        void setImmediateValueColorSpace(VLRColorSpace colorSpace) {
            errorCheck(vlrTripletSpectrumShaderNodeSetImmediateValueColorSpace((VLRTripletSpectrumShaderNode)m_raw, colorSpace));
        }
        void setImmediateValueTriplet(float e0, float e1, float e2) {
            errorCheck(vlrTripletSpectrumShaderNodeSetImmediateValueTriplet((VLRTripletSpectrumShaderNode)m_raw, e0, e1, e2));
        }
    };



    class RegularSampledSpectrumShaderNodeHolder : public ShaderNodeHolder {
    public:
        RegularSampledSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrRegularSampledSpectrumShaderNodeCreate(getRaw(m_context), (VLRRegularSampledSpectrumShaderNode*)&m_raw));
        }
        ~RegularSampledSpectrumShaderNodeHolder() {
            errorCheck(vlrRegularSampledSpectrumShaderNodeDestroy(getRaw(m_context), (VLRRegularSampledSpectrumShaderNode)m_raw));
        }

        void setImmediateValueSpectrum(VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples) {
            errorCheck(vlrRegularSampledSpectrumShaderNodeSetImmediateValueSpectrum((VLRRegularSampledSpectrumShaderNode)m_raw, spectrumType, minLambda, maxLambda, values, numSamples));
        }
    };



    class IrregularSampledSpectrumShaderNodeHolder : public ShaderNodeHolder {
    public:
        IrregularSampledSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrIrregularSampledSpectrumShaderNodeCreate(getRaw(m_context), (VLRIrregularSampledSpectrumShaderNode*)&m_raw));
        }
        ~IrregularSampledSpectrumShaderNodeHolder() {
            errorCheck(vlrIrregularSampledSpectrumShaderNodeDestroy(getRaw(m_context), (VLRIrregularSampledSpectrumShaderNode)m_raw));
        }

        void setImmediateValueSpectrum(VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples) {
            errorCheck(vlrIrregularSampledSpectrumShaderNodeSetImmediateValueSpectrum((VLRIrregularSampledSpectrumShaderNode)m_raw, spectrumType, lambdas, values, numSamples));
        }
    };



    class Float3ToSpectrumShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_nodeFloat3;

    public:
        Float3ToSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat3ToSpectrumShaderNodeCreate(getRaw(m_context), (VLRFloat3ToSpectrumShaderNode*)&m_raw));
        }
        ~Float3ToSpectrumShaderNodeHolder() {
            errorCheck(vlrFloat3ToSpectrumShaderNodeDestroy(getRaw(m_context), (VLRFloat3ToSpectrumShaderNode)m_raw));
        }

        void setNodeFloat3(const ShaderNodeSocket &nodeFloat3) {
            m_nodeFloat3 = nodeFloat3;
            errorCheck(vlrFloat3ToSpectrumShaderNodeSetNodeVector3D((VLRFloat3ToSpectrumShaderNode)m_raw, m_nodeFloat3.getNode(), m_nodeFloat3.socketInfo));
        }
        void setImmediateValueFloat3(const float value[3]) {
            errorCheck(vlrFloat3ToSpectrumShaderNodeSetImmediateValueVector3D((VLRFloat3ToSpectrumShaderNode)m_raw, value));
        }
        void setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
            errorCheck(vlrFloat3ToSpectrumShaderNodeSetImmediateValueSpectrumTypeAndColorSpace((VLRFloat3ToSpectrumShaderNode)m_raw, spectrumType, colorSpace));
        }
    };



    class ScaleAndOffsetUVTextureMap2DShaderNodeHolder : public ShaderNodeHolder {
    public:
        ScaleAndOffsetUVTextureMap2DShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrScaleAndOffsetUVTextureMap2DShaderNodeCreate(getRaw(m_context), (VLRScaleAndOffsetUVTextureMap2DShaderNode*)&m_raw));
        }
        ~ScaleAndOffsetUVTextureMap2DShaderNodeHolder() {
            errorCheck(vlrScaleAndOffsetUVTextureMap2DShaderNodeDestroy(getRaw(m_context), (VLRScaleAndOffsetUVTextureMap2DShaderNode)m_raw));
        }

        void setValues(const float offset[2], const float scale[2]) {
            errorCheck(vlrScaleAndOffsetUVTextureMap2DShaderNodeSetValues((VLRScaleAndOffsetUVTextureMap2DShaderNode)m_raw, offset, scale));
        }
    };



    class Image2DTextureShaderNodeHolder : public ShaderNodeHolder {
        Image2DRef m_image;
        ShaderNodeSocket m_nodeTexCoord;

    public:
        Image2DTextureShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrImage2DTextureShaderNodeCreate(getRaw(m_context), (VLRImage2DTextureShaderNode*)&m_raw));
        }
        ~Image2DTextureShaderNodeHolder() {
            errorCheck(vlrImage2DTextureShaderNodeDestroy(getRaw(m_context), (VLRImage2DTextureShaderNode)m_raw));
        }

        void setImage(const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrImage2DTextureShaderNodeSetImage((VLRImage2DTextureShaderNode)m_raw, m_image ? (VLRImage2D)m_image->get() : nullptr));
        }
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            errorCheck(vlrImage2DTextureShaderNodeSetFilterMode((VLRImage2DTextureShaderNode)m_raw, minification, magnification, mipmapping));
        }
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y) {
            errorCheck(vlrImage2DTextureShaderNodeSetWrapMode((VLRImage2DTextureShaderNode)m_raw, x, y));
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
        EnvironmentTextureShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrEnvironmentTextureShaderNodeCreate(getRaw(m_context), (VLREnvironmentTextureShaderNode*)&m_raw));
        }
        ~EnvironmentTextureShaderNodeHolder() {
            errorCheck(vlrEnvironmentTextureShaderNodeDestroy(getRaw(m_context), (VLREnvironmentTextureShaderNode)m_raw));
        }

        void setImage(const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrEnvironmentTextureShaderNodeSetImage((VLREnvironmentTextureShaderNode)m_raw, m_image ? (VLRImage2D)m_image->get() : nullptr));
        }
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            errorCheck(vlrEnvironmentTextureShaderNodeSetFilterMode((VLREnvironmentTextureShaderNode)m_raw, minification, magnification, mipmapping));
        }
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y) {
            errorCheck(vlrEnvironmentTextureShaderNodeSetWrapMode((VLREnvironmentTextureShaderNode)m_raw, x, y));
        }
        bool setNodeTexCoord(const ShaderNodeSocket &nodeTexCoord) {
            m_nodeTexCoord = nodeTexCoord;
            errorCheck(vlrEnvironmentTextureShaderNodeSetNodeTexCoord((VLREnvironmentTextureShaderNode)m_raw, m_nodeTexCoord.getNode(), m_nodeTexCoord.socketInfo));
        }
    };



    class SurfaceMaterialHolder : public Object {
    public:
        SurfaceMaterialHolder(const ContextConstRef &context) : Object(context) {}
    };



    class MatteSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeAlbedo;

    public:
        MatteSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMatteSurfaceMaterialCreate(getRaw(m_context), (VLRMatteSurfaceMaterial*)&m_raw));
        }
        ~MatteSurfaceMaterialHolder() {
            errorCheck(vlrMatteSurfaceMaterialDestroy(getRaw(m_context), (VLRMatteSurfaceMaterial)m_raw));
        }

        void setNodeAlbedo(const ShaderNodeSocket &node) {
            m_nodeAlbedo = node;
            errorCheck(vlrMatteSurfaceMaterialSetNodeAlbedo((VLRMatteSurfaceMaterial)m_raw, m_nodeAlbedo.getNode(), m_nodeAlbedo.socketInfo));
        }
        void setImmediateValueAlbedo(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMatteSurfaceMaterialSetImmediateValueAlbedo((VLRMatteSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
    };



    class SpecularReflectionSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeffR;
        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;

    public:
        SpecularReflectionSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialCreate(getRaw(m_context), (VLRSpecularReflectionSurfaceMaterial*)&m_raw));
        }
        ~SpecularReflectionSurfaceMaterialHolder() {
            errorCheck(vlrSpecularReflectionSurfaceMaterialDestroy(getRaw(m_context), (VLRSpecularReflectionSurfaceMaterial)m_raw));
        }

        void setNodeCoeffR(const ShaderNodeSocket &node) {
            m_nodeCoeffR = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR((VLRSpecularReflectionSurfaceMaterial)m_raw, m_nodeCoeffR.getNode(), m_nodeCoeffR.socketInfo));
        }
        void setImmediateValueCoeffR(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR((VLRSpecularReflectionSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeEta(const ShaderNodeSocket &node) {
            m_nodeEta = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNodeEta((VLRSpecularReflectionSurfaceMaterial)m_raw, m_nodeEta.getNode(), m_nodeEta.socketInfo));
        }
        void setImmediateValueEta(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta((VLRSpecularReflectionSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNode_k(const ShaderNodeSocket &node) {
            m_node_k = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNode_k((VLRSpecularReflectionSurfaceMaterial)m_raw, m_node_k.getNode(), m_node_k.socketInfo));
        }
        void setImmediateValue_k(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k((VLRSpecularReflectionSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
    };



    class SpecularScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;

    public:
        SpecularScatteringSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialCreate(getRaw(m_context), (VLRSpecularScatteringSurfaceMaterial*)&m_raw));
        }
        ~SpecularScatteringSurfaceMaterialHolder() {
            errorCheck(vlrSpecularScatteringSurfaceMaterialDestroy(getRaw(m_context), (VLRSpecularScatteringSurfaceMaterial)m_raw));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeCoeff((VLRSpecularScatteringSurfaceMaterial)m_raw, m_nodeCoeff.getNode(), m_nodeCoeff.socketInfo));
        }
        void setImmediateValueCoeff(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff((VLRSpecularScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeEtaExt(const ShaderNodeSocket &node) {
            m_nodeEtaExt = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt((VLRSpecularScatteringSurfaceMaterial)m_raw, m_nodeEtaExt.getNode(), m_nodeEtaExt.socketInfo));
        }
        void setImmediateValueEtaExt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt((VLRSpecularScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeEtaInt(const ShaderNodeSocket &node) {
            m_nodeEtaInt = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt((VLRSpecularScatteringSurfaceMaterial)m_raw, m_nodeEtaInt.getNode(), m_nodeEtaInt.socketInfo));
        }
        void setImmediateValueEtaInt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt((VLRSpecularScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
    };



    class MicrofacetReflectionSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;
        ShaderNodeSocket m_nodeRoughnessAnisotropyRotation;

    public:
        MicrofacetReflectionSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialCreate(getRaw(m_context), (VLRMicrofacetReflectionSurfaceMaterial*)&m_raw));
        }
        ~MicrofacetReflectionSurfaceMaterialHolder() {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialDestroy(getRaw(m_context), (VLRMicrofacetReflectionSurfaceMaterial)m_raw));
        }

        void setNodeEta(const ShaderNodeSocket &node) {
            m_nodeEta = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNodeEta((VLRMicrofacetReflectionSurfaceMaterial)m_raw, m_nodeEta.getNode(), m_nodeEta.socketInfo));
        }
        void setImmediateValueEta(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta((VLRMicrofacetReflectionSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNode_k(const ShaderNodeSocket &node) {
            m_node_k = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNode_k((VLRMicrofacetReflectionSurfaceMaterial)m_raw, m_node_k.getNode(), m_node_k.socketInfo));
        }
        void setImmediateValue_k(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k((VLRMicrofacetReflectionSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeRoughnessAnisotropyRotation(const ShaderNodeSocket &node) {
            m_nodeRoughnessAnisotropyRotation = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughnessAnisotropyRotation((VLRMicrofacetReflectionSurfaceMaterial)m_raw, m_nodeRoughnessAnisotropyRotation.getNode(), m_nodeRoughnessAnisotropyRotation.socketInfo));
        }
        void setImmediateValueRoughness(float value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness((VLRMicrofacetReflectionSurfaceMaterial)m_raw, value));
        }
        void setImmediateValueAnisotropy(float value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueAnisotropy((VLRMicrofacetReflectionSurfaceMaterial)m_raw, value));
        }
        void setImmediateValueRotation(float value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRotation((VLRMicrofacetReflectionSurfaceMaterial)m_raw, value));
        }
    };



    class MicrofacetScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;
        ShaderNodeSocket m_nodeRoughnessAnisotropyRotation;

    public:
        MicrofacetScatteringSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialCreate(getRaw(m_context), (VLRMicrofacetScatteringSurfaceMaterial*)&m_raw));
        }
        ~MicrofacetScatteringSurfaceMaterialHolder() {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialDestroy(getRaw(m_context), (VLRMicrofacetScatteringSurfaceMaterial)m_raw));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeCoeff.getNode(), m_nodeCoeff.socketInfo));
        }
        void setImmediateValueCoeff(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff((VLRMicrofacetScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeEtaExt(const ShaderNodeSocket &node) {
            m_nodeEtaExt = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeEtaExt.getNode(), m_nodeEtaExt.socketInfo));
        }
        void setImmediateValueEtaExt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeEtaInt(const ShaderNodeSocket &node) {
            m_nodeEtaInt = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeEtaInt.getNode(), m_nodeEtaInt.socketInfo));
        }
        void setImmediateValueEtaInt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt((VLRMicrofacetScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeRoughnessAnisotropyRotation(const ShaderNodeSocket &node) {
            m_nodeRoughnessAnisotropyRotation = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughnessAnisotropyRotation((VLRMicrofacetScatteringSurfaceMaterial)m_raw, m_nodeRoughnessAnisotropyRotation.getNode(), m_nodeRoughnessAnisotropyRotation.socketInfo));
        }
        void setImmediateValueRoughness(float value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness((VLRMicrofacetScatteringSurfaceMaterial)m_raw, value));
        }
        void setImmediateValueAnisotropy(float value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueAnisotropy((VLRMicrofacetScatteringSurfaceMaterial)m_raw, value));
        }
        void setImmediateValueRotation(float value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRotation((VLRMicrofacetScatteringSurfaceMaterial)m_raw, value));
        }
    };



    class LambertianScatteringSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeF0;

    public:
        LambertianScatteringSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialCreate(getRaw(m_context), (VLRLambertianScatteringSurfaceMaterial*)&m_raw));
        }
        ~LambertianScatteringSurfaceMaterialHolder() {
            errorCheck(vlrLambertianScatteringSurfaceMaterialDestroy(getRaw(m_context), (VLRLambertianScatteringSurfaceMaterial)m_raw));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetNodeCoeff((VLRLambertianScatteringSurfaceMaterial)m_raw, m_nodeCoeff.getNode(), m_nodeCoeff.socketInfo));
        }
        void setImmediateValueCoeff(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff((VLRLambertianScatteringSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
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
        UE4SurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrUE4SurfaceMaterialCreate(getRaw(m_context), (VLRUE4SurfaceMaterial*)&m_raw));
        }
        ~UE4SurfaceMaterialHolder() {
            errorCheck(vlrUE4SurfaceMaterialDestroy(getRaw(m_context), (VLRUE4SurfaceMaterial)m_raw));
        }

        void setNodeBaseColor(const ShaderNodeSocket &node) {
            m_nodeBaseColor = node;
            errorCheck(vlrUE4SufaceMaterialSetNodeBaseColor((VLRUE4SurfaceMaterial)m_raw, m_nodeBaseColor.getNode(), m_nodeBaseColor.socketInfo));
        }
        void setImmediateValueBaseColor(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueBaseColor((VLRUE4SurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
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



    class OldStyleSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeDiffuseColor;
        ShaderNodeSocket m_nodeSpecularColor;
        ShaderNodeSocket m_nodeGlossiness;

    public:
        OldStyleSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrOldStyleSurfaceMaterialCreate(getRaw(m_context), (VLROldStyleSurfaceMaterial*)&m_raw));
        }
        ~OldStyleSurfaceMaterialHolder() {
            errorCheck(vlrOldStyleSurfaceMaterialDestroy(getRaw(m_context), (VLROldStyleSurfaceMaterial)m_raw));
        }

        void setNodeDiffuseColor(const ShaderNodeSocket &node) {
            m_nodeDiffuseColor = node;
            errorCheck(vlrOldStyleSufaceMaterialSetNodeDiffuseColor((VLROldStyleSurfaceMaterial)m_raw, m_nodeDiffuseColor.getNode(), m_nodeDiffuseColor.socketInfo));
        }
        void setImmediateValueDiffuseColor(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrOldStyleSufaceMaterialSetImmediateValueDiffuseColor((VLROldStyleSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeSpecularColor(const ShaderNodeSocket &node) {
            m_nodeSpecularColor = node;
            errorCheck(vlrOldStyleSufaceMaterialSetNodeSpecularColor((VLROldStyleSurfaceMaterial)m_raw, m_nodeSpecularColor.getNode(), m_nodeSpecularColor.socketInfo));
        }
        void setImmediateValueSpecularColor(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrOldStyleSufaceMaterialSetImmediateValueSpecularColor((VLROldStyleSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setNodeGlossiness(const ShaderNodeSocket &node) {
            m_nodeGlossiness = node;
            errorCheck(vlrOldStyleSufaceMaterialSetNodeGlossiness((VLROldStyleSurfaceMaterial)m_raw, m_nodeSpecularColor.getNode(), m_nodeGlossiness.socketInfo));
        }
        void setImmediateValueGlossiness(float value) {
            errorCheck(vlrOldStyleSufaceMaterialSetImmediateValueGlossiness((VLROldStyleSurfaceMaterial)m_raw, value));
        }
    };



    class DiffuseEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEmittance;

    public:
        DiffuseEmitterSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialCreate(getRaw(m_context), (VLRDiffuseEmitterSurfaceMaterial*)&m_raw));
        }
        ~DiffuseEmitterSurfaceMaterialHolder() {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialDestroy(getRaw(m_context), (VLRDiffuseEmitterSurfaceMaterial)m_raw));
        }

        void setNodeEmittance(const ShaderNodeSocket &node) {
            m_nodeEmittance = node;
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance((VLRDiffuseEmitterSurfaceMaterial)m_raw, m_nodeEmittance.getNode(), m_nodeEmittance.socketInfo));
        }
        void setImmediateValueEmittance(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance((VLRDiffuseEmitterSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setImmediateValueScale(float value) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetImmediateValueScale((VLRDiffuseEmitterSurfaceMaterial)m_raw, value));
        }
    };



    class MultiSurfaceMaterialHolder : public SurfaceMaterialHolder {
        SurfaceMaterialRef m_materials[4];

    public:
        MultiSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMultiSurfaceMaterialCreate(getRaw(m_context), (VLRMultiSurfaceMaterial*)&m_raw));
        }
        ~MultiSurfaceMaterialHolder() {
            errorCheck(vlrMultiSurfaceMaterialDestroy(getRaw(m_context), (VLRMultiSurfaceMaterial)m_raw));
        }

        void setSubMaterial(uint32_t index, const SurfaceMaterialRef &mat) {
            if (!mat)
                return;
            m_materials[index] = mat;
            errorCheck(vlrMultiSurfaceMaterialSetSubMaterial((VLRMultiSurfaceMaterial)m_raw, index, (VLRSurfaceMaterial)mat->get()));
        }
    };



    class EnvironmentEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEmittanceTextured;
        ShaderNodeSocket m_nodeEmittanceContant;

    public:
        EnvironmentEmitterSurfaceMaterialHolder(const ContextConstRef &context) :
            SurfaceMaterialHolder(context) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialCreate(getRaw(m_context), (VLREnvironmentEmitterSurfaceMaterial*)&m_raw));
        }
        ~EnvironmentEmitterSurfaceMaterialHolder() {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialDestroy(getRaw(m_context), (VLREnvironmentEmitterSurfaceMaterial)m_raw));
        }

        void setNodeEmittanceTextured(EnvironmentTextureShaderNodeRef node) {
            m_nodeEmittanceTextured = node->getSocket(VLRShaderNodeSocketType_Spectrum, 0);
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceTextured((VLREnvironmentEmitterSurfaceMaterial)m_raw, (VLREnvironmentTextureShaderNode)m_nodeEmittanceTextured.getNode()));
        }
        void setNodeEmittanceConstant(ShaderNodeRef node) {
            m_nodeEmittanceContant = node->getSocket(VLRShaderNodeSocketType_Spectrum, 0);
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceConstant((VLREnvironmentEmitterSurfaceMaterial)m_raw, (VLRShaderNode)m_nodeEmittanceContant.getNode()));
        }
        void setImmediateValueEmittance(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance((VLREnvironmentEmitterSurfaceMaterial)m_raw, colorSpace, e0, e1, e2));
        }
        void setImmediateValueScale(float value) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueScale((VLREnvironmentEmitterSurfaceMaterial)m_raw, value));
        }
    };



    class TransformHolder : public Object {
    public:
        TransformHolder(const ContextConstRef &context) : Object(context) {}
    };



    class StaticTransformHolder : public TransformHolder {
    public:
        StaticTransformHolder(const ContextConstRef &context, const float mat[16]) : TransformHolder(context) {
            errorCheck(vlrStaticTransformCreate(getRaw(m_context), (VLRStaticTransform*)&m_raw, mat));
        }
        StaticTransformHolder(const ContextConstRef &context, const VLR::Matrix4x4 &mat) : TransformHolder(context) {
            float matArray[16];
            mat.getArray(matArray);
            errorCheck(vlrStaticTransformCreate(getRaw(m_context), (VLRStaticTransform*)&m_raw, matArray));
        }
        ~StaticTransformHolder() {
            errorCheck(vlrStaticTransformDestroy(getRaw(m_context), (VLRStaticTransform)m_raw));
        }
    };



    class NodeHolder : public Object {
    public:
        NodeHolder(const ContextConstRef &context) : Object(context) {}

        VLRNodeType getNodeType() const {
            VLRNodeType type;
            errorCheck(vlrNodeGetType((VLRNode)m_raw, &type));
            return type;
        }
        void setName(const std::string &name) const {
            errorCheck(vlrNodeSetName((VLRNode)m_raw, name.c_str()));
        }
        const char* getName() const {
            const char* name;
            errorCheck(vlrNodeGetName((VLRNode)m_raw, &name));
            return name;
        }
    };



    class SurfaceNodeHolder : public NodeHolder {
    public:
        SurfaceNodeHolder(const ContextConstRef &context) : NodeHolder(context) {}
    };



    class TriangleMeshSurfaceNodeHolder : public SurfaceNodeHolder {
        std::vector<SurfaceMaterialRef> m_materials;
        std::vector<ShaderNodeSocket> m_nodeNormals;
        std::vector<ShaderNodeSocket> m_nodeAlphas;

    public:
        TriangleMeshSurfaceNodeHolder(const ContextConstRef &context, const char* name) :
            SurfaceNodeHolder(context) {
            errorCheck(vlrTriangleMeshSurfaceNodeCreate(getRaw(m_context), (VLRTriangleMeshSurfaceNode*)&m_raw, name));
        }
        ~TriangleMeshSurfaceNodeHolder() {
            errorCheck(vlrTriangleMeshSurfaceNodeDestroy(getRaw(m_context), (VLRTriangleMeshSurfaceNode)m_raw));
        }

        void setVertices(VLR::Vertex* vertices, uint32_t numVertices) {
            errorCheck(vlrTriangleMeshSurfaceNodeSetVertices((VLRTriangleMeshSurfaceNode)m_raw, (VLRVertex*)vertices, numVertices));
        }
        void addMaterialGroup(uint32_t* indices, uint32_t numIndices,
                              const SurfaceMaterialRef &material,
                              const ShaderNodeSocket &nodeNormal, const ShaderNodeSocket &nodeAlpha,
                              VLRTangentType tangentType) {
            m_materials.push_back(material);
            m_nodeNormals.push_back(nodeNormal);
            m_nodeAlphas.push_back(nodeAlpha);
            errorCheck(vlrTriangleMeshSurfaceNodeAddMaterialGroup((VLRTriangleMeshSurfaceNode)m_raw, indices, numIndices,
                                                                  (VLRSurfaceMaterial)material->get(),
                                                                  nodeNormal.getNode(), nodeNormal.socketInfo,
                                                                  nodeAlpha.getNode(), nodeAlpha.socketInfo,
                                                                  tangentType));
        }
    };



    class InternalNodeHolder : public NodeHolder {
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;

    public:
        InternalNodeHolder(const ContextConstRef &context, const char* name, const StaticTransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            errorCheck(vlrInternalNodeCreate(getRaw(m_context), (VLRInternalNode*)&m_raw, name, (VLRTransform)m_transform->get()));
        }
        ~InternalNodeHolder() {
            errorCheck(vlrInternalNodeDestroy(getRaw(m_context), (VLRInternalNode)m_raw));
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrInternalNodeSetTransform((VLRInternalNode)m_raw, (VLRTransform)transform->get()));
        }
        StaticTransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrInternalNodeAddChild((VLRInternalNode)m_raw, child->get()));
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrInternalNodeRemoveChild((VLRInternalNode)m_raw, child->get()));
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrInternalNodeAddChild((VLRInternalNode)m_raw, child->get()));
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrInternalNodeRemoveChild((VLRInternalNode)m_raw, child->get()));
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
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;
        EnvironmentEmitterSurfaceMaterialRef m_matEnv;

    public:
        SceneHolder(const ContextConstRef &context, const StaticTransformRef &transform) :
            Object(context), m_transform(transform) {
            errorCheck(vlrSceneCreate(getRaw(m_context), (VLRScene*)&m_raw, (VLRTransform)m_transform->get()));
        }
        ~SceneHolder() {
            errorCheck(vlrSceneDestroy(getRaw(m_context), (VLRScene)m_raw));
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrSceneSetTransform((VLRScene)m_raw, (VLRTransform)transform->get()));
        }
        StaticTransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrSceneAddChild((VLRScene)m_raw, child->get()));
        }
        void removeChild(const InternalNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrSceneRemoveChild((VLRScene)m_raw, child->get()));
        }
        void addChild(const SurfaceNodeRef &child) {
            m_children.insert(child);
            errorCheck(vlrSceneAddChild((VLRScene)m_raw, child->get()));
        }
        void removeChild(const SurfaceNodeRef &child) {
            m_children.erase(child);
            errorCheck(vlrSceneRemoveChild((VLRScene)m_raw, child->get()));
        }
        uint32_t getNumChildren() const {
            return (uint32_t)m_children.size();
        }
        NodeRef getChildAt(uint32_t index) const {
            auto it = m_children.cbegin();
            std::advance(it, index);
            return *it;
        }

        void setEnvironment(const EnvironmentEmitterSurfaceMaterialRef &matEnv, float rotationPhi) {
            m_matEnv = matEnv;
            errorCheck(vlrSceneSetEnvironment((VLRScene)m_raw, (VLREnvironmentEmitterSurfaceMaterial)m_matEnv->get()));
            errorCheck(vlrSceneSetEnvironmentRotation((VLRScene)m_raw, rotationPhi));
        }
        void setEnvironmentRotation(float rotationPhi) {
            errorCheck(vlrSceneSetEnvironmentRotation((VLRScene)m_raw, rotationPhi));
        }
    };



    class CameraHolder : public Object {
    public:
        CameraHolder(const ContextConstRef &context) : Object(context) {}

        VLRCameraType getCameraType() const {
            VLRCameraType type;
            errorCheck(vlrCameraGetType((VLRCamera)m_raw, &type));
            return type;
        }
    };



    class PerspectiveCameraHolder : public CameraHolder {
    public:
        PerspectiveCameraHolder(const ContextConstRef &context) :
            CameraHolder(context) {
            errorCheck(vlrPerspectiveCameraCreate(getRaw(m_context), (VLRPerspectiveCamera*)&m_raw));
        }
        ~PerspectiveCameraHolder() {
            errorCheck(vlrPerspectiveCameraDestroy(getRaw(m_context), (VLRPerspectiveCamera)m_raw));
        }

        void setPosition(const VLR::Point3D &position) {
            errorCheck(vlrPerspectiveCameraSetPosition((VLRPerspectiveCamera)m_raw, (VLRPoint3D*)&position));
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            errorCheck(vlrPerspectiveCameraSetOrientation((VLRPerspectiveCamera)m_raw, (VLRQuaternion*)&orientation));
        }
        void setAspectRatio(float aspect) {
            errorCheck(vlrPerspectiveCameraSetAspectRatio((VLRPerspectiveCamera)m_raw, aspect));
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

        void getPosition(VLR::Point3D* position) {
            errorCheck(vlrPerspectiveCameraGetPosition((VLRPerspectiveCamera)m_raw, (VLRPoint3D*)position));
        }
        void getOrientation(VLR::Quaternion* orientation) {
            errorCheck(vlrPerspectiveCameraGetOrientation((VLRPerspectiveCamera)m_raw, (VLRQuaternion*)orientation));
        }
        void getAspectRatio(float* aspect) {
            errorCheck(vlrPerspectiveCameraGetAspectRatio((VLRPerspectiveCamera)m_raw, aspect));
        }
        void getSensitivity(float* sensitivity) {
            errorCheck(vlrPerspectiveCameraGetSensitivity((VLRPerspectiveCamera)m_raw, sensitivity));
        }
        void getFovY(float* fovY) {
            errorCheck(vlrPerspectiveCameraGetFovY((VLRPerspectiveCamera)m_raw, fovY));
        }
        void getLensRadius(float* lensRadius) {
            errorCheck(vlrPerspectiveCameraGetLensRadius((VLRPerspectiveCamera)m_raw, lensRadius));
        }
        void getObjectPlaneDistance(float* distance) {
            errorCheck(vlrPerspectiveCameraGetObjectPlaneDistance((VLRPerspectiveCamera)m_raw, distance));
        }
    };



    class EquirectangularCameraHolder : public CameraHolder {
    public:
        EquirectangularCameraHolder(const ContextConstRef &context) :
            CameraHolder(context) {
            errorCheck(vlrEquirectangularCameraCreate(getRaw(m_context), (VLREquirectangularCamera*)&m_raw));
        }
        ~EquirectangularCameraHolder() {
            errorCheck(vlrEquirectangularCameraDestroy(getRaw(m_context), (VLREquirectangularCamera)m_raw));
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

        void getPosition(VLR::Point3D* position) {
            errorCheck(vlrEquirectangularCameraGetPosition((VLREquirectangularCamera)m_raw, (VLRPoint3D*)position));
        }
        void getOrientation(VLR::Quaternion* orientation) {
            errorCheck(vlrEquirectangularCameraGetOrientation((VLREquirectangularCamera)m_raw, (VLRQuaternion*)orientation));
        }
        void getSensitivity(float* sensitivity) {
            errorCheck(vlrEquirectangularCameraGetSensitivity((VLREquirectangularCamera)m_raw, sensitivity));
        }
        void getAngles(float* phiAngle, float* thetaAngle) {
            errorCheck(vlrEquirectangularCameraGetAngles((VLREquirectangularCamera)m_raw, phiAngle, thetaAngle));
        }
    };



    class Context : public std::enable_shared_from_this<Context> {
        VLRContext m_rawContext;
        GeometryShaderNodeRef m_geomShaderNode;

        Context() {}

        void initialize(bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize,
                        const int32_t* devices, uint32_t numDevices) {
            errorCheck(vlrCreateContext(&m_rawContext, logging, enableRTX, maxCallableDepth, stackSize, devices, numDevices));
            m_geomShaderNode = std::make_shared<GeometryShaderNodeHolder>(shared_from_this());
        }

    public:
        static ContextRef create(bool logging, bool enableRTX = true, uint32_t maxCallableDepth = 8, uint32_t stackSize = 0,
                                 const int32_t* devices = nullptr, uint32_t numDevices = 0) {
            auto ret = std::shared_ptr<Context>(new Context());
            ret->initialize(logging, enableRTX, maxCallableDepth, stackSize, devices, numDevices);
            return ret;
        }

        ~Context() {
            errorCheck(vlrDestroyContext(m_rawContext));
        }

        VLRContext get() const {
            return m_rawContext;
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

        void getOutputBufferSize(uint32_t* width, uint32_t* height) const {
            errorCheck(vlrContextGetOutputBufferSize(m_rawContext, width, height));
        }

        void render(const SceneRef &scene, const CameraRef &camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            errorCheck(vlrContextRender(m_rawContext, (VLRScene)scene->get(), (VLRCamera)camera->get(), shrinkCoeff, firstFrame, numAccumFrames));
        }

        void debugRender(const SceneRef &scene, const CameraRef &camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            errorCheck(vlrContextDebugRender(m_rawContext, (VLRScene)scene->get(), (VLRCamera)camera->get(), renderMode, shrinkCoeff, firstFrame, numAccumFrames));
        }



        LinearImage2DRef createLinearImage2D(const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) const {
            return std::make_shared<LinearImage2DHolder>(shared_from_this(), linearData, width, height, format, spectrumType, colorSpace);
        }

        BlockCompressedImage2DRef createBlockCompressedImage2D(uint8_t** data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height, VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) const {
            return std::make_shared<BlockCompressedImage2DHolder>(shared_from_this(), data, sizes, mipCount, width, height, format, spectrumType, colorSpace);
        }



        GeometryShaderNodeRef createGeometryShaderNode() const {
            return m_geomShaderNode;
        }

        FloatShaderNodeRef createFloatShaderNode() const {
            return std::make_shared<FloatShaderNodeHolder>(shared_from_this());
        }

        Float2ShaderNodeRef createFloat2ShaderNode() const {
            return std::make_shared<Float2ShaderNodeHolder>(shared_from_this());
        }

        Float3ShaderNodeRef createFloat3ShaderNode() const {
            return std::make_shared<Float3ShaderNodeHolder>(shared_from_this());
        }

        Float4ShaderNodeRef createFloat4ShaderNode() const {
            return std::make_shared<Float4ShaderNodeHolder>(shared_from_this());
        }

        ScaleAndOffsetFloatShaderNodeRef createScaleAndOffsetFloatShaderNode() const {
            return std::make_shared<ScaleAndOffsetFloatShaderNodeHolder>(shared_from_this());
        }

        TripletSpectrumShaderNodeRef createTripletSpectrumShaderNode() const {
            return std::make_shared<TripletSpectrumShaderNodeHolder>(shared_from_this());
        }

        RegularSampledSpectrumShaderNodeRef createRegularSampledSpectrumShaderNode() const {
            return std::make_shared<RegularSampledSpectrumShaderNodeHolder>(shared_from_this());
        }

        IrregularSampledSpectrumShaderNodeRef createIrregularSampledSpectrumShaderNode() const {
            return std::make_shared<IrregularSampledSpectrumShaderNodeHolder>(shared_from_this());
        }

        Float3ToSpectrumShaderNodeRef createFloat3ToSpectrumShaderNode() const {
            return std::make_shared<Float3ToSpectrumShaderNodeHolder>(shared_from_this());
        }

        ScaleAndOffsetUVTextureMap2DShaderNodeRef createScaleAndOffsetUVTextureMap2DShaderNode() const {
            return std::make_shared<ScaleAndOffsetUVTextureMap2DShaderNodeHolder>(shared_from_this());
        }

        Image2DTextureShaderNodeRef createImage2DTextureShaderNode() const {
            return std::make_shared<Image2DTextureShaderNodeHolder>(shared_from_this());
        }

        EnvironmentTextureShaderNodeRef createEnvironmentTextureShaderNode() const {
            return std::make_shared<EnvironmentTextureShaderNodeHolder>(shared_from_this());
        }



        MatteSurfaceMaterialRef createMatteSurfaceMaterial() const {
            return std::make_shared<MatteSurfaceMaterialHolder>(shared_from_this());
        }

        SpecularReflectionSurfaceMaterialRef createSpecularReflectionSurfaceMaterial() const {
            return std::make_shared<SpecularReflectionSurfaceMaterialHolder>(shared_from_this());
        }

        SpecularScatteringSurfaceMaterialRef createSpecularScatteringSurfaceMaterial() const {
            return std::make_shared<SpecularScatteringSurfaceMaterialHolder>(shared_from_this());
        }

        MicrofacetReflectionSurfaceMaterialRef createMicrofacetReflectionSurfaceMaterial() const {
            return std::make_shared<MicrofacetReflectionSurfaceMaterialHolder>(shared_from_this());
        }

        MicrofacetScatteringSurfaceMaterialRef createMicrofacetScatteringSurfaceMaterial() const {
            return std::make_shared<MicrofacetScatteringSurfaceMaterialHolder>(shared_from_this());
        }

        LambertianScatteringSurfaceMaterialRef createLambertianScatteringSurfaceMaterial() const {
            return std::make_shared<LambertianScatteringSurfaceMaterialHolder>(shared_from_this());
        }

        UE4SurfaceMaterialRef createUE4SurfaceMaterial() const {
            return std::make_shared<UE4SurfaceMaterialHolder>(shared_from_this());
        }

        OldStyleSurfaceMaterialRef createOldStyleSurfaceMaterial() const {
            return std::make_shared<OldStyleSurfaceMaterialHolder>(shared_from_this());
        }

        DiffuseEmitterSurfaceMaterialRef createDiffuseEmitterSurfaceMaterial() const {
            return std::make_shared<DiffuseEmitterSurfaceMaterialHolder>(shared_from_this());
        }

        MultiSurfaceMaterialRef createMultiSurfaceMaterial() const {
            return std::make_shared<MultiSurfaceMaterialHolder>(shared_from_this());
        }

        EnvironmentEmitterSurfaceMaterialRef createEnvironmentEmitterSurfaceMaterial() const {
            return std::make_shared<EnvironmentEmitterSurfaceMaterialHolder>(shared_from_this());
        }



        StaticTransformRef createStaticTransform(const float mat[16]) const {
            return std::make_shared<StaticTransformHolder>(shared_from_this(), mat);
        }

        StaticTransformRef createStaticTransform(const VLR::Matrix4x4 &mat) const {
            return std::make_shared<StaticTransformHolder>(shared_from_this(), mat);
        }

        TriangleMeshSurfaceNodeRef createTriangleMeshSurfaceNode(const char* name) const {
            return std::make_shared<TriangleMeshSurfaceNodeHolder>(shared_from_this(), name);
        }

        InternalNodeRef createInternalNode(const char* name, const StaticTransformRef &transform) const {
            return std::make_shared<InternalNodeHolder>(shared_from_this(), name, transform);
        }

        SceneRef createScene(const StaticTransformRef &transform) const {
            return std::make_shared<SceneHolder>(shared_from_this(), transform);
        }

        PerspectiveCameraRef createPerspectiveCamera() const {
            return std::make_shared<PerspectiveCameraHolder>(shared_from_this());
        }

        EquirectangularCameraRef createEquirectangularCamera() const {
            return std::make_shared<EquirectangularCameraHolder>(shared_from_this());
        }
    };



    inline VLRContext getRaw(const ContextConstRef &context) {
        return context->get();
    }
}
