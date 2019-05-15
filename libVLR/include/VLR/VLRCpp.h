#pragma once

#include <algorithm>
#include <vector>
#include <map>
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
        if (errorCode != VLRResult_NoError)
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

        template <typename VLRType>
        VLRType get() const { return (VLRType)m_raw; }
    };



    class Image2DHolder : public Object {
    public:
        Image2DHolder(const ContextConstRef &context) : Object(context) {}

        uint32_t getWidth() const {
            uint32_t width;
            errorCheck(vlrImage2DGetWidth(get<VLRImage2D>(), &width));
            return width;
        }
        uint32_t getHeight() const {
            uint32_t height;
            errorCheck(vlrImage2DGetHeight(get<VLRImage2D>(), &height));
            return height;
        }
        uint32_t getStride() const {
            uint32_t stride;
            errorCheck(vlrImage2DGetStride(get<VLRImage2D>(), &stride));
            return stride;
        }
        VLRDataFormat getOriginalDataFormat() const {
            VLRDataFormat format;
            errorCheck(vlrImage2DGetOriginalDataFormat(get<VLRImage2D>(), &format));
            return format;
        }
        bool originalHasAlpha() const {
            bool hasAlpha;
            errorCheck(vlrImage2DOriginalHasAlpha(get<VLRImage2D>(), &hasAlpha));
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
            errorCheck(vlrLinearImage2DDestroy(getRaw(m_context), get<VLRLinearImage2D>()));
        }
    };



    class BlockCompressedImage2DHolder : public Image2DHolder {
    public:
        BlockCompressedImage2DHolder(const ContextConstRef &context, const uint8_t* const* data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height, VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) :
            Image2DHolder(context) {
            errorCheck(vlrBlockCompressedImage2DCreate(getRaw(m_context), (VLRBlockCompressedImage2D*)&m_raw, const_cast<uint8_t**>(data), const_cast<size_t*>(sizes), mipCount, width, height, dataFormat, spectrumType, colorSpace));
        }
        ~BlockCompressedImage2DHolder() {
            errorCheck(vlrBlockCompressedImage2DDestroy(getRaw(m_context), get<VLRBlockCompressedImage2D>()));
        }
    };



    struct ShaderNodeSocket {
        ShaderNodeRef node; // To retain reference count.
        VLRShaderNodeSocket socket;

        ShaderNodeSocket() {
            std::memset(&socket, 0, sizeof(socket));
        }
        ShaderNodeSocket(const ShaderNodeRef &_node, const VLRShaderNodeSocket &_socket) :
            node(_node), socket(_socket) {}
    };



    class ShaderNodeHolder : public Object {
    public:
        ShaderNodeHolder(const ContextConstRef &context) : Object(context) {}

        ShaderNodeSocket getSocket(VLRShaderNodeSocketType socketType, uint32_t option) {
            VLRShaderNodeSocket socket;
            errorCheck(vlrShaderNodeGetSocket(get<VLRShaderNode>(), socketType, option, &socket));
            return ShaderNodeSocket(std::dynamic_pointer_cast<ShaderNodeHolder>(shared_from_this()), socket);
        }
    };



    class GeometryShaderNodeHolder : public ShaderNodeHolder {
    public:
        GeometryShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrGeometryShaderNodeCreate(getRaw(m_context), (VLRGeometryShaderNode*)&m_raw));
        }
        ~GeometryShaderNodeHolder() {
            errorCheck(vlrGeometryShaderNodeDestroy(getRaw(m_context), get<VLRGeometryShaderNode>()));
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
            errorCheck(vlrFloat2ShaderNodeDestroy(getRaw(m_context), get<VLRFloat2ShaderNode>()));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloat2ShaderNodeSetNode0(get<VLRFloat2ShaderNode>(), m_node0.socket));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloat2ShaderNodeSetImmediateValue0(get<VLRFloat2ShaderNode>(), value));
        }
        void setNode1(const ShaderNodeSocket &node1) {
            m_node1 = node1;
            errorCheck(vlrFloat2ShaderNodeSetNode1(get<VLRFloat2ShaderNode>(), m_node1.socket));
        }
        void setImmediateValue1(float value) {
            errorCheck(vlrFloat2ShaderNodeSetImmediateValue1(get<VLRFloat2ShaderNode>(), value));
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
            errorCheck(vlrFloat3ShaderNodeDestroy(getRaw(m_context), get<VLRFloat3ShaderNode>()));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloat3ShaderNodeSetNode0(get<VLRFloat3ShaderNode>(), m_node0.socket));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloat3ShaderNodeSetImmediateValue0(get<VLRFloat3ShaderNode>(), value));
        }
        void setNode1(const ShaderNodeSocket &node1) {
            m_node1 = node1;
            errorCheck(vlrFloat3ShaderNodeSetNode1(get<VLRFloat3ShaderNode>(), m_node1.socket));
        }
        void setImmediateValue1(float value) {
            errorCheck(vlrFloat3ShaderNodeSetImmediateValue1(get<VLRFloat3ShaderNode>(), value));
        }
        void setNode2(const ShaderNodeSocket &node2) {
            m_node2 = node2;
            errorCheck(vlrFloat3ShaderNodeSetNode2(get<VLRFloat3ShaderNode>(), m_node2.socket));
        }
        void setImmediateValue2(float value) {
            errorCheck(vlrFloat3ShaderNodeSetImmediateValue2(get<VLRFloat3ShaderNode>(), value));
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
            errorCheck(vlrFloat4ShaderNodeDestroy(getRaw(m_context), get<VLRFloat4ShaderNode>()));
        }

        void setNode0(const ShaderNodeSocket &node0) {
            m_node0 = node0;
            errorCheck(vlrFloat4ShaderNodeSetNode0(get<VLRFloat4ShaderNode>(), m_node0.socket));
        }
        void setImmediateValue0(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue0(get<VLRFloat4ShaderNode>(), value));
        }
        void setNode1(const ShaderNodeSocket &node1) {
            m_node1 = node1;
            errorCheck(vlrFloat4ShaderNodeSetNode1(get<VLRFloat4ShaderNode>(), m_node1.socket));
        }
        void setImmediateValue1(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue1(get<VLRFloat4ShaderNode>(), value));
        }
        void setNode2(const ShaderNodeSocket &node2) {
            m_node2 = node2;
            errorCheck(vlrFloat4ShaderNodeSetNode2(get<VLRFloat4ShaderNode>(), m_node2.socket));
        }
        void setImmediateValue2(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue2(get<VLRFloat4ShaderNode>(), value));
        }
        void setNode3(const ShaderNodeSocket &node3) {
            m_node3 = node3;
            errorCheck(vlrFloat4ShaderNodeSetNode3(get<VLRFloat4ShaderNode>(), m_node3.socket));
        }
        void setImmediateValue3(float value) {
            errorCheck(vlrFloat4ShaderNodeSetImmediateValue3(get<VLRFloat4ShaderNode>(), value));
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
            errorCheck(vlrScaleAndOffsetFloatShaderNodeDestroy(getRaw(m_context), get<VLRScaleAndOffsetFloatShaderNode>()));
        }

        void setNodeValue(const ShaderNodeSocket &node) {
            m_nodeValue = node;
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetNodeValue(get<VLRScaleAndOffsetFloatShaderNode>(), m_nodeValue.socket));
        }
        void setNodeScale(const ShaderNodeSocket &node) {
            m_nodeScale = node;
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetNodeScale(get<VLRScaleAndOffsetFloatShaderNode>(), m_nodeScale.socket));
        }
        void setNodeOffset(const ShaderNodeSocket &node) {
            m_nodeOffset = node;
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetNodeOffset(get<VLRScaleAndOffsetFloatShaderNode>(), m_nodeOffset.socket));
        }
        void setImmediateValueScale(float value) {
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetImmediateValueScale(get<VLRScaleAndOffsetFloatShaderNode>(), value));
        }
        void setImmediateValueOffset(float value) {
            errorCheck(vlrScaleAndOffsetFloatShaderNodeSetImmediateValueOffset(get<VLRScaleAndOffsetFloatShaderNode>(), value));
        }
    };



    class TripletSpectrumShaderNodeHolder : public ShaderNodeHolder {
    public:
        TripletSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrTripletSpectrumShaderNodeCreate(getRaw(m_context), (VLRTripletSpectrumShaderNode*)&m_raw));
        }
        ~TripletSpectrumShaderNodeHolder() {
            errorCheck(vlrTripletSpectrumShaderNodeDestroy(getRaw(m_context), get<VLRTripletSpectrumShaderNode>()));
        }

        void setImmediateValueSpectrumType(VLRSpectrumType spectrumType) {
            errorCheck(vlrTripletSpectrumShaderNodeSetImmediateValueSpectrumType(get<VLRTripletSpectrumShaderNode>(), spectrumType));
        }
        void setImmediateValueColorSpace(VLRColorSpace colorSpace) {
            errorCheck(vlrTripletSpectrumShaderNodeSetImmediateValueColorSpace(get<VLRTripletSpectrumShaderNode>(), colorSpace));
        }
        void setImmediateValueTriplet(float e0, float e1, float e2) {
            errorCheck(vlrTripletSpectrumShaderNodeSetImmediateValueTriplet(get<VLRTripletSpectrumShaderNode>(), e0, e1, e2));
        }
    };



    class RegularSampledSpectrumShaderNodeHolder : public ShaderNodeHolder {
    public:
        RegularSampledSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrRegularSampledSpectrumShaderNodeCreate(getRaw(m_context), (VLRRegularSampledSpectrumShaderNode*)&m_raw));
        }
        ~RegularSampledSpectrumShaderNodeHolder() {
            errorCheck(vlrRegularSampledSpectrumShaderNodeDestroy(getRaw(m_context), get<VLRRegularSampledSpectrumShaderNode>()));
        }

        void setImmediateValueSpectrum(VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples) {
            errorCheck(vlrRegularSampledSpectrumShaderNodeSetImmediateValueSpectrum(get<VLRRegularSampledSpectrumShaderNode>(), spectrumType, minLambda, maxLambda, values, numSamples));
        }
    };



    class IrregularSampledSpectrumShaderNodeHolder : public ShaderNodeHolder {
    public:
        IrregularSampledSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrIrregularSampledSpectrumShaderNodeCreate(getRaw(m_context), (VLRIrregularSampledSpectrumShaderNode*)&m_raw));
        }
        ~IrregularSampledSpectrumShaderNodeHolder() {
            errorCheck(vlrIrregularSampledSpectrumShaderNodeDestroy(getRaw(m_context), get<VLRIrregularSampledSpectrumShaderNode>()));
        }

        void setImmediateValueSpectrum(VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples) {
            errorCheck(vlrIrregularSampledSpectrumShaderNodeSetImmediateValueSpectrum(get<VLRIrregularSampledSpectrumShaderNode>(), spectrumType, lambdas, values, numSamples));
        }
    };



    class Float3ToSpectrumShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_nodeFloat3;

    public:
        Float3ToSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrFloat3ToSpectrumShaderNodeCreate(getRaw(m_context), (VLRFloat3ToSpectrumShaderNode*)&m_raw));
        }
        ~Float3ToSpectrumShaderNodeHolder() {
            errorCheck(vlrFloat3ToSpectrumShaderNodeDestroy(getRaw(m_context), get<VLRFloat3ToSpectrumShaderNode>()));
        }

        void setNodeFloat3(const ShaderNodeSocket &nodeFloat3) {
            m_nodeFloat3 = nodeFloat3;
            errorCheck(vlrFloat3ToSpectrumShaderNodeSetNodeVector3D(get<VLRFloat3ToSpectrumShaderNode>(), m_nodeFloat3.socket));
        }
        void setImmediateValueFloat3(const float value[3]) {
            errorCheck(vlrFloat3ToSpectrumShaderNodeSetImmediateValueVector3D(get<VLRFloat3ToSpectrumShaderNode>(), value));
        }
        void setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
            errorCheck(vlrFloat3ToSpectrumShaderNodeSetImmediateValueSpectrumTypeAndColorSpace(get<VLRFloat3ToSpectrumShaderNode>(), spectrumType, colorSpace));
        }
    };



    class ScaleAndOffsetUVTextureMap2DShaderNodeHolder : public ShaderNodeHolder {
    public:
        ScaleAndOffsetUVTextureMap2DShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrScaleAndOffsetUVTextureMap2DShaderNodeCreate(getRaw(m_context), (VLRScaleAndOffsetUVTextureMap2DShaderNode*)&m_raw));
        }
        ~ScaleAndOffsetUVTextureMap2DShaderNodeHolder() {
            errorCheck(vlrScaleAndOffsetUVTextureMap2DShaderNodeDestroy(getRaw(m_context), get<VLRScaleAndOffsetUVTextureMap2DShaderNode>()));
        }

        void setValues(const float offset[2], const float scale[2]) {
            errorCheck(vlrScaleAndOffsetUVTextureMap2DShaderNodeSetValues(get<VLRScaleAndOffsetUVTextureMap2DShaderNode>(), offset, scale));
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
            errorCheck(vlrImage2DTextureShaderNodeDestroy(getRaw(m_context), get<VLRImage2DTextureShaderNode>()));
        }

        void setImage(const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrImage2DTextureShaderNodeSetImage(get<VLRImage2DTextureShaderNode>(), m_image ? m_image->get<VLRImage2D>() : nullptr));
        }
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            errorCheck(vlrImage2DTextureShaderNodeSetFilterMode(get<VLRImage2DTextureShaderNode>(), minification, magnification, mipmapping));
        }
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y) {
            errorCheck(vlrImage2DTextureShaderNodeSetWrapMode(get<VLRImage2DTextureShaderNode>(), x, y));
        }
        void setNodeTexCoord(const ShaderNodeSocket &nodeTexCoord) {
            m_nodeTexCoord = nodeTexCoord;
            errorCheck(vlrImage2DTextureShaderNodeSetNodeTexCoord(get<VLRImage2DTextureShaderNode>(), m_nodeTexCoord.socket));
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
            errorCheck(vlrEnvironmentTextureShaderNodeDestroy(getRaw(m_context), get<VLREnvironmentTextureShaderNode>()));
        }

        void setImage(const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrEnvironmentTextureShaderNodeSetImage(get<VLREnvironmentTextureShaderNode>(), m_image ? m_image->get<VLRImage2D>() : nullptr));
        }
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
            errorCheck(vlrEnvironmentTextureShaderNodeSetFilterMode(get<VLREnvironmentTextureShaderNode>(), minification, magnification, mipmapping));
        }
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y) {
            errorCheck(vlrEnvironmentTextureShaderNodeSetWrapMode(get<VLREnvironmentTextureShaderNode>(), x, y));
        }
        bool setNodeTexCoord(const ShaderNodeSocket &nodeTexCoord) {
            m_nodeTexCoord = nodeTexCoord;
            errorCheck(vlrEnvironmentTextureShaderNodeSetNodeTexCoord(get<VLREnvironmentTextureShaderNode>(), m_nodeTexCoord.socket));
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
            errorCheck(vlrMatteSurfaceMaterialDestroy(getRaw(m_context), get<VLRMatteSurfaceMaterial>()));
        }

        void setNodeAlbedo(const ShaderNodeSocket &node) {
            m_nodeAlbedo = node;
            errorCheck(vlrMatteSurfaceMaterialSetNodeAlbedo(get<VLRMatteSurfaceMaterial>(), m_nodeAlbedo.socket));
        }
        void setImmediateValueAlbedo(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMatteSurfaceMaterialSetImmediateValueAlbedo(get<VLRMatteSurfaceMaterial>(), colorSpace, e0, e1, e2));
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
            errorCheck(vlrSpecularReflectionSurfaceMaterialDestroy(getRaw(m_context), get<VLRSpecularReflectionSurfaceMaterial>()));
        }

        void setNodeCoeffR(const ShaderNodeSocket &node) {
            m_nodeCoeffR = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR(get<VLRSpecularReflectionSurfaceMaterial>(), m_nodeCoeffR.socket));
        }
        void setImmediateValueCoeffR(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR(get<VLRSpecularReflectionSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeEta(const ShaderNodeSocket &node) {
            m_nodeEta = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNodeEta(get<VLRSpecularReflectionSurfaceMaterial>(), m_nodeEta.socket));
        }
        void setImmediateValueEta(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta(get<VLRSpecularReflectionSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNode_k(const ShaderNodeSocket &node) {
            m_node_k = node;
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetNode_k(get<VLRSpecularReflectionSurfaceMaterial>(), m_node_k.socket));
        }
        void setImmediateValue_k(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k(get<VLRSpecularReflectionSurfaceMaterial>(), colorSpace, e0, e1, e2));
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
            errorCheck(vlrSpecularScatteringSurfaceMaterialDestroy(getRaw(m_context), get<VLRSpecularScatteringSurfaceMaterial>()));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeCoeff(get<VLRSpecularScatteringSurfaceMaterial>(), m_nodeCoeff.socket));
        }
        void setImmediateValueCoeff(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff(get<VLRSpecularScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeEtaExt(const ShaderNodeSocket &node) {
            m_nodeEtaExt = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt(get<VLRSpecularScatteringSurfaceMaterial>(), m_nodeEtaExt.socket));
        }
        void setImmediateValueEtaExt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt(get<VLRSpecularScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeEtaInt(const ShaderNodeSocket &node) {
            m_nodeEtaInt = node;
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt(get<VLRSpecularScatteringSurfaceMaterial>(), m_nodeEtaInt.socket));
        }
        void setImmediateValueEtaInt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt(get<VLRSpecularScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
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
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialDestroy(getRaw(m_context), get<VLRMicrofacetReflectionSurfaceMaterial>()));
        }

        void setNodeEta(const ShaderNodeSocket &node) {
            m_nodeEta = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNodeEta(get<VLRMicrofacetReflectionSurfaceMaterial>(), m_nodeEta.socket));
        }
        void setImmediateValueEta(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta(get<VLRMicrofacetReflectionSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNode_k(const ShaderNodeSocket &node) {
            m_node_k = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNode_k(get<VLRMicrofacetReflectionSurfaceMaterial>(), m_node_k.socket));
        }
        void setImmediateValue_k(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k(get<VLRMicrofacetReflectionSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeRoughnessAnisotropyRotation(const ShaderNodeSocket &node) {
            m_nodeRoughnessAnisotropyRotation = node;
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughnessAnisotropyRotation(get<VLRMicrofacetReflectionSurfaceMaterial>(), m_nodeRoughnessAnisotropyRotation.socket));
        }
        void setImmediateValueRoughness(float value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness(get<VLRMicrofacetReflectionSurfaceMaterial>(), value));
        }
        void setImmediateValueAnisotropy(float value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueAnisotropy(get<VLRMicrofacetReflectionSurfaceMaterial>(), value));
        }
        void setImmediateValueRotation(float value) {
            errorCheck(vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRotation(get<VLRMicrofacetReflectionSurfaceMaterial>(), value));
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
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialDestroy(getRaw(m_context), get<VLRMicrofacetScatteringSurfaceMaterial>()));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff(get<VLRMicrofacetScatteringSurfaceMaterial>(), m_nodeCoeff.socket));
        }
        void setImmediateValueCoeff(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff(get<VLRMicrofacetScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeEtaExt(const ShaderNodeSocket &node) {
            m_nodeEtaExt = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt(get<VLRMicrofacetScatteringSurfaceMaterial>(), m_nodeEtaExt.socket));
        }
        void setImmediateValueEtaExt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt(get<VLRMicrofacetScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeEtaInt(const ShaderNodeSocket &node) {
            m_nodeEtaInt = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt(get<VLRMicrofacetScatteringSurfaceMaterial>(), m_nodeEtaInt.socket));
        }
        void setImmediateValueEtaInt(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt(get<VLRMicrofacetScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeRoughnessAnisotropyRotation(const ShaderNodeSocket &node) {
            m_nodeRoughnessAnisotropyRotation = node;
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughnessAnisotropyRotation(get<VLRMicrofacetScatteringSurfaceMaterial>(), m_nodeRoughnessAnisotropyRotation.socket));
        }
        void setImmediateValueRoughness(float value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness(get<VLRMicrofacetScatteringSurfaceMaterial>(), value));
        }
        void setImmediateValueAnisotropy(float value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueAnisotropy(get<VLRMicrofacetScatteringSurfaceMaterial>(), value));
        }
        void setImmediateValueRotation(float value) {
            errorCheck(vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRotation(get<VLRMicrofacetScatteringSurfaceMaterial>(), value));
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
            errorCheck(vlrLambertianScatteringSurfaceMaterialDestroy(getRaw(m_context), get<VLRLambertianScatteringSurfaceMaterial>()));
        }

        void setNodeCoeff(const ShaderNodeSocket &node) {
            m_nodeCoeff = node;
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetNodeCoeff(get<VLRLambertianScatteringSurfaceMaterial>(), m_nodeCoeff.socket));
        }
        void setImmediateValueCoeff(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff(get<VLRLambertianScatteringSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeF0(const ShaderNodeSocket &node) {
            m_nodeF0 = node;
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetNodeF0(get<VLRLambertianScatteringSurfaceMaterial>(), m_nodeF0.socket));
        }
        void setImmediateValueF0(float value) {
            errorCheck(vlrLambertianScatteringSurfaceMaterialSetImmediateValueF0(get<VLRLambertianScatteringSurfaceMaterial>(), value));
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
            errorCheck(vlrUE4SurfaceMaterialDestroy(getRaw(m_context), get<VLRUE4SurfaceMaterial>()));
        }

        void setNodeBaseColor(const ShaderNodeSocket &node) {
            m_nodeBaseColor = node;
            errorCheck(vlrUE4SufaceMaterialSetNodeBaseColor(get<VLRUE4SurfaceMaterial>(), m_nodeBaseColor.socket));
        }
        void setImmediateValueBaseColor(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueBaseColor(get<VLRUE4SurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeOcclusionRoughnessMetallic(const ShaderNodeSocket &node) {
            m_nodeOcclusionRoughnessMetallic = node;
            errorCheck(vlrUE4SufaceMaterialSetNodeOcclusionRoughnessMetallic(get<VLRUE4SurfaceMaterial>(), m_nodeOcclusionRoughnessMetallic.socket));
        }
        void setImmediateValueOcclusion(float value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueOcclusion(get<VLRUE4SurfaceMaterial>(), value));
        }
        void setImmediateValueRoughness(float value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueRoughness(get<VLRUE4SurfaceMaterial>(), value));
        }
        void setImmediateValueMetallic(float value) {
            errorCheck(vlrUE4SufaceMaterialSetImmediateValueMetallic(get<VLRUE4SurfaceMaterial>(), value));
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
            errorCheck(vlrOldStyleSurfaceMaterialDestroy(getRaw(m_context), get<VLROldStyleSurfaceMaterial>()));
        }

        void setNodeDiffuseColor(const ShaderNodeSocket &node) {
            m_nodeDiffuseColor = node;
            errorCheck(vlrOldStyleSufaceMaterialSetNodeDiffuseColor(get<VLROldStyleSurfaceMaterial>(), m_nodeDiffuseColor.socket));
        }
        void setImmediateValueDiffuseColor(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrOldStyleSufaceMaterialSetImmediateValueDiffuseColor(get<VLROldStyleSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeSpecularColor(const ShaderNodeSocket &node) {
            m_nodeSpecularColor = node;
            errorCheck(vlrOldStyleSufaceMaterialSetNodeSpecularColor(get<VLROldStyleSurfaceMaterial>(), m_nodeSpecularColor.socket));
        }
        void setImmediateValueSpecularColor(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrOldStyleSufaceMaterialSetImmediateValueSpecularColor(get<VLROldStyleSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setNodeGlossiness(const ShaderNodeSocket &node) {
            m_nodeGlossiness = node;
            errorCheck(vlrOldStyleSufaceMaterialSetNodeGlossiness(get<VLROldStyleSurfaceMaterial>(), m_nodeSpecularColor.socket));
        }
        void setImmediateValueGlossiness(float value) {
            errorCheck(vlrOldStyleSufaceMaterialSetImmediateValueGlossiness(get<VLROldStyleSurfaceMaterial>(), value));
        }
    };



    class DiffuseEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        ShaderNodeSocket m_nodeEmittance;

    public:
        DiffuseEmitterSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialCreate(getRaw(m_context), (VLRDiffuseEmitterSurfaceMaterial*)&m_raw));
        }
        ~DiffuseEmitterSurfaceMaterialHolder() {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialDestroy(getRaw(m_context), get<VLRDiffuseEmitterSurfaceMaterial>()));
        }

        void setNodeEmittance(const ShaderNodeSocket &node) {
            m_nodeEmittance = node;
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance(get<VLRDiffuseEmitterSurfaceMaterial>(), m_nodeEmittance.socket));
        }
        void setImmediateValueEmittance(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance(get<VLRDiffuseEmitterSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setImmediateValueScale(float value) {
            errorCheck(vlrDiffuseEmitterSurfaceMaterialSetImmediateValueScale(get<VLRDiffuseEmitterSurfaceMaterial>(), value));
        }
    };



    class MultiSurfaceMaterialHolder : public SurfaceMaterialHolder {
        SurfaceMaterialRef m_materials[4];

    public:
        MultiSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMultiSurfaceMaterialCreate(getRaw(m_context), (VLRMultiSurfaceMaterial*)&m_raw));
        }
        ~MultiSurfaceMaterialHolder() {
            errorCheck(vlrMultiSurfaceMaterialDestroy(getRaw(m_context), get<VLRMultiSurfaceMaterial>()));
        }

        void setSubMaterial(uint32_t index, const SurfaceMaterialRef &mat) {
            if (!mat)
                return;
            m_materials[index] = mat;
            errorCheck(vlrMultiSurfaceMaterialSetSubMaterial(get<VLRMultiSurfaceMaterial>(), index, mat->get<VLRSurfaceMaterial>()));
        }
    };



    class EnvironmentEmitterSurfaceMaterialHolder : public SurfaceMaterialHolder {
        EnvironmentTextureShaderNodeRef m_nodeEmittanceTextured;
        ShaderNodeRef m_nodeEmittanceContant;

    public:
        EnvironmentEmitterSurfaceMaterialHolder(const ContextConstRef &context) :
            SurfaceMaterialHolder(context) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialCreate(getRaw(m_context), (VLREnvironmentEmitterSurfaceMaterial*)&m_raw));
        }
        ~EnvironmentEmitterSurfaceMaterialHolder() {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialDestroy(getRaw(m_context), get<VLREnvironmentEmitterSurfaceMaterial>()));
        }

        void setNodeEmittanceTextured(const EnvironmentTextureShaderNodeRef &node) {
            m_nodeEmittanceTextured = node;
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceTextured(get<VLREnvironmentEmitterSurfaceMaterial>(), m_nodeEmittanceTextured->get<VLREnvironmentTextureShaderNode>()));
        }
        void setNodeEmittanceConstant(const ShaderNodeRef &node) {
            m_nodeEmittanceContant = node;
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceConstant(get<VLREnvironmentEmitterSurfaceMaterial>(), m_nodeEmittanceContant->get<VLRShaderNode>()));
        }
        void setImmediateValueEmittance(VLRColorSpace colorSpace, float e0, float e1, float e2) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance(get<VLREnvironmentEmitterSurfaceMaterial>(), colorSpace, e0, e1, e2));
        }
        void setImmediateValueScale(float value) {
            errorCheck(vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueScale(get<VLREnvironmentEmitterSurfaceMaterial>(), value));
        }
    };



    class TransformHolder : public Object {
    public:
        TransformHolder(const ContextConstRef &context) : Object(context) {}

        VLRTransformType getTransformType() const {
            VLRTransformType type;
            errorCheck(vlrTransformGetType(get<VLRTransform>(), &type));
            return type;
        }
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
            errorCheck(vlrStaticTransformDestroy(getRaw(m_context), get<VLRStaticTransform>()));
        }

        void getArrays(float mat[16], float invMat[16]) const {
            errorCheck(vlrStaticTransformGetArrays(get<VLRStaticTransform>(), mat, invMat));
        }
        void getMatrices(VLR::Matrix4x4* mat, VLR::Matrix4x4* invMat) const {
            float aMat[16], aInvMat[16];
            getArrays(aMat, aInvMat);
            *mat = VLR::Matrix4x4(aMat);
            *invMat = VLR::Matrix4x4(aInvMat);
        }
    };



    class NodeHolder : public Object {
    public:
        NodeHolder(const ContextConstRef &context) : Object(context) {}

        VLRNodeType getNodeType() const {
            VLRNodeType type;
            errorCheck(vlrNodeGetType(get<VLRNode>(), &type));
            return type;
        }
        void setName(const std::string &name) const {
            errorCheck(vlrNodeSetName(get<VLRNode>(), name.c_str()));
        }
        const char* getName() const {
            const char* name;
            errorCheck(vlrNodeGetName(get<VLRNode>(), &name));
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
            errorCheck(vlrTriangleMeshSurfaceNodeDestroy(getRaw(m_context), get<VLRTriangleMeshSurfaceNode>()));
        }

        void setVertices(VLR::Vertex* vertices, uint32_t numVertices) {
            errorCheck(vlrTriangleMeshSurfaceNodeSetVertices(get<VLRTriangleMeshSurfaceNode>(), (VLRVertex*)vertices, numVertices));
        }
        void addMaterialGroup(uint32_t* indices, uint32_t numIndices,
                              const SurfaceMaterialRef &material,
                              const ShaderNodeSocket &nodeNormal, const ShaderNodeSocket &nodeAlpha,
                              VLRTangentType tangentType) {
            m_materials.push_back(material);
            m_nodeNormals.push_back(nodeNormal);
            m_nodeAlphas.push_back(nodeAlpha);
            errorCheck(vlrTriangleMeshSurfaceNodeAddMaterialGroup(get<VLRTriangleMeshSurfaceNode>(), indices, numIndices,
                                                                  material->get<VLRSurfaceMaterial>(), nodeNormal.socket, nodeAlpha.socket,
                                                                  tangentType));
        }
    };



    class InternalNodeHolder : public NodeHolder {
        TransformRef m_transform;
        std::map<VLRNode, NodeRef> m_children;

    public:
        InternalNodeHolder(const ContextConstRef &context, const char* name, const TransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            errorCheck(vlrInternalNodeCreate(getRaw(m_context), (VLRInternalNode*)&m_raw, name, m_transform->get<VLRTransform>()));
        }
        ~InternalNodeHolder() {
            errorCheck(vlrInternalNodeDestroy(getRaw(m_context), get<VLRInternalNode>()));
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrInternalNodeSetTransform(get<VLRInternalNode>(), transform->get<VLRTransform>()));
        }
        TransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrInternalNodeAddChild(get<VLRInternalNode>(), rawChild));
        }
        void removeChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrInternalNodeRemoveChild(get<VLRInternalNode>(), rawChild));
        }
        void addChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrInternalNodeAddChild(get<VLRInternalNode>(), rawChild));
        }
        void removeChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrInternalNodeRemoveChild(get<VLRInternalNode>(), rawChild));
        }
        uint32_t getNumChildren() const {
            uint32_t numChildren;
            errorCheck(vlrInternalNodeGetNumChildren(get<VLRInternalNode>(), &numChildren));
            return numChildren;
        }
        void getChildren(uint32_t numChildren, NodeRef* children) const {
            auto rawChildren = new VLRNode[numChildren];
            errorCheck(vlrInternalNodeGetChildren(get<VLRInternalNode>(), rawChildren));

            for (int i = 0; i < numChildren; ++i) {
                VLRNode rawChild = rawChildren[i];
                if (m_children.count(rawChild))
                    children[i] = m_children.at(rawChild);
            }

            delete[] rawChildren;
        }
        NodeRef getChildAt(uint32_t index) const {
            VLRNode rawChild;
            errorCheck(vlrInternalNodeGetChildAt(get<VLRInternalNode>(), index, &rawChild));

            if (m_children.count(rawChild))
                return m_children.at(rawChild);

            return nullptr;
        }
    };



    class SceneHolder : public Object {
        TransformRef m_transform;
        std::map<VLRNode, NodeRef> m_children;
        EnvironmentEmitterSurfaceMaterialRef m_matEnv;

    public:
        SceneHolder(const ContextConstRef &context, const TransformRef &transform) :
            Object(context), m_transform(transform) {
            errorCheck(vlrSceneCreate(getRaw(m_context), (VLRScene*)&m_raw, m_transform->get<VLRTransform>()));
        }
        ~SceneHolder() {
            errorCheck(vlrSceneDestroy(getRaw(m_context), get<VLRScene>()));
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrSceneSetTransform(get<VLRScene>(), transform->get<VLRTransform>()));
        }
        TransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrSceneAddChild(get<VLRScene>(), rawChild));
        }
        void removeChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrSceneRemoveChild(get<VLRScene>(), rawChild));
        }
        void addChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrSceneAddChild(get<VLRScene>(), rawChild));
        }
        void removeChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->get<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrSceneRemoveChild(get<VLRScene>(), rawChild));
        }
        uint32_t getNumChildren() const {
            uint32_t numChildren;
            errorCheck(vlrSceneGetNumChildren(get<VLRScene>(), &numChildren));
            return numChildren;
        }
        void getChildren(uint32_t numChildren, NodeRef* children) const {
            auto rawChildren = new VLRNode[numChildren];
            errorCheck(vlrSceneGetChildren(get<VLRScene>(), rawChildren));

            for (int i = 0; i < numChildren; ++i) {
                VLRNode rawChild = rawChildren[i];
                if (m_children.count(rawChild))
                    children[i] = m_children.at(rawChild);
            }

            delete[] rawChildren;
        }
        NodeRef getChildAt(uint32_t index) const {
            VLRNode rawChild;
            errorCheck(vlrSceneGetChildAt(get<VLRScene>(), index, &rawChild));

            if (m_children.count(rawChild))
                return m_children.at(rawChild);

            return nullptr;
        }

        void setEnvironment(const EnvironmentEmitterSurfaceMaterialRef &matEnv, float rotationPhi) {
            m_matEnv = matEnv;
            errorCheck(vlrSceneSetEnvironment(get<VLRScene>(), m_matEnv->get<VLREnvironmentEmitterSurfaceMaterial>()));
            errorCheck(vlrSceneSetEnvironmentRotation(get<VLRScene>(), rotationPhi));
        }
        void setEnvironmentRotation(float rotationPhi) {
            errorCheck(vlrSceneSetEnvironmentRotation(get<VLRScene>(), rotationPhi));
        }
    };



    class CameraHolder : public Object {
    public:
        CameraHolder(const ContextConstRef &context) : Object(context) {}

        VLRCameraType getCameraType() const {
            VLRCameraType type;
            errorCheck(vlrCameraGetType(get<VLRCamera>(), &type));
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
            errorCheck(vlrPerspectiveCameraDestroy(getRaw(m_context), get<VLRPerspectiveCamera>()));
        }

        void setPosition(const VLR::Point3D &position) {
            errorCheck(vlrPerspectiveCameraSetPosition(get<VLRPerspectiveCamera>(), (VLRPoint3D*)&position));
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            errorCheck(vlrPerspectiveCameraSetOrientation(get<VLRPerspectiveCamera>(), (VLRQuaternion*)&orientation));
        }
        void setAspectRatio(float aspect) {
            errorCheck(vlrPerspectiveCameraSetAspectRatio(get<VLRPerspectiveCamera>(), aspect));
        }
        void setSensitivity(float sensitivity) {
            errorCheck(vlrPerspectiveCameraSetSensitivity(get<VLRPerspectiveCamera>(), sensitivity));
        }
        void setFovY(float fovY) {
            errorCheck(vlrPerspectiveCameraSetFovY(get<VLRPerspectiveCamera>(), fovY));
        }
        void setLensRadius(float lensRadius) {
            errorCheck(vlrPerspectiveCameraSetLensRadius(get<VLRPerspectiveCamera>(), lensRadius));
        }
        void setObjectPlaneDistance(float distance) {
            errorCheck(vlrPerspectiveCameraSetObjectPlaneDistance(get<VLRPerspectiveCamera>(), distance));
        }

        void getPosition(VLR::Point3D* position) {
            errorCheck(vlrPerspectiveCameraGetPosition(get<VLRPerspectiveCamera>(), (VLRPoint3D*)position));
        }
        void getOrientation(VLR::Quaternion* orientation) {
            errorCheck(vlrPerspectiveCameraGetOrientation(get<VLRPerspectiveCamera>(), (VLRQuaternion*)orientation));
        }
        void getAspectRatio(float* aspect) {
            errorCheck(vlrPerspectiveCameraGetAspectRatio(get<VLRPerspectiveCamera>(), aspect));
        }
        void getSensitivity(float* sensitivity) {
            errorCheck(vlrPerspectiveCameraGetSensitivity(get<VLRPerspectiveCamera>(), sensitivity));
        }
        void getFovY(float* fovY) {
            errorCheck(vlrPerspectiveCameraGetFovY(get<VLRPerspectiveCamera>(), fovY));
        }
        void getLensRadius(float* lensRadius) {
            errorCheck(vlrPerspectiveCameraGetLensRadius(get<VLRPerspectiveCamera>(), lensRadius));
        }
        void getObjectPlaneDistance(float* distance) {
            errorCheck(vlrPerspectiveCameraGetObjectPlaneDistance(get<VLRPerspectiveCamera>(), distance));
        }
    };



    class EquirectangularCameraHolder : public CameraHolder {
    public:
        EquirectangularCameraHolder(const ContextConstRef &context) :
            CameraHolder(context) {
            errorCheck(vlrEquirectangularCameraCreate(getRaw(m_context), (VLREquirectangularCamera*)&m_raw));
        }
        ~EquirectangularCameraHolder() {
            errorCheck(vlrEquirectangularCameraDestroy(getRaw(m_context), get<VLREquirectangularCamera>()));
        }

        void setPosition(const VLR::Point3D &position) {
            errorCheck(vlrEquirectangularCameraSetPosition(get<VLREquirectangularCamera>(), (VLRPoint3D*)&position));
        }
        void setOrientation(const VLR::Quaternion &orientation) {
            errorCheck(vlrEquirectangularCameraSetOrientation(get<VLREquirectangularCamera>(), (VLRQuaternion*)&orientation));
        }
        void setSensitivity(float sensitivity) {
            errorCheck(vlrEquirectangularCameraSetSensitivity(get<VLREquirectangularCamera>(), sensitivity));
        }
        void setAngles(float phiAngle, float thetaAngle) {
            errorCheck(vlrEquirectangularCameraSetAngles(get<VLREquirectangularCamera>(), phiAngle, thetaAngle));
        }

        void getPosition(VLR::Point3D* position) {
            errorCheck(vlrEquirectangularCameraGetPosition(get<VLREquirectangularCamera>(), (VLRPoint3D*)position));
        }
        void getOrientation(VLR::Quaternion* orientation) {
            errorCheck(vlrEquirectangularCameraGetOrientation(get<VLREquirectangularCamera>(), (VLRQuaternion*)orientation));
        }
        void getSensitivity(float* sensitivity) {
            errorCheck(vlrEquirectangularCameraGetSensitivity(get<VLREquirectangularCamera>(), sensitivity));
        }
        void getAngles(float* phiAngle, float* thetaAngle) {
            errorCheck(vlrEquirectangularCameraGetAngles(get<VLREquirectangularCamera>(), phiAngle, thetaAngle));
        }
    };



    class Context : public std::enable_shared_from_this<Context> {
        VLRContext m_rawContext;
        GeometryShaderNodeRef m_geomShaderNode;
        StaticTransformRef m_identityTransform;

        Context() {}

        void initialize(bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize,
                        const int32_t* devices, uint32_t numDevices) {
            errorCheck(vlrCreateContext(&m_rawContext, logging, enableRTX, maxCallableDepth, stackSize, devices, numDevices));
            m_geomShaderNode = std::make_shared<GeometryShaderNodeHolder>(shared_from_this());
            m_identityTransform = std::make_shared<StaticTransformHolder>(shared_from_this(), VLR::Matrix4x4::Identity());
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

        const void* mapOutputBuffer() const {
            const void* ptr = nullptr;
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
            errorCheck(vlrContextRender(m_rawContext, scene->get<VLRScene>(), camera->get<VLRCamera>(), shrinkCoeff, firstFrame, numAccumFrames));
        }

        void debugRender(const SceneRef &scene, const CameraRef &camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            errorCheck(vlrContextDebugRender(m_rawContext, scene->get<VLRScene>(), camera->get<VLRCamera>(), renderMode, shrinkCoeff, firstFrame, numAccumFrames));
        }



        LinearImage2DRef createLinearImage2D(const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) const {
            return std::make_shared<LinearImage2DHolder>(shared_from_this(), linearData, width, height, format, spectrumType, colorSpace);
        }

        BlockCompressedImage2DRef createBlockCompressedImage2D(uint8_t** data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height, VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) const {
            return std::make_shared<BlockCompressedImage2DHolder>(shared_from_this(), data, sizes, mipCount, width, height, format, spectrumType, colorSpace);
        }



        const GeometryShaderNodeRef &createGeometryShaderNode() const {
            return m_geomShaderNode;
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

        const StaticTransformRef &getIdentityTransform() const {
            return m_identityTransform;
        }

        TriangleMeshSurfaceNodeRef createTriangleMeshSurfaceNode(const char* name) const {
            return std::make_shared<TriangleMeshSurfaceNodeHolder>(shared_from_this(), name);
        }

        InternalNodeRef createInternalNode(const char* name, const StaticTransformRef &transform = nullptr) const {
            return std::make_shared<InternalNodeHolder>(shared_from_this(), name, transform ? transform : getIdentityTransform());
        }

        SceneRef createScene(const StaticTransformRef &transform = nullptr) const {
            return std::make_shared<SceneHolder>(shared_from_this(), transform ? transform : getIdentityTransform());
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
