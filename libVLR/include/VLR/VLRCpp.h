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
    VLR_DECLARE_HOLDER_AND_REFERENCE(GeometryShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(FloatShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float2ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float3ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Float4ShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(TripletSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(RegularSampledSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(IrregularSampledSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(Vector3DToSpectrumShaderNode);
    VLR_DECLARE_HOLDER_AND_REFERENCE(OffsetAndScaleUVTextureMap2DShaderNode);
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

    public:
        Object(const ContextConstRef &context) : m_context(context) {}
        virtual ~Object() {}

        virtual VLRObject get() const = 0;
    };



    class Image2DHolder : public Object {
    public:
        Image2DHolder(const ContextConstRef &context) : Object(context) {}
    };



    class LinearImage2DHolder : public Image2DHolder {
        VLRLinearImage2D m_raw;

    public:
        LinearImage2DHolder(const ContextConstRef &context, uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData) :
            Image2DHolder(context) {
            errorCheck(vlrLinearImage2DCreate(getRaw(m_context), &m_raw, width, height, format, applyDegamma, linearData));
        }
        ~LinearImage2DHolder() {
            errorCheck(vlrLinearImage2DDestroy(getRaw(m_context), m_raw));
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
        ShaderNodeHolder(const ContextConstRef &context) : Object(context) {}

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



    class Vector3DToSpectrumShaderNodeHolder : public ShaderNodeHolder {
        ShaderNodeSocket m_nodeVector3D;

    public:
        Vector3DToSpectrumShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrVector3DToSpectrumShaderNodeCreate(getRaw(m_context), (VLRVector3DToSpectrumShaderNode*)&m_raw));
        }
        ~Vector3DToSpectrumShaderNodeHolder() {
            errorCheck(vlrVector3DToSpectrumShaderNodeDestroy(getRaw(m_context), (VLRVector3DToSpectrumShaderNode)m_raw));
        }

        void setNodeVector3D(const ShaderNodeSocket &nodeVector3D) {
            m_nodeVector3D = nodeVector3D;
            errorCheck(vlrVector3DToSpectrumShaderNodeSetNodeVector3D((VLRVector3DToSpectrumShaderNode)m_raw, m_nodeVector3D.getNode(), m_nodeVector3D.socketInfo));
        }
        void setImmediateValueVector3D(const VLR::Vector3D &value) {
            VLRVector3D v{ value.x, value.y, value.z };
            errorCheck(vlrVector3DToSpectrumShaderNodeSetImmediateValueVector3D((VLRVector3DToSpectrumShaderNode)m_raw, &v));
        }
        void setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
            errorCheck(vlrVector3DToSpectrumShaderNodeSetImmediateValueSpectrumTypeAndColorSpace((VLRVector3DToSpectrumShaderNode)m_raw, spectrumType, colorSpace));
        }
    };



    class OffsetAndScaleUVTextureMap2DShaderNodeHolder : public ShaderNodeHolder {
    public:
        OffsetAndScaleUVTextureMap2DShaderNodeHolder(const ContextConstRef &context) : ShaderNodeHolder(context) {
            errorCheck(vlrOffsetAndScaleUVTextureMap2DShaderNodeCreate(getRaw(m_context), (VLROffsetAndScaleUVTextureMap2DShaderNode*)&m_raw));
        }
        ~OffsetAndScaleUVTextureMap2DShaderNodeHolder() {
            errorCheck(vlrOffsetAndScaleUVTextureMap2DShaderNodeDestroy(getRaw(m_context), (VLROffsetAndScaleUVTextureMap2DShaderNode)m_raw));
        }

        void setValues(const float offset[2], const float scale[2]) {
            errorCheck(vlrOffsetAndScaleUVTextureMap2DShaderNodeSetValues((VLROffsetAndScaleUVTextureMap2DShaderNode)m_raw, offset, scale));
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

        void setImage(VLRSpectrumType spectrumType, VLRColorSpace colorSpace, const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrImage2DTextureShaderNodeSetImage((VLRImage2DTextureShaderNode)m_raw, spectrumType, colorSpace, m_image ? (VLRImage2D)m_image->get() : nullptr));
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

        void setImage(VLRColorSpace colorSpace, const Image2DRef &image) {
            m_image = image;
            errorCheck(vlrEnvironmentTextureShaderNodeSetImage((VLREnvironmentTextureShaderNode)m_raw, colorSpace, (VLRImage2D)m_image->get()));
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
    protected:
        VLRSurfaceMaterial m_raw;

    public:
        SurfaceMaterialHolder(const ContextConstRef &context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
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
    };



    class MultiSurfaceMaterialHolder : public SurfaceMaterialHolder {
        std::vector<SurfaceMaterialRef> m_materials;

    public:
        MultiSurfaceMaterialHolder(const ContextConstRef &context) : SurfaceMaterialHolder(context) {
            errorCheck(vlrMultiSurfaceMaterialCreate(getRaw(m_context), (VLRMultiSurfaceMaterial*)&m_raw));
        }
        ~MultiSurfaceMaterialHolder() {
            errorCheck(vlrMultiSurfaceMaterialDestroy(getRaw(m_context), (VLRMultiSurfaceMaterial)m_raw));
        }

        void setSubMaterial(uint32_t index, VLRSurfaceMaterial mat) {
            errorCheck(vlrMultiSurfaceMaterialSetSubMaterial((VLRMultiSurfaceMaterial)m_raw, index, mat));
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
    protected:
        VLRTransform m_raw;
    public:
        TransformHolder(const ContextConstRef &context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
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



    enum class NodeType {
        TriangleMeshSurfaceNode = 0,
        InternalNode,
    };

    class NodeHolder : public Object {
    public:
        NodeHolder(const ContextConstRef &context) : Object(context) {}

        virtual NodeType getNodeType() const = 0;
        virtual void setName(const std::string &name) const = 0;
        virtual const char* getName() const = 0;
    };



    class SurfaceNodeHolder : public NodeHolder {
    public:
        SurfaceNodeHolder(const ContextConstRef &context) : NodeHolder(context) {}
    };



    class TriangleMeshSurfaceNodeHolder : public SurfaceNodeHolder {
        VLRTriangleMeshSurfaceNode m_raw;
        std::vector<SurfaceMaterialRef> m_materials;
        std::vector<ShaderNodeSocket> m_nodeNormals;
        std::vector<ShaderNodeSocket> m_nodeAlphas;

    public:
        TriangleMeshSurfaceNodeHolder(const ContextConstRef &context, const char* name) :
            SurfaceNodeHolder(context) {
            errorCheck(vlrTriangleMeshSurfaceNodeCreate(getRaw(m_context), &m_raw, name));
        }
        ~TriangleMeshSurfaceNodeHolder() {
            errorCheck(vlrTriangleMeshSurfaceNodeDestroy(getRaw(m_context), m_raw));
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
        VLRInternalNode m_raw;
        StaticTransformRef m_transform;
        std::set<NodeRef> m_children;

    public:
        InternalNodeHolder(const ContextConstRef &context, const char* name, const StaticTransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            errorCheck(vlrInternalNodeCreate(getRaw(m_context), &m_raw, name, (VLRTransform)m_transform->get()));
        }
        ~InternalNodeHolder() {
            errorCheck(vlrInternalNodeDestroy(getRaw(m_context), m_raw));
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
        SceneHolder(const ContextConstRef &context, const StaticTransformRef &transform) :
            Object(context), m_transform(transform) {
            errorCheck(vlrSceneCreate(getRaw(m_context), &m_raw, (VLRTransform)m_transform->get()));
        }
        ~SceneHolder() {
            errorCheck(vlrSceneDestroy(getRaw(m_context), m_raw));
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
        CameraHolder(const ContextConstRef &context) : Object(context) {}

        VLRObject get() const override { return (VLRObject)m_raw; }
    };



    class PerspectiveCameraHolder : public CameraHolder {
    public:
        PerspectiveCameraHolder(const ContextConstRef &context, const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) :
            CameraHolder(context) {
            VLRResult res = vlrPerspectiveCameraCreate(getRaw(m_context), (VLRPerspectiveCamera*)&m_raw, (VLRPoint3D*)&position, (VLRQuaternion*)&orientation,
                                                       sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist);
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
        EquirectangularCameraHolder(const ContextConstRef &context, const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                    float sensitivity, float phiAngle, float thetaAngle) :
            CameraHolder(context) {
            VLRResult res = vlrEquirectangularCameraCreate(getRaw(m_context), (VLREquirectangularCamera*)&m_raw, (VLRPoint3D*)&position, (VLRQuaternion*)&orientation,
                                                           sensitivity, phiAngle, thetaAngle);
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
    };



    class Context : public std::enable_shared_from_this<Context> {
        VLRContext m_rawContext;
        GeometryShaderNodeRef m_geomShaderNode;

        Context() {}

        void initialize(bool logging, uint32_t stackSize, const int32_t* devices, uint32_t numDevices) {
            errorCheck(vlrCreateContext(&m_rawContext, logging, stackSize, devices, numDevices));
            m_geomShaderNode = std::make_shared<GeometryShaderNodeHolder>(shared_from_this());
        }

    public:
        static ContextRef create(bool logging, uint32_t stackSize, const int32_t* devices = nullptr, uint32_t numDevices = 0) {
            auto ret = std::shared_ptr<Context>(new Context());
            ret->initialize(logging, stackSize, devices, numDevices);
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



        LinearImage2DRef createLinearImage2D(uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData) const {
            return std::make_shared<LinearImage2DHolder>(shared_from_this(), width, height, format, applyDegamma, linearData);
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

        TripletSpectrumShaderNodeRef createTripletSpectrumShaderNode() const {
            return std::make_shared<TripletSpectrumShaderNodeHolder>(shared_from_this());
        }

        RegularSampledSpectrumShaderNodeRef createRegularSampledSpectrumShaderNode() const {
            return std::make_shared<RegularSampledSpectrumShaderNodeHolder>(shared_from_this());
        }

        IrregularSampledSpectrumShaderNodeRef createIrregularSampledSpectrumShaderNode() const {
            return std::make_shared<IrregularSampledSpectrumShaderNodeHolder>(shared_from_this());
        }

        Vector3DToSpectrumShaderNodeRef createVector3DToSpectrumShaderNode() const {
            return std::make_shared<Vector3DToSpectrumShaderNodeHolder>(shared_from_this());
        }

        OffsetAndScaleUVTextureMap2DShaderNodeRef createOffsetAndScaleUVTextureMap2DShaderNode() const {
            return std::make_shared<OffsetAndScaleUVTextureMap2DShaderNodeHolder>(shared_from_this());
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

        PerspectiveCameraRef createPerspectiveCamera(const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                                     float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) const {
            return std::make_shared<PerspectiveCameraHolder>(shared_from_this(), position, orientation, sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist);
        }

        EquirectangularCameraRef createEquirectangularCamera(const VLR::Point3D &position, const VLR::Quaternion &orientation,
                                                             float sensitivity, float phiAngle, float thetaAngle) const {
            return std::make_shared<EquirectangularCameraHolder>(shared_from_this(), position, orientation, sensitivity, phiAngle, thetaAngle);
        }
    };



    inline VLRContext getRaw(const ContextConstRef &context) {
        return context->get();
    }
}
