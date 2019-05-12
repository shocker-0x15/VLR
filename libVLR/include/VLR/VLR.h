#pragma once

#include "public_types.h"

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define VLR_API_Platform_Windows
#    if defined(_MSC_VER)
#        define VLR_API_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define VLR_API_Platform_macOS
#endif

#if defined(VLR_API_Platform_Windows_MSVC)
#   if defined(VLR_API_EXPORTS)
#       define VLR_API __declspec(dllexport)
#   else
#       define VLR_API __declspec(dllimport)
#   endif
#else
#   define VLR_API
#endif



#if defined(__cplusplus)
extern "C" {
#endif
    enum VLRResult {
        VLRResult_NoError = 0,
        VLRResult_InvalidContext,
        VLRResult_InvalidInstance,
        VLRResult_InvalidArgument,
        VLRResult_IncompatibleNodeType,
        VLRResult_InternalError,
    };

#if !defined(__cplusplus)
    typedef enum VLRResult VLRResult;
#endif

#if !defined(VLR_API_EXPORTS)
#   define VLR_DEFINE_OPAQUE_TYPE(name) typedef struct VLR ## name ## _API* VLR ## name

    VLR_DEFINE_OPAQUE_TYPE(Object);

    VLR_DEFINE_OPAQUE_TYPE(Context);

    VLR_DEFINE_OPAQUE_TYPE(Image2D);
    VLR_DEFINE_OPAQUE_TYPE(LinearImage2D);
    VLR_DEFINE_OPAQUE_TYPE(BlockCompressedImage2D);

    VLR_DEFINE_OPAQUE_TYPE(ShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(GeometryShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(Float2ShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(Float3ShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(Float4ShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(ScaleAndOffsetFloatShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(TripletSpectrumShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(RegularSampledSpectrumShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(IrregularSampledSpectrumShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(Float3ToSpectrumShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(ScaleAndOffsetUVTextureMap2DShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(Image2DTextureShaderNode);
    VLR_DEFINE_OPAQUE_TYPE(EnvironmentTextureShaderNode);

    VLR_DEFINE_OPAQUE_TYPE(SurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(MatteSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(SpecularReflectionSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(SpecularScatteringSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(MicrofacetReflectionSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(MicrofacetScatteringSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(LambertianScatteringSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(UE4SurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(OldStyleSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(DiffuseEmitterSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(MultiSurfaceMaterial);
    VLR_DEFINE_OPAQUE_TYPE(EnvironmentEmitterSurfaceMaterial);

    VLR_DEFINE_OPAQUE_TYPE(Transform);
    typedef const struct VLRTransform_API* VLRTransformConst;
    VLR_DEFINE_OPAQUE_TYPE(StaticTransform);

    VLR_DEFINE_OPAQUE_TYPE(Node);
    VLR_DEFINE_OPAQUE_TYPE(SurfaceNode);
    VLR_DEFINE_OPAQUE_TYPE(TriangleMeshSurfaceNode);
    VLR_DEFINE_OPAQUE_TYPE(InternalNode);
    VLR_DEFINE_OPAQUE_TYPE(Scene);

    VLR_DEFINE_OPAQUE_TYPE(Camera);
    VLR_DEFINE_OPAQUE_TYPE(PerspectiveCamera);
    VLR_DEFINE_OPAQUE_TYPE(EquirectangularCamera);
#endif



    VLR_API VLRResult vlrPrintDevices();
    VLR_API VLRResult vlrGetDeviceName(uint32_t index, char* name, uint32_t bufferLength);

    VLR_API const char* vlrGetErrorMessage(VLRResult code);

    VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices);
    VLR_API VLRResult vlrDestroyContext(VLRContext context);

    VLR_API VLRResult vlrContextBindOutputBuffer(VLRContext context, uint32_t width, uint32_t height, uint32_t bufferID);
    VLR_API VLRResult vlrContextMapOutputBuffer(VLRContext context, void** ptr);
    VLR_API VLRResult vlrContextUnmapOutputBuffer(VLRContext context);
    VLR_API VLRResult vlrContextGetOutputBufferSize(VLRContext context, uint32_t* width, uint32_t* height);
    VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCamera camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);
    VLR_API VLRResult vlrContextDebugRender(VLRContext context, VLRScene scene, VLRCamera camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);



    VLR_API VLRResult vlrImage2DGetWidth(VLRImage2D image, uint32_t* width);
    VLR_API VLRResult vlrImage2DGetHeight(VLRImage2D image, uint32_t* height);
    VLR_API VLRResult vlrImage2DGetStride(VLRImage2D image, uint32_t* stride);
    VLR_API VLRResult vlrImage2DGetOriginalDataFormat(VLRImage2D image, VLRDataFormat* format);
    VLR_API VLRResult vlrImage2DOriginalHasAlpha(VLRImage2D image, bool* hasAlpha);

    VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                             uint8_t* linearData, uint32_t width, uint32_t height,
                                             VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace);
    VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image);

    VLR_API VLRResult vlrBlockCompressedImage2DCreate(VLRContext context, VLRBlockCompressedImage2D* image,
                                                      uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                                      VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace);
    VLR_API VLRResult vlrBlockCompressedImage2DDestroy(VLRContext context, VLRBlockCompressedImage2D image);



    VLR_API VLRResult vlrShaderNodeGetSocket(VLRShaderNode node, VLRShaderNodeSocketType socketType, uint32_t option, 
                                             VLRShaderNodeSocket* socket);

    VLR_API VLRResult vlrGeometryShaderNodeCreate(VLRContext context, VLRGeometryShaderNode* node);
    VLR_API VLRResult vlrGeometryShaderNodeDestroy(VLRContext context, VLRGeometryShaderNode node);

    VLR_API VLRResult vlrFloat2ShaderNodeCreate(VLRContext context, VLRFloat2ShaderNode* node);
    VLR_API VLRResult vlrFloat2ShaderNodeDestroy(VLRContext context, VLRFloat2ShaderNode node);
    VLR_API VLRResult vlrFloat2ShaderNodeSetNode0(VLRFloat2ShaderNode node, VLRShaderNodeSocket node0);
    VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue0(VLRFloat2ShaderNode node, float value);
    VLR_API VLRResult vlrFloat2ShaderNodeSetNode1(VLRFloat2ShaderNode node, VLRShaderNodeSocket node1);
    VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue1(VLRFloat2ShaderNode node, float value);

    VLR_API VLRResult vlrFloat3ShaderNodeCreate(VLRContext context, VLRFloat3ShaderNode* node);
    VLR_API VLRResult vlrFloat3ShaderNodeDestroy(VLRContext context, VLRFloat3ShaderNode node);
    VLR_API VLRResult vlrFloat3ShaderNodeSetNode0(VLRFloat3ShaderNode node, VLRShaderNodeSocket node0);
    VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue0(VLRFloat3ShaderNode node, float value);
    VLR_API VLRResult vlrFloat3ShaderNodeSetNode1(VLRFloat3ShaderNode node, VLRShaderNodeSocket node1);
    VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue1(VLRFloat3ShaderNode node, float value);
    VLR_API VLRResult vlrFloat3ShaderNodeSetNode2(VLRFloat3ShaderNode node, VLRShaderNodeSocket node2);
    VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue2(VLRFloat3ShaderNode node, float value);

    VLR_API VLRResult vlrFloat4ShaderNodeCreate(VLRContext context, VLRFloat4ShaderNode* node);
    VLR_API VLRResult vlrFloat4ShaderNodeDestroy(VLRContext context, VLRFloat4ShaderNode node);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode0(VLRFloat4ShaderNode node, VLRShaderNodeSocket node0);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue0(VLRFloat4ShaderNode node, float value);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode1(VLRFloat4ShaderNode node, VLRShaderNodeSocket node1);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue1(VLRFloat4ShaderNode node, float value);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode2(VLRFloat4ShaderNode node, VLRShaderNodeSocket node2);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue2(VLRFloat4ShaderNode node, float value);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode3(VLRFloat4ShaderNode node, VLRShaderNodeSocket node3);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue3(VLRFloat4ShaderNode node, float value);

    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeCreate(VLRContext context, VLRScaleAndOffsetFloatShaderNode* node);
    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeDestroy(VLRContext context, VLRScaleAndOffsetFloatShaderNode node);
    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeValue(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNodeSocket nodeValue);
    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeScale(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNodeSocket nodeScale);
    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeOffset(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNodeSocket nodeOffset);
    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetImmediateValueScale(VLRScaleAndOffsetFloatShaderNode node, float value);
    VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetImmediateValueOffset(VLRScaleAndOffsetFloatShaderNode node, float value);

    VLR_API VLRResult vlrTripletSpectrumShaderNodeCreate(VLRContext context, VLRTripletSpectrumShaderNode* node);
    VLR_API VLRResult vlrTripletSpectrumShaderNodeDestroy(VLRContext context, VLRTripletSpectrumShaderNode node);
    VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueSpectrumType(VLRTripletSpectrumShaderNode node, VLRSpectrumType spectrumType);
    VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueColorSpace(VLRTripletSpectrumShaderNode node, VLRColorSpace colorSpace);
    VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueTriplet(VLRTripletSpectrumShaderNode node, float e0, float e1, float e2);

    VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeCreate(VLRContext context, VLRRegularSampledSpectrumShaderNode* node);
    VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeDestroy(VLRContext context, VLRRegularSampledSpectrumShaderNode node);
    VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeSetImmediateValueSpectrum(VLRRegularSampledSpectrumShaderNode node, VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples);

    VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeCreate(VLRContext context, VLRIrregularSampledSpectrumShaderNode* node);
    VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeDestroy(VLRContext context, VLRIrregularSampledSpectrumShaderNode node);
    VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeSetImmediateValueSpectrum(VLRIrregularSampledSpectrumShaderNode node, VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples);

    VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeCreate(VLRContext context, VLRFloat3ToSpectrumShaderNode* node);
    VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeDestroy(VLRContext context, VLRFloat3ToSpectrumShaderNode node);
    VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeSetNodeVector3D(VLRFloat3ToSpectrumShaderNode node, VLRShaderNodeSocket nodeFloat3);
    VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeSetImmediateValueVector3D(VLRFloat3ToSpectrumShaderNode node, const float value[3]);
    VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeSetImmediateValueSpectrumTypeAndColorSpace(VLRFloat3ToSpectrumShaderNode node, VLRSpectrumType spectrumType, VLRColorSpace colorSpace);

    VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeCreate(VLRContext context, VLRScaleAndOffsetUVTextureMap2DShaderNode* node);
    VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeDestroy(VLRContext context, VLRScaleAndOffsetUVTextureMap2DShaderNode node);
    VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeSetValues(VLRScaleAndOffsetUVTextureMap2DShaderNode node, const float offset[2], const float scale[2]);

    VLR_API VLRResult vlrImage2DTextureShaderNodeCreate(VLRContext context, VLRImage2DTextureShaderNode* node);
    VLR_API VLRResult vlrImage2DTextureShaderNodeDestroy(VLRContext context, VLRImage2DTextureShaderNode node);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetImage(VLRImage2DTextureShaderNode node, VLRImage2D image);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetFilterMode(VLRImage2DTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetWrapMode(VLRImage2DTextureShaderNode node, VLRTextureWrapMode x, VLRTextureWrapMode y);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetNodeTexCoord(VLRImage2DTextureShaderNode node, VLRShaderNodeSocket nodeTexCoord);

    VLR_API VLRResult vlrEnvironmentTextureShaderNodeCreate(VLRContext context, VLREnvironmentTextureShaderNode* node);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeDestroy(VLRContext context, VLREnvironmentTextureShaderNode node);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetImage(VLREnvironmentTextureShaderNode node, VLRImage2D image);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetFilterMode(VLREnvironmentTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetWrapMode(VLREnvironmentTextureShaderNode node, VLRTextureWrapMode x, VLRTextureWrapMode y);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetNodeTexCoord(VLREnvironmentTextureShaderNode node, VLRShaderNodeSocket nodeTexCoord);



    VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material);
    VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material);
    VLR_API VLRResult vlrMatteSurfaceMaterialSetNodeAlbedo(VLRMatteSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMatteSurfaceMaterialSetImmediateValueAlbedo(VLRMatteSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);

    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialCreate(VLRContext context, VLRSpecularReflectionSurfaceMaterial* material);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialDestroy(VLRContext context, VLRSpecularReflectionSurfaceMaterial material);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeEta(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNode_k(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);

    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialCreate(VLRContext context, VLRSpecularScatteringSurfaceMaterial* material);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialDestroy(VLRContext context, VLRSpecularScatteringSurfaceMaterial material);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);

    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialCreate(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial* material);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial material);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNode_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetReflectionSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetReflectionSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetReflectionSurfaceMaterial material, float value);

    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialCreate(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial* material);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial material);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetScatteringSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetScatteringSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetScatteringSurfaceMaterial material, float value);

    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialCreate(VLRContext context, VLRLambertianScatteringSurfaceMaterial* material);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialDestroy(VLRContext context, VLRLambertianScatteringSurfaceMaterial material);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeF0(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueF0(VLRLambertianScatteringSurfaceMaterial material, float value);

    VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material);
    VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material);
    VLR_API VLRResult vlrUE4SufaceMaterialSetNodeBaseColor(VLRUE4SurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueBaseColor(VLRUE4SurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrUE4SufaceMaterialSetNodeOcclusionRoughnessMetallic(VLRUE4SurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueOcclusion(VLRUE4SurfaceMaterial material, float value);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueRoughness(VLRUE4SurfaceMaterial material, float value);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueMetallic(VLRUE4SurfaceMaterial material, float value);

    VLR_API VLRResult vlrOldStyleSurfaceMaterialCreate(VLRContext context, VLROldStyleSurfaceMaterial* material);
    VLR_API VLRResult vlrOldStyleSurfaceMaterialDestroy(VLRContext context, VLROldStyleSurfaceMaterial material);
    VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeDiffuseColor(VLROldStyleSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeSpecularColor(VLROldStyleSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeGlossiness(VLROldStyleSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueDiffuseColor(VLROldStyleSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueSpecularColor(VLROldStyleSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueGlossiness(VLROldStyleSurfaceMaterial material, float value);

    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialCreate(VLRContext context, VLRDiffuseEmitterSurfaceMaterial* material);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialDestroy(VLRContext context, VLRDiffuseEmitterSurfaceMaterial material);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRShaderNodeSocket node);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueScale(VLRDiffuseEmitterSurfaceMaterial material, float value);

    VLR_API VLRResult vlrMultiSurfaceMaterialCreate(VLRContext context, VLRMultiSurfaceMaterial* material);
    VLR_API VLRResult vlrMultiSurfaceMaterialDestroy(VLRContext context, VLRMultiSurfaceMaterial material);
    VLR_API VLRResult vlrMultiSurfaceMaterialSetSubMaterial(VLRMultiSurfaceMaterial material, uint32_t index, VLRSurfaceMaterial mat);

    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialCreate(VLRContext context, VLREnvironmentEmitterSurfaceMaterial* material);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialDestroy(VLRContext context, VLREnvironmentEmitterSurfaceMaterial material);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceTextured(VLREnvironmentEmitterSurfaceMaterial material, VLREnvironmentTextureShaderNode node);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceConstant(VLREnvironmentEmitterSurfaceMaterial material, VLRShaderNode node);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance(VLREnvironmentEmitterSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueScale(VLREnvironmentEmitterSurfaceMaterial material, float value);



    VLR_API VLRResult vlrStaticTransformCreate(VLRContext context, VLRStaticTransform* transform, 
                                               const float mat[16]);
    VLR_API VLRResult vlrStaticTransformDestroy(VLRContext context, VLRStaticTransform transform);



    VLR_API VLRResult vlrNodeSetName(VLRNode node, const char* name);
    VLR_API VLRResult vlrNodeGetName(VLRNode node, const char** name);
    VLR_API VLRResult vlrNodeGetType(VLRNode node, VLRNodeType* type);
    
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode,
                                                       const char* name);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, VLRVertex* vertices, uint32_t numVertices);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, uint32_t* indices, uint32_t numIndices, 
                                                                 VLRSurfaceMaterial material, VLRShaderNodeSocket nodeNormal, VLRShaderNodeSocket nodeAlpha,
                                                                 VLRTangentType tangentType);

    VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                            const char* name, VLRTransform transform);
    VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node);
    VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, VLRTransform localToWorld);
    VLR_API VLRResult vlrInternalNodeGetTransform(VLRInternalNode node, VLRTransformConst* localToWorld);
    VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRNode child);
    VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRNode child);



    VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                     VLRTransform transform);
    VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene);
    VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, VLRTransform localToWorld);
    VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRNode child);
    VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRNode child);
    VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLREnvironmentEmitterSurfaceMaterial material);
    VLR_API VLRResult vlrSceneSetEnvironmentRotation(VLRScene scene, float rotationPhi);



    VLR_API VLRResult vlrCameraGetType(VLRCamera camera, VLRCameraType* type);
    
    VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera);
    VLR_API VLRResult vlrPerspectiveCameraDestroy(VLRContext context, VLRPerspectiveCamera camera);
    VLR_API VLRResult vlrPerspectiveCameraSetPosition(VLRPerspectiveCamera camera, const VLRPoint3D* position);
    VLR_API VLRResult vlrPerspectiveCameraSetOrientation(VLRPerspectiveCamera camera, const VLRQuaternion* orientation);
    VLR_API VLRResult vlrPerspectiveCameraSetAspectRatio(VLRPerspectiveCamera camera, float aspect);
    VLR_API VLRResult vlrPerspectiveCameraSetSensitivity(VLRPerspectiveCamera camera, float sensitivity);
    VLR_API VLRResult vlrPerspectiveCameraSetFovY(VLRPerspectiveCamera camera, float fovY);
    VLR_API VLRResult vlrPerspectiveCameraSetLensRadius(VLRPerspectiveCamera camera, float lensRadius);
    VLR_API VLRResult vlrPerspectiveCameraSetObjectPlaneDistance(VLRPerspectiveCamera camera, float distance);
    VLR_API VLRResult vlrPerspectiveCameraGetPosition(VLRPerspectiveCamera camera, VLRPoint3D* position);
    VLR_API VLRResult vlrPerspectiveCameraGetOrientation(VLRPerspectiveCamera camera, VLRQuaternion* orientation);
    VLR_API VLRResult vlrPerspectiveCameraGetAspectRatio(VLRPerspectiveCamera camera, float* aspect);
    VLR_API VLRResult vlrPerspectiveCameraGetSensitivity(VLRPerspectiveCamera camera, float* sensitivity);
    VLR_API VLRResult vlrPerspectiveCameraGetFovY(VLRPerspectiveCamera camera, float* fovY);
    VLR_API VLRResult vlrPerspectiveCameraGetLensRadius(VLRPerspectiveCamera camera, float* lensRadius);
    VLR_API VLRResult vlrPerspectiveCameraGetObjectPlaneDistance(VLRPerspectiveCamera camera, float* distance);

    VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera);
    VLR_API VLRResult vlrEquirectangularCameraDestroy(VLRContext context, VLREquirectangularCamera camera);
    VLR_API VLRResult vlrEquirectangularCameraSetPosition(VLREquirectangularCamera camera, const VLRPoint3D* position);
    VLR_API VLRResult vlrEquirectangularCameraSetOrientation(VLREquirectangularCamera camera, const VLRQuaternion* orientation);
    VLR_API VLRResult vlrEquirectangularCameraSetSensitivity(VLREquirectangularCamera camera, float sensitivity);
    VLR_API VLRResult vlrEquirectangularCameraSetAngles(VLREquirectangularCamera camera, float phiAngle, float thetaAngle);
    VLR_API VLRResult vlrEquirectangularCameraGetPosition(VLREquirectangularCamera camera, VLRPoint3D* position);
    VLR_API VLRResult vlrEquirectangularCameraGetOrientation(VLREquirectangularCamera camera, VLRQuaternion* orientation);
    VLR_API VLRResult vlrEquirectangularCameraGetSensitivity(VLREquirectangularCamera camera, float* sensitivity);
    VLR_API VLRResult vlrEquirectangularCameraGetAngles(VLREquirectangularCamera camera, float* phiAngle, float* thetaAngle);
#if defined(__cplusplus)
}
#endif
