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
        VLRResult_NumErrors
    };

#if !defined(__cplusplus)
    typedef enum VLRResult VLRResult;
#endif

#if !defined(VLR_API_EXPORTS)
    // e.g. Object
    // typedef struct VLRObject_API* VLRObject;
    // typedef const struct VLRObject_API* VLRObjectConst
#   define VLR_PROCESS_CLASS(name) \
        typedef struct VLR ## name ## _API* VLR ## name; \
        typedef const struct VLR ## name ## _API* VLR ## name ## Const

    VLR_PROCESS_CLASS(Context);

    VLR_PROCESS_CLASS_LIST();
#   undef VLR_PROCESS_CLASS
#endif



    VLR_API VLRResult vlrPrintDevices();
    VLR_API VLRResult vlrGetDeviceName(uint32_t index, char* name, uint32_t bufferLength);

    VLR_API const char* vlrGetErrorMessage(VLRResult code);

    VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices);
    VLR_API VLRResult vlrDestroyContext(VLRContext context);

    VLR_API VLRResult vlrContextGetNumDevices(VLRContext context, uint32_t* numDevices);
    VLR_API VLRResult vlrContextGetDeviceIndexAt(VLRContext context, uint32_t index, int32_t* deviceIndex);

    VLR_API VLRResult vlrContextBindOutputBuffer(VLRContext context, uint32_t width, uint32_t height, uint32_t bufferID);
    VLR_API VLRResult vlrContextMapOutputBuffer(VLRContext context, const void** ptr);
    VLR_API VLRResult vlrContextUnmapOutputBuffer(VLRContext context);
    VLR_API VLRResult vlrContextGetOutputBufferSize(VLRContext context, uint32_t* width, uint32_t* height);
    VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCameraConst camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);
    VLR_API VLRResult vlrContextDebugRender(VLRContext context, VLRScene scene, VLRCameraConst camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);



    VLR_API VLRResult vlrImage2DGetWidth(VLRImage2DConst image, uint32_t* width);
    VLR_API VLRResult vlrImage2DGetHeight(VLRImage2DConst image, uint32_t* height);
    VLR_API VLRResult vlrImage2DGetStride(VLRImage2DConst image, uint32_t* stride);
    VLR_API VLRResult vlrImage2DGetOriginalDataFormat(VLRImage2DConst image, VLRDataFormat* format);
    VLR_API VLRResult vlrImage2DOriginalHasAlpha(VLRImage2DConst image, bool* hasAlpha);

    VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                             uint8_t* linearData, uint32_t width, uint32_t height,
                                             VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace);
    VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image);

    VLR_API VLRResult vlrBlockCompressedImage2DCreate(VLRContext context, VLRBlockCompressedImage2D* image,
                                                      uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                                      VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace);
    VLR_API VLRResult vlrBlockCompressedImage2DDestroy(VLRContext context, VLRBlockCompressedImage2D image);



    VLR_API VLRResult vlrParameterInfoGetName(VLRParameterInfoConst paramInfo, const char** name);
    VLR_API VLRResult vlrParameterInfoGetSocketForm(VLRParameterInfoConst paramInfo, VLRParameterFormFlag* form);
    VLR_API VLRResult vlrParameterInfoGetType(VLRParameterInfoConst paramInfo, const char** type);
    VLR_API VLRResult vlrParameterInfoGetTupleSize(VLRParameterInfoConst paramInfo, uint32_t* size);



    VLR_API VLRResult vlrGetNumEnumMembers(const char* typeName, uint32_t* numMembers);
    VLR_API VLRResult vlrGetEnumMember(const char* typeName, uint32_t index, const char** value);
    
    // Connectable
    // Image2D, ShaderNode, SurfaceMaterial

    VLR_API VLRResult vlrConnectableGetNumParameters(VLRConnectableConst node, uint32_t* numParams);
    VLR_API VLRResult vlrConnectableGetParameterInfo(VLRConnectableConst node, uint32_t index, VLRParameterInfoConst* paramInfo);

    VLR_API VLRResult vlrConnectableGetEnumValue(VLRConnectableConst node, const char* paramName,
                                                 const char** value);
    VLR_API VLRResult vlrConnectableGetPoint3D(VLRConnectableConst node, const char* paramName,
                                               VLRPoint3D* value);
    VLR_API VLRResult vlrConnectableGetVector3D(VLRConnectableConst node, const char* paramName,
                                                VLRVector3D* value);
    VLR_API VLRResult vlrConnectableGetNormal3D(VLRConnectableConst node, const char* paramName,
                                                VLRNormal3D* value);
    VLR_API VLRResult vlrConnectableGetFloat(VLRConnectableConst node, const char* paramName,
                                             float* value);
    VLR_API VLRResult vlrConnectableGetFloatTuple(VLRConnectableConst node, const char* paramName,
                                                  float* values, uint32_t length);
    VLR_API VLRResult vlrConnectableGetFloatArray(VLRConnectableConst node, const char* paramName,
                                                  const float** values, uint32_t* length);
    VLR_API VLRResult vlrConnectableGetImage2D(VLRConnectableConst node, const char* paramName,
                                               VLRImage2DConst* image);
    VLR_API VLRResult vlrConnectableGetImmediateSpectrum(VLRConnectableConst node, const char* paramName,
                                                         VLRImmediateSpectrum* value);
    VLR_API VLRResult vlrConnectableGetSurfaceMaterial(VLRConnectableConst node, const char* paramName,
                                                       VLRSurfaceMaterialConst* value);
    VLR_API VLRResult vlrConnectableGetShaderNodePlug(VLRConnectableConst node, const char* paramName,
                                                      VLRShaderNodePlug* plug);

    VLR_API VLRResult vlrConnectableSetEnumValue(VLRConnectable node, const char* paramName,
                                                 const char* value);
    VLR_API VLRResult vlrConnectableSetPoint3D(VLRConnectable node, const char* paramName,
                                               const VLRPoint3D* value);
    VLR_API VLRResult vlrConnectableSetVector3D(VLRConnectable node, const char* paramName,
                                                const VLRVector3D* value);
    VLR_API VLRResult vlrConnectableSetNormal3D(VLRConnectable node, const char* paramName,
                                                const VLRNormal3D* value);
    VLR_API VLRResult vlrConnectableSetFloat(VLRConnectable node, const char* paramName,
                                             float value);
    VLR_API VLRResult vlrConnectableSetFloatTuple(VLRConnectable node, const char* paramName,
                                                  const float* values, uint32_t length);
    VLR_API VLRResult vlrConnectableSetImage2D(VLRConnectable node, const char* paramName,
                                               VLRImage2DConst image);
    VLR_API VLRResult vlrConnectableSetImmediateSpectrum(VLRConnectable node, const char* paramName,
                                                         const VLRImmediateSpectrum* value);
    VLR_API VLRResult vlrConnectableSetSurfaceMaterial(VLRConnectable node, const char* paramName,
                                                       VLRSurfaceMaterialConst value);
    VLR_API VLRResult vlrConnectableSetShaderNodePlug(VLRConnectable node, const char* paramName,
                                                      VLRShaderNodePlug plug);



    VLR_API VLRResult vlrShaderNodeCreate(VLRContext context, const char* typeName, VLRShaderNode* node);
    VLR_API VLRResult vlrShaderNodeDestroy(VLRContext context, VLRShaderNode node);

    VLR_API VLRResult vlrShaderNodeGetPlug(VLRShaderNodeConst node, VLRShaderNodePlugType plugType, uint32_t option, 
                                           VLRShaderNodePlug* plug);



    VLR_API VLRResult vlrSurfaceMaterialCreate(VLRContext context, const char* typeName, VLRSurfaceMaterial* material);
    VLR_API VLRResult vlrSurfaceMaterialDestroy(VLRContext context, VLRSurfaceMaterial material);



    VLR_API VLRResult vlrTransformGetType(VLRTransformConst transform, VLRTransformType* type);

    VLR_API VLRResult vlrStaticTransformCreate(VLRContext context, VLRStaticTransform* transform, 
                                               const float mat[16]);
    VLR_API VLRResult vlrStaticTransformDestroy(VLRContext context, VLRStaticTransform transform);
    VLR_API VLRResult vlrStaticTransformGetArrays(VLRStaticTransformConst transform, float mat[16], float invMat[16]);



    VLR_API VLRResult vlrNodeSetName(VLRNode node, const char* name);
    VLR_API VLRResult vlrNodeGetName(VLRNodeConst node, const char** name);
    VLR_API VLRResult vlrNodeGetType(VLRNodeConst node, VLRNodeType* type);
    
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode,
                                                       const char* name);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, const VLRVertex* vertices, uint32_t numVertices);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, const uint32_t* indices, uint32_t numIndices, 
                                                                 VLRSurfaceMaterialConst material, VLRShaderNodePlug nodeNormal, VLRShaderNodePlug nodeAlpha,
                                                                 VLRTangentType tangentType);

    VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                            const char* name, VLRTransformConst transform);
    VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node);
    VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, VLRTransformConst localToWorld);
    VLR_API VLRResult vlrInternalNodeGetTransform(VLRInternalNodeConst node, VLRTransformConst* localToWorld);
    VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRNode child);
    VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRNode child);
    VLR_API VLRResult vlrInternalNodeGetNumChildren(VLRInternalNodeConst node, uint32_t* numChildren);
    VLR_API VLRResult vlrInternalNodeGetChildren(VLRInternalNodeConst node, VLRNode* children);
    VLR_API VLRResult vlrInternalNodeGetChildAt(VLRInternalNodeConst node, uint32_t index, VLRNode* child);



    VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                     VLRTransformConst transform);
    VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene);
    VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, VLRTransformConst localToWorld);
    VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRNode child);
    VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRNode child);
    VLR_API VLRResult vlrSceneGetNumChildren(VLRSceneConst scene, uint32_t* numChildren);
    VLR_API VLRResult vlrSceneGetChildren(VLRSceneConst scene, VLRNode* children);
    VLR_API VLRResult vlrSceneGetChildAt(VLRSceneConst scene, uint32_t index, VLRNode* child);
    VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLRSurfaceMaterial material);
    VLR_API VLRResult vlrSceneSetEnvironmentRotation(VLRScene scene, float rotationPhi);



    VLR_API VLRResult vlrCameraGetType(VLRCameraConst camera, VLRCameraType* type);
    
    VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera);
    VLR_API VLRResult vlrPerspectiveCameraDestroy(VLRContext context, VLRPerspectiveCamera camera);
    VLR_API VLRResult vlrPerspectiveCameraSetPosition(VLRPerspectiveCamera camera, const VLRPoint3D* position);
    VLR_API VLRResult vlrPerspectiveCameraSetOrientation(VLRPerspectiveCamera camera, const VLRQuaternion* orientation);
    VLR_API VLRResult vlrPerspectiveCameraSetAspectRatio(VLRPerspectiveCamera camera, float aspect);
    VLR_API VLRResult vlrPerspectiveCameraSetSensitivity(VLRPerspectiveCamera camera, float sensitivity);
    VLR_API VLRResult vlrPerspectiveCameraSetFovY(VLRPerspectiveCamera camera, float fovY);
    VLR_API VLRResult vlrPerspectiveCameraSetLensRadius(VLRPerspectiveCamera camera, float lensRadius);
    VLR_API VLRResult vlrPerspectiveCameraSetObjectPlaneDistance(VLRPerspectiveCamera camera, float distance);
    VLR_API VLRResult vlrPerspectiveCameraGetPosition(VLRPerspectiveCameraConst camera, VLRPoint3D* position);
    VLR_API VLRResult vlrPerspectiveCameraGetOrientation(VLRPerspectiveCameraConst camera, VLRQuaternion* orientation);
    VLR_API VLRResult vlrPerspectiveCameraGetAspectRatio(VLRPerspectiveCameraConst camera, float* aspect);
    VLR_API VLRResult vlrPerspectiveCameraGetSensitivity(VLRPerspectiveCameraConst camera, float* sensitivity);
    VLR_API VLRResult vlrPerspectiveCameraGetFovY(VLRPerspectiveCameraConst camera, float* fovY);
    VLR_API VLRResult vlrPerspectiveCameraGetLensRadius(VLRPerspectiveCameraConst camera, float* lensRadius);
    VLR_API VLRResult vlrPerspectiveCameraGetObjectPlaneDistance(VLRPerspectiveCameraConst camera, float* distance);

    VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera);
    VLR_API VLRResult vlrEquirectangularCameraDestroy(VLRContext context, VLREquirectangularCamera camera);
    VLR_API VLRResult vlrEquirectangularCameraSetPosition(VLREquirectangularCamera camera, const VLRPoint3D* position);
    VLR_API VLRResult vlrEquirectangularCameraSetOrientation(VLREquirectangularCamera camera, const VLRQuaternion* orientation);
    VLR_API VLRResult vlrEquirectangularCameraSetSensitivity(VLREquirectangularCamera camera, float sensitivity);
    VLR_API VLRResult vlrEquirectangularCameraSetAngles(VLREquirectangularCamera camera, float phiAngle, float thetaAngle);
    VLR_API VLRResult vlrEquirectangularCameraGetPosition(VLREquirectangularCameraConst camera, VLRPoint3D* position);
    VLR_API VLRResult vlrEquirectangularCameraGetOrientation(VLREquirectangularCameraConst camera, VLRQuaternion* orientation);
    VLR_API VLRResult vlrEquirectangularCameraGetSensitivity(VLREquirectangularCameraConst camera, float* sensitivity);
    VLR_API VLRResult vlrEquirectangularCameraGetAngles(VLREquirectangularCameraConst camera, float* phiAngle, float* thetaAngle);
#if defined(__cplusplus)
}
#endif
