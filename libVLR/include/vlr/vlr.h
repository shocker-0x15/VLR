#pragma once

#include "public_types.h"
#include <cuda.h>

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

#if !defined(VLR_TYPEDEF_DONE)
    // e.g. Object
    // typedef struct VLRObject_API* VLRObject;
    // typedef const struct VLRObject_API* VLRObjectConst
#   define VLR_PROCESS_CLASS(name) \
        typedef struct VLR ## name ## _API* VLR ## name; \
        typedef const struct VLR ## name ## _API* VLR ## name ## Const

VLR_PROCESS_CLASS(Context);

VLR_PROCESS_CLASS_LIST();
#   undef VLR_PROCESS_CLASS

#   define VLR_TYPEDEF_DONE
#endif



VLR_API const char* vlrGetErrorMessage(VLRResult code);



// Context

VLR_API VLRResult vlrCreateContext(
    CUcontext cuContext, bool logging, uint32_t maxCallableDepth,
    VLRContext* context);
VLR_API VLRResult vlrDestroyContext(
    VLRContext context);
VLR_API VLRResult vlrContextGetCUcontext(
    VLRContext context,
    CUcontext* cuContext);

VLR_API VLRResult vlrContextBindOutputBuffer(
    VLRContext context,
    uint32_t width, uint32_t height, uint32_t glTexID);
VLR_API VLRResult vlrContextGetOutputBufferSize(
    VLRContext context,
    uint32_t* width, uint32_t* height);
VLR_API VLRResult vlrContextGetOutputBuffer(
    VLRContext context,
    CUarray* array);
VLR_API VLRResult vlrContextReadOutputBuffer(
    VLRContext context,
    float* data);
VLR_API VLRResult vlrContextSetScene(
    VLRContext context,
    VLRScene scene);
VLR_API VLRResult vlrContextRender(
    VLRContext context,
    CUstream stream, VLRCameraConst camera, bool denoise, uint32_t shrinkCoeff, bool firstFrame,
    uint32_t limitNumAccumFrames, uint32_t* numAccumFrames);
VLR_API VLRResult vlrContextDebugRender(
    VLRContext context,
    CUstream stream, VLRCameraConst camera, VLRDebugRenderingMode renderMode,
    uint32_t shrinkCoeff, bool firstFrame,
    uint32_t limitNumAccumFrames, uint32_t* numAccumFrames);



VLR_API VLRResult vlrObjectGetType(
    VLRObjectConst object,
    const char** typeName);



VLR_API VLRResult vlrParameterInfoGetName(
    VLRParameterInfoConst paramInfo,
    const char** name);
VLR_API VLRResult vlrParameterInfoGetSocketForm(
    VLRParameterInfoConst paramInfo,
    VLRParameterFormFlag* form);
VLR_API VLRResult vlrParameterInfoGetType(
    VLRParameterInfoConst paramInfo,
    const char** type);
VLR_API VLRResult vlrParameterInfoGetTupleSize(
    VLRParameterInfoConst paramInfo,
    uint32_t* size);



VLR_API VLRResult vlrGetNumEnumMembers(
    const char* typeName,
    uint32_t* numMembers);
VLR_API VLRResult vlrGetEnumMember(
    const char* typeName, uint32_t index,
    const char** value);


    
// Queryable
// Image2D, ShaderNode, SurfaceMaterial, Camera

VLR_API VLRResult vlrQueryableGetNumParameters(
    VLRQueryableConst queryable,
    uint32_t* numParams);
VLR_API VLRResult vlrQueryableGetParameterInfo(
    VLRQueryableConst queryable,
    uint32_t index, VLRParameterInfoConst* paramInfo);

VLR_API VLRResult vlrQueryableGetEnumValue(
    VLRQueryableConst queryable,
    const char* paramName,
    const char** value);
VLR_API VLRResult vlrQueryableGetPoint3D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRPoint3D* value);
VLR_API VLRResult vlrQueryableGetVector3D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRVector3D* value);
VLR_API VLRResult vlrQueryableGetNormal3D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRNormal3D* value);
VLR_API VLRResult vlrQueryableGetQuaternion(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRQuaternion* value);
VLR_API VLRResult vlrQueryableGetFloat(
    VLRQueryableConst queryable,
    const char* paramName,
    float* value);
VLR_API VLRResult vlrQueryableGetFloatTuple(
    VLRQueryableConst queryable,
    const char* paramName,
    float* values, uint32_t length);
VLR_API VLRResult vlrQueryableGetFloatArray(
    VLRQueryableConst queryable,
    const char* paramName,
    const float** values, uint32_t* length);
VLR_API VLRResult vlrQueryableGetImage2D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRImage2DConst* image);
VLR_API VLRResult vlrQueryableGetImmediateSpectrum(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRImmediateSpectrum* value);
VLR_API VLRResult vlrQueryableGetSurfaceMaterial(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRSurfaceMaterialConst* value);
VLR_API VLRResult vlrQueryableGetShaderNodePlug(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRShaderNodePlug* plug);

VLR_API VLRResult vlrQueryableSetEnumValue(
    VLRQueryable queryable,
    const char* paramName, const char* value);
VLR_API VLRResult vlrQueryableSetPoint3D(
    VLRQueryable queryable,
    const char* paramName, const VLRPoint3D* value);
VLR_API VLRResult vlrQueryableSetVector3D(
    VLRQueryable queryable,
    const char* paramName, const VLRVector3D* value);
VLR_API VLRResult vlrQueryableSetNormal3D(
    VLRQueryable queryable,
    const char* paramName, const VLRNormal3D* value);
VLR_API VLRResult vlrQueryableSetQuaternion(
    VLRQueryable queryable,
    const char* paramName, const VLRQuaternion* value);
VLR_API VLRResult vlrQueryableSetFloat(
    VLRQueryable queryable,
    const char* paramName, float value);
VLR_API VLRResult vlrQueryableSetFloatTuple(
    VLRQueryable queryable,
    const char* paramName, const float* values, uint32_t length);
VLR_API VLRResult vlrQueryableSetImage2D(
    VLRQueryable queryable,
    const char* paramName, VLRImage2DConst image);
VLR_API VLRResult vlrQueryableSetImmediateSpectrum(
    VLRQueryable queryable,
    const char* paramName, const VLRImmediateSpectrum* value);
VLR_API VLRResult vlrQueryableSetSurfaceMaterial(
    VLRQueryable queryable,
    const char* paramName, VLRSurfaceMaterialConst value);
VLR_API VLRResult vlrQueryableSetShaderNodePlug(
    VLRQueryable queryable,
    const char* paramName, VLRShaderNodePlug plug);



// Image2D
// LinearImage2D, BlockCompressedImage2D
    
VLR_API VLRResult vlrImage2DGetWidth(
    VLRImage2DConst image,
    uint32_t* width);
VLR_API VLRResult vlrImage2DGetHeight(
    VLRImage2DConst image,
    uint32_t* height);
VLR_API VLRResult vlrImage2DGetStride(
    VLRImage2DConst image,
    uint32_t* stride);
VLR_API VLRResult vlrImage2DGetOriginalDataFormat(
    VLRImage2DConst image,
    const char** format);
VLR_API VLRResult vlrImage2DOriginalHasAlpha(
    VLRImage2DConst image,
    bool* hasAlpha);



// LinearImage2D
    
VLR_API VLRResult vlrLinearImage2DCreate(
    VLRContext context,
    uint8_t* linearData, uint32_t width, uint32_t height,
    const char* format, const char* spectrumType, const char* colorSpace,
    VLRLinearImage2D* image);
VLR_API VLRResult vlrLinearImage2DDestroy(
    VLRContext context,
    VLRLinearImage2D image);



// BlockCompressedImage2D
    
VLR_API VLRResult vlrBlockCompressedImage2DCreate(
    VLRContext context,
    uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
    const char* dataFormat, const char* spectrumType, const char* colorSpace,
    VLRBlockCompressedImage2D* image);
VLR_API VLRResult vlrBlockCompressedImage2DDestroy(
    VLRContext context,
    VLRBlockCompressedImage2D image);



// ShaderNode
    
VLR_API VLRResult vlrShaderNodeCreate(
    VLRContext context,
    const char* typeName,
    VLRShaderNode* node);
VLR_API VLRResult vlrShaderNodeDestroy(
    VLRContext context,
    VLRShaderNode node);

VLR_API VLRResult vlrShaderNodeGetPlug(
    VLRShaderNodeConst node,
    VLRShaderNodePlugType plugType, uint32_t option, 
    VLRShaderNodePlug* plug);



// SurfaceMaterial
    
VLR_API VLRResult vlrSurfaceMaterialCreate(
    VLRContext context,
    const char* typeName,
    VLRSurfaceMaterial* material);
VLR_API VLRResult vlrSurfaceMaterialDestroy(
    VLRContext context,
    VLRSurfaceMaterial material);



// Transform
    
VLR_API VLRResult vlrStaticTransformCreate(
    VLRContext context,
    const float mat[16],
    VLRStaticTransform* transform);
VLR_API VLRResult vlrStaticTransformDestroy(
    VLRContext context,
    VLRStaticTransform transform);
VLR_API VLRResult vlrStaticTransformGetArrays(
    VLRStaticTransformConst transform,
    float mat[16], float invMat[16]);



// Node
// TriangleMeshSurfaceNode, ParentNode, Scene
    
VLR_API VLRResult vlrNodeSetName(
    VLRNode node,
    const char* name);
VLR_API VLRResult vlrNodeGetName(
    VLRNodeConst node,
    const char** name);



// TriangleMeshSurfaceNode
    
VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(
    VLRContext context,
    const char* name,
    VLRTriangleMeshSurfaceNode* surfaceNode);
VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(
    VLRContext context,
    VLRTriangleMeshSurfaceNode surfaceNode);
VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(
    VLRTriangleMeshSurfaceNode surfaceNode,
    const VLRVertex* vertices, uint32_t numVertices);
VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(
    VLRTriangleMeshSurfaceNode surfaceNode,
    const uint32_t* indices, uint32_t numIndices,
    VLRSurfaceMaterialConst material,
    VLRShaderNodePlug nodeNormal, VLRShaderNodePlug nodeTangent, VLRShaderNodePlug nodeAlpha);



// PointSurfaceNode

VLR_API VLRResult vlrPointSurfaceNodeCreate(
    VLRContext context,
    const char* name,
    VLRPointSurfaceNode* surfaceNode);
VLR_API VLRResult vlrPointSurfaceNodeDestroy(
    VLRContext context,
    VLRPointSurfaceNode surfaceNode);
VLR_API VLRResult vlrPointSurfaceNodeSetVertices(
    VLRPointSurfaceNode surfaceNode,
    const VLRVertex* vertices, uint32_t numVertices);
VLR_API VLRResult vlrPointSurfaceNodeAddMaterialGroup(
    VLRPointSurfaceNode surfaceNode,
    const uint32_t* indices, uint32_t numIndices,
    VLRSurfaceMaterialConst material);



// ParentNode
// InternalNode, Scene

VLR_API VLRResult vlrParentNodeSetTransform(
    VLRParentNode node,
    VLRTransformConst localToWorld);
VLR_API VLRResult vlrParentNodeGetTransform(
    VLRParentNodeConst node,
    VLRTransformConst* localToWorld);
VLR_API VLRResult vlrParentNodeAddChild(
    VLRParentNode node,
    VLRNode child);
VLR_API VLRResult vlrParentNodeRemoveChild(
    VLRParentNode node,
    VLRNode child);
VLR_API VLRResult vlrParentNodeGetNumChildren(
    VLRParentNodeConst node,
    uint32_t* numChildren);
VLR_API VLRResult vlrParentNodeGetChildren(
    VLRParentNodeConst node,
    VLRNode* children);
VLR_API VLRResult vlrParentNodeGetChildAt(
    VLRParentNodeConst node,
    uint32_t index,
    VLRNode* child);



// InternalNode

VLR_API VLRResult vlrInternalNodeCreate(
    VLRContext context,
    const char* name, VLRTransformConst transform,
    VLRInternalNode* node);
VLR_API VLRResult vlrInternalNodeDestroy(
    VLRContext context,
    VLRInternalNode node);



// Scene
    
VLR_API VLRResult vlrSceneCreate(
    VLRContext context,
    VLRTransformConst transform,
    VLRScene* scene);
VLR_API VLRResult vlrSceneDestroy(
    VLRContext context,
    VLRScene scene);
VLR_API VLRResult vlrSceneSetEnvironment(
    VLRScene scene,
    VLRSurfaceMaterial material);
VLR_API VLRResult vlrSceneSetEnvironmentRotation(
    VLRScene scene,
    float rotationPhi);



// Camera
    
VLR_API VLRResult vlrCameraCreate(
    VLRContext context,
    const char* typeName,
    VLRCamera* camera);
VLR_API VLRResult vlrCameraDestroy(
    VLRContext context,
    VLRCamera camera);

#if defined(__cplusplus)
}
#endif
