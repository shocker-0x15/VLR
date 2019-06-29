#pragma once

#if defined(__cplusplus)
#include <cstdint>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

enum VLRParameterFormFlag {
    VLRParameterFormFlag_ImmediateValue = 1 << 0,
    VLRParameterFormFlag_Node = 1 << 1,
    VLRParameterFormFlag_Both = (VLRParameterFormFlag_ImmediateValue | VLRParameterFormFlag_Node)
};

enum VLRShaderNodePlugType {
    VLRShaderNodePlugType_float1 = 0,
    VLRShaderNodePlugType_float2,
    VLRShaderNodePlugType_float3,
    VLRShaderNodePlugType_float4,
    VLRShaderNodePlugType_Point3D,
    VLRShaderNodePlugType_Vector3D,
    VLRShaderNodePlugType_Normal3D,
    VLRShaderNodePlugType_Spectrum,
    VLRShaderNodePlugType_Alpha,
    VLRShaderNodePlugType_TextureCoordinates,
    NumVLRShaderNodePlugTypes
};

struct VLRShaderNodePlug {
    uintptr_t nodeRef;
    uint32_t info;
};



enum VLRDebugRenderingMode {
    VLRDebugRenderingMode_BaseColor = 0,
    VLRDebugRenderingMode_GeometricNormal,
    VLRDebugRenderingMode_ShadingTangent,
    VLRDebugRenderingMode_ShadingBitangent,
    VLRDebugRenderingMode_ShadingNormal,
    VLRDebugRenderingMode_TC0Direction,
    VLRDebugRenderingMode_TextureCoordinates,
    VLRDebugRenderingMode_GeometricVsShadingNormal,
    VLRDebugRenderingMode_ShadingFrameLengths,
    VLRDebugRenderingMode_ShadingFrameOrthogonality,
};

#if !defined(__cplusplus)
typedef enum VLRParameterFormFlag VLRParameterFormFlag;
typedef enum VLRShaderNodePlugType VLRShaderNodePlugType;
typedef struct VLRShaderNodePlug VLRShaderNodePlug;
typedef enum VLRDebugRenderingMode VLRDebugRenderingMode;
#endif



struct VLRImmediateSpectrum {
    const char* colorSpace;
    float e0;
    float e1;
    float e2;
};

struct VLRPoint3D {
    float x, y, z;
};

struct VLRNormal3D {
    float x, y, z;
};

struct VLRVector3D {
    float x, y, z;
};

struct VLRTexCoord2D {
    float u, v;
};

struct VLRQuaternion {
    float x, y, z, w;
};

#if !defined(__cplusplus)
typedef struct VLRImmediateSpectrum VLRImmediateSpectrum;
typedef struct VLRPoint3D VLRPoint3D;
typedef struct VLRNormal3D VLRNormal3D;
typedef struct VLRVector3D VLRVector3D;
typedef struct VLRTexCoord2D VLRTexCoord2D;
typedef struct VLRQuaternion VLRQuaternion;
#endif



struct VLRVertex {
    VLRPoint3D position;
    VLRNormal3D normal;
    VLRVector3D tc0Direction;
    VLRTexCoord2D texCoord;
};

#if !defined(__cplusplus)
typedef struct VLRVertex VLRVertex;
#endif



#define VLR_PROCESS_CLASS_LIST() \
    VLR_PROCESS_CLASS(Object); \
\
    VLR_PROCESS_CLASS(ParameterInfo); \
\
    VLR_PROCESS_CLASS(Queryable); \
\
    VLR_PROCESS_CLASS(Image2D); \
    VLR_PROCESS_CLASS(LinearImage2D); \
    VLR_PROCESS_CLASS(BlockCompressedImage2D); \
\
    VLR_PROCESS_CLASS(ShaderNode); \
\
    VLR_PROCESS_CLASS(SurfaceMaterial); \
\
    VLR_PROCESS_CLASS(Transform); \
    VLR_PROCESS_CLASS(StaticTransform); \
\
    VLR_PROCESS_CLASS(Node); \
    VLR_PROCESS_CLASS(SurfaceNode); \
    VLR_PROCESS_CLASS(TriangleMeshSurfaceNode); \
    VLR_PROCESS_CLASS(InternalNode); \
    VLR_PROCESS_CLASS(Scene); \
\
    VLR_PROCESS_CLASS(Camera);
