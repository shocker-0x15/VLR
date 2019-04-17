#pragma once

#include "common.h"

enum VLRSpectrumType {
    VLRSpectrumType_Reflectance = 0,
    VLRSpectrumType_Transmittance = VLRSpectrumType_Reflectance,
    VLRSpectrumType_LightSource,
    VLRSpectrumType_IndexOfRefraction,
    VLRSpectrumType_NA,
    NumVLRSpectrumTypes
};

enum VLRColorSpace {
    VLRColorSpace_Rec709_D65_sRGBGamma = 0,
    VLRColorSpace_Rec709_D65,
    VLRColorSpace_XYZ,
    VLRColorSpace_xyY,
    NumVLRColorSpaces
};



enum VLRDataFormat {
    VLRDataFormat_RGB8x3 = 0,
    VLRDataFormat_RGB_8x4,
    VLRDataFormat_RGBA8x4,
    VLRDataFormat_RGBA16Fx4,
    VLRDataFormat_RGBA32Fx4,
    VLRDataFormat_RG32Fx2,
    VLRDataFormat_Gray32F,
    VLRDataFormat_Gray8,
    VLRDataFormat_GrayA8x2,
    VLRDataFormat_BC1,
    VLRDataFormat_BC2,
    VLRDataFormat_BC3,
    VLRDataFormat_BC4,
    VLRDataFormat_BC4_Signed,
    VLRDataFormat_BC5,
    VLRDataFormat_BC5_Signed,
    VLRDataFormat_BC6H,
    VLRDataFormat_BC6H_Signed,
    VLRDataFormat_BC7,
    NumVLRDataFormats
};



enum VLRShaderNodeSocketType {
    VLRShaderNodeSocketType_float = 0,
    VLRShaderNodeSocketType_float2,
    VLRShaderNodeSocketType_float3,
    VLRShaderNodeSocketType_float4,
    VLRShaderNodeSocketType_Point3D,
    VLRShaderNodeSocketType_Vector3D,
    VLRShaderNodeSocketType_Normal3D,
    VLRShaderNodeSocketType_Spectrum,
    VLRShaderNodeSocketType_Alpha,
    VLRShaderNodeSocketType_TextureCoordinates,
    NumVLRShaderNodeSocketTypes
};

struct VLRShaderNodeSocketInfo {
    uint32_t dummy;
};



enum VLRTextureFilter {
    VLRTextureFilter_Nearest = 0,
    VLRTextureFilter_Linear,
    VLRTextureFilter_None
};

enum VLRTextureWrapMode {
    VLRTextureWrapMode_Repeat = 0,
    VLRTextureWrapMode_ClampToEdge,
    VLRTextureWrapMode_Mirror,
    VLRTextureWrapMode_ClampToBorder,
};



enum VLRTangentType {
    VLRTangentType_TC0Direction = 0,
    VLRTangentType_RadialX,
    VLRTangentType_RadialY,
    VLRTangentType_RadialZ,
};



enum VLRNodeType {
    VLRNodeType_TriangleMeshSurfaceNode = 0,
    VLRNodeType_InternalNode,
};

enum VLRCameraType {
    VLRCameraType_Perspective = 0,
    VLRCameraType_Equirectangular,
};

enum VLRDebugRenderingMode {
    VLRDebugRenderingMode_GeometricNormal = 0,
    VLRDebugRenderingMode_ShadingTangent,
    VLRDebugRenderingMode_ShadingBitangent,
    VLRDebugRenderingMode_ShadingNormal,
    VLRDebugRenderingMode_TC0Direction,
    VLRDebugRenderingMode_TextureCoordinates,
    VLRDebugRenderingMode_GeometricVsShadingNormal,
    VLRDebugRenderingMode_ShadingFrameLengths,
    VLRDebugRenderingMode_ShadingFrameOrthogonality,
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



struct VLRVertex {
    VLRPoint3D position;
    VLRNormal3D normal;
    VLRVector3D tc0Direction;
    VLRTexCoord2D texCoord;
};
