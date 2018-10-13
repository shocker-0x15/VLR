#pragma once

#include "common.h"

enum VLRDataFormat {
    VLRDataFormat_RGB8x3 = 0,
    VLRDataFormat_RGB_8x4,
    VLRDataFormat_RGBA8x4,
    VLRDataFormat_RGBA16Fx4,
    VLRDataFormat_RGBA32Fx4,
    VLRDataFormat_RG32Fx2,
    VLRDataFormat_Gray32F,
    VLRDataFormat_Gray8,
    NumVLRDataFormats
};



enum VLRTextureFilter {
    VLRTextureFilter_Nearest = 0,
    VLRTextureFilter_Linear,
    VLRTextureFilter_None
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
    VLRVector3D tangent;
    VLRTexCoord2D texCoord;
};
