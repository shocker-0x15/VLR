#pragma once

#include "common_internal.h"
#include "../include/VLR/basic_types.h"

namespace vlr {
    //template struct VLR_API Vector3DTemplate<float>;
    //template struct VLR_API Vector4DTemplate<float>;
    //template struct VLR_API Normal3DTemplate<float>;
    //template struct VLR_API Point3DTemplate<float>;
    //template struct VLR_API TexCoord2DTemplate<float>;
    //template struct VLR_API BoundingBox3DTemplate<float>;
    //template struct VLR_API Matrix3x3Template<float>;
    //template struct VLR_API Matrix4x4Template<float>;
    //template struct VLR_API QuaternionTemplate<float>;
    //template struct VLR_API RGBTemplate<float>;



    CUDA_DEVICE_FUNCTION constexpr Vector3D asVector3D(const float3 &v) {
        return Vector3D(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION float3 asOptiXType(const Vector3D &v) {
        return make_float3(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION constexpr Vector4D asVector4D(const float4 &v) {
        return Vector4D(v.x, v.y, v.z, v.w);
    }
    CUDA_DEVICE_FUNCTION float4 asOptiXType(const Vector4D &v) {
        return make_float4(v.x, v.y, v.z, v.w);
    }
    CUDA_DEVICE_FUNCTION constexpr Normal3D asNormal3D(const float3 &v) {
        return Normal3D(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION float3 asOptiXType(const Normal3D &n) {
        return make_float3(n.x, n.y, n.z);
    }
    CUDA_DEVICE_FUNCTION constexpr Point3D asPoint3D(const float3 &v) {
        return Point3D(v.x, v.y, v.z);
    }
    CUDA_DEVICE_FUNCTION float3 asOptiXType(const Point3D &p) {
        return make_float3(p.x, p.y, p.z);
    }
    CUDA_DEVICE_FUNCTION constexpr TexCoord2D asTexCoord2D(const float2 &v) {
        return TexCoord2D(v.x, v.y);
    }
    CUDA_DEVICE_FUNCTION float2 asOptiXType(const TexCoord2D &p) {
        return make_float2(p.u, p.v);
    }
}
