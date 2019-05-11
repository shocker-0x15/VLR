#pragma once

#include "common_internal.h"
#include "../include/VLR/basic_types.h"

namespace VLR {
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



    RT_FUNCTION constexpr Vector3D asVector3D(const optix::float3 &v) {
        return Vector3D(v.x, v.y, v.z);
    }
    RT_FUNCTION inline optix::float3 asOptiXType(const Vector3D &v) {
        return optix::make_float3(v.x, v.y, v.z);
    }
    RT_FUNCTION constexpr Vector4D asVector4D(const optix::float4 &v) {
        return Vector4D(v.x, v.y, v.z, v.w);
    }
    RT_FUNCTION inline optix::float4 asOptiXType(const Vector4D &v) {
        return optix::make_float4(v.x, v.y, v.z, v.w);
    }
    RT_FUNCTION constexpr Normal3D asNormal3D(const optix::float3 &v) {
        return Normal3D(v.x, v.y, v.z);
    }
    RT_FUNCTION inline optix::float3 asOptiXType(const Normal3D &n) {
        return optix::make_float3(n.x, n.y, n.z);
    }
    RT_FUNCTION constexpr Point3D asPoint3D(const optix::float3 &v) {
        return Point3D(v.x, v.y, v.z);
    }
    RT_FUNCTION inline optix::float3 asOptiXType(const Point3D &p) {
        return optix::make_float3(p.x, p.y, p.z);
    }
    RT_FUNCTION constexpr TexCoord2D asTexCoord2D(const optix::float2 &v) {
        return TexCoord2D(v.x, v.y);
    }
    RT_FUNCTION inline optix::float2 asOptiXType(const TexCoord2D &p) {
        return optix::make_float2(p.u, p.v);
    }
    RT_FUNCTION inline Matrix3x3 asMatrix3x3(const optix::Matrix3x3 &mat) {
        optix::float3 col0 = mat.getCol(0);
        optix::float3 col1 = mat.getCol(1);
        optix::float3 col2 = mat.getCol(2);
        return Matrix3x3(asVector3D(col0), asVector3D(col1), asVector3D(col2));
    }
    RT_FUNCTION inline optix::Matrix3x3 asOptiXType(const Matrix3x3 &mat) {
        optix::Matrix3x3 ret;
        ret.setCol(0, asOptiXType(mat.c0));
        ret.setCol(1, asOptiXType(mat.c1));
        ret.setCol(2, asOptiXType(mat.c2));
        return ret;
    }
}
