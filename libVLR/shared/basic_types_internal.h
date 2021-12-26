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



    CUDA_DEVICE_FUNCTION int32_t floatToOrderedInt(float fVal) {
#if defined(VLR_Host)
        int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
#else
        int32_t iVal = __float_as_int(fVal);
#endif
        return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
    }

    CUDA_DEVICE_FUNCTION float orderedIntToFloat(int32_t iVal) {
        int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
#if defined(VLR_Host)
        return *reinterpret_cast<float*>(&orgVal);
#else
        return __int_as_float(orgVal);
#endif
    }

    CUDA_DEVICE_FUNCTION uint32_t floatToOrderedUInt(float fVal) {
#if defined(VLR_Host)
        int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
#else
        int32_t iVal = __float_as_int(fVal);
#endif
        return iVal ^ (iVal >= 0 ? 0x80000000 : 0xFFFFFFFF);
    }

    struct Point3DAsOrderedInt {
        int32_t x, y, z;

        CUDA_DEVICE_FUNCTION Point3DAsOrderedInt() : x(0), y(0), z(0) {
        }
        CUDA_DEVICE_FUNCTION Point3DAsOrderedInt(const Point3D &v) :
            x(floatToOrderedInt(v.x)), y(floatToOrderedInt(v.y)), z(floatToOrderedInt(v.z)) {
        }

        CUDA_DEVICE_FUNCTION Point3DAsOrderedInt& operator=(const Point3DAsOrderedInt &v) {
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }
        CUDA_DEVICE_FUNCTION Point3DAsOrderedInt& operator=(const volatile Point3DAsOrderedInt &v) {
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }
        CUDA_DEVICE_FUNCTION volatile Point3DAsOrderedInt& operator=(const Point3DAsOrderedInt &v) volatile {
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }
        CUDA_DEVICE_FUNCTION volatile Point3DAsOrderedInt& operator=(const volatile Point3DAsOrderedInt &v) volatile {
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }

        CUDA_DEVICE_FUNCTION explicit operator Point3D() const {
            return Point3D(orderedIntToFloat(x), orderedIntToFloat(y), orderedIntToFloat(z));
        }
        CUDA_DEVICE_FUNCTION explicit operator Point3D() const volatile {
            return Point3D(orderedIntToFloat(x), orderedIntToFloat(y), orderedIntToFloat(z));
        }
    };

    CUDA_DEVICE_FUNCTION Point3DAsOrderedInt min(const Point3DAsOrderedInt &a, const Point3DAsOrderedInt &b) {
        Point3DAsOrderedInt ret;
        ret.x = vlr::min(a.x, b.x);
        ret.y = vlr::min(a.y, b.y);
        ret.z = vlr::min(a.z, b.z);
        return ret;
    }

    CUDA_DEVICE_FUNCTION Point3DAsOrderedInt max(const Point3DAsOrderedInt &a, const Point3DAsOrderedInt &b) {
        Point3DAsOrderedInt ret;
        ret.x = vlr::max(a.x, b.x);
        ret.y = vlr::max(a.y, b.y);
        ret.z = vlr::max(a.z, b.z);
        return ret;
    }

    struct BoundingBox3DAsOrderedInt {
        Point3DAsOrderedInt minP;
        Point3DAsOrderedInt maxP;

        CUDA_DEVICE_FUNCTION BoundingBox3DAsOrderedInt() :
            minP(Point3D(INFINITY)), maxP(Point3D(-INFINITY)) {
        }
        CUDA_DEVICE_FUNCTION BoundingBox3DAsOrderedInt(const BoundingBox3D &v) :
            minP(v.minP), maxP(v.maxP) {
        }

        CUDA_DEVICE_FUNCTION BoundingBox3DAsOrderedInt& operator=(const BoundingBox3DAsOrderedInt &v) {
            minP = v.minP;
            maxP = v.maxP;
            return *this;
        }
        CUDA_DEVICE_FUNCTION BoundingBox3DAsOrderedInt& operator=(const volatile BoundingBox3DAsOrderedInt &v) {
            minP = v.minP;
            maxP = v.maxP;
            return *this;
        }
        CUDA_DEVICE_FUNCTION volatile BoundingBox3DAsOrderedInt& operator=(const BoundingBox3DAsOrderedInt &v) volatile {
            minP = v.minP;
            maxP = v.maxP;
            return *this;
        }
        CUDA_DEVICE_FUNCTION volatile BoundingBox3DAsOrderedInt& operator=(const volatile BoundingBox3DAsOrderedInt &v) volatile {
            minP = v.minP;
            maxP = v.maxP;
            return *this;
        }

        CUDA_DEVICE_FUNCTION explicit operator BoundingBox3D() const {
            return BoundingBox3D(static_cast<Point3D>(minP), static_cast<Point3D>(maxP));
        }
        CUDA_DEVICE_FUNCTION explicit operator BoundingBox3D() const volatile {
            return BoundingBox3D(static_cast<Point3D>(minP), static_cast<Point3D>(maxP));
        }

        struct unify {
            CUDA_DEVICE_FUNCTION BoundingBox3DAsOrderedInt operator()(const BoundingBox3DAsOrderedInt &a, const BoundingBox3DAsOrderedInt &b) {
                BoundingBox3DAsOrderedInt ret;
                ret.minP = min(a.minP, b.minP);
                ret.maxP = max(a.maxP, b.maxP);
                return ret;
            }
        };
    };

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
#if __CUDA_ARCH__ < 600
#   define atomicOr_block atomicOr
#   define atomicAnd_block atomicAnd
#   define atomicAdd_block atomicAdd
#   define atomicMin_block atomicMin
#   define atomicMax_block atomicMax
#endif

    CUDA_DEVICE_FUNCTION void atomicMinPoint3D(Point3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMin(&dst->x, v.x);
        atomicMin(&dst->y, v.y);
        atomicMin(&dst->z, v.z);
    }

    CUDA_DEVICE_FUNCTION void atomicMaxPoint3D(Point3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMax(&dst->x, v.x);
        atomicMax(&dst->y, v.y);
        atomicMax(&dst->z, v.z);
    }

    CUDA_DEVICE_FUNCTION void atomicMinPoint3D_block(Point3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMin_block(&dst->x, v.x);
        atomicMin_block(&dst->y, v.y);
        atomicMin_block(&dst->z, v.z);
    }

    CUDA_DEVICE_FUNCTION void atomicMaxPoint3D_block(Point3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMax_block(&dst->x, v.x);
        atomicMax_block(&dst->y, v.y);
        atomicMax_block(&dst->z, v.z);
    }

    CUDA_DEVICE_FUNCTION void atomicUnifyBoundingBox3D(BoundingBox3DAsOrderedInt* dst, const BoundingBox3DAsOrderedInt &v) {
        atomicMinPoint3D(&dst->minP, v.minP);
        atomicMaxPoint3D(&dst->maxP, v.maxP);
    }

    CUDA_DEVICE_FUNCTION void atomicUnifyBoundingBox3D(BoundingBox3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMinPoint3D(&dst->minP, v);
        atomicMaxPoint3D(&dst->maxP, v);
    }

    CUDA_DEVICE_FUNCTION void atomicUnifyBoundingBox3D_block(BoundingBox3DAsOrderedInt* dst, const BoundingBox3DAsOrderedInt &v) {
        atomicMinPoint3D_block(&dst->minP, v.minP);
        atomicMaxPoint3D_block(&dst->maxP, v.maxP);
    }

    CUDA_DEVICE_FUNCTION void atomicUnifyBoundingBox3D_block(BoundingBox3DAsOrderedInt* dst, const Point3DAsOrderedInt &v) {
        atomicMinPoint3D_block(&dst->minP, v);
        atomicMaxPoint3D_block(&dst->maxP, v);
    }
#endif
}
