#pragma once

#if !defined(VLR_Device)
#   define VLR_Host
#endif

// Platform defines
#if defined(VLR_Host)
#   if defined(_WIN32) || defined(_WIN64)
#       define VLR_Platform_Windows
#       if defined(_MSC_VER)
#           define VLR_Platform_Windows_MSVC
#       endif
#   elif defined(__APPLE__)
#       define VLR_Platform_macOS
#   endif
#endif

#if defined(VLR_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#   if defined(VLR_API_EXPORTS)
#       define VLR_CPP_API __declspec(dllexport)
#   else
#       define VLR_CPP_API __declspec(dllimport)
#   endif
#else
#   define VLR_CPP_API
#endif

#define VLR_M_PI 3.14159265358979323846f
#ifndef VLR_HUGE_ENUF
#   define VLR_HUGE_ENUF  1e+300  // VLR_HUGE_ENUF*VLR_HUGE_ENUF must overflow
#endif
#define VLR_INFINITY   ((float)(VLR_HUGE_ENUF * VLR_HUGE_ENUF))
#define VLR_NAN        ((float)(VLR_INFINITY * 0.0f))

#define VLR_USE_DEVPRINTF

#if defined(VLR_Host)

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

#endif

#include <cstdint>
#include <cmath>
#include <cfloat>

#if defined(DEBUG)
#   define ENABLE_ASSERT
#endif

// vlrDevPrintf / vlrprintf
#if defined(VLR_Host)
#   if defined(VLR_Platform_Windows_MSVC)
VLR_CPP_API void vlrDevPrintf(const char* fmt, ...);
#   else
#       define vlrDevPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   endif

VLR_CPP_API void vlrprintf(const char* fmt, ...);
#endif

#if defined(ENABLE_ASSERT)
#   if defined(VLR_Host)
#       define VLRAssert(expr, fmt, ...) do { if (!(expr)) { vlrDevPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); vlrDevPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#   else
#       define VLRAssert(expr, fmt, ...) do { if (!(expr)) { vlrDevPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); vlrDevPrintf(fmt"\n", ##__VA_ARGS__); assert(false); } } while (0)
#   endif
#else
#   define VLRAssert(expr, fmt, ...)
#endif

#define VLRUnused(var) (void)var

#define VLRAssert_ShouldNotBeCalled() VLRAssert(false, "Should not be called!")
#define VLRAssert_NotImplemented() VLRAssert(false, "Not implemented yet!")



#if defined(VLR_Host)
#   define CUDA_DEVICE_FUNCTION inline
#   define HOST_STATIC_CONSTEXPR static constexpr
#else
#   define CUDA_DEVICE_FUNCTION __device__ __forceinline__
#   define HOST_STATIC_CONSTEXPR
#endif

namespace VLR {
    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T min(const T a, const T b) {
        return a < b ? a : b;
    }
    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T max(const T a, const T b) {
        return a > b ? a : b;
    }
    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr T clamp(const T v, const T minv, const T maxv) {
        return VLR::min(VLR::max(v, minv), maxv);
    }

    CUDA_DEVICE_FUNCTION bool isinf(float x) {
#if defined(VLR_Host)
        return std::isinf(x);
#else
        return isinf(x);
#endif
    }

    CUDA_DEVICE_FUNCTION bool isnan(float x) {
#if defined(VLR_Host)
        return std::isnan(x);
#else
        return isnan(x);
#endif
    }

    CUDA_DEVICE_FUNCTION bool isfinite(float x) {
#if defined(VLR_Host)
        return std::isfinite(x);
#else
        return isfinite(x);
#endif
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION void sincos(T angle, T* s, T* c);

    template <>
    CUDA_DEVICE_FUNCTION void sincos<float>(float angle, float* s, float* c) {
#if defined(VLR_Host)
        *s = std::sin(angle);
        *c = std::cos(angle);
#else
        ::sincosf(angle, s, c);
#endif
    }
}