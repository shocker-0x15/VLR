#pragma once

#if defined(__CUDACC__)
#   define VLR_Device
#   define RT_FUNCTION __forceinline__ __device__
#   define RT_VARIABLE __device__ const
#   define HOST_INLINE
#   define HOST_CONSTEXPR
#else
#   define VLR_Host
#   define RT_FUNCTION
#   define RT_VARIABLE
#   define HOST_INLINE inline
#   define HOST_CONSTEXPR constexpr
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
#   if defined(VLR_API_EXPORTS)
#       define VLR_API __declspec(dllexport)
#   else
#       define VLR_API __declspec(dllimport)
#   endif
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#else
#   define VLR_API
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
#include <cstdint>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <cfloat>

#endif

#if defined(DEBUG)
#   define ENABLE_ASSERT
#endif

// vlrDevPrintf
#if defined(VLR_Host)
#   if defined(VLR_Platform_Windows_MSVC)
VLR_API void vlrDevPrintf(const char* fmt, ...);
#   else
#       define vlrDevPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   endif
#else
#   define vlrDevPrintf(fmt, ...) rtPrintf(fmt, ##__VA_ARGS__)
#endif

// vlrprintf
#if defined(VLR_Host)
#   if defined(VLR_USE_DEVPRINTF)
#       define vlrprintf vlrDevPrintf
#   else
#       define vlrprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   endif
#else
#   define vlrprintf(fmt, ...) rtPrintf(fmt, ##__VA_ARGS__)
#endif

#if defined(ENABLE_ASSERT)
#   if defined(VLR_Host)
#       define VLRAssert(expr, fmt, ...) if (!(expr)) { vlrDevPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); vlrDevPrintf(fmt"\n", ##__VA_ARGS__); abort(); }
#   else
#       define VLRAssert(expr, fmt, ...) if (!(expr)) { vlrDevPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); vlrDevPrintf(fmt"\n", ##__VA_ARGS__); }
#   endif
#else
#   define VLRAssert(expr, fmt, ...)
#endif

#define VLRAssert_ShouldNotBeCalled() VLRAssert(false, "Should not be called!")
#define VLRAssert_NotImplemented() VLRAssert(false, "Not implemented yet!")



// ----------------------------------------------------------------
// JP: よく使用される基礎的な関数の定義。
// EN: define fundamental functions often used.

#if defined(VLR_Device)
namespace std {
    template <typename T>
    RT_FUNCTION constexpr T min(const T &a, const T &b) {
        return a < b ? a : b;
    }

    template <typename T>
    RT_FUNCTION constexpr T max(const T &a, const T &b) {
        return a > b ? a : b;
    }

    RT_FUNCTION constexpr bool isinf(float x) {
        return ::isinf(x);
    }

    RT_FUNCTION constexpr bool isnan(float x) {
        return ::isnan(x);
    }

    RT_FUNCTION constexpr bool isfinite(float x) {
        return ::isfinite(x);
    }
}

namespace VLR {
    template <typename T>
    RT_FUNCTION constexpr void sincos(T angle, T* s, T* c);

    template <>
    RT_FUNCTION constexpr void sincos<float>(float angle, float* s, float* c) {
        ::sincosf(angle, s, c);
    }
    template <>
    RT_FUNCTION constexpr void sincos<double>(double angle, double* s, double* c) {
        ::sincos(angle, s, c);
    }
}
#else
namespace VLR {
    template <typename T>
    RT_FUNCTION HOST_INLINE constexpr void sincos(T angle, T* s, T* c) {
        *s = std::sin(angle);
        *c = std::cos(angle);
    }
}
#endif

namespace VLR {
    template <typename T>
    RT_FUNCTION HOST_INLINE constexpr T clamp(const T &v, const T &minv, const T &maxv) {
        return std::min<T>(maxv, std::max<T>(minv, v));
    }

    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType lerp(RealType a, RealType b, RealType t) {
        return a * (1 - t) + b * t;
    }

    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType pow1(RealType x) {
        return x;
    }
    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType pow2(RealType x) {
        return x * x;
    }
    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType pow3(RealType x) {
        return x * x * x;
    }
    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType pow4(RealType x) {
        return x * x * x * x;
    }
    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType pow5(RealType x) {
        return x * x * x * x *x;
    }

    template <typename T>
    RT_FUNCTION HOST_INLINE constexpr bool realEq(T a, T b, T epsilon) {
        bool forAbsolute = std::fabs(a - b) < epsilon;
        bool forRelative = std::fabs(a - b) < epsilon * std::fmax(std::fabs(a), std::fabs(b));
        return forAbsolute || forRelative;
    }
    template <typename T>
    RT_FUNCTION HOST_INLINE constexpr bool realGE(T a, T b, T epsilon) { return a > b || realEq(a, b, epsilon); }
    template <typename T>
    RT_FUNCTION HOST_INLINE constexpr bool realLE(T a, T b, T epsilon) { return a < b || realEq(a, b, epsilon); }
}

// END: define fundamental functions often used.
// ----------------------------------------------------------------
