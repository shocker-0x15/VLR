#pragma once

#if defined(__CUDACC__)
#   define VLR_Device
#   define RT_FUNCTION __forceinline__ __device__
#else
#   define VLR_Host
#   define RT_FUNCTION
#endif

#define VLR_M_PI 3.14159265358979323846f
//#define VLR_INFINITY
//#define VLR_NAN

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

#ifdef VLR_Platform_Windows_MSVC
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   ifdef VLR_API_EXPORTS
#       define VLR_API __declspec(dllexport)
#   else
#       define VLR_API __declspec(dllimport)
#   endif
#   include <Windows.h>
#   undef near
#   undef far
#else
#   define VLR_API
#endif

#if defined(VLR_Host)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <cfloat>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>

#include <array>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <stack>

#include <chrono>
#include <limits>
#include <algorithm>
#include <memory>
#include <functional>

#endif

#ifdef DEBUG
#   define ENABLE_ASSERT
#endif

#if defined(VLR_Host)
#   if defined(VLR_Platform_Windows_MSVC)
void VLRDebugPrintf(const char* fmt, ...);
#   else
#       define VLRDebugPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#   endif
#else
#   define VLRDebugPrintf(fmt, ...) rtPrintf(fmt, ##__VA_ARGS__);
#endif

#ifdef ENABLE_ASSERT
#   define VLRAssert(expr, fmt, ...) if (!(expr)) { VLRDebugPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); VLRDebugPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define VLRAssert(expr, fmt, ...)
#endif

#define VLRAssert_ShouldNotBeCalled() VLRAssert(false, "Should not be called!")
#define VLRAssert_NotImplemented() VLRAssert(false, "Not implemented yet!")

#define VLR_Minimum_Machine_Alignment 16
#define VLR_L1_Cacheline_Size 64

// For memalign, free, alignof
#if defined(VLR_Platform_Windows_MSVC)
#   include <malloc.h>
#   define VLR_memalign(size, alignment) _aligned_malloc(size, alignment)
#   define VLR_freealign(ptr) _aligned_free(ptr)
#   define VLR_alignof(T) __alignof(T)
#elif defined(VLR_Platform_macOS)
inline void* VLR_memalign(size_t size, size_t alignment) {
    void* ptr;
    if (posix_memalign(&ptr, alignment, size))
        ptr = nullptr;
    return ptr;
}
#   define VLR_freealign(ptr) ::free(ptr)
#   define VLR_alignof(T) alignof(T)
#endif

// For getcwd
#if defined(VLR_Platform_Windows_MSVC)
#   define VLR_getcwd(size, buf) GetCurrentDirectory(size, buf)
#elif defined(VLR_Platform_macOS)
#   include <unistd.h>
#   define VLR_getcwd(size, buf) getcwd(buf, size)
#endif



// ----------------------------------------------------------------
// JP: よく使用される基礎的な関数の定義。
// EN: define fundamental functions often used.

#if defined(VLR_Device)
namespace std {
    template <typename T>
    __device__ constexpr T min(const T &a, const T &b) {
        return a < b ? a : b;
    }

    template <typename T>
    __device__ constexpr T max(const T &a, const T &b) {
        return a > b ? a : b;
    }

    __device__ constexpr bool isinf(float x) {
        return ::isinf(x);
    }
    
    __device__ constexpr bool isnan(float x) {
        return ::isnan(x);
    }
}
#endif

namespace VLR {
    template <typename T, size_t size>
    RT_FUNCTION constexpr size_t lengthof(const T(&array)[size]) {
        return size;
    }

    template <typename T>
    RT_FUNCTION constexpr T clamp(const T &v, const T &minv, const T &maxv) {
        return std::min(maxv, std::max(minv, v));
    }

    template <typename RealType>
    RT_FUNCTION constexpr RealType saturate(RealType x) {
        return clamp<RealType>(x, 0, 1);
    }

    template <typename RealType>
    RT_FUNCTION RealType smoothstep(RealType edge0, RealType edge1, RealType x) {
        // Scale, bias and saturate x to 0..1 range
        x = saturate((x - edge0) / (edge1 - edge0));
        // Evaluate polynomial
        return x * x * (3 - 2 * x);
    }

    template <typename RealType>
    RT_FUNCTION RealType remap(RealType orgValue, RealType orgMin, RealType orgMax, RealType newMin, RealType newMax) {
        RealType percenrage = (orgValue - orgMin) / (orgMax - orgMin);
        return newMin + percenrage * (newMax - newMin);
    }

    template <typename T>
    RT_FUNCTION bool realEq(T a, T b, T epsilon) {
        bool forAbsolute = std::fabs(a - b) < epsilon;
        bool forRelative = std::fabs(a - b) < epsilon * std::fmax(std::fabs(a), std::fabs(b));
        return forAbsolute || forRelative;
    }
    template <typename T>
    RT_FUNCTION bool realGE(T a, T b, T epsilon) { return a > b || realEq(a, b, epsilon); }
    template <typename T>
    RT_FUNCTION bool realLE(T a, T b, T epsilon) { return a < b || realEq(a, b, epsilon); }

    RT_FUNCTION constexpr uint32_t prevPowerOf2(uint32_t x) {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

#if defined(VLR_Host)
    template <typename T, typename ...Args>
    inline std::array<T, sizeof...(Args)> make_array(Args &&...args) {
        return std::array<T, sizeof...(Args)>{ std::forward<Args>(args)... };
    }

    template <typename T, typename ...ArgTypes>
    std::shared_ptr<T> createShared(ArgTypes&&... args) {
        return std::shared_ptr<T>(new T(std::forward<ArgTypes>(args)...));
    }

    template <typename T, typename ...ArgTypes>
    std::unique_ptr<T> createUnique(ArgTypes&&... args) {
        return std::unique_ptr<T>(new T(std::forward<ArgTypes>(args)...));
    }
#endif
}

// END: define fundamental functions often used.
// ----------------------------------------------------------------
