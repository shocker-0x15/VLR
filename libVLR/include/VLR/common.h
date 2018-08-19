#pragma once

#if defined(__CUDACC__)
#   define VLR_Device
#   define RT_FUNCTION __forceinline__ __device__
#   define HOST_INLINE
#else
#   define VLR_Host
#   define RT_FUNCTION
#   define HOST_INLINE inline
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

#include <immintrin.h>

#endif

#if defined(DEBUG)
#   define ENABLE_ASSERT
#endif

#if defined(VLR_Host)
#   if defined(VLR_Platform_Windows_MSVC)
VLR_API void VLRDebugPrintf(const char* fmt, ...);
#   else
#       define VLRDebugPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#   endif
#else
#   define VLRDebugPrintf(fmt, ...) rtPrintf(fmt, ##__VA_ARGS__);
#endif

#if defined(ENABLE_ASSERT)
#   if defined(VLR_Host)
#       define VLRAssert(expr, fmt, ...) if (!(expr)) { VLRDebugPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); VLRDebugPrintf(fmt"\n", ##__VA_ARGS__); abort(); }
#   else
#       define VLRAssert(expr, fmt, ...) if (!(expr)) { VLRDebugPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); VLRDebugPrintf(fmt"\n", ##__VA_ARGS__); }
#   endif
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
    RT_FUNCTION HOST_INLINE constexpr size_t lengthof(const T(&array)[size]) {
        return size;
    }

    template <typename T>
    RT_FUNCTION HOST_INLINE constexpr T clamp(const T &v, const T &minv, const T &maxv) {
        return std::min(maxv, std::max(minv, v));
    }

    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType lerp(RealType a, RealType b, RealType t) {
        return a * (1 - t) + b * t;
    }

    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType saturate(RealType x) {
        return clamp<RealType>(x, 0, 1);
    }

    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType smoothstep(RealType edge0, RealType edge1, RealType x) {
        // Scale, bias and saturate x to 0..1 range
        x = saturate((x - edge0) / (edge1 - edge0));
        // Evaluate polynomial
        return x * x * (3 - 2 * x);
    }

    template <typename RealType>
    RT_FUNCTION HOST_INLINE constexpr RealType remap(RealType orgValue, RealType orgMin, RealType orgMax, RealType newMin, RealType newMax) {
        RealType percenrage = (orgValue - orgMin) / (orgMax - orgMin);
        return newMin + percenrage * (newMax - newMin);
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

    RT_FUNCTION HOST_INLINE constexpr uint32_t prevPowerOf2(uint32_t x) {
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return x - (x >> 1);
    }

    RT_FUNCTION HOST_INLINE constexpr uint32_t nextPowerOf2(uint32_t x) {
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        ++x;
        return x;
    }

    RT_FUNCTION HOST_INLINE uint32_t nextExpOf2(uint32_t n) {
        if (n == 0)
            return 0;
        uint32_t np = nextPowerOf2(n);
        //return _tzcnt_u64(np);
        uint32_t exp = 0;
        while ((np & (1 << exp)) == 0 && exp < 32)
            ++exp;
        return exp;
    }

    template <typename IntType>
    RT_FUNCTION HOST_INLINE constexpr IntType nextMultiplierForPowOf2(IntType value, uint32_t powOf2) {
        return (value + powOf2 - 1) / powOf2;
    }

    template <typename IntType>
    RT_FUNCTION HOST_INLINE constexpr IntType nextMultiplesOfPowOf2(IntType value, uint32_t powOf2) {
        return nextMultiplierForPowOf2(value, powOf2) * powOf2;
    }

    RT_FUNCTION HOST_INLINE uint32_t countTrailingZeroes(uint32_t value) {
        uint32_t count = 0;
        for (int i = 0; i < 32; ++i) {
            if ((value & 0x1) == 1)
                break;
            ++count;
            value >>= 1;
        }
        return count;
    }

    template <typename RealType>
    struct CompensatedSum {
        RealType result;
        RealType comp;
        RT_FUNCTION CompensatedSum(const RealType &value) : result(value), comp(0.0) { };
        RT_FUNCTION CompensatedSum &operator=(const RealType &value) {
            result = value;
            comp = 0;
            return *this;
        }
        RT_FUNCTION CompensatedSum &operator+=(const RealType &value) {
            RealType cInput = value - comp;
            RealType sumTemp = result + cInput;
            comp = (sumTemp - result) - cInput;
            result = sumTemp;
            return *this;
        }
        RT_FUNCTION operator RealType() const { return result; };
    };

#if defined(VLR_Host)
    inline uint32_t nthSetBit(uint32_t value, uint32_t n) {
        uint32_t idx = 0;
        uint32_t count;
        if (n >= _mm_popcnt_u32(value))
            return 0xFFFFFFFF;

        for (uint32_t width = 16; width >= 1; width >>= 1) {
            if (value == 0)
                return 0xFFFFFFFF;

            uint32_t mask = (1 << width) - 1;
            count = _mm_popcnt_u32(value & mask);
            if (n >= count) {
                value >>= width;
                n -= count;
                idx += width;
            }
        }

        return idx;
    }



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
