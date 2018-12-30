#pragma once

#include "common.h"

#define VLR_ENABLE_VALIDATION
#define VLR_ENABLE_TIMEOUT_CALLBACK

#define VLR_Color_System_CIE_1931_2deg  0
#define VLR_Color_System_CIE_1964_10deg 1
#define VLR_Color_System_CIE_2012_2deg  2
#define VLR_Color_System_CIE_2012_10deg 3

#define VLR_USE_SPECTRAL_RENDERING
#define VLR_Color_System_is_based_on VLR_Color_System_CIE_1931_2deg
static constexpr uint32_t NumSpectralSamples = 4;
static constexpr uint32_t NumStrataForStorage = 16;

#if VLR_Color_System_is_based_on == VLR_Color_System_CIE_1931_2deg
#   define xbarReferenceValues xbar_CIE1931_2deg
#   define ybarReferenceValues ybar_CIE1931_2deg
#   define zbarReferenceValues zbar_CIE1931_2deg
#elif VLR_Color_System_is_based_on == VLR_Color_System_CIE_1964_10deg
#   define xbarReferenceValues xbar_CIE1964_10deg
#   define ybarReferenceValues ybar_CIE1964_10deg
#   define zbarReferenceValues zbar_CIE1964_10deg
#elif VLR_Color_System_is_based_on == VLR_Color_System_CIE_2012_2deg
#   define xbarReferenceValues xbar_CIE2012_2deg
#   define ybarReferenceValues ybar_CIE2012_2deg
#   define zbarReferenceValues zbar_CIE2012_2deg
#elif VLR_Color_System_is_based_on == VLR_Color_System_CIE_2012_10deg
#   define xbarReferenceValues xbar_CIE2012_10deg
#   define ybarReferenceValues ybar_CIE2012_10deg
#   define zbarReferenceValues zbar_CIE2012_10deg
#endif

#if defined(VLR_Host)

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

namespace VLR {
    template <typename T, size_t size>
    RT_FUNCTION HOST_INLINE constexpr size_t lengthof(const T(&array)[size]) {
        return size;
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
        RealType percentage = (orgValue - orgMin) / (orgMax - orgMin);
        return newMin + percentage * (newMax - newMin);
    }

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
