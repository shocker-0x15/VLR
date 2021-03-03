#pragma once

#if defined(__CUDACC__)
#   define VLR_Device

#   define vlrDevPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   define vlrprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#   define VLR_Host
#endif

#include "common.h"

#define VLR_Color_System_CIE_1931_2deg  0
#define VLR_Color_System_CIE_1964_10deg 1
#define VLR_Color_System_CIE_2012_2deg  2
#define VLR_Color_System_CIE_2012_10deg 3

#define MENG_SPECTRAL_UPSAMPLING 0
#define JAKOB_SPECTRAL_UPSAMPLING 1 // TODO: 光源など1.0を超えるスペクトラムへの対応。

//#define VLR_USE_SPECTRAL_RENDERING
#define SPECTRAL_UPSAMPLING_METHOD MENG_SPECTRAL_UPSAMPLING
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
#include <filesystem>

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

#include "../utils/optixu_on_cudau.h"
#if defined(VLR_Host)
#   undef CUDA_DEVICE_FUNCTION
#   define CUDA_DEVICE_FUNCTION inline
#endif
#include <vector_types.h>
#include <vector_functions.h>



#if defined(DEBUG)
#   define VLR_DEBUG_SELECT(A, B) A
#else
#   define VLR_DEBUG_SELECT(A, B) B
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



// ----------------------------------------------------------------
// JP: よく使用される基礎的な関数の定義。
// EN: define fundamental functions often used.

namespace vlr {
    template <typename T, size_t size>
    CUDA_DEVICE_FUNCTION constexpr size_t lengthof(const T(&array)[size]) {
        return size;
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType lerp(RealType a, RealType b, RealType t) {
        return a * (1 - t) + b * t;
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType pow1(RealType x) {
        return x;
    }
    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType pow2(RealType x) {
        return x * x;
    }
    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType pow3(RealType x) {
        return x * x * x;
    }
    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType pow4(RealType x) {
        return x * x * x * x;
    }
    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType pow5(RealType x) {
        return x * x * x * x * x;
    }

    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr bool realEq(T a, T b, T epsilon) {
        bool forAbsolute = std::fabs(a - b) < epsilon;
        bool forRelative = std::fabs(a - b) < epsilon * std::fmax(std::fabs(a), std::fabs(b));
        return forAbsolute || forRelative;
    }
    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr bool realGE(T a, T b, T epsilon) { return a > b || realEq(a, b, epsilon); }
    template <typename T>
    CUDA_DEVICE_FUNCTION constexpr bool realLE(T a, T b, T epsilon) { return a < b || realEq(a, b, epsilon); }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType saturate(RealType x) {
        return std::clamp<RealType>(x, 0, 1);
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType smoothstep(RealType edge0, RealType edge1, RealType x) {
        // Scale, bias and saturate x to 0..1 range
        x = saturate((x - edge0) / (edge1 - edge0));
        // Evaluate polynomial
        return x * x * (3 - 2 * x);
    }

    template <typename RealType>
    CUDA_DEVICE_FUNCTION constexpr RealType remap(RealType orgValue, RealType orgMin, RealType orgMax, RealType newMin, RealType newMax) {
        RealType percentage = (orgValue - orgMin) / (orgMax - orgMin);
        return newMin + percentage * (newMax - newMin);
    }

    CUDA_DEVICE_FUNCTION uint32_t lzcnt(uint32_t x) {
#if defined(VLR_Host)
        return _lzcnt_u32(x);
#else
        return __clz(x);
#endif
    }

    CUDA_DEVICE_FUNCTION constexpr uint32_t lzcntConst(uint32_t x) {
        uint32_t count = 0;
        for (int bit = 31; bit >= 0; --bit) {
            if ((x >> bit) & 0b1)
                break;
            ++count;
        }
        return count;
    }

    CUDA_DEVICE_FUNCTION uint32_t tzcnt(uint32_t x) {
#if defined(VLR_Host)
        return _tzcnt_u32(x);
#else
        return __clz(__brev(x));
#endif
    }

    CUDA_DEVICE_FUNCTION constexpr uint32_t tzcntConst(uint32_t x) {
        uint32_t count = 0;
        for (int bit = 0; bit < 32; ++bit) {
            if ((x >> bit) & 0b1)
                break;
            ++count;
        }
        return count;
    }

    CUDA_DEVICE_FUNCTION int32_t popcnt(uint32_t x) {
#if defined(VLR_Host)
        return _mm_popcnt_u32(x);
#else
        return __popc(x);
#endif
    }

    CUDA_DEVICE_FUNCTION constexpr int32_t popcntConst(uint32_t x) {
        uint32_t count = 0;
        for (int bit = 0; bit < 32; ++bit) {
            if ((x >> bit) & 0b1)
                ++count;
        }
        return count;
    }

    CUDA_DEVICE_FUNCTION uint32_t nthSetBit(uint32_t value, int32_t n) {
        uint32_t idx = 0;
        int32_t count;
        if (n >= popcnt(value))
            return 0xFFFFFFFF;

        for (uint32_t width = 16; width >= 1; width >>= 1) {
            if (value == 0)
                return 0xFFFFFFFF;

            uint32_t mask = (1 << width) - 1;
            count = popcnt(value & mask);
            if (n >= count) {
                value >>= width;
                n -= count;
                idx += width;
            }
        }

        return idx;
    }

    //     0: 0
    //     1: 0
    //  2- 3: 1
    //  4- 7: 2
    //  8-15: 3
    // 16-31: 4
    // ...
    CUDA_DEVICE_FUNCTION uint32_t prevPowOf2Exponent(uint32_t x) {
        if (x == 0)
            return 0;
        return 31 - lzcnt(x);
    }

    CUDA_DEVICE_FUNCTION constexpr uint32_t prevPowOf2ExponentConst(uint32_t x) {
        if (x == 0)
            return 0;
        return 31 - lzcntConst(x);
    }

    //    0: 0
    //    1: 0
    //    2: 1
    // 3- 4: 2
    // 5- 8: 3
    // 9-16: 4
    // ...
    CUDA_DEVICE_FUNCTION uint32_t nextPowOf2Exponent(uint32_t x) {
        if (x == 0)
            return 0;
        return 32 - lzcnt(x - 1);
    }

    CUDA_DEVICE_FUNCTION constexpr uint32_t nextPowOf2ExponentConst(uint32_t x) {
        if (x == 0)
            return 0;
        return 32 - lzcntConst(x - 1);
    }

    //     0: 0
    //     1: 1
    //  2- 3: 2
    //  4- 7: 4
    //  8-15: 8
    // 16-31: 16
    // ...
    CUDA_DEVICE_FUNCTION uint32_t prevPowerOf2(uint32_t x) {
        if (x == 0)
            return 0;
        return 1 << prevPowOf2Exponent(x);
    }

    CUDA_DEVICE_FUNCTION constexpr uint32_t prevPowerOf2Const(uint32_t x) {
        if (x == 0)
            return 0;
        return 1 << prevPowOf2ExponentConst(x);
    }

    //    0: 0
    //    1: 1
    //    2: 2
    // 3- 4: 4
    // 5- 8: 8
    // 9-16: 16
    // ...
    CUDA_DEVICE_FUNCTION uint32_t nextPowerOf2(uint32_t x) {
        if (x == 0)
            return 0;
        return 1 << nextPowOf2Exponent(x);
    }

    CUDA_DEVICE_FUNCTION constexpr uint32_t nextPowerOf2Const(uint32_t x) {
        if (x == 0)
            return 0;
        return 1 << nextPowOf2ExponentConst(x);
    }

    template <typename IntType>
    CUDA_DEVICE_FUNCTION constexpr IntType nextMultiplesForPowOf2(IntType x, uint32_t exponent) {
        IntType mask = (1 << exponent) - 1;
        return (x + mask) & ~mask;
    }

    template <typename IntType>
    CUDA_DEVICE_FUNCTION constexpr IntType nextMultiplierForPowOf2(IntType x, uint32_t exponent) {
        return nextMultiplesForPowOf2(x, exponent) >> exponent;
    }

    template <typename RealType>
    struct CompensatedSum {
        RealType result;
        RealType comp;
        CUDA_DEVICE_FUNCTION CompensatedSum(const RealType &value) : result(value), comp(0.0) { }
        CUDA_DEVICE_FUNCTION CompensatedSum &operator=(const RealType &value) {
            result = value;
            comp = 0;
            return *this;
        }
        CUDA_DEVICE_FUNCTION CompensatedSum &operator+=(const RealType &value) {
            RealType cInput = value - comp;
            RealType sumTemp = result + cInput;
            comp = (sumTemp - result) - cInput;
            result = sumTemp;
            return *this;
        }
        CUDA_DEVICE_FUNCTION operator RealType() const { return result; }
    };

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



    inline std::string tolower(std::string str) {
        const auto tolower = [](unsigned char c) { return std::tolower(c); };
        std::transform(str.cbegin(), str.cend(), str.begin(), tolower);
        return str;
    }
#endif
}

// END: define fundamental functions often used.
// ----------------------------------------------------------------



// filesystem
#if defined(VLR_Host)
namespace vlr {
    std::filesystem::path getExecutableDirectory();
}
#endif
