/*

   Copyright 2020 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#pragma once

/*

Note:
JP:
- 現状ではあらゆるAPIに破壊的変更が入る可能性が非常に高い。
- (少なくともホスト側コンパイラーがMSVC 16.7.6の場合は)"-std=c++17"をptxのコンパイル時に設定する必要あり。
- Visual StudioにおけるCUDAのプロパティ"Use Fast Math"はptxコンパイルに対して機能していない？
EN:
- It is very likely for now that any API will have breaking changes.
- Setting "-std=c++17" is required for ptx compilation (at least for the case the host compiler is MSVC 16.7.6).
- In Visual Studio, does the CUDA property "Use Fast Math" not work for ptx compilation??

----------------------------------------------------------------
TODO:
- Linux環境でのテスト。
- CMake整備。
- setPayloads/getPayloadsなどで引数側が必要以上の引数を渡していてもエラーが出ない問題。
- ASのRelocationサポート。
- Curve Primitiveサポート。
- AOV Denoiserサポート。
- 途中で各オブジェクトのパラメターを変更した際の処理。
  パイプラインのセットアップ順などが現状は暗黙的に固定されている。これを自由な順番で変えられるようにする。
- Multi GPUs?
- Assertとexceptionの整理。

検討事項:
- HitGroup以外のProgramGroupにユーザーデータを持たせる。
- GAS/IASに関してユーザーが気にするところはAS云々ではなくグループ化なので
  名前を変えるべき？GeometryGroup/InstanceGroupのような感じ。
  しかしビルドやアップデートを明示的にしているため結局ASであるということをユーザーが意識する必要がある。
- ユーザーがあるSBTレコード中の各データのストライドを意識せずともそれぞれのオフセットを取得する関数。
- GAS中のGeometryInstanceのインデックスを取得できるようにする。
  各GeometryInstanceが1つのSBTレコードしか使っていない場合はoptixGetSbtGASIndex()で代用できる。

*/

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define OPTIXU_Platform_Windows
#   if defined(_MSC_VER)
#       define OPTIXU_Platform_Windows_MSVC
#       if defined(__INTELLISENSE__)
#           define OPTIXU_Platform_CodeCompletion
#       endif
#   endif
#elif defined(__APPLE__)
#   define OPTIXU_Platform_macOS
#endif

#if defined(__CUDACC_RTC__)
// Defining cstdint and cfloat (under cuda/std) is left to the user.
#else
#include <cstdint>
#include <cfloat>
#include <string>
#endif
#include <optix.h>

#if !defined(__CUDA_ARCH__)
#include <optix_stubs.h>
#endif

#if defined(__CUDA_ARCH__)
#   define RT_CALLABLE_PROGRAM extern "C" __device__
#   define RT_DEVICE_FUNCTION __device__ __forceinline__
#   if !defined(RT_PIPELINE_LAUNCH_PARAMETERS)
#       define RT_PIPELINE_LAUNCH_PARAMETERS extern "C" __constant__
#   endif
#else
#   define RT_CALLABLE_PROGRAM
#   define RT_DEVICE_FUNCTION
#   define RT_PIPELINE_LAUNCH_PARAMETERS
#endif

#define RT_RG_NAME(name) __raygen__ ## name
#define RT_MS_NAME(name) __miss__ ## name
#define RT_EX_NAME(name) __exception__ ## name
#define RT_CH_NAME(name) __closesthit__ ## name
#define RT_AH_NAME(name) __anyhit__ ## name
#define RT_IS_NAME(name) __intersection__ ## name
#define RT_DC_NAME(name) __direct_callable__ ## name
#define RT_CC_NAME(name) __continuation_callable__ ## name

#define RT_RG_NAME_STR(name) "__raygen__" name
#define RT_MS_NAME_STR(name) "__miss__" name
#define RT_EX_NAME_STR(name) "__exception__" name
#define RT_CH_NAME_STR(name) "__closesthit__" name
#define RT_AH_NAME_STR(name) "__anyhit__" name
#define RT_IS_NAME_STR(name) "__intersection__" name
#define RT_DC_NAME_STR(name) "__direct_callable__" name
#define RT_CC_NAME_STR(name) "__continuation_callable__" name



inline OptixGeometryFlags operator|(OptixGeometryFlags a, OptixGeometryFlags b) {
    return static_cast<OptixGeometryFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline OptixPrimitiveTypeFlags operator|(OptixPrimitiveTypeFlags a, OptixPrimitiveTypeFlags b) {
    return static_cast<OptixPrimitiveTypeFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline OptixInstanceFlags operator|(OptixInstanceFlags a, OptixInstanceFlags b) {
    return static_cast<OptixInstanceFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline OptixMotionFlags operator|(OptixMotionFlags a, OptixMotionFlags b) {
    return static_cast<OptixMotionFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline OptixRayFlags operator|(OptixRayFlags a, OptixRayFlags b) {
    return static_cast<OptixRayFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline OptixTraversableGraphFlags operator|(OptixTraversableGraphFlags a, OptixTraversableGraphFlags b) {
    return static_cast<OptixTraversableGraphFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline OptixExceptionFlags operator|(OptixExceptionFlags a, OptixExceptionFlags b) {
    return static_cast<OptixExceptionFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}



namespace optixu {
#ifdef _DEBUG
#   define OPTIXU_ENABLE_ASSERT
#endif

    void devPrintf(const char* fmt, ...);

#if 1
#   define optixuPrintf(fmt, ...) \
        do { \
            optixu::devPrintf(fmt, ##__VA_ARGS__); \
            printf(fmt, ##__VA_ARGS__); \
        } while (0)
#else
#   define optixuPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#if defined(OPTIXU_ENABLE_ASSERT)
#   if defined(__CUDA_ARCH__)
#       define optixuAssert(expr, fmt, ...) \
            do { \
                if (!(expr)) { \
                    printf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
                    printf(fmt"\n", ##__VA_ARGS__); \
                    assert(0); \
                } \
            } while (0)
#   else
#       define optixuAssert(expr, fmt, ...) \
            do { \
                if (!(expr)) { \
                    optixu::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
                    optixu::devPrintf(fmt"\n", ##__VA_ARGS__); \
                    abort(); \
                } \
            } while (0)
#   endif
#else
#   define optixuAssert(expr, fmt, ...)
#endif

#define optixuAssert_ShouldNotBeCalled() optixuAssert(false, "Should not be called!")
#define optixuAssert_NotImplemented() optixuAssert(false, "Not implemented yet!")



    // ----------------------------------------------------------------
    // JP: ホスト・デバイス共有のクラス定義
    // EN: Definitions of Host-/Device-shared classes

    template <typename FuncType>
    class DirectCallableProgramID;

    template <typename ReturnType, typename... ArgTypes>
    class DirectCallableProgramID<ReturnType(ArgTypes...)> {
        uint32_t m_sbtIndex;

    public:
        RT_DEVICE_FUNCTION DirectCallableProgramID() {}
        RT_DEVICE_FUNCTION explicit DirectCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_DEVICE_FUNCTION explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
            return optixDirectCall<ReturnType, ArgTypes...>(m_sbtIndex, args...);
        }
#endif
    };

    template <typename FuncType>
    class ContinuationCallableProgramID;

    template <typename ReturnType, typename... ArgTypes>
    class ContinuationCallableProgramID<ReturnType(ArgTypes...)> {
        uint32_t m_sbtIndex;

    public:
        RT_DEVICE_FUNCTION ContinuationCallableProgramID() {}
        RT_DEVICE_FUNCTION explicit ContinuationCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_DEVICE_FUNCTION explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
            return optixContinuationCall<ReturnType, ArgTypes...>(m_sbtIndex, args...);
        }
#endif
    };

    // END: Definitions of Host-/Device-shared classes
    // ----------------------------------------------------------------




    // ----------------------------------------------------------------
    // JP: デバイス関数のラッパー
    // EN: Device-side function wrappers
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

    namespace detail {
        template <typename T>
        RT_DEVICE_FUNCTION constexpr size_t getNumDwords() {
            return (sizeof(T) + 3) / 4;
        }

        template <typename... Types>
        RT_DEVICE_FUNCTION constexpr size_t calcSumDwords() {
            return (0 + ... + getNumDwords<Types>());
        }

        template <uint32_t start, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void packToUInts(uint32_t* v, const HeadType &head, const TailTypes &... tails) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            constexpr uint32_t numDwords = sizeof(HeadType) / sizeof(uint32_t);
#pragma unroll
            for (int i = 0; i < numDwords; ++i)
                v[start + i] = *(reinterpret_cast<const uint32_t*>(&head) + i);
            if constexpr (sizeof...(tails) > 0)
                packToUInts<start + numDwords>(v, tails...);
        }

        template <typename Func, typename Type, uint32_t offsetInDst, uint32_t srcSlot>
        RT_DEVICE_FUNCTION void getValue(Type* value) {
            if (!value)
                return;
            *(reinterpret_cast<uint32_t*>(value) + offsetInDst) = Func::get<srcSlot>();
            if constexpr (offsetInDst + 1 < getNumDwords<Type>())
                getValue<Func, Type, offsetInDst + 1, srcSlot + 1>(value);
        }

        template <typename Func, uint32_t srcStartSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void getValues(HeadType* head, TailTypes*... tails) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            getValue<Func, HeadType, 0, srcStartSlot>(head);
            if constexpr (sizeof...(tails) > 0)
                getValues<Func, srcStartSlot + getNumDwords<HeadType>()>(tails...);
        }

        template <typename Func, typename Type, uint32_t offsetInSrc, uint32_t dstSlot>
        RT_DEVICE_FUNCTION void setValue(const Type* value) {
            if (!value)
                return;
            Func::set<dstSlot>(*(reinterpret_cast<const uint32_t*>(value) + offsetInSrc));
            if constexpr (offsetInSrc + 1 < getNumDwords<Type>())
                setValue<Func, Type, offsetInSrc + 1, dstSlot + 1>(value);
        }

        template <typename Func, uint32_t dstStartSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void setValues(const HeadType* head, const TailTypes*... tails) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Value type of size not multiple of Dword is not supported.");
            setValue<Func, HeadType, 0, dstStartSlot>(head);
            if constexpr (sizeof...(tails) > 0)
                setValues<Func, dstStartSlot + getNumDwords<HeadType>()>(tails...);
        }

        template <uint32_t startSlot, typename HeadType, typename... TailTypes>
        RT_DEVICE_FUNCTION void traceSetPayloads(uint32_t** p, HeadType &headPayload, TailTypes &... tailPayloads) {
            static_assert(sizeof(HeadType) % sizeof(uint32_t) == 0,
                          "Payload type of size not multiple of Dword is not supported.");
            constexpr uint32_t numDwords = getNumDwords<HeadType>();
#pragma unroll
            for (int i = 0; i < numDwords; ++i)
                p[startSlot + i] = reinterpret_cast<uint32_t*>(&headPayload) + i;
            if constexpr (sizeof...(tailPayloads) > 0)
                traceSetPayloads<startSlot + numDwords>(p, tailPayloads...);
        }

        struct PayloadFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION static uint32_t get() {
                if constexpr (index == 0)
                    return optixGetPayload_0();
                if constexpr (index == 1)
                    return optixGetPayload_1();
                if constexpr (index == 2)
                    return optixGetPayload_2();
                if constexpr (index == 3)
                    return optixGetPayload_3();
                if constexpr (index == 4)
                    return optixGetPayload_4();
                if constexpr (index == 5)
                    return optixGetPayload_5();
                if constexpr (index == 6)
                    return optixGetPayload_6();
                if constexpr (index == 7)
                    return optixGetPayload_7();
                return 0;
            }

            template <uint32_t index>
            RT_DEVICE_FUNCTION static void set(uint32_t p) {
                if constexpr (index == 0)
                    optixSetPayload_0(p);
                if constexpr (index == 1)
                    optixSetPayload_1(p);
                if constexpr (index == 2)
                    optixSetPayload_2(p);
                if constexpr (index == 3)
                    optixSetPayload_3(p);
                if constexpr (index == 4)
                    optixSetPayload_4(p);
                if constexpr (index == 5)
                    optixSetPayload_5(p);
                if constexpr (index == 6)
                    optixSetPayload_6(p);
                if constexpr (index == 7)
                    optixSetPayload_7(p);
            }
        };

        struct AttributeFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION static uint32_t get() {
                if constexpr (index == 0)
                    return optixGetAttribute_0();
                if constexpr (index == 1)
                    return optixGetAttribute_1();
                if constexpr (index == 2)
                    return optixGetAttribute_2();
                if constexpr (index == 3)
                    return optixGetAttribute_3();
                if constexpr (index == 4)
                    return optixGetAttribute_4();
                if constexpr (index == 5)
                    return optixGetAttribute_5();
                if constexpr (index == 6)
                    return optixGetAttribute_6();
                if constexpr (index == 7)
                    return optixGetAttribute_7();
                return 0;
            }
        };

        struct ExceptionDetailFunc {
            template <uint32_t index>
            RT_DEVICE_FUNCTION static uint32_t get() {
                if constexpr (index == 0)
                    return optixGetExceptionDetail_0();
                if constexpr (index == 1)
                    return optixGetExceptionDetail_1();
                if constexpr (index == 2)
                    return optixGetExceptionDetail_2();
                if constexpr (index == 3)
                    return optixGetExceptionDetail_3();
                if constexpr (index == 4)
                    return optixGetExceptionDetail_4();
                if constexpr (index == 5)
                    return optixGetExceptionDetail_5();
                if constexpr (index == 6)
                    return optixGetExceptionDetail_6();
                if constexpr (index == 7)
                    return optixGetExceptionDetail_7();
                return 0;
            }
        };
    }

    // JP: 右辺値参照でペイロードを受け取れば右辺値も受け取れて、かつ値の書き換えも反映できる。
    //     が、optixTraceに仕様をあわせることと、テンプレート引数の整合性チェックを簡単にするためただの参照で受け取る。
    // EN: Taking payloads as rvalue reference makes it possible to take rvalue while reflecting value changes.
    //     However take them as normal reference to ease consistency check of template arguments and for
    //     conforming optixTrace.
    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION void trace(OptixTraversableHandle handle,
                                  const float3 &origin, const float3 &direction,
                                  float tmin, float tmax, float rayTime,
                                  OptixVisibilityMask visibilityMask, OptixRayFlags rayFlags,
                                  uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex,
                                  PayloadTypes &... payloads) {
        constexpr size_t numDwords = detail::calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= 8, "Maximum number of payloads is 8 dwords.");

#define OPTIXU_TRACE_ARGUMENTS \
    handle, \
    origin, direction, \
    tmin, tmax, rayTime, \
    visibilityMask, rayFlags, \
    SBToffset, SBTstride, missSBTIndex

        if constexpr (numDwords == 0) {
            optixTrace(OPTIXU_TRACE_ARGUMENTS);
        }
        else {
            uint32_t* p[numDwords];
            detail::traceSetPayloads<0>(p, payloads...);

            if constexpr (numDwords == 1)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0]);
            if constexpr (numDwords == 2)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1]);
            if constexpr (numDwords == 3)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1], *p[2]);
            if constexpr (numDwords == 4)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1], *p[2], *p[3]);
            if constexpr (numDwords == 5)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1], *p[2], *p[3], *p[4]);
            if constexpr (numDwords == 6)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1], *p[2], *p[3], *p[4], *p[5]);
            if constexpr (numDwords == 7)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1], *p[2], *p[3], *p[4], *p[5], *p[6]);
            if constexpr (numDwords == 8)
                optixTrace(OPTIXU_TRACE_ARGUMENTS, *p[0], *p[1], *p[2], *p[3], *p[4], *p[5], *p[6], *p[7]);
        }
#undef OPTIXU_TRACE_ARGUMENTS
    }



    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION void getPayloads(PayloadTypes*... payloads) {
        constexpr size_t numDwords = detail::calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= 8, "Maximum number of payloads is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without payloads has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::PayloadFunc, 0>(payloads...);
    }

    template <typename... PayloadTypes>
    RT_DEVICE_FUNCTION void setPayloads(const PayloadTypes*... payloads) {
        constexpr size_t numDwords = detail::calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= 8, "Maximum number of payloads is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without payloads has no effect.");
        if constexpr (numDwords > 0)
            detail::setValues<detail::PayloadFunc, 0>(payloads...);
    }



    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION void reportIntersection(float hitT, uint32_t hitKind,
                                               const AttributeTypes &... attributes) {
        constexpr size_t numDwords = detail::calcSumDwords<AttributeTypes...>();
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        if constexpr (numDwords == 0) {
            optixReportIntersection(hitT, hitKind);
        }
        else {
            uint32_t a[numDwords];
            detail::packToUInts<0>(a, attributes...);

            if constexpr (numDwords == 1)
                optixReportIntersection(hitT, hitKind, a[0]);
            if constexpr (numDwords == 2)
                optixReportIntersection(hitT, hitKind, a[0], a[1]);
            if constexpr (numDwords == 3)
                optixReportIntersection(hitT, hitKind, a[0], a[1], a[2]);
            if constexpr (numDwords == 4)
                optixReportIntersection(hitT, hitKind, a[0], a[1], a[2], a[3]);
            if constexpr (numDwords == 5)
                optixReportIntersection(hitT, hitKind, a[0], a[1], a[2], a[3], a[4]);
            if constexpr (numDwords == 6)
                optixReportIntersection(hitT, hitKind, a[0], a[1], a[2], a[3], a[4], a[5]);
            if constexpr (numDwords == 7)
                optixReportIntersection(hitT, hitKind, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
            if constexpr (numDwords == 8)
                optixReportIntersection(hitT, hitKind, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        }
    }

    template <typename... AttributeTypes>
    RT_DEVICE_FUNCTION void getAttributes(AttributeTypes*... attributes) {
        constexpr size_t numDwords = detail::calcSumDwords<AttributeTypes...>();
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without attributes has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::AttributeFunc, 0>(attributes...);
    }



    template <typename... ExceptionDetailTypes>
    RT_DEVICE_FUNCTION void throwException(int32_t exceptionCode,
                                           const ExceptionDetailTypes &... exceptionDetails) {
        constexpr size_t numDwords = detail::calcSumDwords<ExceptionDetailTypes...>();
        static_assert(numDwords <= 8, "Maximum number of exception details is 8 dwords.");
        if constexpr (numDwords == 0) {
            optixThrowException(exceptionCode);
        }
        else {
            uint32_t ed[numDwords];
            detail::packToUInts<0>(ed, exceptionDetails...);

            if constexpr (numDwords == 1)
                optixThrowException(exceptionCode, ed[0]);
            if constexpr (numDwords == 2)
                optixThrowException(exceptionCode, ed[0], ed[1]);
            if constexpr (numDwords == 3)
                optixThrowException(exceptionCode, ed[0], ed[1], ed[2]);
            if constexpr (numDwords == 4)
                optixThrowException(exceptionCode, ed[0], ed[1], ed[2], ed[3]);
            if constexpr (numDwords == 5)
                optixThrowException(exceptionCode, ed[0], ed[1], ed[2], ed[3], ed[4]);
            if constexpr (numDwords == 6)
                optixThrowException(exceptionCode, ed[0], ed[1], ed[2], ed[3], ed[4], ed[5]);
            if constexpr (numDwords == 7)
                optixThrowException(exceptionCode, ed[0], ed[1], ed[2], ed[3], ed[4], ed[5], ed[6]);
            if constexpr (numDwords == 8)
                optixThrowException(exceptionCode, ed[0], ed[1], ed[2], ed[3], ed[4], ed[5], ed[6], ed[7]);
        }
    }

    template <typename... ExceptionDetailTypes>
    RT_DEVICE_FUNCTION void getExceptionDetails(ExceptionDetailTypes*... details) {
        constexpr size_t numDwords = detail::calcSumDwords<ExceptionDetailTypes...>();
        static_assert(numDwords <= 8, "Maximum number of exception details is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without exception details has no effect.");
        if constexpr (numDwords > 0)
            detail::getValues<detail::ExceptionDetailFunc, 0>(details...);
    }

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    // END: Device-side function wrappers
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ホスト側API
    // EN: Host-side API.
#if !defined(__CUDA_ARCH__)
    /*

    Context --+-- Pipeline --+-- Module
              |              |
              |              +-- ProgramGroup
              |
              +-- Material
              |
              |
              |
              +-- Scene    --+-- IAS
              |              |
              |              +-- Instance
              |              |
              |              +-- Transform
              |              |
              |              +-- GAS
              |              |
              |              +-- GeomInst
              |
              +-- Denoiser

    JP: 
    EN: 

    */

    class Context;
    class Material;
    class Scene;
    class GeometryInstance;
    class GeometryAccelerationStructure;
    class Transform;
    class Instance;
    class InstanceAccelerationStructure;
    class Pipeline;
    class Module;
    class ProgramGroup;
    class Denoiser;

    enum class ASTradeoff {
        Default = 0,
        PreferFastTrace,
        PreferFastBuild,
    };

    enum class TransformType {
        MatrixMotion = 0,
        SRTMotion,
        Static,
        Invalid
    };

    class BufferView {
        CUdeviceptr m_devicePtr;
        size_t m_numElements;
        uint32_t m_stride;

    public:
        BufferView() :
            m_devicePtr(0),
            m_numElements(0), m_stride(0) {}
        BufferView(CUdeviceptr devicePtr, size_t numElements, uint32_t stride) :
            m_devicePtr(devicePtr),
            m_numElements(numElements), m_stride(stride) {}

        CUdeviceptr getCUdeviceptr() const { return m_devicePtr; }
        size_t numElements() const { return m_numElements; }
        uint32_t stride() const { return m_stride; }
        size_t sizeInBytes() const { return m_numElements * m_stride; }

        bool isValid() const {
            return m_devicePtr != 0;
        }
    };



#define OPTIXU_PIMPL() \
public: \
    class Priv; \
private: \
    Priv* m = nullptr

#define OPTIXU_COMMON_FUNCTIONS(SelfType) \
    operator bool() const { return m; } \
    bool operator==(const SelfType &r) const { return m == r.m; } \
    bool operator!=(const SelfType &r) const { return m != r.m; } \
    bool operator<(const SelfType &r) const { \
        static_assert(std::is_same<decltype(r), decltype(*this)>::value, \
                      "This function can be defined only for the self type."); \
        return m < r.m; \
    }



    class Context {
        OPTIXU_PIMPL();

    public:
        static Context create(CUcontext cudaContext, bool enableValidation = false);
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Context);

        CUcontext getCUcontext() const;

        [[nodiscard]]
        Pipeline createPipeline() const;
        [[nodiscard]]
        Material createMaterial() const;
        [[nodiscard]]
        Scene createScene() const;
        [[nodiscard]]
        Denoiser createDenoiser(OptixDenoiserInputKind inputKind) const;
    };



    class Material {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Material);

        // JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
        //     パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
        // EN: Updating a shader binding table is required when calling the following APIs.
        //     Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
        void setHitGroup(uint32_t rayType, ProgramGroup hitGroup);
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }
    };



    class Scene {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Scene);

        [[nodiscard]]
        GeometryInstance createGeometryInstance(bool forCustomPrimitives = false) const;
        [[nodiscard]]
        GeometryAccelerationStructure createGeometryAccelerationStructure(bool forCustomPrimitives = false) const;
        [[nodiscard]]
        Transform createTransform() const;
        [[nodiscard]]
        Instance createInstance() const;
        [[nodiscard]]
        InstanceAccelerationStructure createInstanceAccelerationStructure() const;

        void generateShaderBindingTableLayout(size_t* memorySize) const;
    };



    class GeometryInstance {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(GeometryInstance);

        // JP: 以下のAPIを呼んだ場合は所属するGASのmarkDirty()を呼ぶ必要がある。
        //     (頂点/AABBバッファーの変更のみの場合は、markDirty()を呼ばずにGASのアップデートだけでも良い。)
        // EN: Calling markDirty() of a GAS to which the geometry instance belongs is
        //     required when calling the following APIs.
        //     (It is okay to use update instead of calling markDirty() when changing only vertex/AABB buffer.)
        void setNumMotionSteps(uint32_t n) const;
        void setVertexFormat(OptixVertexFormat format) const;
        void setVertexBuffer(const BufferView &vertexBuffer, uint32_t motionStep = 0) const;
        void setTriangleBuffer(const BufferView &triangleBuffer, OptixIndicesFormat format = OPTIX_INDICES_FORMAT_UNSIGNED_INT3) const;
        void setCustomPrimitiveAABBBuffer(const BufferView &primitiveAABBBuffer, uint32_t motionStep = 0) const;
        void setPrimitiveIndexOffset(uint32_t offset) const;
        void setNumMaterials(uint32_t numMaterials, const BufferView &matIndexOffsetBuffer, uint32_t indexOffsetSize = sizeof(uint32_t)) const;
        void setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const;

        // JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
        //     パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
        // EN: Updating a shader binding table is required when calling the following APIs.
        //     Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
        void setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const;
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }
    };



    class GeometryAccelerationStructure {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(GeometryAccelerationStructure);

        // JP: 以下のAPIを呼んだ場合はGASがdirty状態になる。
        // EN: Calling the following APIs marks the GAS dirty.
        void setConfiguration(ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction, bool allowRandomVertexAccess) const;
        void setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void addChild(GeometryInstance geomInst, CUdeviceptr preTransform = 0) const;
        void removeChild(GeometryInstance geomInst, CUdeviceptr preTransform = 0) const;
        void markDirty() const;

        // JP: 以下のAPIを呼んだ場合はヒットグループのシェーダーバインディングテーブルレイアウトが無効化される。
        // EN: Calling the following APIs invalidate the shader binding table layout of hit group.
        void setNumMaterialSets(uint32_t numMatSets) const;
        void setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const;

        // JP: リビルド・コンパクトを行った場合はこのGASが(間接的に)所属するTraversable (例: IAS)
        //     のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which this GAS (indirectly) belongs
        //     is required when performing rebuild / compact.
        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const;
        OptixTraversableHandle rebuild(CUstream stream, const BufferView &accelBuffer, const BufferView &scratchBuffer) const;
        // JP: リビルドが完了するのをホスト側で待つ。
        // EN: Wait on the host until rebuild operation finishes.
        void prepareForCompact(size_t* compactedAccelBufferSize) const;
        OptixTraversableHandle compact(CUstream stream, const BufferView &compactedAccelBuffer) const;
        // JP: コンパクトが完了するのをホスト側で待つ。
        // EN: Wait on the host until compact operation finishes.
        void removeUncompacted() const;

        // JP: アップデートを行った場合はこのGASが(間接的に)所属するTraversable (例: IAS)
        //     もアップデートもしくはリビルドする必要がある。
        // EN: Updating or rebuilding a traversable (e.g. IAS) to which this GAS (indirectly) belongs
        //     is required when performing update.
        void update(CUstream stream, const BufferView &scratchBuffer) const;

        // JP: 以下のAPIを呼んだ場合はシェーダーバインディングテーブルを更新する必要がある。
        //     パイプラインのmarkHitGroupShaderBindingTableDirty()を呼べばローンチ時にセットアップされる。
        // EN: Updating a shader binding table is required when calling the following APIs.
        //     Calling pipeline's markHitGroupShaderBindingTableDirty() triggers re-setup of the table at launch.
        void setUserData(const void* data, uint32_t size, uint32_t alignment) const;
        template <typename T>
        void setUserData(const T &data) const {
            setUserData(&data, sizeof(T), alignof(T));
        }

        bool isReady() const;
        OptixTraversableHandle getHandle() const;
    };



    class Transform {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Transform);

        // JP: 以下のAPIを呼んだ場合はTransformがdirty状態になる。
        // EN: Calling the following APIs marks the transform dirty.
        void setConfiguration(TransformType type, uint32_t numKeys,
                              size_t* transformSize);
        void setMotionOptions(float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void setMatrixMotionKey(uint32_t keyIdx, const float matrix[12]) const;
        void setSRTMotionKey(uint32_t keyIdx, const float scale[3], const float orientation[4], const float translation[3]) const;
        void setStaticTransform(const float matrix[12]) const;
        void setChild(GeometryAccelerationStructure child) const;
        void setChild(InstanceAccelerationStructure child) const;
        void setChild(Transform child) const;
        void markDirty() const;

        // JP: (間接的に)所属するTraversable (例: IAS)のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which the transform
        //     (indirectly) belongs is required.
        OptixTraversableHandle rebuild(CUstream stream, const BufferView &trDeviceMem);

        bool isReady() const;
        OptixTraversableHandle getHandle() const;
    };



    class Instance {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Instance);

        // JP: 所属するIASのmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a IAS to which the instance belongs is required.
        void setChild(GeometryAccelerationStructure child, uint32_t matSetIdx = 0) const;
        void setChild(InstanceAccelerationStructure child) const;
        void setChild(Transform child, uint32_t matSetIdx = 0) const;

        // JP: 所属するIASをリビルドもしくはアップデートする必要がある。
        // EN: Rebulding or Updating of a IAS to which the instance belongs is required.
        void setID(uint32_t value) const;
        void setVisibilityMask(uint32_t mask) const;
        void setFlags(OptixInstanceFlags flags) const;
        void setTransform(const float transform[12]) const;
        void setMaterialSetIndex(uint32_t matSetIdx) const;
    };



    // TODO: インスタンスバッファーもユーザー管理にしたいため、rebuild()が今の形になっているが微妙かもしれない。
    //       インスタンスバッファーを内部で1つ持つようにすると、
    //       あるフレームでIASをビルド、次のフレームでインスタンスの追加がありリビルドの必要が生じた場合に
    //       1フレーム目のGPU処理の終了を待たないと危険という状況になってしまう。
    //       OptiX的にはASのビルド完了後にはインスタンスバッファーは不要となるが、
    //       アップデート処理はリビルド時に書かれたインスタンスバッファーの内容を期待しているため、
    //       基本的にインスタンスバッファーとASのメモリ(コンパクション版にもなり得る)は同じ寿命で扱ったほうが良さそう。
    class InstanceAccelerationStructure {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(InstanceAccelerationStructure);

        // JP: 以下のAPIを呼んだ場合はIASがdirty状態になる。
        // EN: Calling the following APIs marks the IAS dirty.
        void setConfiguration(ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction) const;
        void setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const;
        void addChild(Instance instance) const;
        void removeChild(Instance instance) const;
        void markDirty() const;

        // JP: リビルド・コンパクトを行った場合はこのIASが(間接的に)所属するTraversable (例: IAS)
        //     のmarkDirty()を呼ぶ必要がある。
        // EN: Calling markDirty() of a traversable (e.g. IAS) to which this IAS (indirectly) belongs
        //     is required when performing rebuild / compact.
        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement, uint32_t* numInstances) const;
        OptixTraversableHandle rebuild(CUstream stream, const BufferView &instanceBuffer,
                                       const BufferView &accelBuffer, const BufferView &scratchBuffer) const;
        // JP: リビルドが完了するのをホスト側で待つ。
        // EN: Wait on the host until rebuild operation finishes.
        void prepareForCompact(size_t* compactedAccelBufferSize) const;
        OptixTraversableHandle compact(CUstream stream, const BufferView &compactedAccelBuffer) const;
        // JP: コンパクトが完了するのをホスト側で待つ。
        // EN: Wait on the host until compact operation finishes.
        void removeUncompacted() const;

        // JP: アップデートを行った場合はこのIASが(間接的に)所属するTraversable (例: IAS)
        //     もアップデートもしくはリビルドする必要がある。
        // EN: Updating or rebuilding a traversable (e.g. IAS) to which this IAS (indirectly) belongs
        //     is required when performing update.
        void update(CUstream stream, const BufferView &scratchBuffer) const;

        bool isReady() const;
        OptixTraversableHandle getHandle() const;
    };



    class Pipeline {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Pipeline);

        void setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues,
                                const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                bool useMotionBlur,
                                OptixTraversableGraphFlags traversableGraphFlags,
                                OptixExceptionFlags exceptionFlags,
                                OptixPrimitiveTypeFlags supportedPrimitiveTypeFlags) const;

        [[nodiscard]]
        Module createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount,
                                         OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel,
                                         OptixModuleCompileBoundValueEntry* boundValues = nullptr, uint32_t numBoundValues = 0) const;

        [[nodiscard]]
        ProgramGroup createRayGenProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        ProgramGroup createExceptionProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        ProgramGroup createMissProgram(Module module, const char* entryFunctionName) const;
        [[nodiscard]]
        ProgramGroup createHitProgramGroup(Module module_CH, const char* entryFunctionNameCH,
                                           Module module_AH, const char* entryFunctionNameAH,
                                           Module module_IS, const char* entryFunctionNameIS) const;
        [[nodiscard]]
        ProgramGroup createCallableProgramGroup(Module module_DC, const char* entryFunctionNameDC,
                                                Module module_CC, const char* entryFunctionNameCC) const;

        void link(uint32_t maxTraceDepth, OptixCompileDebugLevel debugLevel) const;

        // JP: 以下のAPIを呼んだ場合は(非ヒットグループの)シェーダーバインディングテーブルレイアウトが無効化される。
        // EN: Calling the following APIs invalidate the (non-hit group) shader binding table layout.
        void setNumMissRayTypes(uint32_t numMissRayTypes) const;
        void setNumCallablePrograms(uint32_t numCallablePrograms) const;

        void generateShaderBindingTableLayout(size_t* memorySize) const;

        // JP: 以下のAPIを呼んだ場合は(非ヒットグループの)シェーダーバインディングテーブルがdirty状態になり
        //     ローンチ時に再セットアップされる。
        //     ただしローンチ時のセットアップはSBTバッファーの内容変更・転送を伴うので、
        //     非同期書き換えを行う場合は安全のためにはSBTバッファーをダブルバッファリングする必要がある。
        // EN: Calling the following API marks the (non-hit group) shader binding table dirty
        //     then triggers re-setup of the table at launch.
        //     However note that the setup in the launch involves the change of the SBT buffer's contents
        //     and transfer, so double buffered SBT is required for safety
        //     in the case performing asynchronous update.
        void setRayGenerationProgram(ProgramGroup program) const;
        void setExceptionProgram(ProgramGroup program) const;
        void setMissProgram(uint32_t rayType, ProgramGroup program) const;
        void setCallableProgram(uint32_t index, ProgramGroup program) const;
        void setShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const;

        // JP: 以下のAPIを呼んだ場合はヒットグループのシェーダーバインディングテーブルがdirty状態になり
        //     ローンチ時に再セットアップされる。
        //     ただしローンチ時のセットアップはSBTバッファーの内容変更・転送を伴うので、
        //     非同期書き換えを行う場合は安全のためにはSBTバッファーをダブルバッファリングする必要がある。
        // EN: Calling the following APIs marks the hit group's shader binding table dirty,
        //     then triggers re-setup of the table at launch.
        //     However note that the setup in the launch involves the change of the SBT buffer's contents
        //     and transfer, so double buffered SBT is required for safety
        //     in the case performing asynchronous update.
        void setScene(const Scene &scene) const;
        void setHitGroupShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const;
        void markHitGroupShaderBindingTableDirty() const;

        void setStackSize(uint32_t directCallableStackSizeFromTraversal,
                          uint32_t directCallableStackSizeFromState,
                          uint32_t continuationStackSize,
                          uint32_t maxTraversableGraphDepth) const;

        // JP: セットされたシーンを基にシェーダーバインディングテーブルのセットアップを行い、
        //     Ray Generationシェーダーを起動する。
        // EN: Setup the shader binding table based on the scene set, then launch the ray generation shader.
        void launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const;
    };



    // JP: Moduleの寿命はそれを参照するあらゆるProgramGroupの寿命よりも長い必要がある。
    // EN: The lifetime of a module must extend to the lifetime of any ProgramGroup that reference that module.
    class Module {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Module);
    };



    class ProgramGroup {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(ProgramGroup);

        void getStackSize(OptixStackSizes* sizes) const;
    };



    class DenoisingTask {
        uint32_t placeHolder[6];

        // TODO: ? implement a function to query required window (tile + overlap).
    };
    
    class Denoiser {
        OPTIXU_PIMPL();

    public:
        void destroy();
        OPTIXU_COMMON_FUNCTIONS(Denoiser);

        void setModel(OptixDenoiserModelKind kind, void* data, size_t sizeInBytes) const;
        void prepare(uint32_t imageWidth, uint32_t imageHeight, uint32_t tileWidth, uint32_t tileHeight,
                     size_t* stateBufferSize, size_t* scratchBufferSize, size_t* scratchBufferSizeForComputeIntensity,
                     uint32_t* numTasks) const;
        void getTasks(DenoisingTask* tasks) const;
        void setLayers(const BufferView &color, const BufferView &albedo, const BufferView &normal, const BufferView &denoisedColor,
                       OptixPixelFormat colorFormat, OptixPixelFormat albedoFormat, OptixPixelFormat normalFormat) const;
        void setupState(CUstream stream, const BufferView &stateBuffer, const BufferView &scratchBuffer) const;

        void computeIntensity(CUstream stream, const BufferView &scratchBuffer, CUdeviceptr outputIntensity);
        void invoke(CUstream stream, bool denoiseAlpha, CUdeviceptr hdrIntensity, float blendFactor,
                    const DenoisingTask &task);
    };



#undef OPTIXU_COMMON_FUNCTIONS
#undef OPTIXU_PIMPL

#endif // #if !defined(__CUDA_ARCH__)
    // END: Host-side API.
    // ----------------------------------------------------------------
} // namespace optixu
