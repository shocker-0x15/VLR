#pragma once

/*

JP: 現状ではあらゆるAPIに破壊的変更が入る可能性が非常に高い。
EN: It is very likely for now that any API will have breaking changes.

----------------------------------------------------------------
TODO:
- スタックサイズ調整。
- HitGroup以外のプログラムの非同期更新。
- HitGroup以外のProgramGroupにユーザーデータを持たせる。
- 途中で各オブジェクトのパラメターを変更した際の処理。
  パイプラインのセットアップ順などが現状は暗黙的に固定されている。これを自由な順番で変えられるようにする。
- Assertとexceptionの整理。
- GAS/IASに関してユーザーが気にするところはAS云々ではなくグループ化なので
  名前を変えるべき？GeometryGroup/InstanceGroupのような感じ。

----------------------------------------------------------------
- GASの構築と更新
  - ジオメトリの変形
    - シングルバッファリング
      GeomInstに登録済みの頂点バッファー(+インデックスバッファー)の情報を更新してGASのupdate()を呼ぶ。
      要素数・フォーマットは変更しない。
      OptiXカーネル実行中にCPUから内容を更新するのは危険。
    - マルチバッファリング
      GeomInstに登録済みの頂点バッファー(+インデックスバッファー)と同じ要素数、
      同じフォーマットのバッファーを新たに登録してGASのupdate()を呼ぶ。
  - GeomInstの追加・削除
    prepareForBuild()を呼びメモリ要件を取得、GAS用のメモリを確保してrebuild()を呼ぶ。
    すでに確保済みのメモリを使用する場合、GASを使用しているOptiXカーネル実行中に、他のCUDA streamからrebuild()を呼ぶのは危険。
- IASの構築と更新
  - インスタンスの変形
    - Instanceのトランスフォームを更新してIASのupdate()を呼ぶ。
  - インスタンスの追加・削除
    prepareForBuild()を呼びメモリ要件を取得、インスタンスバッファーとIAS用のメモリを確保してrebuild()を呼ぶ。
    すでに確保済みのメモリを使用する場合、IASを使用しているOptiXカーネル実行中に、他のCUDA streamからrebuild()を呼ぶのは危険。
- SBTの更新
  - マテリアルの更新
    マテリアルには32bitの情報しか記録できないようにしているため、
    典型的にはユーザーが用意したマテリアル情報本体を格納したバッファーのインデックスとして使用することを期待している。
    そのためマテリアルの変化はユーザーの管理する世界の中で起きることを想定している。
    が、バッファーのインデックス自体を変えるケースも考えうる。
    その場合にはSBT自体をユーザーがダブルバッファリングなどして非同期に更新することを想定している。
  - プログラムグループの更新
    SBT中のレコードヘッダー、つまりプログラムグループを書き換えることは頻繁には起こらないと想定している。
    が、可能性としてはゼロではない。
    その場合にはSBT自体をユーザーがダブルバッファリングなどして非同期に更新することを想定している。

AS/SBT Layoutのdirty状態はUtil側で検知できるdirty状態をカーネルローンチ時に検出したらエラーを出してくれるだけのもの。
リビルド・アップデート・レイアウト生成などはユーザーが行う必要がある。
さらにUtil側で検知できないdirty状態はユーザーが意識する必要がある。

*/

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define OPTIX_Platform_Windows
#    if defined(_MSC_VER)
#        define OPTIX_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define OPTIX_Platform_macOS
#endif

#include <optix.h>
#include <cuda.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__)
#include <optix_stubs.h>
#include "cuda_util.h"
#endif

#if defined(__CUDA_ARCH__)
#   define RT_FUNCTION __forceinline__ __device__
#   define RT_PROGRAM extern "C" __global__
#   define RT_CALLABLE_PROGRAM extern "C" __device__
#else
#   define RT_FUNCTION
#   define RT_PROGRAM
#   define RT_CALLABLE_PROGRAM
#endif



namespace optixu {
#if !defined(__CUDA_ARCH__)
    using namespace cudau;
#endif

#ifdef _DEBUG
#   define OPTIX_ENABLE_ASSERT
#endif

#if defined(OPTIX_Platform_Windows_MSVC)
    void devPrintf(const char* fmt, ...);
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if 1
#   define optixPrintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define optixPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#if defined(OPTIX_ENABLE_ASSERT)
#   if defined(__CUDA_ARCH__)
#   define optixAssert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); } 0
#   else
#   define optixAssert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#   endif
#else
#   define optixAssert(expr, fmt, ...)
#endif

#define optixAssert_ShouldNotBeCalled() optixAssert(false, "Should not be called!")
#define optixAssert_NotImplemented() optixAssert(false, "Not implemented yet!")

    template <typename T>
    RT_FUNCTION constexpr bool false_T() { return false; }



    struct HitGroupSBTRecordData {
        uint32_t materialData;
        uint32_t geomInstData;
    };

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
    RT_FUNCTION HitGroupSBTRecordData getHitGroupSBTRecordData() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
#endif



    template <typename FuncType>
    class DirectCallableProgramID;

    template <typename ReturnType, typename... ArgTypes>
    class DirectCallableProgramID<ReturnType(ArgTypes...)> {
        uint32_t m_sbtIndex;

    public:
        RT_FUNCTION DirectCallableProgramID() {}
        RT_FUNCTION explicit DirectCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_FUNCTION explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
        RT_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
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
        RT_FUNCTION ContinuationCallableProgramID() {}
        RT_FUNCTION explicit ContinuationCallableProgramID(uint32_t sbtIndex) : m_sbtIndex(sbtIndex) {}
        RT_FUNCTION explicit operator uint32_t() const { return m_sbtIndex; }

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
        RT_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
            return optixContinuationCall<ReturnType, ArgTypes...>(m_sbtIndex, args...);
        }
#endif
    };



    template <typename T>
    class NativeBlockBuffer2D {
        CUsurfObject m_surfObject;

    public:
        NativeBlockBuffer2D() : m_surfObject(0) {}
        NativeBlockBuffer2D(CUsurfObject surfObject) : m_surfObject(surfObject) {};

        NativeBlockBuffer2D &operator=(CUsurfObject surfObject) {
            m_surfObject = surfObject;
            return *this;
        }

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
        RT_FUNCTION T operator[](uint2 idx) const {
            return surf2Dread<T>(m_surfObject, idx.x * sizeof(T), idx.y);
        }
        RT_FUNCTION void write(uint2 idx, const T &value) {
            surf2Dwrite(value, m_surfObject, idx.x * sizeof(T), idx.y);
        }
#endif
    };


    
    template <typename T, uint32_t log2BlockWidth>
    class BlockBuffer2D {
        T* m_rawBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numXBlocks;

        RT_FUNCTION constexpr uint32_t calcLinearIndex(uint2 idx) const {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t blockIdxX = idx.x >> log2BlockWidth;
            uint32_t blockIdxY = idx.y >> log2BlockWidth;
            uint32_t blockOffset = (blockIdxY * m_numXBlocks + blockIdxX) * (blockWidth * blockWidth);
            uint32_t idxXInBlock = idx.x & mask;
            uint32_t idxYInBlock = idx.y & mask;
            uint32_t linearIndexInBlock = idxYInBlock * blockWidth + idxXInBlock;
            return blockOffset + linearIndexInBlock;
        }

    public:
        RT_FUNCTION BlockBuffer2D() {}
        RT_FUNCTION BlockBuffer2D(T* rawBuffer, uint32_t width, uint32_t height) :
        m_rawBuffer(rawBuffer), m_width(width), m_height(height) {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            m_numXBlocks = ((width + mask) & ~mask) >> log2BlockWidth;
        }

        RT_FUNCTION const T &operator[](uint2 idx) const {
            return m_rawBuffer[calcLinearIndex(idx)];
        }
        RT_FUNCTION T &operator[](uint2 idx) {
            return m_rawBuffer[calcLinearIndex(idx)];
        }
    };
    
    
    
#if !defined(__CUDA_ARCH__)
    template <typename T, uint32_t log2BlockWidth>
    class HostBlockBuffer2D {
        TypedBuffer<T> m_rawBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numXBlocks;
        T* m_mappedPointer;

        constexpr uint32_t calcLinearIndex(uint2 idx) const {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t blockIdxX = idx.x >> log2BlockWidth;
            uint32_t blockIdxY = idx.y >> log2BlockWidth;
            uint32_t blockOffset = (blockIdxY * m_numXBlocks + blockIdxX) * (blockWidth * blockWidth);
            uint32_t idxXInBlock = idx.x & mask;
            uint32_t idxYInBlock = idx.y & mask;
            uint32_t linearIndexInBlock = idxYInBlock * blockWidth + idxXInBlock;
            return blockOffset + linearIndexInBlock;
        }

    public:
        HostBlockBuffer2D() : m_mappedPointer(nullptr) {}
        HostBlockBuffer2D(HostBlockBuffer2D &&b) {
            m_width = b.m_width;
            m_height = b.m_height;
            m_numXBlocks = b.m_numXBlocks;
            m_mappedPointer = b.m_mappedPointer;
            m_rawBuffer = std::move(b);
        }
        HostBlockBuffer2D &operator=(HostBlockBuffer2D &&b) {
            m_rawBuffer.finalize();

            m_width = b.m_width;
            m_height = b.m_height;
            m_numXBlocks = b.m_numXBlocks;
            m_mappedPointer = b.m_mappedPointer;
            m_rawBuffer = std::move(b.m_rawBuffer);

            return *this;
        }

        void initialize(CUcontext context, BufferType type, uint32_t width, uint32_t height) {
            m_width = width;
            m_height = height;
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            m_numXBlocks = ((width + mask) & ~mask) >> log2BlockWidth;
            uint32_t numYBlocks = ((height + mask) & ~mask) >> log2BlockWidth;
            uint32_t numElements = numYBlocks * m_numXBlocks * blockWidth * blockWidth;
            m_rawBuffer.initialize(context, type, numElements);
        }
        void finalize() {
            m_rawBuffer.finalize();
        }

        void resize(uint32_t width, uint32_t height) {
            if (!m_rawBuffer.isInitialized())
                throw std::runtime_error("Buffer is not initialized.");

            if (m_width == width && m_height == height)
                return;

            HostBlockBuffer2D newBuffer;
            newBuffer.initialize(m_rawBuffer.getCUcontext(), m_rawBuffer.getBufferType(), width, height);

            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t numSrcYBlocks = ((m_height + mask) & ~mask) >> log2BlockWidth;
            uint32_t numDstYBlocks = ((height + mask) & ~mask) >> log2BlockWidth;
            uint32_t numXBlocksToCopy = std::min(m_numXBlocks, newBuffer.m_numXBlocks);
            uint32_t numYBlocksToCopy = std::min(numSrcYBlocks, numDstYBlocks);
            if (numXBlocksToCopy == m_numXBlocks) {
                size_t numBytesToCopy = (numXBlocksToCopy * numYBlocksToCopy * blockWidth * blockWidth) * sizeof(T);
                CUDADRV_CHECK(cuMemcpyDtoD(newBuffer.m_rawBuffer.getCUdeviceptr(),
                                           m_rawBuffer.getCUdeviceptr(),
                                           numBytesToCopy));
            }
            else {
                for (int yb = 0; yb < numYBlocksToCopy; ++yb) {
                    size_t srcOffset = (m_numXBlocks * blockWidth * blockWidth * yb) * sizeof(T);
                    size_t dstOffset = (newBuffer.m_numXBlocks * blockWidth * blockWidth * yb) * sizeof(T);
                    size_t numBytesToCopy = (numXBlocksToCopy * blockWidth * blockWidth) * sizeof(T);
                    CUDADRV_CHECK(cuMemcpyDtoD(newBuffer.m_rawBuffer.getCUdeviceptr() + dstOffset,
                                               m_rawBuffer.getCUdeviceptr() + srcOffset,
                                               numBytesToCopy));
                }
            }

            *this = std::move(newBuffer);
        }

        CUcontext getCUcontext() const {
            return m_rawBuffer.getCUcontext();
        }
        BufferType getBufferType() const {
            return m_rawBuffer.getBufferType();
        }

        CUdeviceptr getCUdeviceptr() const {
            return m_rawBuffer.getCUdeviceptr();
        }

        void map() {
            m_mappedPointer = reinterpret_cast<T*>(m_rawBuffer.map());
        }
        void unmap() {
            m_rawBuffer.unmap();
            m_mappedPointer = nullptr;
        }
        const T &operator[](uint2 idx) const {
            return m_mappedPointer[calcLinearIndex(idx)];
        }
        T &operator[](uint2 idx) {
            return m_mappedPointer[calcLinearIndex(idx)];
        }

        BlockBuffer2D<T, log2BlockWidth> getBlockBuffer2D() const {
            return BlockBuffer2D<T, log2BlockWidth>(m_rawBuffer.getDevicePointer(), m_width, m_height);
        }
    };
#endif



    // Device-side function wrappers
#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
    template <typename T>
    RT_FUNCTION constexpr size_t __calcSumDwords() {
        return sizeof(T) / 4;
    }
    
    template <typename HeadType0, typename... TailTypes>
    RT_FUNCTION constexpr size_t __calcSumDwords() {
        return sizeof(HeadType0) / 4 + _calcSumDwords<TailTypes...>();
    }

    template <typename... PayloadTypes>
    RT_FUNCTION constexpr size_t _calcSumDwords() {
        if constexpr (sizeof...(PayloadTypes) > 0)
            return __calcSumDwords<PayloadTypes...>();
        else
            return 0;
    }



    template <uint32_t start, typename HeadType, typename... TailTypes>
    RT_FUNCTION void _traceSetPayloads(uint32_t** p, HeadType &headPayload, TailTypes &... tailPayloads) {
        constexpr uint32_t numDwords = sizeof(HeadType) / 4;
#pragma unroll
        for (int i = 0; i < numDwords; ++i)
            p[start + i] = reinterpret_cast<uint32_t*>(&headPayload) + i;
        if constexpr (sizeof...(tailPayloads) > 0)
            _traceSetPayloads<start + numDwords>(p, tailPayloads...);
    }

#define OPTIXU_TRACE_PARAMETERS \
    OptixTraversableHandle handle, \
    const float3 &origin, const float3 &direction, \
    float tmin, float tmax, float rayTime, \
    OptixVisibilityMask visibilityMask, uint32_t rayFlags, \
    uint32_t SBToffset, uint32_t SBTstride, uint32_t missSBTIndex
#define OPTIXU_TRACE_ARGUMENTS \
    handle, \
    origin, direction, \
    tmin, tmax, rayTime, \
    visibilityMask, rayFlags, \
    SBToffset, SBTstride, missSBTIndex

    template <uint32_t numDwords, typename... PayloadTypes>
    RT_FUNCTION void _trace(OPTIXU_TRACE_PARAMETERS, PayloadTypes &... payloads) {
        uint32_t* p[numDwords];
        if constexpr (numDwords > 0)
            _traceSetPayloads<0>(p, payloads...);

        if constexpr (numDwords == 0)
            optixTrace(OPTIXU_TRACE_ARGUMENTS);
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
    
    // JP: 右辺値参照でペイロードを受け取れば右辺値も受け取れて、かつ値の書き換えも反映できる。
    //     が、optixTraceに仕様をあわせることと、テンプレート引数の整合性チェックを簡単にするためただの参照で受け取る。
    // EN: 
    template <typename... PayloadTypes>
    RT_FUNCTION void trace(OPTIXU_TRACE_PARAMETERS, PayloadTypes &... payloads) {
        constexpr size_t numDwords = _calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= 8, "Maximum number of payloads is 8 dwords.");
        _trace<numDwords>(OPTIXU_TRACE_ARGUMENTS, payloads...);
    }



    template <uint32_t index>
    RT_FUNCTION uint32_t _optixGetPayload() {
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

    template <typename PayloadType, uint32_t offset, uint32_t start>
    RT_FUNCTION void _getPayload(PayloadType* payload) {
        if (!payload)
            return;
        constexpr uint32_t numDwords = sizeof(PayloadType) / 4;
        *(reinterpret_cast<uint32_t*>(payload) + offset) = _optixGetPayload<start>();
        if constexpr (offset + 1 < numDwords)
            _getPayload<PayloadType, offset + 1, start + 1>(payload);
    }

    template <uint32_t start, typename HeadType, typename... TailTypes>
    RT_FUNCTION void _getPayloads(HeadType* headPayload, TailTypes*... tailPayloads) {
        _getPayload<HeadType, 0, start>(headPayload);
        if constexpr (sizeof...(tailPayloads) > 0)
            _getPayloads<start + sizeof(HeadType) / 4>(tailPayloads...);
    }

    template <typename... PayloadTypes>
    RT_FUNCTION void getPayloads(PayloadTypes*... payloads) {
        constexpr size_t numDwords = _calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= 8, "Maximum number of payloads is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without payloads has no effect.");
        if constexpr (numDwords > 0)
            _getPayloads<0>(payloads...);
    }



    template <uint32_t index>
    RT_FUNCTION void _optixSetPayload(uint32_t p) {
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

    template <typename PayloadType, uint32_t offset, uint32_t start>
    RT_FUNCTION void _setPayload(const PayloadType* payload) {
        if (!payload)
            return;
        constexpr uint32_t numDwords = sizeof(PayloadType) / 4;
        _optixSetPayload<start>(*(reinterpret_cast<const uint32_t*>(payload) + offset));
        if constexpr (offset + 1 < numDwords)
            _setPayload<PayloadType, offset + 1, start + 1>(payload);
    }

    template <uint32_t start, typename HeadType, typename... TailTypes>
    RT_FUNCTION void _setPayloads(const HeadType* headPayload, const TailTypes*... tailPayloads) {
        _setPayload<HeadType, 0, start>(headPayload);
        if constexpr (sizeof...(tailPayloads) > 0)
            _setPayloads<start + sizeof(HeadType) / 4>(tailPayloads...);
    }

    template <typename... PayloadTypes>
    RT_FUNCTION void setPayloads(PayloadTypes*... payloads) {
        constexpr size_t numDwords = _calcSumDwords<PayloadTypes...>();
        static_assert(numDwords <= 8, "Maximum number of payloads is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without payloads has no effect.");
        if constexpr (numDwords > 0)
            _setPayloads<0>(payloads...);
    }



    template <uint32_t start, typename HeadType, typename... TailTypes>
    RT_FUNCTION void _setAttributes(uint32_t* a, const HeadType &headAttribute, const TailTypes &... tailAttributes) {
        constexpr uint32_t numDwords = sizeof(HeadType) / 4;
#pragma unroll
        for (int i = 0; i < numDwords; ++i)
            a[start + i] = *(reinterpret_cast<const uint32_t*>(&headAttribute) + i);
        if constexpr (sizeof...(tailAttributes) > 0)
            _setAttributes<start + numDwords>(a, tailAttributes...);
    }
    
    template <uint32_t numDwords, typename... AttributeTypes>
    RT_FUNCTION void _reportIntersection(float hitT, uint32_t hitKind, const AttributeTypes &... attributes) {
        uint32_t a[numDwords];
        if constexpr (numDwords > 0)
            _setAttributes<0>(a, attributes...);

        if constexpr (numDwords == 0)
            optixReportIntersection(hitT, hitKind);
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
    
    template <typename... AttributeTypes>
    RT_FUNCTION void reportIntersection(float hitT, uint32_t hitKind,
                                        const AttributeTypes &... attributes) {
        constexpr size_t numDwords = _calcSumDwords<AttributeTypes...>();
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        _reportIntersection<numDwords>(hitT, hitKind, attributes...);
    }



    template <uint32_t index>
    RT_FUNCTION uint32_t _optixGetAttribute() {
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

    template <typename AttributeType, uint32_t offset, uint32_t start>
    RT_FUNCTION void _getAttribute(AttributeType* attribute) {
        if (!attribute)
            return;
        constexpr uint32_t numDwords = sizeof(AttributeType) / 4;
        *(reinterpret_cast<uint32_t*>(attribute) + offset) = _optixGetAttribute<start>();
        if constexpr (offset + 1 < numDwords)
            _getAttribute<AttributeType, offset + 1, start + 1>(attribute);
    }

    template <uint32_t start, typename HeadType, typename... TailTypes>
    RT_FUNCTION void _getAttributes(HeadType* headAttribute, TailTypes*... tailAttributes) {
        _getAttribute<HeadType, 0, start>(headAttribute);
        if constexpr (sizeof...(tailAttributes) > 0)
            _getAttributes<start + sizeof(HeadType) / 4>(tailAttributes...);
    }

    template <typename... AttributeTypes>
    RT_FUNCTION void getAttributes(AttributeTypes*... attributes) {
        constexpr size_t numDwords = _calcSumDwords<AttributeTypes...>();
        static_assert(numDwords <= 8, "Maximum number of attributes is 8 dwords.");
        static_assert(numDwords > 0, "Calling this function without attributes has no effect.");
        if constexpr (numDwords > 0)
            _getAttributes<0>(attributes...);
    }
#endif



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
                             |
                             +-- Instance
                             |
                             +-- GAS
                             |
                             +-- GeomInst

    JP: 
    EN: 

    */

    class Context;
    class Material;
    class Scene;
    class GeometryInstance;
    class GeometryAccelerationStructure;
    class Instance;
    class InstanceAccelerationStructure;
    class Pipeline;
    class Module;
    class ProgramGroup;

#define OPTIX_PIMPL() \
public: \
    class Priv; \
private: \
    Priv* m = nullptr



    class Context {
        OPTIX_PIMPL();

    public:
        static Context create(CUcontext cudaContext);
        void destroy();

        Material createMaterial() const;
        Scene createScene() const;

        Pipeline createPipeline() const;

        CUcontext getCUcontext() const;
    };



    class Material {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setHitGroup(uint32_t rayType, ProgramGroup hitGroup);
        void setUserData(uint32_t data) const;
    };



    class Scene {
        OPTIX_PIMPL();

    public:
        void destroy();

        GeometryInstance createGeometryInstance(bool forCustomPrimitives = false) const;
        GeometryAccelerationStructure createGeometryAccelerationStructure(bool forCustomPrimitives = false) const;
        Instance createInstance() const;
        InstanceAccelerationStructure createInstanceAccelerationStructure() const;

        void generateShaderBindingTableLayout(size_t* memorySize) const;
    };



    class GeometryInstance {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setVertexBuffer(Buffer* vertexBuffer) const;
        void setTriangleBuffer(Buffer* triangleBuffer) const;
        void setCustomPrimitiveAABBBuffer(TypedBuffer<OptixAabb>* primitiveAABBBuffer) const;
        void setNumMaterials(uint32_t numMaterials, TypedBuffer<uint32_t>* matIdxOffsetBuffer) const;

        void setUserData(uint32_t data) const;

        void setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const;
        void setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const;
    };



    class GeometryAccelerationStructure {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) const;
        void setNumMaterialSets(uint32_t numMatSets) const;
        void setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const;

        void addChild(GeometryInstance geomInst) const;
        void removeChild(GeometryInstance geomInst) const;

        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const;
        OptixTraversableHandle rebuild(CUstream stream, const Buffer &accelBuffer, const Buffer &scratchBuffer) const;
        void prepareForCompact(size_t* compactedAccelBufferSize) const;
        OptixTraversableHandle compact(CUstream stream, const Buffer &compactedAccelBuffer) const;
        void removeUncompacted() const;
        OptixTraversableHandle update(CUstream stream, const Buffer &scratchBuffer) const;

        bool isReady() const;
    };



    class Instance {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setGAS(GeometryAccelerationStructure gas, uint32_t matSetIdx = 0) const;
        void setTransform(const float transform[12]) const;
    };



    class InstanceAccelerationStructure {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) const;

        void addChild(Instance instance) const;
        void removeChild(Instance instance) const;

        void prepareForBuild(OptixAccelBufferSizes* memoryRequirement, uint32_t* numInstances) const;
        // JP: インスタンスバッファーもユーザー管理にしたいため、今の形になっているが微妙かもしれない。
        //     インスタンスバッファーを内部で1つ持つようにすると、
        //     あるフレームでIASをビルド、次のフレームでインスタンスの追加がありリビルドの必要が生じた場合に
        //     1フレーム目のGPU処理の終了を待たないと危険という状況になってしまう。
        //     OptiX的にはASのビルド完了後にはインスタンスバッファーは不要となるが、
        //     アップデート処理はリビルド時に書かれたインスタンスバッファーの内容を期待しているため、
        //     基本的にインスタンスバッファーとASのメモリ(コンパクション版にもなり得る)は同じ寿命で扱ったほうが良さそう。
        // EN: 
        OptixTraversableHandle rebuild(CUstream stream, const TypedBuffer<OptixInstance> &instanceBuffer,
                                       const Buffer &accelBuffer, const Buffer &scratchBuffer) const;
        void prepareForCompact(size_t* compactedAccelBufferSize) const;
        OptixTraversableHandle compact(CUstream stream, const Buffer &compactedAccelBuffer) const;
        void removeUncompacted() const;
        OptixTraversableHandle update(CUstream stream, const Buffer &scratchBuffer) const;

        bool isReady() const;
    };



    class Pipeline {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setMaxTraceDepth(uint32_t maxTraceDepth) const;
        void setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues, const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags) const;

        Module createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount, OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) const;

        ProgramGroup createRayGenProgram(Module module, const char* entryFunctionName) const;
        ProgramGroup createExceptionProgram(Module module, const char* entryFunctionName) const;
        ProgramGroup createMissProgram(Module module, const char* entryFunctionName) const;
        ProgramGroup createHitProgramGroup(Module module_CH, const char* entryFunctionNameCH,
                                           Module module_AH, const char* entryFunctionNameAH,
                                           Module module_IS, const char* entryFunctionNameIS) const;
        ProgramGroup createCallableGroup(Module module_DC, const char* entryFunctionNameDC,
                                         Module module_CC, const char* entryFunctionNameCC) const;

        void link(OptixCompileDebugLevel debugLevel, bool overrideUseMotionBlur) const;

        void setNumMissRayTypes(uint32_t numMissRayTypes) const;

        void setRayGenerationProgram(ProgramGroup program) const;
        void setExceptionProgram(ProgramGroup program) const;
        void setMissProgram(uint32_t rayType, ProgramGroup program) const;
        void setCallableProgram(uint32_t index, ProgramGroup program) const;

        void setScene(const Scene &scene) const;
        void setHitGroupShaderBindingTable(Buffer* shaderBindingTable) const;
        void markHitGroupShaderBindingTableDirty() const;

        void launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const;

        void setStackSize(uint32_t directCallableStackSizeFromTraversal,
                          uint32_t directCallableStackSizeFromState,
                          uint32_t continuationStackSize) const;
    };



    // The lifetime of a module must extend to the lifetime of any ProgramGroup that reference that module.
    class Module {
        OPTIX_PIMPL();

    public:
        void destroy();
    };



    class ProgramGroup {
        OPTIX_PIMPL();

    public:
        void destroy();

        void getStackSize(OptixStackSizes* sizes) const;
    };

#endif // #if !defined(__CUDA_ARCH__)
}
