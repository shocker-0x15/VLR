#pragma once

#include "cuda_util.h"
#include "optix_util.h"

namespace optixu {
    template <typename T>
    class NativeBlockBuffer2D {
        CUsurfObject m_surfObject;

    public:
        RT_DEVICE_FUNCTION NativeBlockBuffer2D() : m_surfObject(0) {}
        RT_DEVICE_FUNCTION NativeBlockBuffer2D(CUsurfObject surfObject) : m_surfObject(surfObject) {};

        RT_DEVICE_FUNCTION NativeBlockBuffer2D &operator=(CUsurfObject surfObject) {
            m_surfObject = surfObject;
            return *this;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION T read(uint2 idx) const {
            return surf2Dread<T>(m_surfObject, idx.x * sizeof(T), idx.y);
        }
        RT_DEVICE_FUNCTION void write(uint2 idx, const T &value) const {
            surf2Dwrite(value, m_surfObject, idx.x * sizeof(T), idx.y);
        }
        template <uint32_t comp, typename U>
        RT_DEVICE_FUNCTION void writeComp(uint2 idx, U value) const {
            surf2Dwrite(value, m_surfObject, idx.x * sizeof(T) + comp * sizeof(U), idx.y);
        }
        RT_DEVICE_FUNCTION T read(int2 idx) const {
            return surf2Dread<T>(m_surfObject, idx.x * sizeof(T), idx.y);
        }
        RT_DEVICE_FUNCTION void write(int2 idx, const T &value) const {
            surf2Dwrite(value, m_surfObject, idx.x * sizeof(T), idx.y);
        }
        template <uint32_t comp, typename U>
        RT_DEVICE_FUNCTION void writeComp(int2 idx, U value) const {
            surf2Dwrite(value, m_surfObject, idx.x * sizeof(T) + comp * sizeof(U), idx.y);
        }
#endif
    };



    template <typename T, uint32_t log2BlockWidth>
    class BlockBuffer2D {
        T* m_rawBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numXBlocks;

#if defined(__CUDA_ARCH__)
        RT_DEVICE_FUNCTION constexpr uint32_t calcLinearIndex(uint32_t idxX, uint32_t idxY) const {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t blockIdxX = idxX >> log2BlockWidth;
            uint32_t blockIdxY = idxY >> log2BlockWidth;
            uint32_t blockOffset = (blockIdxY * m_numXBlocks + blockIdxX) * (blockWidth * blockWidth);
            uint32_t idxXInBlock = idxX & mask;
            uint32_t idxYInBlock = idxY & mask;
            uint32_t linearIndexInBlock = idxYInBlock * blockWidth + idxXInBlock;
            return blockOffset + linearIndexInBlock;
        }
#endif

    public:
        RT_DEVICE_FUNCTION BlockBuffer2D() {}
        RT_DEVICE_FUNCTION BlockBuffer2D(T* rawBuffer, uint32_t width, uint32_t height) :
            m_rawBuffer(rawBuffer), m_width(width), m_height(height) {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            m_numXBlocks = ((width + mask) & ~mask) >> log2BlockWidth;
        }

#if defined(__CUDA_ARCH__)
        RT_DEVICE_FUNCTION uint2 getSize() const {
            return make_uint2(m_width, m_height);
        }

        RT_DEVICE_FUNCTION const T &operator[](uint2 idx) const {
            optixuAssert(idx.x < m_width && idx.y < m_height,
                         "Out of bounds: %u, %u", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
        RT_DEVICE_FUNCTION T &operator[](uint2 idx) {
            optixuAssert(idx.x < m_width && idx.y < m_height,
                         "Out of bounds: %u, %u", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
        RT_DEVICE_FUNCTION const T &operator[](int2 idx) const {
            optixuAssert(idx.x >= 0 && idx.x < m_width && idx.y >= 0 && idx.y < m_height,
                         "Out of bounds: %d, %d", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
        RT_DEVICE_FUNCTION T &operator[](int2 idx) {
            optixuAssert(idx.x >= 0 && idx.x < m_width && idx.y >= 0 && idx.y < m_height,
                         "Out of bounds: %d, %d", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
#endif
    };

#if !defined(__CUDA_ARCH__)
    template <typename T, uint32_t log2BlockWidth>
    class HostBlockBuffer2D {
        cudau::TypedBuffer<T> m_rawBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numXBlocks;
        T* m_mappedPointer;

        constexpr uint32_t calcLinearIndex(uint32_t x, uint32_t y) const {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t blockIdxX = x >> log2BlockWidth;
            uint32_t blockIdxY = y >> log2BlockWidth;
            uint32_t blockOffset = (blockIdxY * m_numXBlocks + blockIdxX) * (blockWidth * blockWidth);
            uint32_t idxXInBlock = x & mask;
            uint32_t idxYInBlock = y & mask;
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

        void initialize(CUcontext context, cudau::BufferType type, uint32_t width, uint32_t height) {
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
                for (uint32_t yb = 0; yb < numYBlocksToCopy; ++yb) {
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
        cudau::BufferType getBufferType() const {
            return m_rawBuffer.getBufferType();
        }

        CUdeviceptr getCUdeviceptr() const {
            return m_rawBuffer.getCUdeviceptr();
        }
        bool isInitialized() const {
            return m_rawBuffer.isInitialized();
        }

        void map() {
            m_mappedPointer = reinterpret_cast<T*>(m_rawBuffer.map());
        }
        void unmap() {
            m_rawBuffer.unmap();
            m_mappedPointer = nullptr;
        }
        const T &operator()(uint32_t x, uint32_t y) const {
            return m_mappedPointer[calcLinearIndex(x, y)];
        }
        T &operator()(uint32_t x, uint32_t y) {
            return m_mappedPointer[calcLinearIndex(x, y)];
        }

        BlockBuffer2D<T, log2BlockWidth> getBlockBuffer2D() const {
            return BlockBuffer2D<T, log2BlockWidth>(m_rawBuffer.getDevicePointer(), m_width, m_height);
        }
    };
#endif // !defined(__CUDA_ARCH__)
}

#if !defined(__CUDA_ARCH__)
inline optixu::BufferView getView(const cudau::Buffer &buffer) {
    return optixu::BufferView(buffer.getCUdeviceptr(), buffer.numElements(), buffer.stride());
}
#endif // !defined(__CUDA_ARCH__)
