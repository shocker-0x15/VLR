#pragma once

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define CUDAHPlatform_Windows
#   if defined(__MINGW32__) // Defined for both 32 bit/64 bit MinGW
#       define CUDAHPlatform_Windows_MinGW
#   elif defined(_MSC_VER)
#       define CUDAHPlatform_Windows_MSVC
#   endif
#elif defined(__linux__)
#   define CUDAHPlatform_Linux
#elif defined(__APPLE__)
#   define CUDAHPlatform_macOS
#elif defined(__OpenBSD__)
#   define CUDAHPlatform_OpenBSD
#endif

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <vector>
#include <sstream>

// Enable this macro if CUDA/OpenGL interoperability is required.
#define CUDA_UTIL_USE_GL_INTEROP
#if defined(CUDA_UTIL_USE_GL_INTEROP)
#include <GL/gl3w.h>
#endif

#include <cuda.h>
#if defined(CUDA_UTIL_USE_GL_INTEROP)
#include <cudaGL.h>
#endif
#include <vector_types.h>

#undef min
#undef max

#ifdef _DEBUG
#   define CUDAU_ENABLE_ASSERT
#endif

#ifdef CUDAU_ENABLE_ASSERT
#   define CUDAUAssert(expr, fmt, ...) \
    if (!(expr)) { \
        cudau::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        cudau::devPrintf(fmt"\n", ##__VA_ARGS__); \
        abort(); \
    } 0
#else
#   define CUDAUAssert(expr, fmt, ...)
#endif

#define CUDAUAssert_ShouldNotBeCalled() CUDAUAssert(false, "Should not be called!")
#define CUDAUAssert_NotImplemented() CUDAUAssert(false, "Not implemented yet!")

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << cudaGetErrorString(error) \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



namespace cudau {
    void devPrintf(const char* fmt, ...);



    using ConstVoidPtr = const void*;
    
    inline void addArgPointer(ConstVoidPtr* pointer) {}

    template <typename HeadType, typename... TailTypes>
    void addArgPointer(ConstVoidPtr* pointer, HeadType &&head, TailTypes&&... tails) {
        *pointer = &head;
        addArgPointer(pointer + 1, std::forward<TailTypes>(tails)...);
    }

    template <typename... ArgTypes>
    void callKernel(CUstream stream, CUfunction kernel, const dim3 &gridDim, const dim3 &blockDim, uint32_t sharedMemSize,
                    ArgTypes&&... args) {
        ConstVoidPtr argPointers[sizeof...(args)];
        addArgPointer(argPointers, std::forward<ArgTypes>(args)...);

        CUDADRV_CHECK(cuLaunchKernel(kernel,
                                     gridDim.x, gridDim.y, gridDim.z,
                                     blockDim.x, blockDim.y, blockDim.z,
                                     sharedMemSize, stream, const_cast<void**>(argPointers), nullptr));
    }



    class Kernel {
        CUfunction m_kernel;
        dim3 m_blockDim;
        uint32_t m_sharedMemSize;

    public:
        Kernel(CUmodule module, const char* name, const dim3 blockDim, uint32_t sharedMemSize) :
            m_blockDim(blockDim), m_sharedMemSize(sharedMemSize) {
            CUDADRV_CHECK(cuModuleGetFunction(&m_kernel, module, name));
        }

        void setBlockDimensions(const dim3 &blockDim) {
            m_blockDim = blockDim;
        }
        void setSharedMemorySize(uint32_t sharedMemSize) {
            m_sharedMemSize = sharedMemSize;
        }

        uint32_t getBlockDimX() const { return m_blockDim.x; }
        uint32_t getBlockDimY() const { return m_blockDim.y; }
        uint32_t getBlockDimZ() const { return m_blockDim.z; }
        dim3 calcGridDim(uint32_t numItemsX) const {
            return dim3((numItemsX + m_blockDim.x - 1) / m_blockDim.x);
        }
        dim3 calcGridDim(uint32_t numItemsX, uint32_t numItemsY) const {
            return dim3((numItemsX + m_blockDim.x - 1) / m_blockDim.x,
                        (numItemsY + m_blockDim.y - 1) / m_blockDim.y);
        }
        dim3 calcGridDim(uint32_t numItemsX, uint32_t numItemsY, uint32_t numItemsZ) const {
            return dim3((numItemsX + m_blockDim.x - 1) / m_blockDim.x,
                        (numItemsY + m_blockDim.y - 1) / m_blockDim.y,
                        (numItemsZ + m_blockDim.z - 1) / m_blockDim.z);
        }

        template <typename... ArgTypes>
        void operator()(CUstream stream, const dim3 &gridDim, ArgTypes&&... args) const {
            callKernel(stream, m_kernel, gridDim, m_blockDim, m_sharedMemSize, std::forward<ArgTypes>(args)...);
        }
    };



    class Timer {
        CUcontext m_context;
        CUevent m_startEvent;
        CUevent m_endEvent;

    public:
        void initialize(CUcontext context) {
            m_context = context;
            CUDADRV_CHECK(cuCtxSetCurrent(m_context));
            CUDADRV_CHECK(cuEventCreate(&m_startEvent, CU_EVENT_BLOCKING_SYNC));
            CUDADRV_CHECK(cuEventCreate(&m_endEvent, CU_EVENT_BLOCKING_SYNC));
        }
        void finalize() {
            CUDADRV_CHECK(cuCtxSetCurrent(m_context));
            CUDADRV_CHECK(cuEventDestroy(m_endEvent));
            CUDADRV_CHECK(cuEventDestroy(m_startEvent));
            m_context = nullptr;
        }

        void start(CUstream stream) const {
            CUDADRV_CHECK(cuEventRecord(m_startEvent, stream));
        }
        void stop(CUstream stream) const {
            CUDADRV_CHECK(cuEventRecord(m_endEvent, stream));
        }

        float report() const {
            float ret = 0.0f;
            CUDADRV_CHECK(cuEventSynchronize(m_endEvent));
            CUDADRV_CHECK(cuEventElapsedTime(&ret, m_startEvent, m_endEvent));
            return ret;
        }
    };



    enum class BufferType {
        Device = 0,     // not preferred, typically slower than ZERO_COPY
        GL_Interop = 1, // single device only, preferred for single device
        ZeroCopy = 2,   // general case, preferred for multi-gpu if not fully nvlink connected
        P2P = 3         // fully connected only, preferred for fully nvlink connected
    };

    class Buffer {
        CUcontext m_cudaContext;
        BufferType m_type;

        uint32_t m_numElements;
        uint32_t m_stride;

        void* m_hostPointer;
        CUdeviceptr m_devicePointer;
        void* m_mappedPointer;

        uint32_t m_GLBufferID;
        CUgraphicsResource m_cudaGfxResource;

        struct {
            unsigned int m_initialized : 1;
            unsigned int m_mapped : 1;
        };

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        void initialize(CUcontext context, BufferType type,
                        uint32_t numElements, uint32_t stride, uint32_t glBufferID);

    public:
        Buffer();
        ~Buffer();

        Buffer(Buffer &&b);
        Buffer &operator=(Buffer &&b);

        void initialize(CUcontext context, BufferType type,
                        uint32_t numElements, uint32_t stride) {
            initialize(context, type, numElements, stride, 0);
        }
        void initializeFromGLBuffer(CUcontext context, uint32_t glBufferID) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            GLint currentBuffer;
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currentBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, glBufferID);
            GLint size;
            glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
            glBindBuffer(GL_ARRAY_BUFFER, currentBuffer);
            initialize(context, BufferType::GL_Interop, size, 1, glBufferID);
#else
            CUDAUAssert(false, "Enable \"CUDA_UTIL_USE_GL_INTEROP\" at the top of this file if you use CUDA/OpenGL interoperability.");
#endif
        }
        void finalize();

        void resize(uint32_t numElements, uint32_t stride);

        CUcontext getCUcontext() const {
            return m_cudaContext;
        }
        BufferType getBufferType() const {
            return m_type;
        }

        CUdeviceptr getCUdeviceptr() const {
            return m_devicePointer;
        }
        CUdeviceptr getCUdeviceptrAt(uint32_t idx) const {
            return m_devicePointer + m_stride * idx;
        }
        size_t sizeInBytes() const {
            return m_numElements * m_stride;
        }
        size_t stride() const {
            return m_stride;
        }
        size_t numElements() const {
            return m_numElements;
        }
        bool isInitialized() const {
            return m_initialized;
        }

        void beginCUDAAccess(CUstream stream);
        void endCUDAAccess(CUstream stream);

        void* map();
        template <typename T>
        T* map() {
            return reinterpret_cast<T*>(map());
        }
        void unmap();

        Buffer copy() const;
    };



    template <typename T>
    class TypedBuffer : public Buffer {
    public:
        TypedBuffer() {}
        TypedBuffer(CUcontext context, BufferType type, int32_t numElements) {
            Buffer::initialize(context, type, numElements, sizeof(T));
        }
        TypedBuffer(CUcontext context, BufferType type, int32_t numElements, const T &value) {
            Buffer::initialize(context, type, numElements, sizeof(T));
            T* values = (T*)map();
            for (int i = 0; i < numElements; ++i)
                values[i] = value;
            unmap();
        }

        void initialize(CUcontext context, BufferType type, int32_t numElements) {
            Buffer::initialize(context, type, numElements, sizeof(T));
        }
        void initialize(CUcontext context, BufferType type, int32_t numElements, const T &value) {
            Buffer::initialize(context, type, numElements, sizeof(T));
            T* values = (T*)Buffer::map();
            for (int i = 0; i < numElements; ++i)
                values[i] = value;
            Buffer::unmap();
        }
        void initialize(CUcontext context, BufferType type, const T* v, uint32_t numElements) {
            initialize(context, type, numElements);
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), v, numElements * sizeof(T)));
        }
        void initialize(CUcontext context, BufferType type, const std::vector<T> &v) {
            initialize(context, type, v.size());
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), v.data(), v.size() * sizeof(T)));
        }
        void finalize() {
            Buffer::finalize();
        }

        void resize(int32_t numElements) {
            Buffer::resize(numElements, sizeof(T));
        }

        T* getDevicePointer() const {
            return reinterpret_cast<T*>(getCUdeviceptr());
        }
        T* getDevicePointerAt(uint32_t idx) const {
            return reinterpret_cast<T*>(getCUdeviceptrAt(idx));
        }

        T* map() {
            return reinterpret_cast<T*>(Buffer::map());
        }

        T operator[](uint32_t idx) {
            const T* values = map();
            T ret = values[idx];
            unmap();
            return ret;
        }

        TypedBuffer<T> copy() const {
            TypedBuffer<T> ret;
            // safe ?
            *reinterpret_cast<Buffer*>(&ret) = Buffer::copy();
            return ret;
        }
    };

    template <typename T>
    class TypedHostBuffer {
        std::vector<T> m_values;

    public:
        TypedHostBuffer() {}
        TypedHostBuffer(TypedBuffer<T> &b) {
            m_values.resize(b.numElements());
            auto srcValues = b.map();
            std::copy_n(srcValues, b.numElements(), m_values.data());
            b.unmap();
        }

        T* getPointer() {
            return m_values.data();
        }
        size_t numElements() const {
            return m_values.size();
        }

        const T &operator[](uint32_t idx) const {
            return m_values[idx];
        }
        T &operator[](uint32_t idx) {
            return m_values[idx];
        }
    };



    enum class ArrayElementType {
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        Float16,
        Float32,
        BC1_UNorm,
        BC2_UNorm,
        BC3_UNorm,
        BC4_UNorm,
        BC4_SNorm,
        BC5_UNorm,
        BC5_SNorm,
        BC6H_UF16,
        BC6H_SF16,
        BC7_UNorm
    };

    enum class ArraySurface {
        Enable = 0,
        Disable,
    };

    void getArrayElementFormat(GLenum internalFormat, ArrayElementType* elemType, uint32_t* numChannels);
    
    class Array {
        CUcontext m_cudaContext;

        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_depth;
        uint32_t m_numMipmapLevels;
        uint32_t m_stride;
        ArrayElementType m_elemType;
        uint32_t m_numChannels;

        union {
            CUarray m_array;
            CUmipmappedArray m_mipmappedArray;
        };
        void** m_mappedPointers;
        CUarray* m_mappedArrays;

        uint32_t m_GLTexID;
        CUgraphicsResource m_cudaGfxResource;

        struct {
            unsigned int m_surfaceLoadStore : 1;
            unsigned int m_cubemap : 1;
            unsigned int m_layered : 1;
            unsigned int m_initialized : 1;
        };

        Array(const Array &) = delete;
        Array &operator=(const Array &) = delete;

        void initialize(CUcontext context, ArrayElementType elemType, uint32_t numChannels,
                        uint32_t width, uint32_t height, uint32_t depth, uint32_t numMipmapLevels,
                        bool writable, bool cubemap, bool layered, uint32_t glTexID);

    public:
        Array();
        ~Array();

        Array(Array &&b);
        Array &operator=(Array &&b);

        void initialize1D(CUcontext context, ArrayElementType elemType, uint32_t numChannels, ArraySurface surfaceLoadStore,
                          uint32_t length, uint32_t numMipmapLevels) {
            initialize(context, elemType, numChannels, length, 0, 0, numMipmapLevels,
                       surfaceLoadStore == ArraySurface::Enable, false, false, 0);
        }
        void initialize2D(CUcontext context, ArrayElementType elemType, uint32_t numChannels, ArraySurface surfaceLoadStore,
                          uint32_t width, uint32_t height, uint32_t numMipmapLevels) {
            initialize(context, elemType, numChannels, width, height, 0, numMipmapLevels,
                       surfaceLoadStore == ArraySurface::Enable, false, false, 0);
        }
        void initialize3D(CUcontext context, ArrayElementType elemType, uint32_t numChannels, ArraySurface surfaceLoadStore,
                          uint32_t width, uint32_t height, uint32_t depth, uint32_t numMipmapLevels) {
            initialize(context, elemType, numChannels, width, height, 0, numMipmapLevels,
                       surfaceLoadStore == ArraySurface::Enable, false, false, 0);
        }
        void initializeFromGLTexture2D(CUcontext context, uint32_t glTexID, ArraySurface surfaceLoadStore) {
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            GLint currentTexture;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &currentTexture);
            glBindTexture(GL_TEXTURE_2D, glTexID);
            GLint width, height;
            GLint numMipmapLevels;
            GLint format;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
            glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_VIEW_NUM_LEVELS, &numMipmapLevels);
            numMipmapLevels = std::max(numMipmapLevels, 1);
            glBindTexture(GL_TEXTURE_2D, currentTexture);
            ArrayElementType elemType;
            uint32_t numChannels;
            getArrayElementFormat((GLenum)format, &elemType, &numChannels);
            initialize(context, elemType, numChannels, width, height, 0, numMipmapLevels,
                       surfaceLoadStore == ArraySurface::Enable, false, false, glTexID);
#else
            CUDAUAssert(false, "Enable \"CUDA_UTIL_USE_GL_INTEROP\" at the top of this file if you use CUDA/OpenGL interoperability.");
#endif
        }
        void finalize();

        void resize(uint32_t length);
        void resize(uint32_t width, uint32_t height);
        void resize(uint32_t width, uint32_t height, uint32_t depth);

        CUarray getCUarray(uint32_t mipmapLevel) const {
            if (m_numMipmapLevels > 1) {
                CUarray ret;
                CUDADRV_CHECK(cuMipmappedArrayGetLevel(&ret, m_mipmappedArray, mipmapLevel));
                return ret;
            }
            else {
                return m_array;
            }
        }
        CUmipmappedArray getCUmipmappedArray() const {
            return m_mipmappedArray;
        }

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        uint32_t getDepth() const {
            return m_depth;
        }
        uint32_t getNumMipmapLevels() const {
            return m_numMipmapLevels;
        }

        void beginCUDAAccess(CUstream stream, uint32_t mipmapLevel);
        void endCUDAAccess(CUstream stream, uint32_t mipmapLevel);

        void* map(uint32_t mipmapLevel = 0);
        template <typename T>
        T* map(uint32_t mipmapLevel = 0) {
            return reinterpret_cast<T*>(map(mipmapLevel));
        }
        void unmap(uint32_t mipmapLevel = 0);

        CUDA_RESOURCE_VIEW_DESC getResourceViewDesc() const;

        CUsurfObject getSurfaceObject(uint32_t mipmapLevel) const {
            CUsurfObject ret;
            CUDA_RESOURCE_DESC resDesc = {};
            resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            if (m_GLTexID == 0)
                resDesc.res.array.hArray = getCUarray(mipmapLevel);
            else
                resDesc.res.array.hArray = m_mappedArrays[mipmapLevel];
            CUDADRV_CHECK(cuSurfObjectCreate(&ret, &resDesc));
            return ret;
        }
    };



    enum class TextureWrapMode {
        Repeat = CU_TR_ADDRESS_MODE_WRAP,
        Clamp = CU_TR_ADDRESS_MODE_CLAMP,
        Mirror = CU_TR_ADDRESS_MODE_MIRROR,
        Border = CU_TR_ADDRESS_MODE_BORDER,
    };

    enum class TextureFilterMode {
        Point = CU_TR_FILTER_MODE_POINT,
        Linear = CU_TR_FILTER_MODE_LINEAR,
    };

    enum class TextureIndexingMode {
        NormalizedCoordinates = 0,
        ArrayIndex,
    };

    enum class TextureReadMode {
        ElementType = 0,
        NormalizedFloat,
        NormalizedFloat_sRGB
    };
    
    class TextureSampler {
        CUDA_RESOURCE_DESC m_resDesc;
        CUDA_TEXTURE_DESC m_texDesc;
        CUDA_RESOURCE_VIEW_DESC m_resViewDesc;
        CUtexObject m_texObject;
        struct {
            unsigned int m_texObjectCreated : 1;
            unsigned int m_texObjectIsUpToDate : 1;
        };

        TextureSampler(const TextureSampler &) = delete;
        TextureSampler &operator=(const TextureSampler &) = delete;

    public:
        TextureSampler() : 
            m_texObjectCreated(false), m_texObjectIsUpToDate(false) {
            m_resDesc = {};
            m_texDesc = {};
            m_texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
            m_resViewDesc = {};
        }
        ~TextureSampler() {
            if (m_texObjectCreated)
                cuTexObjectDestroy(m_texObject);
        }

        void destroyTextureObject() {
            if (m_texObjectCreated) {
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
                m_texObjectIsUpToDate = false;
                m_texObjectCreated = false;
            }
        }

        TextureSampler(TextureSampler &&t) {
            m_resDesc = t.m_resDesc;
            m_texDesc = t.m_texDesc;
            m_resViewDesc = t.m_resViewDesc;
            m_texObject = t.m_texObject;
            m_texObjectCreated = t.m_texObjectCreated;
            m_texObjectIsUpToDate = t.m_texObjectIsUpToDate;

            t.m_texObjectCreated = false;
        }
        TextureSampler &operator=(TextureSampler &&t) {
            m_resDesc = t.m_resDesc;
            m_texDesc = t.m_texDesc;
            m_resViewDesc = t.m_resViewDesc;
            m_texObject = t.m_texObject;
            m_texObjectCreated = t.m_texObjectCreated;
            m_texObjectIsUpToDate = t.m_texObjectIsUpToDate;

            t.m_texObjectCreated = false;
        }

        void setArray(const Array &array) {
            if (array.getNumMipmapLevels() > 1) {
                m_resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
                m_resDesc.res.mipmap.hMipmappedArray = array.getCUmipmappedArray();
                m_texDesc.maxMipmapLevelClamp = array.getNumMipmapLevels() - 1;
            }
            else {
                m_resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
                m_resDesc.res.array.hArray = array.getCUarray(0);
            }
            m_resViewDesc = array.getResourceViewDesc();
            m_texObjectIsUpToDate = false;
        }

        void setFilterMode(TextureFilterMode xy, TextureFilterMode mipmap) {
            m_texDesc.filterMode = static_cast<CUfilter_mode>(xy);
            m_texDesc.mipmapFilterMode = static_cast<CUfilter_mode>(mipmap);
            m_texObjectIsUpToDate = false;
        }
        void setWrapMode(uint32_t dim, TextureWrapMode mode) {
            if (dim >= 3)
                return;
            m_texDesc.addressMode[dim] = static_cast<CUaddress_mode>(mode);
            m_texObjectIsUpToDate = false;
        }
        void setBorderColor(float r, float g, float b, float a) {
            m_texDesc.borderColor[0] = r;
            m_texDesc.borderColor[1] = g;
            m_texDesc.borderColor[2] = b;
            m_texDesc.borderColor[3] = a;
            m_texObjectIsUpToDate = false;
        }
        void setIndexingMode(TextureIndexingMode mode) {
            if (mode == TextureIndexingMode::ArrayIndex)
                m_texDesc.flags &= ~CU_TRSF_NORMALIZED_COORDINATES;
            else
                m_texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
            m_texObjectIsUpToDate = false;
        }
        void setReadMode(TextureReadMode mode) {
            if (mode == TextureReadMode::ElementType)
                m_texDesc.flags |= CU_TRSF_READ_AS_INTEGER;
            else
                m_texDesc.flags &= ~CU_TRSF_READ_AS_INTEGER;

            if (mode == TextureReadMode::NormalizedFloat_sRGB)
                m_texDesc.flags |= CU_TRSF_SRGB;
            else
                m_texDesc.flags &= ~CU_TRSF_SRGB;
            m_texObjectIsUpToDate = false;
        }

        CUtexObject getTextureObject() {
            if (m_texObjectCreated && !m_texObjectIsUpToDate)
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
            if (!m_texObjectIsUpToDate) {
                CUDADRV_CHECK(cuTexObjectCreate(&m_texObject, &m_resDesc, &m_texDesc, &m_resViewDesc));
                m_texObjectCreated = true;
                m_texObjectIsUpToDate = true;
            }
            return m_texObject;
        }
    };
}
