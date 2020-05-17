#include "cuda_util.h"

#ifdef CUDAHPlatform_Windows_MSVC
#   include <Windows.h>
#   undef near
#   undef far
#   undef min
#   undef max
#endif



namespace cudau {
#ifdef CUDAHPlatform_Windows_MSVC
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[1024];
        vsprintf_s(str, fmt, args);
        va_end(args);
        OutputDebugString(str);
    }
#else
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprintf_s(fmt, args);
        va_end(args);
    }
#endif



    Buffer::Buffer() :
        m_cudaContext(nullptr),
        m_hostPointer(nullptr), m_devicePointer(0), m_mappedPointer(nullptr),
        m_GLBufferID(0), m_cudaGfxResource(nullptr),
        m_initialized(false), m_mapped(false) {
    }

    Buffer::~Buffer() {
        if (m_initialized)
            finalize();
    }

    Buffer::Buffer(Buffer &&b) {
        m_cudaContext = b.m_cudaContext;
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;
    }

    Buffer &Buffer::operator=(Buffer &&b) {
        finalize();

        m_cudaContext = b.m_cudaContext;
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;

        return *this;
    }
    
    void Buffer::initialize(CUcontext context, BufferType type,
                            uint32_t numElements, uint32_t stride, uint32_t glBufferID) {
        if (m_initialized)
            throw std::runtime_error("Buffer is already initialized.");

        m_cudaContext = context;
        m_type = type;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        // If using GL Interop, expect that the active device is also the display device.
        if (m_type == BufferType::GL_Interop) {
            CUdevice currentDevice;
            int32_t isDisplayDevice;
            CUDADRV_CHECK(cuCtxGetDevice(&currentDevice));
            CUDADRV_CHECK(cuDeviceGetAttribute(&isDisplayDevice, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, currentDevice));
            if (!isDisplayDevice)
                throw std::runtime_error("GL Interop is only available on the display device.");
        }

        m_numElements = numElements;
        m_stride = stride;

        m_GLBufferID = glBufferID;

        if (m_type == BufferType::Device || m_type == BufferType::P2P)
            CUDADRV_CHECK(cuMemAlloc(&m_devicePointer, m_numElements * m_stride));

        if (m_type == BufferType::GL_Interop)
#if defined(CUDA_UTIL_USE_GL_INTEROP)
            CUDADRV_CHECK(cuGraphicsGLRegisterBuffer(&m_cudaGfxResource, m_GLBufferID, CU_GRAPHICS_REGISTER_FLAGS_NONE));
#else
            CUDAUAssert(false, "Enable \"CUDA_UTIL_USE_GL_INTEROP\" at the top of the header if you use CUDA/OpenGL interoperability.");
#endif

        if (m_type == BufferType::ZeroCopy) {
            CUDADRV_CHECK(cuMemHostAlloc(&m_hostPointer, m_numElements * m_stride, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
            CUDADRV_CHECK(cuMemHostGetDevicePointer(&m_devicePointer, m_hostPointer, 0));
        }

        m_initialized = true;
    }

    void Buffer::finalize() {
        if (!m_initialized)
            return;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        if (m_mapped)
            unmap();

        if (m_type == BufferType::ZeroCopy) {
            CUDADRV_CHECK(cuMemFreeHost(m_hostPointer));
            m_devicePointer = 0;
            m_hostPointer = nullptr;
        }

        if (m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuGraphicsUnregisterResource(m_cudaGfxResource));
            m_devicePointer = 0;
        }

        if (m_type == BufferType::Device || m_type == BufferType::P2P) {
            CUDADRV_CHECK(cuMemFree(m_devicePointer));
            m_devicePointer = 0;
        }

        m_mappedPointer = nullptr;
        m_stride = 0;
        m_numElements = 0;

        m_initialized = false;
    }

    void Buffer::resize(uint32_t numElements, uint32_t stride) {
        if (!m_initialized)
            throw std::runtime_error("Buffer is not initialized.");
        if (m_type == BufferType::GL_Interop)
            throw std::runtime_error("Resize for GL-interop buffer is not supported.");
        if (stride < m_stride)
            throw std::runtime_error("New stride must be >= the current stride.");

        if (numElements == m_numElements && stride == m_stride)
            return;

        Buffer newBuffer;
        newBuffer.initialize(m_cudaContext, m_type, numElements, stride, m_GLBufferID);

        uint32_t numElementsToCopy = std::min(m_numElements, numElements);
        if (stride == m_stride) {
            size_t numBytesToCopy = static_cast<size_t>(numElementsToCopy) * m_stride;
            CUDADRV_CHECK(cuMemcpyDtoD(newBuffer.m_devicePointer, m_devicePointer, numBytesToCopy));
        }
        else {
            auto src = map<const uint8_t>();
            auto dst = newBuffer.map<uint8_t>();
            for (int i = 0; i < numElementsToCopy; ++i) {
                std::memset(dst, 0, stride);
                std::memcpy(dst, src, m_stride);
            }
            newBuffer.unmap();
            unmap();
        }

        *this = std::move(newBuffer);
    }

    void Buffer::beginCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        size_t bufferSize = 0;
        CUDADRV_CHECK(cuGraphicsMapResources(1, &m_cudaGfxResource, stream));
        CUDADRV_CHECK(cuGraphicsResourceGetMappedPointer(&m_devicePointer, &bufferSize, m_cudaGfxResource));
    }

    void Buffer::endCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        CUDADRV_CHECK(cuGraphicsUnmapResources(1, &m_cudaGfxResource, stream));
    }

    void* Buffer::map() {
        if (m_mapped)
            throw std::runtime_error("This buffer is already mapped.");

        m_mapped = true;

        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P ||
            m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            size_t size = (size_t)m_numElements * m_stride;
            m_mappedPointer = new uint8_t[size];

            if (m_type == BufferType::GL_Interop)
                beginCUDAAccess(0);

            CUDADRV_CHECK(cuMemcpyDtoH(m_mappedPointer, m_devicePointer, size));

            return m_mappedPointer;
        }
        else {
            return m_hostPointer;
        }
    }

    void Buffer::unmap() {
        if (!m_mapped)
            throw std::runtime_error("This buffer is not mapped.");

        m_mapped = false;

        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P ||
            m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            size_t size = (size_t)m_numElements * m_stride;

            CUDADRV_CHECK(cuMemcpyHtoD(m_devicePointer, m_mappedPointer, size));

            if (m_type == BufferType::GL_Interop)
                endCUDAAccess(0);

            delete[] m_mappedPointer;
            m_mappedPointer = nullptr;
        }
    }

    Buffer Buffer::copy() const {
        if (m_GLBufferID != 0)
            throw std::runtime_error("Copying OpenGL buffer is not supported.");

        Buffer ret;
        ret.initialize(m_cudaContext, m_type, m_numElements, m_stride, m_GLBufferID);

        size_t size = m_stride * m_numElements;
        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            CUDADRV_CHECK(cuMemcpyDtoD(ret.m_devicePointer, m_devicePointer, size));
        }
        else {
            std::memcpy(ret.m_hostPointer, m_hostPointer, size);
        }

        return ret;
    }



    static bool isBCFormat(ArrayElementType elemType) {
        return (elemType == cudau::ArrayElementType::BC1_UNorm ||
                elemType == cudau::ArrayElementType::BC2_UNorm ||
                elemType == cudau::ArrayElementType::BC3_UNorm ||
                elemType == cudau::ArrayElementType::BC4_UNorm ||
                elemType == cudau::ArrayElementType::BC4_SNorm ||
                elemType == cudau::ArrayElementType::BC5_UNorm ||
                elemType == cudau::ArrayElementType::BC5_SNorm ||
                elemType == cudau::ArrayElementType::BC6H_UF16 ||
                elemType == cudau::ArrayElementType::BC6H_SF16 ||
                elemType == cudau::ArrayElementType::BC7_UNorm);
    }

    static CUresourceViewFormat getResourceViewFormat(ArrayElementType elemType, uint32_t numChannels) {
#define CUDA_UTIL_EXPR0(arrayEnum, BaseType, BitWidth) \
    case cudau::ArrayElementType::arrayEnum ## BitWidth: \
        if (numChannels == 1) \
            return CU_RES_VIEW_FORMAT_ ## BaseType ## _1X ## BitWidth; \
        else if (numChannels == 2) \
            return CU_RES_VIEW_FORMAT_ ## BaseType ## _2X ## BitWidth; \
        else if (numChannels == 4) \
            return CU_RES_VIEW_FORMAT_ ## BaseType ## _4X ## BitWidth; \
        break    

        switch (elemType) {
            CUDA_UTIL_EXPR0(UInt, UINT, 8);
            CUDA_UTIL_EXPR0(Int, SINT, 8);
            CUDA_UTIL_EXPR0(UInt, UINT, 16);
            CUDA_UTIL_EXPR0(Int, SINT, 16);
            CUDA_UTIL_EXPR0(UInt, UINT, 32);
            CUDA_UTIL_EXPR0(Int, SINT, 32);
            CUDA_UTIL_EXPR0(Float, FLOAT, 16);
            CUDA_UTIL_EXPR0(Float, FLOAT, 32);
        case cudau::ArrayElementType::BC1_UNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
            break;
        case cudau::ArrayElementType::BC2_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
            break;
        case cudau::ArrayElementType::BC3_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
            break;
        case cudau::ArrayElementType::BC4_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC4;
            break;
        case cudau::ArrayElementType::BC4_SNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_SIGNED_BC4;
            break;
        case cudau::ArrayElementType::BC5_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC5;
            break;
        case cudau::ArrayElementType::BC5_SNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SIGNED_BC5;
            break;
        case cudau::ArrayElementType::BC6H_UF16:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC6H;
            break;
        case cudau::ArrayElementType::BC6H_SF16:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SIGNED_BC6H;
            break;
        case cudau::ArrayElementType::BC7_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC7;
            break;
        default:
            break;
        }
        CUDAUAssert_ShouldNotBeCalled();
        return CU_RES_VIEW_FORMAT_NONE;

#undef CUDA_UTIL_EXPR0
    }



    void getArrayElementFormat(GLenum internalFormat, ArrayElementType* elemType, uint32_t* numChannels) {
#define CUDA_UTIL_EXPR0(glEnum, arrayEnum, numCh) \
    case glEnum: \
        *elemType = ArrayElementType::arrayEnum; \
        *numChannels = numCh; \
        break
#define CUDA_UTIL_EXPR1(glEnum, numCh) \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 8,    UInt8,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16,   Int16,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16F,  Float16, numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 32F,  Float32, numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 8UI,  UInt8,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16UI, UInt16,  numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 32UI, UInt32,  numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 8I,   Int8,    numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 16I,  Int16,   numCh); \
        CUDA_UTIL_EXPR0(GL_ ## glEnum ## 32I,  Int32,   numCh)

        switch (internalFormat) {
            CUDA_UTIL_EXPR1(R, 1);
            CUDA_UTIL_EXPR1(RG, 2);
            CUDA_UTIL_EXPR1(RGBA, 4);
        default:
            CUDAUAssert_ShouldNotBeCalled();
            break;
        }

#undef CUDA_UTIL_EXPR1
#undef CUDA_UTIL_EXPR0
    }
    
    Array::Array() :
        m_cudaContext(nullptr),
        m_array(0), m_mappedPointers(nullptr), m_mappedArrays(nullptr),
        m_GLTexID(0), m_cudaGfxResource(nullptr),
        m_surfaceLoadStore(false), m_cubemap(false), m_layered(false),
        m_initialized(false) {
    }

    Array::~Array() {
        if (m_initialized)
            finalize();
    }

    Array::Array(Array &&b) {
        m_cudaContext = b.m_cudaContext;
        m_width = b.m_width;
        m_height = b.m_height;
        m_depth = b.m_depth;
        m_stride = b.m_stride;
        m_elemType = b.m_elemType;
        m_numChannels = b.m_numChannels;
        if (m_numMipmapLevels > 1)
            m_mipmappedArray = b.m_mipmappedArray;
        else
            m_array = b.m_array;
        m_mappedPointers = b.m_mappedPointers;
        m_mappedArrays = b.m_mappedArrays;
        m_GLTexID = b.m_GLTexID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_surfaceLoadStore = b.m_surfaceLoadStore;
        m_cubemap = b.m_cubemap;
        m_layered = b.m_layered;
        m_initialized = b.m_initialized;

        b.m_initialized = false;
    }

    Array &Array::operator=(Array &&b) {
        finalize();

        m_cudaContext = b.m_cudaContext;
        m_width = b.m_width;
        m_height = b.m_height;
        m_depth = b.m_depth;
        m_stride = b.m_stride;
        m_elemType = b.m_elemType;
        m_numChannels = b.m_numChannels;
        if (m_numMipmapLevels > 1)
            m_mipmappedArray = b.m_mipmappedArray;
        else
            m_array = b.m_array;
        m_mappedPointers = b.m_mappedPointers;
        m_mappedArrays = b.m_mappedArrays;
        m_GLTexID = b.m_GLTexID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_surfaceLoadStore = b.m_surfaceLoadStore;
        m_cubemap = b.m_cubemap;
        m_layered = b.m_layered;
        m_initialized = b.m_initialized;

        b.m_initialized = false;

        return *this;
    }
    
    void Array::initialize(CUcontext context, ArrayElementType elemType, uint32_t numChannels,
                           uint32_t width, uint32_t height, uint32_t depth, uint32_t numMipmapLevels,
                           bool surfaceLoadStore, bool cubemap, bool layered, uint32_t glTexID) {
        if (m_initialized)
            throw std::runtime_error("Array is already initialized.");
        if (numChannels != 1 && numChannels != 2 && numChannels != 4)
            throw std::runtime_error("numChannels must be 1, 2, or 4.");
        if (elemType >= ArrayElementType::BC1_UNorm &&
            elemType <= ArrayElementType::BC7_UNorm &&
            numChannels != 1)
            throw std::runtime_error("numChannels must be 1 for BC format.");

        m_cudaContext = context;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        m_width = width;
        m_height = height;
        m_depth = depth;
        m_numMipmapLevels = numMipmapLevels;
        m_elemType = elemType;
        m_numChannels = numChannels;
        m_surfaceLoadStore = surfaceLoadStore;
        m_cubemap = cubemap;
        m_layered = layered;

        CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
        if (surfaceLoadStore)
            arrayDesc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
        if (layered)
            arrayDesc.Flags |= CUDA_ARRAY3D_LAYERED;
        if (cubemap)
            arrayDesc.Flags |= CUDA_ARRAY3D_CUBEMAP;
        switch (m_elemType) {
        case cudau::ArrayElementType::UInt8:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            m_stride = 1;
            break;
        case cudau::ArrayElementType::UInt16:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::UInt32:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::Int8:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT8;
            m_stride = 1;
            break;
        case cudau::ArrayElementType::Int16:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT16;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::Int32:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT32;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::Float16:
            arrayDesc.Format = CU_AD_FORMAT_HALF;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::Float32:
            arrayDesc.Format = CU_AD_FORMAT_FLOAT;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::BC1_UNorm:
        case cudau::ArrayElementType::BC4_UNorm:
        case cudau::ArrayElementType::BC4_SNorm:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            m_numChannels = 2;
            numChannels = 2;
            m_width >>= 2;
            m_height >>= 2;
            break;
        case cudau::ArrayElementType::BC2_UNorm:
        case cudau::ArrayElementType::BC3_UNorm:
        case cudau::ArrayElementType::BC5_UNorm:
        case cudau::ArrayElementType::BC5_SNorm:
        case cudau::ArrayElementType::BC6H_UF16:
        case cudau::ArrayElementType::BC6H_SF16:
        case cudau::ArrayElementType::BC7_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            m_numChannels = 4;
            numChannels = 4;
            m_width >>= 2;
            m_height >>= 2;
            break;
        default:
            CUDAUAssert_ShouldNotBeCalled();
            break;
        }
        arrayDesc.Width = m_width;
        arrayDesc.Height = m_height;
        arrayDesc.Depth = m_depth;
        arrayDesc.NumChannels = numChannels;
        m_stride *= numChannels;

        if (glTexID != 0) {
            uint32_t flags = surfaceLoadStore ? CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST : CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY;
            CUDADRV_CHECK(cuGraphicsGLRegisterImage(&m_cudaGfxResource, glTexID, GL_TEXTURE_2D, flags));
            m_GLTexID = glTexID;
        }
        else {
            if (m_numMipmapLevels > 1)
                CUDADRV_CHECK(cuMipmappedArrayCreate(&m_mipmappedArray, &arrayDesc, numMipmapLevels));
            else
                CUDADRV_CHECK(cuArray3DCreate(&m_array, &arrayDesc));
        }

        m_mappedPointers = new void*[m_numMipmapLevels];
        m_mappedArrays = new CUarray[m_numMipmapLevels];
        for (int i = 0; i < m_numMipmapLevels; ++i) {
            m_mappedPointers[i] = nullptr;
            m_mappedArrays[i] = nullptr;
        }

        m_initialized = true;
    }

    void Array::finalize() {
        if (!m_initialized)
            return;

        delete[] m_mappedArrays;
        delete[] m_mappedPointers;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        if (m_GLTexID != 0) {
            CUDADRV_CHECK(cuGraphicsUnregisterResource(m_cudaGfxResource));
            m_GLTexID = 0;
        }
        else {
            if (m_numMipmapLevels > 1)
                CUDADRV_CHECK(cuMipmappedArrayDestroy(m_mipmappedArray));
            else
                CUDADRV_CHECK(cuArrayDestroy(m_array));
        }

        m_initialized = false;
    }

    void Array::resize(uint32_t length) {
        if (m_height > 0 || m_depth > 0)
            throw std::runtime_error("Array dimension cannot be changed.");
        CUDAUAssert_NotImplemented();
    }

    void Array::resize(uint32_t width, uint32_t height) {
        if (m_depth > 0)
            throw std::runtime_error("Array dimension cannot be changed.");
        if (m_numMipmapLevels > 1)
            throw std::runtime_error("resize() is supported only on non-mipmapped array.");
        
        if (width == m_width && height == m_height)
            return;

        Array newArray;
        newArray.initialize(m_cudaContext, m_elemType, m_numChannels, width, height, m_depth, m_numMipmapLevels,
                            m_surfaceLoadStore, m_cubemap, m_layered, 0);

        size_t sizePerRow = std::min(m_width, width) * static_cast<size_t>(m_stride);

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = std::max<size_t>(1, std::min(m_height, height));
        params.Depth = std::max<size_t>(1, m_depth);

        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = m_array;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = m_array;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

        *this = std::move(newArray);
    }

    void Array::resize(uint32_t width, uint32_t height, uint32_t depth) {
        CUDAUAssert_NotImplemented();
    }

    void Array::beginCUDAAccess(CUstream stream, uint32_t mipmapLevel) {
        if (m_GLTexID == 0)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        CUDADRV_CHECK(cuGraphicsMapResources(1, &m_cudaGfxResource, stream));
        CUDADRV_CHECK(cuGraphicsSubResourceGetMappedArray(&m_mappedArrays[mipmapLevel], m_cudaGfxResource, 0, mipmapLevel));
    }

    void Array::endCUDAAccess(CUstream stream, uint32_t mipmapLevel) {
        if (m_GLTexID == 0)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        m_mappedArrays[mipmapLevel] = nullptr;
        CUDADRV_CHECK(cuGraphicsUnmapResources(1, &m_cudaGfxResource, stream));
    }

    void* Array::map(uint32_t mipmapLevel) {
        if (m_mappedPointers[mipmapLevel])
            throw std::runtime_error("This mip-map level is already mapped.");
        if (mipmapLevel >= m_numMipmapLevels)
            throw std::runtime_error("Specified mip-map level is out of bound.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        // JP: CUDAはミップマップレベル数の計算に問題を抱えているように見える。
        //     例: 64x64, BCフォーマットのテクスチャー(16x16ブロック)は次のミップレベルを持つことができる。
        //         64x64(16x16), 32x32(8x8), 16x16(4x4), 8x8(2x2), 4x4(1x1), 2x2(1x1), 1x1(1x1)
        //     しかしCUDAは最後のミップレベルが4x4(1x1)だと仮定しているように見える。
        //     それとも何か勘違いしている？
        // EN: CUDA seems to have an issue in calculation of the number of mipmap levels.
        //     e.g. 64x64 texture with BC format (16x16 blocks) can have the following mipmap levels:
        //          64x64(16x16), 32x32(8x8), 16x16(4x4), 8x8(2x2), 4x4(1x1), 2x2(1x1), 1x1(1x1)
        //     However CUDA seems to suppose that the last mip level is 4x4(1x1).
        //     Or do I misunderstand something?
        //
        // Bug reports:
        // https://forums.developer.nvidia.com/t/how-to-create-cudatextureobject-with-texture-raw-data-in-block-compressed-bc-format/119570
        // https://forums.developer.nvidia.com/t/mip-map-with-block-compressed-texture/72170
        if (m_numMipmapLevels > 1)
            CUDADRV_CHECK(cuMipmappedArrayGetLevel(&m_mappedArrays[mipmapLevel], m_mipmappedArray, mipmapLevel));

        uint32_t width = std::max<size_t>(1, m_width >> mipmapLevel);
        uint32_t height = std::max<size_t>(1, m_height >> mipmapLevel);
        uint32_t depth = std::max<size_t>(1, m_depth);
        size_t sizePerRow = width * static_cast<size_t>(m_stride);
        size_t size = depth * height * sizePerRow;
        m_mappedPointers[mipmapLevel] = new uint8_t[size];

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = height;
        params.Depth = depth;

        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = m_numMipmapLevels > 1 ? m_mappedArrays[mipmapLevel] : m_array;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_HOST;
        params.dstHost = m_mappedPointers[mipmapLevel];
        params.dstPitch = sizePerRow;
        params.dstHeight = height;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstArray, dstDevice, dstLOD are not used in this case.

        CUDADRV_CHECK(cuMemcpy3D(&params));

        return m_mappedPointers[mipmapLevel];
    }

    void Array::unmap(uint32_t mipmapLevel) {
        if (!m_mappedPointers[mipmapLevel])
            throw std::runtime_error("This mip-map level is not mapped.");
        if (mipmapLevel >= m_numMipmapLevels)
            throw std::runtime_error("Specified mip-map level is out of bound.");

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        uint32_t width = std::max<size_t>(1, m_width >> mipmapLevel);
        uint32_t height = std::max<size_t>(1, m_height >> mipmapLevel);
        uint32_t depth = std::max<size_t>(1, m_depth);
        size_t sizePerRow = width * static_cast<size_t>(m_stride);

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = height;
        params.Depth = depth;

        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        params.srcHost = m_mappedPointers[mipmapLevel];
        params.srcPitch = sizePerRow;
        params.srcHeight = height;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcArray, srcDevice, srcLOD are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = m_numMipmapLevels > 1 ? m_mappedArrays[mipmapLevel] : m_array;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

        CUDADRV_CHECK(cuMemcpy3D(&params));

        delete[] m_mappedPointers[mipmapLevel];
        m_mappedPointers[mipmapLevel] = nullptr;
    }

    CUDA_RESOURCE_VIEW_DESC Array::getResourceViewDesc() const {
        CUDA_RESOURCE_VIEW_DESC ret = {};
        bool isBC = isBCFormat(m_elemType);
        ret.format = getResourceViewFormat(m_elemType, m_numChannels);
        ret.width = (isBC ? 4 : 1) * m_width;
        ret.height = (isBC ? 4 : 1) * m_height;
        ret.depth = m_depth;
        ret.firstMipmapLevel = 0;
        ret.lastMipmapLevel = m_numMipmapLevels - 1;
        if (m_layered) {
            CUDAUAssert_NotImplemented();
        }
        else {
            ret.firstLayer = 0;
            ret.lastLayer = 0;
        }

        return ret;
    }
}
