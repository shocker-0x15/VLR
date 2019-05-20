#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <map>

#include "GL/gl3w.h"

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define GLTKPlatform_Windows
#   if defined(__MINGW32__) // Defined for both 32 bit/64 bit MinGW
#       define GLTKPlatform_Windows_MinGW
#   elif defined(_MSC_VER)
#       define GLTKPlatform_Windows_MSVC
#   endif
#elif defined(__linux__)
#   define GLTKPlatform_Linux
#elif defined(__APPLE__)
#   define GLTKPlatform_macOS
#elif defined(__OpenBSD__)
#   define GLTKPlatform_OpenBSD
#endif

#ifdef GLTKPlatform_Windows_MSVC
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#endif

#ifdef GLTKPlatform_Windows_MSVC
static void GLTKDebugPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[1024];
    vsprintf_s(str, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#else
#   define GLTKDebugPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#ifdef _DEBUG
#   define GLTK_ENABLE_ASSERT
#endif

#ifdef GLTK_ENABLE_ASSERT
#   define GLTKAssert(expr, fmt, ...) if (!(expr)) { GLTKDebugPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); GLTKDebugPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define GLTKAssert(expr, fmt, ...)
#endif

#define GLTKAssert_ShouldNotBeCalled() GLTKAssert(false, "Should not be called!")
#define GLTKAssert_NotImplemented() GLTKAssert(false, "Not implemented yet!")



namespace GLTK {
    enum class Error {
        NoError = GL_NO_ERROR,
        InvalidEnum = GL_INVALID_ENUM,
        InvalidValue = GL_INVALID_VALUE,
        InvalidOperation = GL_INVALID_OPERATION,
        InvalidFramebufferOperation = GL_INVALID_FRAMEBUFFER_OPERATION,
        OutOfMemory = GL_OUT_OF_MEMORY,
        StackUnderflow = GL_STACK_UNDERFLOW,
        StackOverflow = GL_STACK_OVERFLOW,
    };

    static std::string getErrorString(Error err) {
        switch (err) {
        case Error::NoError:
            return "NoError: No error has been recorded. The value of this symbolic constant is guaranteed to be 0.";
        case Error::InvalidEnum:
            return "InvalidEnum: An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.";
        case Error::InvalidValue:
            return "InvalidValue: A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.";
        case Error::InvalidOperation:
            return "InvalidOperation: The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.";
        case Error::InvalidFramebufferOperation:
            return "InvalidFramebufferOperation: The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag.";
        case Error::OutOfMemory:
            return "OutOfMemory: There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
        case Error::StackUnderflow:
            return "StackUnderflow: An attempt has been made to perform an operation that would cause an internal stack to underflow.";
        case Error::StackOverflow:
            return "StackOverflow: An attempt has been made to perform an operation that would cause an internal stack to overflow.";
        default:
            break;
        }
        return "";
    }

    static void printErrorString() {
        GLTKDebugPrintf("%s\n", getErrorString((Error)glGetError()).c_str());
    }

    static inline void errorCheck() {
#if defined(GLTK_ENABLE_ASSERT)
        Error error = (Error)glGetError();
        GLTKAssert(error == Error::NoError, "%s", getErrorString(error).c_str());
#endif
    }



    struct InternalFormat {
        int32_t value;

        InternalFormat(int32_t v) : value(v) {}

        operator GLint() const {
            return (GLint)value;
        }

        operator GLenum() const {
            return (GLenum)value;
        }
    };

    struct BaseInternalFormat {
        enum Value {
            DepthComponent = GL_DEPTH_COMPONENT,
            DepthStencil = GL_DEPTH_STENCIL,
            RED = GL_RED,
            RG = GL_RG,
            RGB = GL_RGB,
            RGBA = GL_RGBA,
        };
        Value value;

        BaseInternalFormat(Value v) : value(v) {}

        operator InternalFormat() const {
            return InternalFormat(value);
        }

        operator GLenum() const {
            return (GLenum)value;
        }
    };

    struct SizedInternalFormat {
        enum Value {
            R8 = GL_R8,
            R8_SNorm = GL_R8_SNORM,
            R16 = GL_R16,
            R16_SNorm = GL_R16_SNORM,
            RG8 = GL_RG8,
            RG8_SNorm = GL_RG8_SNORM,
            RG16 = GL_RG16,
            RG16_SNorm = GL_RG16_SNORM,
            R3_G3_B2 = GL_R3_G3_B2,
            RGB4 = GL_RGB4,
            RGB5 = GL_RGB5,
            RGB8 = GL_RGB8,
            RGB8_SNorm = GL_RGB8_SNORM,
            RGB10 = GL_RGB10,
            RGB12 = GL_RGB12,
            RGB16_SNorm = GL_RGB16_SNORM,
            RGBA2 = GL_RGBA2,
            RGBA4 = GL_RGBA4,
            RGB5_A1 = GL_RGB5_A1,
            RGBA8 = GL_RGBA8,
            RGBA8_SNorm = GL_RGBA8_SNORM,
            RGB10_A2 = GL_RGB10_A2,
            RGB10_A2UI = GL_RGB10_A2UI,
            RGBA12 = GL_RGBA12,
            RGBA16 = GL_RGBA16,
            SRGB8 = GL_SRGB8,
            SRGB8_ALPHA8 = GL_SRGB8_ALPHA8,
            R16F = GL_R16F,
            RG16F = GL_RG16F,
            RGB16F = GL_RGB16F,
            RGBA16F = GL_RGBA16F,
            R32F = GL_R32F,
            RG32F = GL_RG32F,
            RGB32F = GL_RGB32F,
            RGBA32F = GL_RGBA32F,
            R11F_G11F_B10F = GL_R11F_G11F_B10F,
            RGB9_E5 = GL_RGB9_E5,
            R8I = GL_R8I,
            R8UI = GL_R8UI,
            R16I = GL_R16I,
            R16UI = GL_R16UI,
            R32I = GL_R32I,
            R32UI = GL_R32UI,
            RG8I = GL_RG8I,
            RG8UI = GL_RG8UI,
            RG16I = GL_RG16I,
            RG16UI = GL_RG16UI,
            RG32I = GL_RG32I,
            RG32UI = GL_RG32UI,
            RGB8I = GL_RGB8I,
            RGB8UI = GL_RGB8UI,
            RGB16I = GL_RGB16I,
            RGB16UI = GL_RGB16UI,
            RGB32I = GL_RGB32I,
            RGB32UI = GL_RGB32UI,
            RGBA8I = GL_RGBA8I,
            RGBA8UI = GL_RGBA8UI,
            RGBA16I = GL_RGBA16I,
            RGBA16UI = GL_RGBA16UI,
            RGBA32I = GL_RGBA32I,
            RGBA32UI = GL_RGBA32UI,
        };
        Value value;

        SizedInternalFormat(Value v) : value(v) {}

        operator InternalFormat() const {
            return InternalFormat(value);
        }

        operator GLenum() const {
            return (GLenum)value;
        }
    };

    struct CompressedInternalFormat {
        enum Value {
            Compressed_Red = GL_COMPRESSED_RED,
            Compressed_RG = GL_COMPRESSED_RG,
            Compressed_RGB = GL_COMPRESSED_RGB,
            Compressed_RGBA = GL_COMPRESSED_RGBA,
            Compressed_sRGB = GL_COMPRESSED_SRGB,
            Compressed_sRGB_Alpha = GL_COMPRESSED_SRGB_ALPHA,
            Compressed_Red_RGTC1 = GL_COMPRESSED_RED_RGTC1,
            Compressed_SignedRed_RGTC1 = GL_COMPRESSED_SIGNED_RED_RGTC1,
            Compressed_RG_RGTC2 = GL_COMPRESSED_RG_RGTC2,
            Compressed_SignedRG_RGTC2 = GL_COMPRESSED_SIGNED_RG_RGTC2,
        };
        Value value;

        CompressedInternalFormat(Value v) : value(v) {}

        operator InternalFormat() const {
            return InternalFormat(value);
        }

        operator GLenum() const {
            return (GLenum)value;
        }
    };

    struct Format {
        enum Value {
            Red = GL_RED,
            RG = GL_RG,
            RGB = GL_RGB,
            BGR = GL_BGR,
            RGBA = GL_RGBA,
            BGRA = GL_BGRA,
            Red_Integer = GL_RED_INTEGER,
            RG_Integer = GL_RG_INTEGER,
            RGB_Integer = GL_RGB_INTEGER,
            BGR_Integer = GL_BGR_INTEGER,
            RGBA_Integer = GL_RGBA_INTEGER,
            BGRA_Integer = GL_BGRA_INTEGER,
            StencilIndex = GL_STENCIL_INDEX,
            DepthComponent = GL_DEPTH_COMPONENT,
            DepthStencil = GL_DEPTH_STENCIL,
        };
        Value value;

        Format(Value v) : value(v) {}

        operator GLenum() const {
            return (GLenum)value;
        }
    };

    struct NumericalType {
        enum Value {
            UnsignedByte = GL_UNSIGNED_BYTE,
            Byte = GL_BYTE,
            UnsignedShort = GL_UNSIGNED_SHORT,
            Short = GL_SHORT,
            UnsignedInt = GL_UNSIGNED_INT,
            Int = GL_INT,
            Float = GL_FLOAT,
            UnsignedByte332 = GL_UNSIGNED_BYTE_3_3_2,
            UnsignedByte233Rev = GL_UNSIGNED_BYTE_2_3_3_REV,
            UnsignedShort565 = GL_UNSIGNED_SHORT_5_6_5,
            UnsignedShort565Rev = GL_UNSIGNED_SHORT_5_6_5_REV,
            UnsignedShort4444 = GL_UNSIGNED_SHORT_4_4_4_4,
            UnsignedShort4444Rev = GL_UNSIGNED_SHORT_4_4_4_4_REV,
            UnsignedShort5551 = GL_UNSIGNED_SHORT_5_5_5_1,
            UnsignedShort1555Rev = GL_UNSIGNED_SHORT_1_5_5_5_REV,
            UnsignedInt8888 = GL_UNSIGNED_INT_8_8_8_8,
            UnsignedInt8888Rev = GL_UNSIGNED_INT_8_8_8_8_REV,
            UnsignedInt1010102 = GL_UNSIGNED_INT_10_10_10_2,
            UnsignedInt2101010Rev = GL_UNSIGNED_INT_2_10_10_10_REV,
        };
        Value value;

        NumericalType(Value v) : value(v) {}

        operator GLenum() const {
            return (GLenum)value;
        }
    };



    class Buffer {
    public:
        enum class Target {
            ArrayBuffer = GL_ARRAY_BUFFER,
            AtomicCounterBuffer = GL_ATOMIC_COUNTER_BUFFER,
            CopyReadBuffer = GL_COPY_READ_BUFFER,
            CopyWriteBuffer = GL_COPY_WRITE_BUFFER,
            DispatchIndirectBuffer = GL_DISPATCH_INDIRECT_BUFFER,
            DrawIndirectBuffer = GL_DRAW_INDIRECT_BUFFER,
            ElementArrayBuffer = GL_ELEMENT_ARRAY_BUFFER,
            PixelPackBuffer = GL_PIXEL_PACK_BUFFER,
            PixelUnpackBuffer = GL_PIXEL_UNPACK_BUFFER,
            ShaderStorageBuffer = GL_SHADER_STORAGE_BUFFER,
            TextureBuffer = GL_TEXTURE_BUFFER,
            TransformFeedbackBuffer = GL_TRANSFORM_FEEDBACK_BUFFER,
            UniformBuffer = GL_UNIFORM_BUFFER,
            Unbound = 0,
        };

        enum class Usage {
            StreamDraw = GL_STREAM_DRAW,
            StreamRead = GL_STREAM_READ,
            StreamCopy = GL_STREAM_COPY,
            StaticDraw = GL_STATIC_DRAW,
            StaticRead = GL_STATIC_READ,
            StaticCopy = GL_STATIC_COPY,
            DynamicDraw = GL_DYNAMIC_DRAW,
            DynamicRead = GL_DYNAMIC_READ,
            DynamicCopy = GL_DYNAMIC_COPY,
        };

    private:
        GLuint m_handle;
        Target m_target;
        uint32_t m_stride;
        uint32_t m_numElements;
        Usage m_usage;

    public:
        Buffer() : m_handle(0), m_stride(0), m_numElements(0) {}
        ~Buffer() {
            if (m_handle)
                finalize();
        }

        void initialize(Target target, uint32_t stride, uint32_t numElements, void* data, Usage usage) {
            glGenBuffers(1, &m_handle); errorCheck();
            m_target = target;
            m_stride = stride;
            m_numElements = numElements;
            m_usage = usage;

            bind();

            glBufferData((GLenum)m_target, m_stride * m_numElements, data, (GLenum)m_usage); errorCheck();

            unbind();
        }

        void finalize() {
            if (m_handle) {
                glDeleteBuffers(1, &m_handle); errorCheck();
            }
            m_handle = 0;
        }

        GLuint getRawHandle() const {
            return m_handle;
        }

        void bind() const {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glBindBuffer((GLenum)m_target, m_handle); errorCheck();
        }

        void unbind() const {
            glBindBuffer((GLenum)m_target, 0); errorCheck();
        }
    };



    class VertexArray {
    public:
    private:
        GLuint m_handle;

    public:
        VertexArray() : m_handle(0) {}
        ~VertexArray() {
            if (m_handle)
                finalize();
        }

        void initialize() {
            glGenVertexArrays(1, &m_handle); errorCheck();
        }

        void finalize() {
            if (m_handle) {
                glDeleteVertexArrays(1, &m_handle); errorCheck();
            }
            m_handle = 0;
        }

        void bind() const {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glBindVertexArray(m_handle); errorCheck();
        }

        void unbind() const {
            glBindVertexArray(0); errorCheck();
        }
    };



    class BufferTexture {
    public:
    private:
        GLuint m_handle;

    public:
        BufferTexture() : m_handle(0) {}
        ~BufferTexture() {
            if (m_handle)
                finalize();
        }

        void initialize(const Buffer &buffer, SizedInternalFormat internalFormat) {
            glGenTextures(1, &m_handle); errorCheck();

            bind();

            glTexBuffer(GL_TEXTURE_BUFFER, internalFormat, buffer.getRawHandle()); errorCheck();

            unbind();
        }

        void finalize() {
            if (m_handle) {
                glDeleteTextures(1, &m_handle); errorCheck();
            }
            m_handle = 0;
        }

        void bind() const {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glBindTexture(GL_TEXTURE_BUFFER, m_handle); errorCheck();
        }

        void unbind() const {
            glBindTexture(GL_TEXTURE_BUFFER, 0); errorCheck();
        }
    };



    class Texture2D {
    public:
    private:
        GLuint m_handle;
        uint32_t m_width;
        uint32_t m_height;

    public:
        Texture2D() : m_handle(0), m_width(0), m_height(0) {}
        ~Texture2D() {
            if (m_handle)
                finalize();
        }

        void initialize(uint32_t width, uint32_t height, InternalFormat internalFormat) {
            glGenTextures(1, &m_handle); errorCheck();
            m_width = width;
            m_height = height;

            bind();

            glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_width, m_height, 0, Format::Red, NumericalType::UnsignedByte, nullptr); errorCheck();

            unbind();
        }

        void finalize() {
            if (m_handle) {
                glDeleteTextures(1, &m_handle); errorCheck();
            }
            m_handle = 0;
        }

        GLuint getRawHandle() const {
            return m_handle;
        }

        void bind() const {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glBindTexture(GL_TEXTURE_2D, m_handle); errorCheck();
        }

        void unbind() const {
            glBindTexture(GL_TEXTURE_2D, 0); errorCheck();
        }
    };



    class Sampler {
    public:
        enum class MinFilter {
            Nearest = GL_NEAREST,
            Linear = GL_LINEAR,
            NearestMipMapNearest = GL_NEAREST_MIPMAP_NEAREST,
            LinearMipMapNearest = GL_LINEAR_MIPMAP_NEAREST,
            NearestMipMapLinear = GL_NEAREST_MIPMAP_LINEAR,
            LinearMipMapLinear = GL_LINEAR_MIPMAP_LINEAR,
        };
        enum class MagFilter {
            Nearest = GL_NEAREST,
            Linear = GL_LINEAR,
        };

        enum class WrapMode {
            Repeat = GL_REPEAT,
            ClampToEdge = GL_CLAMP_TO_EDGE,
            ClampToBorder = GL_CLAMP_TO_BORDER,
            MirroredRepeat = GL_MIRRORED_REPEAT,
        };

    private:
        GLuint m_handle;

    public:
        Sampler() : m_handle(0) {}
        ~Sampler() {
            if (m_handle)
                finalize();
        }

        void initialize(MinFilter minFilter, MagFilter magFilter, WrapMode wrapModeS, WrapMode wrapModeT) {
            glGenSamplers(1, &m_handle); errorCheck();
            glSamplerParameteri(m_handle, (GLuint)GL_TEXTURE_MIN_FILTER, (GLint)minFilter); errorCheck();
            glSamplerParameteri(m_handle, (GLuint)GL_TEXTURE_MAG_FILTER, (GLint)magFilter); errorCheck();
            glSamplerParameteri(m_handle, (GLuint)GL_TEXTURE_WRAP_S, (GLint)wrapModeS); errorCheck();
            glSamplerParameteri(m_handle, (GLuint)GL_TEXTURE_WRAP_T, (GLint)wrapModeT); errorCheck();
        }

        void finalize() {
            if (m_handle) {
                glDeleteSamplers(1, &m_handle); errorCheck();
            }
            m_handle = 0;
        }

        void bindToTextureUnit(GLuint unit) const {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glBindSampler(unit, m_handle); errorCheck();
        }
    };



    class FrameBuffer {
    public:
        enum class Target {
            Draw = GL_DRAW_FRAMEBUFFER,
            Read = GL_READ_FRAMEBUFFER,
            ReadDraw = GL_FRAMEBUFFER,
            Unbound = 0,
        };

        enum class Status {
            Complete = GL_FRAMEBUFFER_COMPLETE,
            Undefined = GL_FRAMEBUFFER_UNDEFINED,
            IncompleteAttachment = GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT,
            IncompleteMissingAttachment = GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT,
            IncompleteDrawBuffer = GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER,
            IncompleteReadBuffer = GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER,
            Unsupported = GL_FRAMEBUFFER_UNSUPPORTED,
            IncompleteMultisample = GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE,
            IncompleteLayerTargets = GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS,
        };

        static void checkStatus(Target target) {
            Status status = (Status)glCheckFramebufferStatus((GLenum)target); errorCheck();
            switch (status) {
            case Status::Complete:
                break;
            case Status::Undefined:
                GLTKDebugPrintf("The specified framebuffer is the default read or draw framebuffer, but the default framebuffer does not exist.\n");
                break;
            case Status::IncompleteAttachment:
                GLTKDebugPrintf("Any of the framebuffer attachment points are framebuffer incomplete.\n");
                break;
            case Status::IncompleteMissingAttachment:
                GLTKDebugPrintf("The framebuffer does not have at least one image attached to it.\n");
                break;
            case Status::IncompleteDrawBuffer:
                GLTKDebugPrintf("The value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAW_BUFFERi.\n");
                break;
            case Status::IncompleteReadBuffer:
                GLTKDebugPrintf("GL_READ_BUFFER is not GL_NONE and the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER.\n");
                break;
            case Status::Unsupported:
                GLTKDebugPrintf("The combination of internal formats of the attached images violates an implementation-dependent set of restrictions.\n");
                break;
            case Status::IncompleteMultisample:
                GLTKDebugPrintf("The value of GL_RENDERBUFFER_SAMPLES is not the same for all attached renderbuffers; if the value of GL_TEXTURE_SAMPLES is the not same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_RENDERBUFFER_SAMPLES does not match the value of GL_TEXTURE_SAMPLES.\n");
                GLTKDebugPrintf("The value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not GL_TRUE for all attached textures.\n");;
                break;
            case Status::IncompleteLayerTargets:
                GLTKDebugPrintf("Any framebuffer attachment is layered, and any populated attachment is not layered, or if all populated color attachments are not from textures of the same target.\n");
                break;
            default:
                break;
            }
        }

    private:
        GLuint m_handle;
        //GLuint m_renderTargetHandle;
        Texture2D m_renderTargetTexture;
        GLuint m_depthRenderTargetHandle;
        uint32_t m_width;
        uint32_t m_height;
        Target m_curTarget;

    public:
        FrameBuffer() : m_handle(0), m_depthRenderTargetHandle(0), m_width(0), m_height(0) {}
        ~FrameBuffer() {
            if (m_handle)
                finalize();
        }

        void initialize(uint32_t width, uint32_t height, InternalFormat internalFormat, InternalFormat depthInternalFormat) {
            glGenFramebuffers(1, &m_handle); errorCheck();
            m_width = width;
            m_height = height;

            bind(Target::ReadDraw);

            // JP: テクスチャー経由でレンダーターゲットを初期化する。
            m_renderTargetTexture.initialize(m_width, m_height, internalFormat);
            m_renderTargetTexture.bind();
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_renderTargetTexture.getRawHandle(), 0); errorCheck();
            m_renderTargetTexture.unbind();

            // JP: デプスレンダーターゲットの初期化。
            glGenRenderbuffers(1, &m_depthRenderTargetHandle); errorCheck();
            glBindRenderbuffer(GL_RENDERBUFFER, m_depthRenderTargetHandle); errorCheck();
            glRenderbufferStorage(GL_RENDERBUFFER, depthInternalFormat, m_width, m_height); errorCheck();
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthRenderTargetHandle); errorCheck();
            glBindRenderbuffer(GL_RENDERBUFFER, 0); errorCheck();

            checkStatus(Target::ReadDraw);

            unbind();
        }

        void finalize() {
            if (m_depthRenderTargetHandle) {
                glDeleteRenderbuffers(1, &m_depthRenderTargetHandle); errorCheck();
            }
            m_depthRenderTargetHandle = 0;

            m_renderTargetTexture.finalize();

            if (m_handle) {
                glDeleteFramebuffers(1, &m_handle); errorCheck();
            }
            m_handle = 0;
        }

        GLuint getRawHandle() const {
            return m_handle;
        }

        Texture2D &getRenderTargetTexture() {
            return m_renderTargetTexture;
        }

        uint32_t getWidth() const {
            return m_width;
        }

        uint32_t getHeight() const {
            return m_height;
        }

        void bind(Target target) {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glBindFramebuffer((GLenum)target, m_handle); errorCheck();
            m_curTarget = target;
        }

        void unbind() {
            glBindFramebuffer((GLenum)m_curTarget, 0); errorCheck();
            m_curTarget = Target::Unbound;
        }
    };



    class GraphicsShader {
    public:
    private:
        GLuint m_handle;
        GLuint m_VSHandle;
        GLuint m_PSHandle;
        std::map<std::string, GLuint> m_attribLocations;

        static GLuint compileShader(GLenum shaderType, const std::string &source) {
            GLuint handle = glCreateShader(shaderType); errorCheck();

            const GLchar* glStrSource = (const GLchar*)source.c_str();
            glShaderSource(handle, 1, &glStrSource, NULL); errorCheck();
            glCompileShader(handle); errorCheck();

            GLint logLength;
            glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLength); errorCheck();
            if (logLength > 0) {
                GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
                glGetShaderInfoLog(handle, logLength, &logLength, log); errorCheck();
                GLTKDebugPrintf("Shader Compile Error:\n%s\n", log);
                free(log);
            }

            GLint status;
            glGetShaderiv(handle, GL_COMPILE_STATUS, &status); errorCheck();
            if (status == 0) {
                glDeleteShader(handle); errorCheck();
                return 0;
            }

            return handle;
        }

        void validateProgram() const {
            GLint status;
            GLint logLength;
            glValidateProgram(m_handle); errorCheck();
            glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &logLength); errorCheck();
            if (logLength > 0) {
                GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
                glGetProgramInfoLog(m_handle, logLength, &logLength, log); errorCheck();
                GLTKDebugPrintf("%s", log);
                free(log);
            }
            glGetProgramiv(m_handle, GL_VALIDATE_STATUS, &status); errorCheck();
            if (status == 0) {
                GLTKDebugPrintf("Program Status : GL_FALSE\n");
            }
        }
    public:
        GraphicsShader() : m_handle(0), m_VSHandle(0), m_PSHandle(0) {}
        ~GraphicsShader() {
            if (m_handle)
                finalize();
        }

        void initializeVSPS(const std::string &vsSource, const std::string &psSource) {
            m_VSHandle = compileShader(GL_VERTEX_SHADER, vsSource);
            m_PSHandle = compileShader(GL_FRAGMENT_SHADER, psSource);
            if (m_VSHandle == 0 || m_PSHandle == 0)
                return;

            m_handle = glCreateProgram(); errorCheck();
            glAttachShader(m_handle, m_VSHandle); errorCheck();
            glAttachShader(m_handle, m_PSHandle); errorCheck();
            glLinkProgram(m_handle); errorCheck();

            GLint logLength;
            glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &logLength); errorCheck();
            if (logLength > 0) {
                GLchar* log = (GLchar*)malloc(logLength * sizeof(GLchar));
                glGetProgramInfoLog(m_handle, logLength, &logLength, log); errorCheck();
                GLTKDebugPrintf("%s\n", log);
                free(log);
            }

            GLint status;
            glGetProgramiv(m_handle, GL_LINK_STATUS, &status); errorCheck();
            if (status == 0) {
                finalize();
                return;
            }

            validateProgram();
        }

        void finalize() {
            if (m_handle) {
                glDeleteProgram(m_handle); errorCheck();
            }
            m_handle = 0;

            if (m_PSHandle) {
                glDeleteShader(m_PSHandle); errorCheck();
            }
            m_PSHandle = 0;

            if (m_VSHandle) {
                glDeleteShader(m_VSHandle); errorCheck();
            }
            m_VSHandle = 0;
        }

        void useProgram() const {
            GLTKAssert(m_handle != 0, "This is an invalid object.");
            glUseProgram(m_handle); errorCheck();
        }
    };



    class ComputeShader {
    public:
    private:
    public:
    };
}
