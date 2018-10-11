#pragma once

#include "context.h"
#include "ext/include/half.hpp"

using half_float::half;

namespace VLR {
    // ----------------------------------------------------------------
    // Textures

    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    struct RGBA16Fx4 { half r, g, b, a; };
    struct RGBA32Fx4 { float r, g, b, a; };
    struct RG32Fx2 { float r, g; };
    struct Gray32F { float v; };
    struct Gray8 { uint8_t v; };

    extern const size_t sizesOfDataFormats[(uint32_t)DataFormat::Num];

    class Image2D : public Object {
        uint32_t m_width, m_height;
        DataFormat m_dataFormat;
        mutable optix::Buffer m_optixDataBuffer;
        mutable bool m_initOptiXObject;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static DataFormat getInternalFormat(DataFormat inputFormat);

        Image2D(Context &context, uint32_t width, uint32_t height, DataFormat dataFormat);
        virtual ~Image2D();

        virtual Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const = 0;
        virtual Image2D* createLuminanceImage2D() const = 0;
        virtual void* createLinearImageData() const = 0;

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        DataFormat getDataFormat() const {
            return m_dataFormat;
        }
        uint32_t getStride() const {
            return (uint32_t)sizesOfDataFormats[(uint32_t)m_dataFormat];
        }

        virtual optix::Buffer getOptiXObject() const;
    };



    class LinearImage2D : public Image2D {
        std::vector<uint8_t> m_data;
        mutable bool m_copyDone;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        // EN: "linearData" means data layout is linear.
        LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat, bool applyDegamma);

        template <typename PixelType>
        PixelType get(uint32_t x, uint32_t y) const {
            return *(PixelType*)(m_data.data() + (y * getWidth() + x) * getStride());
        }

        Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const override;
        Image2D* createLuminanceImage2D() const override;
        void* createLinearImageData() const override;

        optix::Buffer getOptiXObject() const override;
    };



    class TextureMap2D : public Object {
    protected:
        struct OptiXProgramSet {
            optix::Program callableProgramMap;
        };

        uint32_t m_texMapIndex;

        static void commonInitializeProcedure(Context &context, const char* identifiers[1], OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet);

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        TextureMap2D(Context &context);
        virtual ~TextureMap2D();

        uint32_t getTextureMapIndex() const {
            return m_texMapIndex;
        }

        virtual void setupTextureMapDescriptor(Shared::TextureMapDescriptor* texMapDesc) const = 0;
    };



    class OffsetAndScaleUVTextureMap2D : public TextureMap2D {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, OffsetAndScaleUVTextureMap2D*> s_defaultInstance;

        float m_offset[2];
        float m_scale[2];

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        OffsetAndScaleUVTextureMap2D(Context &context, const float offset[2], const float scale[2]);
        ~OffsetAndScaleUVTextureMap2D();

        void setupTextureMapDescriptor(Shared::TextureMapDescriptor* texMapDesc) const override;

        static const OffsetAndScaleUVTextureMap2D* getDefault(Context &context) {
            return s_defaultInstance.at(context.getID());
        }
    };



    class FloatTexture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        FloatTexture(Context &context);
        virtual ~FloatTexture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class Float2Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Float2Texture(Context &context);
        virtual ~Float2Texture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class ConstantFloat2Texture : public Float2Texture {
        Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ConstantFloat2Texture(Context &context, const float value[2]);
        ~ConstantFloat2Texture();
    };



    class Float3Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Float3Texture(Context &context);
        virtual ~Float3Texture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);

        virtual void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const {
            VLRAssert_NotImplemented();
        }
    };



    class ConstantFloat3Texture : public Float3Texture {
        Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ConstantFloat3Texture(Context &context, const float value[3]);
        ~ConstantFloat3Texture();
    };



    class ImageFloat3Texture : public Float3Texture {
        const Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ImageFloat3Texture(Context &context, const Image2D* image);

        void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const override;
    };



    class Float4Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Float4Texture(Context &context);
        virtual ~Float4Texture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class ConstantFloat4Texture : public Float4Texture {
        Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ConstantFloat4Texture(Context &context, const float value[4]);
        ~ConstantFloat4Texture();
    };



    class ImageFloat4Texture : public Float4Texture {
        const Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ImageFloat4Texture(Context &context, const Image2D* image);
    };

    // END: Textures
    // ----------------------------------------------------------------
}
