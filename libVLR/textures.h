#pragma once

#include "context.h"
#include "ext/include/half.hpp"

using half_float::half;

namespace VLR {
    class ShaderNode : public Object {
    protected:
        struct OptiXProgramSet {
            optix::Program callableProgram;
        };

        uint32_t m_nodeIndex;

        static void commonInitializeProcedure(Context &context, const char* identifiers[1], OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet);

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ShaderNode(Context &context);
        virtual ~ShaderNode();

        virtual bool hasTextureCoordinateOutput() const { return false; }
        virtual bool hasRGBSpectrumOutput() const { return false; }
        virtual bool hasFloatOutput() const { return false; }

        virtual void setupNodeDescriptor(Shared::NodeDescriptor* nodeDesc) const = 0;

        uint32_t getShaderNodeIndex() const { return m_nodeIndex; }
    };



    class OffsetAndScaleUVTextureMap2DShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, OffsetAndScaleUVTextureMap2DShaderNode*> s_defaultInstance;

        float m_offset[2];
        float m_scale[2];

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        OffsetAndScaleUVTextureMap2DShaderNode(Context &context, const float offset[2], const float scale[2]);
        ~OffsetAndScaleUVTextureMap2DShaderNode();

        bool hasTextureCoordinateOutput() const override { return true; }

        void setupNodeDescriptor(Shared::NodeDescriptor* nodeDesc) const override;

        static const OffsetAndScaleUVTextureMap2DShaderNode* getDefault(Context &context) {
            return s_defaultInstance.at(context.getID());
        }
    };



    class ConstantTextureShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, ConstantTextureShaderNode*> s_gray18;

        RGBSpectrum m_spectrum;
        float m_alpha;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ConstantTextureShaderNode(Context &context, const RGBSpectrum &spectrum, float alpha);
        ~ConstantTextureShaderNode();

        bool hasRGBSpectrumOutput() const override { return true; }
        bool hasFloatOutput() const override { return true; }

        void setupNodeDescriptor(Shared::NodeDescriptor* nodeDesc) const override;

        static const ConstantTextureShaderNode* getGray18(Context &context) {
            return s_gray18.at(context.getID());
        }
    };



    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    struct RGBA16Fx4 { half r, g, b, a; };
    struct RGBA32Fx4 { float r, g, b, a; };
    struct RG32Fx2 { float r, g; };
    struct Gray32F { float v; };
    struct Gray8 { uint8_t v; };

    extern const size_t sizesOfDataFormats[(uint32_t)VLRDataFormat::NumVLRDataFormats];

    class Image2D : public Object {
        uint32_t m_width, m_height;
        VLRDataFormat m_dataFormat;
        mutable optix::Buffer m_optixDataBuffer;
        mutable bool m_initOptiXObject;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static VLRDataFormat getInternalFormat(VLRDataFormat inputFormat);

        Image2D(Context &context, uint32_t width, uint32_t height, VLRDataFormat dataFormat);
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
        VLRDataFormat getDataFormat() const {
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

        // EN: "linearData" means data layout is linear, it doesn't mean gamma curve.
        LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat dataFormat, bool applyDegamma);

        template <typename PixelType>
        PixelType get(uint32_t x, uint32_t y) const {
            return *(PixelType*)(m_data.data() + (y * getWidth() + x) * getStride());
        }

        Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const override;
        Image2D* createLuminanceImage2D() const override;
        void* createLinearImageData() const override;

        optix::Buffer getOptiXObject() const override;
    };



    class Image2DTextureShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        const ShaderNode* m_nodeTexCoord;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Image2DTextureShaderNode(Context &context, const Image2D* image, const ShaderNode* nodeTexCoord);
        ~Image2DTextureShaderNode();

        bool hasRGBSpectrumOutput() const override { return true; }

        void setupNodeDescriptor(Shared::NodeDescriptor* nodeDesc) const override;

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
    };



    // ----------------------------------------------------------------
    // Textures

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

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
    };



    class ConstantFloatTexture : public FloatTexture {
        Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ConstantFloatTexture(Context &context, const float value);
        ~ConstantFloatTexture();
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

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
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

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);

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

        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
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
