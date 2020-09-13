#pragma once

#include "queryable.h"

namespace VLR {
    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    struct RGBA16Fx4 { half r, g, b, a; };
    struct RGBA32Fx4 { float r, g, b, a; };
    struct RG32Fx2 { float r, g; };
    struct Gray32F { float v; };
    struct Gray8 { uint8_t v; };
    struct GrayA8x2 { uint8_t v; uint8_t a; };
    struct uvsA8x4 { uint8_t u, v, s, a; };
    struct uvsA16Fx4 { half u, v, s, a; };

    extern const size_t sizesOfDataFormats[(uint32_t)DataFormat::NumFormats];

    uint32_t getComponentStartIndex(DataFormat dataFormat, BumpType bumpType, ShaderNodePlugType ptype, uint32_t index);

    class Image2D : public Queryable {
        uint32_t m_width, m_height;
        DataFormat m_originalDataFormat;
        DataFormat m_dataFormat;
        bool m_needsHW_sRGB_degamma;
        SpectrumType m_spectrumType;
        ColorSpace m_colorSpace;

    protected:
        mutable cudau::Array m_optixDataBuffer;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        static DataFormat getInternalFormat(DataFormat inputFormat, SpectrumType spectrumType);

        Image2D(Context &context, uint32_t width, uint32_t height,
                DataFormat originalDataFormat, SpectrumType spectrumType, ColorSpace colorSpace);
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
        uint32_t getStride() const {
            return (uint32_t)sizesOfDataFormats[(uint32_t)m_dataFormat];
        }
        DataFormat getDataFormat() const {
            return m_dataFormat;
        }
        DataFormat getOriginalDataFormat() const {
          return m_originalDataFormat;
        }
        bool originalHasAlpha() const {
            return (m_originalDataFormat == DataFormat::RGBA8x4 ||
                    m_originalDataFormat == DataFormat::RGBA16Fx4 ||
                    m_originalDataFormat == DataFormat::RGBA32Fx4 ||
                    m_originalDataFormat == DataFormat::GrayA8x2 ||
                    m_originalDataFormat == DataFormat::BC1 ||
                    m_originalDataFormat == DataFormat::BC2 ||
                    m_originalDataFormat == DataFormat::BC3 ||
                    m_originalDataFormat == DataFormat::BC7);
        }
        bool needsHW_sRGB_degamma() const {
            return m_needsHW_sRGB_degamma;
        }
        SpectrumType getSpectrumType() const {
            return m_spectrumType;
        }
        ColorSpace getColorSpace() const {
            return m_colorSpace;
        }

        virtual const cudau::Array &getOptiXObject() const;
    };



    class LinearImage2D : public Image2D {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        std::vector<uint8_t> m_data;
        mutable bool m_copyDone;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        // JP: "linearData" はメモリ上のレイアウトがリニアであることを意味しており、ガンマカーブ云々を表しているのではない。
        // EN: "linearData" means data layout is linear, it doesn't mean gamma curve.
        LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height,
                      DataFormat dataFormat, SpectrumType spectrumType, ColorSpace colorSpace);

        template <typename PixelType>
        PixelType get(uint32_t x, uint32_t y) const {
            return *(PixelType*)(m_data.data() + (y * getWidth() + x) * getStride());
        }

        Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const override;
        Image2D* createLuminanceImage2D() const override;
        void* createLinearImageData() const override;

        const cudau::Array &getOptiXObject() const override;
    };



    class BlockCompressedImage2D : public Image2D {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        std::vector<std::vector<uint8_t>> m_data;
        mutable bool m_copyDone;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        BlockCompressedImage2D(Context &context, const uint8_t* const* data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                               DataFormat dataFormat, SpectrumType spectrumType, ColorSpace colorSpace);

        Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const override;
        Image2D* createLuminanceImage2D() const override;
        void* createLinearImageData() const override;

        const cudau::Array &getOptiXObject() const override;
    };
}
