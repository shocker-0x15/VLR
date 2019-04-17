#include "shader_nodes.h"

namespace VLR {
    const size_t sizesOfDataFormats[DataFormat::NumDataFormats] = {
    sizeof(RGB8x3),
    sizeof(RGB_8x4),
    sizeof(RGBA8x4),
    sizeof(RGBA16Fx4),
    sizeof(RGBA32Fx4),
    sizeof(RG32Fx2),
    sizeof(Gray32F),
    sizeof(Gray8),
    sizeof(GrayA8x2),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    sizeof(uvsA8x4),
    sizeof(uvsA16Fx4),
    };

    uint32_t getComponentStartIndex(DataFormat dataFormat, VLRShaderNodeSocketType stype, uint32_t index) {
        uint32_t ret = 0xFFFFFFFF;

        switch (dataFormat.value) {
        case DataFormat::RGBA8x4:
        case DataFormat::RGBA16Fx4:
        case DataFormat::RGBA32Fx4:
        case DataFormat::BC1:
        case DataFormat::BC2:
        case DataFormat::BC3:
        case DataFormat::BC7: {
            switch (stype) {
            case VLRShaderNodeSocketType_float:
                if (index < 4)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float2:
                if (index < 3)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float3:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float4:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Point3D:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Vector3D:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Normal3D:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Alpha:
                if (index < 1)
                    ret = 3;
                break;
            case VLRShaderNodeSocketType_TextureCoordinates:
                if (index < 3)
                    ret = index;
                break;
            default:
                break;
            }
            break;
        }
        case DataFormat::BC6H:
        case DataFormat::BC6H_Signed: {
            switch (stype) {
            case VLRShaderNodeSocketType_float:
                if (index < 3)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float2:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float3:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Point3D:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Vector3D:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Normal3D:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_TextureCoordinates:
                if (index < 2)
                    ret = index;
                break;
            default:
                break;
            }
            break;
        }
        case DataFormat::RG32Fx2:
        case DataFormat::BC5:
        case DataFormat::BC5_Signed: {
            switch (stype) {
            case VLRShaderNodeSocketType_float:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float2:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_TextureCoordinates:
                if (index < 1)
                    ret = index;
                break;
            default:
                break;
            }
            break;
        }
        case DataFormat::Gray32F:
        case DataFormat::Gray8:
        case DataFormat::BC4:
        case DataFormat::BC4_Signed: {
            switch (stype) {
            case VLRShaderNodeSocketType_float:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Spectrum:
                if (index < 1)
                    ret = index;
                break;
            default:
                break;
            }
            break;
        }
        case DataFormat::GrayA8x2: {
            switch (stype) {
            case VLRShaderNodeSocketType_float:
                if (index < 2)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_float2:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Alpha:
                if (index < 1)
                    ret = 1;
                break;
            case VLRShaderNodeSocketType_TextureCoordinates:
                if (index < 1)
                    ret = index;
                break;
            default:
                break;
            }
            break;
        }
        case DataFormat::uvsA8x4:
        case DataFormat::uvsA16Fx4: {
            switch (stype) {
            case VLRShaderNodeSocketType_Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case VLRShaderNodeSocketType_Alpha:
                if (index < 1)
                    ret = 3;
                break;
            default:
                break;
            }
            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }

        return ret;
    }



    DataFormat Image2D::getInternalFormat(DataFormat inputFormat, VLRSpectrumType spectrumType) {
        const auto asis = [](DataFormat inputFormat) {
            switch (inputFormat.value) {
            case DataFormat::RGB8x3:
                return DataFormat::RGBA8x4;
            case DataFormat::RGB_8x4:
                return DataFormat::RGBA8x4;
            case DataFormat::RGBA8x4:
                return DataFormat::RGBA8x4;
            case DataFormat::RGBA16Fx4:
                return DataFormat::RGBA16Fx4;
            case DataFormat::RGBA32Fx4:
                return DataFormat::RGBA32Fx4;
            case DataFormat::RG32Fx2:
                return DataFormat::RG32Fx2;
            case DataFormat::Gray32F:
                return DataFormat::Gray32F;
            case DataFormat::Gray8:
                return DataFormat::Gray8;
            case DataFormat::GrayA8x2:
                return DataFormat::GrayA8x2;
            case DataFormat::BC1:
                return DataFormat::BC1;
            case DataFormat::BC2:
                return DataFormat::BC2;
            case DataFormat::BC3:
                return DataFormat::BC3;
            case DataFormat::BC4:
                return DataFormat::BC4;
            case DataFormat::BC4_Signed:
                return DataFormat::BC4_Signed;
            case DataFormat::BC5:
                return DataFormat::BC5;
            case DataFormat::BC5_Signed:
                return DataFormat::BC5_Signed;
            case DataFormat::BC6H:
                return DataFormat::BC6H;
            case DataFormat::BC6H_Signed:
                return DataFormat::BC6H_Signed;
            case DataFormat::BC7:
                return DataFormat::BC7;
            case DataFormat::uvsA8x4:
                return DataFormat::uvsA8x4;
            case DataFormat::uvsA16Fx4:
                return DataFormat::uvsA16Fx4;
            default:
                VLRAssert(false, "Data format is invalid.");
                break;
            }
            return (DataFormat::Value)0;
        };

        if (spectrumType == VLRSpectrumType_NA) {
            return asis(inputFormat);
        }
        else {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            switch (inputFormat.value) {
            case DataFormat::RGB8x3:
                if (spectrumType == VLRSpectrumType_Reflectance)
                    return DataFormat::uvsA8x4;
                else
                    return DataFormat::uvsA16Fx4;
            case DataFormat::RGB_8x4:
                if (spectrumType == VLRSpectrumType_Reflectance)
                    return DataFormat::uvsA8x4;
                else
                    return DataFormat::uvsA16Fx4;
            case DataFormat::RGBA8x4:
                if (spectrumType == VLRSpectrumType_Reflectance)
                    return DataFormat::uvsA8x4;
                else
                    return DataFormat::uvsA16Fx4;
            case DataFormat::RGBA16Fx4:
                return DataFormat::uvsA16Fx4;
            case DataFormat::RGBA32Fx4:
                VLRAssert_NotImplemented();
                return DataFormat::uvsA16Fx4;
            default:
                break;
            }
            return asis(inputFormat);
#else
            return asis(inputFormat);
#endif
        }
        return DataFormat((DataFormat::Value)0);
    }

    Image2D::Image2D(Context &context, uint32_t width, uint32_t height,
                     DataFormat originalDataFormat, VLRSpectrumType spectrumType, ColorSpace colorSpace) :
        Object(context), m_width(width), m_height(height), 
        m_originalDataFormat(originalDataFormat), m_spectrumType(spectrumType), m_colorSpace(colorSpace),
        m_initOptiXObject(false)
    {
        m_dataFormat = getInternalFormat(m_originalDataFormat, spectrumType);
        m_needsHW_sRGB_degamma = false;
        if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma) {
            if (m_dataFormat == DataFormat::RGBA8x4 ||
                m_dataFormat == DataFormat::Gray8 ||
                m_dataFormat == DataFormat::BC1 ||
                m_dataFormat == DataFormat::BC2 ||
                m_dataFormat == DataFormat::BC3 ||
                m_dataFormat == DataFormat::BC4 ||
                m_dataFormat == DataFormat::BC7)
                m_needsHW_sRGB_degamma = true;

            // GrayA8はRT_FORMAT_UNSIGNED_BYTE2なのでテクスチャーユニットにデガンマさせるとA成分もデガンマされて不都合。
            // BC4_Signedは符号付きなのでデガンマは不適切。
            // BC5も法線マップなどに使われる2成分テクスチャーなのでデガンマは考えにくい。
        }
    }

    Image2D::~Image2D() {
        if (m_optixDataBuffer)
            m_optixDataBuffer->destroy();
    }

    optix::Buffer Image2D::getOptiXObject() const {
        if (m_initOptiXObject)
            return m_optixDataBuffer;

        optix::Context optixContext = m_context.getOptiXContext();

        if (m_dataFormat >= DataFormat::BC1 && m_dataFormat <= DataFormat::BC7) {
            uint32_t widthInBlocks = nextMultiplierForPowOf2(m_width, 4);
            uint32_t heightInBlocks = nextMultiplierForPowOf2(m_height, 4);

            switch (m_dataFormat.value) {
            case DataFormat::BC1:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC1, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC2:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC2, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC3:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC3, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC4, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC4_Signed:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BC4, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC5:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC5, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC5_Signed:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BC5, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC6H:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC6H, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC6H_Signed:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BC6H, widthInBlocks, heightInBlocks);
                break;
            case DataFormat::BC7:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BC7, widthInBlocks, heightInBlocks);
                break;
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
        }
        else {
            switch (m_dataFormat.value) {
            case DataFormat::RGB8x3:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE3, m_width, m_height);
                break;
            case DataFormat::RGB_8x4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height);
                break;
            case DataFormat::RGBA8x4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height);
                break;
            case DataFormat::RGBA16Fx4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_HALF4, m_width, m_height);
                break;
            case DataFormat::RGBA32Fx4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_width, m_height);
                break;
            case DataFormat::RG32Fx2:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, m_width, m_height);
                break;
            case DataFormat::Gray32F:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, m_width, m_height);
                break;
            case DataFormat::Gray8:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, m_width, m_height);
                break;
            case DataFormat::GrayA8x2:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE2, m_width, m_height);
                break;
            case DataFormat::uvsA8x4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height);
                break;
            case DataFormat::uvsA16Fx4:
                m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_HALF4, m_width, m_height);
                break;
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
        }

        m_initOptiXObject = true;

        return m_optixDataBuffer;
    }



    template <bool enableDegamma>
    struct sRGB_D65_ColorSpaceFunc {
        static float degamma(float v) {
            if /*constexpr*/ (enableDegamma)
                return sRGB_degamma(v);
            else
                return v;
        }
        static uint8_t degamma(uint8_t v) {
            if /*constexpr*/ (enableDegamma)
                return (uint8_t)std::min<uint32_t>(255, 256 * sRGB_degamma(v / 255.0f));
            else
                return v;
        }

        static void RGB_to_XYZ(const float RGB[3], float XYZ[3]) {
            transformTristimulus(mat_Rec709_D65_to_XYZ, RGB, XYZ);
        }
    };

    template <bool enableDegamma>
    struct sRGB_E_ColorSpaceFunc {
        static float degamma(float v) {
            if /*constexpr*/ (enableDegamma)
                return sRGB_degamma(v);
            else
                return v;
        }
        static uint8_t degamma(uint8_t v) {
            if /*constexpr*/ (enableDegamma)
                return (uint8_t)std::min<uint32_t>(255, 256 * sRGB_degamma(v / 255.0f));
            else
                return v;
        }

        static void RGB_to_XYZ(const float RGB[3], float XYZ[3]) {
            transformTristimulus(mat_Rec709_E_to_XYZ, RGB, XYZ);
        }
    };

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGB8x3 &srcData, RGBA8x4 &dstData) {
        dstData.r = ColorSpaceFunc::degamma(srcData.r);
        dstData.g = ColorSpaceFunc::degamma(srcData.g);
        dstData.b = ColorSpaceFunc::degamma(srcData.b);
        dstData.a = 255;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGB_8x4 &srcData, RGBA8x4 &dstData) {
        dstData.r = ColorSpaceFunc::degamma(srcData.r);
        dstData.g = ColorSpaceFunc::degamma(srcData.g);
        dstData.b = ColorSpaceFunc::degamma(srcData.b);
        dstData.a = 255;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGBA8x4 &srcData, RGBA8x4 &dstData) {
        dstData.r = ColorSpaceFunc::degamma(srcData.r);
        dstData.g = ColorSpaceFunc::degamma(srcData.g);
        dstData.b = ColorSpaceFunc::degamma(srcData.b);
        dstData.a = srcData.a;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGBA16Fx4 &srcData, RGBA16Fx4 &dstData) {
        dstData.r = (half)ColorSpaceFunc::degamma((float)srcData.r);
        dstData.g = (half)ColorSpaceFunc::degamma((float)srcData.g);
        dstData.b = (half)ColorSpaceFunc::degamma((float)srcData.b);
        dstData.a = srcData.a;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGBA32Fx4 &srcData, RGBA32Fx4 &dstData) {
        dstData.r = ColorSpaceFunc::degamma(srcData.r);
        dstData.g = ColorSpaceFunc::degamma(srcData.g);
        dstData.b = ColorSpaceFunc::degamma(srcData.b);
        dstData.a = srcData.a;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RG32Fx2 &srcData, RG32Fx2 &dstData) {
        dstData.r = ColorSpaceFunc::degamma(srcData.r);
        dstData.g = ColorSpaceFunc::degamma(srcData.g);
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const Gray32F &srcData, Gray32F &dstData) {
        dstData.v = ColorSpaceFunc::degamma(srcData.v);
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const Gray8 &srcData, Gray8 &dstData) {
        dstData.v = ColorSpaceFunc::degamma(srcData.v);
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const GrayA8x2 &srcData, GrayA8x2 &dstData) {
        dstData.v = ColorSpaceFunc::degamma(srcData.v);
        dstData.a = srcData.a;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(float RGB[3], uvsA8x4 &dstData) {
        RGB[0] = ColorSpaceFunc::degamma(RGB[0]);
        RGB[1] = ColorSpaceFunc::degamma(RGB[1]);
        RGB[2] = ColorSpaceFunc::degamma(RGB[2]);

        float XYZ[3];
        ColorSpaceFunc::RGB_to_XYZ(RGB, XYZ);

        float b = XYZ[0] + XYZ[1] + XYZ[2];
        float xy[2];
        xy[0] = b > 0.0f ? XYZ[0] / b : (1.0f / 3.0f);
        xy[1] = b > 0.0f ? XYZ[1] / b : (1.0f / 3.0f);

        float uv[2];
        UpsampledSpectrum::xy_to_uv(xy, uv);

        dstData.u = (uint8_t)(255 * clamp<float>(uv[0] / UpsampledSpectrum::GridWidth(), 0, 1));
        dstData.v = (uint8_t)(255 * clamp<float>(uv[1] / UpsampledSpectrum::GridHeight(), 0, 1));
        dstData.s = (uint8_t)(255 * clamp<float>(b / 3.0f, 0, 1));
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(float RGB[3], uvsA16Fx4 &dstData) {
        RGB[0] = ColorSpaceFunc::degamma(RGB[0]);
        RGB[1] = ColorSpaceFunc::degamma(RGB[1]);
        RGB[2] = ColorSpaceFunc::degamma(RGB[2]);

        float XYZ[3];
        ColorSpaceFunc::RGB_to_XYZ(RGB, XYZ);

        float b = XYZ[0] + XYZ[1] + XYZ[2];
        float xy[2];
        xy[0] = b > 0.0f ? XYZ[0] / b : (1.0f / 3.0f);
        xy[1] = b > 0.0f ? XYZ[1] / b : (1.0f / 3.0f);

        float uv[2];
        UpsampledSpectrum::xy_to_uv(xy, uv);

        dstData.u = (half)uv[0];
        dstData.v = (half)uv[1];
        // JP: よくあるテクスチャーの値だとInfになってしまうため
        //     本来は割るべきところを割らないままにしておく。
        // EN: 
        dstData.s = (half)(b/* / UpsampledSpectrum::EqualEnergyReflectance()*/);
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGB8x3 &srcData, uvsA8x4 &dstData) {
        float RGB[] = { srcData.r / 255.0f, srcData.g / 255.0f, srcData.b / 255.0f };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = 255;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGB8x3 &srcData, uvsA16Fx4 &dstData) {
        float RGB[] = { srcData.r / 255.0f, srcData.g / 255.0f, srcData.b / 255.0f };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = (half)1.0f;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGB_8x4 &srcData, uvsA8x4 &dstData) {
        float RGB[] = { srcData.r / 255.0f, srcData.g / 255.0f, srcData.b / 255.0f };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = 255;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGB_8x4 &srcData, uvsA16Fx4 &dstData) {
        float RGB[] = { srcData.r / 255.0f, srcData.g / 255.0f, srcData.b / 255.0f };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = (half)1.0f;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGBA8x4 &srcData, uvsA8x4 &dstData) {
        float RGB[] = { srcData.r / 255.0f, srcData.g / 255.0f, srcData.b / 255.0f };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = srcData.a;
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGBA8x4 &srcData, uvsA16Fx4 &dstData) {
        float RGB[] = { srcData.r / 255.0f, srcData.g / 255.0f, srcData.b / 255.0f };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = (half)(srcData.a / 255.0f);
    }

    template <typename ColorSpaceFunc>
    void perPixelFunc(const RGBA16Fx4 &srcData, uvsA16Fx4 &dstData) {
        float RGB[] = { srcData.r, srcData.g, srcData.b };
        perPixelFunc<ColorSpaceFunc>(RGB, dstData);
        dstData.a = srcData.a;
    }

    template <typename SrcType, typename DstType, typename ColorSpaceFunc>
    void processAllPixels(const uint8_t* srcData, uint8_t* dstData, uint32_t width, uint32_t height) {
        if /* constexpr */ (std::is_same<SrcType, DstType>::value &&
            (std::is_same<ColorSpaceFunc, sRGB_D65_ColorSpaceFunc<false>>::value ||
             std::is_same<ColorSpaceFunc, sRGB_E_ColorSpaceFunc<false>>::value)) {
            auto srcHead = (const SrcType*)srcData;
            auto dstHead = (SrcType*)dstData;
            std::copy_n(srcHead, width * height, dstHead);
        }
        else {
            auto srcHead = (const SrcType*)srcData;
            auto dstHead = (DstType*)dstData;
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    auto &src = *(srcLineHead + x);
                    auto &dst = *(dstLineHead + x);
                    perPixelFunc<ColorSpaceFunc>(src, dst);
                }
            }
        }
    }


    
    LinearImage2D::LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height,
                                 DataFormat dataFormat, VLRSpectrumType spectrumType, ColorSpace colorSpace) :
        Image2D(context, width, height, dataFormat, spectrumType, colorSpace), m_copyDone(false) {
        VLRAssert(dataFormat < DataFormat::BC1 || dataFormat > DataFormat::BC7, "Specified data format is a block compressed format.");
        m_data.resize(getStride() * getWidth() * getHeight());

        switch (dataFormat.value) {
        case DataFormat::RGB8x3: {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            if (spectrumType != VLRSpectrumType_NA) {
                if (spectrumType == VLRSpectrumType_Reflectance) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB8x3, uvsA8x4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB8x3, uvsA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else if (spectrumType == VLRSpectrumType_IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB8x3, uvsA16Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB8x3, uvsA16Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }
            else {
                processAllPixels<RGB8x3, RGBA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            }
#else
            processAllPixels<RGB8x3, RGBA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
#endif
            break;
        }
        case DataFormat::RGB_8x4: {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            if (spectrumType != VLRSpectrumType_NA) {
                if (spectrumType == VLRSpectrumType_Reflectance) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB_8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB_8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else if (spectrumType == VLRSpectrumType_IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB_8x4, uvsA16Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB_8x4, uvsA16Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }
            else {
                processAllPixels<RGB_8x4, RGBA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            }
#else
            processAllPixels<RGB_8x4, RGBA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
#endif
            break;
        }
        case DataFormat::RGBA8x4: {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            if (spectrumType != VLRSpectrumType_NA) {
                if (spectrumType == VLRSpectrumType_Reflectance) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGBA8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGBA8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else if (spectrumType == VLRSpectrumType_IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGBA8x4, uvsA16Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGBA8x4, uvsA16Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }
            else {
                processAllPixels<RGBA8x4, RGBA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            }
#else
            processAllPixels<RGBA8x4, RGBA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
#endif
            break;
        }
        case DataFormat::RGBA16Fx4: {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            if (spectrumType != VLRSpectrumType_NA) {
                if (spectrumType == VLRSpectrumType_Reflectance ||
                    spectrumType == VLRSpectrumType_IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGBA16Fx4, uvsA16Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGBA16Fx4, uvsA16Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGBA16Fx4, uvsA16Fx4, sRGB_D65_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGBA16Fx4, uvsA16Fx4, sRGB_D65_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
            }
            else {
                if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                    processAllPixels<RGBA16Fx4, RGBA16Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                else
                    processAllPixels<RGBA16Fx4, RGBA16Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            }
#else
            if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                processAllPixels<RGBA16Fx4, RGBA16Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
            else
                processAllPixels<RGBA16Fx4, RGBA16Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
#endif
            break;
        }
        case DataFormat::RGBA32Fx4: {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            VLRAssert_NotImplemented();
#else
            if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                processAllPixels<RGBA32Fx4, RGBA32Fx4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
            else
                processAllPixels<RGBA32Fx4, RGBA32Fx4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
#endif
            break;
        }
        case DataFormat::RG32Fx2: {
            if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                processAllPixels<RG32Fx2, RG32Fx2, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
            else
                processAllPixels<RG32Fx2, RG32Fx2, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            break;
        }
        case DataFormat::Gray32F: {
            if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                processAllPixels<Gray32F, Gray32F, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
            else
                processAllPixels<Gray32F, Gray32F, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            break;
        }
        case DataFormat::Gray8: {
            processAllPixels<Gray8, Gray8, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            break;
        }
        case DataFormat::GrayA8x2: {
            if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                processAllPixels<GrayA8x2, GrayA8x2, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
            else
                processAllPixels<GrayA8x2, GrayA8x2, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
            break;
        }
        case DataFormat::uvsA8x4:
        case DataFormat::uvsA16Fx4:
            std::copy(linearData, linearData + getStride() * width * height, (uint8_t*)m_data.data());
            break;
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }
    }

    Image2D* LinearImage2D::createShrinkedImage2D(uint32_t width, uint32_t height) const {
        uint32_t orgWidth = getWidth();
        uint32_t orgHeight = getHeight();
        uint32_t stride = getStride();
        VLRAssert(width < orgWidth && height < orgHeight, "Image size must be smaller than the original.");
        std::vector<uint8_t> data;
        data.resize(stride * width * height);

        float deltaOrgX = orgWidth / width;
        float deltaOrgY = orgHeight / height;
        for (int y = 0; y < height; ++y) {
            float top = deltaOrgY * y;
            float bottom = deltaOrgY * (y + 1);
            uint32_t topPix = (uint32_t)top;
            uint32_t bottomPix = (uint32_t)ceilf(bottom) - 1;

            for (int x = 0; x < width; ++x) {
                float left = deltaOrgX * x;
                float right = deltaOrgX * (x + 1);
                uint32_t leftPix = (uint32_t)left;
                uint32_t rightPix = (uint32_t)ceilf(right) - 1;

                float area = (bottom - top) * (right - left);

                // UL, UR, LL, LR
                float weightsCorners[] = {
                    (leftPix + 1 - left) * (topPix + 1 - top),
                    (right - rightPix) * (topPix + 1 - top),
                    (leftPix + 1 - left) * (bottom - bottomPix),
                    (right - rightPix) * (bottom - bottomPix)
                };
                // Top, Left, Right, Bottom
                float weightsEdges[] = {
                    topPix + 1 - top,
                    leftPix + 1 - left,
                    right - rightPix,
                    bottom - bottomPix
                };

                switch (getDataFormat().value) {
                case DataFormat::RGBA16Fx4: {
                    CompensatedSum<float> sumR(0), sumG(0), sumB(0), sumA(0);
                    RGBA16Fx4 pix;

                    uint32_t corners[] = { leftPix, topPix, rightPix, topPix, leftPix, bottomPix, rightPix, bottomPix };
                    for (int i = 0; i < 4; ++i) {
                        pix = get<RGBA16Fx4>(corners[2 * i + 0], corners[2 * i + 1]);
                        sumR += weightsCorners[i] * float(pix.r);
                        sumG += weightsCorners[i] * float(pix.g);
                        sumB += weightsCorners[i] * float(pix.b);
                        sumA += weightsCorners[i] * float(pix.a);
                    }

                    for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                        pix = get<RGBA16Fx4>(x, topPix);
                        sumR += weightsEdges[0] * float(pix.r);
                        sumG += weightsEdges[0] * float(pix.g);
                        sumB += weightsEdges[0] * float(pix.b);
                        sumA += weightsEdges[0] * float(pix.a);

                        pix = get<RGBA16Fx4>(x, bottomPix);
                        sumR += weightsEdges[3] * float(pix.r);
                        sumG += weightsEdges[3] * float(pix.g);
                        sumB += weightsEdges[3] * float(pix.b);
                        sumA += weightsEdges[3] * float(pix.a);
                    }
                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        pix = get<RGBA16Fx4>(leftPix, y);
                        sumR += weightsEdges[1] * float(pix.r);
                        sumG += weightsEdges[1] * float(pix.g);
                        sumB += weightsEdges[1] * float(pix.b);
                        sumA += weightsEdges[1] * float(pix.a);

                        pix = get<RGBA16Fx4>(rightPix, y);
                        sumR += weightsEdges[2] * float(pix.r);
                        sumG += weightsEdges[2] * float(pix.g);
                        sumB += weightsEdges[2] * float(pix.b);
                        sumA += weightsEdges[2] * float(pix.a);
                    }

                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                            pix = get<RGBA16Fx4>(x, y);
                            sumR += float(pix.r);
                            sumG += float(pix.g);
                            sumB += float(pix.b);
                            sumA += float(pix.a);
                        }
                    }

                    *(RGBA16Fx4*)&data[(y * width + x) * stride] = RGBA16Fx4{ half(sumR / area), half(sumG / area), half(sumB / area), half(sumA / area) };
                    break;
                }
                case DataFormat::uvsA16Fx4: {
                    CompensatedSum<float> sum_u(0), sum_v(0), sum_s(0), sumA(0);
                    uvsA16Fx4 pix;

                    uint32_t corners[] = { leftPix, topPix, rightPix, topPix, leftPix, bottomPix, rightPix, bottomPix };
                    for (int i = 0; i < 4; ++i) {
                        pix = get<uvsA16Fx4>(corners[2 * i + 0], corners[2 * i + 1]);
                        sum_u += weightsCorners[i] * float(pix.u);
                        sum_v += weightsCorners[i] * float(pix.v);
                        sum_s += weightsCorners[i] * float(pix.s);
                        sumA += weightsCorners[i] * float(pix.a);
                    }

                    for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                        pix = get<uvsA16Fx4>(x, topPix);
                        sum_u += weightsEdges[0] * float(pix.u);
                        sum_v += weightsEdges[0] * float(pix.v);
                        sum_s += weightsEdges[0] * float(pix.s);
                        sumA += weightsEdges[0] * float(pix.a);

                        pix = get<uvsA16Fx4>(x, bottomPix);
                        sum_u += weightsEdges[3] * float(pix.u);
                        sum_v += weightsEdges[3] * float(pix.v);
                        sum_s += weightsEdges[3] * float(pix.s);
                        sumA += weightsEdges[3] * float(pix.a);
                    }
                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        pix = get<uvsA16Fx4>(leftPix, y);
                        sum_u += weightsEdges[1] * float(pix.u);
                        sum_v += weightsEdges[1] * float(pix.v);
                        sum_s += weightsEdges[1] * float(pix.s);
                        sumA += weightsEdges[1] * float(pix.a);

                        pix = get<uvsA16Fx4>(rightPix, y);
                        sum_u += weightsEdges[2] * float(pix.u);
                        sum_v += weightsEdges[2] * float(pix.v);
                        sum_s += weightsEdges[2] * float(pix.s);
                        sumA += weightsEdges[2] * float(pix.a);
                    }

                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                            pix = get<uvsA16Fx4>(x, y);
                            sum_u += float(pix.u);
                            sum_v += float(pix.v);
                            sum_s += float(pix.s);
                            sumA += float(pix.a);
                        }
                    }

                    *(uvsA16Fx4*)&data[(y * width + x) * stride] = uvsA16Fx4{ half(sum_u / area), half(sum_v / area), half(sum_s / area), half(sumA / area) };
                    break;
                }
                default:
                    VLRAssert_ShouldNotBeCalled();
                    break;
                }
            }
        }

        return new LinearImage2D(m_context, data.data(), width, height, getDataFormat(), getSpectrumType(), getColorSpace());
    }

    Image2D* LinearImage2D::createLuminanceImage2D() const {
        uint32_t width = getWidth();
        uint32_t height = getHeight();
        uint32_t stride;
        DataFormat newDataFormat;
        switch (getDataFormat().value) {
        case DataFormat::RGBA16Fx4: {
            stride = sizeof(float);
            newDataFormat = DataFormat::Gray32F;
            break;
        }
        case DataFormat::uvsA16Fx4: {
            stride = sizeof(float);
            newDataFormat = DataFormat::Gray32F;
            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
        std::vector<uint8_t> data;
        data.resize(stride * width * height);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                switch (getDataFormat().value) {
                case DataFormat::RGBA16Fx4: {
                    RGBA16Fx4 pix = get<RGBA16Fx4>(x, y);
                    float Y = mat_Rec709_D65_to_XYZ[1] * pix.r + mat_Rec709_D65_to_XYZ[4] * pix.g + mat_Rec709_D65_to_XYZ[7] * pix.b;
                    *(float*)&data[(y * width + x) * stride] = Y;
                    break;
                }
                case DataFormat::uvsA16Fx4: {
                    uvsA16Fx4 pix = get<uvsA16Fx4>(x, y);
                    float uv[2] = { pix.u, pix.v };
                    float xy[2];
                    UpsampledSpectrum::uv_to_xy(uv, xy);
                    float b = pix.s/* * UpsampledSpectrum::EqualEnergyReflectance()*/;
                    float Y = xy[1] * b;
                    *(float*)&data[(y * width + x) * stride] = Y;
                    break;
                }
                default:
                    VLRAssert_ShouldNotBeCalled();
                    break;
                }
            }
        }

        return new LinearImage2D(m_context, data.data(), width, height, newDataFormat, getSpectrumType(), getColorSpace());
    }

    void* LinearImage2D::createLinearImageData() const {
        uint8_t* ret = new uint8_t[m_data.size()];
        std::copy(m_data.cbegin(), m_data.cend(), ret);
        return ret;
    }

    optix::Buffer LinearImage2D::getOptiXObject() const {
        optix::Buffer buffer = Image2D::getOptiXObject();
        if (!m_copyDone) {
            auto dstData = (uint8_t*)buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy(m_data.cbegin(), m_data.cend(), dstData);
            buffer->unmap();
            m_copyDone = true;
        }
        return buffer;
    }



    BlockCompressedImage2D::BlockCompressedImage2D(Context &context, const uint8_t* const* data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height, 
                                                   DataFormat dataFormat, VLRSpectrumType spectrumType, ColorSpace colorSpace) :
        Image2D(context, width, height, dataFormat, spectrumType, colorSpace), m_copyDone(false) {
        VLRAssert(dataFormat >= DataFormat::BC1 && dataFormat <= DataFormat::BC7, "Specified data format is not block compressed format.");
        m_data.resize(mipCount);
        for (int i = 0; i < mipCount; ++i) {
            m_data[i].resize(sizes[i]);
            std::copy(data[i], data[i] + sizes[i], m_data[i].data());
        }
    }

    Image2D* BlockCompressedImage2D::createShrinkedImage2D(uint32_t width, uint32_t height) const {
        VLRAssert_NotImplemented();
        return nullptr;
    }

    Image2D* BlockCompressedImage2D::createLuminanceImage2D() const {
        VLRAssert_NotImplemented();
        return nullptr;
    }

    void* BlockCompressedImage2D::createLinearImageData() const {
        VLRAssert_NotImplemented();
        return nullptr;
    }

    optix::Buffer BlockCompressedImage2D::getOptiXObject() const {
        optix::Buffer buffer = Image2D::getOptiXObject();
        if (!m_copyDone) {
            // JP: OptiXのBCブロックカウントの計算がおかしいらしく。
            //     非2のべき乗テクスチャーだとサイズがずれる。
            //     要問い合わせ。
            int32_t mipCount = 1;// m_data.size();

            buffer->setMipLevelCount(mipCount);
            auto dstData = new uint8_t*[mipCount];

            for (int mipLevel = 0; mipLevel < mipCount; ++mipLevel)
                dstData[mipLevel] = (uint8_t*)buffer->map(mipLevel, RT_BUFFER_MAP_WRITE_DISCARD);

            for (int mipLevel = 0; mipLevel < mipCount; ++mipLevel) {
                const auto &mipData = m_data[mipLevel];
                std::copy(mipData.cbegin(), mipData.cend(), dstData[mipLevel]);
            }

            for (int mipLevel = mipCount - 1; mipLevel >= 0; --mipLevel)
                buffer->unmap(mipLevel);

            delete[] dstData;

            m_copyDone = true;
        }
        return buffer;
    }



    Shared::ShaderNodeSocketID ShaderNodeSocketIdentifier::getSharedType() const {
        if (node) {
            Shared::ShaderNodeSocketID ret;
            ret.nodeDescIndex = node->getShaderNodeIndex();
            ret.socketType = socketInfo.outputType;
            ret.option = socketInfo.option;
            ret.isSpectrumNode = node->isSpectrumNode();
            return ret;
        }
        return Shared::ShaderNodeSocketID::Invalid();
    }



    // static 
    void ShaderNode::commonInitializeProcedure(Context &context, const SocketTypeToProgramPair* pairs, uint32_t numPairs, OptiXProgramSet* programSet) {
        std::string ptx = readTxtFile(VLR_PTX_DIR"shader_nodes.ptx");

        optix::Context optixContext = context.getOptiXContext();

        Shared::NodeProcedureSet nodeProcSet;
        for (int i = 0; i < lengthof(nodeProcSet.progs); ++i)
            nodeProcSet.progs[i] = 0xFFFFFFFF;
        for (int i = 0; i < numPairs; ++i) {
            VLRShaderNodeSocketType stype = pairs[i].stype;
            programSet->callablePrograms[stype] = optixContext->createProgramFromPTXString(ptx, pairs[i].programName);
            nodeProcSet.progs[stype] = programSet->callablePrograms[stype]->getId();
        }

        programSet->nodeProcedureSetIndex = context.allocateNodeProcedureSet();
        context.updateNodeProcedureSet(programSet->nodeProcedureSetIndex, nodeProcSet);
    }

    // static 
    void ShaderNode::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        context.releaseNodeProcedureSet(programSet.nodeProcedureSetIndex);

        for (int i = lengthof(programSet.callablePrograms) - 1; i >= 0; --i) {
            if (programSet.callablePrograms[i])
                programSet.callablePrograms[i]->destroy();
        }
    }

    // static
    void ShaderNode::initialize(Context &context) {
        GeometryShaderNode::initialize(context);
        FloatShaderNode::initialize(context);
        Float2ShaderNode::initialize(context);
        Float3ShaderNode::initialize(context);
        Float4ShaderNode::initialize(context);
        ScaleAndOffsetFloatShaderNode::initialize(context);
        TripletSpectrumShaderNode::initialize(context);
        RegularSampledSpectrumShaderNode::initialize(context);
        IrregularSampledSpectrumShaderNode::initialize(context);
        Vector3DToSpectrumShaderNode::initialize(context);
        ScaleAndOffsetUVTextureMap2DShaderNode::initialize(context);
        Image2DTextureShaderNode::initialize(context);
        EnvironmentTextureShaderNode::initialize(context);
    }

    // static
    void ShaderNode::finalize(Context &context) {
        EnvironmentTextureShaderNode::finalize(context);
        Image2DTextureShaderNode::finalize(context);
        ScaleAndOffsetUVTextureMap2DShaderNode::finalize(context);
        Vector3DToSpectrumShaderNode::finalize(context);
        IrregularSampledSpectrumShaderNode::finalize(context);
        RegularSampledSpectrumShaderNode::finalize(context);
        TripletSpectrumShaderNode::finalize(context);
        ScaleAndOffsetFloatShaderNode::finalize(context);
        Float4ShaderNode::finalize(context);
        Float3ShaderNode::finalize(context);
        Float2ShaderNode::finalize(context);
        FloatShaderNode::finalize(context);
        GeometryShaderNode::finalize(context);
    }

    ShaderNode::ShaderNode(Context &context, bool isSpectrumNode) : Object(context), m_isSpectrumNode(isSpectrumNode) {
        if (m_isSpectrumNode)
            m_nodeIndex = m_context.allocateSpectrumNodeDescriptor();
        else
            m_nodeIndex = m_context.allocateNodeDescriptor();
    }

    ShaderNode::~ShaderNode() {
        if (m_nodeIndex != 0xFFFFFFFF)
            if (m_isSpectrumNode)
                m_context.releaseSpectrumNodeDescriptor(m_nodeIndex);
            else
                m_context.releaseNodeDescriptor(m_nodeIndex);
        m_nodeIndex = 0xFFFFFFFF;
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> GeometryShaderNode::OptiXProgramSets;
    std::map<uint32_t, GeometryShaderNode*> GeometryShaderNode::Instances;

    // static
    void GeometryShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Point3D, "VLR::GeometryShaderNode_Point3D",
            VLRShaderNodeSocketType_Normal3D, "VLR::GeometryShaderNode_Normal3D",
            VLRShaderNodeSocketType_Vector3D, "VLR::GeometryShaderNode_Vector3D",
            VLRShaderNodeSocketType_TextureCoordinates, "VLR::GeometryShaderNode_TextureCoordinates",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;

        Instances[context.getID()] = new GeometryShaderNode(context);
    }

    // static
    void GeometryShaderNode::finalize(Context &context) {
        delete Instances.at(context.getID());

        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    GeometryShaderNode::GeometryShaderNode(Context &context) :
        ShaderNode(context) {
        setupNodeDescriptor();
    }

    GeometryShaderNode::~GeometryShaderNode() {
    }

    void GeometryShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::GeometryShaderNode>();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    GeometryShaderNode* GeometryShaderNode::getInstance(Context &context) {
        return Instances.at(context.getID());
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> FloatShaderNode::OptiXProgramSets;

    // static
    void FloatShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_float, "VLR::FloatShaderNode_float",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void FloatShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    FloatShaderNode::FloatShaderNode(Context &context) :
        ShaderNode(context), m_imm0(0.0f) {
        setupNodeDescriptor();
    }

    FloatShaderNode::~FloatShaderNode() {
    }

    void FloatShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::FloatShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.imm0 = m_imm0;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool FloatShaderNode::setNode0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node0 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void FloatShaderNode::setImmediateValue0(float value) {
        m_imm0 = value;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float2ShaderNode::OptiXProgramSets;

    // static
    void Float2ShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_float, "VLR::Float2ShaderNode_float",
            VLRShaderNodeSocketType_float2, "VLR::Float2ShaderNode_float2",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float2ShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Float2ShaderNode::Float2ShaderNode(Context &context) :
        ShaderNode(context),
        m_imm0(0.0f), m_imm1(0.0f) {
        setupNodeDescriptor();
    }

    Float2ShaderNode::~Float2ShaderNode() {
    }

    void Float2ShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::Float2ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool Float2ShaderNode::setNode0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node0 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float2ShaderNode::setImmediateValue0(float value) {
        m_imm0 = value;
        setupNodeDescriptor();
    }

    bool Float2ShaderNode::setNode1(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node1 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float2ShaderNode::setImmediateValue1(float value) {
        m_imm1 = value;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float3ShaderNode::OptiXProgramSets;

    // static
    void Float3ShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_float, "VLR::Float3ShaderNode_float",
            VLRShaderNodeSocketType_float2, "VLR::Float3ShaderNode_float2",
            VLRShaderNodeSocketType_float3, "VLR::Float3ShaderNode_float3",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float3ShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Float3ShaderNode::Float3ShaderNode(Context &context) :
        ShaderNode(context),
        m_imm0(0.0f), m_imm1(0.0f), m_imm2(0.0f) {
        setupNodeDescriptor();
    }

    Float3ShaderNode::~Float3ShaderNode() {
    }

    void Float3ShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::Float3ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.node2 = m_node2.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;
        nodeData.imm2 = m_imm2;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool Float3ShaderNode::setNode0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node0 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float3ShaderNode::setImmediateValue0(float value) {
        m_imm0 = value;
        setupNodeDescriptor();
    }

    bool Float3ShaderNode::setNode1(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node1 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float3ShaderNode::setImmediateValue1(float value) {
        m_imm1 = value;
        setupNodeDescriptor();
    }

    bool Float3ShaderNode::setNode2(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node2 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float3ShaderNode::setImmediateValue2(float value) {
        m_imm2 = value;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float4ShaderNode::OptiXProgramSets;

    // static
    void Float4ShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_float, "VLR::Float4ShaderNode_float",
            VLRShaderNodeSocketType_float2, "VLR::Float4ShaderNode_float2",
            VLRShaderNodeSocketType_float3, "VLR::Float4ShaderNode_float3",
            VLRShaderNodeSocketType_float4, "VLR::Float4ShaderNode_float4",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float4ShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Float4ShaderNode::Float4ShaderNode(Context &context) :
        ShaderNode(context), m_imm0(0.0f), m_imm1(0.0f), m_imm2(0.0f), m_imm3(0.0f) {
        setupNodeDescriptor();
    }

    Float4ShaderNode::~Float4ShaderNode() {
    }

    void Float4ShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::Float4ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.node2 = m_node2.getSharedType();
        nodeData.node3 = m_node3.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;
        nodeData.imm2 = m_imm2;
        nodeData.imm3 = m_imm3;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool Float4ShaderNode::setNode0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node0 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float4ShaderNode::setImmediateValue0(float value) {
        m_imm0 = value;
        setupNodeDescriptor();
    }

    bool Float4ShaderNode::setNode1(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node1 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float4ShaderNode::setImmediateValue1(float value) {
        m_imm1 = value;
        setupNodeDescriptor();
    }

    bool Float4ShaderNode::setNode2(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node2 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float4ShaderNode::setImmediateValue2(float value) {
        m_imm2 = value;
        setupNodeDescriptor();
    }

    bool Float4ShaderNode::setNode3(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_node3 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float4ShaderNode::setImmediateValue3(float value) {
        m_imm3 = value;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> ScaleAndOffsetFloatShaderNode::OptiXProgramSets;

    // static
    void ScaleAndOffsetFloatShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_float, "VLR::ScaleAndOffsetFloatShaderNode_float",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void ScaleAndOffsetFloatShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    ScaleAndOffsetFloatShaderNode::ScaleAndOffsetFloatShaderNode(Context &context) :
        ShaderNode(context), m_immScale(1.0f), m_immOffset(0.0f) {
        setupNodeDescriptor();
    }

    ScaleAndOffsetFloatShaderNode::~ScaleAndOffsetFloatShaderNode() {
    }

    void ScaleAndOffsetFloatShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::ScaleAndOffsetFloatShaderNode>();
        nodeData.nodeValue = m_nodeValue.getSharedType();
        nodeData.nodeScale = m_nodeScale.getSharedType();
        nodeData.nodeOffset = m_nodeOffset.getSharedType();
        nodeData.immScale = m_immScale;
        nodeData.immOffset = m_immOffset;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool ScaleAndOffsetFloatShaderNode::setNodeValue(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_nodeValue = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::setNodeScale(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_nodeScale = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::setNodeOffset(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_nodeOffset = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void ScaleAndOffsetFloatShaderNode::setImmediateValueScale(float value) {
        m_immScale = value;
        setupNodeDescriptor();
    }

    void ScaleAndOffsetFloatShaderNode::setImmediateValueOffset(float value) {
        m_immOffset = value;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> TripletSpectrumShaderNode::OptiXProgramSets;

    // static
    void TripletSpectrumShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::TripletSpectrumShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TripletSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    TripletSpectrumShaderNode::TripletSpectrumShaderNode(Context &context) :
        ShaderNode(context, true), m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(ColorSpace::Rec709_D65),
        m_immE0(0.18f), m_immE1(0.18f), m_immE2(0.18f) {
        setupNodeDescriptor();
    }

    TripletSpectrumShaderNode::~TripletSpectrumShaderNode() {
    }

    void TripletSpectrumShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SpectrumNodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::TripletSpectrumShaderNode>();
        nodeData.value = createTripletSpectrum(m_spectrumType, m_colorSpace, m_immE0, m_immE1, m_immE2);

        m_context.updateSpectrumNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void TripletSpectrumShaderNode::setImmediateValueSpectrumType(VLRSpectrumType spectrumType) {
        m_spectrumType = spectrumType;
        setupNodeDescriptor();
    }

    void TripletSpectrumShaderNode::setImmediateValueColorSpace(ColorSpace colorSpace) {
        m_colorSpace = colorSpace;
        setupNodeDescriptor();
    }

    void TripletSpectrumShaderNode::setImmediateValueTriplet(float e0, float e1, float e2) {
        m_immE0 = e0;
        m_immE1 = e1;
        m_immE2 = e2;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> RegularSampledSpectrumShaderNode::OptiXProgramSets;

    // static
    void RegularSampledSpectrumShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::RegularSampledSpectrumShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void RegularSampledSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    RegularSampledSpectrumShaderNode::RegularSampledSpectrumShaderNode(Context &context) :
        ShaderNode(context, true), m_spectrumType(VLRSpectrumType_NA), m_minLambda(0.0f), m_maxLambda(1000.0f), m_values(nullptr), m_numSamples(2) {
        m_values = new float[2];
        m_values[0] = m_values[1] = 1.0f;
        setupNodeDescriptor();
    }

    RegularSampledSpectrumShaderNode::~RegularSampledSpectrumShaderNode() {
        if (m_values)
            delete[] m_values;
    }

    void RegularSampledSpectrumShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SpectrumNodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::RegularSampledSpectrumShaderNode>();
#if defined(VLR_USE_SPECTRAL_RENDERING)
        VLRAssert(m_numSamples <= lengthof(nodeData.values), "Number of sample points must not be greater than %u.", lengthof(nodeData.values));
        nodeData.minLambda = m_minLambda;
        nodeData.maxLambda = m_maxLambda;
        std::copy_n(m_values, m_numSamples, nodeData.values);
        nodeData.numSamples = m_numSamples;
#else
        RegularSampledSpectrum spectrum(m_minLambda, m_maxLambda, m_values, m_numSamples);
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        float RGB[3];
        transformToRenderingRGB(m_spectrumType, XYZ, RGB);
        nodeData.value = RGBSpectrum(std::fmax(0.0f, RGB[0]), std::fmax(0.0f, RGB[1]), std::fmax(0.0f, RGB[2]));
#endif

        m_context.updateSpectrumNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void RegularSampledSpectrumShaderNode::setImmediateValueSpectrum(VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples) {
        if (m_values)
            delete[] m_values;
        m_spectrumType = spectrumType;
        m_minLambda = minLambda;
        m_maxLambda = maxLambda;
        m_values = new float[numSamples];
        std::copy_n(values, numSamples, m_values);
        m_numSamples = numSamples;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> IrregularSampledSpectrumShaderNode::OptiXProgramSets;

    // static
    void IrregularSampledSpectrumShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::IrregularSampledSpectrumShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void IrregularSampledSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    IrregularSampledSpectrumShaderNode::IrregularSampledSpectrumShaderNode(Context &context) :
        ShaderNode(context, true), m_spectrumType(VLRSpectrumType_NA), m_lambdas(nullptr), m_values(nullptr), m_numSamples(2) {
        m_lambdas = new float[2];
        m_values = new float[2];
        m_lambdas[0] = 0.0f;
        m_lambdas[1] = 1000.0f;
        m_values[0] = m_values[1] = 1.0f;
        setupNodeDescriptor();
    }

    IrregularSampledSpectrumShaderNode::~IrregularSampledSpectrumShaderNode() {
        if (m_values) {
            delete[] m_lambdas;
            delete[] m_values;
        }
    }

    void IrregularSampledSpectrumShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SpectrumNodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::IrregularSampledSpectrumShaderNode>();
#if defined(VLR_USE_SPECTRAL_RENDERING)
        VLRAssert(m_numSamples <= lengthof(nodeData.values), "Number of sample points must not be greater than %u.", lengthof(nodeData.values));
        std::copy_n(m_lambdas, m_numSamples, nodeData.lambdas);
        std::copy_n(m_values, m_numSamples, nodeData.values);
        nodeData.numSamples = m_numSamples;
#else
        IrregularSampledSpectrum spectrum(m_lambdas, m_values, m_numSamples);
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        float RGB[3];
        transformToRenderingRGB(m_spectrumType, XYZ, RGB);
        nodeData.value = RGBSpectrum(std::fmax(0.0f, RGB[0]), std::fmax(0.0f, RGB[1]), std::fmax(0.0f, RGB[2]));
#endif

        m_context.updateSpectrumNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void IrregularSampledSpectrumShaderNode::setImmediateValueSpectrum(VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples) {
        if (m_values) {
            delete[] m_lambdas;
            delete[] m_values;
        }
        m_spectrumType = spectrumType;
        m_lambdas = new float[numSamples];
        m_values = new float[numSamples];
        std::copy_n(lambdas, numSamples, m_lambdas);
        std::copy_n(values, numSamples, m_values);
        m_numSamples = numSamples;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Vector3DToSpectrumShaderNode::OptiXProgramSets;

    // static
    void Vector3DToSpectrumShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::Vector3DToSpectrumShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Vector3DToSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Vector3DToSpectrumShaderNode::Vector3DToSpectrumShaderNode(Context &context) :
        ShaderNode(context), m_immVector3D(Vector3D(0, 0, 0)), m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(ColorSpace::Rec709_D65) {
        setupNodeDescriptor();
    }

    Vector3DToSpectrumShaderNode::~Vector3DToSpectrumShaderNode() {
    }

    void Vector3DToSpectrumShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::Vector3DToSpectrumShaderNode>();
        nodeData.nodeVector3D = m_nodeVector3D.getSharedType();
        nodeData.immVector3D = m_immVector3D;
        nodeData.spectrumType = m_spectrumType;
        nodeData.colorSpace = m_colorSpace;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool Vector3DToSpectrumShaderNode::setNodeVector3D(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Vector3D)
            return false;
        m_nodeVector3D = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Vector3DToSpectrumShaderNode::setImmediateValueVector3D(const Vector3D &value) {
        m_immVector3D = value;
        setupNodeDescriptor();
    }

    void Vector3DToSpectrumShaderNode::setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, ColorSpace colorSpace) {
        m_spectrumType = spectrumType;
        m_colorSpace = colorSpace;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> ScaleAndOffsetUVTextureMap2DShaderNode::OptiXProgramSets;

    // static
    void ScaleAndOffsetUVTextureMap2DShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_TextureCoordinates, "VLR::ScaleAndOffsetUVTextureMap2DShaderNode_TextureCoordinates",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void ScaleAndOffsetUVTextureMap2DShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    ScaleAndOffsetUVTextureMap2DShaderNode::ScaleAndOffsetUVTextureMap2DShaderNode(Context &context) :
        ShaderNode(context), m_offset{ 0.0f, 0.0f }, m_scale{ 1.0f, 1.0f } {
        setupNodeDescriptor();
    }

    ScaleAndOffsetUVTextureMap2DShaderNode::~ScaleAndOffsetUVTextureMap2DShaderNode() {
    }

    void ScaleAndOffsetUVTextureMap2DShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::ScaleAndOffsetUVTextureMap2DShaderNode>();
        nodeData.offset[0] = m_offset[0];
        nodeData.offset[1] = m_offset[1];
        nodeData.scale[0] = m_scale[0];
        nodeData.scale[1] = m_scale[1];

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void ScaleAndOffsetUVTextureMap2DShaderNode::setValues(const float offset[2], const float scale[2]) {
        std::copy_n(offset, 2, m_offset);
        std::copy_n(scale, 2, m_scale);
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Image2DTextureShaderNode::OptiXProgramSets;
    std::map<uint32_t, LinearImage2D*> Image2DTextureShaderNode::NullImages;

    // static
    void Image2DTextureShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_float, "VLR::Image2DTextureShaderNode_float",
            VLRShaderNodeSocketType_float2, "VLR::Image2DTextureShaderNode_float2",
            VLRShaderNodeSocketType_float3, "VLR::Image2DTextureShaderNode_float3",
            VLRShaderNodeSocketType_float4, "VLR::Image2DTextureShaderNode_float4",
            VLRShaderNodeSocketType_Normal3D, "VLR::Image2DTextureShaderNode_Normal3D",
            VLRShaderNodeSocketType_Spectrum, "VLR::Image2DTextureShaderNode_Spectrum",
            VLRShaderNodeSocketType_Alpha, "VLR::Image2DTextureShaderNode_Alpha",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        uint8_t nullData[] = { 255, 0, 255, 255 };
        LinearImage2D* nullImage = new LinearImage2D(context, nullData, 1, 1, DataFormat::RGBA8x4, VLRSpectrumType_Reflectance, ColorSpace::Rec709_D65);

        OptiXProgramSets[context.getID()] = programSet;
        NullImages[context.getID()] = nullImage;
    }

    // static
    void Image2DTextureShaderNode::finalize(Context &context) {
        delete NullImages.at(context.getID());

        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Image2DTextureShaderNode::Image2DTextureShaderNode(Context &context) :
        ShaderNode(context), m_image(nullptr) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setBuffer(NullImages.at(m_context.getID())->getOptiXObject());
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);

        setupNodeDescriptor();
    }

    Image2DTextureShaderNode::~Image2DTextureShaderNode() {
        m_optixTextureSampler->destroy();
    }

    void Image2DTextureShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::Image2DTextureShaderNode>();
        nodeData.textureID = m_optixTextureSampler->getId();
        if (m_image) {
            nodeData.dataFormat = (unsigned int)m_image->getDataFormat();
            nodeData.spectrumType = m_image->getSpectrumType();

            // JP: GPUカーネル内でHWによってsRGBデガンマされて読まれる場合には、デガンマ済みとして捉える必要がある。
            // EN: Data should be regarded as post-degamma in the case that reading with sRGB degamma by HW in a GPU kernel.
            ColorSpace colorSpace = m_image->getColorSpace();
            if (m_image->needsHW_sRGB_degamma() && colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                colorSpace = ColorSpace::Rec709_D65;
            nodeData.colorSpace = (unsigned int)colorSpace;
        }
        else {
            nodeData.dataFormat = 0;
            nodeData.spectrumType = 0;
            nodeData.colorSpace = 0;
        }
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void Image2DTextureShaderNode::setImage(const Image2D* image) {
        m_image = image;
        if (m_image) {
            m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
            m_optixTextureSampler->setReadMode(m_image->needsHW_sRGB_degamma() ? RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB : RT_TEXTURE_READ_NORMALIZED_FLOAT);
        }
        else {
            m_optixTextureSampler->setBuffer(NullImages.at(m_context.getID())->getOptiXObject());
            m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        }
        setupNodeDescriptor();
    }

    void Image2DTextureShaderNode::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }

    void Image2DTextureShaderNode::setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y) {
        m_optixTextureSampler->setWrapMode(0, (RTwrapmode)x);
        m_optixTextureSampler->setWrapMode(1, (RTwrapmode)y);
    }

    bool Image2DTextureShaderNode::setNodeTexCoord(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_TextureCoordinates)
            return false;
        m_nodeTexCoord = outputSocket;
        setupNodeDescriptor();
        return true;
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> EnvironmentTextureShaderNode::OptiXProgramSets;

    // static
    void EnvironmentTextureShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::EnvironmentTextureShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EnvironmentTextureShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    EnvironmentTextureShaderNode::EnvironmentTextureShaderNode(Context &context) :
        ShaderNode(context), m_image(nullptr) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);

        setupNodeDescriptor();
    }

    EnvironmentTextureShaderNode::~EnvironmentTextureShaderNode() {
        m_optixTextureSampler->destroy();
    }

    void EnvironmentTextureShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        auto &nodeData = *nodeDesc.getData<Shared::EnvironmentTextureShaderNode>();
        nodeData.textureID = m_optixTextureSampler->getId();
        if (m_image) {
            nodeData.dataFormat = (unsigned int)m_image->getDataFormat();
            nodeData.colorSpace = (unsigned int)m_image->getColorSpace();
        }
        else {
            nodeData.dataFormat = 0;
            nodeData.colorSpace = 0;
        }
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void EnvironmentTextureShaderNode::setImage(const Image2D* image) {
        m_image = image;
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
        setupNodeDescriptor();
    }

    void EnvironmentTextureShaderNode::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }

    void EnvironmentTextureShaderNode::setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y) {
        m_optixTextureSampler->setWrapMode(0, (RTwrapmode)x);
        m_optixTextureSampler->setWrapMode(1, (RTwrapmode)y);
    }

    bool EnvironmentTextureShaderNode::setNodeTexCoord(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_TextureCoordinates)
            return false;
        m_nodeTexCoord = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void EnvironmentTextureShaderNode::createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const {
        uint32_t mapWidth = m_image->getWidth() / 4;
        uint32_t mapHeight = m_image->getHeight() / 4;
        Image2D* shrinkedImage = m_image->createShrinkedImage2D(mapWidth, mapHeight);
        Image2D* shrinkedYImage = shrinkedImage->createLuminanceImage2D();
        delete shrinkedImage;
        float* linearData = (float*)shrinkedYImage->createLinearImageData();
        for (int y = 0; y < mapHeight; ++y) {
            float theta = M_PI * (y + 0.5f) / mapHeight;
            for (int x = 0; x < mapWidth; ++x) {
                linearData[y * mapWidth + x] *= std::sin(theta);
            }
        }
        delete shrinkedYImage;

        importanceMap->initialize(m_context, linearData, mapWidth, mapHeight);

        delete[] linearData;
    }
}
