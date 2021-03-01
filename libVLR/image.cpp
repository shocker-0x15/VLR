#include "image.h"

namespace vlr {
    const size_t sizesOfDataFormats[static_cast<uint32_t>(DataFormat::NumFormats)] = {
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

    // TODO: ちょっとわかりにくい。
    // ShaderNodePlugのoptionとコンポーネント位置を分離して指定できるようにするとわかりやすそう。
    uint32_t getComponentStartIndex(DataFormat dataFormat, BumpType bumpType, ShaderNodePlugType ptype, uint32_t index) {
        uint32_t ret = 0xFFFFFFFF;

        switch (dataFormat) {
        case DataFormat::RGBA8x4:
        case DataFormat::RGBA16Fx4:
        case DataFormat::RGBA32Fx4:
        case DataFormat::BC1:
        case DataFormat::BC2:
        case DataFormat::BC3:
        case DataFormat::BC7: {
            switch (ptype) {
            case ShaderNodePlugType::float1:
                if (index < 4)
                    ret = index;
                break;
            case ShaderNodePlugType::float2:
                if (index < 3)
                    ret = index;
                break;
            case ShaderNodePlugType::float3:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::float4:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Point3D:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::Vector3D:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::Normal3D:
                // JP: いずれのバンプマップにも対応可能。
                if (bumpType == BumpType::HeightMap && index < 4)
                    ret = index;
                else if (bumpType != BumpType::HeightMap && index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Alpha:
                if (index < 1)
                    ret = 3;
                break;
            case ShaderNodePlugType::TextureCoordinates:
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
            switch (ptype) {
            case ShaderNodePlugType::float1:
                if (index < 3)
                    ret = index;
                break;
            case ShaderNodePlugType::float2:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::float3:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::Point3D:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Vector3D:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Normal3D:
                // JP: いずれのバンプマップにも対応可能。
                if (bumpType == BumpType::HeightMap && index < 3)
                    ret = index;
                else if (bumpType != BumpType::HeightMap && index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::TextureCoordinates:
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
            switch (ptype) {
            case ShaderNodePlugType::float1:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::float2:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Normal3D:
                VLRAssert_NotImplemented();
                break;
            case ShaderNodePlugType::TextureCoordinates:
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
            switch (ptype) {
            case ShaderNodePlugType::float1:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Normal3D:
                // JP: Height Map のみサポート。
                if (bumpType == BumpType::HeightMap && index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Spectrum:
                if (index < 1)
                    ret = index;
                break;
            default:
                break;
            }
            break;
        }
        case DataFormat::GrayA8x2: {
            switch (ptype) {
            case ShaderNodePlugType::float1:
                if (index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::float2:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Normal3D:
                // JP: Height Map のみサポート。
                if (bumpType == BumpType::HeightMap && index < 2)
                    ret = index;
                break;
            case ShaderNodePlugType::Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Alpha:
                if (index < 1)
                    ret = 1;
                break;
            case ShaderNodePlugType::TextureCoordinates:
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
            switch (ptype) {
            case ShaderNodePlugType::Spectrum:
                if (index < 1)
                    ret = index;
                break;
            case ShaderNodePlugType::Alpha:
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



    // static
    void Image2D::initialize(Context &context) {
        LinearImage2D::initialize(context);
        BlockCompressedImage2D::initialize(context);
    }

    // static
    void Image2D::finalize(Context &context) {
        BlockCompressedImage2D::finalize(context);
        LinearImage2D::finalize(context);
    }

    DataFormat Image2D::getInternalFormat(DataFormat inputFormat, SpectrumType spectrumType) {
        const auto asis = [](DataFormat inputFormat) {
            switch (inputFormat) {
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
            return DataFormat(0);
        };

        if (spectrumType == SpectrumType::NA) {
            return asis(inputFormat);
        }
        else {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            switch (inputFormat) {
            case DataFormat::RGB8x3:
                return DataFormat::uvsA8x4;
            case DataFormat::RGB_8x4:
                return DataFormat::uvsA8x4;
            case DataFormat::RGBA8x4:
                return DataFormat::uvsA8x4;
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
        return DataFormat(0);
    }

    Image2D::Image2D(Context &context, uint32_t width, uint32_t height,
                     DataFormat originalDataFormat, SpectrumType spectrumType, ColorSpace colorSpace) :
        Queryable(context), m_width(width), m_height(height),
        m_originalDataFormat(originalDataFormat), m_spectrumType(spectrumType), m_colorSpace(colorSpace)
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
        if (m_optixDataBuffer.isInitialized())
            m_optixDataBuffer.finalize();
    }

    const cudau::Array &Image2D::getOptiXObject() const {
        if (m_optixDataBuffer.isInitialized())
            return m_optixDataBuffer;

        optixu::Context optixContext = m_context.getOptiXContext();
        CUcontext cudaContext = optixContext.getCUcontext();
        cudau::ArraySurface useSurfaceLoadStore = cudau::ArraySurface::Disable;
        cudau::ArrayTextureGather useTextureGather = cudau::ArrayTextureGather::Enable;
        uint32_t numMipmapLevels = 1;

#define VLR_TEMP_EXPR0(format, elementType, numChs) \
    case format: \
        m_optixDataBuffer.initialize2D(cudaContext, elementType, numChs, \
                                       useSurfaceLoadStore, useTextureGather, \
                                       m_width, m_height, numMipmapLevels); \
        break

        if (m_dataFormat >= DataFormat::BC1 && m_dataFormat <= DataFormat::BC7) {
            switch (m_dataFormat) {
                VLR_TEMP_EXPR0(DataFormat::BC1, cudau::ArrayElementType::BC1_UNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC2, cudau::ArrayElementType::BC2_UNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC3, cudau::ArrayElementType::BC3_UNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC4, cudau::ArrayElementType::BC4_UNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC4_Signed, cudau::ArrayElementType::BC4_SNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC5, cudau::ArrayElementType::BC5_UNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC5_Signed, cudau::ArrayElementType::BC5_SNorm, 1);
                VLR_TEMP_EXPR0(DataFormat::BC6H, cudau::ArrayElementType::BC6H_UF16, 1);
                VLR_TEMP_EXPR0(DataFormat::BC6H_Signed, cudau::ArrayElementType::BC6H_SF16, 1);
                VLR_TEMP_EXPR0(DataFormat::BC7, cudau::ArrayElementType::BC7_UNorm, 1);
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
        }
        else {
            switch (m_dataFormat) {
                VLR_TEMP_EXPR0(DataFormat::RGB_8x4, cudau::ArrayElementType::UInt8, 4);
                VLR_TEMP_EXPR0(DataFormat::RGBA8x4, cudau::ArrayElementType::UInt8, 4);
                VLR_TEMP_EXPR0(DataFormat::RGBA16Fx4, cudau::ArrayElementType::Float16, 4);
                VLR_TEMP_EXPR0(DataFormat::RGBA32Fx4, cudau::ArrayElementType::Float32, 4);
                VLR_TEMP_EXPR0(DataFormat::RG32Fx2, cudau::ArrayElementType::Float32, 2);
                VLR_TEMP_EXPR0(DataFormat::Gray32F, cudau::ArrayElementType::Float32, 1);
                VLR_TEMP_EXPR0(DataFormat::Gray8, cudau::ArrayElementType::UInt8, 1);
                VLR_TEMP_EXPR0(DataFormat::GrayA8x2, cudau::ArrayElementType::UInt8, 2);
                VLR_TEMP_EXPR0(DataFormat::uvsA8x4, cudau::ArrayElementType::UInt8, 4);
                VLR_TEMP_EXPR0(DataFormat::uvsA16Fx4, cudau::ArrayElementType::Float16, 4);
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
        }

        return m_optixDataBuffer;

#undef VLR_TEMP_EXPR0
    }



    template <bool enableDegamma>
    struct sRGB_D65_ColorSpaceFunc {
        static float degamma(float v) {
            if constexpr (enableDegamma)
                return sRGB_degamma(v);
            else
                return v;
        }
        static uint8_t degamma(uint8_t v) {
            if constexpr (enableDegamma)
                return static_cast<uint8_t>(std::min<uint32_t>(255, 256 * sRGB_degamma(v / 255.0f)));
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
            if constexpr (enableDegamma)
                return sRGB_degamma(v);
            else
                return v;
        }
        static uint8_t degamma(uint8_t v) {
            if constexpr (enableDegamma)
                return static_cast<uint8_t>(std::min<uint32_t>(255, 256 * sRGB_degamma(v / 255.0f)));
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
        dstData.r = (half)ColorSpaceFunc::degamma(static_cast<float>(srcData.r));
        dstData.g = (half)ColorSpaceFunc::degamma(static_cast<float>(srcData.g));
        dstData.b = (half)ColorSpaceFunc::degamma(static_cast<float>(srcData.b));
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

        dstData.u = static_cast<half>(uv[0]);
        dstData.v = static_cast<half>(uv[1]);
        // JP: よくあるテクスチャーの値だとInfになってしまうため
        //     本来は割るべきところを割らないままにしておく。
        // EN: 
        dstData.s = static_cast<half>(b/* / UpsampledSpectrum::EqualEnergyReflectance()*/);
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
        dstData.a = static_cast<half>(1.0f);
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
        dstData.a = static_cast<half>(1.0f);
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
        dstData.a = static_cast<half>(srcData.a / 255.0f);
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
            auto srcHead = reinterpret_cast<const SrcType*>(srcData);
            auto dstHead = reinterpret_cast<SrcType*>(dstData);
            std::copy_n(srcHead, width * height, dstHead);
        }
        else {
            auto srcHead = reinterpret_cast<const SrcType*>(srcData);
            auto dstHead = reinterpret_cast<DstType*>(dstData);
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

    

    std::vector<ParameterInfo> LinearImage2D::ParameterInfos;

    // static
    void LinearImage2D::initialize(Context &context) {
    }

    // static
    void LinearImage2D::finalize(Context &context) {
    }

    LinearImage2D::LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height,
                                 DataFormat dataFormat, SpectrumType spectrumType, ColorSpace colorSpace) :
        Image2D(context, width, height, dataFormat, spectrumType, colorSpace), m_copyDone(false) {
        VLRAssert(dataFormat < DataFormat::BC1 || dataFormat > DataFormat::BC7, "Specified data format is a block compressed format.");
        m_data.resize(getStride() * getWidth() * getHeight());

        switch (dataFormat) {
        case DataFormat::RGB8x3: {
#if defined(VLR_USE_SPECTRAL_RENDERING)
            if (spectrumType != SpectrumType::NA) {
                if (spectrumType == SpectrumType::Reflectance ||
                    spectrumType == SpectrumType::IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB8x3, uvsA8x4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB8x3, uvsA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB8x3, uvsA8x4, sRGB_D65_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB8x3, uvsA8x4, sRGB_D65_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
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
            if (spectrumType != SpectrumType::NA) {
                if (spectrumType == SpectrumType::Reflectance ||
                    spectrumType == SpectrumType::IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB_8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB_8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGB_8x4, uvsA8x4, sRGB_D65_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGB_8x4, uvsA8x4, sRGB_D65_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
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
            if (spectrumType != SpectrumType::NA) {
                if (spectrumType == SpectrumType::Reflectance ||
                    spectrumType == SpectrumType::IndexOfRefraction) {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGBA8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGBA8x4, uvsA8x4, sRGB_E_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
                }
                else {
                    if (colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
                        processAllPixels<RGBA8x4, uvsA8x4, sRGB_D65_ColorSpaceFunc<true>>(linearData, m_data.data(), width, height);
                    else
                        processAllPixels<RGBA8x4, uvsA8x4, sRGB_D65_ColorSpaceFunc<false>>(linearData, m_data.data(), width, height);
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
            if (spectrumType != SpectrumType::NA) {
                if (spectrumType == SpectrumType::Reflectance ||
                    spectrumType == SpectrumType::IndexOfRefraction) {
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
        VLRAssert(width <= orgWidth && height <= orgHeight, "Image size must be smaller than the original.");
        std::vector<uint8_t> data;
        data.resize(stride * width * height);

        float deltaOrgX = orgWidth / width;
        float deltaOrgY = orgHeight / height;
        for (int y = 0; y < height; ++y) {
            float top = deltaOrgY * y;
            float bottom = deltaOrgY * (y + 1);
            uint32_t topPix = static_cast<uint32_t>(top);
            uint32_t bottomPix = static_cast<uint32_t>(ceilf(bottom)) - 1;

            for (int x = 0; x < width; ++x) {
                float left = deltaOrgX * x;
                float right = deltaOrgX * (x + 1);
                uint32_t leftPix = static_cast<uint32_t>(left);
                uint32_t rightPix = static_cast<uint32_t>(ceilf(right)) - 1;

                CompensatedSum<float> sumWeight(0);
                //float area = (bottom - top) * (right - left);

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

                switch (getDataFormat()) {
                case DataFormat::RGBA16Fx4: {
                    CompensatedSum<float> sumR(0), sumG(0), sumB(0), sumA(0);
                    RGBA16Fx4 pix;

                    uint32_t corners[] = { leftPix, topPix, rightPix, topPix, leftPix, bottomPix, rightPix, bottomPix };
                    for (int i = 0; i < 4; ++i) {
                        pix = get<RGBA16Fx4>(corners[2 * i + 0], corners[2 * i + 1]);
                        float weight = weightsCorners[i];
                        sumR += weight * float(pix.r);
                        sumG += weight * float(pix.g);
                        sumB += weight * float(pix.b);
                        sumA += weight * float(pix.a);
                        sumWeight += weight;
                    }

                    for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                        pix = get<RGBA16Fx4>(x, topPix);
                        float weightT = weightsEdges[0];
                        sumR += weightT * float(pix.r);
                        sumG += weightT * float(pix.g);
                        sumB += weightT * float(pix.b);
                        sumA += weightT * float(pix.a);

                        pix = get<RGBA16Fx4>(x, bottomPix);
                        float weightB = weightsEdges[3];
                        sumR += weightB * float(pix.r);
                        sumG += weightB * float(pix.g);
                        sumB += weightB * float(pix.b);
                        sumA += weightB * float(pix.a);
                    }
                    if (rightPix > (leftPix + 1))
                        sumWeight += 2 * (rightPix - leftPix - 1);
                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        pix = get<RGBA16Fx4>(leftPix, y);
                        float weightL = weightsEdges[1];
                        sumR += weightL * float(pix.r);
                        sumG += weightL * float(pix.g);
                        sumB += weightL * float(pix.b);
                        sumA += weightL * float(pix.a);

                        pix = get<RGBA16Fx4>(rightPix, y);
                        float weightR = weightsEdges[2];
                        sumR += weightR * float(pix.r);
                        sumG += weightR * float(pix.g);
                        sumB += weightR * float(pix.b);
                        sumA += weightR * float(pix.a);
                    }
                    if (bottomPix > (topPix + 1))
                        sumWeight += 2 * (bottomPix - topPix - 1);

                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                            pix = get<RGBA16Fx4>(x, y);
                            sumR += float(pix.r);
                            sumG += float(pix.g);
                            sumB += float(pix.b);
                            sumA += float(pix.a);
                        }
                    }
                    if (bottomPix > topPix && rightPix > leftPix)
                        sumWeight += (bottomPix - topPix - 1) * (rightPix - leftPix - 1);

                    *(RGBA16Fx4*)&data[(y * width + x) * stride] = RGBA16Fx4{ half(sumR / sumWeight), half(sumG / sumWeight), half(sumB / sumWeight), half(sumA / sumWeight) };
                    break;
                }
                case DataFormat::uvsA16Fx4: {
                    CompensatedSum<float> sum_u(0), sum_v(0), sum_s(0), sumA(0);
                    uvsA16Fx4 pix;

                    uint32_t corners[] = { leftPix, topPix, rightPix, topPix, leftPix, bottomPix, rightPix, bottomPix };
                    for (int i = 0; i < 4; ++i) {
                        pix = get<uvsA16Fx4>(corners[2 * i + 0], corners[2 * i + 1]);
                        float weight = weightsCorners[i];
                        sum_u += weightsCorners[i] * float(pix.u);
                        sum_v += weightsCorners[i] * float(pix.v);
                        sum_s += weightsCorners[i] * float(pix.s);
                        sumA += weightsCorners[i] * float(pix.a);
                        sumWeight += weight;
                    }

                    for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                        pix = get<uvsA16Fx4>(x, topPix);
                        float weightT = weightsEdges[0];
                        sum_u += weightT * float(pix.u);
                        sum_v += weightT * float(pix.v);
                        sum_s += weightT * float(pix.s);
                        sumA += weightT * float(pix.a);

                        pix = get<uvsA16Fx4>(x, bottomPix);
                        float weightB = weightsEdges[3];
                        sum_u += weightB * float(pix.u);
                        sum_v += weightB * float(pix.v);
                        sum_s += weightB * float(pix.s);
                        sumA += weightB * float(pix.a);
                    }
                    if (rightPix > (leftPix + 1))
                        sumWeight += 2 * (rightPix - leftPix - 1);
                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        pix = get<uvsA16Fx4>(leftPix, y);
                        float weightL = weightsEdges[1];
                        sum_u += weightL * float(pix.u);
                        sum_v += weightL * float(pix.v);
                        sum_s += weightL * float(pix.s);
                        sumA += weightL * float(pix.a);

                        pix = get<uvsA16Fx4>(rightPix, y);
                        float weightR = weightsEdges[2];
                        sum_u += weightR * float(pix.u);
                        sum_v += weightR * float(pix.v);
                        sum_s += weightR * float(pix.s);
                        sumA += weightR * float(pix.a);
                    }
                    if (bottomPix > (topPix + 1))
                        sumWeight += 2 * (bottomPix - topPix - 1);

                    for (uint32_t y = topPix + 1; y < bottomPix; ++y) {
                        for (uint32_t x = leftPix + 1; x < rightPix; ++x) {
                            pix = get<uvsA16Fx4>(x, y);
                            sum_u += float(pix.u);
                            sum_v += float(pix.v);
                            sum_s += float(pix.s);
                            sumA += float(pix.a);
                        }
                    }
                    if (bottomPix > topPix && rightPix > leftPix)
                        sumWeight += (bottomPix - topPix - 1) * (rightPix - leftPix - 1);

                    *reinterpret_cast<uvsA16Fx4*>(&data[(y * width + x) * stride]) = uvsA16Fx4{ half(sum_u / sumWeight), half(sum_v / sumWeight), half(sum_s / sumWeight), half(sumA / sumWeight) };
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
        switch (getDataFormat()) {
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
                switch (getDataFormat()) {
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

    const cudau::Array &LinearImage2D::getOptiXObject() const {
        const cudau::Array &buffer = Image2D::getOptiXObject();
        if (!m_copyDone) {
            auto dstData = m_optixDataBuffer.map<uint8_t>(0);
            std::copy(m_data.cbegin(), m_data.cend(), dstData);
            m_optixDataBuffer.unmap(0);
            m_copyDone = true;
        }
        return buffer;
    }



    std::vector<ParameterInfo> BlockCompressedImage2D::ParameterInfos;

    // static
    void BlockCompressedImage2D::initialize(Context& context) {
    }

    // static
    void BlockCompressedImage2D::finalize(Context& context) {
    }
    
    BlockCompressedImage2D::BlockCompressedImage2D(Context &context, const uint8_t* const* data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height, 
                                                   DataFormat dataFormat, SpectrumType spectrumType, ColorSpace colorSpace) :
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

    const cudau::Array &BlockCompressedImage2D::getOptiXObject() const {
        const cudau::Array &buffer = Image2D::getOptiXObject();
        if (!m_copyDone) {
            for (int mipLevel = 0; mipLevel < m_optixDataBuffer.getNumMipmapLevels(); ++mipLevel) {
                const std::vector<uint8_t> &srcMipData = m_data[mipLevel];
                auto dstData = m_optixDataBuffer.map<uint8_t>(mipLevel);
                std::copy(srcMipData.cbegin(), srcMipData.cend(), dstData);
                m_optixDataBuffer.unmap(mipLevel);
            }
            m_copyDone = true;
        }
        return buffer;
    }
}
