#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>



namespace DDS {
    enum class Format : uint32_t {
        BC1_UNorm = 71,
        BC1_UNorm_sRGB = 72,
        BC2_UNorm = 74,
        BC2_UNorm_sRGB = 75,
        BC3_UNorm = 77,
        BC3_UNorm_sRGB = 78,
        BC4_UNorm = 80,
        BC4_SNorm = 81,
        BC5_UNorm = 83,
        BC5_SNorm = 84,
        BC6H_UF16 = 95,
        BC6H_SF16 = 96,
        BC7_UNorm = 98,
        BC7_UNorm_sRGB = 99,
    };

    struct Header {
        struct Flags {
            enum Value : uint32_t {
                Caps = 1 << 0,
                Height = 1 << 1,
                Width = 1 << 2,
                Pitch = 1 << 3,
                PixelFormat = 1 << 12,
                MipMapCount = 1 << 17,
                LinearSize = 1 << 19,
                Depth = 1 << 23
            } value;

            Flags() : value((Value)0) {}
            Flags(Value v) : value(v) {}

            Flags operator&(Flags v) const {
                return (Value)(value & v.value);
            }
            Flags operator|(Flags v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct PFFlags {
            enum Value : uint32_t {
                AlphaPixels = 1 << 0,
                Alpha = 1 << 1,
                FourCC = 1 << 2,
                PaletteIndexed4 = 1 << 3,
                PaletteIndexed8 = 1 << 5,
                RGB = 1 << 6,
                Luminance = 1 << 17,
                BumpDUDV = 1 << 19,
            } value;

            PFFlags() : value((Value)0) {}
            PFFlags(Value v) : value(v) {}

            PFFlags operator&(PFFlags v) const {
                return (Value)(value & v.value);
            }
            PFFlags operator|(PFFlags v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct Caps {
            enum Value : uint32_t {
                Alpha = 1 << 1,
                Complex = 1 << 3,
                Texture = 1 << 12,
                MipMap = 1 << 22,
            } value;

            Caps() : value((Value)0) {}
            Caps(Value v) : value(v) {}

            Caps operator&(Caps v) const {
                return (Value)(value & v.value);
            }
            Caps operator|(Caps v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct Caps2 {
            enum Value : uint32_t {
                CubeMap = 1 << 9,
                CubeMapPositiveX = 1 << 10,
                CubeMapNegativeX = 1 << 11,
                CubeMapPositiveY = 1 << 12,
                CubeMapNegativeY = 1 << 13,
                CubeMapPositiveZ = 1 << 14,
                CubeMapNegativeZ = 1 << 15,
                Volume = 1 << 22,
            } value;

            Caps2() : value((Value)0) {}
            Caps2(Value v) : value(v) {}

            Caps2 operator&(Caps2 v) const {
                return (Value)(value & v.value);
            }
            Caps2 operator|(Caps2 v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        uint32_t m_magic;
        uint32_t m_size;
        Flags m_flags;
        uint32_t m_height;
        uint32_t m_width;
        uint32_t m_pitchOrLinearSize;
        uint32_t m_depth;
        uint32_t m_mipmapCount;
        uint32_t m_reserved1[11];
        uint32_t m_PFSize;
        PFFlags m_PFFlags;
        uint32_t m_fourCC;
        uint32_t m_RGBBitCount;
        uint32_t m_RBitMask;
        uint32_t m_GBitMask;
        uint32_t m_BBitMask;
        uint32_t m_RGBAlphaBitMask;
        Caps m_caps;
        Caps2 m_caps2;
        uint32_t m_reservedCaps[2];
        uint32_t m_reserved2;
    };
    static_assert(sizeof(Header) == 128, "sizeof(Header) must be 128.");

    struct HeaderDX10 {
        Format m_format;
        uint32_t m_dimension;
        uint32_t m_miscFlag;
        uint32_t m_arraySize;
        uint32_t m_miscFlag2;
    };
    static_assert(sizeof(HeaderDX10) == 20, "sizeof(HeaderDX10) must be 20.");

    static uint8_t** load(const char* filepath, int32_t* width, int32_t* height, int32_t* mipCount, size_t** sizes, Format* format) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            hpprintf("Not found: %s\n", filepath);
            return nullptr;
        }

        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();

        ifs.clear();
        ifs.seekg(0, std::ios::beg);

        Header header;
        ifs.read((char*)& header, sizeof(Header));
        if (header.m_magic != 0x20534444 || header.m_fourCC != 0x30315844) {
            hpprintf("Non dds (dx10) file: %s", filepath);
            return nullptr;
        }

        HeaderDX10 dx10Header;
        ifs.read((char*)& dx10Header, sizeof(HeaderDX10));

        *width = header.m_width;
        *height = header.m_height;
        *format = (Format)dx10Header.m_format;

        if (*format != Format::BC1_UNorm && *format != Format::BC1_UNorm_sRGB &&
            *format != Format::BC2_UNorm && *format != Format::BC2_UNorm_sRGB &&
            *format != Format::BC3_UNorm && *format != Format::BC3_UNorm_sRGB &&
            *format != Format::BC4_UNorm && *format != Format::BC4_SNorm &&
            *format != Format::BC5_UNorm && *format != Format::BC5_SNorm &&
            *format != Format::BC6H_UF16 && *format != Format::BC6H_SF16 &&
            *format != Format::BC7_UNorm && *format != Format::BC7_UNorm_sRGB) {
            hpprintf("No support for non block compressed formats: %s", filepath);
            return nullptr;
        }

        const size_t dataSize = fileSize - (sizeof(Header) + sizeof(HeaderDX10));

        *mipCount = 1;
        if ((header.m_flags & Header::Flags::MipMapCount) != 0)
            *mipCount = header.m_mipmapCount;

        uint8_t** data = new uint8_t * [*mipCount];
        *sizes = new size_t[*mipCount];
        int32_t mipWidth = *width;
        int32_t mipHeight = *height;
        uint32_t blockSize = 16;
        if (*format == Format::BC1_UNorm || *format == Format::BC1_UNorm_sRGB ||
            *format == Format::BC4_UNorm || *format == Format::BC4_SNorm)
            blockSize = 8;
        size_t cumDataSize = 0;
        for (int i = 0; i < *mipCount; ++i) {
            int32_t bw = (mipWidth + 3) / 4;
            int32_t bh = (mipHeight + 3) / 4;
            size_t mipDataSize = bw * bh * blockSize;

            data[i] = new uint8_t[mipDataSize];
            (*sizes)[i] = mipDataSize;
            ifs.read((char*)data[i], mipDataSize);
            cumDataSize += mipDataSize;

            mipWidth = std::max<int32_t>(1, mipWidth / 2);
            mipHeight = std::max<int32_t>(1, mipHeight / 2);
        }
        Assert(cumDataSize == dataSize, "Data size mismatch.");

        return data;
    }

    static void free(uint8_t** data, int32_t mipCount, size_t* sizes) {
        for (int i = mipCount - 1; i >= 0; --i)
            delete[] data[i];
        delete[] sizes;
        delete[] data;
    }
}



static std::map<std::tuple<std::string, std::string, std::string>, VLRCpp::Image2DRef> s_image2DCache;

// TODO: Should colorSpace be determined from the read image?
VLRCpp::Image2DRef loadImage2D(const VLRCpp::ContextRef &context, const std::string &filepath, const std::string &spectrumType, const std::string &colorSpace) {
    using namespace VLRCpp;
    using namespace VLR;

    Image2DRef ret;

    const auto tolower = [](std::string str) {
        const auto tolower = [](unsigned char c) { return std::tolower(c); };
        std::transform(str.cbegin(), str.cend(), str.begin(), tolower);
        return str;
    };

    auto key = std::make_tuple(filepath, tolower(spectrumType), tolower(colorSpace));
    if (s_image2DCache.count(key))
        return s_image2DCache.at(key);

    hpprintf("Read image: %s...", filepath.c_str());

    bool fileExists = false;
    {
        std::ifstream ifs(filepath);
        fileExists = ifs.is_open();
    }
    if (!fileExists) {
        hpprintf("Not found.\n");
        return ret;
    }

    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

    //#define OVERRIDE_BY_DDS

#if defined(OVERRIDE_BY_DDS)
    std::string ddsFilepath = filepath;
    ddsFilepath = filepath.substr(0, filepath.find_last_of('.'));
    ddsFilepath += ".dds";
    {
        std::ifstream ifs(ddsFilepath);
        if (ifs.is_open())
            ext = "dds";
    }
#endif

    if (ext == "exr") {
        using namespace Imf;
        using namespace Imath;
        RgbaInputFile file(filepath.c_str());
        Imf::Header header = file.header();

        Box2i dw = file.dataWindow();
        long width = dw.max.x - dw.min.x + 1;
        long height = dw.max.y - dw.min.y + 1;
        Array2D<Rgba> pixels{ height, width };
        pixels.resizeErase(height, width);
        file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
        file.readPixels(dw.min.y, dw.max.y);

        Rgba* linearImageData = new Rgba[width * height];
        Rgba* curDataHead = linearImageData;
        for (int i = 0; i < height; ++i) {
            std::copy_n(pixels[i], width, (Rgba*)curDataHead);
            for (int j = 0; j < width; ++j) {
                Rgba &pix = curDataHead[j];
                pix.r = pix.r >= 0.0f ? pix.r : (half)0.0f;
                pix.g = pix.g >= 0.0f ? pix.g : (half)0.0f;
                pix.b = pix.b >= 0.0f ? pix.b : (half)0.0f;
                pix.a = pix.a >= 0.0f ? pix.a : (half)0.0f;
            }
            curDataHead += width;
        }

        ret = context->createLinearImage2D((uint8_t*)linearImageData, width, height, "RGBA16Fx4", spectrumType.c_str(), colorSpace.c_str());

        delete[] linearImageData;
    }
    else if (ext == "dds") {
        int32_t width, height, mipCount;
        size_t* sizes;
        DDS::Format format;
#if defined(OVERRIDE_BY_DDS)
        uint8_t** data = DDS::load(ddsFilepath.c_str(), &width, &height, &mipCount, &sizes, &format);
#else
        uint8_t** data = DDS::load(filepath.c_str(), &width, &height, &mipCount, &sizes, &format);
#endif

        const auto translate = [](DDS::Format ddsFormat, const char** vlrFormat, bool* needsDegamma) {
            *needsDegamma = false;
            switch (ddsFormat) {
            case DDS::Format::BC1_UNorm:
                *vlrFormat = "BC1";
                break;
            case DDS::Format::BC1_UNorm_sRGB:
                *vlrFormat = "BC1";
                *needsDegamma = true;
                break;
            case DDS::Format::BC2_UNorm:
                *vlrFormat = "BC2";
                break;
            case DDS::Format::BC2_UNorm_sRGB:
                *vlrFormat = "BC2";
                *needsDegamma = true;
                break;
            case DDS::Format::BC3_UNorm:
                *vlrFormat = "BC3";
                break;
            case DDS::Format::BC3_UNorm_sRGB:
                *vlrFormat = "BC3";
                *needsDegamma = true;
                break;
            case DDS::Format::BC4_UNorm:
                *vlrFormat = "BC4";
                break;
            case DDS::Format::BC4_SNorm:
                *vlrFormat = "BC4_Signed";
                break;
            case DDS::Format::BC5_UNorm:
                *vlrFormat = "BC5";
                break;
            case DDS::Format::BC5_SNorm:
                *vlrFormat = "BC5_Signed";
                break;
            case DDS::Format::BC6H_UF16:
                *vlrFormat = "BC6H";
                break;
            case DDS::Format::BC6H_SF16:
                *vlrFormat = "BC6H_Signed";
                break;
            case DDS::Format::BC7_UNorm:
                *vlrFormat = "BC7";
                break;
            case DDS::Format::BC7_UNorm_sRGB:
                *vlrFormat = "BC7";
                *needsDegamma = true;
                break;
            default:
                break;
            }
        };

        const char* vlrFormat;
        bool needsDegamma;
        translate(format, &vlrFormat, &needsDegamma);

        ret = context->createBlockCompressedImage2D(data, sizes, mipCount, width, height, vlrFormat, spectrumType.c_str(), colorSpace.c_str());
        Assert(ret, "failed to load a block compressed texture.");

        DDS::free(data, mipCount, sizes);
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filepath.c_str(), &width, &height, &n, 0);
        if (n == 4)
            ret = context->createLinearImage2D(linearImageData, width, height, "RGBA8x4", spectrumType.c_str(), colorSpace.c_str());
        else if (n == 3)
            ret = context->createLinearImage2D(linearImageData, width, height, "RGB8x3", spectrumType.c_str(), colorSpace.c_str());
        else if (n == 2)
            ret = context->createLinearImage2D(linearImageData, width, height, "GrayA8x2", spectrumType.c_str(), colorSpace.c_str());
        else if (n == 1)
            ret = context->createLinearImage2D(linearImageData, width, height, "Gray8", spectrumType.c_str(), colorSpace.c_str());
        else
            Assert_ShouldNotBeCalled();
        stbi_image_free(linearImageData);
    }

    hpprintf("done.\n");

    s_image2DCache[key] = ret;

    return ret;
}
