#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>

#include "dds_loader.h"



static std::map<std::tuple<std::string, std::string, std::string>, vlr::Image2DRef> s_image2DCache;

// TODO: Should colorSpace be determined from the read image?
vlr::Image2DRef loadImage2D(const vlr::ContextRef &context, const std::string &filepath, const std::string &spectrumType, const std::string &colorSpace) {
    using namespace vlr;

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
        dds::Format format;
#if defined(OVERRIDE_BY_DDS)
        uint8_t** data = dds::load(ddsFilepath.c_str(), &width, &height, &mipCount, &sizes, &format);
#else
        uint8_t** data = dds::load(filepath.c_str(), &width, &height, &mipCount, &sizes, &format);
#endif

        const auto translate = [](dds::Format ddsFormat, const char** vlrFormat, bool* needsDegamma) {
            *needsDegamma = false;
            switch (ddsFormat) {
            case dds::Format::BC1_UNorm:
                *vlrFormat = "BC1";
                break;
            case dds::Format::BC1_UNorm_sRGB:
                *vlrFormat = "BC1";
                *needsDegamma = true;
                break;
            case dds::Format::BC2_UNorm:
                *vlrFormat = "BC2";
                break;
            case dds::Format::BC2_UNorm_sRGB:
                *vlrFormat = "BC2";
                *needsDegamma = true;
                break;
            case dds::Format::BC3_UNorm:
                *vlrFormat = "BC3";
                break;
            case dds::Format::BC3_UNorm_sRGB:
                *vlrFormat = "BC3";
                *needsDegamma = true;
                break;
            case dds::Format::BC4_UNorm:
                *vlrFormat = "BC4";
                break;
            case dds::Format::BC4_SNorm:
                *vlrFormat = "BC4_Signed";
                break;
            case dds::Format::BC5_UNorm:
                *vlrFormat = "BC5";
                break;
            case dds::Format::BC5_SNorm:
                *vlrFormat = "BC5_Signed";
                break;
            case dds::Format::BC6H_UF16:
                *vlrFormat = "BC6H";
                break;
            case dds::Format::BC6H_SF16:
                *vlrFormat = "BC6H_Signed";
                break;
            case dds::Format::BC7_UNorm:
                *vlrFormat = "BC7";
                break;
            case dds::Format::BC7_UNorm_sRGB:
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

        dds::free(data, mipCount, sizes);
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
