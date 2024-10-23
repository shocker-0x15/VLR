#include "image.h"

#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#include "dds_loader.h"
#include "tinyexr.h"

#include "half/half.hpp"



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
        int32_t width, height;
        float* fp32Data;
        const char* errMsg = nullptr;
        int exrRet;
        EXRVersion exrVersion;
        exrRet = ParseEXRVersionFromFile(&exrVersion, filepath.c_str());
        VLRAssert(exrRet == TINYEXR_SUCCESS, "failed to parse the exr version.");
        EXRHeader exrHeader;
        exrRet = ParseEXRHeaderFromFile(&exrHeader, &exrVersion, filepath.c_str(), &errMsg);
        VLRAssert(exrRet == TINYEXR_SUCCESS, "failed to parse the exr header.");
        exrRet = LoadEXR(&fp32Data, &width, &height, filepath.c_str(), &errMsg);
        VLRAssert(exrRet == TINYEXR_SUCCESS, "failed to read the exr."); // TODO: error handling.

        std::vector<half_float::half> fp16Data(width * height * 4);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t idx = y * width + x;
                fp16Data[4 * idx + 0] = static_cast<half_float::half>(fp32Data[4 * idx + 0]);
                fp16Data[4 * idx + 1] = static_cast<half_float::half>(fp32Data[4 * idx + 1]);
                fp16Data[4 * idx + 2] = static_cast<half_float::half>(fp32Data[4 * idx + 2]);
                fp16Data[4 * idx + 3] = static_cast<half_float::half>(fp32Data[4 * idx + 3]);
            }
        }
        free(fp32Data);

        ret = context->createLinearImage2D(
            reinterpret_cast<uint8_t*>(fp16Data.data()), width, height, "RGBA16Fx4",
            spectrumType.c_str(), colorSpace.c_str());
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

        ret = context->createBlockCompressedImage2D(
            data, sizes, mipCount, width, height, vlrFormat,
            spectrumType.c_str(), colorSpace.c_str());
        Assert(ret, "failed to load a block compressed texture.");

        dds::free(data, sizes);
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

void writePNG(const std::filesystem::path &filePath, uint32_t width, uint32_t height, const uint32_t* data) {
    stbi_write_png(filePath.string().c_str(), width, height, 4, data, width * 4);
}

void writeEXR(const std::filesystem::path &filePath, uint32_t width, uint32_t height, const float* data) {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 4;

    std::vector<float> images[4];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);
    images[3].resize(width * height);

    bool flipY = false;
    float brightnessScale = 1.0f;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t srcIdx = 4 * (y * width + x);
            uint32_t dstIdx = (flipY ? (height - 1 - y) : y) * width + x;
            images[0][dstIdx] = brightnessScale * data[srcIdx + 0];
            images[1][dstIdx] = brightnessScale * data[srcIdx + 1];
            images[2][dstIdx] = brightnessScale * data[srcIdx + 2];
            images[3][dstIdx] = brightnessScale * data[srcIdx + 3];
        }
    }

    float* image_ptr[4];
    image_ptr[0] = &(images[3].at(0)); // A
    image_ptr[1] = &(images[2].at(0)); // B
    image_ptr[2] = &(images[1].at(0)); // G
    image_ptr[3] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 4;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
    strncpy(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
    strncpy(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
    strncpy(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

    header.pixel_types = (int32_t*)malloc(sizeof(int32_t) * header.num_channels);
    header.requested_pixel_types = (int32_t*)malloc(sizeof(int32_t) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = nullptr;
    int32_t ret = SaveEXRImageToFile(&image, &header, filePath.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}
