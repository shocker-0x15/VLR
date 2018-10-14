#include "textures.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Textures

    const size_t sizesOfDataFormats[(uint32_t)NumVLRDataFormats] = {
        sizeof(RGB8x3),
        sizeof(RGB_8x4),
        sizeof(RGBA8x4),
        sizeof(RGBA16Fx4),
        sizeof(RGBA32Fx4),
        sizeof(RG32Fx2),
        sizeof(Gray32F),
        sizeof(Gray8),
    };

    VLRDataFormat Image2D::getInternalFormat(VLRDataFormat inputFormat) {
        switch (inputFormat) {
        case VLRDataFormat_RGB8x3:
            return VLRDataFormat_RGBA8x4;
        case VLRDataFormat_RGB_8x4:
            return VLRDataFormat_RGBA8x4;
        case VLRDataFormat_RGBA8x4:
            return VLRDataFormat_RGBA8x4;
        case VLRDataFormat_RGBA16Fx4:
            return VLRDataFormat_RGBA16Fx4;
        case VLRDataFormat_RGBA32Fx4:
            return VLRDataFormat_RGBA32Fx4;
        case VLRDataFormat_RG32Fx2:
            return VLRDataFormat_RG32Fx2;
        case VLRDataFormat_Gray32F:
            return VLRDataFormat_Gray32F;
        case VLRDataFormat_Gray8:
            return VLRDataFormat_Gray8;
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }
        return VLRDataFormat_RGBA8x4;
    }

    Image2D::Image2D(Context &context, uint32_t width, uint32_t height, VLRDataFormat dataFormat) :
        Object(context), m_width(width), m_height(height), m_dataFormat(dataFormat), m_initOptiXObject(false) {
    }

    Image2D::~Image2D() {
        if (m_optixDataBuffer)
            m_optixDataBuffer->destroy();
    }

    optix::Buffer Image2D::getOptiXObject() const {
        if (m_initOptiXObject)
            return m_optixDataBuffer;

        optix::Context optixContext = m_context.getOptiXContext();
        switch (m_dataFormat) {
        case VLRDataFormat_RGB8x3:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE3, m_width, m_height);
            break;
        case VLRDataFormat_RGB_8x4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height);
            break;
        case VLRDataFormat_RGBA8x4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height);
            break;
        case VLRDataFormat_RGBA16Fx4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_HALF4, m_width, m_height);
            break;
        case VLRDataFormat_RGBA32Fx4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_width, m_height);
            break;
        case VLRDataFormat_RG32Fx2:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, m_width, m_height);
            break;
        case VLRDataFormat_Gray32F:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, m_width, m_height);
            break;
        case VLRDataFormat_Gray8:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, m_width, m_height);
            break;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }

        m_initOptiXObject = true;

        return m_optixDataBuffer;
    }



    LinearImage2D::LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat dataFormat, bool applyDegamma) :
        Image2D(context, width, height, Image2D::getInternalFormat(dataFormat)), m_copyDone(false) {
        m_data.resize(getStride() * getWidth() * getHeight());

        switch (dataFormat) {
        case VLRDataFormat_RGB8x3: {
            auto srcHead = (const RGB8x3*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    const RGB8x3 &src = *(srcLineHead + x);
                    RGBA8x4 &dst = *(dstLineHead + x);
                    dst.r = applyDegamma ? std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.r / 255.0f)) : src.r;
                    dst.g = applyDegamma ? std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.g / 255.0f)) : src.g;
                    dst.b = applyDegamma ? std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.b / 255.0f)) : src.b;
                    dst.a = 255;
                }
            }
            break;
        }
        case VLRDataFormat_RGB_8x4: {
            auto srcHead = (const RGB_8x4*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    const RGB_8x4 &src = *(srcLineHead + x);
                    RGBA8x4 &dst = *(dstLineHead + x);
                    dst.r = applyDegamma ? std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.r / 255.0f)) : src.r;
                    dst.g = applyDegamma ? std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.g / 255.0f)) : src.g;
                    dst.b = applyDegamma ? std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.b / 255.0f)) : src.b;
                    dst.a = 255;
                }
            }
            break;
        }
        case VLRDataFormat_RGBA8x4: {
            auto srcHead = (const RGBA8x4*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            if (applyDegamma) {
                for (int y = 0; y < height; ++y) {
                    auto srcLineHead = srcHead + width * y;
                    auto dstLineHead = dstHead + width * y;
                    for (int x = 0; x < width; ++x) {
                        const RGBA8x4 &src = *(srcLineHead + x);
                        RGBA8x4 &dst = *(dstLineHead + x);
                        dst.r = std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.r / 255.0f));
                        dst.g = std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.g / 255.0f));
                        dst.b = std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.b / 255.0f));
                        dst.a = src.a;
                    }
                }
            }
            else {
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_RGBA16Fx4: {
            auto srcHead = (const RGBA16Fx4*)linearData;
            auto dstHead = (RGBA16Fx4*)m_data.data();
            if (applyDegamma) {
                for (int y = 0; y < height; ++y) {
                    auto srcLineHead = srcHead + width * y;
                    auto dstLineHead = dstHead + width * y;
                    for (int x = 0; x < width; ++x) {
                        const RGBA16Fx4 &src = *(srcLineHead + x);
                        RGBA16Fx4 &dst = *(dstLineHead + x);
                        dst.r = (half)sRGB_degamma_s((float)src.r);
                        dst.g = (half)sRGB_degamma_s((float)src.g);
                        dst.b = (half)sRGB_degamma_s((float)src.b);
                        dst.a = src.a;
                    }
                }
            }
            else {
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_RGBA32Fx4: {
            auto srcHead = (const RGBA32Fx4*)linearData;
            auto dstHead = (RGBA32Fx4*)m_data.data();
            if (applyDegamma) {
                for (int y = 0; y < height; ++y) {
                    auto srcLineHead = srcHead + width * y;
                    auto dstLineHead = dstHead + width * y;
                    for (int x = 0; x < width; ++x) {
                        const RGBA32Fx4 &src = *(srcLineHead + x);
                        RGBA32Fx4 &dst = *(dstLineHead + x);
                        dst.r = sRGB_degamma_s(src.r);
                        dst.g = sRGB_degamma_s(src.g);
                        dst.b = sRGB_degamma_s(src.b);
                        dst.a = src.a;
                    }
                }
            }
            else {
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_RG32Fx2: {
            auto srcHead = (const RG32Fx2*)linearData;
            auto dstHead = (RG32Fx2*)m_data.data();
            if (applyDegamma) {
                for (int y = 0; y < height; ++y) {
                    auto srcLineHead = srcHead + width * y;
                    auto dstLineHead = dstHead + width * y;
                    for (int x = 0; x < width; ++x) {
                        const RG32Fx2 &src = *(srcLineHead + x);
                        RG32Fx2 &dst = *(dstLineHead + x);
                        dst.r = sRGB_degamma_s(src.r);
                        dst.g = sRGB_degamma_s(src.g);
                    }
                }
            }
            else {
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_Gray32F: {
            auto srcHead = (const Gray32F*)linearData;
            auto dstHead = (Gray32F*)m_data.data();
            if (applyDegamma) {
                for (int y = 0; y < height; ++y) {
                    auto srcLineHead = srcHead + width * y;
                    auto dstLineHead = dstHead + width * y;
                    for (int x = 0; x < width; ++x) {
                        const Gray32F &src = *(srcLineHead + x);
                        Gray32F &dst = *(dstLineHead + x);
                        dst.v = sRGB_degamma_s(src.v);
                    }
                }
            }
            else {
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_Gray8: {
            auto srcHead = (const Gray8*)linearData;
            auto dstHead = (Gray8*)m_data.data();
            if (applyDegamma) {
                for (int y = 0; y < height; ++y) {
                    auto srcLineHead = srcHead + width * y;
                    auto dstLineHead = dstHead + width * y;
                    for (int x = 0; x < width; ++x) {
                        const Gray8 &src = *(srcLineHead + x);
                        Gray8 &dst = *(dstLineHead + x);
                        dst.v = std::min<uint32_t>(255, 256 * sRGB_degamma_s(src.v / 255.0f));
                    }
                }
            }
            else {
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
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

                switch (getDataFormat()) {
                case VLRDataFormat_RGBA16Fx4: {
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
                default:
                    VLRAssert_ShouldNotBeCalled();
                    break;
                }
            }
        }

        return new LinearImage2D(m_context, data.data(), width, height, getDataFormat(), false);
    }

    Image2D* LinearImage2D::createLuminanceImage2D() const {
        uint32_t width = getWidth();
        uint32_t height = getHeight();
        uint32_t stride;
        VLRDataFormat newDataFormat;
        switch (getDataFormat()) {
        case VLRDataFormat_RGBA16Fx4: {
            stride = sizeof(float);
            newDataFormat = VLRDataFormat_Gray32F;
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
                case VLRDataFormat_RGBA16Fx4: {
                    RGBA16Fx4 pix = get<RGBA16Fx4>(x, y);
                    float Y = RGBSpectrum(pix.r, pix.g, pix.b).luminance(RGBColorSpace::sRGB);
                    *(float*)&data[(y * width + x) * stride] = Y;
                    break;
                }
                default:
                    VLRAssert_ShouldNotBeCalled();
                    break;
                }
            }
        }

        return new LinearImage2D(m_context, data.data(), width, height, newDataFormat, false);
    }

    void* LinearImage2D::createLinearImageData() const {
        uint8_t* ret = new uint8_t[m_data.size()];
        std::copy(m_data.cbegin(), m_data.cend(), ret);
        return ret;
    }

    optix::Buffer LinearImage2D::getOptiXObject() const {
        optix::Buffer buffer = Image2D::getOptiXObject();
        if (!m_copyDone) {
            auto dstData = (uint8_t*)buffer->map();
            std::copy(m_data.cbegin(), m_data.cend(), dstData);
            buffer->unmap();
            m_copyDone = true;
        }
        return buffer;
    }



    // static
    void TextureMap2D::commonInitializeProcedure(Context &context, const char* identifiers[1], OptiXProgramSet* programSet) {
        std::string ptx = readTxtFile("resources/ptxes/materials.ptx");

        optix::Context optixContext = context.getOptiXContext();

        programSet->callableProgramMap = optixContext->createProgramFromPTXString(ptx, identifiers[0]);
    }

    // static
    void TextureMap2D::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        programSet.callableProgramMap->destroy();
    }

    // static
    void TextureMap2D::initialize(Context &context) {
        OffsetAndScaleUVTextureMap2D::initialize(context);
    }

    // static
    void TextureMap2D::finalize(Context &context) {
        OffsetAndScaleUVTextureMap2D::finalize(context);
    }

    TextureMap2D::TextureMap2D(Context &context) : Object(context) {
        m_texMapIndex = 0xFFFFFFFF;
    }

    TextureMap2D::~TextureMap2D() {
        if (m_texMapIndex != 0xFFFFFFFF)
            m_context.unsetTextureMapDescriptor(m_texMapIndex);
        m_texMapIndex = 0xFFFFFFFF;
    }



    std::map<uint32_t, TextureMap2D::OptiXProgramSet> OffsetAndScaleUVTextureMap2D::OptiXProgramSets;
    std::map<uint32_t, OffsetAndScaleUVTextureMap2D*> OffsetAndScaleUVTextureMap2D::s_defaultInstance;

    // static
    void OffsetAndScaleUVTextureMap2D::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::OffsetAndScaleUVTextureMap2D_map",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;

        float offset[] = { 0.0f, 0.0f };
        float scale[] = { 1.0f, 1.0f };
        s_defaultInstance[context.getID()] = new OffsetAndScaleUVTextureMap2D(context, offset, scale);
    }

    // static
    void OffsetAndScaleUVTextureMap2D::finalize(Context &context) {
        delete s_defaultInstance.at(context.getID());
        s_defaultInstance.erase(context.getID());

        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    OffsetAndScaleUVTextureMap2D::OffsetAndScaleUVTextureMap2D(Context &context, const float offset[2], const float scale[2]) :
        TextureMap2D(context), m_offset{ offset[0], offset[1] }, m_scale{ scale[0], scale[1] } {
        Shared::TextureMapDescriptor texMatDesc;
        setupTextureMapDescriptor(&texMatDesc);

        m_texMapIndex = m_context.setTextureMapDescriptor(texMatDesc);
    }

    OffsetAndScaleUVTextureMap2D::~OffsetAndScaleUVTextureMap2D() {
    }

    void OffsetAndScaleUVTextureMap2D::setupTextureMapDescriptor(Shared::TextureMapDescriptor* texMapDesc) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        texMapDesc->progTextureMap = progSet.callableProgramMap->getId();
        Shared::OffsetAndScaleUVTextureMap2D &texMap = *(Shared::OffsetAndScaleUVTextureMap2D*)&texMapDesc->data;
        texMap.offset[0] = m_offset[0];
        texMap.offset[1] = m_offset[1];
        texMap.scale[0] = m_scale[0];
        texMap.scale[1] = m_scale[1];
    }



    FloatTexture::FloatTexture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    FloatTexture::~FloatTexture() {
        m_optixTextureSampler->destroy();
    }

    void FloatTexture::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    ConstantFloatTexture::ConstantFloatTexture(Context &context, const float value) :
        FloatTexture(context) {
        m_image = new LinearImage2D(context, (const uint8_t*)&value, 1, 1, VLRDataFormat_Gray32F, false);
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    ConstantFloatTexture::~ConstantFloatTexture() {
        delete m_image;
    }



    Float2Texture::Float2Texture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    Float2Texture::~Float2Texture() {
        m_optixTextureSampler->destroy();
    }

    void Float2Texture::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    ConstantFloat2Texture::ConstantFloat2Texture(Context &context, const float value[2]) :
        Float2Texture(context) {
        float value2[] = {value[0], value[1]};
        m_image = new LinearImage2D(context, (const uint8_t*)value2, 1, 1, VLRDataFormat_RG32Fx2, false);
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    ConstantFloat2Texture::~ConstantFloat2Texture() {
        delete m_image;
    }



    Float3Texture::Float3Texture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    Float3Texture::~Float3Texture() {
        m_optixTextureSampler->destroy();
    }

    void Float3Texture::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    ConstantFloat3Texture::ConstantFloat3Texture(Context &context, const float value[3]) :
        Float3Texture(context) {
        float value4[] = { value[0], value[1], value[2], 0 };
        m_image = new LinearImage2D(context, (const uint8_t*)value4, 1, 1, VLRDataFormat_RGBA32Fx4, false);
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    ConstantFloat3Texture::~ConstantFloat3Texture() {
        delete m_image;
    }



    ImageFloat3Texture::ImageFloat3Texture(Context &context, const Image2D* image) :
        Float3Texture(context), m_image(image) {
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    void ImageFloat3Texture::createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const {
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



    Float4Texture::Float4Texture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    Float4Texture::~Float4Texture() {
        m_optixTextureSampler->destroy();
    }

    void Float4Texture::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    ConstantFloat4Texture::ConstantFloat4Texture(Context &context, const float value[4]) :
        Float4Texture(context) {
        m_image = new LinearImage2D(context, (const uint8_t*)value, 1, 1, VLRDataFormat_RGBA32Fx4, false);
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    ConstantFloat4Texture::~ConstantFloat4Texture() {
        delete m_image;
    }



    ImageFloat4Texture::ImageFloat4Texture(Context &context, const Image2D* image) :
        Float4Texture(context), m_image(image) {
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    // END: Textures
    // ----------------------------------------------------------------
}
