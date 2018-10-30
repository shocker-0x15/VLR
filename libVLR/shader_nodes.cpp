#include "shader_nodes.h"

namespace VLR {
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



    Shared::ShaderNodeSocketID ShaderNodeSocketIdentifier::getSharedType() const {
        if (node && socketInfo.type != VLRShaderNodeSocketType_Invalid) {
            Shared::ShaderNodeSocketID ret;
            ret.nodeDescIndex = node->getShaderNodeIndex();
            ret.socketIndex = socketInfo.outputIndex;
            ret.option = socketInfo.option;
            return ret;
        }
        return Shared::ShaderNodeSocketID::Invalid();
    }



    // static 
    void ShaderNode::commonInitializeProcedure(Context &context, const char** identifiers, uint32_t numIDs, OptiXProgramSet* programSet) {
        std::string ptx = readTxtFile("resources/ptxes/shader_nodes.ptx");

        optix::Context optixContext = context.getOptiXContext();

        Shared::NodeProcedureSet nodeProcSet;
        for (int i = 0; i < numIDs; ++i) {
            programSet->callablePrograms[i] = optixContext->createProgramFromPTXString(ptx, identifiers[i]);
            nodeProcSet.progs[i] = programSet->callablePrograms[i]->getId();
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
        FloatShaderNode::initialize(context);
        Float2ShaderNode::initialize(context);
        Float3ShaderNode::initialize(context);
        Float4ShaderNode::initialize(context);
        OffsetAndScaleUVTextureMap2DShaderNode::initialize(context);
        ConstantTextureShaderNode::initialize(context);
        Image2DTextureShaderNode::initialize(context);
        EnvironmentTextureShaderNode::initialize(context);
    }

    // static
    void ShaderNode::finalize(Context &context) {
        EnvironmentTextureShaderNode::finalize(context);
        Image2DTextureShaderNode::finalize(context);
        ConstantTextureShaderNode::finalize(context);
        OffsetAndScaleUVTextureMap2DShaderNode::finalize(context);
        Float4ShaderNode::finalize(context);
        Float3ShaderNode::finalize(context);
        Float2ShaderNode::finalize(context);
        FloatShaderNode::finalize(context);
    }

    ShaderNode::ShaderNode(Context &context) : Object(context) {
        m_nodeIndex = m_context.allocateNodeDescriptor();
    }

    ShaderNode::~ShaderNode() {
        if (m_nodeIndex != 0xFFFFFFFF)
            m_context.releaseNodeDescriptor(m_nodeIndex);
        m_nodeIndex = 0xFFFFFFFF;
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> FloatShaderNode::OptiXProgramSets;

    // static
    void FloatShaderNode::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::FloatShaderNode_float",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        Shared::FloatShaderNode &nodeData = *(Shared::FloatShaderNode*)&nodeDesc.data;
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
        const char* identifiers[] = {
            "VLR::Float2ShaderNode_float",
            "VLR::Float2ShaderNode_float2",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        Shared::Float2ShaderNode &nodeData = *(Shared::Float2ShaderNode*)&nodeDesc.data;
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
        const char* identifiers[] = {
            "VLR::Float3ShaderNode_float",
            "VLR::Float3ShaderNode_float2",
            "VLR::Float3ShaderNode_float3",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        Shared::Float3ShaderNode &nodeData = *(Shared::Float3ShaderNode*)&nodeDesc.data;
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
        const char* identifiers[] = {
            "VLR::Float4ShaderNode_float",
            "VLR::Float4ShaderNode_float2",
            "VLR::Float4ShaderNode_float3",
            "VLR::Float4ShaderNode_float4",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        Shared::Float4ShaderNode &nodeData = *(Shared::Float4ShaderNode*)&nodeDesc.data;
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



    std::map<uint32_t, ShaderNode::OptiXProgramSet> OffsetAndScaleUVTextureMap2DShaderNode::OptiXProgramSets;

    // static
    void OffsetAndScaleUVTextureMap2DShaderNode::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::OffsetAndScaleUVTextureMap2DShaderNode_TexCoord",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void OffsetAndScaleUVTextureMap2DShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    OffsetAndScaleUVTextureMap2DShaderNode::OffsetAndScaleUVTextureMap2DShaderNode(Context &context) :
        ShaderNode(context), m_offset{ 0.0f, 0.0f }, m_scale{ 1.0f, 1.0f } {
        setupNodeDescriptor();
    }

    OffsetAndScaleUVTextureMap2DShaderNode::~OffsetAndScaleUVTextureMap2DShaderNode() {
    }

    void OffsetAndScaleUVTextureMap2DShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        Shared::OffsetAndScaleUVTextureMap2DShaderNode &nodeData = *(Shared::OffsetAndScaleUVTextureMap2DShaderNode*)&nodeDesc.data;
        nodeData.offset[0] = m_offset[0];
        nodeData.offset[1] = m_offset[1];
        nodeData.scale[0] = m_scale[0];
        nodeData.scale[1] = m_scale[1];

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void OffsetAndScaleUVTextureMap2DShaderNode::setValues(const float offset[2], const float scale[2]) {
        std::copy_n(offset, 2, m_offset);
        std::copy_n(scale, 2, m_scale);
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> ConstantTextureShaderNode::OptiXProgramSets;

    // static
    void ConstantTextureShaderNode::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::ConstantTextureShaderNode_RGBSpectrum",
            "VLR::ConstantTextureShaderNode_Alpha",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void ConstantTextureShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    ConstantTextureShaderNode::ConstantTextureShaderNode(Context &context) :
        ShaderNode(context), m_spectrum(RGBSpectrum(0.18f)), m_alpha(1.0f) {
        setupNodeDescriptor();
    }

    ConstantTextureShaderNode::~ConstantTextureShaderNode() {
    }

    void ConstantTextureShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        Shared::ConstantTextureShaderNode &nodeData = *(Shared::ConstantTextureShaderNode*)&nodeDesc.data;
        nodeData.spectrum = m_spectrum;
        nodeData.alpha = m_alpha;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void ConstantTextureShaderNode::setValues(const RGBSpectrum &spectrum, float alpha) {
        m_spectrum = spectrum;
        m_alpha = alpha;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Image2DTextureShaderNode::OptiXProgramSets;

    // static
    void Image2DTextureShaderNode::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::Image2DTextureShaderNode_RGBSpectrum",
            "VLR::Image2DTextureShaderNode_float",
            "VLR::Image2DTextureShaderNode_float2",
            "VLR::Image2DTextureShaderNode_float3",
            "VLR::Image2DTextureShaderNode_float4",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Image2DTextureShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Image2DTextureShaderNode::Image2DTextureShaderNode(Context &context) :
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

    Image2DTextureShaderNode::~Image2DTextureShaderNode() {
        m_optixTextureSampler->destroy();
    }

    void Image2DTextureShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        nodeDesc.procSetIndex = progSet.nodeProcedureSetIndex;
        Shared::Image2DTextureShaderNode &nodeData = *(Shared::Image2DTextureShaderNode*)&nodeDesc.data;
        nodeData.textureID = m_optixTextureSampler->getId();
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void Image2DTextureShaderNode::setImage(const Image2D* image) {
        m_image = image;
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
        setupNodeDescriptor();
    }

    void Image2DTextureShaderNode::setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
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
        const char* identifiers[] = {
            "VLR::EnvironmentTextureShaderNode_RGBSpectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        Shared::EnvironmentTextureShaderNode &nodeData = *(Shared::EnvironmentTextureShaderNode*)&nodeDesc.data;
        nodeData.textureID = m_optixTextureSampler->getId();
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
