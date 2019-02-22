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
    sizeof(GrayA8x2),
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
        case VLRDataFormat_GrayA8x2:
            return VLRDataFormat_GrayA8x2;
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }
        return VLRDataFormat_RGBA8x4;
    }

    Image2D::Image2D(Context &context, uint32_t width, uint32_t height, VLRDataFormat originalDataFormat) :
        Object(context), m_width(width), m_height(height), m_originalDataFormat(originalDataFormat), m_initOptiXObject(false) {
        m_dataFormat = getInternalFormat(m_originalDataFormat);
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
        case VLRDataFormat_GrayA8x2:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE2, m_width, m_height);
            break;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }

        m_initOptiXObject = true;

        return m_optixDataBuffer;
    }



    template <typename SrcType, typename DstType>
    using FuncProcessPixel = void (*)(const SrcType &src, DstType &dst);
    
    template <typename SrcType, typename DstType>
    void processAllPixels(const uint8_t* srcData, uint8_t* dstData, uint32_t width, uint32_t height, FuncProcessPixel<SrcType, DstType> func) {
        auto srcHead = (const SrcType*)srcData;
        auto dstHead = (DstType*)dstData;
        for (int y = 0; y < height; ++y) {
            auto srcLineHead = srcHead + width * y;
            auto dstLineHead = dstHead + width * y;
            for (int x = 0; x < width; ++x) {
                auto &src = *(srcLineHead + x);
                auto &dst = *(dstLineHead + x);
                func(src, dst);
            }
        }
    }
    
    LinearImage2D::LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat dataFormat, bool applyDegamma) :
        Image2D(context, width, height, Image2D::getInternalFormat(dataFormat)), m_copyDone(false) {
        m_data.resize(getStride() * getWidth() * getHeight());

        switch (dataFormat) {
        case VLRDataFormat_RGB8x3: {
            FuncProcessPixel<RGB8x3, RGBA8x4> funcApplyDegamma = [](const RGB8x3 &src, RGBA8x4 &dst) {
                dst.r = std::min<uint32_t>(255, 256 * sRGB_degamma(src.r / 255.0f));
                dst.g = std::min<uint32_t>(255, 256 * sRGB_degamma(src.g / 255.0f));
                dst.b = std::min<uint32_t>(255, 256 * sRGB_degamma(src.b / 255.0f));
                dst.a = 255;
            };
            FuncProcessPixel<RGB8x3, RGBA8x4> funcAsIs = [](const RGB8x3 &src, RGBA8x4 &dst) {
                dst.r = src.r;
                dst.g = src.g;
                dst.b = src.b;
                dst.a = 255;
            };
            processAllPixels(linearData, m_data.data(), width, height, applyDegamma ? funcApplyDegamma : funcAsIs);
            break;
        }
        case VLRDataFormat_RGB_8x4: {
            FuncProcessPixel<RGB_8x4, RGBA8x4> funcApplyDegamma = [](const RGB_8x4 &src, RGBA8x4 &dst) {
                dst.r = std::min<uint32_t>(255, 256 * sRGB_degamma(src.r / 255.0f));
                dst.g = std::min<uint32_t>(255, 256 * sRGB_degamma(src.g / 255.0f));
                dst.b = std::min<uint32_t>(255, 256 * sRGB_degamma(src.b / 255.0f));
                dst.a = 255;
            };
            FuncProcessPixel<RGB_8x4, RGBA8x4> funcAsIs = [](const RGB_8x4 &src, RGBA8x4 &dst) {
                dst.r = src.r;
                dst.g = src.g;
                dst.b = src.b;
                dst.a = 255;
            };
            processAllPixels(linearData, m_data.data(), width, height, applyDegamma ? funcApplyDegamma : funcAsIs);
            break;
        }
        case VLRDataFormat_RGBA8x4: {
            if (applyDegamma) {
                FuncProcessPixel<RGBA8x4, RGBA8x4> funcApplyDegamma = [](const RGBA8x4 &src, RGBA8x4 &dst) {
                    dst.r = std::min<uint32_t>(255, 256 * sRGB_degamma(src.r / 255.0f));
                    dst.g = std::min<uint32_t>(255, 256 * sRGB_degamma(src.g / 255.0f));
                    dst.b = std::min<uint32_t>(255, 256 * sRGB_degamma(src.b / 255.0f));
                    dst.a = src.a;
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const RGBA8x4*)linearData;
                auto dstHead = (RGBA8x4*)m_data.data();
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_RGBA16Fx4: {
            if (applyDegamma) {
                FuncProcessPixel<RGBA16Fx4, RGBA16Fx4> funcApplyDegamma = [](const RGBA16Fx4 &src, RGBA16Fx4 &dst) {
                    dst.r = (half)sRGB_degamma((float)src.r);
                    dst.g = (half)sRGB_degamma((float)src.g);
                    dst.b = (half)sRGB_degamma((float)src.b);
                    dst.a = src.a;
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const RGBA16Fx4*)linearData;
                auto dstHead = (RGBA16Fx4*)m_data.data();
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_RGBA32Fx4: {
            if (applyDegamma) {
                FuncProcessPixel<RGBA32Fx4, RGBA32Fx4> funcApplyDegamma = [](const RGBA32Fx4 &src, RGBA32Fx4 &dst) {
                    dst.r = sRGB_degamma(src.r);
                    dst.g = sRGB_degamma(src.g);
                    dst.b = sRGB_degamma(src.b);
                    dst.a = src.a;
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const RGBA32Fx4*)linearData;
                auto dstHead = (RGBA32Fx4*)m_data.data();
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_RG32Fx2: {
            if (applyDegamma) {
                FuncProcessPixel<RG32Fx2, RG32Fx2> funcApplyDegamma = [](const RG32Fx2 &src, RG32Fx2 &dst) {
                    dst.r = sRGB_degamma(src.r);
                    dst.g = sRGB_degamma(src.g);
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const RG32Fx2*)linearData;
                auto dstHead = (RG32Fx2*)m_data.data();
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_Gray32F: {
            if (applyDegamma) {
                FuncProcessPixel<Gray32F, Gray32F> funcApplyDegamma = [](const Gray32F &src, Gray32F &dst) {
                    dst.v = sRGB_degamma(src.v);
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const Gray32F*)linearData;
                auto dstHead = (Gray32F*)m_data.data();
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_Gray8: {
            if (applyDegamma) {
                FuncProcessPixel<Gray8, Gray8> funcApplyDegamma = [](const Gray8 &src, Gray8 &dst) {
                    dst.v = std::min<uint32_t>(255, 256 * sRGB_degamma(src.v / 255.0f));
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const Gray8*)linearData;
                auto dstHead = (Gray8*)m_data.data();
                std::copy_n(srcHead, width * height, dstHead);
            }
            break;
        }
        case VLRDataFormat_GrayA8x2: {
            if (applyDegamma) {
                FuncProcessPixel<GrayA8x2, GrayA8x2> funcApplyDegamma = [](const GrayA8x2 &src, GrayA8x2 &dst) {
                    dst.v = std::min<uint32_t>(255, 256 * sRGB_degamma(src.v / 255.0f));
                    dst.a = src.a;
                };
                processAllPixels(linearData, m_data.data(), width, height, funcApplyDegamma);
            }
            else {
                auto srcHead = (const GrayA8x2*)linearData;
                auto dstHead = (GrayA8x2*)m_data.data();
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
                    float Y = mat_Rec709_D65_to_XYZ[1] * pix.r + mat_Rec709_D65_to_XYZ[4] * pix.g + mat_Rec709_D65_to_XYZ[7] * pix.b;
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
            ret.isSpectrumNode = node->isSpectrumNode();
            return ret;
        }
        return Shared::ShaderNodeSocketID::Invalid();
    }



    // static 
    void ShaderNode::commonInitializeProcedure(Context &context, const char** identifiers, uint32_t numIDs, OptiXProgramSet* programSet) {
        std::string ptx = readTxtFile(VLR_PTX_DIR"shader_nodes.ptx");

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
        const char* identifiers[] = {
            "VLR::GeometryShaderNode_Point3D",
            "VLR::GeometryShaderNode_Normal3D",
            "VLR::GeometryShaderNode_Vector3D",
            "VLR::GeometryShaderNode_textureCoordinates",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        const char* identifiers[] = {
            "VLR::ScaleAndOffsetFloatShaderNode_float",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        const char* identifiers[] = {
            "VLR::TripletSpectrumShaderNode_spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TripletSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    TripletSpectrumShaderNode::TripletSpectrumShaderNode(Context &context) :
        ShaderNode(context, true), m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(VLRColorSpace_Rec709_D65),
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

    void TripletSpectrumShaderNode::setImmediateValueColorSpace(VLRColorSpace colorSpace) {
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
        const char* identifiers[] = {
            "VLR::RegularSampledSpectrumShaderNode_spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        const char* identifiers[] = {
            "VLR::IrregularSampledSpectrumShaderNode_spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        const char* identifiers[] = {
            "VLR::Vector3DToSpectrumShaderNode_spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Vector3DToSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Vector3DToSpectrumShaderNode::Vector3DToSpectrumShaderNode(Context &context) :
        ShaderNode(context), m_immVector3D(Vector3D(0, 0, 0)), m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(VLRColorSpace_Rec709_D65) {
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

    void Vector3DToSpectrumShaderNode::setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
        m_spectrumType = spectrumType;
        m_colorSpace = colorSpace;
        setupNodeDescriptor();
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> ScaleAndOffsetUVTextureMap2DShaderNode::OptiXProgramSets;

    // static
    void ScaleAndOffsetUVTextureMap2DShaderNode::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::ScaleAndOffsetUVTextureMap2DShaderNode_textureCoordinates",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

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
        const char* identifiers[] = {
            "VLR::Image2DTextureShaderNode_spectrum",
            "VLR::Image2DTextureShaderNode_float",
            "VLR::Image2DTextureShaderNode_float2",
            "VLR::Image2DTextureShaderNode_float3",
            "VLR::Image2DTextureShaderNode_float4",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, lengthof(identifiers), &programSet);

        uint8_t nullData[] = { 255, 0, 255, 255 };
        LinearImage2D* nullImage = new LinearImage2D(context, nullData, 1, 1, VLRDataFormat_RGBA8x4, false);

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
        ShaderNode(context), m_image(nullptr), m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(VLRColorSpace_Rec709_D65) {
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
        nodeData.format = m_image ? m_image->getDataFormat() : (VLRDataFormat)0;
        nodeData.spectrumType = m_spectrumType;
        nodeData.colorSpace = m_colorSpace;
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void Image2DTextureShaderNode::setImage(VLRSpectrumType spectrumType, VLRColorSpace colorSpace, const Image2D* image) {
        m_spectrumType = spectrumType;
        m_colorSpace = colorSpace;
        m_image = image;
        if (m_image)
            m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
        else
            m_optixTextureSampler->setBuffer(NullImages.at(m_context.getID())->getOptiXObject());
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
        const char* identifiers[] = {
            "VLR::EnvironmentTextureShaderNode_spectrum",
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
        ShaderNode(context), m_image(nullptr), m_colorSpace(VLRColorSpace_Rec709_D65) {
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
        nodeData.colorSpace = m_colorSpace;
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void EnvironmentTextureShaderNode::setImage(VLRColorSpace colorSpace, const Image2D* image) {
        m_colorSpace = colorSpace;
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
