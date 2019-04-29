#include "shader_nodes.h"

namespace VLR {
    Shared::ShaderNodeSocketID ShaderNodeSocketIdentifier::getSharedType() const {
        if (node) {
            Shared::ShaderNodeSocketID ret;
            ret.nodeType = node->getProcedureSetIndex();
            ret.socketType = socketInfo.outputType;
            ret.nodeDescIndex = node->getShaderNodeIndex();
            ret.option = socketInfo.option;
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
        Float3ToSpectrumShaderNode::initialize(context);
        ScaleAndOffsetUVTextureMap2DShaderNode::initialize(context);
        Image2DTextureShaderNode::initialize(context);
        EnvironmentTextureShaderNode::initialize(context);
    }

    // static
    void ShaderNode::finalize(Context &context) {
        EnvironmentTextureShaderNode::finalize(context);
        Image2DTextureShaderNode::finalize(context);
        ScaleAndOffsetUVTextureMap2DShaderNode::finalize(context);
        Float3ToSpectrumShaderNode::finalize(context);
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

    ShaderNode::ShaderNode(Context &context, bool isSpectrumNode) : Object(context), m_isLargeNode(isSpectrumNode) {
        if (m_isLargeNode)
            m_nodeIndex = m_context.allocateLargeNodeDescriptor();
        else
            m_nodeIndex = m_context.allocateNodeDescriptor();
    }

    ShaderNode::~ShaderNode() {
        if (m_nodeIndex != 0xFFFFFFFF)
            if (m_isLargeNode)
                m_context.releaseLargeNodeDescriptor(m_nodeIndex);
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
        auto &nodeData = *nodeDesc.getData<Shared::FloatShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.imm0 = m_imm0;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool FloatShaderNode::setNode0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        auto &nodeData = *nodeDesc.getData<Shared::Float2ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool Float2ShaderNode::setNode0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        auto &nodeData = *nodeDesc.getData<Shared::ScaleAndOffsetFloatShaderNode>();
        nodeData.nodeValue = m_nodeValue.getSharedType();
        nodeData.nodeScale = m_nodeScale.getSharedType();
        nodeData.nodeOffset = m_nodeOffset.getSharedType();
        nodeData.immScale = m_immScale;
        nodeData.immOffset = m_immOffset;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool ScaleAndOffsetFloatShaderNode::setNodeValue(const ShaderNodeSocketIdentifier &outputSocket) {
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
            return false;
        m_nodeValue = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::setNodeScale(const ShaderNodeSocketIdentifier &outputSocket) {
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
            return false;
        m_nodeScale = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::setNodeOffset(const ShaderNodeSocketIdentifier &outputSocket) {
        if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFor(outputSocket.getType()))
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
        ShaderNode(context, false), m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(ColorSpace::Rec709_D65),
        m_immE0(0.18f), m_immE1(0.18f), m_immE2(0.18f) {
        setupNodeDescriptor();
    }

    TripletSpectrumShaderNode::~TripletSpectrumShaderNode() {
    }

    void TripletSpectrumShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        auto &nodeData = *nodeDesc.getData<Shared::TripletSpectrumShaderNode>();
        nodeData.value = createTripletSpectrum(m_spectrumType, m_colorSpace, m_immE0, m_immE1, m_immE2);

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
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
#if defined(VLR_USE_SPECTRAL_RENDERING)
        ShaderNode(context, true),
#else
        ShaderNode(context, false),
#endif
        m_spectrumType(VLRSpectrumType_NA), m_minLambda(0.0f), m_maxLambda(1000.0f), m_values(nullptr), m_numSamples(2) {
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

#if defined(VLR_USE_SPECTRAL_RENDERING)
        Shared::LargeNodeDescriptor nodeDesc;
        auto &nodeData = *nodeDesc.getData<Shared::RegularSampledSpectrumShaderNode>();

        VLRAssert(m_numSamples <= lengthof(nodeData.values), "Number of sample points must not be greater than %u.", lengthof(nodeData.values));
        nodeData.minLambda = m_minLambda;
        nodeData.maxLambda = m_maxLambda;
        std::copy_n(m_values, m_numSamples, nodeData.values);
        nodeData.numSamples = m_numSamples;

        m_context.updateLargeNodeDescriptor(m_nodeIndex, nodeDesc);
#else
        Shared::NodeDescriptor nodeDesc;
        auto &nodeData = *nodeDesc.getData<Shared::RegularSampledSpectrumShaderNode>();

        RegularSampledSpectrum spectrum(m_minLambda, m_maxLambda, m_values, m_numSamples);
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        float RGB[3];
        transformToRenderingRGB(m_spectrumType, XYZ, RGB);
        nodeData.value = RGBSpectrum(std::fmax(0.0f, RGB[0]), std::fmax(0.0f, RGB[1]), std::fmax(0.0f, RGB[2]));

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
#endif
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
#if defined(VLR_USE_SPECTRAL_RENDERING)
        ShaderNode(context, true),
#else
        ShaderNode(context, false),
#endif
        m_spectrumType(VLRSpectrumType_NA), m_lambdas(nullptr), m_values(nullptr), m_numSamples(2) {
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

#if defined(VLR_USE_SPECTRAL_RENDERING)
        Shared::LargeNodeDescriptor nodeDesc;
        auto &nodeData = *nodeDesc.getData<Shared::IrregularSampledSpectrumShaderNode>();

        VLRAssert(m_numSamples <= lengthof(nodeData.values), "Number of sample points must not be greater than %u.", lengthof(nodeData.values));
        std::copy_n(m_lambdas, m_numSamples, nodeData.lambdas);
        std::copy_n(m_values, m_numSamples, nodeData.values);
        nodeData.numSamples = m_numSamples;

        m_context.updateLargeNodeDescriptor(m_nodeIndex, nodeDesc);
#else
        Shared::NodeDescriptor nodeDesc;
        auto &nodeData = *nodeDesc.getData<Shared::IrregularSampledSpectrumShaderNode>();

        IrregularSampledSpectrum spectrum(m_lambdas, m_values, m_numSamples);
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        float RGB[3];
        transformToRenderingRGB(m_spectrumType, XYZ, RGB);
        nodeData.value = RGBSpectrum(std::fmax(0.0f, RGB[0]), std::fmax(0.0f, RGB[1]), std::fmax(0.0f, RGB[2]));

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
#endif
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



    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float3ToSpectrumShaderNode::OptiXProgramSets;

    // static
    void Float3ToSpectrumShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::Float3ToSpectrumShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float3ToSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    Float3ToSpectrumShaderNode::Float3ToSpectrumShaderNode(Context &context) :
        ShaderNode(context), m_immFloat3{ 0, 0, 0 }, m_spectrumType(VLRSpectrumType_Reflectance), m_colorSpace(ColorSpace::Rec709_D65) {
        setupNodeDescriptor();
    }

    Float3ToSpectrumShaderNode::~Float3ToSpectrumShaderNode() {
    }

    void Float3ToSpectrumShaderNode::setupNodeDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::NodeDescriptor nodeDesc;
        auto &nodeData = *nodeDesc.getData<Shared::Float3ToSpectrumShaderNode>();
        nodeData.nodeFloat3 = m_nodeFloat3.getSharedType();
        nodeData.immFloat3[0] = m_immFloat3[0];
        nodeData.immFloat3[1] = m_immFloat3[1];
        nodeData.immFloat3[2] = m_immFloat3[2];
        nodeData.spectrumType = m_spectrumType;
        nodeData.colorSpace = m_colorSpace;

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    bool Float3ToSpectrumShaderNode::setNodeFloat3(const ShaderNodeSocketIdentifier &outputSocket) {
        if (!Shared::NodeTypeInfo<optix::float3>::ConversionIsDefinedFor(outputSocket.getType()))
            return false;
        m_nodeFloat3 = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void Float3ToSpectrumShaderNode::setImmediateValueFloat3(const float value[3]) {
        m_immFloat3[0] = value[0];
        m_immFloat3[1] = value[1];
        m_immFloat3[2] = value[2];
        setupNodeDescriptor();
    }

    void Float3ToSpectrumShaderNode::setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, ColorSpace colorSpace) {
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
        ShaderNode(context), m_image(NullImages.at(m_context.getID())) {
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
        auto &nodeData = *nodeDesc.getData<Shared::Image2DTextureShaderNode>();
        nodeData.textureID = m_optixTextureSampler->getId();
        nodeData.dataFormat = (unsigned int)m_image->getDataFormat();
        nodeData.spectrumType = m_image->getSpectrumType();
        // JP: GPUカーネル内でHWによってsRGBデガンマされて読まれる場合には、デガンマ済みとして捉える必要がある。
        // EN: Data should be regarded as post-degamma in the case that reading with sRGB degamma by HW in a GPU kernel.
        ColorSpace colorSpace = m_image->getColorSpace();
        if (m_image->needsHW_sRGB_degamma() && colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
            colorSpace = ColorSpace::Rec709_D65;
        nodeData.colorSpace = (unsigned int)colorSpace;
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void Image2DTextureShaderNode::setImage(const Image2D* image) {
        m_image = image ? image : NullImages.at(m_context.getID());
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
        m_optixTextureSampler->setReadMode(m_image->needsHW_sRGB_degamma() ? RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB : RT_TEXTURE_READ_NORMALIZED_FLOAT);
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
        if (!Shared::NodeTypeInfo<Point3D>::ConversionIsDefinedFor(outputSocket.getType()))
            return false;
        m_nodeTexCoord = outputSocket;
        setupNodeDescriptor();
        return true;
    }



    std::map<uint32_t, ShaderNode::OptiXProgramSet> EnvironmentTextureShaderNode::OptiXProgramSets;
    std::map<uint32_t, LinearImage2D*> EnvironmentTextureShaderNode::NullImages;

    // static
    void EnvironmentTextureShaderNode::initialize(Context &context) {
        const SocketTypeToProgramPair pairs[] = {
            VLRShaderNodeSocketType_Spectrum, "VLR::EnvironmentTextureShaderNode_Spectrum",
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        half nullData[] = { (half)1.0f, (half)0.0f, (half)1.0f, (half)1.0f };
        LinearImage2D* nullImage = new LinearImage2D(context, (uint8_t*)nullData, 1, 1, DataFormat::RGBA16Fx4, VLRSpectrumType_LightSource, ColorSpace::Rec709_D65);

        OptiXProgramSets[context.getID()] = programSet;
        NullImages[context.getID()] = nullImage;
    }

    // static
    void EnvironmentTextureShaderNode::finalize(Context &context) {
        delete NullImages.at(context.getID());

        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    EnvironmentTextureShaderNode::EnvironmentTextureShaderNode(Context &context) :
        ShaderNode(context), m_image(NullImages.at(m_context.getID())) {
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
        auto &nodeData = *nodeDesc.getData<Shared::EnvironmentTextureShaderNode>();
        nodeData.textureID = m_optixTextureSampler->getId();
        nodeData.dataFormat = (unsigned int)m_image->getDataFormat();
        nodeData.colorSpace = (unsigned int)m_image->getColorSpace();
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        m_context.updateNodeDescriptor(m_nodeIndex, nodeDesc);
    }

    void EnvironmentTextureShaderNode::setImage(const Image2D* image) {
        m_image = image ? image : NullImages.at(m_context.getID());
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
        if (!Shared::NodeTypeInfo<Point3D>::ConversionIsDefinedFor(outputSocket.getType()))
            return false;
        m_nodeTexCoord = outputSocket;
        setupNodeDescriptor();
        return true;
    }

    void EnvironmentTextureShaderNode::createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const {
        uint32_t mapWidth = std::max<uint32_t>(1, m_image->getWidth() / 4);
        uint32_t mapHeight = std::max<uint32_t>(1, m_image->getHeight() / 4);
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
