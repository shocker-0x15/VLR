#include "shader_nodes.h"

namespace VLR {
    Shared::ShaderNodePlug ShaderNodePlug::getSharedType() const {
        if (node) {
            Shared::ShaderNodePlug ret;
            ret.nodeType = node->getProcedureSetIndex();
            ret.plugType = info.outputType;
            ret.nodeDescIndex = node->getShaderNodeIndex();
            ret.option = info.option;
            return ret;
        }
        return Shared::ShaderNodePlug::Invalid();
    }



    optixu::Module ShaderNode::s_shaderNodeModule;
    
    // static 
    void ShaderNode::commonInitializeProcedure(Context &context, const PlugTypeToProgramPair* pairs, uint32_t numPairs, OptiXProgramSet* programSet) {
        optixu::Pipeline pipeline = context.getOptixPipeline();

        Shared::NodeProcedureSet nodeProcSet;
        for (int i = 0; i < lengthof(nodeProcSet.progs); ++i)
            nodeProcSet.progs[i] = 0xFFFFFFFF;
        for (int i = 0; i < numPairs; ++i) {
            uint32_t ptype = static_cast<uint32_t>(pairs[i].ptype);
            programSet->callablePrograms[ptype].create(
                pipeline,
                s_shaderNodeModule, pairs[i].programName,
                context.getEmptyModule(), nullptr);
            nodeProcSet.progs[ptype] = programSet->callablePrograms[ptype].ID;
        }

        programSet->nodeProcedureSetIndex = context.allocateNodeProcedureSet();
        context.updateNodeProcedureSet(programSet->nodeProcedureSetIndex, nodeProcSet);
    }

    // static 
    void ShaderNode::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        context.releaseNodeProcedureSet(programSet.nodeProcedureSetIndex);

        for (int i = lengthof(programSet.callablePrograms) - 1; i >= 0; --i) {
            if (programSet.callablePrograms[i])
                programSet.callablePrograms[i].destroy();
        }
    }

    void ShaderNode::updateNodeDescriptor() const {
        if (m_nodeSizeClass == 0)
            m_context.updateSmallNodeDescriptor(m_nodeIndex, smallNodeDesc);
        else if (m_nodeSizeClass == 1)
            m_context.updateMediumNodeDescriptor(m_nodeIndex, mediumNodeDesc);
        else if (m_nodeSizeClass == 2)
            m_context.updateLargeNodeDescriptor(m_nodeIndex, largeNodeDesc);
        else
            VLRAssert_ShouldNotBeCalled();
    }

    // static
    void ShaderNode::initialize(Context &context) {
        optixu::Pipeline pipeline = context.getOptixPipeline();
        s_shaderNodeModule = pipeline.createModuleFromPTXString(
            readTxtFile(getExecutableDirectory() / "ptxes/shader_nodes.ptx"),
            OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        GeometryShaderNode::initialize(context);
        TangentShaderNode::initialize(context);
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
        TangentShaderNode::finalize(context);
        GeometryShaderNode::finalize(context);

        s_shaderNodeModule.destroy();
    }

    ShaderNode::ShaderNode(Context &context, size_t sizeOfNode) : Queryable(context) {
        size_t sizeOfNodeInDW = sizeOfNode / 4;
        if (sizeOfNodeInDW <= Shared::SmallNodeDescriptor::NumDWSlots()) {
            m_nodeSizeClass = 0;
            m_nodeIndex = m_context.allocateSmallNodeDescriptor();
        }
        else if (sizeOfNodeInDW <= Shared::MediumNodeDescriptor::NumDWSlots()) {
            m_nodeSizeClass = 1;
            m_nodeIndex = m_context.allocateMediumNodeDescriptor();
        }
        else if (sizeOfNodeInDW <= Shared::LargeNodeDescriptor::NumDWSlots()) {
            m_nodeSizeClass = 2;
            m_nodeIndex = m_context.allocateLargeNodeDescriptor();
        }
    }

    ShaderNode::~ShaderNode() {
        if (m_nodeIndex != 0xFFFFFFFF) {
            if (m_nodeSizeClass == 0)
                m_context.releaseSmallNodeDescriptor(m_nodeIndex);
            else if (m_nodeSizeClass == 1)
                m_context.releaseMediumNodeDescriptor(m_nodeIndex);
            else if (m_nodeSizeClass == 2)
                m_context.releaseLargeNodeDescriptor(m_nodeIndex);
        }
        m_nodeIndex = 0xFFFFFFFF;
    }



    std::vector<ParameterInfo> GeometryShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> GeometryShaderNode::OptiXProgramSets;
    std::map<uint32_t, GeometryShaderNode*> GeometryShaderNode::Instances;

    // static
    void GeometryShaderNode::initialize(Context &context) {
        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Point3D, RT_DC_NAME_STR("GeometryShaderNode_Point3D"),
            ShaderNodePlugType::Normal3D, RT_DC_NAME_STR("GeometryShaderNode_Normal3D"),
            ShaderNodePlugType::Vector3D, RT_DC_NAME_STR("GeometryShaderNode_Vector3D"),
            ShaderNodePlugType::TextureCoordinates, RT_DC_NAME_STR("GeometryShaderNode_TextureCoordinates"),
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
        OptiXProgramSets.erase(context.getID());
    }

    GeometryShaderNode::GeometryShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::GeometryShaderNode)) {
        setupNodeDescriptor();
    }

    GeometryShaderNode::~GeometryShaderNode() {
    }

    void GeometryShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::GeometryShaderNode>();

        updateNodeDescriptor();
    }

    GeometryShaderNode* GeometryShaderNode::getInstance(Context &context) {
        return Instances.at(context.getID());
    }



    std::vector<ParameterInfo> TangentShaderNode::ParameterInfos;

    std::map<uint32_t, ShaderNode::OptiXProgramSet> TangentShaderNode::OptiXProgramSets;

    // static
    void TangentShaderNode::initialize(Context& context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("tangent type", VLRParameterFormFlag_ImmediateValue, EnumTangentType),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Vector3D, RT_DC_NAME_STR("TangentShaderNode_Vector3D"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TangentShaderNode::finalize(Context& context) {
        OptiXProgramSet& programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    TangentShaderNode::TangentShaderNode(Context& context) :
        ShaderNode(context, sizeof(Shared::TangentShaderNode)), m_immTangentType(TangentType::TC0Direction) {
        setupNodeDescriptor();
    }

    TangentShaderNode::~TangentShaderNode() {
    }

    void TangentShaderNode::setupNodeDescriptor() const {
        auto& nodeData = *getData<Shared::TangentShaderNode>();
        nodeData.tangentType = m_immTangentType;

        updateNodeDescriptor();
    }

    bool TangentShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "tangent type")) {
            *enumValue = getEnumMemberFromValue(m_immTangentType);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool TangentShaderNode::set(const char* paramName, const char* enumValue) {
        if (testParamName(paramName, "tangent type")) {
            auto v = getEnumValueFromMember<TangentType>(enumValue);
            if (v == (TangentType)0xFFFFFFFF)
                return false;

            m_immTangentType = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> Float2ShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float2ShaderNode::OptiXProgramSets;

    // static
    void Float2ShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("0", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("1", VLRParameterFormFlag_Both, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::float1, RT_DC_NAME_STR("Float2ShaderNode_float1"),
            ShaderNodePlugType::float2, RT_DC_NAME_STR("Float2ShaderNode_float2"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float2ShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    Float2ShaderNode::Float2ShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::Float2ShaderNode)),
        m_imm0(0.0f), m_imm1(0.0f) {
        setupNodeDescriptor();
    }

    Float2ShaderNode::~Float2ShaderNode() {
    }

    void Float2ShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::Float2ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;

        updateNodeDescriptor();
    }

    bool Float2ShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            if (length != 1)
                return false;

            values[0] = m_imm0;
        }
        else if (testParamName(paramName, "1")) {
            if (length != 1)
                return false;

            values[0] = m_imm1;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float2ShaderNode::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            *plug = m_node0;
        }
        else if (testParamName(paramName, "1")) {
            *plug = m_node1;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float2ShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            if (length != 1)
                return false;

            m_imm0 = values[0];
        }
        else if (testParamName(paramName, "1")) {
            if (length != 1)
                return false;

            m_imm1 = values[0];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Float2ShaderNode::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "0")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node0 = plug;
        }
        else if (testParamName(paramName, "1")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node1 = plug;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> Float3ShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float3ShaderNode::OptiXProgramSets;

    // static
    void Float3ShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("0", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("1", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("2", VLRParameterFormFlag_Both, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::float1, RT_DC_NAME_STR("Float3ShaderNode_float1"),
            ShaderNodePlugType::float2, RT_DC_NAME_STR("Float3ShaderNode_float2"),
            ShaderNodePlugType::float3, RT_DC_NAME_STR("Float3ShaderNode_float3"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float3ShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    Float3ShaderNode::Float3ShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::Float3ShaderNode)),
        m_imm0(0.0f), m_imm1(0.0f), m_imm2(0.0f) {
        setupNodeDescriptor();
    }

    Float3ShaderNode::~Float3ShaderNode() {
    }

    void Float3ShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::Float3ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.node2 = m_node2.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;
        nodeData.imm2 = m_imm2;

        updateNodeDescriptor();
    }

    bool Float3ShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            if (length != 1)
                return false;

            values[0] = m_imm0;
        }
        else if (testParamName(paramName, "1")) {
            if (length != 1)
                return false;

            values[0] = m_imm1;
        }
        else if (testParamName(paramName, "2")) {
            if (length != 1)
                return false;

            values[0] = m_imm2;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float3ShaderNode::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            *plug = m_node0;
        }
        else if (testParamName(paramName, "1")) {
            *plug = m_node1;
        }
        else if (testParamName(paramName, "2")) {
            *plug = m_node2;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float3ShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            if (length != 1)
                return false;

            m_imm0 = values[0];
        }
        else if (testParamName(paramName, "1")) {
            if (length != 1)
                return false;

            m_imm1 = values[0];
        }
        else if (testParamName(paramName, "2")) {
            if (length != 1)
                return false;

            m_imm2 = values[0];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Float3ShaderNode::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "0")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node0 = plug;
        }
        else if (testParamName(paramName, "1")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node1 = plug;
        }
        else if (testParamName(paramName, "2")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node2 = plug;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> Float4ShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float4ShaderNode::OptiXProgramSets;

    // static
    void Float4ShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("0", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("1", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("2", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("3", VLRParameterFormFlag_Both, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::float1, RT_DC_NAME_STR("Float4ShaderNode_float1"),
            ShaderNodePlugType::float2, RT_DC_NAME_STR("Float4ShaderNode_float2"),
            ShaderNodePlugType::float3, RT_DC_NAME_STR("Float4ShaderNode_float3"),
            ShaderNodePlugType::float4, RT_DC_NAME_STR("Float4ShaderNode_float4"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float4ShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    Float4ShaderNode::Float4ShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::Float4ShaderNode)),
        m_imm0(0.0f), m_imm1(0.0f), m_imm2(0.0f), m_imm3(0.0f) {
        setupNodeDescriptor();
    }

    Float4ShaderNode::~Float4ShaderNode() {
    }

    void Float4ShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::Float4ShaderNode>();
        nodeData.node0 = m_node0.getSharedType();
        nodeData.node1 = m_node1.getSharedType();
        nodeData.node2 = m_node2.getSharedType();
        nodeData.node3 = m_node3.getSharedType();
        nodeData.imm0 = m_imm0;
        nodeData.imm1 = m_imm1;
        nodeData.imm2 = m_imm2;
        nodeData.imm3 = m_imm3;

        updateNodeDescriptor();
    }

    bool Float4ShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            if (length != 1)
                return false;

            values[0] = m_imm0;
        }
        else if (testParamName(paramName, "1")) {
            if (length != 1)
                return false;

            values[0] = m_imm1;
        }
        else if (testParamName(paramName, "2")) {
            if (length != 1)
                return false;

            values[0] = m_imm2;
        }
        else if (testParamName(paramName, "3")) {
            if (length != 1)
                return false;

            values[0] = m_imm3;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float4ShaderNode::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            *plug = m_node0;
        }
        else if (testParamName(paramName, "1")) {
            *plug = m_node1;
        }
        else if (testParamName(paramName, "2")) {
            *plug = m_node2;
        }
        else if (testParamName(paramName, "3")) {
            *plug = m_node3;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float4ShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "0")) {
            if (length != 1)
                return false;

            m_imm0 = values[0];
        }
        else if (testParamName(paramName, "1")) {
            if (length != 1)
                return false;

            m_imm1 = values[0];
        }
        else if (testParamName(paramName, "2")) {
            if (length != 1)
                return false;

            m_imm2 = values[0];
        }
        else if (testParamName(paramName, "3")) {
            if (length != 1)
                return false;

            m_imm3 = values[0];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Float4ShaderNode::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "0")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node0 = plug;
        }
        else if (testParamName(paramName, "1")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node1 = plug;
        }
        else if (testParamName(paramName, "2")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node2 = plug;
        }
        else if (testParamName(paramName, "3")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node3 = plug;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> ScaleAndOffsetFloatShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> ScaleAndOffsetFloatShaderNode::OptiXProgramSets;

    // static
    void ScaleAndOffsetFloatShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("value", VLRParameterFormFlag_Node, ParameterFloat),
            ParameterInfo("scale", VLRParameterFormFlag_Both, ParameterFloat),
            ParameterInfo("offset", VLRParameterFormFlag_Both, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::float1, RT_DC_NAME_STR("ScaleAndOffsetFloatShaderNode_float1"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void ScaleAndOffsetFloatShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    ScaleAndOffsetFloatShaderNode::ScaleAndOffsetFloatShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::ScaleAndOffsetFloatShaderNode)), m_immScale(1.0f), m_immOffset(0.0f) {
        setupNodeDescriptor();
    }

    ScaleAndOffsetFloatShaderNode::~ScaleAndOffsetFloatShaderNode() {
    }

    void ScaleAndOffsetFloatShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::ScaleAndOffsetFloatShaderNode>();
        nodeData.nodeValue = m_nodeValue.getSharedType();
        nodeData.nodeScale = m_nodeScale.getSharedType();
        nodeData.nodeOffset = m_nodeOffset.getSharedType();
        nodeData.immScale = m_immScale;
        nodeData.immOffset = m_immOffset;

        updateNodeDescriptor();
    }

    bool ScaleAndOffsetFloatShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "scale")) {
            if (length != 1)
                return false;

            values[0] = m_immScale;
        }
        else if (testParamName(paramName, "offset")) {
            if (length != 1)
                return false;

            values[0] = m_immOffset;
        }
        else {
            return false;
        }

        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "value")) {
            *plug = m_nodeValue;
        }
        else if (testParamName(paramName, "scale")) {
            *plug = m_nodeScale;
        }
        else if (testParamName(paramName, "offset")) {
            *plug = m_nodeOffset;
        }
        else {
            return false;
        }

        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "scale")) {
            if (length != 1)
                return false;

            m_immScale = values[0];
        }
        else if (testParamName(paramName, "offset")) {
            if (length != 1)
                return false;

            m_immOffset = values[0];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool ScaleAndOffsetFloatShaderNode::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "value")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeValue = plug;
        }
        else if (testParamName(paramName, "scale")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeScale = plug;
        }
        else if (testParamName(paramName, "offset")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeOffset = plug;
        }
        else {
            return false;
        }

        return true;
    }



    std::vector<ParameterInfo> TripletSpectrumShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> TripletSpectrumShaderNode::OptiXProgramSets;

    // static
    void TripletSpectrumShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("spectrum type", VLRParameterFormFlag_ImmediateValue, EnumSpectrumType),
            ParameterInfo("color space", VLRParameterFormFlag_ImmediateValue, EnumColorSpace),
            ParameterInfo("triplet", VLRParameterFormFlag_ImmediateValue, ParameterFloat, 3),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Spectrum, RT_DC_NAME_STR("TripletSpectrumShaderNode_Spectrum"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TripletSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    TripletSpectrumShaderNode::TripletSpectrumShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::TripletSpectrumShaderNode)), m_spectrumType(SpectrumType::Reflectance), m_colorSpace(ColorSpace::Rec709_D65),
        m_immE0(0.18f), m_immE1(0.18f), m_immE2(0.18f) {
        setupNodeDescriptor();
    }

    TripletSpectrumShaderNode::~TripletSpectrumShaderNode() {
    }

    void TripletSpectrumShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::TripletSpectrumShaderNode>();
        nodeData.value = createTripletSpectrum(m_spectrumType, m_colorSpace, m_immE0, m_immE1, m_immE2);

        updateNodeDescriptor();
    }

    bool TripletSpectrumShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "spectrum type")) {
            *enumValue = getEnumMemberFromValue(m_spectrumType);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else if (testParamName(paramName, "color space")) {
            *enumValue = getEnumMemberFromValue(m_colorSpace);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool TripletSpectrumShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "triplet")) {
            if (length != 3)
                return false;

            values[0] = m_immE0;
            values[1] = m_immE1;
            values[2] = m_immE2;
        }
        else {
            return false;
        }

        return true;
    }

    bool TripletSpectrumShaderNode::set(const char* paramName, const char* enumValue) {
        if (testParamName(paramName, "spectrum type")) {
            auto v = getEnumValueFromMember<SpectrumType>(enumValue);
            if (v == (SpectrumType)0xFFFFFFFF)
                return false;

            m_spectrumType = v;
        }
        else if (testParamName(paramName, "color space")) {
            auto v = getEnumValueFromMember<ColorSpace>(enumValue);
            if (v == (ColorSpace)0xFFFFFFFF)
                return false;

            m_colorSpace = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool TripletSpectrumShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "triplet")) {
            if (length != 3)
                return false;

            m_immE0 = values[0];
            m_immE1 = values[1];
            m_immE2 = values[2];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> RegularSampledSpectrumShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> RegularSampledSpectrumShaderNode::OptiXProgramSets;

    // static
    void RegularSampledSpectrumShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("spectrum type", VLRParameterFormFlag_ImmediateValue, EnumSpectrumType),
            ParameterInfo("min wavelength", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("max wavelength", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("values", VLRParameterFormFlag_ImmediateValue, ParameterFloat, 0),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Spectrum, RT_DC_NAME_STR("RegularSampledSpectrumShaderNode_Spectrum"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void RegularSampledSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    RegularSampledSpectrumShaderNode::RegularSampledSpectrumShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::RegularSampledSpectrumShaderNode)),
        m_spectrumType(SpectrumType::NA), m_minLambda(0.0f), m_maxLambda(1000.0f), m_values(nullptr), m_numSamples(2) {
        m_values = new float[2];
        m_values[0] = m_values[1] = 1.0f;
        setupNodeDescriptor();
    }

    RegularSampledSpectrumShaderNode::~RegularSampledSpectrumShaderNode() {
        if (m_values)
            delete[] m_values;
    }

    void RegularSampledSpectrumShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::RegularSampledSpectrumShaderNode>();
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
        updateNodeDescriptor();
    }

    bool RegularSampledSpectrumShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "spectrum type")) {
            *enumValue = getEnumMemberFromValue(m_spectrumType);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool RegularSampledSpectrumShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return true;

        if (testParamName(paramName, "min wavelength")) {
            if (length != 1)
                return false;

            values[0] = m_minLambda;
        }
        else if (testParamName(paramName, "max wavelength")) {
            if (length != 1)
                return false;

            values[0] = m_maxLambda;
        }
        else {
            return false;
        }

        return true;
    }

    bool RegularSampledSpectrumShaderNode::get(const char* paramName, const float** values, uint32_t* length) const {
        if (values == nullptr || length == nullptr)
            return true;

        if (testParamName(paramName, "values")) {
            *values = m_values;
            *length = m_numSamples;
        }
        else {
            return false;
        }

        return true;
    }

    bool RegularSampledSpectrumShaderNode::set(const char* paramName, const char* enumValue) {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "spectrum type")) {
            auto v = getEnumValueFromMember<SpectrumType>(enumValue);
            if (v == (SpectrumType)0xFFFFFFFF)
                return false;

            m_spectrumType = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool RegularSampledSpectrumShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "min wavelength")) {
            if (length != 1)
                return false;

            m_minLambda = values[0];
        }
        else if (testParamName(paramName, "max wavelength")) {
            if (length != 1)
                return false;

            m_maxLambda = values[0];
        }
        else if (testParamName(paramName, "values")) {
            if (m_values)
                delete[] m_values;

            m_numSamples = length;
            m_values = new float[m_numSamples];
            std::copy_n(values, m_numSamples, m_values);
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> IrregularSampledSpectrumShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> IrregularSampledSpectrumShaderNode::OptiXProgramSets;

    // static
    void IrregularSampledSpectrumShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("spectrum type", VLRParameterFormFlag_ImmediateValue, EnumSpectrumType),
            ParameterInfo("wavelengths", VLRParameterFormFlag_ImmediateValue, ParameterFloat, 0),
            ParameterInfo("values", VLRParameterFormFlag_ImmediateValue, ParameterFloat, 0),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Spectrum, RT_DC_NAME_STR("IrregularSampledSpectrumShaderNode_Spectrum"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void IrregularSampledSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    IrregularSampledSpectrumShaderNode::IrregularSampledSpectrumShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::IrregularSampledSpectrumShaderNode)),
        m_spectrumType(SpectrumType::NA), m_lambdas(nullptr), m_values(nullptr), m_numSamples(2) {
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
        auto &nodeData = *getData<Shared::IrregularSampledSpectrumShaderNode>();
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
        updateNodeDescriptor();
    }

    bool IrregularSampledSpectrumShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "spectrum type")) {
            *enumValue = getEnumMemberFromValue(m_spectrumType);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool IrregularSampledSpectrumShaderNode::get(const char* paramName, const float** values, uint32_t* length) const {
        if (values == nullptr || length == nullptr)
            return true;

        if (testParamName(paramName, "wavelengths")) {
            *values = m_lambdas;
            *length = m_numSamples;
        }
        else if (testParamName(paramName, "values")) {
            *values = m_values;
            *length = m_numSamples;
        }
        else {
            return false;
        }

        return true;
    }

    bool IrregularSampledSpectrumShaderNode::set(const char* paramName, const char* enumValue) {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "spectrum type")) {
            auto v = getEnumValueFromMember<SpectrumType>(enumValue);
            if (v == (SpectrumType)0xFFFFFFFF)
                return false;

            m_spectrumType = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool IrregularSampledSpectrumShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "wavelengths")) {
            if (m_lambdas)
                delete[] m_lambdas;

            m_numSamples = length;
            m_lambdas = new float[m_numSamples];
            std::copy_n(values, m_numSamples, m_lambdas);
        }
        else if (testParamName(paramName, "values")) {
            if (m_values)
                delete[] m_values;

            m_numSamples = length;
            m_values = new float[m_numSamples];
            std::copy_n(values, m_numSamples, m_values);
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> Float3ToSpectrumShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> Float3ToSpectrumShaderNode::OptiXProgramSets;

    // static
    void Float3ToSpectrumShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("spectrum type", VLRParameterFormFlag_ImmediateValue, EnumSpectrumType),
            ParameterInfo("color space", VLRParameterFormFlag_ImmediateValue, EnumColorSpace),
            ParameterInfo("value", VLRParameterFormFlag_Both, ParameterFloat, 3),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Spectrum, RT_DC_NAME_STR("Float3ToSpectrumShaderNode_Spectrum"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void Float3ToSpectrumShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    Float3ToSpectrumShaderNode::Float3ToSpectrumShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::Float3ToSpectrumShaderNode)), m_immFloat3{ 0, 0, 0 }, m_spectrumType(SpectrumType::Reflectance), m_colorSpace(ColorSpace::Rec709_D65) {
        setupNodeDescriptor();
    }

    Float3ToSpectrumShaderNode::~Float3ToSpectrumShaderNode() {
    }

    void Float3ToSpectrumShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::Float3ToSpectrumShaderNode>();
        nodeData.nodeFloat3 = m_nodeFloat3.getSharedType();
        nodeData.immFloat3[0] = m_immFloat3[0];
        nodeData.immFloat3[1] = m_immFloat3[1];
        nodeData.immFloat3[2] = m_immFloat3[2];
        nodeData.spectrumType = m_spectrumType;
        nodeData.colorSpace = m_colorSpace;

        updateNodeDescriptor();
    }

    bool Float3ToSpectrumShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "spectrum type")) {
            *enumValue = getEnumMemberFromValue(m_spectrumType);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else if (testParamName(paramName, "color space")) {
            *enumValue = getEnumMemberFromValue(m_colorSpace);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool Float3ToSpectrumShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "value")) {
            if (length != 3)
                return false;

            values[0] = m_immFloat3[0];
            values[1] = m_immFloat3[1];
            values[2] = m_immFloat3[2];
        }
        else {
            return false;
        }

        return true;
    }

    bool Float3ToSpectrumShaderNode::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "value")) {
            *plug = m_nodeFloat3;
        }
        else {
            return false;
        }

        return true;
    }

    bool Float3ToSpectrumShaderNode::set(const char* paramName, const char* enumValue) {
        if (testParamName(paramName, "spectrum type")) {
            auto v = getEnumValueFromMember<SpectrumType>(enumValue);
            if (v == (SpectrumType)0xFFFFFFFF)
                return false;

            m_spectrumType = v;
        }
        else if (testParamName(paramName, "color space")) {
            auto v = getEnumValueFromMember<ColorSpace>(enumValue);
            if (v == (ColorSpace)0xFFFFFFFF)
                return false;

            m_colorSpace = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Float3ToSpectrumShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "value")) {
            if (length != 3)
                return false;

            m_immFloat3[0] = values[0];
            m_immFloat3[1] = values[1];
            m_immFloat3[2] = values[2];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Float3ToSpectrumShaderNode::set(const char* paramName, const ShaderNodePlug &plug) {
        if (testParamName(paramName, "value")) {
            if (!Shared::NodeTypeInfo<float3>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeFloat3 = plug;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> ScaleAndOffsetUVTextureMap2DShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> ScaleAndOffsetUVTextureMap2DShaderNode::OptiXProgramSets;

    // static
    void ScaleAndOffsetUVTextureMap2DShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("scale", VLRParameterFormFlag_ImmediateValue, ParameterFloat, 2),
            ParameterInfo("offset", VLRParameterFormFlag_ImmediateValue, ParameterFloat, 2),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::TextureCoordinates, RT_DC_NAME_STR("ScaleAndOffsetUVTextureMap2DShaderNode_TextureCoordinates"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void ScaleAndOffsetUVTextureMap2DShaderNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    ScaleAndOffsetUVTextureMap2DShaderNode::ScaleAndOffsetUVTextureMap2DShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::ScaleAndOffsetUVTextureMap2DShaderNode)), m_offset{ 0.0f, 0.0f }, m_scale{ 1.0f, 1.0f } {
        setupNodeDescriptor();
    }

    ScaleAndOffsetUVTextureMap2DShaderNode::~ScaleAndOffsetUVTextureMap2DShaderNode() {
    }

    void ScaleAndOffsetUVTextureMap2DShaderNode::setupNodeDescriptor() const {
        auto &nodeData = *getData<Shared::ScaleAndOffsetUVTextureMap2DShaderNode>();
        nodeData.offset[0] = m_offset[0];
        nodeData.offset[1] = m_offset[1];
        nodeData.scale[0] = m_scale[0];
        nodeData.scale[1] = m_scale[1];

        updateNodeDescriptor();
    }

    bool ScaleAndOffsetUVTextureMap2DShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "scale")) {
            if (length != 2)
                return false;

            values[0] = m_scale[0];
            values[1] = m_scale[1];
        }
        else if (testParamName(paramName, "offset")) {
            if (length != 2)
                return false;

            values[0] = m_offset[0];
            values[1] = m_offset[1];
        }
        else {
            return false;
        }

        return true;
    }

    bool ScaleAndOffsetUVTextureMap2DShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "scale")) {
            if (length != 2)
                return false;

            m_scale[0] = values[0];
            m_scale[1] = values[1];
        }
        else if (testParamName(paramName, "offset")) {
            if (length != 2)
                return false;

            m_offset[0] = values[0];
            m_offset[1] = values[1];
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> Image2DTextureShaderNode::ParameterInfos;
    
    std::map<uint32_t, ShaderNode::OptiXProgramSet> Image2DTextureShaderNode::OptiXProgramSets;
    std::map<uint32_t, LinearImage2D*> Image2DTextureShaderNode::NullImages;

    // static
    void Image2DTextureShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("image", VLRParameterFormFlag_Node, ParameterImage),
            ParameterInfo("bump type", VLRParameterFormFlag_ImmediateValue, EnumBumpType),
            ParameterInfo("bump coeff", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("min filter", VLRParameterFormFlag_ImmediateValue, EnumTextureFilter),
            ParameterInfo("mag filter", VLRParameterFormFlag_ImmediateValue, EnumTextureFilter),
            ParameterInfo("wrap u", VLRParameterFormFlag_ImmediateValue, EnumTextureWrapMode),
            ParameterInfo("wrap v", VLRParameterFormFlag_ImmediateValue, EnumTextureWrapMode),
            ParameterInfo("texcoord", VLRParameterFormFlag_Node, ParameterTextureCoordinates),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::float1, RT_DC_NAME_STR("Image2DTextureShaderNode_float1"),
            ShaderNodePlugType::float2, RT_DC_NAME_STR("Image2DTextureShaderNode_float2"),
            ShaderNodePlugType::float3, RT_DC_NAME_STR("Image2DTextureShaderNode_float3"),
            ShaderNodePlugType::float4, RT_DC_NAME_STR("Image2DTextureShaderNode_float4"),
            ShaderNodePlugType::Normal3D, RT_DC_NAME_STR("Image2DTextureShaderNode_Normal3D"),
            ShaderNodePlugType::Spectrum, RT_DC_NAME_STR("Image2DTextureShaderNode_Spectrum"),
            ShaderNodePlugType::Alpha, RT_DC_NAME_STR("Image2DTextureShaderNode_Alpha"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        uint8_t nullData[] = { 255, 0, 255, 255 };
        LinearImage2D* nullImage = new LinearImage2D(context, nullData, 1, 1, DataFormat::RGBA8x4, SpectrumType::Reflectance, ColorSpace::Rec709_D65);

        OptiXProgramSets[context.getID()] = programSet;
        NullImages[context.getID()] = nullImage;
    }

    // static
    void Image2DTextureShaderNode::finalize(Context &context) {
        delete NullImages.at(context.getID());

        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    Image2DTextureShaderNode::Image2DTextureShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::Image2DTextureShaderNode)), m_image(NullImages.at(m_context.getID())),
        m_bumpType(BumpType::NormalMap_DirectX), m_bumpCoeff(1.0f),
        m_xyFilter(TextureFilter::Linear),
        m_wrapU(TextureWrapMode::Repeat), m_wrapV(TextureWrapMode::Repeat) {
        m_textureSampler.setXyFilterMode(static_cast<cudau::TextureFilterMode>(m_xyFilter));
        m_textureSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
        m_textureSampler.setWrapMode(0, static_cast<cudau::TextureWrapMode>(m_wrapU));
        m_textureSampler.setWrapMode(1, static_cast<cudau::TextureWrapMode>(m_wrapV));
        m_textureSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        m_textureSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);

        setupNodeDescriptor();
    }

    Image2DTextureShaderNode::~Image2DTextureShaderNode() {
        auto &nodeData = *getData<Shared::Image2DTextureShaderNode>();
        if (nodeData.texture)
            cuTexObjectDestroy(nodeData.texture);
    }

    void Image2DTextureShaderNode::setupNodeDescriptor() {
        m_textureSampler.setXyFilterMode(static_cast<cudau::TextureFilterMode>(m_xyFilter));
        m_textureSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
        m_textureSampler.setWrapMode(0, static_cast<cudau::TextureWrapMode>(m_wrapU));
        m_textureSampler.setWrapMode(1, static_cast<cudau::TextureWrapMode>(m_wrapV));
        m_textureSampler.setReadMode(m_image->needsHW_sRGB_degamma() ?
                                     cudau::TextureReadMode::NormalizedFloat_sRGB :
                                     cudau::TextureReadMode::NormalizedFloat);

        auto &nodeData = *getData<Shared::Image2DTextureShaderNode>();
        if (nodeData.texture)
            CUDADRV_CHECK(cuTexObjectDestroy(nodeData.texture));
        nodeData.texture = m_textureSampler.createTextureObject(m_image->getOptiXObject());
        nodeData.dataFormat = static_cast<unsigned int>(m_image->getDataFormat());
        nodeData.spectrumType = static_cast<unsigned int>(m_image->getSpectrumType());
        // JP: GPUカーネル内でHWによってsRGBデガンマされて読まれる場合には、デガンマ済みとして捉える必要がある。
        // EN: Data should be regarded as post-degamma in the case that reading with sRGB degamma by HW in a GPU kernel.
        ColorSpace colorSpace = m_image->getColorSpace();
        if (m_image->needsHW_sRGB_degamma() && colorSpace == ColorSpace::Rec709_D65_sRGBGamma)
            colorSpace = ColorSpace::Rec709_D65;
        nodeData.colorSpace = static_cast<unsigned int>(colorSpace);
        nodeData.bumpType = static_cast<unsigned int>(m_bumpType);
        const float minCoeff = 1.0f / (1 << (VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH - 1));
        float coeff = std::round(m_bumpCoeff * (1 << (VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH - 1))) - 1;
        nodeData.bumpCoeff = VLR::clamp<int32_t>(coeff, 0, (1 << VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH) - 1);
        nodeData.nodeTexCoord = m_nodeTexCoord.getSharedType();

        updateNodeDescriptor();
    }

    bool Image2DTextureShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "bump type")) {
            *enumValue = getEnumMemberFromValue(m_bumpType);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else if (testParamName(paramName, "filter")) {
            *enumValue = getEnumMemberFromValue(m_xyFilter);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else if (testParamName(paramName, "wrap u")) {
            *enumValue = getEnumMemberFromValue(m_wrapU);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else if (testParamName(paramName, "wrap v")) {
            *enumValue = getEnumMemberFromValue(m_wrapV);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool Image2DTextureShaderNode::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "bump coeff")) {
            if (length != 1)
                return false;

            values[0] = m_bumpCoeff;
        }
        else {
            return false;
        }

        return true;
    }

    bool Image2DTextureShaderNode::get(const char* paramName, const Image2D** image) const {
        if (image == nullptr)
            return false;

        if (testParamName(paramName, "image")) {
            *image = m_image != NullImages.at(m_context.getID()) ? m_image : nullptr;
        }
        else {
            return false;
        }

        return true;
    }

    bool Image2DTextureShaderNode::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "texcoord")) {
            *plug = m_nodeTexCoord;
        }
        else {
            return false;
        }

        return true;
    }

    bool Image2DTextureShaderNode::set(const char* paramName, const char* enumValue) {
        if (testParamName(paramName, "bump type")) {
            auto v = getEnumValueFromMember<BumpType>(enumValue);
            if (v == (BumpType)0xFFFFFFFF)
                return false;

            m_bumpType = v;
        }
        else if (testParamName(paramName, "filter")) {
            auto v = getEnumValueFromMember<TextureFilter>(enumValue);
            if (v == (TextureFilter)0xFFFFFFFF)
                return false;

            m_xyFilter = v;
        }
        else if (testParamName(paramName, "wrap u")) {
            auto v = getEnumValueFromMember<TextureWrapMode>(enumValue);
            if (v == (TextureWrapMode)0xFFFFFFFF)
                return false;

            m_wrapU = v;
        }
        else if (testParamName(paramName, "wrap v")) {
            auto v = getEnumValueFromMember<TextureWrapMode>(enumValue);
            if (v == (TextureWrapMode)0xFFFFFFFF)
                return false;

            m_wrapV = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Image2DTextureShaderNode::set(const char* paramName, const float* values, uint32_t length) {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "bump coeff")) {
            if (length != 1)
                return false;

            const float minCoeff = 1.0f / (1 << (VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH - 1));
            m_bumpCoeff = VLR::clamp(values[0], minCoeff, 2.0f);
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Image2DTextureShaderNode::set(const char* paramName, const Image2D* image) {
        if (testParamName(paramName, "image")) {
            m_image = image ? image : NullImages.at(m_context.getID());
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool Image2DTextureShaderNode::set(const char* paramName, const ShaderNodePlug &plug) {
        if (testParamName(paramName, "texcoord")) {
            if (!Shared::NodeTypeInfo<Point3D>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeTexCoord = plug;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }



    std::vector<ParameterInfo> EnvironmentTextureShaderNode::ParameterInfos;

    std::map<uint32_t, ShaderNode::OptiXProgramSet> EnvironmentTextureShaderNode::OptiXProgramSets;
    std::map<uint32_t, LinearImage2D*> EnvironmentTextureShaderNode::NullImages;

    // static
    void EnvironmentTextureShaderNode::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("image", VLRParameterFormFlag_Node, ParameterImage),
            ParameterInfo("min filter", VLRParameterFormFlag_ImmediateValue, EnumTextureFilter),
            ParameterInfo("mag filter", VLRParameterFormFlag_ImmediateValue, EnumTextureFilter),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const PlugTypeToProgramPair pairs[] = {
            ShaderNodePlugType::Spectrum, RT_DC_NAME_STR("EnvironmentTextureShaderNode_Spectrum"),
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, pairs, lengthof(pairs), &programSet);

        half nullData[] = { (half)1.0f, (half)0.0f, (half)1.0f, (half)1.0f };
        LinearImage2D* nullImage = new LinearImage2D(context, (uint8_t*)nullData, 1, 1, DataFormat::RGBA16Fx4, SpectrumType::LightSource, ColorSpace::Rec709_D65);

        OptiXProgramSets[context.getID()] = programSet;
        NullImages[context.getID()] = nullImage;
    }

    // static
    void EnvironmentTextureShaderNode::finalize(Context &context) {
        delete NullImages.at(context.getID());

        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    EnvironmentTextureShaderNode::EnvironmentTextureShaderNode(Context &context) :
        ShaderNode(context, sizeof(Shared::EnvironmentTextureShaderNode)), m_image(NullImages.at(m_context.getID())),
        m_xyFilter(TextureFilter::Linear) {
        m_textureSampler.setXyFilterMode(static_cast<cudau::TextureFilterMode>(m_xyFilter));
        m_textureSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
        m_textureSampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
        m_textureSampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
        m_textureSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        m_textureSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);

        setupNodeDescriptor();
    }

    EnvironmentTextureShaderNode::~EnvironmentTextureShaderNode() {
        auto &nodeData = *getData<Shared::EnvironmentTextureShaderNode>();
        if (nodeData.texture)
            cuTexObjectDestroy(nodeData.texture);
    }

    void EnvironmentTextureShaderNode::setupNodeDescriptor() {
        m_textureSampler.setXyFilterMode(static_cast<cudau::TextureFilterMode>(m_xyFilter));
        m_textureSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);

        auto &nodeData = *getData<Shared::EnvironmentTextureShaderNode>();
        if (nodeData.texture)
            CUDADRV_CHECK(cuTexObjectDestroy(nodeData.texture));
        nodeData.texture = m_textureSampler.createTextureObject(m_image->getOptiXObject());
        nodeData.dataFormat = static_cast<unsigned int>(m_image->getDataFormat());
        nodeData.colorSpace = static_cast<unsigned int>(m_image->getColorSpace());

        updateNodeDescriptor();
    }

    bool EnvironmentTextureShaderNode::get(const char* paramName, const char** enumValue) const {
        if (enumValue == nullptr)
            return false;

        if (testParamName(paramName, "filter")) {
            *enumValue = getEnumMemberFromValue(m_xyFilter);
            VLRAssert(*enumValue != nullptr, "Invalid enum value");
        }
        else {
            return false;
        }

        return true;
    }

    bool EnvironmentTextureShaderNode::get(const char* paramName, const Image2D** image) const {
        if (image == nullptr)
            return false;

        if (testParamName(paramName, "image")) {
            *image = m_image != NullImages.at(m_context.getID()) ? m_image : nullptr;
        }
        else {
            return false;
        }

        return true;
    }

    bool EnvironmentTextureShaderNode::set(const char* paramName, const char* enumValue) {
        if (testParamName(paramName, "filter")) {
            auto v = getEnumValueFromMember<TextureFilter>(enumValue);
            if (v == (TextureFilter)0xFFFFFFFF)
                return false;

            m_xyFilter = v;
        }
        else {
            return false;
        }
        setupNodeDescriptor();

        return true;
    }

    bool EnvironmentTextureShaderNode::set(const char* paramName, const Image2D* image) {
        if (testParamName(paramName, "image")) {
            m_image = image ? image : NullImages.at(m_context.getID());
        }
        else {
            return false;
        }
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
