#include "materials.h"

namespace VLR {
    optixu::Module SurfaceMaterial::s_materialModule;

    // static
    void SurfaceMaterial::commonInitializeProcedure(Context &context, const char* identifiers[10], OptiXProgramSet* programSet) {
        optixu::Pipeline pipeline = context.getOptixPipeline();

        if (identifiers[0] && identifiers[1] && identifiers[2] && identifiers[3] && identifiers[4] && identifiers[5] && identifiers[6]) {
            programSet->callableProgramSetupBSDF.create(
                pipeline, s_materialModule, identifiers[0],
                context.getEmptyModule(), nullptr);

            programSet->callableProgramBSDFGetBaseColor.create(
                pipeline, s_materialModule, identifiers[1],
                context.getEmptyModule(), nullptr);
            programSet->callableProgramBSDFmatches.create(
                pipeline, s_materialModule, identifiers[2],
                context.getEmptyModule(), nullptr);
            programSet->callableProgramBSDFSampleInternal.create(
                pipeline, s_materialModule, identifiers[3],
                context.getEmptyModule(), nullptr);
            programSet->callableProgramBSDFEvaluateInternal.create(
                pipeline, s_materialModule, identifiers[4],
                context.getEmptyModule(), nullptr);
            programSet->callableProgramBSDFEvaluatePDFInternal.create(
                pipeline, s_materialModule, identifiers[5],
                context.getEmptyModule(), nullptr);
            programSet->callableProgramBSDFWeightInternal.create(
                pipeline, s_materialModule, identifiers[6],
                context.getEmptyModule(), nullptr);

            Shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = programSet->callableProgramBSDFGetBaseColor.ID;
                bsdfProcSet.progMatches = programSet->callableProgramBSDFmatches.ID;
                bsdfProcSet.progSampleInternal = programSet->callableProgramBSDFSampleInternal.ID;
                bsdfProcSet.progEvaluateInternal = programSet->callableProgramBSDFEvaluateInternal.ID;
                bsdfProcSet.progEvaluatePDFInternal = programSet->callableProgramBSDFEvaluatePDFInternal.ID;
                bsdfProcSet.progWeightInternal = programSet->callableProgramBSDFWeightInternal.ID;
            }
            programSet->bsdfProcedureSetIndex = context.allocateBSDFProcedureSet();
            context.updateBSDFProcedureSet(programSet->bsdfProcedureSetIndex, bsdfProcSet);
        }

        if (identifiers[7] && identifiers[8] && identifiers[9]) {
            programSet->callableProgramSetupEDF.create(
                pipeline, s_materialModule, identifiers[7],
                context.getEmptyModule(), nullptr);

            programSet->callableProgramEDFEvaluateEmittanceInternal.create(
                pipeline, s_materialModule, identifiers[8],
                context.getEmptyModule(), nullptr);
            programSet->callableProgramEDFEvaluateInternal.create(
                pipeline, s_materialModule, identifiers[9],
                context.getEmptyModule(), nullptr);

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = programSet->callableProgramEDFEvaluateEmittanceInternal.ID;
                edfProcSet.progEvaluateInternal = programSet->callableProgramEDFEvaluateInternal.ID;
            }
            programSet->edfProcedureSetIndex = context.allocateEDFProcedureSet();
            context.updateEDFProcedureSet(programSet->edfProcedureSetIndex, edfProcSet);
        }
    }

    // static
    void SurfaceMaterial::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        if (programSet.callableProgramSetupEDF) {
            context.releaseEDFProcedureSet(programSet.edfProcedureSetIndex);

            programSet.callableProgramEDFEvaluateInternal.destroy();
            programSet.callableProgramEDFEvaluateEmittanceInternal.destroy();

            programSet.callableProgramSetupEDF.destroy();
        }

        if (programSet.callableProgramSetupBSDF) {
            context.releaseBSDFProcedureSet(programSet.bsdfProcedureSetIndex);

            programSet.callableProgramBSDFWeightInternal.destroy();
            programSet.callableProgramBSDFEvaluatePDFInternal.destroy();
            programSet.callableProgramBSDFEvaluateInternal.destroy();
            programSet.callableProgramBSDFSampleInternal.destroy();
            programSet.callableProgramBSDFmatches.destroy();
            programSet.callableProgramBSDFGetBaseColor.destroy();

            programSet.callableProgramSetupBSDF.destroy();
        }
    }

    // static
    void SurfaceMaterial::setupMaterialDescriptorHead(Context &context, const OptiXProgramSet &progSet, Shared::SurfaceMaterialDescriptor* matDesc) {
        if (progSet.callableProgramSetupBSDF) {
            matDesc->progSetupBSDF = progSet.callableProgramSetupBSDF.ID;
            matDesc->bsdfProcedureSetIndex = progSet.bsdfProcedureSetIndex;
        }
        else {
            matDesc->progSetupBSDF = context.getOptixCallableProgramNullBSDF_setupBSDF().ID;
            matDesc->bsdfProcedureSetIndex = context.getNullBSDFProcedureSetIndex();
        }

        if (progSet.callableProgramSetupEDF) {
            matDesc->progSetupEDF = progSet.callableProgramSetupEDF.ID;
            matDesc->edfProcedureSetIndex = progSet.edfProcedureSetIndex;
        }
        else {
            matDesc->progSetupEDF = context.getOptixCallableProgramNullEDF_setupEDF().ID;
            matDesc->edfProcedureSetIndex = context.getNullEDFProcedureSetIndex();
        }
    }

    // static
    void SurfaceMaterial::initialize(Context &context) {
        optixu::Pipeline pipeline = context.getOptixPipeline();
        s_materialModule = pipeline.createModuleFromPTXString(
            readTxtFile(getExecutableDirectory() / "ptxes/materials.ptx"),
            OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        MatteSurfaceMaterial::initialize(context);
        SpecularReflectionSurfaceMaterial::initialize(context);
        SpecularScatteringSurfaceMaterial::initialize(context);
        MicrofacetReflectionSurfaceMaterial::initialize(context);
        MicrofacetScatteringSurfaceMaterial::initialize(context);
        LambertianScatteringSurfaceMaterial::initialize(context);
        UE4SurfaceMaterial::initialize(context);
        OldStyleSurfaceMaterial::initialize(context);
        DiffuseEmitterSurfaceMaterial::initialize(context);
        MultiSurfaceMaterial::initialize(context);
        EnvironmentEmitterSurfaceMaterial::initialize(context);
    }

    // static
    void SurfaceMaterial::finalize(Context &context) {
        EnvironmentEmitterSurfaceMaterial::finalize(context);
        MultiSurfaceMaterial::finalize(context);
        DiffuseEmitterSurfaceMaterial::finalize(context);
        OldStyleSurfaceMaterial::finalize(context);
        UE4SurfaceMaterial::finalize(context);
        LambertianScatteringSurfaceMaterial::finalize(context);
        MicrofacetScatteringSurfaceMaterial::finalize(context);
        MicrofacetReflectionSurfaceMaterial::finalize(context);
        SpecularScatteringSurfaceMaterial::finalize(context);
        SpecularReflectionSurfaceMaterial::finalize(context);
        MatteSurfaceMaterial::finalize(context);

        s_materialModule.destroy();
    }

    SurfaceMaterial::SurfaceMaterial(Context &context) : Queryable(context) {
        m_matIndex = m_context.allocateSurfaceMaterialDescriptor();
    }

    SurfaceMaterial::~SurfaceMaterial() {
        if (m_matIndex != 0xFFFFFFFF)
            m_context.releaseSurfaceMaterialDescriptor(m_matIndex);
        m_matIndex = 0xFFFFFFFF;
    }



    std::vector<ParameterInfo> MatteSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MatteSurfaceMaterial::OptiXProgramSets;

    // static
    void MatteSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("albedo", VLRParameterFormFlag_Both, ParameterSpectrum),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("MatteSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("MatteBRDF_getBaseColor"),
            RT_DC_NAME_STR("MatteBRDF_matches"),
            RT_DC_NAME_STR("MatteBRDF_sampleInternal"),
            RT_DC_NAME_STR("MatteBRDF_evaluateInternal"),
            RT_DC_NAME_STR("MatteBRDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("MatteBRDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MatteSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    MatteSurfaceMaterial::MatteSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_immAlbedo(ColorSpace::Rec709_D65, 0.18f, 0.18f, 0.18f) {
        setupMaterialDescriptor();
    }

    MatteSurfaceMaterial::~MatteSurfaceMaterial() {
    }

    void MatteSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::MatteSurfaceMaterial>();
        mat.nodeAlbedo = m_nodeAlbedo.getSharedType();
        mat.immAlbedo = m_immAlbedo.createTripletSpectrum(SpectrumType::Reflectance);

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MatteSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "albedo")) {
            *spectrum = m_immAlbedo;
        }
        else {
            return false;
        }

        return true;
    }

    bool MatteSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "albedo")) {
            *plug = m_nodeAlbedo;
        }
        else {
            return false;
        }

        return true;
    }

    bool MatteSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "albedo")) {
            m_immAlbedo = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool MatteSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "albedo")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeAlbedo = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> SpecularReflectionSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularReflectionSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularReflectionSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("coeff", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("eta", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("k", VLRParameterFormFlag_Both, ParameterSpectrum),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("SpecularReflectionSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("SpecularBRDF_getBaseColor"),
            RT_DC_NAME_STR("SpecularBRDF_matches"),
            RT_DC_NAME_STR("SpecularBRDF_sampleInternal"),
            RT_DC_NAME_STR("SpecularBRDF_evaluateInternal"),
            RT_DC_NAME_STR("SpecularBRDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("SpecularBRDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void SpecularReflectionSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    SpecularReflectionSurfaceMaterial::SpecularReflectionSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(ColorSpace::Rec709_D65, 0.8f, 0.8f, 0.8f),
        m_immEta(ColorSpace::Rec709_D65, 1.0f, 1.0f, 1.0f),
        m_imm_k(ColorSpace::Rec709_D65, 0.0f, 0.0f, 0.0f) {
        setupMaterialDescriptor();
    }

    SpecularReflectionSurfaceMaterial::~SpecularReflectionSurfaceMaterial() {
    }

    void SpecularReflectionSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::SpecularReflectionSurfaceMaterial>();
        mat.nodeCoeffR = m_nodeCoeff.getSharedType();
        mat.nodeEta = m_nodeEta.getSharedType();
        mat.node_k = m_node_k.getSharedType();
        mat.immCoeffR = m_immCoeff.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immEta = m_immEta.createTripletSpectrum(SpectrumType::IndexOfRefraction);
        mat.imm_k = m_imm_k.createTripletSpectrum(SpectrumType::IndexOfRefraction);

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool SpecularReflectionSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *spectrum = m_immCoeff;
        }
        else if (testParamName(paramName, "eta")) {
            *spectrum = m_immEta;
        }
        else if (testParamName(paramName, "k")) {
            *spectrum = m_imm_k;
        }
        else {
            return false;
        }

        return true;
    }

    bool SpecularReflectionSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *plug = m_nodeCoeff;
        }
        else if (testParamName(paramName, "eta")) {
            *plug = m_nodeEta;
        }
        else if (testParamName(paramName, "k")) {
            *plug = m_node_k;
        }
        else {
            return false;
        }

        return true;
    }

    bool SpecularReflectionSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "coeff")) {
            m_immCoeff = spectrum;
        }
        else if (testParamName(paramName, "eta")) {
            m_immEta = spectrum;
        }
        else if (testParamName(paramName, "k")) {
            m_imm_k = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool SpecularReflectionSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "coeff")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeCoeff = plug;
        }
        else if (testParamName(paramName, "eta")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEta = plug;
        }
        else if (testParamName(paramName, "k")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node_k = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> SpecularScatteringSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularScatteringSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("coeff", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("eta ext", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("eta int", VLRParameterFormFlag_Both, ParameterSpectrum),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("SpecularScatteringSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("SpecularBSDF_getBaseColor"),
            RT_DC_NAME_STR("SpecularBSDF_matches"),
            RT_DC_NAME_STR("SpecularBSDF_sampleInternal"),
            RT_DC_NAME_STR("SpecularBSDF_evaluateInternal"),
            RT_DC_NAME_STR("SpecularBSDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("SpecularBSDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void SpecularScatteringSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    SpecularScatteringSurfaceMaterial::SpecularScatteringSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(ColorSpace::Rec709_D65, 0.8f, 0.8f, 0.8f),
        m_immEtaExt(ColorSpace::Rec709_D65, 1.0f, 1.0f, 1.0f),
        m_immEtaInt(ColorSpace::Rec709_D65, 1.5f, 1.5f, 1.5f) {
        setupMaterialDescriptor();
    }

    SpecularScatteringSurfaceMaterial::~SpecularScatteringSurfaceMaterial() {
    }

    void SpecularScatteringSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::SpecularScatteringSurfaceMaterial>();
        mat.nodeCoeff = m_nodeCoeff.getSharedType();
        mat.nodeEtaExt = m_nodeEtaExt.getSharedType();
        mat.nodeEtaInt = m_nodeEtaInt.getSharedType();
        mat.immCoeff = m_immCoeff.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immEtaExt = m_immEtaExt.createTripletSpectrum(SpectrumType::IndexOfRefraction);
        mat.immEtaInt = m_immEtaInt.createTripletSpectrum(SpectrumType::IndexOfRefraction);

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool SpecularScatteringSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *spectrum = m_immCoeff;
        }
        else if (testParamName(paramName, "eta ext")) {
            *spectrum = m_immEtaExt;
        }
        else if (testParamName(paramName, "eta int")) {
            *spectrum = m_immEtaInt;
        }
        else {
            return false;
        }

        return true;
    }

    bool SpecularScatteringSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *plug = m_nodeCoeff;
        }
        else if (testParamName(paramName, "eta ext")) {
            *plug = m_nodeEtaExt;
        }
        else if (testParamName(paramName, "eta int")) {
            *plug = m_nodeEtaInt;
        }
        else {
            return false;
        }

        return true;
    }

    bool SpecularScatteringSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "coeff")) {
            m_immCoeff = spectrum;
        }
        else if (testParamName(paramName, "eta ext")) {
            m_immEtaExt = spectrum;
        }
        else if (testParamName(paramName, "eta int")) {
            m_immEtaInt = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool SpecularScatteringSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "coeff")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeCoeff = plug;
        }
        else if (testParamName(paramName, "eta ext")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEtaExt = plug;
        }
        else if (testParamName(paramName, "eta int")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEtaInt = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> MicrofacetReflectionSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MicrofacetReflectionSurfaceMaterial::OptiXProgramSets;

    // static
    void MicrofacetReflectionSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("eta", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("k", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("roughness/anisotropy/rotation", VLRParameterFormFlag_Node, ParameterFloat, 3),
            ParameterInfo("roughness", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("anisotropy", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("rotation", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("MicrofacetReflectionSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("MicrofacetBRDF_getBaseColor"),
            RT_DC_NAME_STR("MicrofacetBRDF_matches"),
            RT_DC_NAME_STR("MicrofacetBRDF_sampleInternal"),
            RT_DC_NAME_STR("MicrofacetBRDF_evaluateInternal"),
            RT_DC_NAME_STR("MicrofacetBRDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("MicrofacetBRDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MicrofacetReflectionSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    MicrofacetReflectionSurfaceMaterial::MicrofacetReflectionSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immEta(ColorSpace::Rec709_D65, 1.0f, 1.0f, 1.0f),
        m_imm_k(ColorSpace::Rec709_D65, 0.0f, 0.0f, 0.0f),
        m_immRoughness(0.1f), m_immAnisotropy(0.0f), m_immRotation(0.0f) {
        setupMaterialDescriptor();
    }

    MicrofacetReflectionSurfaceMaterial::~MicrofacetReflectionSurfaceMaterial() {
    }

    void MicrofacetReflectionSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::MicrofacetReflectionSurfaceMaterial>();
        mat.nodeEta = m_nodeEta.getSharedType();
        mat.node_k = m_node_k.getSharedType();
        mat.nodeRoughnessAnisotropyRotation = m_nodeRoughnessAnisotropyRotation.getSharedType();
        mat.immEta = m_immEta.createTripletSpectrum(SpectrumType::IndexOfRefraction);
        mat.imm_k = m_imm_k.createTripletSpectrum(SpectrumType::IndexOfRefraction);
        mat.immRoughness = m_immRoughness;
        mat.immAnisotropy = m_immAnisotropy;
        mat.immRotation = m_immRotation;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MicrofacetReflectionSurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "roughness")) {
            if (length != 1)
                return false;

            values[0] = m_immRoughness;
        }
        else if (testParamName(paramName, "anisotropy")) {
            if (length != 1)
                return false;

            values[0] = m_immAnisotropy;
        }
        else if (testParamName(paramName, "rotation")) {
            if (length != 1)
                return false;

            values[0] = m_immRotation;
        }
        else {
            return false;
        }

        return true;
    }

    bool MicrofacetReflectionSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "eta")) {
            *spectrum = m_immEta;
        }
        else if (testParamName(paramName, "k")) {
            *spectrum = m_imm_k;
        }
        else {
            return false;
        }

        return true;
    }

    bool MicrofacetReflectionSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "eta")) {
            *plug = m_nodeEta;
        }
        else if (testParamName(paramName, "k")) {
            *plug = m_node_k;
        }
        else if (testParamName(paramName, "roughness/anisotropy/rotation")) {
            *plug = m_nodeRoughnessAnisotropyRotation;
        }
        else {
            return false;
        }

        return true;
    }

    bool MicrofacetReflectionSurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "roughness")) {
            if (length != 1)
                return false;

            m_immRoughness = values[0];
        }
        else if (testParamName(paramName, "anisotropy")) {
            if (length != 1)
                return false;

            m_immAnisotropy = values[0];
        }
        else if (testParamName(paramName, "rotation")) {
            if (length != 1)
                return false;

            m_immRotation = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool MicrofacetReflectionSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "eta")) {
            m_immEta = spectrum;
        }
        else if (testParamName(paramName, "k")) {
            m_imm_k = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool MicrofacetReflectionSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "eta")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEta = plug;
        }
        else if (testParamName(paramName, "k")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_node_k = plug;
        }
        else if (testParamName(paramName, "roughness/anisotropy/rotation")) {
            if (!Shared::NodeTypeInfo<float3>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeRoughnessAnisotropyRotation = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> MicrofacetScatteringSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MicrofacetScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void MicrofacetScatteringSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("coeff", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("eta ext", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("eta Int", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("roughness/anisotropy/rotation", VLRParameterFormFlag_Node, ParameterFloat, 3),
            ParameterInfo("roughness", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("anisotropy", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("rotation", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("MicrofacetScatteringSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("MicrofacetBSDF_getBaseColor"),
            RT_DC_NAME_STR("MicrofacetBSDF_matches"),
            RT_DC_NAME_STR("MicrofacetBSDF_sampleInternal"),
            RT_DC_NAME_STR("MicrofacetBSDF_evaluateInternal"),
            RT_DC_NAME_STR("MicrofacetBSDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("MicrofacetBSDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MicrofacetScatteringSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    MicrofacetScatteringSurfaceMaterial::MicrofacetScatteringSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(ColorSpace::Rec709_D65, 0.8f, 0.8f, 0.8f),
        m_immEtaExt(ColorSpace::Rec709_D65, 1.0f, 1.0f, 1.0f),
        m_immEtaInt(ColorSpace::Rec709_D65, 1.5f, 1.5f, 1.5f),
        m_immRoughness(0.1f), m_immAnisotropy(0.0f), m_immRotation(0.0f) {
        setupMaterialDescriptor();
    }

    MicrofacetScatteringSurfaceMaterial::~MicrofacetScatteringSurfaceMaterial() {
    }

    void MicrofacetScatteringSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::MicrofacetScatteringSurfaceMaterial>();
        mat.nodeCoeff = m_nodeCoeff.getSharedType();
        mat.nodeEtaExt = m_nodeEtaExt.getSharedType();
        mat.nodeEtaInt = m_nodeEtaInt.getSharedType();
        mat.nodeRoughnessAnisotropyRotation = m_nodeRoughnessAnisotropyRotation.getSharedType();
        mat.immCoeff = m_immCoeff.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immEtaExt = m_immEtaExt.createTripletSpectrum(SpectrumType::IndexOfRefraction);
        mat.immEtaInt = m_immEtaInt.createTripletSpectrum(SpectrumType::IndexOfRefraction);
        mat.immRoughness = m_immRoughness;
        mat.immAnisotropy = m_immAnisotropy;
        mat.immRotation = m_immRotation;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MicrofacetScatteringSurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "roughness")) {
            if (length != 1)
                return false;

            values[0] = m_immRoughness;
        }
        else if (testParamName(paramName, "anisotropy")) {
            if (length != 1)
                return false;

            values[0] = m_immAnisotropy;
        }
        else if (testParamName(paramName, "rotation")) {
            if (length != 1)
                return false;

            values[0] = m_immRotation;
        }
        else {
            return false;
        }

        return true;
    }

    bool MicrofacetScatteringSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *spectrum = m_immCoeff;
        }
        else if (testParamName(paramName, "eta ext")) {
            *spectrum = m_immEtaExt;
        }
        else if (testParamName(paramName, "eta int")) {
            *spectrum = m_immEtaInt;
        }
        else {
            return false;
        }

        return true;
    }

    bool MicrofacetScatteringSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *plug = m_nodeCoeff;
        }
        else if (testParamName(paramName, "eta ext")) {
            *plug = m_nodeEtaExt;
        }
        else if (testParamName(paramName, "eta int")) {
            *plug = m_nodeEtaInt;
        }
        else if (testParamName(paramName, "roughness/anisotropy/rotation")) {
            *plug = m_nodeRoughnessAnisotropyRotation;
        }
        else {
            return false;
        }

        return true;
    }

    bool MicrofacetScatteringSurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "roughness")) {
            if (length != 1)
                return false;

            m_immRoughness = values[0];
        }
        else if (testParamName(paramName, "anisotropy")) {
            if (length != 1)
                return false;

            m_immAnisotropy = values[0];
        }
        else if (testParamName(paramName, "rotation")) {
            if (length != 1)
                return false;

            m_immRotation = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool MicrofacetScatteringSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "coeff")) {
            m_immCoeff = spectrum;
        }
        else if (testParamName(paramName, "eta ext")) {
            m_immEtaExt = spectrum;
        }
        else if (testParamName(paramName, "eta int")) {
            m_immEtaInt = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool MicrofacetScatteringSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "coeff")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeCoeff = plug;
        }
        else if (testParamName(paramName, "eta ext")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEtaExt = plug;
        }
        else if (testParamName(paramName, "eta int")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEtaInt = plug;
        }
        else if (testParamName(paramName, "roughness/anisotropy/rotation")) {
            if (!Shared::NodeTypeInfo<float3>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeRoughnessAnisotropyRotation = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> LambertianScatteringSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> LambertianScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void LambertianScatteringSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("coeff", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("f0", VLRParameterFormFlag_Both, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("LambertianScatteringSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("LambertianBSDF_getBaseColor"),
            RT_DC_NAME_STR("LambertianBSDF_matches"),
            RT_DC_NAME_STR("LambertianBSDF_sampleInternal"),
            RT_DC_NAME_STR("LambertianBSDF_evaluateInternal"),
            RT_DC_NAME_STR("LambertianBSDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("LambertianBSDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void LambertianScatteringSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    LambertianScatteringSurfaceMaterial::LambertianScatteringSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(ColorSpace::Rec709_D65, 0.8f, 0.8f, 0.8f), m_immF0(0.04f) {
        setupMaterialDescriptor();
    }

    LambertianScatteringSurfaceMaterial::~LambertianScatteringSurfaceMaterial() {
    }

    void LambertianScatteringSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::LambertianScatteringSurfaceMaterial>();
        mat.nodeCoeff = m_nodeCoeff.getSharedType();
        mat.nodeF0 = m_nodeF0.getSharedType();
        mat.immCoeff = m_immCoeff.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immF0 = m_immF0;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool LambertianScatteringSurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "f0")) {
            if (length != 1)
                return false;

            values[0] = m_immF0;
        }
        else {
            return false;
        }

        return true;
    }

    bool LambertianScatteringSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *spectrum = m_immCoeff;
        }
        else {
            return false;
        }

        return true;
    }

    bool LambertianScatteringSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "coeff")) {
            *plug = m_nodeCoeff;
        }
        else if (testParamName(paramName, "f0")) {
            *plug = m_nodeF0;
        }
        else {
            return false;
        }

        return true;
    }

    bool LambertianScatteringSurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "f0")) {
            if (length != 1)
                return false;

            m_immF0 = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool LambertianScatteringSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "coeff")) {
            m_immCoeff = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool LambertianScatteringSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "coeff")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeCoeff = plug;
        }
        else if (testParamName(paramName, "f0")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeF0 = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> UE4SurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> UE4SurfaceMaterial::OptiXProgramSets;

    // static
    void UE4SurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("base color", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("occlusion/roughness/metallic", VLRParameterFormFlag_Node, ParameterFloat, 3),
            ParameterInfo("occlusion", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("roughness", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
            ParameterInfo("metallic", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("UE4SurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_getBaseColor"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_matches"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_sampleInternal"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluateInternal"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void UE4SurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    UE4SurfaceMaterial::UE4SurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immBaseColor(ColorSpace::Rec709_D65, 0.18f, 0.18f, 0.18f), m_immOcculusion(0.0f), m_immRoughness(0.1f), m_immMetallic(0.0f) {
        setupMaterialDescriptor();
    }

    UE4SurfaceMaterial::~UE4SurfaceMaterial() {
    }

    void UE4SurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::UE4SurfaceMaterial>();
        mat.nodeBaseColor = m_nodeBaseColor.getSharedType();
        mat.nodeOcclusionRoughnessMetallic = m_nodeOcclusionRoughnessMetallic.getSharedType();
        mat.immBaseColor = m_immBaseColor.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immOcclusion = m_immOcculusion;
        mat.immRoughness = m_immRoughness;
        mat.immMetallic = m_immMetallic;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool UE4SurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "occlusion")) {
            if (length != 1)
                return false;

            values[0] = m_immOcculusion;
        }
        else if (testParamName(paramName, "roughness")) {
            if (length != 1)
                return false;

            values[0] = m_immRoughness;
        }
        else if (testParamName(paramName, "metallic")) {
            if (length != 1)
                return false;

            values[0] = m_immMetallic;
        }
        else {
            return false;
        }

        return true;
    }

    bool UE4SurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "base color")) {
            *spectrum = m_immBaseColor;
        }
        else {
            return false;
        }

        return true;
    }

    bool UE4SurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "base color")) {
            *plug = m_nodeBaseColor;
        }
        else if (testParamName(paramName, "occlusion/roughness/metallic")) {
            *plug = m_nodeOcclusionRoughnessMetallic;
        }
        else {
            return false;
        }

        return true;
    }

    bool UE4SurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "occlusion")) {
            if (length != 1)
                return false;

            m_immOcculusion = values[0];
        }
        else if (testParamName(paramName, "roughness")) {
            if (length != 1)
                return false;

            m_immRoughness = values[0];
        }
        else if (testParamName(paramName, "metallic")) {
            if (length != 1)
                return false;

            m_immMetallic = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool UE4SurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "base color")) {
            m_immBaseColor = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool UE4SurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "base color")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeBaseColor = plug;
        }
        else if (testParamName(paramName, "occlusion/roughness/metallic")) {
            if (!Shared::NodeTypeInfo<float3>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeOcclusionRoughnessMetallic = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> OldStyleSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> OldStyleSurfaceMaterial::OptiXProgramSets;

    // static
    void OldStyleSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("diffuse", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("specular", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("glossiness", VLRParameterFormFlag_Both, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("OldStyleSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_getBaseColor"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_matches"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_sampleInternal"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluateInternal"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("DiffuseAndSpecularBRDF_weightInternal"),
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void OldStyleSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    OldStyleSurfaceMaterial::OldStyleSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immDiffuseColor(ColorSpace::Rec709_D65, 0.18f, 0.18f, 0.18f),
        m_immSpecularColor(ColorSpace::Rec709_D65, 0.04f, 0.04f, 0.04f),
        m_immGlossiness(0.6f) {
        setupMaterialDescriptor();
    }

    OldStyleSurfaceMaterial::~OldStyleSurfaceMaterial() {
    }

    void OldStyleSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::OldStyleSurfaceMaterial>();
        mat.nodeDiffuseColor = m_nodeDiffuseColor.getSharedType();
        mat.nodeSpecularColor = m_nodeSpecularColor.getSharedType();
        mat.nodeGlossiness = m_nodeGlossiness.getSharedType();
        mat.immDiffuseColor = m_immDiffuseColor.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immSpecularColor = m_immSpecularColor.createTripletSpectrum(SpectrumType::Reflectance);
        mat.immGlossiness = m_immGlossiness;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool OldStyleSurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "glossiness")) {
            if (length != 1)
                return false;

            values[0] = m_immGlossiness;
        }
        else {
            return false;
        }

        return true;
    }

    bool OldStyleSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "diffuse")) {
            *spectrum = m_immDiffuseColor;
        }
        else if (testParamName(paramName, "specular")) {
            *spectrum = m_immSpecularColor;
        }
        else {
            return false;
        }

        return true;
    }

    bool OldStyleSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "diffuse")) {
            *plug = m_nodeDiffuseColor;
        }
        else if (testParamName(paramName, "specular")) {
            *plug = m_nodeSpecularColor;
        }
        else if (testParamName(paramName, "glossiness")) {
            *plug = m_nodeGlossiness;
        }
        else {
            return false;
        }

        return true;
    }

    bool OldStyleSurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "glossiness")) {
            if (length != 1)
                return false;

            m_immGlossiness = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool OldStyleSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "diffuse")) {
            m_immDiffuseColor = spectrum;
        }
        else if (testParamName(paramName, "specular")) {
            m_immSpecularColor = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool OldStyleSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "diffuse")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeDiffuseColor = plug;
        }
        else if (testParamName(paramName, "specular")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeSpecularColor = plug;
        }
        else if (testParamName(paramName, "glossiness")) {
            if (!Shared::NodeTypeInfo<float>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeGlossiness = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> DiffuseEmitterSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> DiffuseEmitterSurfaceMaterial::OptiXProgramSets;

    // static
    void DiffuseEmitterSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("emittance", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("scale", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            RT_DC_NAME_STR("DiffuseEmitterSurfaceMaterial_setupEDF"),
            RT_DC_NAME_STR("DiffuseEDF_evaluateEmittanceInternal"),
            RT_DC_NAME_STR("DiffuseEDF_evaluateInternal")
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void DiffuseEmitterSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    DiffuseEmitterSurfaceMaterial::DiffuseEmitterSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_immEmittance(ColorSpace::Rec709_D65, M_PI, M_PI, M_PI), m_immScale(1.0f) {
        setupMaterialDescriptor();
    }

    DiffuseEmitterSurfaceMaterial::~DiffuseEmitterSurfaceMaterial() {
    }

    void DiffuseEmitterSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::DiffuseEmitterSurfaceMaterial>();
        mat.nodeEmittance = m_nodeEmittance.getSharedType();
        mat.immEmittance = m_immEmittance.createTripletSpectrum(SpectrumType::LightSource);
        mat.immScale = m_immScale;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool DiffuseEmitterSurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "scale")) {
            if (length != 1)
                return false;

            values[0] = m_immScale;
        }
        else {
            return false;
        }

        return true;
    }

    bool DiffuseEmitterSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "emittance")) {
            *spectrum = m_immEmittance;
        }
        else {
            return false;
        }

        return true;
    }

    bool DiffuseEmitterSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "emittance")) {
            *plug = m_nodeEmittance;
        }
        else {
            return false;
        }

        return true;
    }

    bool DiffuseEmitterSurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "scale")) {
            if (length != 1)
                return false;

            m_immScale = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool DiffuseEmitterSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "emittance")) {
            m_immEmittance = spectrum;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool DiffuseEmitterSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "emittance")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEmittance = plug;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }



    std::vector<ParameterInfo> MultiSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MultiSurfaceMaterial::OptiXProgramSets;

    // static
    void MultiSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("0", VLRParameterFormFlag_Node, ParameterSurfaceMaterial),
            ParameterInfo("1", VLRParameterFormFlag_Node, ParameterSurfaceMaterial),
            ParameterInfo("2", VLRParameterFormFlag_Node, ParameterSurfaceMaterial),
            ParameterInfo("3", VLRParameterFormFlag_Node, ParameterSurfaceMaterial),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            RT_DC_NAME_STR("MultiSurfaceMaterial_setupBSDF"),
            RT_DC_NAME_STR("MultiBSDF_getBaseColor"),
            RT_DC_NAME_STR("MultiBSDF_matches"),
            RT_DC_NAME_STR("MultiBSDF_sampleInternal"),
            RT_DC_NAME_STR("MultiBSDF_evaluateInternal"),
            RT_DC_NAME_STR("MultiBSDF_evaluatePDFInternal"),
            RT_DC_NAME_STR("MultiBSDF_weightInternal"),
            RT_DC_NAME_STR("MultiSurfaceMaterial_setupEDF"),
            RT_DC_NAME_STR("MultiEDF_evaluateEmittanceInternal"),
            RT_DC_NAME_STR("MultiEDF_evaluateInternal")
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MultiSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    MultiSurfaceMaterial::MultiSurfaceMaterial(Context &context) :
        SurfaceMaterial(context) {
        std::fill_n(m_subMaterials, lengthof(m_subMaterials), nullptr);
        setupMaterialDescriptor();
    }

    MultiSurfaceMaterial::~MultiSurfaceMaterial() {
    }

    void MultiSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::MultiSurfaceMaterial>();

        mat.numSubMaterials = 0;
        std::fill_n(mat.subMatIndices, lengthof(mat.subMatIndices), 0xFFFFFFFF);
        for (int i = 0; i < lengthof(m_subMaterials); ++i) {
            if (m_subMaterials[i])
                mat.subMatIndices[mat.numSubMaterials++] = m_subMaterials[i]->getMaterialIndex();
        }

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MultiSurfaceMaterial::get(const char* paramName, const SurfaceMaterial** material) const {
        if (material == nullptr)
            return false;

        if (strcmp(paramName, "0") == 0) {
            *material = m_subMaterials[0];
        }
        else if (strcmp(paramName, "1") == 0) {
            *material = m_subMaterials[1];
        }
        else if (strcmp(paramName, "2") == 0) {
            *material = m_subMaterials[2];
        }
        else if (strcmp(paramName, "3") == 0) {
            *material = m_subMaterials[3];
        }
        else {
            return false;
        }

        return true;
    }

    bool MultiSurfaceMaterial::set(const char* paramName, const SurfaceMaterial* material) {
        if (strcmp(paramName, "0") == 0) {
            m_subMaterials[0] = material;
        }
        else if (strcmp(paramName, "1") == 0) {
            m_subMaterials[1] = material;
        }
        else if (strcmp(paramName, "2") == 0) {
            m_subMaterials[2] = material;
        }
        else if (strcmp(paramName, "3") == 0) {
            m_subMaterials[3] = material;
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool MultiSurfaceMaterial::isEmitting() const {
        for (int i = 0; i < lengthof(m_subMaterials); ++i) {
            if (m_subMaterials[i]) {
                if (m_subMaterials[i]->isEmitting())
                    return true;
            }
        }
        return false;
    }



    std::vector<ParameterInfo> EnvironmentEmitterSurfaceMaterial::ParameterInfos;
    
    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> EnvironmentEmitterSurfaceMaterial::OptiXProgramSets;

    // static
    void EnvironmentEmitterSurfaceMaterial::initialize(Context &context) {
        const ParameterInfo paramInfos[] = {
            ParameterInfo("emittance", VLRParameterFormFlag_Both, ParameterSpectrum),
            ParameterInfo("scale", VLRParameterFormFlag_ImmediateValue, ParameterFloat),
        };

        if (ParameterInfos.size() == 0) {
            ParameterInfos.resize(lengthof(paramInfos));
            std::copy_n(paramInfos, lengthof(paramInfos), ParameterInfos.data());
        }

        const char* identifiers[] = {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            RT_DC_NAME_STR("EnvironmentEmitterSurfaceMaterial_setupEDF"),
            RT_DC_NAME_STR("EnvironmentEDF_evaluateEmittanceInternal"),
            RT_DC_NAME_STR("EnvironmentEDF_evaluateInternal")
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EnvironmentEmitterSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
        OptiXProgramSets.erase(context.getID());
    }

    EnvironmentEmitterSurfaceMaterial::EnvironmentEmitterSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_immEmittance(ColorSpace::Rec709_D65, M_PI, M_PI, M_PI), m_immScale(1.0f) {
        setupMaterialDescriptor();
    }

    EnvironmentEmitterSurfaceMaterial::~EnvironmentEmitterSurfaceMaterial() {
        m_importanceMap.finalize(m_context);
    }

    void EnvironmentEmitterSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        auto &mat = *matDesc.getData<Shared::EnvironmentEmitterSurfaceMaterial>();
        mat.nodeEmittance = m_nodeEmittance.getSharedType();
        mat.immEmittance = m_immEmittance.createTripletSpectrum(SpectrumType::LightSource);
        mat.immScale = m_immScale;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool EnvironmentEmitterSurfaceMaterial::get(const char* paramName, float* values, uint32_t length) const {
        if (values == nullptr)
            return false;

        if (testParamName(paramName, "scale")) {
            if (length != 1)
                return false;

            values[0] = m_immScale;
        }
        else {
            return false;
        }

        return true;
    }

    bool EnvironmentEmitterSurfaceMaterial::get(const char* paramName, ImmediateSpectrum* spectrum) const {
        if (spectrum == nullptr)
            return false;

        if (testParamName(paramName, "emittance")) {
            *spectrum = m_immEmittance;
        }
        else {
            return false;
        }

        return true;
    }

    bool EnvironmentEmitterSurfaceMaterial::get(const char* paramName, ShaderNodePlug* plug) const {
        if (plug == nullptr)
            return false;

        if (testParamName(paramName, "emittance")) {
            *plug = m_nodeEmittance;
        }
        else {
            return false;
        }

        return true;
    }

    bool EnvironmentEmitterSurfaceMaterial::set(const char* paramName, const float* values, uint32_t length) {
        if (testParamName(paramName, "scale")) {
            if (length != 1)
                return false;

            m_immScale = values[0];
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool EnvironmentEmitterSurfaceMaterial::set(const char* paramName, const ImmediateSpectrum& spectrum) {
        if (testParamName(paramName, "emittance")) {
            m_immEmittance = spectrum;
            if (m_importanceMap.isInitialized())
                m_importanceMap.finalize(m_context);
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    bool EnvironmentEmitterSurfaceMaterial::set(const char* paramName, const ShaderNodePlug& plug) {
        if (testParamName(paramName, "emittance")) {
            if (!Shared::NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom(plug.getType()))
                return false;

            m_nodeEmittance = plug;
            if (m_importanceMap.isInitialized())
                m_importanceMap.finalize(m_context);
        }
        else {
            return false;
        }
        setupMaterialDescriptor();

        return true;
    }

    const RegularConstantContinuousDistribution2D &EnvironmentEmitterSurfaceMaterial::getImportanceMap() {
        if (!m_importanceMap.isInitialized()) {
            if (m_nodeEmittance.node && m_nodeEmittance.node->is<EnvironmentTextureShaderNode>()) {
                auto node = (EnvironmentTextureShaderNode*)m_nodeEmittance.node;
                node->createImportanceMap(&m_importanceMap);
            }
            else {
                uint32_t mapWidth = 512;
                uint32_t mapHeight = 256;
                float* linearData = new float[mapWidth * mapHeight];
                std::fill_n(linearData, mapWidth * mapHeight, 1.0f);
                for (int y = 0; y < mapHeight; ++y) {
                    float theta = M_PI * (y + 0.5f) / mapHeight;
                    for (int x = 0; x < mapWidth; ++x) {
                        linearData[y * mapWidth + x] *= std::sin(theta);
                    }
                }

                m_importanceMap.initialize(m_context, linearData, mapWidth, mapHeight);

                delete[] linearData;
            }
        }

        return m_importanceMap;
    }
}
