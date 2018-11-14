#include "materials.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Material

    // static
    void SurfaceMaterial::commonInitializeProcedure(Context &context, const char* identifiers[10], OptiXProgramSet* programSet) {
        std::string ptx = readTxtFile("resources/ptxes/materials.ptx");

        optix::Context optixContext = context.getOptiXContext();

        if (identifiers[0] && identifiers[1] && identifiers[2] && identifiers[3] && identifiers[4] && identifiers[5] && identifiers[6]) {
            programSet->callableProgramSetupBSDF = optixContext->createProgramFromPTXString(ptx, identifiers[0]);

            programSet->callableProgramBSDFGetBaseColor = optixContext->createProgramFromPTXString(ptx, identifiers[1]);
            programSet->callableProgramBSDFmatches = optixContext->createProgramFromPTXString(ptx, identifiers[2]);
            programSet->callableProgramBSDFSampleInternal = optixContext->createProgramFromPTXString(ptx, identifiers[3]);
            programSet->callableProgramBSDFEvaluateInternal = optixContext->createProgramFromPTXString(ptx, identifiers[4]);
            programSet->callableProgramBSDFEvaluatePDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[5]);
            programSet->callableProgramBSDFWeightInternal = optixContext->createProgramFromPTXString(ptx, identifiers[6]);

            Shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = programSet->callableProgramBSDFGetBaseColor->getId();
                bsdfProcSet.progMatches = programSet->callableProgramBSDFmatches->getId();
                bsdfProcSet.progSampleInternal = programSet->callableProgramBSDFSampleInternal->getId();
                bsdfProcSet.progEvaluateInternal = programSet->callableProgramBSDFEvaluateInternal->getId();
                bsdfProcSet.progEvaluatePDFInternal = programSet->callableProgramBSDFEvaluatePDFInternal->getId();
                bsdfProcSet.progWeightInternal = programSet->callableProgramBSDFWeightInternal->getId();
            }
            programSet->bsdfProcedureSetIndex = context.allocateBSDFProcedureSet();
            context.updateBSDFProcedureSet(programSet->bsdfProcedureSetIndex, bsdfProcSet);
        }

        if (identifiers[7] && identifiers[8] && identifiers[9]) {
            programSet->callableProgramSetupEDF = optixContext->createProgramFromPTXString(ptx, identifiers[7]);

            programSet->callableProgramEDFEvaluateEmittanceInternal = optixContext->createProgramFromPTXString(ptx, identifiers[8]);
            programSet->callableProgramEDFEvaluateInternal = optixContext->createProgramFromPTXString(ptx, identifiers[9]);

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = programSet->callableProgramEDFEvaluateEmittanceInternal->getId();
                edfProcSet.progEvaluateInternal = programSet->callableProgramEDFEvaluateInternal->getId();
            }
            programSet->edfProcedureSetIndex = context.allocateEDFProcedureSet();
            context.updateEDFProcedureSet(programSet->edfProcedureSetIndex, edfProcSet);
        }
    }

    // static
    void SurfaceMaterial::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        if (programSet.callableProgramSetupEDF) {
            context.releaseEDFProcedureSet(programSet.edfProcedureSetIndex);

            programSet.callableProgramEDFEvaluateInternal->destroy();
            programSet.callableProgramEDFEvaluateEmittanceInternal->destroy();

            programSet.callableProgramSetupEDF->destroy();
        }

        if (programSet.callableProgramSetupBSDF) {
            context.releaseBSDFProcedureSet(programSet.bsdfProcedureSetIndex);

            programSet.callableProgramBSDFWeightInternal->destroy();
            programSet.callableProgramBSDFEvaluatePDFInternal->destroy();
            programSet.callableProgramBSDFEvaluateInternal->destroy();
            programSet.callableProgramBSDFSampleInternal->destroy();
            programSet.callableProgramBSDFmatches->destroy();
            programSet.callableProgramBSDFGetBaseColor->destroy();

            programSet.callableProgramSetupBSDF->destroy();
        }
    }

    // static
    void SurfaceMaterial::setupMaterialDescriptorHead(Context &context, const OptiXProgramSet &progSet, Shared::SurfaceMaterialDescriptor* matDesc) {
        Shared::SurfaceMaterialHead &head = *(Shared::SurfaceMaterialHead*)&matDesc->data[0];

        if (progSet.callableProgramSetupBSDF) {
            head.progSetupBSDF = progSet.callableProgramSetupBSDF->getId();
            head.bsdfProcedureSetIndex = progSet.bsdfProcedureSetIndex;
        }
        else {
            head.progSetupBSDF = context.getOptixCallableProgramNullBSDF_setupBSDF()->getId();
            head.bsdfProcedureSetIndex = context.getNullBSDFProcedureSetIndex();
        }

        if (progSet.callableProgramSetupEDF) {
            head.progSetupEDF = progSet.callableProgramSetupEDF->getId();
            head.edfProcedureSetIndex = progSet.edfProcedureSetIndex;
        }
        else {
            head.progSetupEDF = context.getOptixCallableProgramNullEDF_setupEDF()->getId();
            head.edfProcedureSetIndex = context.getNullEDFProcedureSetIndex();
        }
    }

    // static
    void SurfaceMaterial::initialize(Context &context) {
        MatteSurfaceMaterial::initialize(context);
        SpecularReflectionSurfaceMaterial::initialize(context);
        SpecularScatteringSurfaceMaterial::initialize(context);
        MicrofacetReflectionSurfaceMaterial::initialize(context);
        MicrofacetScatteringSurfaceMaterial::initialize(context);
        LambertianScatteringSurfaceMaterial::initialize(context);
        UE4SurfaceMaterial::initialize(context);
        DiffuseEmitterSurfaceMaterial::initialize(context);
        MultiSurfaceMaterial::initialize(context);
        EnvironmentEmitterSurfaceMaterial::initialize(context);
    }

    // static
    void SurfaceMaterial::finalize(Context &context) {
        EnvironmentEmitterSurfaceMaterial::finalize(context);
        MultiSurfaceMaterial::finalize(context);
        DiffuseEmitterSurfaceMaterial::finalize(context);
        UE4SurfaceMaterial::finalize(context);
        LambertianScatteringSurfaceMaterial::finalize(context);
        MicrofacetScatteringSurfaceMaterial::finalize(context);
        MicrofacetReflectionSurfaceMaterial::finalize(context);
        SpecularScatteringSurfaceMaterial::finalize(context);
        SpecularReflectionSurfaceMaterial::finalize(context);
        MatteSurfaceMaterial::finalize(context);
    }

    SurfaceMaterial::SurfaceMaterial(Context &context) : Object(context) {
        m_matIndex = m_context.allocateSurfaceMaterialDescriptor();
    }

    SurfaceMaterial::~SurfaceMaterial() {
        if (m_matIndex != 0xFFFFFFFF)
            m_context.releaseSurfaceMaterialDescriptor(m_matIndex);
        m_matIndex = 0xFFFFFFFF;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MatteSurfaceMaterial::OptiXProgramSets;

    // static
    void MatteSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MatteSurfaceMaterial_setupBSDF",
            "VLR::MatteBRDF_getBaseColor",
            "VLR::MatteBRDF_matches",
            "VLR::MatteBRDF_sampleInternal",
            "VLR::MatteBRDF_evaluateInternal",
            "VLR::MatteBRDF_evaluatePDFInternal",
            "VLR::MatteBRDF_weightInternal",
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
    }

    MatteSurfaceMaterial::MatteSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_immAlbedo(RGBSpectrum(0.18f)) {
        setupMaterialDescriptor();
    }

    MatteSurfaceMaterial::~MatteSurfaceMaterial() {
    }

    void MatteSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::MatteSurfaceMaterial &mat = *(Shared::MatteSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeAlbedo = m_nodeAlbedo.getSharedType();
        mat.immAlbedo = m_immAlbedo;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MatteSurfaceMaterial::setNodeAlbedo(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeAlbedo = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MatteSurfaceMaterial::setImmediateValueAlbedo(const RGBSpectrum &value) {
        m_immAlbedo = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularReflectionSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularReflectionSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::SpecularReflectionSurfaceMaterial_setupBSDF",
            "VLR::SpecularBRDF_getBaseColor",
            "VLR::SpecularBRDF_matches",
            "VLR::SpecularBRDF_sampleInternal",
            "VLR::SpecularBRDF_evaluateInternal",
            "VLR::SpecularBRDF_evaluatePDFInternal",
            "VLR::SpecularBRDF_weightInternal",
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
    }

    SpecularReflectionSurfaceMaterial::SpecularReflectionSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeffR(RGBSpectrum(0.8f)), m_immEta(RGBSpectrum(1.0f)), m_imm_k(RGBSpectrum(0.0f)) {
        setupMaterialDescriptor();
    }

    SpecularReflectionSurfaceMaterial::~SpecularReflectionSurfaceMaterial() {
    }

    void SpecularReflectionSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::SpecularReflectionSurfaceMaterial &mat = *(Shared::SpecularReflectionSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeCoeffR = m_nodeCoeffR.getSharedType();
        mat.nodeEta = m_nodeEta.getSharedType();
        mat.node_k = m_node_k.getSharedType();
        mat.immCoeffR = m_immCoeffR;
        mat.immEta = m_immEta;
        mat.imm_k = m_imm_k;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool SpecularReflectionSurfaceMaterial::setNodeCoeffR(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeCoeffR = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void SpecularReflectionSurfaceMaterial::setImmediateValueCoeffR(const RGBSpectrum &value) {
        m_immCoeffR = value;
        setupMaterialDescriptor();
    }

    bool SpecularReflectionSurfaceMaterial::setNodeEta(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEta = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void SpecularReflectionSurfaceMaterial::setImmediateValueEta(const RGBSpectrum &value) {
        m_immEta = value;
        setupMaterialDescriptor();
    }

    bool SpecularReflectionSurfaceMaterial::setNode_k(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_node_k = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void SpecularReflectionSurfaceMaterial::setImmediateValue_k(const RGBSpectrum &value) {
        m_imm_k = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularScatteringSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::SpecularScatteringSurfaceMaterial_setupBSDF",
            "VLR::SpecularBSDF_getBaseColor",
            "VLR::SpecularBSDF_matches",
            "VLR::SpecularBSDF_sampleInternal",
            "VLR::SpecularBSDF_evaluateInternal",
            "VLR::SpecularBSDF_evaluatePDFInternal",
            "VLR::SpecularBSDF_weightInternal",
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
    }

    SpecularScatteringSurfaceMaterial::SpecularScatteringSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(RGBSpectrum(0.8f)), m_immEtaExt(RGBSpectrum(1.0f)), m_immEtaInt(RGBSpectrum(1.5f)) {
        setupMaterialDescriptor();
    }

    SpecularScatteringSurfaceMaterial::~SpecularScatteringSurfaceMaterial() {
    }

    void SpecularScatteringSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::SpecularScatteringSurfaceMaterial &mat = *(Shared::SpecularScatteringSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeCoeff = m_nodeCoeff.getSharedType();
        mat.nodeEtaExt = m_nodeEtaExt.getSharedType();
        mat.nodeEtaInt = m_nodeEtaInt.getSharedType();
        mat.immCoeff = m_immCoeff;
        mat.immEtaExt = m_immEtaExt;
        mat.immEtaInt = m_immEtaInt;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool SpecularScatteringSurfaceMaterial::setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeCoeff = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void SpecularScatteringSurfaceMaterial::setImmediateValueCoeff(const RGBSpectrum &value) {
        m_immCoeff = value;
        setupMaterialDescriptor();
    }

    bool SpecularScatteringSurfaceMaterial::setNodeEtaExt(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEtaExt = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void SpecularScatteringSurfaceMaterial::setImmediateValueEtaExt(const RGBSpectrum &value) {
        m_immEtaExt = value;
        setupMaterialDescriptor();
    }

    bool SpecularScatteringSurfaceMaterial::setNodeEtaInt(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEtaInt = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void SpecularScatteringSurfaceMaterial::setImmediateValueEtaInt(const RGBSpectrum &value) {
        m_immEtaInt = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MicrofacetReflectionSurfaceMaterial::OptiXProgramSets;

    // static
    void MicrofacetReflectionSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MicrofacetReflectionSurfaceMaterial_setupBSDF",
            "VLR::MicrofacetBRDF_getBaseColor",
            "VLR::MicrofacetBRDF_matches",
            "VLR::MicrofacetBRDF_sampleInternal",
            "VLR::MicrofacetBRDF_evaluateInternal",
            "VLR::MicrofacetBRDF_evaluatePDFInternal",
            "VLR::MicrofacetBRDF_weightInternal",
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
    }

    MicrofacetReflectionSurfaceMaterial::MicrofacetReflectionSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immEta(RGBSpectrum(1.0f)), m_imm_k(RGBSpectrum(0.0f)), m_immRoughness(0.1f), m_immAnisotropy(0.0f), m_immRotation(0.0f) {
        setupMaterialDescriptor();
    }

    MicrofacetReflectionSurfaceMaterial::~MicrofacetReflectionSurfaceMaterial() {
    }

    void MicrofacetReflectionSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::MicrofacetReflectionSurfaceMaterial &mat = *(Shared::MicrofacetReflectionSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeEta = m_nodeEta.getSharedType();
        mat.node_k = m_node_k.getSharedType();
        mat.nodeRoughnessAnisotropyRotation = m_nodeRoughnessAnisotropyRotation.getSharedType();
        mat.immEta = m_immEta;
        mat.imm_k = m_imm_k;
        mat.immRoughness = m_immRoughness;
        mat.immAnisotropy = m_immAnisotropy;
        mat.immRotation = m_immRotation;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MicrofacetReflectionSurfaceMaterial::setNodeEta(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEta = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetReflectionSurfaceMaterial::setImmediateValueEta(const RGBSpectrum &value) {
        m_immEta = value;
        setupMaterialDescriptor();
    }

    bool MicrofacetReflectionSurfaceMaterial::setNode_k(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_node_k = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetReflectionSurfaceMaterial::setImmediateValue_k(const RGBSpectrum &value) {
        m_imm_k = value;
        setupMaterialDescriptor();
    }

    bool MicrofacetReflectionSurfaceMaterial::setNodeRoughnessAnisotropyRotation(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float2)
            return false;
        m_nodeRoughnessAnisotropyRotation = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetReflectionSurfaceMaterial::setImmediateValueRoughness(float value) {
        m_immRoughness = value;
        setupMaterialDescriptor();
    }

    void MicrofacetReflectionSurfaceMaterial::setImmediateValueAnisotropy(float value) {
        m_immAnisotropy = value;
        setupMaterialDescriptor();
    }

    void MicrofacetReflectionSurfaceMaterial::setImmediateValueRotation(float value) {
        m_immRotation = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MicrofacetScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void MicrofacetScatteringSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MicrofacetScatteringSurfaceMaterial_setupBSDF",
            "VLR::MicrofacetBSDF_getBaseColor",
            "VLR::MicrofacetBSDF_matches",
            "VLR::MicrofacetBSDF_sampleInternal",
            "VLR::MicrofacetBSDF_evaluateInternal",
            "VLR::MicrofacetBSDF_evaluatePDFInternal",
            "VLR::MicrofacetBSDF_weightInternal",
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
    }

    MicrofacetScatteringSurfaceMaterial::MicrofacetScatteringSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(RGBSpectrum(0.8f)), m_immEtaExt(RGBSpectrum(1.0f)), m_immEtaInt(RGBSpectrum(1.5f)), m_immRoughness(0.1f), m_immAnisotropy(0.0f), m_immRotation(0.0f) {
        setupMaterialDescriptor();
    }

    MicrofacetScatteringSurfaceMaterial::~MicrofacetScatteringSurfaceMaterial() {
    }

    void MicrofacetScatteringSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::MicrofacetScatteringSurfaceMaterial &mat = *(Shared::MicrofacetScatteringSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeCoeff = m_nodeCoeff.getSharedType();
        mat.nodeEtaExt = m_nodeEtaExt.getSharedType();
        mat.nodeEtaInt = m_nodeEtaInt.getSharedType();
        mat.nodeRoughnessAnisotropyRotation = m_nodeRoughnessAnisotropyRotation.getSharedType();
        mat.immCoeff = m_immCoeff;
        mat.immEtaExt = m_immEtaExt;
        mat.immEtaInt = m_immEtaInt;
        mat.immRoughness = m_immRoughness;
        mat.immAnisotropy = m_immAnisotropy;
        mat.immRotation = m_immRotation;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MicrofacetScatteringSurfaceMaterial::setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeCoeff = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetScatteringSurfaceMaterial::setImmediateValueCoeff(const RGBSpectrum &value) {
        m_immCoeff = value;
        setupMaterialDescriptor();
    }

    bool MicrofacetScatteringSurfaceMaterial::setNodeEtaExt(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEtaExt = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetScatteringSurfaceMaterial::setImmediateValueEtaExt(const RGBSpectrum &value) {
        m_immEtaExt = value;
        setupMaterialDescriptor();
    }

    bool MicrofacetScatteringSurfaceMaterial::setNodeEtaInt(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEtaInt = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetScatteringSurfaceMaterial::setImmediateValueEtaInt(const RGBSpectrum &value) {
        m_immEtaInt = value;
        setupMaterialDescriptor();
    }

    bool MicrofacetScatteringSurfaceMaterial::setNodeRoughnessAnisotropyRotation(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float2)
            return false;
        m_nodeRoughnessAnisotropyRotation = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void MicrofacetScatteringSurfaceMaterial::setImmediateValueRoughness(float value) {
        m_immRoughness = value;
        setupMaterialDescriptor();
    }

    void MicrofacetScatteringSurfaceMaterial::setImmediateValueAnisotropy(float value) {
        m_immAnisotropy = value;
        setupMaterialDescriptor();
    }

    void MicrofacetScatteringSurfaceMaterial::setImmediateValueRotation(float value) {
        m_immRotation = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> LambertianScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void LambertianScatteringSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::LambertianScatteringSurfaceMaterial_setupBSDF",
            "VLR::LambertianBSDF_getBaseColor",
            "VLR::LambertianBSDF_matches",
            "VLR::LambertianBSDF_sampleInternal",
            "VLR::LambertianBSDF_evaluateInternal",
            "VLR::LambertianBSDF_evaluatePDFInternal",
            "VLR::LambertianBSDF_weightInternal",
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
    }

    LambertianScatteringSurfaceMaterial::LambertianScatteringSurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immCoeff(RGBSpectrum(0.8f)), m_immF0(0.04f) {
        setupMaterialDescriptor();
    }

    LambertianScatteringSurfaceMaterial::~LambertianScatteringSurfaceMaterial() {
    }

    void LambertianScatteringSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::LambertianScatteringSurfaceMaterial &mat = *(Shared::LambertianScatteringSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeCoeff = m_nodeCoeff.getSharedType();
        mat.nodeF0 = m_nodeF0.getSharedType();
        mat.immCoeff = m_immCoeff;
        mat.immF0 = m_immF0;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool LambertianScatteringSurfaceMaterial::setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeCoeff = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void LambertianScatteringSurfaceMaterial::setImmediateValueCoeff(const RGBSpectrum &value) {
        m_immCoeff = value;
        setupMaterialDescriptor();
    }

    bool LambertianScatteringSurfaceMaterial::setNodeF0(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float)
            return false;
        m_nodeF0 = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void LambertianScatteringSurfaceMaterial::setImmediateValueF0(float value) {
        m_immF0 = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> UE4SurfaceMaterial::OptiXProgramSets;

    // static
    void UE4SurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::UE4SurfaceMaterial_setupBSDF",
            "VLR::UE4BRDF_getBaseColor",
            "VLR::UE4BRDF_matches",
            "VLR::UE4BRDF_sampleInternal",
            "VLR::UE4BRDF_evaluateInternal",
            "VLR::UE4BRDF_evaluatePDFInternal",
            "VLR::UE4BRDF_weightInternal",
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
    }

    UE4SurfaceMaterial::UE4SurfaceMaterial(Context &context) :
        SurfaceMaterial(context),
        m_immBaseColor(RGBSpectrum(0.18f)), m_immOcculusion(0.0f), m_immRoughness(0.1f), m_immMetallic(0.0f) {
        setupMaterialDescriptor();
    }

    UE4SurfaceMaterial::~UE4SurfaceMaterial() {
    }

    void UE4SurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::UE4SurfaceMaterial &mat = *(Shared::UE4SurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeBaseColor = m_nodeBaseColor.getSharedType();
        mat.nodeOcclusionRoughnessMetallic = m_nodeOcclusionRoughnessMetallic.getSharedType();
        mat.immBaseColor = m_immBaseColor;
        mat.immOcclusion = m_immOcculusion;
        mat.immRoughness = m_immRoughness;
        mat.immMetallic = m_immMetallic;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool UE4SurfaceMaterial::setNodeBaseColor(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeBaseColor = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void UE4SurfaceMaterial::setImmediateValueBaseColor(const RGBSpectrum &value) {
        m_immBaseColor = value;
        setupMaterialDescriptor();
    }

    bool UE4SurfaceMaterial::setNodeOcclusionRoughnessMetallic(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_float3)
            return false;
        m_nodeOcclusionRoughnessMetallic = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void UE4SurfaceMaterial::setImmediateValueOcclusion(float value) {
        m_immOcculusion = value;
        setupMaterialDescriptor();
    }

    void UE4SurfaceMaterial::setImmediateValueRoughness(float value) {
        m_immRoughness = value;
        setupMaterialDescriptor();
    }

    void UE4SurfaceMaterial::setImmediateValueMetallic(float value) {
        m_immMetallic = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> DiffuseEmitterSurfaceMaterial::OptiXProgramSets;

    // static
    void DiffuseEmitterSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            "VLR::DiffuseEmitterSurfaceMaterial_setupEDF",
            "VLR::DiffuseEDF_evaluateEmittanceInternal",
            "VLR::DiffuseEDF_evaluateInternal"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void DiffuseEmitterSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    DiffuseEmitterSurfaceMaterial::DiffuseEmitterSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_immEmittance(RGBSpectrum(M_PI)) {
        setupMaterialDescriptor();
    }

    DiffuseEmitterSurfaceMaterial::~DiffuseEmitterSurfaceMaterial() {
    }

    void DiffuseEmitterSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::DiffuseEmitterSurfaceMaterial &mat = *(Shared::DiffuseEmitterSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        mat.nodeEmittance = m_nodeEmittance.getSharedType();
        mat.immEmittance = m_immEmittance;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool DiffuseEmitterSurfaceMaterial::setNodeEmittance(const ShaderNodeSocketIdentifier &outputSocket) {
        if (outputSocket.getType() != VLRShaderNodeSocketType_Spectrum)
            return false;
        m_nodeEmittance = outputSocket;
        setupMaterialDescriptor();
        return true;
    }

    void DiffuseEmitterSurfaceMaterial::setImmediateValueEmittance(const RGBSpectrum &value) {
        m_immEmittance = value;
        setupMaterialDescriptor();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MultiSurfaceMaterial::OptiXProgramSets;

    // static
    void MultiSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MultiSurfaceMaterial_setupBSDF",
            "VLR::MultiBSDF_getBaseColor",
            "VLR::MultiBSDF_matches",
            "VLR::MultiBSDF_sampleInternal",
            "VLR::MultiBSDF_evaluateInternal",
            "VLR::MultiBSDF_evaluatePDFInternal",
            "VLR::MultiBSDF_weightInternal",
            "VLR::MultiSurfaceMaterial_setupEDF",
            "VLR::MultiEDF_evaluateEmittanceInternal",
            "VLR::MultiEDF_evaluateInternal"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MultiSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    MultiSurfaceMaterial::MultiSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_numSubMaterials(0) {
        setupMaterialDescriptor();
    }

    MultiSurfaceMaterial::~MultiSurfaceMaterial() {
    }

    void MultiSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::MultiSurfaceMaterial &mat = *(Shared::MultiSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];

        for (int i = 0; i < lengthof(m_subMaterials); ++i)
            mat.subMatIndices[i] = m_subMaterials[i]->getMaterialIndex();
        mat.numSubMaterials = m_numSubMaterials;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool MultiSurfaceMaterial::isEmitting() const {
        for (int i = 0; i < m_numSubMaterials; ++i) {
            if (m_subMaterials[i]->isEmitting())
                return true;
        }
        return false;
    }

    void MultiSurfaceMaterial::setSubMaterial(uint32_t index, const SurfaceMaterial* mat) {
        VLRAssert(index < lengthof(m_subMaterials), "Out of range.");
        m_subMaterials[index] = mat;
        m_numSubMaterials = 0;
        for (int i = 0; i < lengthof(m_subMaterials); ++i)
            if (m_subMaterials[i] != nullptr)
                ++m_numSubMaterials;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> EnvironmentEmitterSurfaceMaterial::OptiXProgramSets;

    // static
    void EnvironmentEmitterSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            "VLR::EnvironmentEmitterSurfaceMaterial_setupEDF",
            "VLR::EnvironmentEDF_evaluateEmittanceInternal",
            "VLR::EnvironmentEDF_evaluateInternal"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EnvironmentEmitterSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    EnvironmentEmitterSurfaceMaterial::EnvironmentEmitterSurfaceMaterial(Context &context) :
        SurfaceMaterial(context), m_nodeEmittance(nullptr), m_immEmittance(RGBSpectrum(M_PI)) {
        setupMaterialDescriptor();
    }

    EnvironmentEmitterSurfaceMaterial::~EnvironmentEmitterSurfaceMaterial() {
        m_importanceMap.finalize(m_context);
    }

    void EnvironmentEmitterSurfaceMaterial::setupMaterialDescriptor() const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptorHead(m_context, progSet, &matDesc);
        Shared::EnvironmentEmitterSurfaceMaterial &mat = *(Shared::EnvironmentEmitterSurfaceMaterial*)&matDesc.data[sizeof(Shared::SurfaceMaterialHead) / 4];
        Shared::ShaderNodeSocketID nodeIndexEmittance = Shared::ShaderNodeSocketID::Invalid();
        if (m_nodeEmittance) {
            nodeIndexEmittance.nodeDescIndex = m_nodeEmittance->getShaderNodeIndex();
            nodeIndexEmittance.socketIndex = 0;
            nodeIndexEmittance.option = 0;
        }
        mat.nodeEmittance = nodeIndexEmittance;
        mat.immEmittance = m_immEmittance;

        m_context.updateSurfaceMaterialDescriptor(m_matIndex, matDesc);
    }

    bool EnvironmentEmitterSurfaceMaterial::setNodeEmittance(const EnvironmentTextureShaderNode* node) {
        m_nodeEmittance = node;
        setupMaterialDescriptor();
        if (m_importanceMap.isInitialized())
            m_importanceMap.finalize(m_context);
        return true;
    }

    void EnvironmentEmitterSurfaceMaterial::setImmediateValueEmittance(const RGBSpectrum &value) {
        m_immEmittance = value;
        setupMaterialDescriptor();
        if (m_importanceMap.isInitialized())
            m_importanceMap.finalize(m_context);
    }

    const RegularConstantContinuousDistribution2D &EnvironmentEmitterSurfaceMaterial::getImportanceMap() {
        if (!m_importanceMap.isInitialized()) {
            if (m_nodeEmittance) {
                m_nodeEmittance->createImportanceMap(&m_importanceMap);
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

    // END: Material
    // ----------------------------------------------------------------
}
