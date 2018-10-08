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

            programSet->callableProgramGetBaseColor = optixContext->createProgramFromPTXString(ptx, identifiers[1]);
            programSet->callableProgramBSDFmatches = optixContext->createProgramFromPTXString(ptx, identifiers[2]);
            programSet->callableProgramSampleBSDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[3]);
            programSet->callableProgramEvaluateBSDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[4]);
            programSet->callableProgramEvaluateBSDF_PDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[5]);
            programSet->callableProgramBSDFWeightInternal = optixContext->createProgramFromPTXString(ptx, identifiers[6]);

            Shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = programSet->callableProgramGetBaseColor->getId();
                bsdfProcSet.progBSDFmatches = programSet->callableProgramBSDFmatches->getId();
                bsdfProcSet.progSampleBSDFInternal = programSet->callableProgramSampleBSDFInternal->getId();
                bsdfProcSet.progEvaluateBSDFInternal = programSet->callableProgramEvaluateBSDFInternal->getId();
                bsdfProcSet.progEvaluateBSDF_PDFInternal = programSet->callableProgramEvaluateBSDF_PDFInternal->getId();
                bsdfProcSet.progWeightInternal = programSet->callableProgramBSDFWeightInternal->getId();
            }
            programSet->bsdfProcedureSetIndex = context.setBSDFProcedureSet(bsdfProcSet);
        }

        if (identifiers[7] && identifiers[8] && identifiers[9]) {
            programSet->callableProgramSetupEDF = optixContext->createProgramFromPTXString(ptx, identifiers[7]);

            programSet->callableProgramEvaluateEmittanceInternal = optixContext->createProgramFromPTXString(ptx, identifiers[8]);
            programSet->callableProgramEvaluateEDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[9]);

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = programSet->callableProgramEvaluateEmittanceInternal->getId();
                edfProcSet.progEvaluateEDFInternal = programSet->callableProgramEvaluateEDFInternal->getId();
            }
            programSet->edfProcedureSetIndex = context.setEDFProcedureSet(edfProcSet);
        }
    }

    // static
    void SurfaceMaterial::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        if (programSet.callableProgramSetupEDF) {
            context.unsetEDFProcedureSet(programSet.edfProcedureSetIndex);

            programSet.callableProgramEvaluateEDFInternal->destroy();
            programSet.callableProgramEvaluateEmittanceInternal->destroy();

            programSet.callableProgramSetupEDF->destroy();
        }

        if (programSet.callableProgramSetupBSDF) {
            context.unsetBSDFProcedureSet(programSet.bsdfProcedureSetIndex);

            programSet.callableProgramBSDFWeightInternal->destroy();
            programSet.callableProgramEvaluateBSDF_PDFInternal->destroy();
            programSet.callableProgramEvaluateBSDFInternal->destroy();
            programSet.callableProgramSampleBSDFInternal->destroy();
            programSet.callableProgramBSDFmatches->destroy();
            programSet.callableProgramGetBaseColor->destroy();

            programSet.callableProgramSetupBSDF->destroy();
        }
    }

    // static
    uint32_t SurfaceMaterial::setupMaterialDescriptorHead(Context &context, const OptiXProgramSet &progSet, Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) {
        Shared::SurfaceMaterialHead &head = *(Shared::SurfaceMaterialHead*)&matDesc->i1[baseIndex];

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

        return baseIndex + sizeof(Shared::SurfaceMaterialHead) / 4;
    }

    // static
    void SurfaceMaterial::initialize(Context &context) {
        MatteSurfaceMaterial::initialize(context);
        SpecularReflectionSurfaceMaterial::initialize(context);
        SpecularScatteringSurfaceMaterial::initialize(context);
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
        SpecularScatteringSurfaceMaterial::finalize(context);
        SpecularReflectionSurfaceMaterial::finalize(context);
        MatteSurfaceMaterial::finalize(context);
    }

    SurfaceMaterial::SurfaceMaterial(Context &context) : Object(context) {
        m_matIndex = 0xFFFFFFFF;
    }

    SurfaceMaterial::~SurfaceMaterial() {
        if (m_matIndex != 0xFFFFFFFF)
            m_context.unsetSurfaceMaterialDescriptor(m_matIndex);
        m_matIndex = 0xFFFFFFFF;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MatteSurfaceMaterial::OptiXProgramSets;

    // static
    void MatteSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MatteSurfaceMaterial_setupBSDF",
            "VLR::MatteBRDF_getBaseColor",
            "VLR::MatteBRDF_matches",
            "VLR::MatteBRDF_sampleBSDFInternal",
            "VLR::MatteBRDF_evaluateBSDFInternal",
            "VLR::MatteBRDF_evaluateBSDF_PDFInternal",
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

    MatteSurfaceMaterial::MatteSurfaceMaterial(Context &context, const Float4Texture* texAlbedoRoughness, const TextureMap2D* texMap) :
        SurfaceMaterial(context), m_texAlbedoRoughness(texAlbedoRoughness), m_texMap(texMap) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
    }

    MatteSurfaceMaterial::~MatteSurfaceMaterial() {
    }

    uint32_t MatteSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::MatteSurfaceMaterial &mat = *(Shared::MatteSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texAlbedoRoughness = m_texAlbedoRoughness->getOptiXObject()->getId();
        if (m_texMap)
            mat.texMap = m_texMap->getTextureMapIndex();
        else
            mat.texMap = OffsetAndScaleUVTextureMap2D::getDefault(m_context)->getTextureMapIndex();

        return baseIndex + sizeof(Shared::MatteSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularReflectionSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularReflectionSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::SpecularReflectionSurfaceMaterial_setupBSDF",
            "VLR::SpecularBRDF_getBaseColor",
            "VLR::SpecularBRDF_matches",
            "VLR::SpecularBRDF_sampleBSDFInternal",
            "VLR::SpecularBRDF_evaluateBSDFInternal",
            "VLR::SpecularBRDF_evaluateBSDF_PDFInternal",
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

    SpecularReflectionSurfaceMaterial::SpecularReflectionSurfaceMaterial(Context &context, const Float3Texture* texCoeffR, const Float3Texture* texEta, const Float3Texture* tex_k, const TextureMap2D* texMap) :
        SurfaceMaterial(context), m_texCoeffR(texCoeffR), m_texEta(texEta), m_tex_k(tex_k), m_texMap(texMap) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
    }

    SpecularReflectionSurfaceMaterial::~SpecularReflectionSurfaceMaterial() {
    }

    uint32_t SpecularReflectionSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::SpecularReflectionSurfaceMaterial &mat = *(Shared::SpecularReflectionSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texCoeffR = m_texCoeffR->getOptiXObject()->getId();
        mat.texEta = m_texEta->getOptiXObject()->getId();
        mat.tex_k = m_tex_k->getOptiXObject()->getId();
        if (m_texMap)
            mat.texMap = m_texMap->getTextureMapIndex();
        else
            mat.texMap = OffsetAndScaleUVTextureMap2D::getDefault(m_context)->getTextureMapIndex();

        return baseIndex + sizeof(Shared::SpecularReflectionSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularScatteringSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::SpecularScatteringSurfaceMaterial_setupBSDF",
            "VLR::SpecularBSDF_getBaseColor",
            "VLR::SpecularBSDF_matches",
            "VLR::SpecularBSDF_sampleBSDFInternal",
            "VLR::SpecularBSDF_evaluateBSDFInternal",
            "VLR::SpecularBSDF_evaluateBSDF_PDFInternal",
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

    SpecularScatteringSurfaceMaterial::SpecularScatteringSurfaceMaterial(Context &context, const Float3Texture* texCoeff, const Float3Texture* texEtaExt, const Float3Texture* texEtaInt, const TextureMap2D* texMap) :
        SurfaceMaterial(context), m_texCoeff(texCoeff), m_texEtaExt(texEtaExt), m_texEtaInt(texEtaInt), m_texMap(texMap) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
    }

    SpecularScatteringSurfaceMaterial::~SpecularScatteringSurfaceMaterial() {
    }

    uint32_t SpecularScatteringSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::SpecularScatteringSurfaceMaterial &mat = *(Shared::SpecularScatteringSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texCoeff = m_texCoeff->getOptiXObject()->getId();
        mat.texEtaExt = m_texEtaExt->getOptiXObject()->getId();
        mat.texEtaInt = m_texEtaInt->getOptiXObject()->getId();
        if (m_texMap)
            mat.texMap = m_texMap->getTextureMapIndex();
        else
            mat.texMap = OffsetAndScaleUVTextureMap2D::getDefault(m_context)->getTextureMapIndex();

        return baseIndex + sizeof(Shared::SpecularScatteringSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> UE4SurfaceMaterial::OptiXProgramSets;

    // static
    void UE4SurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::UE4SurfaceMaterial_setupBSDF",
            "VLR::UE4BRDF_getBaseColor",
            "VLR::UE4BRDF_matches",
            "VLR::UE4BRDF_sampleBSDFInternal",
            "VLR::UE4BRDF_evaluateBSDFInternal",
            "VLR::UE4BRDF_evaluateBSDF_PDFInternal",
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

    UE4SurfaceMaterial::UE4SurfaceMaterial(Context &context, const Float3Texture* texBaseColor, const Float3Texture* texOcclusionRoughnessMetallic, const TextureMap2D* texMap) :
        SurfaceMaterial(context), m_texBaseColor(texBaseColor), m_texOcclusionRoughnessMetallic(texOcclusionRoughnessMetallic), m_texMap(texMap) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
    }

    UE4SurfaceMaterial::~UE4SurfaceMaterial() {
    }

    uint32_t UE4SurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::UE4SurfaceMaterial &mat = *(Shared::UE4SurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texBaseColor = m_texBaseColor->getOptiXObject()->getId();
        mat.texOcclusionRoughnessMetallic = m_texOcclusionRoughnessMetallic->getOptiXObject()->getId();
        if (m_texMap)
            mat.texMap = m_texMap->getTextureMapIndex();
        else
            mat.texMap = OffsetAndScaleUVTextureMap2D::getDefault(m_context)->getTextureMapIndex();

        return baseIndex + sizeof(Shared::UE4SurfaceMaterial) / 4;
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
            "VLR::DiffuseEDF_evaluateEDFInternal"
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

    DiffuseEmitterSurfaceMaterial::DiffuseEmitterSurfaceMaterial(Context &context, const Float3Texture* texEmittance, const TextureMap2D* texMap) :
        SurfaceMaterial(context), m_texEmittance(texEmittance), m_texMap(texMap) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
    }

    DiffuseEmitterSurfaceMaterial::~DiffuseEmitterSurfaceMaterial() {
    }

    uint32_t DiffuseEmitterSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::DiffuseEmitterSurfaceMaterial &mat = *(Shared::DiffuseEmitterSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texEmittance = m_texEmittance->getOptiXObject()->getId();
        if (m_texMap)
            mat.texMap = m_texMap->getTextureMapIndex();
        else
            mat.texMap = OffsetAndScaleUVTextureMap2D::getDefault(m_context)->getTextureMapIndex();

        return baseIndex + sizeof(Shared::DiffuseEmitterSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MultiSurfaceMaterial::OptiXProgramSets;

    // static
    void MultiSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MultiSurfaceMaterial_setupBSDF",
            "VLR::MultiBSDF_getBaseColor",
            "VLR::MultiBSDF_matches",
            "VLR::MultiBSDF_sampleBSDFInternal",
            "VLR::MultiBSDF_evaluateBSDFInternal",
            "VLR::MultiBSDF_evaluateBSDF_PDFInternal",
            "VLR::MultiBSDF_weightInternal",
            "VLR::MultiSurfaceMaterial_setupEDF",
            "VLR::MultiEDF_evaluateEmittanceInternal",
            "VLR::MultiEDF_evaluateEDFInternal"
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

    MultiSurfaceMaterial::MultiSurfaceMaterial(Context &context, const SurfaceMaterial** materials, uint32_t numMaterials) :
        SurfaceMaterial(context) {
        VLRAssert(numMaterials <= lengthof(m_materials), "numMaterials should be less than or equal to %u", lengthof(m_materials));
        std::copy_n(materials, numMaterials, m_materials);
        m_numMaterials = numMaterials;

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
    }

    MultiSurfaceMaterial::~MultiSurfaceMaterial() {
    }

    uint32_t MultiSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::MultiSurfaceMaterial &mat = *(Shared::MultiSurfaceMaterial*)&matDesc->i1[baseIndex];
        baseIndex += sizeof(Shared::MultiSurfaceMaterial) / 4;

        uint32_t matOffsets[4] = { 0, 0, 0, 0 };
        VLRAssert(lengthof(matOffsets) == lengthof(m_materials), "Two sizes must match.");
        for (int i = 0; i < m_numMaterials; ++i) {
            const SurfaceMaterial* mat = m_materials[i];
            matOffsets[i] = baseIndex;
            baseIndex = mat->setupMaterialDescriptor(matDesc, baseIndex);
        }
        VLRAssert(baseIndex <= VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS, "exceeds the size of SurfaceMaterialDescriptor.");

        mat.matOffset0 = matOffsets[0];
        mat.matOffset1 = matOffsets[1];
        mat.matOffset2 = matOffsets[2];
        mat.matOffset3 = matOffsets[3];
        mat.numMaterials = m_numMaterials;

        return baseIndex;
    }

    bool MultiSurfaceMaterial::isEmitting() const {
        for (int i = 0; i < m_numMaterials; ++i) {
            if (m_materials[i]->isEmitting())
                return true;
        }
        return false;
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
            "VLR::EnvironmentEDF_evaluateEDFInternal"
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

    EnvironmentEmitterSurfaceMaterial::EnvironmentEmitterSurfaceMaterial(Context &context, const Float3Texture* texEmittance) :
        SurfaceMaterial(context), m_texEmittance(texEmittance) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);

        m_texEmittance->createImportanceMap(&m_importanceMap);
    }

    EnvironmentEmitterSurfaceMaterial::~EnvironmentEmitterSurfaceMaterial() {
        m_importanceMap.finalize(m_context);
    }

    uint32_t EnvironmentEmitterSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::EnvironmentEmitterSurfaceMaterial &mat = *(Shared::EnvironmentEmitterSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texEmittance = m_texEmittance->getOptiXObject()->getId();

        return baseIndex + sizeof(Shared::EnvironmentEmitterSurfaceMaterial) / 4;
    }

    // END: Material
    // ----------------------------------------------------------------
}
