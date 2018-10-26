#pragma once

#include "textures.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Material

    class SurfaceMaterial : public Object {
    protected:
        struct OptiXProgramSet {
            optix::Program callableProgramSetupBSDF;
            optix::Program callableProgramBSDFGetBaseColor;
            optix::Program callableProgramBSDFmatches;
            optix::Program callableProgramBSDFSampleInternal;
            optix::Program callableProgramBSDFEvaluateInternal;
            optix::Program callableProgramBSDFEvaluatePDFInternal;
            optix::Program callableProgramBSDFWeightInternal;
            uint32_t bsdfProcedureSetIndex;

            optix::Program callableProgramSetupEDF;
            optix::Program callableProgramEDFEvaluateEmittanceInternal;
            optix::Program callableProgramEDFEvaluateInternal;
            uint32_t edfProcedureSetIndex;
        };

        uint32_t m_matIndex;

        static void commonInitializeProcedure(Context &context, const char* identifiers[10], OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet);
        static uint32_t setupMaterialDescriptorHead(Context &context, const OptiXProgramSet &progSet, Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex);

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SurfaceMaterial(Context &context);
        virtual ~SurfaceMaterial();

        uint32_t getMaterialIndex() const {
            return m_matIndex;
        }

        virtual uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const = 0;
        virtual bool isEmitting() const { return false; }
    };



    class MatteSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeAlbedo;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context, const ShaderNode* nodeAlbedo);
        ~MatteSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeCoeffR;
        const ShaderNode* m_nodeEta;
        const ShaderNode* m_node_k;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context, const ShaderNode* nodeCoeffR, const ShaderNode* nodeEta, const ShaderNode* node_k);
        ~SpecularReflectionSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeCoeff;
        const ShaderNode* m_nodeEtaExt;
        const ShaderNode* m_nodeEtaInt;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context, const ShaderNode* nodeCoeff, const ShaderNode* nodeEtaExt, const ShaderNode* nodeEtaInt);
        ~SpecularScatteringSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class MicrofacetReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeEta;
        const ShaderNode* m_node_k;
        const ShaderNode* m_nodeRoughness;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MicrofacetReflectionSurfaceMaterial(Context &context, const ShaderNode* nodeEta, const ShaderNode* node_k, const ShaderNode* nodeRoughness);
        ~MicrofacetReflectionSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class MicrofacetScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeCoeff;
        const ShaderNode* m_nodeEtaExt;
        const ShaderNode* m_nodeEtaInt;
        const ShaderNode* m_nodeRoughness;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MicrofacetScatteringSurfaceMaterial(Context &context, const ShaderNode* nodeCoeff, const ShaderNode* nodeEtaExt, const ShaderNode* nodeEtaInt, const ShaderNode* nodeRoughness);
        ~MicrofacetScatteringSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class LambertianScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeCoeff;
        const ShaderNode* m_nodeF0;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        LambertianScatteringSurfaceMaterial(Context &context, const ShaderNode* nodeCoeff, const ShaderNode* nodeF0);
        ~LambertianScatteringSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeBaseColor;
        const ShaderNode* m_nodeOcclusionRoughnessMetallic;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context, const ShaderNode* nodeBaseColor, const ShaderNode* nodeOcclusionRoughnessMetallic);
        ~UE4SurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeEmittance;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        DiffuseEmitterSurfaceMaterial(Context &context, const ShaderNode* nodeEmittance);
        ~DiffuseEmitterSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
        bool isEmitting() const override { return true; }
    };



    class MultiSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const SurfaceMaterial* m_materials[4];
        uint32_t m_numMaterials;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MultiSurfaceMaterial(Context &context, const SurfaceMaterial** materials, uint32_t numMaterials);
        ~MultiSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
        bool isEmitting() const override;
    };



    class EnvironmentEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const ShaderNode* m_nodeEmittance;
        RegularConstantContinuousDistribution2D m_importanceMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentEmitterSurfaceMaterial(Context &context, const ShaderNode* nodeEmittance);
        ~EnvironmentEmitterSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
        bool isEmitting() const override { return true; }

        const RegularConstantContinuousDistribution2D &getImportanceMap() const {
            return m_importanceMap;
        }
    };

    // END: Material
    // ----------------------------------------------------------------
}
