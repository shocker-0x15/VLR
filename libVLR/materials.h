#pragma once

#include "shader_nodes.h"

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
        static void setupMaterialDescriptorHead(Context &context, const OptiXProgramSet &progSet, Shared::SurfaceMaterialDescriptor* matDesc);

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

        virtual bool isEmitting() const { return false; }
    };



    class MatteSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeAlbedo;
        UpsampledSpectrum m_immAlbedo;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context);
        ~MatteSurfaceMaterial();

        bool setNodeAlbedo(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueAlbedo(const UpsampledSpectrum &value);
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeffR;
        ShaderNodeSocketIdentifier m_nodeEta;
        ShaderNodeSocketIdentifier m_node_k;
        UpsampledSpectrum m_immCoeffR;
        UpsampledSpectrum m_immEta;
        UpsampledSpectrum m_imm_k;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context);
        ~SpecularReflectionSurfaceMaterial();

        bool setNodeCoeffR(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueCoeffR(const UpsampledSpectrum &value);
        bool setNodeEta(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEta(const UpsampledSpectrum &value);
        bool setNode_k(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue_k(const UpsampledSpectrum &value);
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeff;
        ShaderNodeSocketIdentifier m_nodeEtaExt;
        ShaderNodeSocketIdentifier m_nodeEtaInt;
        UpsampledSpectrum m_immCoeff;
        UpsampledSpectrum m_immEtaExt;
        UpsampledSpectrum m_immEtaInt;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context);
        ~SpecularScatteringSurfaceMaterial();

        bool setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueCoeff(const UpsampledSpectrum &value);
        bool setNodeEtaExt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaExt(const UpsampledSpectrum &value);
        bool setNodeEtaInt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaInt(const UpsampledSpectrum &value);
    };



    class MicrofacetReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeEta;
        ShaderNodeSocketIdentifier m_node_k;
        ShaderNodeSocketIdentifier m_nodeRoughnessAnisotropyRotation;
        UpsampledSpectrum m_immEta;
        UpsampledSpectrum m_imm_k;
        float m_immRoughness;
        float m_immAnisotropy;
        float m_immRotation;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MicrofacetReflectionSurfaceMaterial(Context &context);
        ~MicrofacetReflectionSurfaceMaterial();

        bool setNodeEta(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEta(const UpsampledSpectrum &value);
        bool setNode_k(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue_k(const UpsampledSpectrum &value);
        bool setNodeRoughnessAnisotropyRotation(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueRoughness(float value);
        void setImmediateValueAnisotropy(float value);
        void setImmediateValueRotation(float value);
    };



    class MicrofacetScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeff;
        ShaderNodeSocketIdentifier m_nodeEtaExt;
        ShaderNodeSocketIdentifier m_nodeEtaInt;
        ShaderNodeSocketIdentifier m_nodeRoughnessAnisotropyRotation;
        UpsampledSpectrum m_immCoeff;
        UpsampledSpectrum m_immEtaExt;
        UpsampledSpectrum m_immEtaInt;
        float m_immRoughness;
        float m_immAnisotropy;
        float m_immRotation;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MicrofacetScatteringSurfaceMaterial(Context &context);
        ~MicrofacetScatteringSurfaceMaterial();

        bool setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueCoeff(const UpsampledSpectrum &value);
        bool setNodeEtaExt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaExt(const UpsampledSpectrum &value);
        bool setNodeEtaInt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaInt(const UpsampledSpectrum &value);
        bool setNodeRoughnessAnisotropyRotation(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueRoughness(float value);
        void setImmediateValueAnisotropy(float value);
        void setImmediateValueRotation(float value);
    };



    class LambertianScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeff;
        ShaderNodeSocketIdentifier m_nodeF0;
        UpsampledSpectrum m_immCoeff;
        float m_immF0;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        LambertianScatteringSurfaceMaterial(Context &context);
        ~LambertianScatteringSurfaceMaterial();

        bool setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueCoeff(const UpsampledSpectrum &value);
        bool setNodeF0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueF0(float value);
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeBaseColor;
        ShaderNodeSocketIdentifier m_nodeOcclusionRoughnessMetallic;
        UpsampledSpectrum m_immBaseColor;
        float m_immOcculusion;
        float m_immRoughness;
        float m_immMetallic;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context);
        ~UE4SurfaceMaterial();

        bool setNodeBaseColor(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueBaseColor(const UpsampledSpectrum &value);
        bool setNodeOcclusionRoughnessMetallic(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueOcclusion(float value);
        void setImmediateValueRoughness(float value);
        void setImmediateValueMetallic(float value);
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeEmittance;
        UpsampledSpectrum m_immEmittance;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        DiffuseEmitterSurfaceMaterial(Context &context);
        ~DiffuseEmitterSurfaceMaterial();

        bool isEmitting() const override { return true; }

        bool setNodeEmittance(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEmittance(const UpsampledSpectrum &value);
    };



    class MultiSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const SurfaceMaterial* m_subMaterials[4];
        uint32_t m_numSubMaterials;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MultiSurfaceMaterial(Context &context);
        ~MultiSurfaceMaterial();

        bool isEmitting() const override;

        void setSubMaterial(uint32_t index, const SurfaceMaterial* mat);
    };



    class EnvironmentEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const EnvironmentTextureShaderNode* m_nodeEmittance;
        UpsampledSpectrum m_immEmittance;
        RegularConstantContinuousDistribution2D m_importanceMap;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentEmitterSurfaceMaterial(Context &context);
        ~EnvironmentEmitterSurfaceMaterial();

        bool isEmitting() const override { return true; }

        bool setNodeEmittance(const EnvironmentTextureShaderNode* node);
        void setImmediateValueEmittance(const UpsampledSpectrum &value);

        const RegularConstantContinuousDistribution2D &getImportanceMap();
    };

    // END: Material
    // ----------------------------------------------------------------
}
