#pragma once

#include "shader_nodes.h"

namespace VLR {
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
        TripletSpectrum m_immAlbedo;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context);
        ~MatteSurfaceMaterial();

        bool setNodeAlbedo(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueAlbedo(ColorSpace colorSpace, float e0, float e1, float e2);
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeffR;
        ShaderNodeSocketIdentifier m_nodeEta;
        ShaderNodeSocketIdentifier m_node_k;
        TripletSpectrum m_immCoeffR;
        TripletSpectrum m_immEta;
        TripletSpectrum m_imm_k;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context);
        ~SpecularReflectionSurfaceMaterial();

        bool setNodeCoeffR(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueCoeffR(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeEta(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEta(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNode_k(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue_k(ColorSpace colorSpace, float e0, float e1, float e2);
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeff;
        ShaderNodeSocketIdentifier m_nodeEtaExt;
        ShaderNodeSocketIdentifier m_nodeEtaInt;
        TripletSpectrum m_immCoeff;
        TripletSpectrum m_immEtaExt;
        TripletSpectrum m_immEtaInt;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context);
        ~SpecularScatteringSurfaceMaterial();

        bool setNodeCoeff(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueCoeff(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeEtaExt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaExt(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeEtaInt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaInt(ColorSpace colorSpace, float e0, float e1, float e2);
    };



    class MicrofacetReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeEta;
        ShaderNodeSocketIdentifier m_node_k;
        ShaderNodeSocketIdentifier m_nodeRoughnessAnisotropyRotation;
        TripletSpectrum m_immEta;
        TripletSpectrum m_imm_k;
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
        void setImmediateValueEta(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNode_k(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue_k(ColorSpace colorSpace, float e0, float e1, float e2);
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
        TripletSpectrum m_immCoeff;
        TripletSpectrum m_immEtaExt;
        TripletSpectrum m_immEtaInt;
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
        void setImmediateValueCoeff(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeEtaExt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaExt(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeEtaInt(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueEtaInt(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeRoughnessAnisotropyRotation(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueRoughness(float value);
        void setImmediateValueAnisotropy(float value);
        void setImmediateValueRotation(float value);
    };



    class LambertianScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeCoeff;
        ShaderNodeSocketIdentifier m_nodeF0;
        TripletSpectrum m_immCoeff;
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
        void setImmediateValueCoeff(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeF0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueF0(float value);
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeBaseColor;
        ShaderNodeSocketIdentifier m_nodeOcclusionRoughnessMetallic;
        TripletSpectrum m_immBaseColor;
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
        void setImmediateValueBaseColor(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeOcclusionRoughnessMetallic(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueOcclusion(float value);
        void setImmediateValueRoughness(float value);
        void setImmediateValueMetallic(float value);
    };



    class OldStyleSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeDiffuseColor;
        ShaderNodeSocketIdentifier m_nodeSpecularColor;
        ShaderNodeSocketIdentifier m_nodeGlossiness;
        TripletSpectrum m_immDiffuseColor;
        TripletSpectrum m_immSpecularColor;
        float m_immGlossiness;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        OldStyleSurfaceMaterial(Context &context);
        ~OldStyleSurfaceMaterial();

        bool setNodeDiffuseColor(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueDiffuseColor(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeSpecularColor(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueSpecularColor(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setNodeGlossiness(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueGlossiness(float value);
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeEmittance;
        TripletSpectrum m_immEmittance;
        float m_immScale;

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
        void setImmediateValueEmittance(ColorSpace colorSpace, float e0, float e1, float e2);
        void setImmediateValueScale(float value);
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

        const EnvironmentTextureShaderNode* m_nodeEmittanceTextured;
        const ShaderNode* m_nodeEmittanceConstant;
        TripletSpectrum m_immEmittance;
        RegularConstantContinuousDistribution2D m_importanceMap;
        float m_immScale;

        void setupMaterialDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentEmitterSurfaceMaterial(Context &context);
        ~EnvironmentEmitterSurfaceMaterial();

        bool isEmitting() const override { return true; }

        bool setNodeEmittanceTextured(const EnvironmentTextureShaderNode* node);
        bool setNodeEmittanceConstant(const ShaderNode* spectrumNode);
        void setImmediateValueEmittance(ColorSpace colorSpace, float e0, float e1, float e2);
        void setImmediateValueScale(float value);

        const RegularConstantContinuousDistribution2D &getImportanceMap();
    };
}
