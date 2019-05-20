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
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

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

        ShaderNodeSocket m_nodeAlbedo;
        TripletSpectrum m_immAlbedo;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context);
        ~MatteSurfaceMaterial();

        bool setAlbedo(const ShaderNodeSocket &outputSocket);
        void setAlbedo(ColorSpace colorSpace, float e0, float e1, float e2);
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeffR;
        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;
        TripletSpectrum m_immCoeffR;
        TripletSpectrum m_immEta;
        TripletSpectrum m_imm_k;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context);
        ~SpecularReflectionSurfaceMaterial();

        bool setCoeffR(const ShaderNodeSocket &outputSocket);
        void setCoeffR(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setEta(const ShaderNodeSocket &outputSocket);
        void setEta(ColorSpace colorSpace, float e0, float e1, float e2);
        bool set_k(const ShaderNodeSocket &outputSocket);
        void set_k(ColorSpace colorSpace, float e0, float e1, float e2);
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;
        TripletSpectrum m_immCoeff;
        TripletSpectrum m_immEtaExt;
        TripletSpectrum m_immEtaInt;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context);
        ~SpecularScatteringSurfaceMaterial();

        bool setCoeff(const ShaderNodeSocket &outputSocket);
        void setCoeff(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setEtaExt(const ShaderNodeSocket &outputSocket);
        void setEtaExt(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setEtaInt(const ShaderNodeSocket &outputSocket);
        void setEtaInt(ColorSpace colorSpace, float e0, float e1, float e2);
    };



    class MicrofacetReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;
        ShaderNodeSocket m_nodeRoughnessAnisotropyRotation;
        TripletSpectrum m_immEta;
        TripletSpectrum m_imm_k;
        float m_immRoughness;
        float m_immAnisotropy;
        float m_immRotation;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        MicrofacetReflectionSurfaceMaterial(Context &context);
        ~MicrofacetReflectionSurfaceMaterial();

        bool setEta(const ShaderNodeSocket &outputSocket);
        void setEta(ColorSpace colorSpace, float e0, float e1, float e2);
        bool set_k(const ShaderNodeSocket &outputSocket);
        void set_k(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setRoughnessAnisotropyRotation(const ShaderNodeSocket &outputSocket);
        void setRoughness(float value);
        void setAnisotropy(float value);
        void setRotation(float value);
    };



    class MicrofacetScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;
        ShaderNodeSocket m_nodeRoughnessAnisotropyRotation;
        TripletSpectrum m_immCoeff;
        TripletSpectrum m_immEtaExt;
        TripletSpectrum m_immEtaInt;
        float m_immRoughness;
        float m_immAnisotropy;
        float m_immRotation;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        MicrofacetScatteringSurfaceMaterial(Context &context);
        ~MicrofacetScatteringSurfaceMaterial();

        bool setCoeff(const ShaderNodeSocket &outputSocket);
        void setCoeff(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setEtaExt(const ShaderNodeSocket &outputSocket);
        void setEtaExt(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setEtaInt(const ShaderNodeSocket &outputSocket);
        void setEtaInt(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setRoughnessAnisotropyRotation(const ShaderNodeSocket &outputSocket);
        void setRoughness(float value);
        void setAnisotropy(float value);
        void setRotation(float value);
    };



    class LambertianScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeF0;
        TripletSpectrum m_immCoeff;
        float m_immF0;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        LambertianScatteringSurfaceMaterial(Context &context);
        ~LambertianScatteringSurfaceMaterial();

        bool setCoeff(const ShaderNodeSocket &outputSocket);
        void setCoeff(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setF0(const ShaderNodeSocket &outputSocket);
        void setF0(float value);
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeBaseColor;
        ShaderNodeSocket m_nodeOcclusionRoughnessMetallic;
        TripletSpectrum m_immBaseColor;
        float m_immOcculusion;
        float m_immRoughness;
        float m_immMetallic;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context);
        ~UE4SurfaceMaterial();

        bool setBaseColor(const ShaderNodeSocket &outputSocket);
        void setBaseColor(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setOcclusionRoughnessMetallic(const ShaderNodeSocket &outputSocket);
        void setOcclusion(float value);
        void setRoughness(float value);
        void setMetallic(float value);
    };



    class OldStyleSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeDiffuseColor;
        ShaderNodeSocket m_nodeSpecularColor;
        ShaderNodeSocket m_nodeGlossiness;
        TripletSpectrum m_immDiffuseColor;
        TripletSpectrum m_immSpecularColor;
        float m_immGlossiness;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        OldStyleSurfaceMaterial(Context &context);
        ~OldStyleSurfaceMaterial();

        bool setDiffuseColor(const ShaderNodeSocket &outputSocket);
        void setDiffuseColor(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setSpecularColor(const ShaderNodeSocket &outputSocket);
        void setSpecularColor(ColorSpace colorSpace, float e0, float e1, float e2);
        bool setGlossiness(const ShaderNodeSocket &outputSocket);
        void setGlossiness(float value);
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeEmittance;
        TripletSpectrum m_immEmittance;
        float m_immScale;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        DiffuseEmitterSurfaceMaterial(Context &context);
        ~DiffuseEmitterSurfaceMaterial();

        bool isEmitting() const override { return true; }

        bool setEmittance(const ShaderNodeSocket &outputSocket);
        void setEmittance(ColorSpace colorSpace, float e0, float e1, float e2);
        void setScale(float value);
    };



    class MultiSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const SurfaceMaterial* m_subMaterials[4];
        uint32_t m_numSubMaterials;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

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
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentEmitterSurfaceMaterial(Context &context);
        ~EnvironmentEmitterSurfaceMaterial();

        bool isEmitting() const override { return true; }

        bool setEmittanceTextured(const EnvironmentTextureShaderNode* node);
        bool setEmittanceConstant(const ShaderNode* spectrumNode);
        void setEmittance(ColorSpace colorSpace, float e0, float e1, float e2);
        void setScale(float value);

        const RegularConstantContinuousDistribution2D &getImportanceMap();
    };
}
