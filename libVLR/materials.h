#pragma once

#include "shader_nodes.h"

namespace VLR {
    struct ImmediateSpectrum {
        ColorSpace colorSpace;
        float e0;
        float e1;
        float e2;

        ImmediateSpectrum() :
            colorSpace(ColorSpace::Rec709_D65), e0(1.0f), e1(0.0f), e2(1.0f) {}
        ImmediateSpectrum(ColorSpace _colorSpace, float _e0, float _e1, float _e2) :
            colorSpace(_colorSpace), e0(_e0), e1(_e1), e2(_e2) {}
        ImmediateSpectrum(const VLRImmediateSpectrum& spectrum) {
            auto v = getEnumValueFromMember<ColorSpace>(spectrum.colorSpace);
            if (v != (ColorSpace)0xFFFFFFFF) {
                colorSpace = v;
                e0 = spectrum.e0;
                e1 = spectrum.e1;
                e2 = spectrum.e2;
            }
            else {
                colorSpace = ColorSpace::Rec709_D65;
                e0 = 1.0f;
                e1 = 0.0f;
                e2 = 1.0f;
            }
        }
        VLRImmediateSpectrum getPublicType() const {
            VLRImmediateSpectrum ret;
            ret.colorSpace = getEnumMemberFromValue(colorSpace);
            ret.e0 = e0;
            ret.e1 = e1;
            ret.e2 = e2;
            return ret;
        }
        TripletSpectrum createTripletSpectrum(SpectrumType spectrumType) const {
            return VLR::createTripletSpectrum(spectrumType, colorSpace, e0, e1, e2);
        }
    };



    class SurfaceMaterial : public Queryable {
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

        static std::string s_materials_ptx;
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
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeAlbedo;
        ImmediateSpectrum m_immAlbedo;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context);
        ~MatteSurfaceMaterial();

        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const ImmediateSpectrum &spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeEta;
        ShaderNodePlug m_node_k;
        ImmediateSpectrum m_immCoeff;
        ImmediateSpectrum m_immEta;
        ImmediateSpectrum m_imm_k;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context);
        ~SpecularReflectionSurfaceMaterial();

        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeEtaExt;
        ShaderNodePlug m_nodeEtaInt;
        ImmediateSpectrum m_immCoeff;
        ImmediateSpectrum m_immEtaExt;
        ImmediateSpectrum m_immEtaInt;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context);
        ~SpecularScatteringSurfaceMaterial();

        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class MicrofacetReflectionSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeEta;
        ShaderNodePlug m_node_k;
        ShaderNodePlug m_nodeRoughnessAnisotropyRotation;
        ImmediateSpectrum m_immEta;
        ImmediateSpectrum m_imm_k;
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

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum &spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class MicrofacetScatteringSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeEtaExt;
        ShaderNodePlug m_nodeEtaInt;
        ShaderNodePlug m_nodeRoughnessAnisotropyRotation;
        ImmediateSpectrum m_immCoeff;
        ImmediateSpectrum m_immEtaExt;
        ImmediateSpectrum m_immEtaInt;
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

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class LambertianScatteringSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeF0;
        ImmediateSpectrum m_immCoeff;
        float m_immF0;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        LambertianScatteringSurfaceMaterial(Context &context);
        ~LambertianScatteringSurfaceMaterial();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeBaseColor;
        ShaderNodePlug m_nodeOcclusionRoughnessMetallic;
        ImmediateSpectrum m_immBaseColor;
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

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class OldStyleSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeDiffuseColor;
        ShaderNodePlug m_nodeSpecularColor;
        ShaderNodePlug m_nodeGlossiness;
        ImmediateSpectrum m_immDiffuseColor;
        ImmediateSpectrum m_immSpecularColor;
        float m_immGlossiness;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        OldStyleSurfaceMaterial(Context &context);
        ~OldStyleSurfaceMaterial();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeEmittance;
        ImmediateSpectrum m_immEmittance;
        float m_immScale;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        DiffuseEmitterSurfaceMaterial(Context &context);
        ~DiffuseEmitterSurfaceMaterial();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        bool isEmitting() const override { return true; }
    };



    class MultiSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const SurfaceMaterial* m_subMaterials[4];

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        MultiSurfaceMaterial(Context &context);
        ~MultiSurfaceMaterial();

        bool get(const char* paramName, const SurfaceMaterial** material) const override;

        bool set(const char* paramName, const SurfaceMaterial* material) override;

        bool isEmitting() const override;
    };



    class EnvironmentEmitterSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodePlug m_nodeEmittance;
        ImmediateSpectrum m_immEmittance;
        RegularConstantContinuousDistribution2D m_importanceMap;
        float m_immScale;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentEmitterSurfaceMaterial(Context &context);
        ~EnvironmentEmitterSurfaceMaterial();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        bool isEmitting() const override { return true; }

        const RegularConstantContinuousDistribution2D &getImportanceMap();
    };
}
