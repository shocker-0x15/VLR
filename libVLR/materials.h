#pragma once

#include "shader_nodes.h"

namespace vlr {
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
            return vlr::createTripletSpectrum(spectrumType, colorSpace, e0, e1, e2);
        }
    };



    class SurfaceMaterial : public Queryable {
        enum BSDFCallableName {
            BSDFCallableName_setupBSDF = 0,
            BSDFCallableName_BSDF_getBaseColor,
            BSDFCallableName_BSDF_matches,
            BSDFCallableName_BSDF_sampleInternal,
            BSDFCallableName_BSDF_sampleWithRevInternal,
            BSDFCallableName_BSDF_evaluateInternal,
            BSDFCallableName_BSDF_evaluateWithRevInternal,
            BSDFCallableName_BSDF_evaluatePDFInternal,
            BSDFCallableName_BSDF_evaluatePDFWithRevInternal,
            BSDFCallableName_BSDF_weightInternal,
            NumBSDFCallableNames
        };
        enum EDFCallableName {
            EDFCallableName_setupEDF,
            EDFCallableName_EDF_matches,
            EDFCallableName_EDF_sampleInternal,
            EDFCallableName_EDF_evaluateEmittanceInternal,
            EDFCallableName_EDF_evaluateInternal,
            EDFCallableName_EDF_evaluatePDFInternal,
            EDFCallableName_EDF_weightInternal,
            EDFCallableName_EDF_as_BSDF_getBaseColor,
            EDFCallableName_EDF_as_BSDF_matches,
            EDFCallableName_EDF_as_BSDF_sampleInternal,
            EDFCallableName_EDF_as_BSDF_sampleWithRevInternal,
            EDFCallableName_EDF_as_BSDF_evaluateInternal,
            EDFCallableName_EDF_as_BSDF_evaluateWithRevInternal,
            EDFCallableName_EDF_as_BSDF_evaluatePDFInternal,
            EDFCallableName_EDF_as_BSDF_evaluatePDFWithRevInternal,
            EDFCallableName_EDF_as_BSDF_weightInternal,
            NumEDFCallableNames
        };

    protected:
        struct OptiXProgramSet {
            uint32_t dcSetupBSDF;
            uint32_t dcBSDFGetBaseColor;
            uint32_t dcBSDFmatches;
            uint32_t dcBSDFSampleInternal;
            uint32_t dcBSDFSampleWithRevInternal;
            uint32_t dcBSDFEvaluateInternal;
            uint32_t dcBSDFEvaluateWithRevInternal;
            uint32_t dcBSDFEvaluatePDFInternal;
            uint32_t dcBSDFEvaluatePDFWithRevInternal;
            uint32_t dcBSDFWeightInternal;
            uint32_t bsdfProcedureSetIndex;

            uint32_t dcSetupEDF;
            uint32_t dcEDFmatches;
            uint32_t dcEDFSampleInternal;
            uint32_t dcEDFEvaluateEmittanceInternal;
            uint32_t dcEDFEvaluateInternal;
            uint32_t dcEDFEvaluatePDFInternal;
            uint32_t dcEDFWeightInternal;
            uint32_t dcEDFAsBSDFGetBaseColor;
            uint32_t dcEDFAsBSDFmatches;
            uint32_t dcEDFAsBSDFSampleInternal;
            uint32_t dcEDFAsBSDFSampleWithRevInternal;
            uint32_t dcEDFAsBSDFEvaluateInternal;
            uint32_t dcEDFAsBSDFEvaluateWithRevInternal;
            uint32_t dcEDFAsBSDFEvaluatePDFInternal;
            uint32_t dcEDFAsBSDFEvaluatePDFWithRevInternal;
            uint32_t dcEDFAsBSDFWeightInternal;
            uint32_t edfProcedureSetIndex;

            OptiXProgramSet() {
                std::memset(this, 0, sizeof(*this));
            }
        };

        uint32_t m_matIndex;

        static void commonInitializeProcedure(
            Context &context,
            const char* bsdfIDs[NumBSDFCallableNames], const char* edfIDs[NumEDFCallableNames],
            OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(
            Context &context, OptiXProgramSet &programSet);
        static void setupMaterialDescriptorHead(
            Context &context, const OptiXProgramSet &progSet, shared::SurfaceMaterialDescriptor* matDesc);

        virtual void setupMaterialDescriptor(CUstream stream) const = 0;

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

        void setup(CUstream stream) const {
            setupMaterialDescriptor(stream);
        }
    };



    class MatteSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeAlbedo;
        ImmediateSpectrum m_immAlbedo;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeEta;
        ShaderNodePlug m_node_k;
        ImmediateSpectrum m_immCoeff;
        ImmediateSpectrum m_immEta;
        ImmediateSpectrum m_imm_k;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeEtaExt;
        ShaderNodePlug m_nodeEtaInt;
        ImmediateSpectrum m_immCoeff;
        ImmediateSpectrum m_immEtaExt;
        ImmediateSpectrum m_immEtaInt;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeEta;
        ShaderNodePlug m_node_k;
        ShaderNodePlug m_nodeRoughnessAnisotropyRotation;
        ImmediateSpectrum m_immEta;
        ImmediateSpectrum m_imm_k;
        float m_immRoughness;
        float m_immAnisotropy;
        float m_immRotation;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

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

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeCoeff;
        ShaderNodePlug m_nodeF0;
        ImmediateSpectrum m_immCoeff;
        float m_immF0;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeBaseColor;
        ShaderNodePlug m_nodeOcclusionRoughnessMetallic;
        ImmediateSpectrum m_immBaseColor;
        float m_immOcculusion;
        float m_immRoughness;
        float m_immMetallic;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeDiffuseColor;
        ShaderNodePlug m_nodeSpecularColor;
        ShaderNodePlug m_nodeGlossiness;
        ImmediateSpectrum m_immDiffuseColor;
        ImmediateSpectrum m_immSpecularColor;
        float m_immGlossiness;

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeEmittance;
        ImmediateSpectrum m_immEmittance;
        float m_immScale;

        void setupMaterialDescriptor(CUstream stream) const override;

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



    class DirectionalEmitterSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeEmittance;
        ImmediateSpectrum m_immEmittance;
        float m_immScale;
        ShaderNodePlug m_nodeDirection;
        Vector3D m_immDirection;

        void setupMaterialDescriptor(CUstream stream) const override;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        DirectionalEmitterSurfaceMaterial(Context &context);
        ~DirectionalEmitterSurfaceMaterial();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, Vector3D* dir) const override;
        bool get(const char* paramName, ImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const Vector3D& dir) override;
        bool set(const char* paramName, const ImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        bool isEmitting() const override { return true; }
    };



    class PointEmitterSurfaceMaterial : public SurfaceMaterial {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeIntensity;
        ImmediateSpectrum m_immIntensity;
        float m_immScale;

        void setupMaterialDescriptor(CUstream stream) const override;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        PointEmitterSurfaceMaterial(Context &context);
        ~PointEmitterSurfaceMaterial();

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        const SurfaceMaterial* m_subMaterials[4];

        void setupMaterialDescriptor(CUstream stream) const override;

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

        static std::unordered_map<uint32_t, OptiXProgramSet> s_optiXProgramSets;

        ShaderNodePlug m_nodeEmittance;
        ImmediateSpectrum m_immEmittance;
        RegularConstantContinuousDistribution2D m_importanceMap;
        float m_immScale;

        void setupMaterialDescriptor(CUstream stream) const override;

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
