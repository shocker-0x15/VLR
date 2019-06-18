#pragma once

#include "shader_nodes.h"

namespace VLR {
#define VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS() \
    static std::vector<ParameterInfo> ParameterInfos; \
    const std::vector<ParameterInfo>& getParamInfos() const override { \
        return ParameterInfos; \
    }



    struct ImmediateSpectrum {
        ColorSpace colorSpace;
        float e0;
        float e1;
        float e2;

        ImmediateSpectrum(ColorSpace _colorSpace, float _e0, float _e1, float _e2) :
            colorSpace(_colorSpace), e0(_e0), e1(_e1), e2(_e2) {}
        ImmediateSpectrum(const VLRImmediateSpectrum& spectrum) {
            const auto &table = g_enumTables.at(ParameterColorSpace);
            if (table.count(spectrum.colorSpace) > 0) {
                colorSpace = (ColorSpace)table.at(spectrum.colorSpace);
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
            const auto& table = g_invEnumTables.at(ParameterColorSpace);
            ret.colorSpace = table.at((uint32_t)colorSpace).c_str();
            ret.e0 = e0;
            ret.e1 = e1;
            ret.e2 = e2;
            return ret;
        }
        TripletSpectrum createTripletSpectrum(SpectrumType spectrumType) const {
            return VLR::createTripletSpectrum(spectrumType, colorSpace, e0, e1, e2);
        }
    };



    class SurfaceMaterial : public Object {
    protected:
        virtual const std::vector<ParameterInfo>& getParamInfos() const = 0;

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

        virtual bool get(const char* paramName, const char** enumValue) const {
            return false;
        };
        virtual bool get(const char* paramName, float* values, uint32_t length) const {
            return false;
        }
        virtual bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const {
            return false;
        }
        virtual bool get(const char* paramName, const SurfaceMaterial** material) const {
            return false;
        }
        virtual bool get(const char* paramName, ShaderNodeSocket* socket) const {
            return false;
        }

        virtual bool set(const char* paramName, const char* enumValue) {
            return false;
        };
        virtual bool set(const char* paramName, const float* values, uint32_t length) {
            return false;
        }
        virtual bool set(const char* paramName, const VLRImmediateSpectrum &spectrum) {
            return false;
        }
        virtual bool set(const char* paramName, const SurfaceMaterial* material) {
            return false;
        }
        virtual bool set(const char* paramName, const ShaderNodeSocket& socket) {
            return false;
        }

        uint32_t getNumParameters() const {
            auto paramInfos = getParamInfos();
            return (uint32_t)paramInfos.size();
        }
        const ParameterInfo* getParameterInfo(uint32_t index) const {
            auto paramInfos = getParamInfos();
            if (index < paramInfos.size())
                return &paramInfos[index];
            return nullptr;
        }

        uint32_t getMaterialIndex() const {
            return m_matIndex;
        }

        virtual bool isEmitting() const { return false; }
    };



    class MatteSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeAlbedo;
        ImmediateSpectrum m_immAlbedo;

        void setupMaterialDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context);
        ~MatteSurfaceMaterial();

        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const VLRImmediateSpectrum &spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;
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

        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;
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

        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class MicrofacetReflectionSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeEta;
        ShaderNodeSocket m_node_k;
        ShaderNodeSocket m_nodeRoughnessAnisotropyRotation;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum &spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class MicrofacetScatteringSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeEtaExt;
        ShaderNodeSocket m_nodeEtaInt;
        ShaderNodeSocket m_nodeRoughnessAnisotropyRotation;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class LambertianScatteringSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeCoeff;
        ShaderNodeSocket m_nodeF0;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeBaseColor;
        ShaderNodeSocket m_nodeOcclusionRoughnessMetallic;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class OldStyleSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeDiffuseColor;
        ShaderNodeSocket m_nodeSpecularColor;
        ShaderNodeSocket m_nodeGlossiness;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeEmittance;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;

        bool isEmitting() const override { return true; }
    };



    class MultiSurfaceMaterial : public SurfaceMaterial {
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

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
        VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_nodeEmittance;
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
        bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket) override;

        bool isEmitting() const override { return true; }

        const RegularConstantContinuousDistribution2D &getImportanceMap();
    };
}

#undef VLR_SURFACE_MATERIAL_DECLARE_PARAMETER_INFOS
