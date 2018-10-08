#pragma once

#include "context.h"
#include "textures.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Material

    class SurfaceMaterial : public Object {
    protected:
        struct OptiXProgramSet {
            optix::Program callableProgramSetupBSDF;
            optix::Program callableProgramGetBaseColor;
            optix::Program callableProgramBSDFmatches;
            optix::Program callableProgramSampleBSDFInternal;
            optix::Program callableProgramEvaluateBSDFInternal;
            optix::Program callableProgramEvaluateBSDF_PDFInternal;
            optix::Program callableProgramBSDFWeightInternal;
            uint32_t bsdfProcedureSetIndex;

            optix::Program callableProgramSetupEDF;
            optix::Program callableProgramEvaluateEmittanceInternal;
            optix::Program callableProgramEvaluateEDFInternal;
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

        const Float4Texture* m_texAlbedoRoughness;
        const TextureMap2D* m_texMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context, const Float4Texture* texAlbedoRoughness, const TextureMap2D* texMap);
        ~MatteSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texCoeffR;
        const Float3Texture* m_texEta;
        const Float3Texture* m_tex_k;
        const TextureMap2D* m_texMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context, const Float3Texture* texCoeffR, const Float3Texture* texEta, const Float3Texture* tex_k, const TextureMap2D* texMap);
        ~SpecularReflectionSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texCoeff;
        const Float3Texture* m_texEtaExt;
        const Float3Texture* m_texEtaInt;
        const TextureMap2D* m_texMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context, const Float3Texture* texCoeff, const Float3Texture* texEtaExt, const Float3Texture* texEtaInt, const TextureMap2D* texMap);
        ~SpecularScatteringSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texBaseColor;
        const Float3Texture* m_texOcclusionRoughnessMetallic;
        const TextureMap2D* m_texMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context, const Float3Texture* texBaseColor, const Float3Texture* texOcclusionRoughnessMetallic, const TextureMap2D* texMap);
        ~UE4SurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texEmittance;
        const TextureMap2D* m_texMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        DiffuseEmitterSurfaceMaterial(Context &context, const Float3Texture* texEmittance, const TextureMap2D* texMap);
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

        const Float3Texture* m_texEmittance;
        RegularConstantContinuousDistribution2D m_importanceMap;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentEmitterSurfaceMaterial(Context &context, const Float3Texture* texEmittance);
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
