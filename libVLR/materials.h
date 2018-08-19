#pragma once

#include "context.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Miscellaneous

    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        optix::Buffer m_PMF;
        optix::Buffer m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        void getInternalType(Shared::DiscreteDistribution1DTemplate<RealType>* instance);
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution1DTemplate {
        optix::Buffer m_PDF;
        optix::Buffer m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        RealType getIntegral() const { return m_integral; }

        void getInternalType(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance);
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        optix::Buffer m_1DDists;
        uint32_t m_num1DDists;
        RealType m_integral;
        RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

    public:
        void initialize(Context &context, const RealType* values, size_t numD1, size_t numD2);
        void finalize(Context &context);

        void getInternalType(Shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance);
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // Material

    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    struct RGBA16Fx4 { uint16_t/*half*/ r, g, b, a; };
    struct RGBA32Fx4 { float r, g, b, a; };
    struct Gray8 { uint8_t v; };

    extern const size_t sizesOfDataFormats[(uint32_t)DataFormat::Num];

    class Image2D : public Object {
        uint32_t m_width, m_height;
        DataFormat m_dataFormat;
        optix::Buffer m_optixDataBuffer;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static DataFormat getInternalFormat(DataFormat inputFormat);

        Image2D(Context &context, uint32_t width, uint32_t height, DataFormat dataFormat);
        virtual ~Image2D();

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        uint32_t getStride() const {
            return (uint32_t)sizesOfDataFormats[(uint32_t)m_dataFormat];
        }

        const optix::Buffer &getOptiXObject() const {
            return m_optixDataBuffer;
        }
    };



    class LinearImage2D : public Image2D {
        std::vector<uint8_t> m_data;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat);
    };



    class FloatTexture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        FloatTexture(Context &context);
        virtual ~FloatTexture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class Float2Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Float2Texture(Context &context);
        virtual ~Float2Texture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class Float3Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Float3Texture(Context &context);
        virtual ~Float3Texture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class ConstantFloat3Texture : public Float3Texture {
        Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ConstantFloat3Texture(Context &context, const float value[3]);
        ~ConstantFloat3Texture();
    };



    class ImageFloat3Texture : public Float3Texture {
        const Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ImageFloat3Texture(Context &context, const Image2D* image);
    };



    class Float4Texture : public Object {
    protected:
        optix::TextureSampler m_optixTextureSampler;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Float4Texture(Context &context);
        virtual ~Float4Texture();

        const optix::TextureSampler &getOptiXObject() const {
            return m_optixTextureSampler;
        }

        void setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping);
    };



    class ConstantFloat4Texture : public Float4Texture {
        Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ConstantFloat4Texture(Context &context, const float value[4]);
        ~ConstantFloat4Texture();
    };



    class ImageFloat4Texture : public Float4Texture {
        const Image2D* m_image;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ImageFloat4Texture(Context &context, const Image2D* image);
    };



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

        optix::Material m_optixMaterial;
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

        optix::Material &getOptiXObject() {
            return m_optixMaterial;
        }
        uint32_t getMaterialIndex() const {
            return m_matIndex;
        }

        virtual uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const = 0;
        virtual bool isEmitting() const { return false; }
    };



    class MatteSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float4Texture* m_texAlbedoRoughness;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        MatteSurfaceMaterial(Context &context, const Float4Texture* texAlbedoRoughness);
        ~MatteSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class SpecularReflectionSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texCoeffR;
        const Float3Texture* m_texEta;
        const Float3Texture* m_tex_k;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularReflectionSurfaceMaterial(Context &context, const Float3Texture* texCoeffR, const Float3Texture* texEta, const Float3Texture* tex_k);
        ~SpecularReflectionSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class SpecularScatteringSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texCoeff;
        const Float3Texture* m_texEtaExt;
        const Float3Texture* m_texEtaInt;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SpecularScatteringSurfaceMaterial(Context &context, const Float3Texture* texCoeff, const Float3Texture* texEtaExt, const Float3Texture* texEtaInt);
        ~SpecularScatteringSurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class UE4SurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texBaseColor;
        const Float3Texture* m_texOcclusionRoughnessMetallic;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context, const Float3Texture* texBaseColor, const Float3Texture* texOcclusionRoughnessMetallic);
        ~UE4SurfaceMaterial();

        uint32_t setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const override;
    };



    class DiffuseEmitterSurfaceMaterial : public SurfaceMaterial {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        const Float3Texture* m_texEmittance;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        DiffuseEmitterSurfaceMaterial(Context &context, const Float3Texture* texEmittance);
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

    // END: Material
    // ----------------------------------------------------------------
}
