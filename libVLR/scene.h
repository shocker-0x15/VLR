#pragma once

#include <VLR.h>
#include "basic_types_internal.h"
#include "shared.h"

#include "slot_manager.h"

namespace VLR {
    class Context {
        static uint32_t NextID;
        static uint32_t getInstanceID() {
            return NextID++;
        }

        uint32_t m_ID;
        optix::Context m_optixContext;

        optix::Program m_optixCallableProgramNullFetchAlpha;
        optix::Program m_optixCallableProgramNullFetchNormal;
        optix::Program m_optixCallableProgramFetchAlpha;
        optix::Program m_optixCallableProgramFetchNormal;

        optix::Program m_optixProgramStochasticAlphaAnyHit; // -- Any Hit Program
        optix::Program m_optixProgramAlphaAnyHit; // ------------ Any Hit Program
        optix::Program m_optixProgramPathTracingIteration; // --- Closest Hit Program

        optix::Program m_optixProgramPathTracing; // ------------ Ray Generation Program
        optix::Program m_optixProgramPathTracingMiss; // -------- Miss Program
        optix::Program m_optixProgramException; // -------------- Exception Program

        optix::Buffer m_optixBSDFProcedureSetBuffer;
        uint32_t m_maxNumBSDFProcSet;
        SlotManager m_bsdfProcSetSlotManager;

        optix::Buffer m_optixEDFProcedureSetBuffer;
        uint32_t m_maxNumEDFProcSet;
        SlotManager m_edfProcSetSlotManager;

        optix::Program m_optixCallableProgramNullBSDF_setupBSDF;
        optix::Program m_optixCallableProgramNullBSDF_getBaseColor;
        optix::Program m_optixCallableProgramNullBSDF_matches;
        optix::Program m_optixCallableProgramNullBSDF_sampleBSDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_evaluateBSDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_evaluateBSDF_PDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_weightInternal;
        uint32_t m_nullBSDFProcedureSetIndex;

        optix::Program m_optixCallableProgramNullEDF_setupEDF;
        optix::Program m_optixCallableProgramNullEDF_evaluateEmittanceInternal;
        optix::Program m_optixCallableProgramNullEDF_evaluateEDFInternal;
        uint32_t m_nullEDFProcedureSetIndex;

        optix::Buffer m_optixSurfaceMaterialDescriptorBuffer;
        uint32_t m_maxNumSurfaceMaterialDescriptors;
        SlotManager m_surfMatDescSlotManager;

        optix::Buffer m_outputBuffer;
        optix::Buffer m_rngBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numAccumFrames;

    public:
        Context();
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        void bindOpenGLBuffer(uint32_t bufferID, uint32_t width, uint32_t height);

        void render(Scene &scene, Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

        const optix::Context &getOptiXContext() const {
            return m_optixContext;
        }

        const optix::Program &getOptiXCallableProgramNullFetchAlpha() const {
            return m_optixCallableProgramNullFetchAlpha;
        }
        const optix::Program &getOptiXCallableProgramNullFetchNormal() const {
            return m_optixCallableProgramNullFetchNormal;
        }
        const optix::Program &getOptiXCallableProgramFetchAlpha() const {
            return m_optixCallableProgramFetchAlpha;
        }
        const optix::Program &getOptiXCallableProgramFetchNormal() const {
            return m_optixCallableProgramFetchNormal;
        }

        // JP: 全マテリアルが共通のClosest Hit, Any Hit Programをバインドする。
        const optix::Program &getOptiXProgramStochasticAlphaAnyHit() const {
            return m_optixProgramStochasticAlphaAnyHit;
        }
        const optix::Program &getOptiXProgramAlphaAnyHit() const {
            return m_optixProgramAlphaAnyHit;
        }
        const optix::Program &getOptiXProgramPathTracingIteration() const {
            return m_optixProgramPathTracingIteration;
        }

        uint32_t setBSDFProcedureSet(const Shared::BSDFProcedureSet &procSet);
        void unsetBSDFProcedureSet(uint32_t index);

        uint32_t setEDFProcedureSet(const Shared::EDFProcedureSet &procSet);
        void unsetEDFProcedureSet(uint32_t index);

        const optix::Program &getOptixCallableProgramNullBSDF_setupBSDF() const {
            return m_optixCallableProgramNullBSDF_setupBSDF;
        }
        uint32_t getNullBSDFProcedureSetIndex() const { return m_nullBSDFProcedureSetIndex; }
        const optix::Program &getOptixCallableProgramNullEDF_setupEDF() const {
            return m_optixCallableProgramNullEDF_setupEDF;
        }
        uint32_t getNullEDFProcedureSetIndex() const { return m_nullEDFProcedureSetIndex; }

        uint32_t setSurfaceMaterialDescriptor(const Shared::SurfaceMaterialDescriptor &matDesc);
        void unsetSurfaceMaterialDescriptor(uint32_t index);
    };



    class ClassIdentifier {
        ClassIdentifier &operator=(const ClassIdentifier &) = delete;

        const ClassIdentifier* m_baseClass;

    public:
        ClassIdentifier(const ClassIdentifier* baseClass) : m_baseClass(baseClass) {}

        const ClassIdentifier* getBaseClass() const {
            return m_baseClass;
        }
    };

    class Object {
    protected:
        Context &m_context;

    public:
        Object(Context &context);
        virtual ~Object() {}

        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        template <class T>
        bool is() const {
            return &getClass() == &T::ClassID;
        }

        template <class T>
        bool isMemberOf() const {
            const ClassIdentifier* curClass = &getClass();
            while (curClass) {
                if (curClass == &T::ClassID)
                    return true;
                curClass = curClass->getBaseClass();
            }
            return false;
        }

        Context &getContext() {
            return m_context;
        }
    };



    // ----------------------------------------------------------------
    // Shallow Hierarchy

    class SHGroup;
    class SHTransform;
    class SHGeometryGroup;
    class SHGeometryInstance;

    class SHGroup {
        optix::Group m_optixGroup;
        optix::Acceleration m_optixAcceleration;
        struct TransformStatus {
            bool hasGeometryDescendant;
        };
        std::map<const SHTransform*, TransformStatus> m_transforms;
        uint32_t m_numValidTransforms;

    public:
        SHGroup(Context &context) : m_numValidTransforms(0) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixGroup = optixContext->createGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGroup->setAcceleration(m_optixAcceleration);
        }
        ~SHGroup() {
            m_optixAcceleration->destroy();
            m_optixGroup->destroy();
        }

        void addChild(SHTransform* transform);
        void removeChild(SHTransform* transform);
        void updateChild(SHTransform* transform);
        uint32_t getNumValidChildren() const {
            return m_numValidTransforms;
        }

        const optix::Group &getOptiXObject() const {
            return m_optixGroup;
        }

        void printOptiXHierarchy();
    };

    class SHTransform {
        std::string m_name;
        optix::Transform m_optixTransform;

        StaticTransform m_transform;
        union {
            const SHTransform* m_childTransform;
            SHGeometryGroup* m_childGeometryGroup;
        };
        bool m_childIsTransform;

        void resolveTransform();

    public:
        SHTransform(const std::string &name, Context &context, const StaticTransform &transform, const SHTransform* childTransform) :
            m_name(name), m_transform(transform), m_childTransform(childTransform), m_childIsTransform(childTransform != nullptr) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixTransform = optixContext->createTransform();

            resolveTransform();
        }
        ~SHTransform() {
            m_optixTransform->destroy();
        }

        const std::string &getName() const { return m_name; }
        void setName(const std::string &name) {
            m_name = name;
        }

        void setTransform(const StaticTransform &transform);
        void update();
        bool isStatic() const;
        StaticTransform getStaticTransform() const;

        void setChild(SHGeometryGroup* geomGroup);
        bool hasGeometryDescendant(SHGeometryGroup** descendant = nullptr) const;

        const optix::Transform &getOptiXObject() const {
            return m_optixTransform;
        }
    };

    class SHGeometryGroup {
        optix::GeometryGroup m_optixGeometryGroup;
        optix::Acceleration m_optixAcceleration;
        std::set<const SHGeometryInstance*> m_instances;

    public:
        SHGeometryGroup(Context &context) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixGeometryGroup = optixContext->createGeometryGroup();
            m_optixAcceleration = optixContext->createAcceleration("Trbvh");
            m_optixGeometryGroup->setAcceleration(m_optixAcceleration);
        }
        ~SHGeometryGroup() {
            m_optixAcceleration->destroy();
            m_optixGeometryGroup->destroy();
        }

        void addGeometryInstance(const SHGeometryInstance* instance);
        void removeGeometryInstance(const SHGeometryInstance* instance);
        const SHGeometryInstance* getGeometryInstanceAt(uint32_t index) const {
            auto it = m_instances.cbegin();
            std::advance(it, index);
            return *it;
        }
        uint32_t getNumInstances() const {
            return (uint32_t)m_instances.size();
        }

        const optix::GeometryGroup &getOptiXObject() const {
            return m_optixGeometryGroup;
        }
    };

    class SHGeometryInstance {
        optix::GeometryInstance m_optixGeometryInstance;
        Shared::SurfaceLightDescriptor m_surfaceLightDescriptor;

    public:
        SHGeometryInstance(Context &context, const Shared::SurfaceLightDescriptor &lightDesc) : m_surfaceLightDescriptor(lightDesc) {
            optix::Context optixContext = context.getOptiXContext();
            m_optixGeometryInstance = optixContext->createGeometryInstance();
        }
        ~SHGeometryInstance() {
            m_optixGeometryInstance->destroy();
        }

        void getSurfaceLightDescriptor(Shared::SurfaceLightDescriptor* lightDesc) const {
            *lightDesc = m_surfaceLightDescriptor;
        }

        const optix::GeometryInstance &getOptiXObject() const {
            return m_optixGeometryInstance;
        }
    };

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    class Node;
    class ParentNode;
    class RootNode;



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
        const Float2Texture* m_texRoughnessMetallic;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        UE4SurfaceMaterial(Context &context, const Float3Texture* texBaseColor, const Float2Texture* texRoughnessMetallic);
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



    class Node : public Object {
    protected:
        std::string m_name;
    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Node(Context &context, const std::string &name) :
            Object(context), m_name(name) {}
        virtual ~Node() {}

        virtual void setName(const std::string &name) {
            m_name = name;
        }
        const std::string &getName() const {
            return m_name;
        }
    };



    class SurfaceNode : public Node {
    protected:
        std::set<ParentNode*> m_parents;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        SurfaceNode(Context &context, const std::string &name) : Node(context, name) {}
        virtual ~SurfaceNode() {}

        virtual void addParent(ParentNode* parent);
        virtual void removeParent(ParentNode* parent);
    };



    class TriangleMeshSurfaceNode : public SurfaceNode {
        struct OptiXProgramSet {
            optix::Program programIntersectTriangle; // Intersection Program
            optix::Program programCalcBBoxForTriangle; // Bounding Box Program
            optix::Program callableProgramDecodeHitPointForTriangle;
            optix::Program callableProgramDecodeTexCoordForTriangle;
            optix::Program callableProgramSampleTriangleMesh;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        struct OptiXGeometry {
            std::vector<uint32_t> indices;
            optix::Buffer optixIndexBuffer;
            optix::Geometry optixGeometry;
            DiscreteDistribution1D primDist;
        };

        std::vector<Vertex> m_vertices;
        optix::Buffer m_optixVertexBuffer;
        std::vector<OptiXGeometry> m_optixGeometries;
        std::vector<SurfaceMaterial*> m_materials;
        std::vector<SHGeometryInstance*> m_shGeometryInstances;
    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        TriangleMeshSurfaceNode(Context &context, const std::string &name);
        ~TriangleMeshSurfaceNode();

        void addParent(ParentNode* parent) override;
        void removeParent(ParentNode* parent) override;

        void setVertices(std::vector<Vertex> &&vertices);
        void addMaterialGroup(std::vector<uint32_t> &&indices, SurfaceMaterial* material);
    };



    struct TransformAndGeometryInstance {
        const SHTransform* transform;
        const SHGeometryInstance* geomInstance;
    };


    
    class ParentNode : public Node {
    protected:
        std::set<Node*> m_children;
        const Transform* m_localToWorld;

        // key: child SHTransform
        // SHTransform containing only the self transform uses nullptr as the key.
        std::map<const SHTransform*, SHTransform*> m_shTransforms;

        SHGeometryGroup m_shGeomGroup;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        ParentNode(Context &context, const std::string &name, const Transform* localToWorld);
        virtual ~ParentNode();

        void setName(const std::string &name) override;

        enum class UpdateEvent {
            TransformAdded = 0,
            TransformRemoved,
            TransformUpdated,
            GeometryAdded,
            GeometryRemoved,
        };

        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*> &childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) = 0;
        virtual void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) = 0;
        virtual void setTransform(const Transform* localToWorld);
        const Transform* getTransform() const {
            return m_localToWorld;
        }

        void addChild(InternalNode* child);
        void addChild(SurfaceNode* child);
        void removeChild(InternalNode* child);
        void removeChild(SurfaceNode* child);
    };



    class InternalNode : public ParentNode {
        std::set<ParentNode*> m_parents;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        InternalNode(Context &context, const std::string &name, const Transform* localToWorld);

        void setTransform(const Transform* localToWorld) override;

        void addParent(ParentNode* parent);
        void removeParent(ParentNode* parent);
    };



    class RootNode : public ParentNode {
        SHGroup m_shGroup;
        std::map<const SHGeometryInstance*, Shared::SurfaceLightDescriptor> m_surfaceLights;
        optix::Buffer m_optixSurfaceLightDescriptorBuffer;
        DiscreteDistribution1D m_surfaceLightImpDist;
        bool m_surfaceLightsAreSetup;

        void childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) override;
        void childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) override;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        RootNode(Context &context, const Transform* localToWorld);
        ~RootNode();

        void set();
    };



    class Scene : public Object {
        RootNode m_rootNode;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        Scene(Context &context, const Transform* localToWorld);

        void setTransform(const Transform* localToWorld) {
            m_rootNode.setTransform(localToWorld);
        }

        void addChild(InternalNode* child) {
            m_rootNode.addChild(child);
        }
        void addChild(SurfaceNode* child) {
            m_rootNode.addChild(child);
        }
        void removeChild(InternalNode* child) {
            m_rootNode.removeChild(child);
        }
        void removeChild(SurfaceNode* child) {
            m_rootNode.removeChild(child);
        }

        void set();
    };



    class Camera : public Object {
    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Camera(Context &context) : 
            Object(context) {}
        virtual ~Camera() {}

        virtual void set() const = 0;
    };



    class PerspectiveCamera : public Camera {
        struct OptiXProgramSet {
            optix::Program callableProgramSampleLensPosition;
            optix::Program callableProgramSampleIDF;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Shared::PerspectiveCamera m_data;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        PerspectiveCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                          float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist);

        void set() const override;

        void setPosition(const Point3D &position) {
            m_data.position = position;
        }
        void setOrientation(const Quaternion &orientation) {
            m_data.orientation = orientation;
        }
        void setSensitivity(float sensitivity) {
            m_data.sensitivity = sensitivity;
        }
        void setLensRadius(float lensRadius) {
            m_data.lensRadius = lensRadius;
        }
        void setObjectPlaneDistance(float distance) {
            m_data.setObjectPlaneDistance(distance);
        }
    };



    class EquirectangularCamera : public Camera {
        struct OptiXProgramSet {
            optix::Program callableProgramSampleLensPosition;
            optix::Program callableProgramSampleIDF;
        };

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        Shared::EquirectangularCamera m_data;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EquirectangularCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                              float sensitivity, float phiAngle, float thetaAngle);

        void set() const override;

        void setPosition(const Point3D &position) {
            m_data.position = position;
        }
        void setOrientation(const Quaternion &orientation) {
            m_data.orientation = orientation;
        }
        void setSensitivity(float sensitivity) {
            m_data.sensitivity = sensitivity;
        }
        void setAngles(float phiAngle, float thetaAngle) {
            m_data.phiAngle = phiAngle;
            m_data.thetaAngle = thetaAngle;
        }
    };
}
