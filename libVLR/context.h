#pragma once

#include <public_types.h>
#include "shared/light_transport_common.h"

#include "slot_finder.h"

namespace vlr {
    extern cudau::BufferType g_bufferType;

    std::string readTxtFile(const std::filesystem::path& filepath);



    enum class TextureFilter {
        Nearest = 0,
        Linear,
        None
    };

    enum class TextureWrapMode {
        Repeat = 0,
        ClampToEdge,
        Mirror,
        ClampToBorder,
    };



    class Scene;
    class Camera;
    class ShaderNode;
    class SurfaceMaterial;

    template <typename InternalType>
    struct SlotBuffer {
        uint32_t maxNumElements;
        cudau::TypedBuffer<InternalType> optixBuffer;
        SlotFinder slotFinder;

        void initialize(CUcontext cuContext, uint32_t _maxNumElements) {
            maxNumElements = _maxNumElements;
            optixBuffer.initialize(cuContext, g_bufferType, maxNumElements);
            optixBuffer.setMappedMemoryPersistent(true);
            slotFinder.initialize(maxNumElements);
        }
        void initialize(CUcontext cuContext, uint32_t _maxNumElements, const InternalType &defaultValue) {
            maxNumElements = _maxNumElements;
            optixBuffer.initialize(cuContext, g_bufferType, maxNumElements, defaultValue);
            optixBuffer.setMappedMemoryPersistent(true);
            slotFinder.initialize(maxNumElements);
        }
        void finalize() {
            slotFinder.finalize();
            optixBuffer.finalize();
        }

        uint32_t allocate() {
            uint32_t index = slotFinder.getFirstAvailableSlot();
            slotFinder.setInUse(index);
            return index;
        }

        void release(uint32_t index) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            slotFinder.setNotInUse(index);
        }

        void get(uint32_t index, InternalType* value) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            auto values = optixBuffer.map();
            *value = values[index];
            optixBuffer.unmap();
        }

        void update(uint32_t index, const InternalType &value, CUstream stream) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            CUDADRV_CHECK(cuMemcpyHtoDAsync(optixBuffer.getCUdeviceptrAt(index), &value, sizeof(value), stream));
        }
    };



    enum OptiXModule {
        OptiXModule_LightTransport = 0,
        OptiXModule_ShaderNode,
        OptiXModule_Material,
        OptiXModule_Triangle,
        OptiXModule_Point,
        OptiXModule_InfiniteSphere,
        OptiXModule_Camera,
        NumOptiXModules,
    };

    constexpr const char* commonModulePaths[] = {
        "ptxes/shader_nodes.ptx",
        "ptxes/materials.ptx",
        "ptxes/triangle.ptx",
        "ptxes/point.ptx",
        "ptxes/infinite_sphere.ptx",
        "ptxes/cameras.ptx",
    };



    class Context {
        static uint32_t NextID;
        static uint32_t getInstanceID() {
            return NextID++;
        }

        uint32_t m_ID;
        CUcontext m_cuContext;

        struct OptiX {
            SlotBuffer<shared::NodeProcedureSet> nodeProcedureSetBuffer;

            SlotBuffer<shared::SmallNodeDescriptor> smallNodeDescriptorBuffer;
            SlotBuffer<shared::MediumNodeDescriptor> mediumNodeDescriptorBuffer;
            SlotBuffer<shared::LargeNodeDescriptor> largeNodeDescriptorBuffer;
            std::unordered_set<ShaderNode*> dirtyShaderNodes;

            SlotBuffer<shared::BSDFProcedureSet> bsdfProcedureSetBuffer;
            SlotBuffer<shared::EDFProcedureSet> edfProcedureSetBuffer;

            SlotBuffer<shared::SurfaceMaterialDescriptor> surfaceMaterialDescriptorBuffer;
            std::unordered_set<SurfaceMaterial*> dirtySurfaceMaterials;

            optixu::Context context;

            optixu::Material materialDefault;
            optixu::Material materialWithAlpha;

            struct PathTracing {
                optixu::Pipeline pipeline;

                std::vector<optixu::Module> modules;

                optixu::ProgramGroup rayGeneration;
                optixu::ProgramGroup miss;
                optixu::ProgramGroup shadowMiss;
                optixu::ProgramGroup hitGroupDefault;
                optixu::ProgramGroup hitGroupWithAlpha;
                optixu::ProgramGroup hitGroupShadowDefault;
                optixu::ProgramGroup hitGroupShadowWithAlpha;
                optixu::ProgramGroup emptyHitGroup;
                std::vector<optixu::ProgramGroup> callablePrograms;

                cudau::Buffer shaderBindingTable;
                cudau::Buffer hitGroupShaderBindingTable;
            } pathTracing;

            struct DebugRendering {
                optixu::Pipeline pipeline;

                std::vector<optixu::Module> modules;

                optixu::ProgramGroup rayGeneration;
                optixu::ProgramGroup miss;
                optixu::ProgramGroup hitGroupDefault;
                optixu::ProgramGroup hitGroupWithAlpha;
                optixu::ProgramGroup emptyHitGroup;
                std::vector<optixu::ProgramGroup> callablePrograms;

                cudau::Buffer shaderBindingTable;
                cudau::Buffer hitGroupShaderBindingTable;
            } debugRendering;

            uint32_t dcNullBSDF_setupBSDF;
            uint32_t dcNullBSDF_getBaseColor;
            uint32_t dcNullBSDF_matches;
            uint32_t dcNullBSDF_sampleInternal;
            uint32_t dcNullBSDF_evaluateInternal;
            uint32_t dcNullBSDF_evaluatePDFInternal;
            uint32_t dcNullBSDF_weightInternal;
            uint32_t nullBSDFProcedureSetIndex;
            uint32_t dcNullEDF_setupEDF;
            uint32_t dcNullEDF_evaluateEmittanceInternal;
            uint32_t dcNullEDF_evaluateInternal;
            uint32_t nullEDFProcedureSetIndex;

            shared::PipelineLaunchParameters launchParams;
            CUdeviceptr launchParamsOnDevice;

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
            cudau::TypedBuffer<UpsampledSpectrum::spectrum_grid_cell_t> UpsampledSpectrum_spectrum_grid;
            cudau::TypedBuffer<UpsampledSpectrum::spectrum_data_point_t> UpsampledSpectrum_spectrum_data_points;
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
            cudau::TypedBuffer<float> upsampledSpectrum_maxBrightnesses;
            cudau::TypedBuffer<UpsampledSpectrum::PolynomialCoefficients> UpsampledSpectrum_coefficients_sRGB_D65;
            cudau::TypedBuffer<UpsampledSpectrum::PolynomialCoefficients> UpsampledSpectrum_coefficients_sRGB_E;
#endif

            optixu::Denoiser denoiser;
            CUdeviceptr hdrIntensity;
            cudau::Buffer denoiserStateBuffer;
            cudau::Buffer denoiserScratchBuffer;
            std::vector<optixu::DenoisingTask> denoiserTasks;
            cudau::TypedBuffer<DiscretizedSpectrum> accumAlbedoBuffer;
            cudau::TypedBuffer<Normal3D> accumNormalBuffer;
            cudau::TypedBuffer<float4> linearColorBuffer;
            cudau::TypedBuffer<float4> linearAlbedoBuffer;
            cudau::TypedBuffer<float4> linearNormalBuffer;
            cudau::TypedBuffer<float4> linearDenoisedColorBuffer;

            optixu::HostBlockBuffer2D<SpectrumStorage, 0> accumBuffer;

            cudau::Array outputBuffer;
            cudau::InteropSurfaceObjectHolder<2> outputBufferHolder;
            bool useGLTexture;
            cudau::Array rngBuffer;

            cudau::Buffer asScratchMem;
        } m_optix;

        CUmodule m_cudaSetupSceneModule;
        cudau::Kernel m_computeInstanceAABBs;
        cudau::Kernel m_finalizeInstanceAABBs;
        cudau::Kernel m_computeSceneAABB;
        cudau::Kernel m_finalizeSceneBounds;

        CUmodule m_cudaPostProcessModule;
        CUdeviceptr m_cudaPostProcessModuleLaunchParamsPtr;
        cudau::Kernel m_copyBuffers;
        cudau::Kernel m_convertToRGB;

        Scene* m_scene;

        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numAccumFrames;

        void render(CUstream stream, const Camera* camera, bool denoise,
                    bool debugRender, VLRDebugRenderingMode renderMode,
                    uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

    public:
        Context(CUcontext cuContext, bool logging, uint32_t maxCallableDepth);
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glTexID);
        void getOutputBufferSize(uint32_t* width, uint32_t* height);
        const cudau::Array &getOutputBuffer() const;
        void readOutputBuffer(float* data);

        void setScene(Scene* scene);
        void render(CUstream stream, const Camera* camera, bool denoise,
                    uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);
        void debugRender(CUstream stream, const Camera* camera, VLRDebugRenderingMode renderMode,
                         uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

        CUcontext getCUcontext() const {
            return m_cuContext;
        }
        optixu::Context getOptiXContext() const {
            return m_optix.context;
        }

        optixu::Material getOptiXMaterialDefault() const {
            return m_optix.materialDefault;
        }
        optixu::Material getOptiXMaterialWithAlpha() const {
            return m_optix.materialWithAlpha;
        }

        uint32_t createDirectCallableProgram(OptiXModule mdl, const char* dcName);
        void destroyDirectCallableProgram(uint32_t index);

        uint32_t allocateNodeProcedureSet();
        void releaseNodeProcedureSet(uint32_t index);
        void updateNodeProcedureSet(uint32_t index, const shared::NodeProcedureSet &procSet, CUstream stream);

        uint32_t allocateSmallNodeDescriptor();
        void releaseSmallNodeDescriptor(uint32_t index);
        void updateSmallNodeDescriptor(uint32_t index, const shared::SmallNodeDescriptor &nodeDesc, CUstream stream);

        uint32_t allocateMediumNodeDescriptor();
        void releaseMediumNodeDescriptor(uint32_t index);
        void updateMediumNodeDescriptor(uint32_t index, const shared::MediumNodeDescriptor &nodeDesc, CUstream stream);

        uint32_t allocateLargeNodeDescriptor();
        void releaseLargeNodeDescriptor(uint32_t index);
        void updateLargeNodeDescriptor(uint32_t index, const shared::LargeNodeDescriptor &nodeDesc, CUstream stream);

        uint32_t allocateBSDFProcedureSet();
        void releaseBSDFProcedureSet(uint32_t index);
        void updateBSDFProcedureSet(uint32_t index, const shared::BSDFProcedureSet &procSet, CUstream stream);

        uint32_t allocateEDFProcedureSet();
        void releaseEDFProcedureSet(uint32_t index);
        void updateEDFProcedureSet(uint32_t index, const shared::EDFProcedureSet &procSet, CUstream stream);

        uint32_t getOptixCallableProgramNullBSDF_setupBSDF() const {
            return m_optix.dcNullBSDF_setupBSDF;
        }
        uint32_t getNullBSDFProcedureSetIndex() const { return m_optix.nullBSDFProcedureSetIndex; }
        uint32_t getOptixCallableProgramNullEDF_setupEDF() const {
            return m_optix.dcNullEDF_setupEDF;
        }
        uint32_t getNullEDFProcedureSetIndex() const { return m_optix.nullEDFProcedureSetIndex; }

        uint32_t allocateSurfaceMaterialDescriptor();
        void releaseSurfaceMaterialDescriptor(uint32_t index);
        void updateSurfaceMaterialDescriptor(uint32_t index, const shared::SurfaceMaterialDescriptor &matDesc, CUstream stream);

        void markShaderNodeDescriptorDirty(ShaderNode* node);
        void markSurfaceMaterialDescriptorDirty(SurfaceMaterial* mat);

        void computeInstanceAABBs(
            CUstream stream,
            const uint32_t* instIndices, const uint32_t* itemOffsets,
            shared::Instance* instances, const shared::GeometryInstance* geomInsts, uint32_t numItems) const {
            m_computeInstanceAABBs(stream, m_computeInstanceAABBs.calcGridDim(numItems),
                                   instIndices, itemOffsets,
                                   instances, geomInsts, numItems);
        }
        void finalizeInstanceAABBs(
            CUstream stream,
            shared::Instance* instances, uint32_t numInstances) const {
            m_finalizeInstanceAABBs(stream, m_finalizeInstanceAABBs.calcGridDim(numInstances),
                                    instances, numInstances);
        }
        void computeSceneAABB(
            CUstream stream,
            const shared::Instance* instances, uint32_t numInstances,
            shared::SceneBounds* sceneAabbAsInt) const {
            m_computeSceneAABB(stream, m_computeSceneAABB.calcGridDim(numInstances),
                               instances, numInstances, sceneAabbAsInt);
        }
        void finalizeSceneBounds(CUstream stream, shared::SceneBounds* sceneBounds) const {
            m_finalizeSceneBounds(stream, m_finalizeSceneBounds.calcGridDim(1), sceneBounds);
        }
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



    class TypeAwareClass {
    public:
        virtual const char* getType() const = 0;
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier& getClass() const { return ClassID; }

        template <class T>
        constexpr bool is() const {
            return &getClass() == &T::ClassID;
        }

        template <class T>
        constexpr bool belongsTo() const {
            const ClassIdentifier* curClass = &getClass();
            while (curClass) {
                if (curClass == &T::ClassID)
                    return true;
                curClass = curClass->getBaseClass();
            }
            return false;
        }
    };

#define VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE() \
    static const char* const TypeName; \
    virtual const char* getType() const { return TypeName; } \
    static const ClassIdentifier ClassID; \
    virtual const ClassIdentifier &getClass() const override { return ClassID; }



    class Object : public TypeAwareClass {
    protected:
        Context &m_context;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        Object(Context &context);
        virtual ~Object() {}

        Context &getContext() {
            return m_context;
        }
    };



    // ----------------------------------------------------------------
    // Miscellaneous

    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        cudau::TypedBuffer<RealType> m_PMF;
        cudau::TypedBuffer<RealType> m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        DiscreteDistribution1DTemplate &operator=(DiscreteDistribution1DTemplate &&v) {
            m_PMF = std::move(v.m_PMF);
            m_CDF = std::move(v.m_CDF);
            m_integral = v.m_integral;
            m_numValues = v.m_numValues;
            return *this;
        }

        void getInternalType(shared::DiscreteDistribution1DTemplate<RealType>* instance) const;
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution1DTemplate {
        cudau::TypedBuffer<RealType> m_PDF;
        cudau::TypedBuffer<RealType> m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        RegularConstantContinuousDistribution1DTemplate &operator=(RegularConstantContinuousDistribution1DTemplate &&v) {
            m_PDF = std::move(v.m_PDF);
            m_CDF = std::move(v.m_CDF);
            m_integral = v.m_integral;
            m_numValues = v.m_numValues;
            return *this;
        }

        RealType getIntegral() const { return m_integral; }
        uint32_t getNumValues() const { return m_numValues; }

        void getInternalType(shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        cudau::TypedBuffer<shared::RegularConstantContinuousDistribution1DTemplate<RealType>> m_raw1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

    public:
        RegularConstantContinuousDistribution2DTemplate() : m_1DDists(nullptr) {}

        RegularConstantContinuousDistribution2DTemplate &operator=(RegularConstantContinuousDistribution2DTemplate &&v) {
            m_raw1DDists = std::move(v.m_raw1DDists);
            m_1DDists = std::move(v.m_1DDists);
            m_top1DDist = std::move(v.m_top1DDist);
            return *this;
        }

        void initialize(Context &context, const RealType* values, size_t numD1, size_t numD2);
        void finalize(Context &context);

        bool isInitialized() const { return m_1DDists != nullptr; }

        void getInternalType(shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
}
