#pragma once

#include <public_types.h>
#include "shared/shared.h"

#include "slot_finder.h"

namespace VLR {
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

        void update(uint32_t index, const InternalType &value) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            auto values = optixBuffer.map();
            values[index] = value;
            optixBuffer.unmap();
        }
    };



    struct CallableProgram {
        static uint32_t NextID;
        optixu::ProgramGroup programGroup;
        uint32_t ID;

        CallableProgram() : ID(0xFFFFFFFF) {}
        void create(optixu::Pipeline pipeline,
                    optixu::Module moduleDC, const char* baseNameDC,
                    optixu::Module moduleCC, const char* baseNameCC) {
            ID = NextID++;
            programGroup = pipeline.createCallableProgramGroup(moduleDC, baseNameDC, moduleCC, baseNameCC);
        }
        void destroy() {
            programGroup.destroy();
        }
        operator bool() const {
            return ID != 0xFFFFFFFF;
        }
    };



    class Context {
        static uint32_t NextID;
        static uint32_t getInstanceID() {
            return NextID++;
        }

        uint32_t m_ID;
        CUcontext m_cuContext;

        struct {
            SlotBuffer<Shared::NodeProcedureSet> nodeProcedureSetBuffer;

            SlotBuffer<Shared::SmallNodeDescriptor> smallNodeDescriptorBuffer;
            SlotBuffer<Shared::MediumNodeDescriptor> mediumNodeDescriptorBuffer;
            SlotBuffer<Shared::LargeNodeDescriptor> largeNodeDescriptorBuffer;

            SlotBuffer<Shared::BSDFProcedureSet> bsdfProcedureSetBuffer;
            SlotBuffer<Shared::EDFProcedureSet> edfProcedureSetBuffer;

            SlotBuffer<Shared::SurfaceMaterialDescriptor> surfaceMaterialDescriptorBuffer;

            optixu::Context context;

            optixu::Pipeline pipeline;

            optixu::Module pathTracingModule;
            optixu::ProgramGroup pathTracingRayGeneration;
            optixu::ProgramGroup pathTracingMiss;
            optixu::ProgramGroup pathTracingShadowMiss;
            optixu::ProgramGroup pathTracingHitGroupDefault;
            optixu::ProgramGroup pathTracingHitGroupWithAlpha;
            optixu::ProgramGroup pathTracingHitGroupShadowDefault;
            optixu::ProgramGroup pathTracingHitGroupShadowWithAlpha;

            optixu::Module debugRenderingModule;
            optixu::ProgramGroup debugRenderingRayGeneration;
            optixu::ProgramGroup debugRenderingMiss;
            optixu::ProgramGroup debugRenderingHitGroupDefault;
            optixu::ProgramGroup debugRenderingHitGroupWithAlpha;

            optixu::Module nullDFModule;
            CallableProgram dcNullBSDF_setupBSDF;
            CallableProgram dcNullBSDF_getBaseColor;
            CallableProgram dcNullBSDF_matches;
            CallableProgram dcNullBSDF_sampleInternal;
            CallableProgram dcNullBSDF_evaluateInternal;
            CallableProgram dcNullBSDF_evaluatePDFInternal;
            CallableProgram dcNullBSDF_weightInternal;
            uint32_t nullBSDFProcedureSetIndex;
            CallableProgram dcNullEDF_setupEDF;
            CallableProgram dcNullEDF_evaluateEmittanceInternal;
            CallableProgram dcNullEDF_evaluateInternal;
            uint32_t nullEDFProcedureSetIndex;

            optixu::Material materialDefault;
            optixu::Material materialWithAlpha;

            optixu::Scene scene;
            SlotBuffer<Shared::GeometryInstance> geomInstBuffer;
            SlotBuffer<Shared::Instance> instBuffer;

            Shared::PipelineLaunchParameters launchParams;
            CUdeviceptr launchParamsOnDevice;

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
            cudau::TypedBuffer<UpsampledSpectrum::spectrum_grid_cell_t> UpsampledSpectrum_spectrum_grid;
            cudau::TypedBuffer<UpsampledSpectrum::spectrum_data_point_t> UpsampledSpectrum_spectrum_data_points;
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
            cudau::TypedBuffer<float> upsampledSpectrum_maxBrightnesses;
            cudau::TypedBuffer<UpsampledSpectrum::PolynomialCoefficients> UpsampledSpectrum_coefficients_sRGB_D65;
            cudau::TypedBuffer<UpsampledSpectrum::PolynomialCoefficients> UpsampledSpectrum_coefficients_sRGB_E;
#endif
        } m_optix;

        CUmodule m_cudaPostProcessModule;
        cudau::Kernel m_cudaPostProcessConvertToRGB;

        optixu::HostBlockBuffer2D<SpectrumStorage, 0> m_rawOutputBuffer;
        cudau::Array m_outputBuffer;
        optixu::HostBlockBuffer2D<Shared::KernelRNG, 2> m_rngBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numAccumFrames;

    public:
        Context(CUcontext cuContext, bool logging, uint32_t maxCallableDepth);
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glTexID);
        const cudau::Array &getOutputBuffer() const;
        void getOutputBufferSize(uint32_t* width, uint32_t* height);

        void render(Scene &scene, const Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);
        void debugRender(Scene &scene, const Camera* camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

        CUcontext getCuContext() const {
            return m_cuContext;
        }
        optixu::Context getOptiXContext() const {
            return m_optix.context;
        }
        optixu::Pipeline getOptixPipeline() const {
            return m_optix.pipeline;
        }

        optixu::Material getOptiXMaterialDefault() const {
            return m_optix.materialDefault;
        }
        optixu::Material getOptiXMaterialWithAlpha() const {
            return m_optix.materialWithAlpha;
        }

        optixu::Scene getOptiXScene() const {
            return m_optix.scene;
        }

        uint32_t allocateNodeProcedureSet();
        void releaseNodeProcedureSet(uint32_t index);
        void updateNodeProcedureSet(uint32_t index, const Shared::NodeProcedureSet &procSet);

        uint32_t allocateSmallNodeDescriptor();
        void releaseSmallNodeDescriptor(uint32_t index);
        void updateSmallNodeDescriptor(uint32_t index, const Shared::SmallNodeDescriptor &nodeDesc);

        uint32_t allocateMediumNodeDescriptor();
        void releaseMediumNodeDescriptor(uint32_t index);
        void updateMediumNodeDescriptor(uint32_t index, const Shared::MediumNodeDescriptor &nodeDesc);

        uint32_t allocateLargeNodeDescriptor();
        void releaseLargeNodeDescriptor(uint32_t index);
        void updateLargeNodeDescriptor(uint32_t index, const Shared::LargeNodeDescriptor &nodeDesc);

        uint32_t allocateBSDFProcedureSet();
        void releaseBSDFProcedureSet(uint32_t index);
        void updateBSDFProcedureSet(uint32_t index, const Shared::BSDFProcedureSet &procSet);

        uint32_t allocateEDFProcedureSet();
        void releaseEDFProcedureSet(uint32_t index);
        void updateEDFProcedureSet(uint32_t index, const Shared::EDFProcedureSet &procSet);

        CallableProgram getOptixCallableProgramNullBSDF_setupBSDF() const {
            return m_optix.dcNullBSDF_setupBSDF;
        }
        uint32_t getNullBSDFProcedureSetIndex() const { return m_optix.nullBSDFProcedureSetIndex; }
        CallableProgram getOptixCallableProgramNullEDF_setupEDF() const {
            return m_optix.dcNullEDF_setupEDF;
        }
        uint32_t getNullEDFProcedureSetIndex() const { return m_optix.nullEDFProcedureSetIndex; }

        uint32_t allocateSurfaceMaterialDescriptor();
        void releaseSurfaceMaterialDescriptor(uint32_t index);
        void updateSurfaceMaterialDescriptor(uint32_t index, const Shared::SurfaceMaterialDescriptor &matDesc);

        uint32_t allocateGeometryInstance();
        void releaseGeometryInstance(uint32_t index);
        void updateGeometryInstance(uint32_t index, const Shared::GeometryInstance &geomInst);

        uint32_t allocateInstance();
        void releaseInstance(uint32_t index);
        void updateInstance(uint32_t index, const Shared::Instance &inst);
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
        constexpr bool isMemberOf() const {
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
    static const char* TypeName; \
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

        void getInternalType(Shared::DiscreteDistribution1DTemplate<RealType>* instance) const;
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

        void getInternalType(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        cudau::TypedBuffer<Shared::RegularConstantContinuousDistribution1DTemplate<RealType>> m_raw1DDists;
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

        void getInternalType(Shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
}
