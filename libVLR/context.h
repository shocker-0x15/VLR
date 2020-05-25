#pragma once

#include <public_types.h>
#include "shared/shared.h"

#include "slot_finder.h"

namespace VLR {
    extern const cudau::BufferType g_bufferType;

    std::string readTxtFile(const filesystem::path& filepath);



    enum class TextureFilter {
        Nearest = static_cast<TextureFilter>(cudau::TextureFilterMode::Point),
        Linear = static_cast<TextureFilter>(cudau::TextureFilterMode::Linear)
    };

    enum class TextureWrapMode {
        Repeat = static_cast<TextureWrapMode>(cudau::TextureWrapMode::Repeat),
        ClampToEdge = static_cast<TextureWrapMode>(cudau::TextureWrapMode::Clamp),
        Mirror = static_cast<TextureWrapMode>(cudau::TextureWrapMode::Mirror),
        ClampToBorder = static_cast<TextureWrapMode>(cudau::TextureWrapMode::Border),
    };



    class Scene;
    class Camera;

    template <typename InternalType>
    struct SlotBuffer {
        uint32_t maxNumElements;
        optixu::TypedBuffer<InternalType> optixBuffer;
        SlotFinder slotFinder;

        void initialize(optixu::Context &context, uint32_t _maxNumElements) {
            maxNumElements = _maxNumElements;
            optixBuffer.initialize(context.getCUcontext(), g_bufferType, maxNumElements);
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
            programGroup = pipeline.createCallableGroup(moduleDC, baseNameDC, moduleCC, baseNameCC);
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
        optixu::Context m_optixContext;

        Shared::PipelineLaunchParameters m_launchParams;
        cudau::TypedBuffer<DiscretizedSpectrumAlwaysSpectral::CMF> m_discretizedSpectrumCMFs;

        optixu::Pipeline m_optixPipeline;
        optixu::Module m_optixPathTracingModule;
        optixu::Module m_optixMaterialModule;
        optixu::Module m_optixEmptyModule;
        optixu::ProgramGroup m_optixPathTracingRayGeneration;
        optixu::ProgramGroup m_optixPathTracingMiss;
        optixu::ProgramGroup m_optixPathTracingShadowMiss;
        optixu::ProgramGroup m_optixPathTracingHitGroupDefault;
        optixu::ProgramGroup m_optixPathTracingHitGroupWithAlpha;
        optixu::ProgramGroup m_optixPathTracingHitGroupShadowDefault;
        optixu::ProgramGroup m_optixPathTracingHitGroupShadowWithAlpha;

        optixu::Module m_optixDebugRenderingModule;
        optixu::ProgramGroup m_optixDebugRenderingRayGeneration;
        optixu::ProgramGroup m_optixDebugRenderingMiss;
        optixu::ProgramGroup m_optixDebugRenderingHitGroupDefault;
        optixu::ProgramGroup m_optixDebugRenderingHitGroupWithAlpha;

        CUmodule m_cudaPostProcessModule;
        cudau::Kernel m_cudaPostProcessConvertToRGB;

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        cudau::TypedBuffer<UpsampledSpectrum::spectrum_grid_cell_t> m_optixBufferUpsampledSpectrum_spectrum_grid;
        cudau::TypedBuffer<UpsampledSpectrum::spectrum_data_point_t> m_optixBufferUpsampledSpectrum_spectrum_data_points;
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        cudau::TypedBuffer<float> m_optixBufferUpsampledSpectrum_maxBrightnesses;
        cudau::TypedBuffer<UpsampledSpectrum::PolynomialCoefficients> m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65;
        cudau::TypedBuffer<UpsampledSpectrum::PolynomialCoefficients> m_optixBufferUpsampledSpectrum_coefficients_sRGB_E;
#endif

        optixu::Material m_optixMaterialDefault;
        optixu::Material m_optixMaterialWithAlpha;

        SlotBuffer<Shared::NodeProcedureSet> m_nodeProcedureBuffer;

        SlotBuffer<Shared::SmallNodeDescriptor> m_smallNodeDescriptorBuffer;
        SlotBuffer<Shared::MediumNodeDescriptor> m_mediumNodeDescriptorBuffer;
        SlotBuffer<Shared::LargeNodeDescriptor> m_largeNodeDescriptorBuffer;

        SlotBuffer<Shared::BSDFProcedureSet> m_BSDFProcedureBuffer;
        SlotBuffer<Shared::EDFProcedureSet> m_EDFProcedureBuffer;

        CallableProgram m_optixCallableProgramNullBSDF_setupBSDF;
        CallableProgram m_optixCallableProgramNullBSDF_getBaseColor;
        CallableProgram m_optixCallableProgramNullBSDF_matches;
        CallableProgram m_optixCallableProgramNullBSDF_sampleInternal;
        CallableProgram m_optixCallableProgramNullBSDF_evaluateInternal;
        CallableProgram m_optixCallableProgramNullBSDF_evaluatePDFInternal;
        CallableProgram m_optixCallableProgramNullBSDF_weightInternal;
        uint32_t m_nullBSDFProcedureSetIndex;

        CallableProgram m_optixCallableProgramNullEDF_setupEDF;
        CallableProgram m_optixCallableProgramNullEDF_evaluateEmittanceInternal;
        CallableProgram m_optixCallableProgramNullEDF_evaluateInternal;
        uint32_t m_nullEDFProcedureSetIndex;

        SlotBuffer<Shared::SurfaceMaterialDescriptor> m_surfaceMaterialDescriptorBuffer;

        optixu::HostBlockBuffer2D<SpectrumStorage, 0> m_rawOutputBuffer;
        cudau::Array m_outputBuffer;
        optixu::HostBlockBuffer2D<Shared::KernelRNG, 2> m_rngBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numAccumFrames;

    public:
        Context(bool logging, uint32_t maxCallableDepth, CUcontext cuContext);
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glTexID);
        const cudau::Array &getOutputBuffer();
        void getOutputBufferSize(uint32_t* width, uint32_t* height);

        void render(Scene &scene, const Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);
        void debugRender(Scene &scene, const Camera* camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

        CUcontext getCuContext() const {
            return m_cuContext;
        }
        optixu::Context getOptiXContext() const {
            return m_optixContext;
        }
        optixu::Pipeline getOptixPipeline() const {
            return m_optixPipeline;
        }
        optixu::Module getEmptyModule() const {
            return m_optixEmptyModule;
        }

        optixu::Material getOptiXMaterialDefault() const {
            return m_optixMaterialDefault;
        }
        optixu::Material getOptiXMaterialWithAlpha() const {
            return m_optixMaterialWithAlpha;
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
            return m_optixCallableProgramNullBSDF_setupBSDF;
        }
        uint32_t getNullBSDFProcedureSetIndex() const { return m_nullBSDFProcedureSetIndex; }
        CallableProgram getOptixCallableProgramNullEDF_setupEDF() const {
            return m_optixCallableProgramNullEDF_setupEDF;
        }
        uint32_t getNullEDFProcedureSetIndex() const { return m_nullEDFProcedureSetIndex; }

        uint32_t allocateSurfaceMaterialDescriptor();
        void releaseSurfaceMaterialDescriptor(uint32_t index);
        void updateSurfaceMaterialDescriptor(uint32_t index, const Shared::SurfaceMaterialDescriptor &matDesc);
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
        optixu::TypedBuffer<RealType> m_PMF;
        optixu::TypedBuffer<RealType> m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        void getSharedType(Shared::DiscreteDistribution1DTemplate<RealType>* instance) const;
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution1DTemplate {
        optixu::TypedBuffer<RealType> m_PDF;
        optixu::TypedBuffer<RealType> m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        RealType getIntegral() const { return m_integral; }
        uint32_t getNumValues() const { return m_numValues; }

        void getSharedType(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        optixu::TypedBuffer<Shared::RegularConstantContinuousDistribution1DTemplate<RealType>> m_raw1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

    public:
        RegularConstantContinuousDistribution2DTemplate() : m_1DDists(nullptr) {}

        void initialize(Context &context, const RealType* values, size_t numD1, size_t numD2);
        void finalize(Context &context);

        bool isInitialized() const { return m_1DDists != nullptr; }

        void getSharedType(Shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
}
