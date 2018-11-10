#include "context.h"

#include <random>

#include "scene.h"

namespace VLR {
    std::string readTxtFile(const std::string& filepath) {
        std::ifstream ifs;
        ifs.open(filepath, std::ios::in);
        if (ifs.fail())
            return "";

        std::stringstream sstream;
        sstream << ifs.rdbuf();

        return std::string(sstream.str());
    };



    Object::Object(Context &context) : m_context(context) {
    }

#define defineClassID(BaseType, Type) const ClassIdentifier Type::ClassID = ClassIdentifier(&BaseType::ClassID)

    const ClassIdentifier TypeAwareClass::ClassID = ClassIdentifier((ClassIdentifier*)nullptr);

    defineClassID(TypeAwareClass, Object);

    defineClassID(Object, Image2D);
    defineClassID(Image2D, LinearImage2D);

    defineClassID(Object, ShaderNode);
    defineClassID(ShaderNode, GeometryShaderNode);
    defineClassID(ShaderNode, FloatShaderNode);
    defineClassID(ShaderNode, Float2ShaderNode);
    defineClassID(ShaderNode, Float3ShaderNode);
    defineClassID(ShaderNode, Float4ShaderNode);
    defineClassID(ShaderNode, Vector3DToSpectrumShaderNode);
    defineClassID(ShaderNode, OffsetAndScaleUVTextureMap2DShaderNode);
    defineClassID(ShaderNode, ConstantTextureShaderNode);
    defineClassID(ShaderNode, Image2DTextureShaderNode);
    defineClassID(ShaderNode, EnvironmentTextureShaderNode);

    defineClassID(Object, SurfaceMaterial);
    defineClassID(SurfaceMaterial, MatteSurfaceMaterial);
    defineClassID(SurfaceMaterial, SpecularReflectionSurfaceMaterial);
    defineClassID(SurfaceMaterial, SpecularScatteringSurfaceMaterial);
    defineClassID(SurfaceMaterial, MicrofacetReflectionSurfaceMaterial);
    defineClassID(SurfaceMaterial, MicrofacetScatteringSurfaceMaterial);
    defineClassID(SurfaceMaterial, LambertianScatteringSurfaceMaterial);
    defineClassID(SurfaceMaterial, UE4SurfaceMaterial);
    defineClassID(SurfaceMaterial, DiffuseEmitterSurfaceMaterial);
    defineClassID(SurfaceMaterial, MultiSurfaceMaterial);
    defineClassID(SurfaceMaterial, EnvironmentEmitterSurfaceMaterial);

    defineClassID(Object, Transform);
    defineClassID(Transform, StaticTransform);

    defineClassID(Object, Node);
    defineClassID(Node, SurfaceNode);
    defineClassID(SurfaceNode, TriangleMeshSurfaceNode);
    defineClassID(SurfaceNode, InfiniteSphereSurfaceNode);
    defineClassID(Node, ParentNode);
    defineClassID(ParentNode, InternalNode);
    defineClassID(ParentNode, RootNode);
    defineClassID(Object, Scene);

    defineClassID(Object, Camera);
    defineClassID(Camera, PerspectiveCamera);
    defineClassID(Camera, EquirectangularCamera);

#undef defineClassID



    uint32_t Context::NextID = 0;

    Context::Context(bool logging, uint32_t stackSize) {
        m_ID = getInstanceID();

        m_optixContext = optix::Context::create();

        m_optixContext->setEntryPointCount(1);
        m_optixContext->setRayTypeCount(Shared::RayType::NumTypes);

        {
            std::string ptx = readTxtFile("resources/ptxes/path_tracing.ptx");

            m_optixProgramShadowAnyHitDefault = m_optixContext->createProgramFromPTXString(ptx, "VLR::shadowAnyHitDefault");
            m_optixProgramAnyHitWithAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::anyHitWithAlpha");
            m_optixProgramShadowAnyHitWithAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::shadowAnyHitWithAlpha");
            m_optixProgramPathTracingIteration = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingIteration");

            m_optixProgramPathTracing = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracing");
            m_optixProgramPathTracingMiss = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingMiss");
            m_optixProgramException = m_optixContext->createProgramFromPTXString(ptx, "VLR::exception");
        }

        m_optixContext->setRayGenerationProgram(0, m_optixProgramPathTracing);
        m_optixContext->setExceptionProgram(0, m_optixProgramException);

        m_optixContext->setMissProgram(Shared::RayType::Primary, m_optixProgramPathTracingMiss);
        m_optixContext->setMissProgram(Shared::RayType::Scattered, m_optixProgramPathTracingMiss);

        m_optixMaterialDefault = m_optixContext->createMaterial();
        m_optixMaterialDefault->setClosestHitProgram(Shared::RayType::Primary, m_optixProgramPathTracingIteration);
        m_optixMaterialDefault->setClosestHitProgram(Shared::RayType::Scattered, m_optixProgramPathTracingIteration);
        //m_optixMaterialDefault->setAnyHitProgram(Shared::RayType::Primary, );
        //m_optixMaterialDefault->setAnyHitProgram(Shared::RayType::Scattered, );
        m_optixMaterialDefault->setAnyHitProgram(Shared::RayType::Shadow, m_optixProgramShadowAnyHitDefault);

        m_optixMaterialWithAlpha = m_optixContext->createMaterial();
        m_optixMaterialWithAlpha->setClosestHitProgram(Shared::RayType::Primary, m_optixProgramPathTracingIteration);
        m_optixMaterialWithAlpha->setClosestHitProgram(Shared::RayType::Scattered, m_optixProgramPathTracingIteration);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::Primary, m_optixProgramAnyHitWithAlpha);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::Scattered, m_optixProgramAnyHitWithAlpha);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::Shadow, m_optixProgramShadowAnyHitWithAlpha);



        m_maxNumNodeProcSet = 64;
        m_optixNodeProcedureSetBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumNodeProcSet);
        m_optixNodeProcedureSetBuffer->setElementSize(sizeof(Shared::NodeProcedureSet));
        m_nodeProcSetSlotManager.initialize(m_maxNumNodeProcSet);
        m_optixContext["VLR::pv_nodeProcedureSetBuffer"]->set(m_optixNodeProcedureSetBuffer);



        m_maxNumNodeDescriptors = 8192;
        m_optixNodeDescriptorBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumNodeDescriptors);
        m_optixNodeDescriptorBuffer->setElementSize(sizeof(Shared::NodeDescriptor));
        m_nodeDescSlotManager.initialize(m_maxNumNodeDescriptors);

        m_optixContext["VLR::pv_nodeDescriptorBuffer"]->set(m_optixNodeDescriptorBuffer);



        m_maxNumBSDFProcSet = 64;
        m_optixBSDFProcedureSetBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumBSDFProcSet);
        m_optixBSDFProcedureSetBuffer->setElementSize(sizeof(Shared::BSDFProcedureSet));
        m_bsdfProcSetSlotManager.initialize(m_maxNumBSDFProcSet);
        m_optixContext["VLR::pv_bsdfProcedureSetBuffer"]->set(m_optixBSDFProcedureSetBuffer);

        m_maxNumEDFProcSet = 64;
        m_optixEDFProcedureSetBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumEDFProcSet);
        m_optixEDFProcedureSetBuffer->setElementSize(sizeof(Shared::EDFProcedureSet));
        m_edfProcSetSlotManager.initialize(m_maxNumEDFProcSet);
        m_optixContext["VLR::pv_edfProcedureSetBuffer"]->set(m_optixEDFProcedureSetBuffer);

        {
            std::string ptx = readTxtFile("resources/ptxes/materials.ptx");

            m_optixCallableProgramNullBSDF_setupBSDF = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_setupBSDF");
            m_optixCallableProgramNullBSDF_getBaseColor = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_getBaseColor");
            m_optixCallableProgramNullBSDF_matches = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_matches");
            m_optixCallableProgramNullBSDF_sampleInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_sampleInternal");
            m_optixCallableProgramNullBSDF_evaluateInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_evaluateInternal");
            m_optixCallableProgramNullBSDF_evaluatePDFInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_evaluatePDFInternal");
            m_optixCallableProgramNullBSDF_weightInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_weightInternal");

            Shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = m_optixCallableProgramNullBSDF_getBaseColor->getId();
                bsdfProcSet.progMatches = m_optixCallableProgramNullBSDF_matches->getId();
                bsdfProcSet.progSampleInternal = m_optixCallableProgramNullBSDF_sampleInternal->getId();
                bsdfProcSet.progEvaluateInternal = m_optixCallableProgramNullBSDF_evaluateInternal->getId();
                bsdfProcSet.progEvaluatePDFInternal = m_optixCallableProgramNullBSDF_evaluatePDFInternal->getId();
                bsdfProcSet.progWeightInternal = m_optixCallableProgramNullBSDF_weightInternal->getId();
            }
            m_nullBSDFProcedureSetIndex = allocateBSDFProcedureSet();
            updateBSDFProcedureSet(m_nullBSDFProcedureSetIndex, bsdfProcSet);
            VLRAssert(m_nullBSDFProcedureSetIndex == 0, "Index of the null BSDF procedure set is expected to be 0.");



            m_optixCallableProgramNullEDF_setupEDF = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_setupEDF");
            m_optixCallableProgramNullEDF_evaluateEmittanceInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEmittanceInternal");
            m_optixCallableProgramNullEDF_evaluateInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateInternal");

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = m_optixCallableProgramNullEDF_evaluateEmittanceInternal->getId();
                edfProcSet.progEvaluateInternal = m_optixCallableProgramNullEDF_evaluateInternal->getId();
            }
            m_nullEDFProcedureSetIndex = allocateEDFProcedureSet();
            updateEDFProcedureSet(m_nullEDFProcedureSetIndex, edfProcSet);
            VLRAssert(m_nullEDFProcedureSetIndex == 0, "Index of the null EDF procedure set is expected to be 0.");
        }

        m_maxNumSurfaceMaterialDescriptors = 8192;
        m_optixSurfaceMaterialDescriptorBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumSurfaceMaterialDescriptors);
        m_optixSurfaceMaterialDescriptorBuffer->setElementSize(sizeof(Shared::SurfaceMaterialDescriptor));
        m_surfMatDescSlotManager.initialize(m_maxNumSurfaceMaterialDescriptors);

        m_optixContext["VLR::pv_materialDescriptorBuffer"]->set(m_optixSurfaceMaterialDescriptorBuffer);

        SurfaceNode::initialize(*this);
        ShaderNode::initialize(*this);
        SurfaceMaterial::initialize(*this);
        Camera::initialize(*this);

        RTsize defaultStackSize = m_optixContext->getStackSize();
        VLRDebugPrintf("Default Stack Size: %u\n", defaultStackSize);

        if (logging) {
            m_optixContext->setPrintEnabled(true);
            m_optixContext->setPrintBufferSize(4096);
            //m_optixContext->setExceptionEnabled(RT_EXCEPTION_BUFFER_ID_INVALID, true);
            //m_optixContext->setExceptionEnabled(RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true);
            //m_optixContext->setExceptionEnabled(RT_EXCEPTION_INTERNAL_ERROR, true);
            if (stackSize == 0)
                stackSize = 1280;
        }
        else {
            m_optixContext->setExceptionEnabled(RT_EXCEPTION_STACK_OVERFLOW, false);
            if (stackSize == 0)
                stackSize = 640;
        }
        m_optixContext->setStackSize(stackSize);
        VLRDebugPrintf("Stack Size: %u\n", stackSize);
    }

    Context::~Context() {
        if (m_rngBuffer)
            m_rngBuffer->destroy();

        if (m_outputBuffer)
            m_outputBuffer->destroy();

        Camera::finalize(*this);
        SurfaceMaterial::finalize(*this);
        ShaderNode::finalize(*this);
        SurfaceNode::finalize(*this);

        m_surfMatDescSlotManager.finalize();
        m_optixSurfaceMaterialDescriptorBuffer->destroy();

        releaseEDFProcedureSet(m_nullEDFProcedureSetIndex);
        m_optixCallableProgramNullEDF_evaluateInternal->destroy();
        m_optixCallableProgramNullEDF_evaluateEmittanceInternal->destroy();
        m_optixCallableProgramNullEDF_setupEDF->destroy();

        releaseBSDFProcedureSet(m_nullBSDFProcedureSetIndex);
        m_optixCallableProgramNullBSDF_weightInternal->destroy();
        m_optixCallableProgramNullBSDF_evaluatePDFInternal->destroy();
        m_optixCallableProgramNullBSDF_evaluateInternal->destroy();
        m_optixCallableProgramNullBSDF_sampleInternal->destroy();
        m_optixCallableProgramNullBSDF_matches->destroy();
        m_optixCallableProgramNullBSDF_getBaseColor->destroy();
        m_optixCallableProgramNullBSDF_setupBSDF->destroy();

        m_edfProcSetSlotManager.finalize();
        m_optixEDFProcedureSetBuffer->destroy();

        m_bsdfProcSetSlotManager.finalize();
        m_optixBSDFProcedureSetBuffer->destroy();

        m_nodeDescSlotManager.finalize();
        m_optixNodeDescriptorBuffer->destroy();

        m_nodeProcSetSlotManager.finalize();
        m_optixNodeProcedureSetBuffer->destroy();

        m_optixMaterialWithAlpha->destroy();
        m_optixMaterialDefault->destroy();

        m_optixProgramException->destroy();
        m_optixProgramPathTracingMiss->destroy();
        m_optixProgramPathTracing->destroy();

        m_optixProgramPathTracingIteration->destroy();
        m_optixProgramShadowAnyHitWithAlpha->destroy();
        m_optixProgramAnyHitWithAlpha->destroy();
        m_optixProgramShadowAnyHitDefault->destroy();

        m_optixContext->destroy();
    }

    void Context::setDevices(const int32_t* devices, uint32_t numDevices) {
        m_optixContext->setDevices(devices, devices + numDevices);
    }

    void Context::bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID) {
        if (m_outputBuffer)
            m_outputBuffer->destroy();
        if (m_rngBuffer)
            m_rngBuffer->destroy();

        m_width = width;
        m_height = height;

        if (glBufferID > 0) {
            m_outputBuffer = m_optixContext->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, glBufferID);
            m_outputBuffer->setFormat(RT_FORMAT_USER);
            m_outputBuffer->setSize(m_width, m_height);
        }
        else {
            m_outputBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_width, m_height);
        }
        m_outputBuffer->setElementSize(sizeof(RGBSpectrum));
        {
            auto dstData = (RGBSpectrum*)m_outputBuffer->map();
            std::fill_n(dstData, m_width * m_height, RGBSpectrum::Zero());
            m_outputBuffer->unmap();
        }
        m_optixContext["VLR::pv_outputBuffer"]->set(m_outputBuffer);

        m_rngBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_width, m_height);
        m_rngBuffer->setElementSize(sizeof(uint64_t));
        {
            std::mt19937_64 rng(591842031321323413);

            auto dstData = (uint64_t*)m_rngBuffer->map();
            for (int y = 0; y < m_height; ++y) {
                for (int x = 0; x < m_width; ++x) {
                    dstData[y * m_width + x] = rng();
                }
            }
            m_rngBuffer->unmap();
        }
        //m_rngBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_width, m_height);
        //m_rngBuffer->setElementSize(sizeof(uint32_t) * 4);
        //{
        //    std::mt19937 rng(591031321);

        //    auto dstData = (uint32_t*)m_rngBuffer->map();
        //    for (int y = 0; y < m_height; ++y) {
        //        for (int x = 0; x < m_width; ++x) {
        //            uint32_t index = 4 * (y * m_width + x);
        //            dstData[index + 0] = rng();
        //            dstData[index + 1] = rng();
        //            dstData[index + 2] = rng();
        //            dstData[index + 3] = rng();
        //        }
        //    }
        //    m_rngBuffer->unmap();
        //}
        m_optixContext["VLR::pv_rngBuffer"]->set(m_rngBuffer);
    }

    void* Context::mapOutputBuffer() {
        if (!m_outputBuffer)
            return nullptr;

        return m_outputBuffer->map();
    }

    void Context::unmapOutputBuffer() {
        m_outputBuffer->unmap();
    }

    void Context::render(Scene &scene, Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        optix::Context optixContext = getOptiXContext();

        scene.set();
        optix::uint2 imageSize = optix::make_uint2(m_width / shrinkCoeff, m_height / shrinkCoeff);
        optixContext["VLR::pv_imageSize"]->setUint(imageSize);

        if (firstFrame)
            m_numAccumFrames = 0;
        ++m_numAccumFrames;
        *numAccumFrames = m_numAccumFrames;

        //optixContext["VLR::pv_numAccumFrames"]->setUint(m_numAccumFrames);
        optixContext["VLR::pv_numAccumFrames"]->setUserData(sizeof(m_numAccumFrames), &m_numAccumFrames);

        camera->set();

#if defined(VLR_ENABLE_TIMEOUT_CALLBACK)
        optixContext->setTimeoutCallback([]() { return 1; }, 0.1);
#endif

#if defined(VLR_ENABLE_VALIDATION)
        optixContext->validate();
#endif

        optixContext->launch(0, imageSize.x, imageSize.y);
    }



    uint32_t Context::allocateNodeProcedureSet() {
        uint32_t index = m_nodeProcSetSlotManager.getFirstAvailableSlot();
        m_nodeProcSetSlotManager.setInUse(index);
        return index;
    }

    void Context::releaseNodeProcedureSet(uint32_t index) {
        VLRAssert(m_nodeProcSetSlotManager.getUsage(index), "Invalid index.");
        m_nodeProcSetSlotManager.setNotInUse(index);
    }

    void Context::updateNodeProcedureSet(uint32_t index, const Shared::NodeProcedureSet &procSet) {
        VLRAssert(m_nodeProcSetSlotManager.getUsage(index), "Invalid index.");
        auto procSets = (Shared::NodeProcedureSet*)m_optixNodeProcedureSetBuffer->map();
        procSets[index] = procSet;
        m_optixNodeProcedureSetBuffer->unmap();
    }



    uint32_t Context::allocateNodeDescriptor() {
        uint32_t index = m_nodeDescSlotManager.getFirstAvailableSlot();
        m_nodeDescSlotManager.setInUse(index);
        return index;
    }

    void Context::releaseNodeDescriptor(uint32_t index) {
        VLRAssert(m_nodeDescSlotManager.getUsage(index), "Invalid index.");
        m_nodeDescSlotManager.setNotInUse(index);
    }

    void Context::updateNodeDescriptor(uint32_t index, const Shared::NodeDescriptor &nodeDesc) {
        VLRAssert(m_nodeDescSlotManager.getUsage(index), "Invalid index.");
        auto nodeDescs = (Shared::NodeDescriptor*)m_optixNodeDescriptorBuffer->map();
        nodeDescs[index] = nodeDesc;
        m_optixNodeDescriptorBuffer->unmap();
    }



    uint32_t Context::allocateBSDFProcedureSet() {
        uint32_t index = m_bsdfProcSetSlotManager.getFirstAvailableSlot();
        m_bsdfProcSetSlotManager.setInUse(index);
        return index;
    }

    void Context::releaseBSDFProcedureSet(uint32_t index) {
        VLRAssert(m_bsdfProcSetSlotManager.getUsage(index), "Invalid index.");
        m_bsdfProcSetSlotManager.setNotInUse(index);
    }

    void Context::updateBSDFProcedureSet(uint32_t index, const Shared::BSDFProcedureSet &procSet) {
        VLRAssert(m_bsdfProcSetSlotManager.getUsage(index), "Invalid index.");
        auto procSets = (Shared::BSDFProcedureSet*)m_optixBSDFProcedureSetBuffer->map();
        procSets[index] = procSet;
        m_optixBSDFProcedureSetBuffer->unmap();
    }



    uint32_t Context::allocateEDFProcedureSet() {
        uint32_t index = m_edfProcSetSlotManager.getFirstAvailableSlot();
        m_edfProcSetSlotManager.setInUse(index);
        return index;
    }

    void Context::releaseEDFProcedureSet(uint32_t index) {
        VLRAssert(m_edfProcSetSlotManager.getUsage(index), "Invalid index.");
        m_edfProcSetSlotManager.setNotInUse(index);
    }

    void Context::updateEDFProcedureSet(uint32_t index, const Shared::EDFProcedureSet &procSet) {
        VLRAssert(m_edfProcSetSlotManager.getUsage(index), "Invalid index.");
        auto procSets = (Shared::EDFProcedureSet*)m_optixEDFProcedureSetBuffer->map();
        procSets[index] = procSet;
        m_optixEDFProcedureSetBuffer->unmap();
    }



    uint32_t Context::allocateSurfaceMaterialDescriptor() {
        uint32_t index = m_surfMatDescSlotManager.getFirstAvailableSlot();
        m_surfMatDescSlotManager.setInUse(index);
        return index;
    }

    void Context::releaseSurfaceMaterialDescriptor(uint32_t index) {
        VLRAssert(m_surfMatDescSlotManager.getUsage(index), "Invalid index.");
        m_surfMatDescSlotManager.setNotInUse(index);
    }

    void Context::updateSurfaceMaterialDescriptor(uint32_t index, const Shared::SurfaceMaterialDescriptor &matDesc) {
        VLRAssert(m_surfMatDescSlotManager.getUsage(index), "Invalid index.");
        auto matDescs = (Shared::SurfaceMaterialDescriptor*)m_optixSurfaceMaterialDescriptorBuffer->map();
        matDescs[index] = matDesc;
        m_optixSurfaceMaterialDescriptorBuffer->unmap();
    }



    // ----------------------------------------------------------------
    // Miscellaneous

    template <typename RealType>
    static optix::Buffer createBuffer(optix::Context &context, RTbuffertype type, RTsize width);

    template <>
    static optix::Buffer createBuffer<float>(optix::Context &context, RTbuffertype type, RTsize width) {
        return context->createBuffer(type, RT_FORMAT_FLOAT, width);
    }



    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numValues) {
        optix::Context optixContext = context.getOptiXContext();

        m_numValues = (uint32_t)numValues;
        m_PMF = createBuffer<RealType>(optixContext, RT_BUFFER_INPUT, m_numValues);
        m_CDF = createBuffer<RealType>(optixContext, RT_BUFFER_INPUT, m_numValues + 1);

        RealType* PMF = (RealType*)m_PMF->map();
        RealType* CDF = (RealType*)m_CDF->map();
        std::memcpy(PMF, values, sizeof(RealType) * m_numValues);

        CompensatedSum<RealType> sum(0);
        CDF[0] = 0;
        for (int i = 0; i < m_numValues; ++i) {
            sum += PMF[i];
            CDF[i + 1] = sum;
        }
        m_integral = sum;
        for (int i = 0; i < m_numValues; ++i) {
            PMF[i] /= m_integral;
            CDF[i + 1] /= m_integral;
        }

        m_CDF->unmap();
        m_PMF->unmap();
    }

    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::finalize(Context &context) {
        if (m_CDF && m_PMF) {
            m_CDF->destroy();
            m_PMF->destroy();
        }
    }

    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::getInternalType(Shared::DiscreteDistribution1DTemplate<RealType>* instance) const {
        if (m_PMF && m_CDF)
            new (instance) Shared::DiscreteDistribution1DTemplate<RealType>(m_PMF->getId(), m_CDF->getId(), m_integral, m_numValues);
    }

    template class DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numValues) {
        optix::Context optixContext = context.getOptiXContext();

        m_numValues = (uint32_t)numValues;
        m_PDF = createBuffer<RealType>(optixContext, RT_BUFFER_INPUT, m_numValues);
        m_CDF = createBuffer<RealType>(optixContext, RT_BUFFER_INPUT, m_numValues + 1);

        RealType* PDF = (RealType*)m_PDF->map();
        RealType* CDF = (RealType*)m_CDF->map();
        std::memcpy(PDF, values, sizeof(RealType) * m_numValues);

        CompensatedSum<RealType> sum{ 0 };
        CDF[0] = 0;
        for (int i = 0; i < m_numValues; ++i) {
            sum += PDF[i] / m_numValues;
            CDF[i + 1] = sum;
        }
        m_integral = sum;
        for (int i = 0; i < m_numValues; ++i) {
            PDF[i] /= sum;
            CDF[i + 1] /= sum;
        }

        m_CDF->unmap();
        m_PDF->unmap();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::finalize(Context &context) {
        if (m_CDF && m_PDF) {
            m_CDF->destroy();
            m_PDF->destroy();
        }
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::getInternalType(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const {
        new (instance) Shared::RegularConstantContinuousDistribution1DTemplate<RealType>(m_PDF->getId(), m_CDF->getId(), m_integral, m_numValues);
    }

    template class RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numD1, size_t numD2) {
        optix::Context optixContext = context.getOptiXContext();

        m_1DDists = new RegularConstantContinuousDistribution1DTemplate<RealType>[numD2];
        m_raw1DDists = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, numD2);
        m_raw1DDists->setElementSize(sizeof(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>));

        auto rawDists = (Shared::RegularConstantContinuousDistribution1DTemplate<RealType>*)m_raw1DDists->map();

        // JP: まず各行に関するDistribution1Dを作成する。
        // EN: First, create Distribution1D's for every rows.
        CompensatedSum<RealType> sum(0);
        RealType* integrals = new RealType[numD2];
        for (int i = 0; i < numD2; ++i) {
            RegularConstantContinuousDistribution1D &dist = m_1DDists[i];
            dist.initialize(context, values + i * numD1, numD1);
            dist.getInternalType(&rawDists[i]);
            integrals[i] = dist.getIntegral();
            sum += integrals[i];
        }

        // JP: 各行の積分値を用いてDistribution1Dを作成する。
        // EN: create a Distribution1D using integral values of each row.
        m_top1DDist.initialize(context, integrals, numD2);
        delete[] integrals;

        VLRAssert(std::isfinite(m_top1DDist.getIntegral()), "invalid integral value.");

        m_raw1DDists->unmap();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::finalize(Context &context) {
        m_top1DDist.finalize(context);

        for (int i = m_top1DDist.getNumValues() - 1; i >= 0; --i) {
            m_1DDists[i].finalize(context);
        }

        m_raw1DDists->destroy();
        delete[] m_1DDists;
        m_1DDists = nullptr;
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::getInternalType(Shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const {
        Shared::RegularConstantContinuousDistribution1DTemplate<RealType> top1DDist;
        m_top1DDist.getInternalType(&top1DDist);
        new (instance) Shared::RegularConstantContinuousDistribution2DTemplate<RealType>(m_raw1DDists->getId(), top1DDist);
    }

    template class RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
}
