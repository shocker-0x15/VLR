#include "context.h"

#include <random>

#include "scene.h"

namespace VLR {
    const cudau::BufferType g_bufferType = cudau::BufferType::Device;

    std::string readTxtFile(const filesystem::path& filepath) {
        std::ifstream ifs;
        ifs.open(filepath, std::ios::in);
        if (ifs.fail()) {
            vlrprintf("Failed to read the text file: %ls", filepath.c_str());
            return "";
        }

        std::stringstream sstream;
        sstream << ifs.rdbuf();

        return std::string(sstream.str());
    };



    Object::Object(Context &context) : m_context(context) {
    }

#define VLR_DEFINE_CLASS_ID(BaseType, Type) \
    const char* Type::TypeName = #Type; \
    const ClassIdentifier Type::ClassID = ClassIdentifier(&BaseType::ClassID)

    const ClassIdentifier TypeAwareClass::ClassID = ClassIdentifier((ClassIdentifier*)nullptr);

    VLR_DEFINE_CLASS_ID(TypeAwareClass, Object);

    VLR_DEFINE_CLASS_ID(Object, Queryable);

    VLR_DEFINE_CLASS_ID(Queryable, Image2D);
    VLR_DEFINE_CLASS_ID(Image2D, LinearImage2D);
    VLR_DEFINE_CLASS_ID(Image2D, BlockCompressedImage2D);

    VLR_DEFINE_CLASS_ID(Queryable, ShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, GeometryShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, TangentShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, Float2ShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, Float3ShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, Float4ShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, ScaleAndOffsetFloatShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, TripletSpectrumShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, RegularSampledSpectrumShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, IrregularSampledSpectrumShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, Float3ToSpectrumShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, ScaleAndOffsetUVTextureMap2DShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, Image2DTextureShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, EnvironmentTextureShaderNode);

    VLR_DEFINE_CLASS_ID(Queryable, SurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, MatteSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, SpecularReflectionSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, SpecularScatteringSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, MicrofacetReflectionSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, MicrofacetScatteringSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, LambertianScatteringSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, UE4SurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, OldStyleSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, DiffuseEmitterSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, MultiSurfaceMaterial);
    VLR_DEFINE_CLASS_ID(SurfaceMaterial, EnvironmentEmitterSurfaceMaterial);

    VLR_DEFINE_CLASS_ID(Object, Transform);
    VLR_DEFINE_CLASS_ID(Transform, StaticTransform);

    VLR_DEFINE_CLASS_ID(Object, Node);
    VLR_DEFINE_CLASS_ID(Node, SurfaceNode);
    VLR_DEFINE_CLASS_ID(SurfaceNode, TriangleMeshSurfaceNode);
    VLR_DEFINE_CLASS_ID(SurfaceNode, InfiniteSphereSurfaceNode);
    VLR_DEFINE_CLASS_ID(Node, ParentNode);
    VLR_DEFINE_CLASS_ID(ParentNode, InternalNode);
    VLR_DEFINE_CLASS_ID(ParentNode, RootNode);
    VLR_DEFINE_CLASS_ID(Object, Scene);

    VLR_DEFINE_CLASS_ID(Queryable, Camera);
    VLR_DEFINE_CLASS_ID(Camera, PerspectiveCamera);
    VLR_DEFINE_CLASS_ID(Camera, EquirectangularCamera);

#undef VLR_DEFINE_CLASS_ID



    struct EntryPoint {
        enum Value {
            PathTracing = 0,
            DebugRendering,
            NumEntryPoints
        } value;

        constexpr EntryPoint(Value v) : value(v) {}
    };



    const cudau::BufferType g_bufferType = cudau::BufferType::Device;

    uint32_t Context::NextID = 0;

    Context::Context(bool logging, uint32_t maxCallableDepth, CUcontext cuContext) {
        vlrprintf("Start initializing VLR ...");

        initializeColorSystem();

        m_ID = getInstanceID();

        m_cuContext = cuContext;
        m_optixContext = optixu::Context::create(m_cuContext);

        const filesystem::path exeDir = getExecutableDirectory();

        // ----------------------------------------------------------------
        // JP: パイプラインの作成。
        // EN: Create pipelines.

        // Path Tracing
        {
            m_optixPathTracingPipeline = m_optixContext.createPipeline();
            m_optixPathTracingPipeline.setPipelineOptions(
                2, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                VLR_DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE));
            m_optixPathTracingPipeline.setMaxTraceDepth(2);
            m_optixPathTracingPipeline.setNumMissRayTypes(Shared::RayType::NumTypes);



            m_optixPathTracingMainModule = m_optixPathTracingPipeline.createModuleFromPTXString(
                readTxtFile(exeDir / "ptxes/path_tracing.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            m_optixPathTracingRayGeneration = m_optixPathTracingPipeline.createRayGenProgram(
                m_optixPathTracingMainModule, RT_RG_NAME_STR("pathTracing"));

            m_optixPathTracingMiss = m_optixPathTracingPipeline.createRayGenProgram(
                m_optixPathTracingMainModule, RT_MS_NAME_STR("pathTracing"));

            m_optixPathTracingShadowMiss = m_optixPathTracingPipeline.createRayGenProgram(
                m_optixPathTracingEmptyModule, nullptr);

            m_optixPathTracingHitGroupDefault = m_optixPathTracingPipeline.createHitProgramGroup(
                m_optixPathTracingMainModule, RT_CH_NAME_STR("pathTracingIteration"),
                m_optixPathTracingEmptyModule, nullptr,
                m_optixPathTracingEmptyModule, nullptr);
            m_optixPathTracingHitGroupWithAlpha = m_optixPathTracingPipeline.createHitProgramGroup(
                m_optixPathTracingMainModule, RT_CH_NAME_STR("pathTracingIteration"),
                m_optixPathTracingMainModule, RT_AH_NAME_STR("withAlpha"),
                m_optixPathTracingEmptyModule, nullptr);

            m_optixPathTracingHitGroupShadowDefault = m_optixPathTracingPipeline.createHitProgramGroup(
                m_optixPathTracingEmptyModule, nullptr,
                m_optixPathTracingMainModule, RT_AH_NAME_STR("shadowDefault"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixPathTracingHitGroupShadowWithAlpha = m_optixPathTracingPipeline.createHitProgramGroup(
                m_optixPathTracingEmptyModule, nullptr,
                m_optixPathTracingMainModule, RT_AH_NAME_STR("shadowWithAlpha"),
                m_optixPathTracingEmptyModule, nullptr);



            m_optixPathTracingPipeline.setRayGenerationProgram(m_optixPathTracingRayGeneration);
            m_optixPathTracingPipeline.setMissProgram(Shared::RayType::ClosestSearch, m_optixPathTracingMiss);
            m_optixPathTracingPipeline.setMissProgram(Shared::RayType::Shadow, m_optixPathTracingShadowMiss);
        }

        // Debug Rendering
        {
            m_optixDebugRenderingPipeline = m_optixContext.createPipeline();
            m_optixDebugRenderingPipeline.setPipelineOptions(
                2, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                VLR_DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE));
            m_optixDebugRenderingPipeline.setMaxTraceDepth(1);
            m_optixDebugRenderingPipeline.setNumMissRayTypes(Shared::DebugRayType::NumTypes);



            m_optixDebugRenderingMainModule = m_optixDebugRenderingPipeline.createModuleFromPTXString(
                readTxtFile(exeDir / "ptxes/debug_rendering.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            m_optixDebugRenderingRayGeneration = m_optixDebugRenderingPipeline.createRayGenProgram(
                m_optixDebugRenderingMainModule, RT_RG_NAME_STR("debugRendering"));

            m_optixDebugRenderingMiss = m_optixDebugRenderingPipeline.createRayGenProgram(
                m_optixDebugRenderingMainModule, RT_MS_NAME_STR("debugRendering"));

            m_optixDebugRenderingHitGroupDefault = m_optixDebugRenderingPipeline.createHitProgramGroup(
                m_optixDebugRenderingMainModule, RT_CH_NAME_STR("debugRendering"),
                m_optixDebugRenderingEmptyModule, nullptr,
                m_optixDebugRenderingEmptyModule, nullptr);
            m_optixDebugRenderingHitGroupWithAlpha = m_optixDebugRenderingPipeline.createHitProgramGroup(
                m_optixDebugRenderingMainModule, RT_CH_NAME_STR("debugRendering"),
                m_optixDebugRenderingMainModule, RT_AH_NAME_STR("debugRenderingWithAlpha"),
                m_optixDebugRenderingEmptyModule, nullptr);



            m_optixDebugRenderingPipeline.setRayGenerationProgram(m_optixDebugRenderingRayGeneration);
            m_optixDebugRenderingPipeline.setMissProgram(Shared::DebugRayType::Primary, m_optixDebugRenderingMiss);
        }

        // END: Create pipelines.
        // ----------------------------------------------------------------

        CUDADRV_CHECK(cuModuleLoad(&m_cudaPostProcessModule, (exeDir / "ptxes/path_tracing.ptx").string().c_str()));
        m_cudaPostProcessConvertToRGB.set(m_cudaPostProcessModule, "convertToRGB", dim3(8, 8), 0);



#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        const uint32_t NumSpectrumGridCells = 168;
        const uint32_t NumSpectrumDataPoints = 186;
        m_optixBufferUpsampledSpectrum_spectrum_grid.initialize(m_cuContext, g_bufferType, NumSpectrumGridCells);
        {
            auto values = m_optixBufferUpsampledSpectrum_spectrum_grid.map();
            std::copy_n(UpsampledSpectrum::spectrum_grid, NumSpectrumGridCells, values);
            m_optixBufferUpsampledSpectrum_spectrum_grid.unmap();
        }
        m_optixBufferUpsampledSpectrum_spectrum_data_points.initialize(m_cuContext, g_bufferType, NumSpectrumDataPoints);
        {
            auto values = m_optixBufferUpsampledSpectrum_spectrum_data_points.map();
            std::copy_n(UpsampledSpectrum::spectrum_data_points, NumSpectrumDataPoints, values);
            m_optixBufferUpsampledSpectrum_spectrum_data_points.unmap();
        }
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        m_optixBufferUpsampledSpectrum_maxBrightnesses = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, UpsampledSpectrum::kTableResolution);
        {
            auto values = (float*)m_optixBufferUpsampledSpectrum_maxBrightnesses->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy_n(UpsampledSpectrum::maxBrightnesses, UpsampledSpectrum::kTableResolution, values);
            m_optixBufferUpsampledSpectrum_maxBrightnesses->unmap();
        }
        m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65 = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 3 * pow3(UpsampledSpectrum::kTableResolution));
        m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65->setElementSize(sizeof(UpsampledSpectrum::PolynomialCoefficients));
        {
            auto values = (UpsampledSpectrum::PolynomialCoefficients*)m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy_n(UpsampledSpectrum::coefficients_sRGB_D65, 3 * pow3(UpsampledSpectrum::kTableResolution), values);
            m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65->unmap();
        }
        m_optixBufferUpsampledSpectrum_coefficients_sRGB_E = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 3 * pow3(UpsampledSpectrum::kTableResolution));
        m_optixBufferUpsampledSpectrum_coefficients_sRGB_E->setElementSize(sizeof(UpsampledSpectrum::PolynomialCoefficients));
        {
            auto values = (UpsampledSpectrum::PolynomialCoefficients*)m_optixBufferUpsampledSpectrum_coefficients_sRGB_E->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy_n(UpsampledSpectrum::coefficients_sRGB_E, 3 * pow3(UpsampledSpectrum::kTableResolution), values);
            m_optixBufferUpsampledSpectrum_coefficients_sRGB_E->unmap();
        }
        setInt32(m_optixContext, "VLR::UpsampledSpectrum_maxBrightnesses", m_optixBufferUpsampledSpectrum_maxBrightnesses->getId());
        setInt32(m_optixContext, "VLR::UpsampledSpectrum_coefficients_sRGB_D65", m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65->getId());
        setInt32(m_optixContext, "VLR::UpsampledSpectrum_coefficients_sRGB_E", m_optixBufferUpsampledSpectrum_coefficients_sRGB_E->getId());
#endif
        
        m_discretizedSpectrumCMFs.initialize(m_cuContext, g_bufferType, 3);
        {
            auto CMFs = m_discretizedSpectrumCMFs.map();
            CMFs[0] = DiscretizedSpectrumAlwaysSpectral::xbar;
            CMFs[1] = DiscretizedSpectrumAlwaysSpectral::ybar;
            CMFs[2] = DiscretizedSpectrumAlwaysSpectral::zbar;
            m_discretizedSpectrumCMFs.unmap();
        }



        m_optixMaterialDefault = m_optixContext.createMaterial();
        m_optixMaterialDefault.setHitGroup(Shared::RayType::ClosestSearch, m_optixPathTracingHitGroupDefault);
        m_optixMaterialDefault.setHitGroup(Shared::RayType::Shadow, m_optixPathTracingHitGroupShadowDefault);
        m_optixMaterialDefault.setHitGroup(Shared::DebugRayType::Primary, m_optixDebugRenderingHitGroupDefault);

        m_optixMaterialWithAlpha = m_optixContext.createMaterial();
        m_optixMaterialWithAlpha.setHitGroup(Shared::RayType::ClosestSearch, m_optixPathTracingHitGroupWithAlpha);
        m_optixMaterialWithAlpha.setHitGroup(Shared::RayType::Shadow, m_optixPathTracingHitGroupShadowWithAlpha);
        m_optixMaterialWithAlpha.setHitGroup(Shared::DebugRayType::Primary, m_optixDebugRenderingHitGroupWithAlpha);



        m_nodeProcedureBuffer.initialize(m_optixContext, 256);



        m_smallNodeDescriptorBuffer.initialize(m_optixContext, 8192);
        m_mediumNodeDescriptorBuffer.initialize(m_optixContext, 8192);
        m_largeNodeDescriptorBuffer.initialize(m_optixContext, 1024);

        m_BSDFProcedureBuffer.initialize(m_optixContext, 64);
        m_EDFProcedureBuffer.initialize(m_optixContext, 64);

        {
            m_optixPathTracingMaterialModule = m_optixPathTracingPipeline.createModuleFromPTXString(
                readTxtFile(exeDir / "ptxes/materials.ptx"),
                OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            m_optixCallableProgramNullBSDF_setupBSDF = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_setupBSDF"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullBSDF_getBaseColor = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_getBaseColor"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullBSDF_matches = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_matches"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullBSDF_sampleInternal = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_sampleInternal"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullBSDF_evaluateInternal = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_evaluateInternal"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullBSDF_evaluatePDFInternal = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_evaluatePDFInternal"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullBSDF_weightInternal = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullBSDF_weightInternal"),
                m_optixPathTracingEmptyModule, nullptr);

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



            m_optixCallableProgramNullEDF_setupEDF = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullEDF_setupEDF"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullEDF_evaluateEmittanceInternal = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullEDF_evaluateEmittanceInternal"),
                m_optixPathTracingEmptyModule, nullptr);
            m_optixCallableProgramNullEDF_evaluateInternal = m_optixPathTracingPipeline.createCallableGroup(
                m_optixPathTracingMaterialModule, RT_DC_NAME_STR("NullEDF_evaluateInternal"),
                m_optixPathTracingEmptyModule, nullptr);

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = m_optixCallableProgramNullEDF_evaluateEmittanceInternal->getId();
                edfProcSet.progEvaluateInternal = m_optixCallableProgramNullEDF_evaluateInternal->getId();
            }
            m_nullEDFProcedureSetIndex = allocateEDFProcedureSet();
            updateEDFProcedureSet(m_nullEDFProcedureSetIndex, edfProcSet);
            VLRAssert(m_nullEDFProcedureSetIndex == 0, "Index of the null EDF procedure set is expected to be 0.");
        }

        m_surfaceMaterialDescriptorBuffer.initialize(m_optixContext, 8192, "VLR::pv_materialDescriptorBuffer");

        Image2D::initialize(*this);
        ShaderNode::initialize(*this);
        SurfaceMaterial::initialize(*this);
        SurfaceNode::initialize(*this);
        Camera::initialize(*this);

        vlrprintf(" done.\n");



        RTsize defaultStackSize = 0;
        if (m_RTXEnabled) {
            if (stackSize > 0)
                vlrprintf("Specified stack size is ignored in RTX mode.\n");
        }
        else {
            defaultStackSize = m_optixContext->getStackSize();

            vlrprintf("Default Stack Size: %llu\n", defaultStackSize);
            vlrprintf("Requested Stack Size: %u\n", stackSize);
        }

        if (logging) {
            m_optixContext->setPrintEnabled(true);
            m_optixContext->setPrintBufferSize(4096);
            //m_optixContext->setExceptionEnabled(RT_EXCEPTION_BUFFER_ID_INVALID, true);
            //m_optixContext->setExceptionEnabled(RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true);
            //m_optixContext->setExceptionEnabled(RT_EXCEPTION_INTERNAL_ERROR, true);
            //m_optixContext->setPrintLaunchIndex(0, 0, 0);
            if (stackSize == 0)
                stackSize = 3072;
        }
        else {
            m_optixContext->setExceptionEnabled(RT_EXCEPTION_STACK_OVERFLOW, false);
            if (stackSize == 0)
                stackSize = 2560;
        }

        if (m_RTXEnabled) {
            m_optixContext->setMaxTraceDepth(2); // Iterative path tracing needs only depth 2 (shadow ray in closest hit program).
            m_optixContext->setMaxCallableProgramDepth(std::max<uint32_t>(3, maxCallableDepth));
        }
        else {
            // Dirty hack for OptiX stack size management where the relation between set/getStackSize() is inconsistent
            // depending on a device or a version of OptiX.
            m_optixContext->setStackSize(defaultStackSize);
            stackSize = stackSize * defaultStackSize / m_optixContext->getStackSize();

            m_optixContext->setStackSize(stackSize);
            RTsize actuallyUsedStackSize = m_optixContext->getStackSize();
            vlrprintf("Stack Size: %llu\n", actuallyUsedStackSize);
        }
    }

    Context::~Context() {
        if (m_rngBuffer)
            m_rngBuffer->destroy();

        if (m_rawOutputBuffer)
            m_rawOutputBuffer->destroy();

        if (m_outputBuffer)
            m_outputBuffer->destroy();

        Camera::finalize(*this);
        SurfaceNode::finalize(*this);
        SurfaceMaterial::finalize(*this);
        ShaderNode::finalize(*this);
        Image2D::finalize(*this);

        m_surfaceMaterialDescriptorBuffer.finalize();

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

        m_EDFProcedureBuffer.finalize();
        m_BSDFProcedureBuffer.finalize();

        m_largeNodeDescriptorBuffer.finalize();
        m_mediumNodeDescriptorBuffer.finalize();
        m_smallNodeDescriptorBuffer.finalize();

        m_nodeProcedureBuffer.finalize();

        m_optixMaterialWithAlpha->destroy();
        m_optixMaterialDefault->destroy();

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        m_optixBufferUpsampledSpectrum_spectrum_data_points->destroy();
        m_optixBufferUpsampledSpectrum_spectrum_grid->destroy();
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        m_optixBufferUpsampledSpectrum_coefficients_sRGB_E->destroy();
        m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65->destroy();
        m_optixBufferUpsampledSpectrum_maxBrightnesses->destroy();
#endif

        m_optixProgramConvertToRGB->destroy();

        m_optixProgramDebugRenderingException->destroy();
        m_optixProgramDebugRenderingRayGeneration->destroy();
        m_optixProgramDebugRenderingMiss->destroy();
        m_optixProgramDebugRenderingAnyHitWithAlpha->destroy();
        m_optixProgramDebugRenderingClosestHit->destroy();

        m_optixProgramException->destroy();
        m_optixProgramPathTracingMiss->destroy();
        m_optixProgramPathTracing->destroy();

        m_optixProgramPathTracingIteration->destroy();
        m_optixProgramShadowAnyHitWithAlpha->destroy();
        m_optixProgramAnyHitWithAlpha->destroy();
        m_optixProgramShadowAnyHitDefault->destroy();

        m_optixContext.destroy();

        finalizeColorSystem();
    }

    void Context::bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID) {
        if (m_outputBuffer.isInitialized())
            m_outputBuffer.finalize();
        if (m_rawOutputBuffer.isInitialized())
            m_rawOutputBuffer.finalize();
        if (m_rngBuffer.isInitialized())
            m_rngBuffer.finalize();

        m_width = width;
        m_height = height;

        if (glBufferID > 0) {
            m_outputBuffer.initializeFromGLTexture2D(m_cuContext, glBufferID, cudau::ArraySurface::Enable);
        }
        else {
            m_outputBuffer.initialize2D(m_cuContext, cudau::ArrayElementType::Float32, 4, cudau::ArraySurface::Enable,
                                        m_width, m_height, 1);
        }
        m_optixContext["VLR::pv_RGBBuffer"]->set(m_outputBuffer);

        m_rawOutputBuffer.initialize(m_cuContext, g_bufferType, m_width, m_height);
        m_optixContext["VLR::pv_spectrumBuffer"]->set(m_rawOutputBuffer);
        m_optixContext["VLR::pv_outputBuffer"]->set(m_rawOutputBuffer);

        m_rngBuffer.initialize(m_cuContext, g_bufferType, m_width, m_height);
        {
            std::mt19937_64 rng(591842031321323413);

            m_rngBuffer.map();
            for (int y = 0; y < m_height; ++y) {
                for (int x = 0; x < m_width; ++x) {
                    m_rngBuffer[uint2{ x, y }] = Shared::KernelRNG(rng());
                }
            }
            m_rngBuffer.unmap();
        }
        //m_rngBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_width, m_height);
        //m_rngBuffer->setElementSize(sizeof(uint32_t) * 4);
        //{
        //    std::mt19937 rng(591031321);

        //    auto dstData = (uint32_t*)m_rngBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
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

    const cudau::Array &Context::getOutputBuffer() {
        return m_outputBuffer;
    }

    void Context::getOutputBufferSize(uint32_t* width, uint32_t* height) {
        *width = m_width;
        *height = m_height;
    }

    void Context::render(Scene &scene, const Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        optix::Context optixContext = getOptiXContext();

        optix::uint2 imageSize = optix::make_uint2(m_width / shrinkCoeff, m_height / shrinkCoeff);
        if (firstFrame) {
            scene.setup();
            camera->setup();

            optixContext["VLR::pv_imageSize"]->setUint(imageSize);

            m_numAccumFrames = 0;
        }

        ++m_numAccumFrames;
        *numAccumFrames = m_numAccumFrames;
        //optixContext["VLR::pv_numAccumFrames"]->setUint(m_numAccumFrames);
        optixContext["VLR::pv_numAccumFrames"]->setUserData(sizeof(m_numAccumFrames), &m_numAccumFrames);

#if defined(VLR_ENABLE_TIMEOUT_CALLBACK)
        optixContext->setTimeoutCallback([]() { return 1; }, 0.1);
#endif

#if defined(VLR_ENABLE_VALIDATION)
        optixContext->validate();
#endif

        optixContext->launch(EntryPoint::PathTracing, imageSize.x, imageSize.y);

        optixContext->launch(EntryPoint::ConvertToRGB, imageSize.x, imageSize.y);
    }

    void Context::debugRender(Scene &scene, const Camera* camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        optix::Context optixContext = getOptiXContext();

        optix::uint2 imageSize = optix::make_uint2(m_width / shrinkCoeff, m_height / shrinkCoeff);
        if (firstFrame) {
            scene.setup();
            camera->setup();

            optixContext["VLR::pv_imageSize"]->setUint(imageSize);

            m_numAccumFrames = 0;
        }

        ++m_numAccumFrames;
        *numAccumFrames = m_numAccumFrames;
        //optixContext["VLR::pv_numAccumFrames"]->setUint(m_numAccumFrames);
        optixContext["VLR::pv_numAccumFrames"]->setUserData(sizeof(m_numAccumFrames), &m_numAccumFrames);

#if defined(VLR_ENABLE_TIMEOUT_CALLBACK)
        optixContext->setTimeoutCallback([]() { return 1; }, 0.1);
#endif

#if defined(VLR_ENABLE_VALIDATION)
        optixContext->validate();
#endif

        auto attr = Shared::DebugRenderingAttribute((Shared::DebugRenderingAttribute)renderMode);
        optixContext["VLR::pv_debugRenderingAttribute"]->setUserData(sizeof(attr), &attr);
        optixContext->launch(EntryPoint::DebugRendering, imageSize.x, imageSize.y);

        optixContext->launch(EntryPoint::ConvertToRGB, imageSize.x, imageSize.y);
    }



    uint32_t Context::allocateNodeProcedureSet() {
        return m_nodeProcedureBuffer.allocate();
    }
    void Context::releaseNodeProcedureSet(uint32_t index) {
        m_nodeProcedureBuffer.release(index);
    }
    void Context::updateNodeProcedureSet(uint32_t index, const Shared::NodeProcedureSet &procSet) {
        m_nodeProcedureBuffer.update(index, procSet);
    }



    uint32_t Context::allocateSmallNodeDescriptor() {
        return m_smallNodeDescriptorBuffer.allocate();
    }
    void Context::releaseSmallNodeDescriptor(uint32_t index) {
        m_smallNodeDescriptorBuffer.release(index);
    }
    void Context::updateSmallNodeDescriptor(uint32_t index, const Shared::SmallNodeDescriptor &nodeDesc) {
        m_smallNodeDescriptorBuffer.update(index, nodeDesc);
    }



    uint32_t Context::allocateMediumNodeDescriptor() {
        return m_mediumNodeDescriptorBuffer.allocate();
    }
    void Context::releaseMediumNodeDescriptor(uint32_t index) {
        m_mediumNodeDescriptorBuffer.release(index);
    }
    void Context::updateMediumNodeDescriptor(uint32_t index, const Shared::MediumNodeDescriptor &nodeDesc) {
        m_mediumNodeDescriptorBuffer.update(index, nodeDesc);
    }



    uint32_t Context::allocateLargeNodeDescriptor() {
        return m_largeNodeDescriptorBuffer.allocate();
    }
    void Context::releaseLargeNodeDescriptor(uint32_t index) {
        m_largeNodeDescriptorBuffer.release(index);
    }
    void Context::updateLargeNodeDescriptor(uint32_t index, const Shared::LargeNodeDescriptor &nodeDesc) {
        m_largeNodeDescriptorBuffer.update(index, nodeDesc);
    }



    uint32_t Context::allocateBSDFProcedureSet() {
        return m_BSDFProcedureBuffer.allocate();
    }
    void Context::releaseBSDFProcedureSet(uint32_t index) {
        m_BSDFProcedureBuffer.release(index);
    }
    void Context::updateBSDFProcedureSet(uint32_t index, const Shared::BSDFProcedureSet &procSet) {
        m_BSDFProcedureBuffer.update(index, procSet);
    }



    uint32_t Context::allocateEDFProcedureSet() {
        return m_EDFProcedureBuffer.allocate();
    }
    void Context::releaseEDFProcedureSet(uint32_t index) {
        m_EDFProcedureBuffer.release(index);
    }
    void Context::updateEDFProcedureSet(uint32_t index, const Shared::EDFProcedureSet &procSet) {
        m_EDFProcedureBuffer.update(index, procSet);
    }



    uint32_t Context::allocateSurfaceMaterialDescriptor() {
        return m_surfaceMaterialDescriptorBuffer.allocate();
    }
    void Context::releaseSurfaceMaterialDescriptor(uint32_t index) {
        m_surfaceMaterialDescriptorBuffer.release(index);
    }
    void Context::updateSurfaceMaterialDescriptor(uint32_t index, const Shared::SurfaceMaterialDescriptor &matDesc) {
        m_surfaceMaterialDescriptorBuffer.update(index, matDesc);
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
        optixu::Context optixContext = context.getOptiXContext();

        m_numValues = (uint32_t)numValues;
        m_PMF.initialize(optixContext.getCUcontext(), s_bufferType, m_numValues);
        m_CDF.initialize(optixContext.getCUcontext(), s_bufferType, m_numValues + 1);

        RealType* PMF = m_PMF.map();
        RealType* CDF = m_CDF.map();
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

        m_CDF.unmap();
        m_PMF.unmap();
    }

    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::finalize(Context &context) {
        if (m_CDF.isInitialize() && m_PMF.isInitialize()) {
            m_CDF.finalize();
            m_PMF.finalize();
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
