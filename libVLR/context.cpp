#include "context.h"

#include <random>

#include "scene.h"

namespace vlr {
    cudau::BufferType g_bufferType = cudau::BufferType::Device;

    std::string readTxtFile(const std::filesystem::path& filepath) {
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



    uint32_t Context::NextID = 0;

    Context::Context(CUcontext cuContext, bool logging, uint32_t maxCallableDepth) {
        vlrprintf("Start initializing VLR ...");

        initializeColorSystem();

        m_ID = getInstanceID();

        m_cuContext = cuContext;

        m_optix = {};
        m_optix.nodeProcedureSetBuffer.initialize(m_cuContext, 256);
        m_optix.smallNodeDescriptorBuffer.initialize(m_cuContext, 8192);
        m_optix.mediumNodeDescriptorBuffer.initialize(m_cuContext, 8192);
        m_optix.largeNodeDescriptorBuffer.initialize(m_cuContext, 1024);
        m_optix.bsdfProcedureSetBuffer.initialize(m_cuContext, 64);
        m_optix.edfProcedureSetBuffer.initialize(m_cuContext, 64);
        m_optix.launchParams.nodeProcedureSetBuffer = m_optix.nodeProcedureSetBuffer.optixBuffer.getDevicePointer();
        m_optix.launchParams.smallNodeDescriptorBuffer = m_optix.smallNodeDescriptorBuffer.optixBuffer.getDevicePointer();
        m_optix.launchParams.mediumNodeDescriptorBuffer = m_optix.mediumNodeDescriptorBuffer.optixBuffer.getDevicePointer();
        m_optix.launchParams.largeNodeDescriptorBuffer = m_optix.largeNodeDescriptorBuffer.optixBuffer.getDevicePointer();
        m_optix.launchParams.bsdfProcedureSetBuffer = m_optix.bsdfProcedureSetBuffer.optixBuffer.getDevicePointer();
        m_optix.launchParams.edfProcedureSetBuffer = m_optix.edfProcedureSetBuffer.optixBuffer.getDevicePointer();

        m_optix.surfaceMaterialDescriptorBuffer.initialize(m_cuContext, 8192);
        m_optix.launchParams.materialDescriptorBuffer = m_optix.surfaceMaterialDescriptorBuffer.optixBuffer.getDevicePointer();

        m_optix.context = optixu::Context::create(cuContext/*, 4, true*/);

        m_optix.pipeline = m_optix.context.createPipeline();
        m_optix.pipeline.setPipelineOptions(
            8, 2,
            "plp", sizeof(shared::PipelineLaunchParameters),
            false,
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            VLR_DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                             OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                             OPTIX_EXCEPTION_FLAG_DEBUG,
                             OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW),
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
        m_optix.pipeline.setNumMissRayTypes(shared::RayType::NumTypes);
        m_optix.pipeline.setNumCallablePrograms(512);

        const std::filesystem::path exeDir = getExecutableDirectory();

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/path_tracing.ptx");
            m_optix.pathTracingModule = m_optix.pipeline.createModuleFromPTXString(
                ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
                VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            m_optix.pathTracingRayGeneration = m_optix.pipeline.createRayGenProgram(
                m_optix.pathTracingModule, RT_RG_NAME_STR("pathTracing"));

            m_optix.pathTracingMiss = m_optix.pipeline.createMissProgram(
                m_optix.pathTracingModule, RT_MS_NAME_STR("pathTracingMiss"));
            m_optix.pathTracingShadowMiss = m_optix.pipeline.createMissProgram(
                optixu::Module(), nullptr);

            m_optix.pathTracingHitGroupDefault = m_optix.pipeline.createHitProgramGroupForBuiltinIS(
                OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                m_optix.pathTracingModule, RT_CH_NAME_STR("pathTracingIteration"),
                optixu::Module(), nullptr);
            m_optix.pathTracingHitGroupWithAlpha = m_optix.pipeline.createHitProgramGroupForBuiltinIS(
                OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                m_optix.pathTracingModule, RT_CH_NAME_STR("pathTracingIteration"),
                m_optix.pathTracingModule, RT_AH_NAME_STR("anyHitWithAlpha"));
            m_optix.pathTracingHitGroupShadowDefault = m_optix.pipeline.createHitProgramGroupForBuiltinIS(
                OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                optixu::Module(), nullptr,
                m_optix.pathTracingModule, RT_AH_NAME_STR("shadowAnyHitDefault"));
            m_optix.pathTracingHitGroupShadowWithAlpha = m_optix.pipeline.createHitProgramGroupForBuiltinIS(
                OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                optixu::Module(), nullptr,
                m_optix.pathTracingModule, RT_AH_NAME_STR("shadowAnyHitWithAlpha"));
        }

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/debug_rendering.ptx");
            m_optix.debugRenderingModule = m_optix.pipeline.createModuleFromPTXString(
                ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
                VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            m_optix.debugRenderingRayGeneration = m_optix.pipeline.createRayGenProgram(
                m_optix.debugRenderingModule, RT_RG_NAME_STR("debugRenderingRayGeneration"));

            m_optix.debugRenderingMiss = m_optix.pipeline.createMissProgram(
                m_optix.debugRenderingModule, RT_MS_NAME_STR("debugRenderingMiss"));

            m_optix.debugRenderingHitGroupDefault = m_optix.pipeline.createHitProgramGroupForBuiltinIS(
                OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                m_optix.debugRenderingModule, RT_CH_NAME_STR("debugRenderingClosestHit"),
                optixu::Module(), nullptr);
            m_optix.debugRenderingHitGroupWithAlpha = m_optix.pipeline.createHitProgramGroupForBuiltinIS(
                OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                m_optix.debugRenderingModule, RT_CH_NAME_STR("debugRenderingClosestHit"),
                m_optix.debugRenderingModule, RT_AH_NAME_STR("debugRenderingAnyHitWithAlpha"));
        }

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/materials.ptx");
            m_optix.nullDFModule = m_optix.pipeline.createModuleFromPTXString(
                ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                VLR_DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_LEVEL_3),
                VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            m_optix.dcNullBSDF_setupBSDF = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_setupBSDF"));
            m_optix.dcNullBSDF_getBaseColor = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_getBaseColor"));
            m_optix.dcNullBSDF_matches = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_matches"));
            m_optix.dcNullBSDF_sampleInternal = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_sampleInternal"));
            m_optix.dcNullBSDF_evaluateInternal = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_evaluateInternal"));
            m_optix.dcNullBSDF_evaluatePDFInternal = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_evaluatePDFInternal"));
            m_optix.dcNullBSDF_weightInternal = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullBSDF_weightInternal"));

            shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = m_optix.dcNullBSDF_getBaseColor;
                bsdfProcSet.progMatches = m_optix.dcNullBSDF_matches;
                bsdfProcSet.progSampleInternal = m_optix.dcNullBSDF_sampleInternal;
                bsdfProcSet.progEvaluateInternal = m_optix.dcNullBSDF_evaluateInternal;
                bsdfProcSet.progEvaluatePDFInternal = m_optix.dcNullBSDF_evaluatePDFInternal;
                bsdfProcSet.progWeightInternal = m_optix.dcNullBSDF_weightInternal;
            }
            m_optix.nullBSDFProcedureSetIndex = allocateBSDFProcedureSet();
            updateBSDFProcedureSet(m_optix.nullBSDFProcedureSetIndex, bsdfProcSet);
            VLRAssert(m_optix.nullBSDFProcedureSetIndex == 0, "Index of the null BSDF procedure set is expected to be 0.");

            m_optix.dcNullEDF_setupEDF = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullEDF_setupEDF"));
            m_optix.dcNullEDF_evaluateEmittanceInternal = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullEDF_evaluateEmittanceInternal"));
            m_optix.dcNullEDF_evaluateInternal = createDirectCallableProgram(
                m_optix.nullDFModule, RT_DC_NAME_STR("NullEDF_evaluateInternal"));

            shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = m_optix.dcNullEDF_evaluateEmittanceInternal;
                edfProcSet.progEvaluateInternal = m_optix.dcNullEDF_evaluateInternal;
            }
            m_optix.nullEDFProcedureSetIndex = allocateEDFProcedureSet();
            updateEDFProcedureSet(m_optix.nullEDFProcedureSetIndex, edfProcSet);
            VLRAssert(m_optix.nullEDFProcedureSetIndex == 0, "Index of the null EDF procedure set is expected to be 0.");
        }

        m_optix.pipeline.setRayGenerationProgram(m_optix.pathTracingRayGeneration);
        m_optix.pipeline.setMissProgram(shared::RayType::Closest, m_optix.pathTracingMiss);
        m_optix.pipeline.setMissProgram(shared::RayType::Shadow, m_optix.pathTracingShadowMiss);
        m_optix.pipeline.setMissProgram(shared::RayType::DebugPrimary, m_optix.debugRenderingMiss);

        m_optix.materialDefault = m_optix.context.createMaterial();
        m_optix.materialDefault.setHitGroup(shared::RayType::Closest, m_optix.pathTracingHitGroupDefault);
        m_optix.materialDefault.setHitGroup(shared::RayType::Shadow, m_optix.pathTracingHitGroupShadowDefault);
        m_optix.materialDefault.setHitGroup(shared::RayType::DebugPrimary, m_optix.debugRenderingHitGroupDefault);

        m_optix.materialWithAlpha = m_optix.context.createMaterial();
        m_optix.materialWithAlpha.setHitGroup(shared::RayType::Closest, m_optix.pathTracingHitGroupWithAlpha);
        m_optix.materialWithAlpha.setHitGroup(shared::RayType::Shadow, m_optix.pathTracingHitGroupShadowWithAlpha);
        m_optix.materialWithAlpha.setHitGroup(shared::RayType::DebugPrimary, m_optix.debugRenderingHitGroupWithAlpha);



        m_optix.scene = m_optix.context.createScene();

        m_optix.geomInstBuffer.initialize(cuContext, 65536);
        m_optix.instBuffer.initialize(cuContext, 65536);
        m_optix.launchParams.geomInstBuffer = m_optix.geomInstBuffer.optixBuffer.getDevicePointer();
        m_optix.launchParams.instBuffer = m_optix.instBuffer.optixBuffer.getDevicePointer();



        m_optix.launchParams.DiscretizedSpectrum_xbar = DiscretizedSpectrumAlwaysSpectral::xbar;
        m_optix.launchParams.DiscretizedSpectrum_ybar = DiscretizedSpectrumAlwaysSpectral::ybar;
        m_optix.launchParams.DiscretizedSpectrum_zbar = DiscretizedSpectrumAlwaysSpectral::zbar;
        m_optix.launchParams.DiscretizedSpectrum_integralCMF = DiscretizedSpectrumAlwaysSpectral::integralCMF;

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        const uint32_t NumSpectrumGridCells = 168;
        const uint32_t NumSpectrumDataPoints = 186;
        m_optix.UpsampledSpectrum_spectrum_grid.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::spectrum_grid, NumSpectrumGridCells);
        m_optix.UpsampledSpectrum_spectrum_data_points.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::spectrum_data_points, NumSpectrumDataPoints);
        m_optix.launchParams.UpsampledSpectrum_spectrum_grid = m_optix.UpsampledSpectrum_spectrum_grid.getDevicePointer();
        m_optix.launchParams.UpsampledSpectrum_spectrum_data_points = m_optix.UpsampledSpectrum_spectrum_data_points.getDevicePointer();
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        m_optix.UpsampledSpectrum_maxBrightnesses.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::maxBrightnesses, UpsampledSpectrum::kTableResolution);
        m_optix.UpsampledSpectrum_coefficients_sRGB_D65.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::coefficients_sRGB_D65, 3 * pow3(UpsampledSpectrum::kTableResolution));
        m_optix.UpsampledSpectrum_coefficients_sRGB_E.initialize(
            m_cuContext, g_bufferType,
            UpsampledSpectrum::coefficients_sRGB_E, 3 * pow3(UpsampledSpectrum::kTableResolution));
        m_optix.launchParams.UpsampledSpectrum_maxBrightnesses = m_optix.UpsampledSpectrum_maxBrightnesses.getDevicePointer();
        m_optix.launchParams.UpsampledSpectrum_coefficients_sRGB_D65 = m_optix.UpsampledSpectrum_coefficients_sRGB_D65.getDevicePointer();
        m_optix.launchParams.UpsampledSpectrum_coefficients_sRGB_E = m_optix.UpsampledSpectrum_coefficients_sRGB_E.getDevicePointer();
#endif

        CUDADRV_CHECK(cuMemAlloc(&m_optix.launchParamsOnDevice, sizeof(shared::PipelineLaunchParameters)));



        m_optix.denoiser = m_optix.context.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL);
        m_optix.denoiser.setModel(OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0);
        CUDADRV_CHECK(cuMemAlloc(&m_optix.hdrIntensity, sizeof(float)));



        {
            CUDADRV_CHECK(cuModuleLoad(&m_cudaPostProcessModule, (exeDir / "ptxes/post_process.ptx").string().c_str()));

            size_t symbolSize;
            CUDADRV_CHECK(cuModuleGetGlobal(&m_cudaPostProcessModuleLaunchParamsPtr, &symbolSize,
                                            m_cudaPostProcessModule, "plp"));

            m_copyBuffers.set(m_cudaPostProcessModule, "copyBuffers", cudau::dim3(32), 0);
            m_convertToRGB.set(m_cudaPostProcessModule, "convertToRGB", cudau::dim3(32), 0);
        }



        Image2D::initialize(*this);
        ShaderNode::initialize(*this);
        SurfaceMaterial::initialize(*this);
        SurfaceNode::initialize(*this);
        Camera::initialize(*this);
        Scene::initialize(*this);

        m_optix.pipeline.link(2, VLR_DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        m_optix.pipeline.setNumCallablePrograms(m_optix.callablePrograms.size());
        for (int i = 0; i < m_optix.callablePrograms.size(); ++i)
            m_optix.pipeline.setCallableProgram(i, m_optix.callablePrograms[i]);

        size_t sbtSize;
        m_optix.pipeline.generateShaderBindingTableLayout(&sbtSize);
        m_optix.shaderBindingTable.initialize(m_cuContext, g_bufferType, sbtSize, 1);
        m_optix.shaderBindingTable.setMappedMemoryPersistent(true);
        m_optix.pipeline.setShaderBindingTable(m_optix.shaderBindingTable,
                                               m_optix.shaderBindingTable.getMappedPointer());

        vlrprintf(" done.\n");
    }

    Context::~Context() {
        m_optix.hitGroupShaderBindingTable.finalize();
        m_optix.shaderBindingTable.finalize();



        m_optix.linearDenoisedColorBuffer.finalize();
        m_optix.linearNormalBuffer.finalize();
        m_optix.linearAlbedoBuffer.finalize();
        m_optix.linearColorBuffer.finalize();
        m_optix.accumNormalBuffer.finalize();
        m_optix.accumAlbedoBuffer.finalize();
        m_optix.denoiserScratchBuffer.finalize();
        m_optix.denoiserStateBuffer.finalize();

        m_optix.outputBufferHolder.finalize();
        m_optix.outputBuffer.finalize();
        m_optix.accumBuffer.finalize();
        m_optix.rngBuffer.finalize();



        Scene::finalize(*this);
        Camera::finalize(*this);
        SurfaceNode::finalize(*this);
        SurfaceMaterial::finalize(*this);
        ShaderNode::finalize(*this);
        Image2D::finalize(*this);



        cuMemFree(m_optix.hdrIntensity);
        m_optix.denoiser.destroy();



        cuMemFree(m_optix.launchParamsOnDevice);

        m_optix.surfaceMaterialDescriptorBuffer.finalize();
        
        m_optix.edfProcedureSetBuffer.finalize();
        m_optix.bsdfProcedureSetBuffer.finalize();
        m_optix.largeNodeDescriptorBuffer.finalize();
        m_optix.mediumNodeDescriptorBuffer.finalize();
        m_optix.smallNodeDescriptorBuffer.finalize();
        m_optix.nodeProcedureSetBuffer.finalize();

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        m_optix.UpsampledSpectrum_spectrum_data_points.finalize();
        m_optix.UpsampledSpectrum_spectrum_grid.finalize();
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        m_optix.UpsampledSpectrum_coefficients_sRGB_E.finalize();
        m_optix.UpsampledSpectrum_coefficients_sRGB_D65.finalize();
        m_optix.UpsampledSpectrum_maxBrightnesses.finalize();
#endif

        m_optix.instBuffer.finalize();
        m_optix.geomInstBuffer.finalize();
        m_optix.scene.destroy();

        m_optix.materialWithAlpha.destroy();
        m_optix.materialDefault.destroy();

        {
            releaseEDFProcedureSet(m_optix.nullEDFProcedureSetIndex);
            destroyDirectCallableProgram(m_optix.dcNullEDF_evaluateInternal);
            destroyDirectCallableProgram(m_optix.dcNullEDF_evaluateEmittanceInternal);
            destroyDirectCallableProgram(m_optix.dcNullEDF_setupEDF);

            releaseBSDFProcedureSet(m_optix.nullBSDFProcedureSetIndex);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_weightInternal);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_evaluatePDFInternal);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_evaluateInternal);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_sampleInternal);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_matches);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_getBaseColor);
            destroyDirectCallableProgram(m_optix.dcNullBSDF_setupBSDF);

            m_optix.nullDFModule.destroy();
        }

        {
            m_optix.debugRenderingHitGroupWithAlpha.destroy();
            m_optix.debugRenderingHitGroupDefault.destroy();
            m_optix.debugRenderingMiss.destroy();
            m_optix.debugRenderingRayGeneration.destroy();

            m_optix.debugRenderingModule.destroy();
        }

        {
            m_optix.pathTracingHitGroupShadowWithAlpha.destroy();
            m_optix.pathTracingHitGroupShadowDefault.destroy();
            m_optix.pathTracingHitGroupWithAlpha.destroy();
            m_optix.pathTracingHitGroupDefault.destroy();
            m_optix.pathTracingShadowMiss.destroy();
            m_optix.pathTracingMiss.destroy();
            m_optix.pathTracingRayGeneration.destroy();

            m_optix.pathTracingModule.destroy();
        }

        m_optix.pipeline.destroy();

        m_optix.context.destroy();

        finalizeColorSystem();
    }

    void Context::bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glTexID) {
        m_optix.outputBufferHolder.finalize();
        if (m_optix.outputBuffer.isInitialized())
            m_optix.outputBuffer.finalize();
        if (m_optix.accumBuffer.isInitialized())
            m_optix.accumBuffer.finalize();
        if (m_optix.rngBuffer.isInitialized())
            m_optix.rngBuffer.finalize();

        m_width = width;
        m_height = height;

        if (glTexID > 0) {
            // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
            int32_t gl3wRet = gl3wInit();
            VLRAssert(gl3wRet == 0, "gl3wInit() failed.");
            constexpr uint32_t OpenGLMajorVersion = 3;
            constexpr uint32_t OpenGLMinorVersion = 0;
            if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
                vlrprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
                return;
            }

            m_optix.outputBuffer.initializeFromGLTexture2D(
                m_cuContext, glTexID,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
            m_optix.outputBufferHolder.initialize(&m_optix.outputBuffer);
        }
        else {
            m_optix.outputBuffer.initialize2D(
                m_cuContext, cudau::ArrayElementType::Float32, 4,
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                m_width, m_height, 1);
        }

        m_optix.accumBuffer.initialize(m_cuContext, g_bufferType, m_width, m_height);
        m_optix.launchParams.accumBuffer = m_optix.accumBuffer.getBlockBuffer2D();

        m_optix.rngBuffer.initialize(m_cuContext, g_bufferType, m_width, m_height);
        {
            m_optix.rngBuffer.map();
            std::mt19937_64 rng(591842031321323413);
            for (int y = 0; y < m_height; ++y) {
                for (int x = 0; x < m_width; ++x) {
                    m_optix.rngBuffer(x, y) = rng();
                }
            }
            m_optix.rngBuffer.unmap();
        }
        m_optix.launchParams.rngBuffer = m_optix.rngBuffer.getBlockBuffer2D();



        if (m_optix.linearDenoisedColorBuffer.isInitialized())
            m_optix.linearDenoisedColorBuffer.finalize();
        if (m_optix.linearNormalBuffer.isInitialized())
            m_optix.linearNormalBuffer.finalize();
        if (m_optix.linearAlbedoBuffer.isInitialized())
            m_optix.linearAlbedoBuffer.finalize();
        if (m_optix.linearColorBuffer.isInitialized())
            m_optix.linearColorBuffer.finalize();
        if (m_optix.accumNormalBuffer.isInitialized())
            m_optix.accumNormalBuffer.finalize();
        if (m_optix.accumAlbedoBuffer.isInitialized())
            m_optix.accumAlbedoBuffer.finalize();
        if (m_optix.denoiserScratchBuffer.isInitialized())
            m_optix.denoiserScratchBuffer.finalize();
        if (m_optix.denoiserStateBuffer.isInitialized())
            m_optix.denoiserStateBuffer.finalize();
        
        size_t stateBufferSize;
        size_t scratchBufferSize;
        size_t scratchBufferSizeForComputeIntensity;
        uint32_t numTasks;
        m_optix.denoiser.prepare(m_width, m_height, 0, 0,
                                 &stateBufferSize, &scratchBufferSize, &scratchBufferSizeForComputeIntensity,
                                 &numTasks);
        m_optix.denoiserStateBuffer.initialize(m_cuContext, g_bufferType, stateBufferSize, 1);
        m_optix.denoiserScratchBuffer.initialize(m_cuContext, g_bufferType,
                                                 std::max(scratchBufferSize, scratchBufferSizeForComputeIntensity), 1);
        m_optix.denoiserTasks.resize(numTasks);
        m_optix.accumAlbedoBuffer.initialize(m_cuContext, g_bufferType, m_width * m_height);
        m_optix.accumNormalBuffer.initialize(m_cuContext, g_bufferType, m_width * m_height);
        m_optix.linearColorBuffer.initialize(m_cuContext, g_bufferType, m_width * m_height);
        m_optix.linearAlbedoBuffer.initialize(m_cuContext, g_bufferType, m_width * m_height);
        m_optix.linearNormalBuffer.initialize(m_cuContext, g_bufferType, m_width * m_height);
        m_optix.linearDenoisedColorBuffer.initialize(m_cuContext, g_bufferType, m_width * m_height);
        m_optix.denoiser.getTasks(m_optix.denoiserTasks.data());
        m_optix.denoiser.setLayers(m_optix.linearColorBuffer,
                                   m_optix.linearAlbedoBuffer,
                                   m_optix.linearNormalBuffer,
                                   m_optix.linearDenoisedColorBuffer,
                                   OPTIX_PIXEL_FORMAT_FLOAT4, OPTIX_PIXEL_FORMAT_FLOAT4, OPTIX_PIXEL_FORMAT_FLOAT4);

        m_optix.launchParams.accumAlbedoBuffer = m_optix.accumAlbedoBuffer.getDevicePointer();
        m_optix.launchParams.accumNormalBuffer = m_optix.accumNormalBuffer.getDevicePointer();



        m_optix.launchParams.imageStrideInPixels = m_width;
    }

    const cudau::Array &Context::getOutputBuffer() const {
        return m_optix.outputBuffer;
    }

    void Context::getOutputBufferSize(uint32_t* width, uint32_t* height) {
        *width = m_width;
        *height = m_height;
    }

    void Context::setScene(Scene &scene) {
        CUstream stream = 0;

        size_t sbtSize;
        m_optix.scene.generateShaderBindingTableLayout(&sbtSize);
        m_optix.hitGroupShaderBindingTable.initialize(m_cuContext, g_bufferType, sbtSize, 1);
        m_optix.hitGroupShaderBindingTable.setMappedMemoryPersistent(true);

        m_optix.pipeline.setScene(m_optix.scene);
        m_optix.pipeline.setHitGroupShaderBindingTable(m_optix.hitGroupShaderBindingTable,
                                                       m_optix.hitGroupShaderBindingTable.getMappedPointer());

        size_t asScratchSize;
        scene.prepareSetup(&asScratchSize);
        cudau::Buffer asScratchMem;
        asScratchMem.initialize(m_cuContext, g_bufferType, asScratchSize, 1);
        scene.setup(stream, asScratchMem, &m_optix.launchParams);
        asScratchMem.finalize();

        CUDADRV_CHECK(cuStreamSynchronize(stream));
    }

    void Context::render(const Camera* camera, bool denoise,
                         bool debugRender, VLRDebugRenderingMode renderMode,
                         uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        CUstream stream = 0;

        uint2 imageSize = make_uint2(m_width / shrinkCoeff, m_height / shrinkCoeff);
        uint32_t imageStrideInPixels = m_optix.launchParams.imageStrideInPixels;
        if (firstFrame) {
            camera->setup(&m_optix.launchParams);

            m_optix.launchParams.imageSize = imageSize;

            if (debugRender)
                m_optix.pipeline.setRayGenerationProgram(m_optix.debugRenderingRayGeneration);
            else
                m_optix.pipeline.setRayGenerationProgram(m_optix.pathTracingRayGeneration);

            m_optix.denoiser.setupState(stream, m_optix.denoiserStateBuffer, m_optix.denoiserScratchBuffer);

            m_numAccumFrames = 0;
        }

        ++m_numAccumFrames;
        *numAccumFrames = m_numAccumFrames;
        m_optix.launchParams.numAccumFrames = m_numAccumFrames;
        m_optix.launchParams.debugRenderingAttribute = static_cast<shared::DebugRenderingAttribute>(renderMode);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(m_optix.launchParamsOnDevice, &m_optix.launchParams,
                                        sizeof(m_optix.launchParams), stream));
        m_optix.pipeline.launch(stream, m_optix.launchParamsOnDevice,
                                imageSize.x, imageSize.y, 1);

        CUDADRV_CHECK(cuMemcpyHtoDAsync(m_cudaPostProcessModuleLaunchParamsPtr, &m_optix.launchParams,
                                        sizeof(m_optix.launchParams), stream));

        if (denoise) {
            m_copyBuffers(stream, m_copyBuffers.calcGridDim(imageSize.x, imageSize.y),
                          m_optix.accumBuffer.getBlockBuffer2D(),
                          m_optix.accumAlbedoBuffer.getDevicePointer(),
                          m_optix.accumNormalBuffer.getDevicePointer(),
                          imageSize, imageStrideInPixels, m_numAccumFrames,
                          m_optix.linearColorBuffer.getDevicePointer(),
                          m_optix.linearAlbedoBuffer.getDevicePointer(),
                          m_optix.linearNormalBuffer.getDevicePointer());

            m_optix.denoiser.computeIntensity(stream, m_optix.denoiserScratchBuffer, m_optix.hdrIntensity);

            for (int i = 0; i < m_optix.denoiserTasks.size(); ++i)
                m_optix.denoiser.invoke(stream, false, m_optix.hdrIntensity, 0.0f, m_optix.denoiserTasks[i]);
        }

        m_optix.outputBufferHolder.beginCUDAAccess(stream);
        m_convertToRGB(stream, m_convertToRGB.calcGridDim(imageSize.x, imageSize.y),
                       m_optix.accumBuffer.getBlockBuffer2D(),
                       m_optix.linearDenoisedColorBuffer.getDevicePointer(),
                       denoise, imageSize, imageStrideInPixels, m_numAccumFrames,
                       m_optix.outputBufferHolder.getNext());
        m_optix.outputBufferHolder.endCUDAAccess(stream);

        CUDADRV_CHECK(cuStreamSynchronize(stream));
    }

    void Context::render(const Camera* camera, bool denoise,
                         uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        render(camera, denoise, false, VLRDebugRenderingMode_BaseColor, shrinkCoeff, firstFrame, numAccumFrames);
    }

    void Context::debugRender(const Camera* camera, VLRDebugRenderingMode renderMode,
                              uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        render(camera, false, true, renderMode, shrinkCoeff, firstFrame, numAccumFrames);
    }



    uint32_t Context::createDirectCallableProgram(optixu::Module dcModule, const char* dcName) {
        optixu::ProgramGroup program = m_optix.pipeline.createCallableProgramGroup(
            dcModule, dcName,
            optixu::Module(), nullptr);
        uint32_t index = m_optix.callablePrograms.size();
        m_optix.callablePrograms.push_back(program);
        return index;
    }
    void Context::destroyDirectCallableProgram(uint32_t index) {
        m_optix.callablePrograms[index].destroy();
    }



    uint32_t Context::allocateNodeProcedureSet() {
        return m_optix.nodeProcedureSetBuffer.allocate();
    }
    void Context::releaseNodeProcedureSet(uint32_t index) {
        m_optix.nodeProcedureSetBuffer.release(index);
    }
    void Context::updateNodeProcedureSet(uint32_t index, const shared::NodeProcedureSet &procSet) {
        m_optix.nodeProcedureSetBuffer.update(index, procSet);
    }



    uint32_t Context::allocateSmallNodeDescriptor() {
        return m_optix.smallNodeDescriptorBuffer.allocate();
    }
    void Context::releaseSmallNodeDescriptor(uint32_t index) {
        m_optix.smallNodeDescriptorBuffer.release(index);
    }
    void Context::updateSmallNodeDescriptor(uint32_t index, const shared::SmallNodeDescriptor &nodeDesc) {
        m_optix.smallNodeDescriptorBuffer.update(index, nodeDesc);
    }



    uint32_t Context::allocateMediumNodeDescriptor() {
        return m_optix.mediumNodeDescriptorBuffer.allocate();
    }
    void Context::releaseMediumNodeDescriptor(uint32_t index) {
        m_optix.mediumNodeDescriptorBuffer.release(index);
    }
    void Context::updateMediumNodeDescriptor(uint32_t index, const shared::MediumNodeDescriptor &nodeDesc) {
        m_optix.mediumNodeDescriptorBuffer.update(index, nodeDesc);
    }



    uint32_t Context::allocateLargeNodeDescriptor() {
        return m_optix.largeNodeDescriptorBuffer.allocate();
    }
    void Context::releaseLargeNodeDescriptor(uint32_t index) {
        m_optix.largeNodeDescriptorBuffer.release(index);
    }
    void Context::updateLargeNodeDescriptor(uint32_t index, const shared::LargeNodeDescriptor &nodeDesc) {
        m_optix.largeNodeDescriptorBuffer.update(index, nodeDesc);
    }



    uint32_t Context::allocateBSDFProcedureSet() {
        return m_optix.bsdfProcedureSetBuffer.allocate();
    }
    void Context::releaseBSDFProcedureSet(uint32_t index) {
        m_optix.bsdfProcedureSetBuffer.release(index);
    }
    void Context::updateBSDFProcedureSet(uint32_t index, const shared::BSDFProcedureSet &procSet) {
        m_optix.bsdfProcedureSetBuffer.update(index, procSet);
    }



    uint32_t Context::allocateEDFProcedureSet() {
        return m_optix.edfProcedureSetBuffer.allocate();
    }
    void Context::releaseEDFProcedureSet(uint32_t index) {
        m_optix.edfProcedureSetBuffer.release(index);
    }
    void Context::updateEDFProcedureSet(uint32_t index, const shared::EDFProcedureSet &procSet) {
        m_optix.edfProcedureSetBuffer.update(index, procSet);
    }



    uint32_t Context::allocateSurfaceMaterialDescriptor() {
        return m_optix.surfaceMaterialDescriptorBuffer.allocate();
    }
    void Context::releaseSurfaceMaterialDescriptor(uint32_t index) {
        m_optix.surfaceMaterialDescriptorBuffer.release(index);
    }
    void Context::updateSurfaceMaterialDescriptor(uint32_t index, const shared::SurfaceMaterialDescriptor &matDesc) {
        m_optix.surfaceMaterialDescriptorBuffer.update(index, matDesc);
    }



    uint32_t Context::allocateGeometryInstance() {
        return m_optix.geomInstBuffer.allocate();
    }
    void Context::releaseGeometryInstance(uint32_t index) {
        m_optix.geomInstBuffer.release(index);
    }
    void Context::updateGeometryInstance(uint32_t index, const shared::GeometryInstance &geomInst) {
        m_optix.geomInstBuffer.update(index, geomInst);
    }



    uint32_t Context::allocateInstance() {
        return m_optix.instBuffer.allocate();
    }
    void Context::releaseInstance(uint32_t index) {
        m_optix.instBuffer.release(index);
    }
    void Context::updateInstance(uint32_t index, const shared::Instance &inst) {
        m_optix.instBuffer.update(index, inst);
    }

    // ----------------------------------------------------------------
    // Miscellaneous

    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numValues) {
        CUcontext cuContext = context.getCUcontext();

        m_numValues = static_cast<uint32_t>(numValues);
        m_PMF.initialize(cuContext, g_bufferType, m_numValues);
        m_CDF.initialize(cuContext, g_bufferType, m_numValues + 1);

        RealType* PMF = m_PMF.map();
        RealType* CDF = m_CDF.map();
        std::memcpy(PMF, values, sizeof(RealType) * m_numValues);

        CompensatedSum<RealType> sum(0);
        for (int i = 0; i < m_numValues; ++i) {
            CDF[i] = sum;
            sum += PMF[i];
        }
        m_integral = sum;
        for (int i = 0; i < m_numValues; ++i) {
            PMF[i] /= m_integral;
            CDF[i] /= m_integral;
        }
        CDF[m_numValues] = 1.0f;

        m_CDF.unmap();
        m_PMF.unmap();
    }

    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::finalize(Context &context) {
        if (m_CDF.isInitialized() && m_PMF.isInitialized()) {
            m_CDF.finalize();
            m_PMF.finalize();
        }
    }

    template <typename RealType>
    void DiscreteDistribution1DTemplate<RealType>::getInternalType(shared::DiscreteDistribution1DTemplate<RealType>* instance) const {
        if (m_PMF.isInitialized() && m_CDF.isInitialized())
            new (instance) shared::DiscreteDistribution1DTemplate<RealType>(
                m_PMF.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
    }

    template class DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numValues) {
        CUcontext cuContext = context.getCUcontext();

        m_numValues = static_cast<uint32_t>(numValues);
        m_PDF.initialize(cuContext, g_bufferType, m_numValues);
        m_CDF.initialize(cuContext, g_bufferType, m_numValues + 1);

        RealType* PDF = m_PDF.map();
        RealType* CDF = m_CDF.map();
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

        m_CDF.unmap();
        m_PDF.unmap();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::finalize(Context &context) {
        if (m_CDF.isInitialized() && m_PDF.isInitialized()) {
            m_CDF.finalize();
            m_PDF.finalize();
        }
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::getInternalType(shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const {
        new (instance) shared::RegularConstantContinuousDistribution1DTemplate<RealType>(
            m_PDF.getDevicePointer(), m_CDF.getDevicePointer(), m_integral, m_numValues);
    }

    template class RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numD1, size_t numD2) {
        CUcontext cuContext = context.getCUcontext();

        m_1DDists = new RegularConstantContinuousDistribution1DTemplate<RealType>[numD2];
        m_raw1DDists.initialize(cuContext, g_bufferType, numD2);

        shared::RegularConstantContinuousDistribution1DTemplate<RealType>* rawDists = m_raw1DDists.map();

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

        m_raw1DDists.unmap();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::finalize(Context &context) {
        m_top1DDist.finalize(context);

        for (int i = m_top1DDist.getNumValues() - 1; i >= 0; --i) {
            m_1DDists[i].finalize(context);
        }

        m_raw1DDists.finalize();
        delete[] m_1DDists;
        m_1DDists = nullptr;
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::getInternalType(shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const {
        shared::RegularConstantContinuousDistribution1DTemplate<RealType> top1DDist;
        m_top1DDist.getInternalType(&top1DDist);
        new (instance) shared::RegularConstantContinuousDistribution2DTemplate<RealType>(
            m_raw1DDists.getDevicePointer(), top1DDist);
    }

    template class RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
}
