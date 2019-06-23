#include "context.h"

#include <random>

#include "scene.h"

namespace VLR {
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

#define VLR_DEFINE_CLASS_ID(BaseType, Type) const ClassIdentifier Type::ClassID = ClassIdentifier(&BaseType::ClassID)

    const ClassIdentifier TypeAwareClass::ClassID = ClassIdentifier((ClassIdentifier*)nullptr);

    VLR_DEFINE_CLASS_ID(TypeAwareClass, Object);

    VLR_DEFINE_CLASS_ID(Object, Connectable);

    VLR_DEFINE_CLASS_ID(Connectable, Image2D);
    VLR_DEFINE_CLASS_ID(Image2D, LinearImage2D);
    VLR_DEFINE_CLASS_ID(Image2D, BlockCompressedImage2D);

    VLR_DEFINE_CLASS_ID(Connectable, ShaderNode);
    VLR_DEFINE_CLASS_ID(ShaderNode, GeometryShaderNode);
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

    VLR_DEFINE_CLASS_ID(Connectable, SurfaceMaterial);
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

    VLR_DEFINE_CLASS_ID(Object, Camera);
    VLR_DEFINE_CLASS_ID(Camera, PerspectiveCamera);
    VLR_DEFINE_CLASS_ID(Camera, EquirectangularCamera);

#undef VLR_DEFINE_CLASS_ID



    struct EntryPoint {
        enum Value {
            PathTracing = 0,
            DebugRendering,
            ConvertToRGB,
            NumEntryPoints
        } value;

        constexpr EntryPoint(Value v) : value(v) {}
    };



    uint32_t Context::NextID = 0;

    static void checkError(RTresult code) {
        if (code != RT_SUCCESS && code != RT_TIMEOUT_CALLBACK)
            throw optix::Exception::makeException(code, 0);
    }

    Context::Context(bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices) {
        // JP: 使用するすべてのGPUがRTXをサポートしている(= Maxwell世代以降のGPU)か調べる。
        // EN: check if all the GPUs to use support RTX (i.e. Maxwell or later generation GPU).
        bool satisfyRequirements = true;
        if (devices == nullptr || numDevices == 0) {
            rtDeviceGetDeviceCount(&m_numDevices);
            m_devices = new int32_t[m_numDevices];
            for (int i = 0; i < m_numDevices; ++i)
                m_devices[i] = i;
        }
        else {
            m_numDevices = numDevices;
            m_devices = new int32_t[m_numDevices];
            std::copy_n(devices, m_numDevices, m_devices);
        }
        for (int i = 0; i < m_numDevices; ++i) {
            int32_t computeCapability[2];
            checkError(rtDeviceGetAttribute(m_devices[i], RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCapability), computeCapability));
            satisfyRequirements &= computeCapability[0] >= 5;
            if (!satisfyRequirements)
                break;
        }
        if (!satisfyRequirements) {
            delete[] m_devices;
            vlrprintf("Selected devices don't satisfy compute capability 5.0.\n");
            checkError(RT_ERROR_INVALID_CONTEXT);
            return;
        }

        m_RTXEnabled = enableRTX;
        int32_t RTXEnabled = m_RTXEnabled;
        if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTXEnabled), &RTXEnabled) == RT_SUCCESS)
            vlrprintf("RTX %s\n", RTXEnabled ? "ON" : "OFF");
        else
            vlrprintf("failed to set the global attribute RT_GLOBAL_ATTRIBUTE_ENABLE_RTX.\n");

        vlrprintf("Start initializing VLR ...");

        initializeColorSystem();

        m_ID = getInstanceID();

        m_optixContext = optix::Context::create();
        m_optixContext->setDevices(m_devices, m_devices + m_numDevices);

        m_optixContext->setEntryPointCount(EntryPoint::NumEntryPoints);
        m_optixContext->setRayTypeCount(Shared::RayType::NumTypes);

        const filesystem::path exeDir = getExecutableDirectory();

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/path_tracing.ptx");

            m_optixProgramShadowAnyHitDefault = m_optixContext->createProgramFromPTXString(ptx, "VLR::shadowAnyHitDefault");
            m_optixProgramAnyHitWithAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::anyHitWithAlpha");
            m_optixProgramShadowAnyHitWithAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::shadowAnyHitWithAlpha");
            m_optixProgramPathTracingIteration = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingIteration");

            m_optixProgramPathTracing = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracing");
            m_optixProgramPathTracingMiss = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingMiss");
            m_optixProgramException = m_optixContext->createProgramFromPTXString(ptx, "VLR::exception");
        }
        m_optixContext->setRayGenerationProgram(EntryPoint::PathTracing, m_optixProgramPathTracing);
        m_optixContext->setExceptionProgram(EntryPoint::PathTracing, m_optixProgramException);

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/debug_rendering.ptx");

            m_optixProgramDebugRenderingClosestHit = m_optixContext->createProgramFromPTXString(ptx, "VLR::debugRenderingClosestHit");
            m_optixProgramDebugRenderingAnyHitWithAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::debugRenderingAnyHitWithAlpha");
            m_optixProgramDebugRenderingMiss = m_optixContext->createProgramFromPTXString(ptx, "VLR::debugRenderingMiss");
            m_optixProgramDebugRenderingRayGeneration = m_optixContext->createProgramFromPTXString(ptx, "VLR::debugRenderingRayGeneration");
            m_optixProgramDebugRenderingException = m_optixContext->createProgramFromPTXString(ptx, "VLR::debugRenderingException");
        }
        m_optixContext->setRayGenerationProgram(EntryPoint::DebugRendering, m_optixProgramDebugRenderingRayGeneration);
        m_optixContext->setExceptionProgram(EntryPoint::DebugRendering, m_optixProgramDebugRenderingException);

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/convert_to_rgb.ptx");
            m_optixProgramConvertToRGB = m_optixContext->createProgramFromPTXString(ptx, "VLR::convertToRGB");
        }
        m_optixContext->setRayGenerationProgram(EntryPoint::ConvertToRGB, m_optixProgramConvertToRGB);

        m_optixContext->setMissProgram(Shared::RayType::Primary, m_optixProgramPathTracingMiss);
        m_optixContext->setMissProgram(Shared::RayType::Scattered, m_optixProgramPathTracingMiss);
        m_optixContext->setMissProgram(Shared::RayType::DebugPrimary, m_optixProgramDebugRenderingMiss);



        const auto setInt32 = [](optix::Context &optixContext, const char* pvname, int32_t data) {
            optixContext[pvname]->setUserData(sizeof(data), &data);
        };
        
        m_optixContext["VLR::DiscretizedSpectrum_xbar"]->setUserData(sizeof(DiscretizedSpectrumAlwaysSpectral::CMF), &DiscretizedSpectrumAlwaysSpectral::xbar);
        m_optixContext["VLR::DiscretizedSpectrum_ybar"]->setUserData(sizeof(DiscretizedSpectrumAlwaysSpectral::CMF), &DiscretizedSpectrumAlwaysSpectral::ybar);
        m_optixContext["VLR::DiscretizedSpectrum_zbar"]->setUserData(sizeof(DiscretizedSpectrumAlwaysSpectral::CMF), &DiscretizedSpectrumAlwaysSpectral::zbar);
        m_optixContext["VLR::DiscretizedSpectrum_integralCMF"]->setFloat(DiscretizedSpectrumAlwaysSpectral::integralCMF);

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        const uint32_t NumSpectrumGridCells = 168;
        const uint32_t NumSpectrumDataPoints = 186;
        m_optixBufferUpsampledSpectrum_spectrum_grid = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, NumSpectrumGridCells);
        m_optixBufferUpsampledSpectrum_spectrum_grid->setElementSize(sizeof(UpsampledSpectrum::spectrum_grid_cell_t));
        {
            auto values = (UpsampledSpectrum::spectrum_grid_cell_t*)m_optixBufferUpsampledSpectrum_spectrum_grid->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy_n(UpsampledSpectrum::spectrum_grid, NumSpectrumGridCells, values);
            m_optixBufferUpsampledSpectrum_spectrum_grid->unmap();
        }
        m_optixBufferUpsampledSpectrum_spectrum_data_points = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, NumSpectrumDataPoints);
        m_optixBufferUpsampledSpectrum_spectrum_data_points->setElementSize(sizeof(UpsampledSpectrum::spectrum_data_point_t));
        {
            auto values = (UpsampledSpectrum::spectrum_data_point_t*)m_optixBufferUpsampledSpectrum_spectrum_data_points->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
            std::copy_n(UpsampledSpectrum::spectrum_data_points, NumSpectrumDataPoints, values);
            m_optixBufferUpsampledSpectrum_spectrum_data_points->unmap();
        }
        setInt32(m_optixContext, "VLR::UpsampledSpectrum_spectrum_grid", m_optixBufferUpsampledSpectrum_spectrum_grid->getId());
        setInt32(m_optixContext, "VLR::UpsampledSpectrum_spectrum_data_points", m_optixBufferUpsampledSpectrum_spectrum_data_points->getId());
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



        m_optixMaterialDefault = m_optixContext->createMaterial();
        m_optixMaterialDefault->setClosestHitProgram(Shared::RayType::Primary, m_optixProgramPathTracingIteration);
        m_optixMaterialDefault->setClosestHitProgram(Shared::RayType::Scattered, m_optixProgramPathTracingIteration);
        m_optixMaterialDefault->setClosestHitProgram(Shared::RayType::DebugPrimary, m_optixProgramDebugRenderingClosestHit);
        //m_optixMaterialDefault->setAnyHitProgram(Shared::RayType::Primary, );
        //m_optixMaterialDefault->setAnyHitProgram(Shared::RayType::Scattered, );
        m_optixMaterialDefault->setAnyHitProgram(Shared::RayType::Shadow, m_optixProgramShadowAnyHitDefault);

        m_optixMaterialWithAlpha = m_optixContext->createMaterial();
        m_optixMaterialWithAlpha->setClosestHitProgram(Shared::RayType::Primary, m_optixProgramPathTracingIteration);
        m_optixMaterialWithAlpha->setClosestHitProgram(Shared::RayType::Scattered, m_optixProgramPathTracingIteration);
        m_optixMaterialWithAlpha->setClosestHitProgram(Shared::RayType::DebugPrimary, m_optixProgramDebugRenderingClosestHit);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::Primary, m_optixProgramAnyHitWithAlpha);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::Scattered, m_optixProgramAnyHitWithAlpha);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::Shadow, m_optixProgramShadowAnyHitWithAlpha);
        m_optixMaterialWithAlpha->setAnyHitProgram(Shared::RayType::DebugPrimary, m_optixProgramDebugRenderingAnyHitWithAlpha);



        m_maxNumNodeProcSet = 256;
        m_optixNodeProcedureSetBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumNodeProcSet);
        m_optixNodeProcedureSetBuffer->setElementSize(sizeof(Shared::NodeProcedureSet));
        m_nodeProcSetSlotFinder.initialize(m_maxNumNodeProcSet);
        m_optixContext["VLR::pv_nodeProcedureSetBuffer"]->set(m_optixNodeProcedureSetBuffer);



        m_maxNumSmallNodeDescriptors = 8192;
        m_optixSmallNodeDescriptorBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumSmallNodeDescriptors);
        m_optixSmallNodeDescriptorBuffer->setElementSize(sizeof(Shared::SmallNodeDescriptor));
        m_smallNodeDescSlotFinder.initialize(m_maxNumSmallNodeDescriptors);

        m_optixContext["VLR::pv_smallNodeDescriptorBuffer"]->set(m_optixSmallNodeDescriptorBuffer);



        m_maxNumMediumNodeDescriptors = 8192;
        m_optixMediumNodeDescriptorBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumMediumNodeDescriptors);
        m_optixMediumNodeDescriptorBuffer->setElementSize(sizeof(Shared::MediumNodeDescriptor));
        m_mediumNodeDescSlotFinder.initialize(m_maxNumMediumNodeDescriptors);

        m_optixContext["VLR::pv_mediumNodeDescriptorBuffer"]->set(m_optixMediumNodeDescriptorBuffer);



        m_maxNumLargeNodeDescriptors = 1024;
        m_optixLargeNodeDescriptorBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumLargeNodeDescriptors);
        m_optixLargeNodeDescriptorBuffer->setElementSize(sizeof(Shared::LargeNodeDescriptor));
        m_largeNodeDescSlotFinder.initialize(m_maxNumLargeNodeDescriptors);

        m_optixContext["VLR::pv_largeNodeDescriptorBuffer"]->set(m_optixLargeNodeDescriptorBuffer);



        m_maxNumBSDFProcSet = 64;
        m_optixBSDFProcedureSetBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumBSDFProcSet);
        m_optixBSDFProcedureSetBuffer->setElementSize(sizeof(Shared::BSDFProcedureSet));
        m_bsdfProcSetSlotFinder.initialize(m_maxNumBSDFProcSet);
        m_optixContext["VLR::pv_bsdfProcedureSetBuffer"]->set(m_optixBSDFProcedureSetBuffer);

        m_maxNumEDFProcSet = 64;
        m_optixEDFProcedureSetBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumEDFProcSet);
        m_optixEDFProcedureSetBuffer->setElementSize(sizeof(Shared::EDFProcedureSet));
        m_edfProcSetSlotFinder.initialize(m_maxNumEDFProcSet);
        m_optixContext["VLR::pv_edfProcedureSetBuffer"]->set(m_optixEDFProcedureSetBuffer);

        {
            std::string ptx = readTxtFile(exeDir / "ptxes/materials.ptx");

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
        m_surfMatDescSlotFinder.initialize(m_maxNumSurfaceMaterialDescriptors);

        m_optixContext["VLR::pv_materialDescriptorBuffer"]->set(m_optixSurfaceMaterialDescriptorBuffer);

        SurfaceNode::initialize(*this);
        ShaderNode::initialize(*this);
        SurfaceMaterial::initialize(*this);
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
        SurfaceMaterial::finalize(*this);
        ShaderNode::finalize(*this);
        SurfaceNode::finalize(*this);

        m_surfMatDescSlotFinder.finalize();
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

        m_edfProcSetSlotFinder.finalize();
        m_optixEDFProcedureSetBuffer->destroy();

        m_bsdfProcSetSlotFinder.finalize();
        m_optixBSDFProcedureSetBuffer->destroy();

        m_largeNodeDescSlotFinder.finalize();
        m_optixLargeNodeDescriptorBuffer->destroy();

        m_smallNodeDescSlotFinder.finalize();
        m_optixSmallNodeDescriptorBuffer->destroy();

        m_nodeProcSetSlotFinder.finalize();
        m_optixNodeProcedureSetBuffer->destroy();

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

        m_optixContext->destroy();

        finalizeColorSystem();

        delete[] m_devices;
    }

    void Context::bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID) {
        if (m_outputBuffer)
            m_outputBuffer->destroy();
        if (m_rawOutputBuffer)
            m_rawOutputBuffer->destroy();
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
        m_optixContext["VLR::pv_RGBBuffer"]->set(m_outputBuffer);

        m_rawOutputBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_width, m_height);
        m_rawOutputBuffer->setElementSize(sizeof(SpectrumStorage));
        m_optixContext["VLR::pv_spectrumBuffer"]->set(m_rawOutputBuffer);
        m_optixContext["VLR::pv_outputBuffer"]->set(m_rawOutputBuffer);

        m_rngBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_width, m_height);
        m_rngBuffer->setElementSize(sizeof(uint64_t));
        {
            std::mt19937_64 rng(591842031321323413);

            auto dstData = (uint64_t*)m_rngBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
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

    const void* Context::mapOutputBuffer() {
        if (!m_outputBuffer)
            return nullptr;

        return m_outputBuffer->map(0, RT_BUFFER_MAP_READ);
    }

    void Context::unmapOutputBuffer() {
        m_outputBuffer->unmap();
    }

    void Context::getOutputBufferSize(uint32_t* width, uint32_t* height) {
        *width = m_width;
        *height = m_height;
    }

    void Context::render(Scene &scene, const Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
        optix::Context optixContext = getOptiXContext();

        optix::uint2 imageSize = optix::make_uint2(m_width / shrinkCoeff, m_height / shrinkCoeff);
        if (firstFrame) {
            scene.set();
            camera->set();

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
            scene.set();
            camera->set();

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
        uint32_t index = m_nodeProcSetSlotFinder.getFirstAvailableSlot();
        m_nodeProcSetSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseNodeProcedureSet(uint32_t index) {
        VLRAssert(m_nodeProcSetSlotFinder.getUsage(index), "Invalid index.");
        m_nodeProcSetSlotFinder.setNotInUse(index);
    }

    void Context::updateNodeProcedureSet(uint32_t index, const Shared::NodeProcedureSet &procSet) {
        VLRAssert(m_nodeProcSetSlotFinder.getUsage(index), "Invalid index.");
        auto procSets = (Shared::NodeProcedureSet*)m_optixNodeProcedureSetBuffer->map(0, RT_BUFFER_MAP_WRITE);
        procSets[index] = procSet;
        m_optixNodeProcedureSetBuffer->unmap();
    }



    uint32_t Context::allocateSmallNodeDescriptor() {
        uint32_t index = m_smallNodeDescSlotFinder.getFirstAvailableSlot();
        m_smallNodeDescSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseSmallNodeDescriptor(uint32_t index) {
        VLRAssert(m_smallNodeDescSlotFinder.getUsage(index), "Invalid index.");
        m_smallNodeDescSlotFinder.setNotInUse(index);
    }

    void Context::updateSmallNodeDescriptor(uint32_t index, const Shared::SmallNodeDescriptor &nodeDesc) {
        VLRAssert(m_smallNodeDescSlotFinder.getUsage(index), "Invalid index.");
        auto nodeDescs = (Shared::SmallNodeDescriptor*)m_optixSmallNodeDescriptorBuffer->map(0, RT_BUFFER_MAP_WRITE);
        nodeDescs[index] = nodeDesc;
        m_optixSmallNodeDescriptorBuffer->unmap();
    }



    uint32_t Context::allocateMediumNodeDescriptor() {
        uint32_t index = m_mediumNodeDescSlotFinder.getFirstAvailableSlot();
        m_mediumNodeDescSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseMediumNodeDescriptor(uint32_t index) {
        VLRAssert(m_mediumNodeDescSlotFinder.getUsage(index), "Invalid index.");
        m_mediumNodeDescSlotFinder.setNotInUse(index);
    }

    void Context::updateMediumNodeDescriptor(uint32_t index, const Shared::MediumNodeDescriptor &nodeDesc) {
        VLRAssert(m_mediumNodeDescSlotFinder.getUsage(index), "Invalid index.");
        auto nodeDescs = (Shared::MediumNodeDescriptor*)m_optixMediumNodeDescriptorBuffer->map(0, RT_BUFFER_MAP_WRITE);
        nodeDescs[index] = nodeDesc;
        m_optixMediumNodeDescriptorBuffer->unmap();
    }



    uint32_t Context::allocateLargeNodeDescriptor() {
        uint32_t index = m_largeNodeDescSlotFinder.getFirstAvailableSlot();
        m_largeNodeDescSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseLargeNodeDescriptor(uint32_t index) {
        VLRAssert(m_largeNodeDescSlotFinder.getUsage(index), "Invalid index.");
        m_largeNodeDescSlotFinder.setNotInUse(index);
    }

    void Context::updateLargeNodeDescriptor(uint32_t index, const Shared::LargeNodeDescriptor &nodeDesc) {
        VLRAssert(m_largeNodeDescSlotFinder.getUsage(index), "Invalid index.");
        auto nodeDescs = (Shared::LargeNodeDescriptor*)m_optixLargeNodeDescriptorBuffer->map(0, RT_BUFFER_MAP_WRITE);
        nodeDescs[index] = nodeDesc;
        m_optixLargeNodeDescriptorBuffer->unmap();
    }



    uint32_t Context::allocateBSDFProcedureSet() {
        uint32_t index = m_bsdfProcSetSlotFinder.getFirstAvailableSlot();
        m_bsdfProcSetSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseBSDFProcedureSet(uint32_t index) {
        VLRAssert(m_bsdfProcSetSlotFinder.getUsage(index), "Invalid index.");
        m_bsdfProcSetSlotFinder.setNotInUse(index);
    }

    void Context::updateBSDFProcedureSet(uint32_t index, const Shared::BSDFProcedureSet &procSet) {
        VLRAssert(m_bsdfProcSetSlotFinder.getUsage(index), "Invalid index.");
        auto procSets = (Shared::BSDFProcedureSet*)m_optixBSDFProcedureSetBuffer->map(0, RT_BUFFER_MAP_WRITE);
        procSets[index] = procSet;
        m_optixBSDFProcedureSetBuffer->unmap();
    }



    uint32_t Context::allocateEDFProcedureSet() {
        uint32_t index = m_edfProcSetSlotFinder.getFirstAvailableSlot();
        m_edfProcSetSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseEDFProcedureSet(uint32_t index) {
        VLRAssert(m_edfProcSetSlotFinder.getUsage(index), "Invalid index.");
        m_edfProcSetSlotFinder.setNotInUse(index);
    }

    void Context::updateEDFProcedureSet(uint32_t index, const Shared::EDFProcedureSet &procSet) {
        VLRAssert(m_edfProcSetSlotFinder.getUsage(index), "Invalid index.");
        auto procSets = (Shared::EDFProcedureSet*)m_optixEDFProcedureSetBuffer->map(0, RT_BUFFER_MAP_WRITE);
        procSets[index] = procSet;
        m_optixEDFProcedureSetBuffer->unmap();
    }



    uint32_t Context::allocateSurfaceMaterialDescriptor() {
        uint32_t index = m_surfMatDescSlotFinder.getFirstAvailableSlot();
        m_surfMatDescSlotFinder.setInUse(index);
        return index;
    }

    void Context::releaseSurfaceMaterialDescriptor(uint32_t index) {
        VLRAssert(m_surfMatDescSlotFinder.getUsage(index), "Invalid index.");
        m_surfMatDescSlotFinder.setNotInUse(index);
    }

    void Context::updateSurfaceMaterialDescriptor(uint32_t index, const Shared::SurfaceMaterialDescriptor &matDesc) {
        VLRAssert(m_surfMatDescSlotFinder.getUsage(index), "Invalid index.");
        auto matDescs = (Shared::SurfaceMaterialDescriptor*)m_optixSurfaceMaterialDescriptorBuffer->map(0, RT_BUFFER_MAP_WRITE);
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
