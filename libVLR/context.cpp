#include "context.h"

#include <random>

#include "materials.h"
#include "scene.h"

namespace VLR {
    static std::string readTxtFile(const std::string& filepath) {
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

    const ClassIdentifier Object::ClassID = ClassIdentifier((ClassIdentifier*)nullptr);

    defineClassID(Object, Image2D);
    defineClassID(Image2D, LinearImage2D);

    defineClassID(Object, FloatTexture);
    defineClassID(Object, Float2Texture);
    defineClassID(Object, Float3Texture);
    defineClassID(Float3Texture, ConstantFloat3Texture);
    defineClassID(Float3Texture, ImageFloat3Texture);
    defineClassID(Object, Float4Texture);
    defineClassID(Float4Texture, ConstantFloat4Texture);
    defineClassID(Float4Texture, ImageFloat4Texture);

    defineClassID(Object, SurfaceMaterial);
    defineClassID(SurfaceMaterial, MatteSurfaceMaterial);
    defineClassID(SurfaceMaterial, SpecularReflectionSurfaceMaterial);
    defineClassID(SurfaceMaterial, SpecularScatteringSurfaceMaterial);
    defineClassID(SurfaceMaterial, UE4SurfaceMaterial);
    defineClassID(SurfaceMaterial, DiffuseEmitterSurfaceMaterial);
    defineClassID(SurfaceMaterial, MultiSurfaceMaterial);
    defineClassID(SurfaceMaterial, EnvironmentEmitterSurfaceMaterial);

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

            m_optixCallableProgramNullFetchAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::Null_NormalAlphaModifier_fetchAlpha");
            m_optixCallableProgramNullFetchNormal = m_optixContext->createProgramFromPTXString(ptx, "VLR::Null_NormalAlphaModifier_fetchNormal");
            m_optixCallableProgramFetchAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::NormalAlphaModifier_fetchAlpha");
            m_optixCallableProgramFetchNormal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NormalAlphaModifier_fetchNormal");

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
            m_optixCallableProgramNullBSDF_sampleBSDFInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_sampleBSDFInternal");
            m_optixCallableProgramNullBSDF_evaluateBSDFInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_evaluateBSDFInternal");
            m_optixCallableProgramNullBSDF_evaluateBSDF_PDFInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_evaluateBSDF_PDFInternal");
            m_optixCallableProgramNullBSDF_weightInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullBSDF_weightInternal");

            Shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = m_optixCallableProgramNullBSDF_getBaseColor->getId();
                bsdfProcSet.progBSDFmatches = m_optixCallableProgramNullBSDF_matches->getId();
                bsdfProcSet.progSampleBSDFInternal = m_optixCallableProgramNullBSDF_sampleBSDFInternal->getId();
                bsdfProcSet.progEvaluateBSDFInternal = m_optixCallableProgramNullBSDF_evaluateBSDFInternal->getId();
                bsdfProcSet.progEvaluateBSDF_PDFInternal = m_optixCallableProgramNullBSDF_evaluateBSDF_PDFInternal->getId();
                bsdfProcSet.progWeightInternal = m_optixCallableProgramNullBSDF_weightInternal->getId();
            }
            m_nullBSDFProcedureSetIndex = setBSDFProcedureSet(bsdfProcSet);
            VLRAssert(m_nullBSDFProcedureSetIndex == 0, "Index of the null BSDF procedure set is expected to be 0.");



            m_optixCallableProgramNullEDF_setupEDF = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_setupEDF");
            m_optixCallableProgramNullEDF_evaluateEmittanceInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEmittanceInternal");
            m_optixCallableProgramNullEDF_evaluateEDFInternal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEDFInternal");

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = m_optixCallableProgramNullEDF_evaluateEmittanceInternal->getId();
                edfProcSet.progEvaluateEDFInternal = m_optixCallableProgramNullEDF_evaluateEDFInternal->getId();
            }
            m_nullEDFProcedureSetIndex = setEDFProcedureSet(edfProcSet);
            VLRAssert(m_nullEDFProcedureSetIndex == 0, "Index of the null EDF procedure set is expected to be 0.");
        }

        m_maxNumSurfaceMaterialDescriptors = 8192;
        m_optixSurfaceMaterialDescriptorBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_maxNumSurfaceMaterialDescriptors);
        m_optixSurfaceMaterialDescriptorBuffer->setElementSize(sizeof(Shared::SurfaceMaterialDescriptor));
        m_surfMatDescSlotManager.initialize(m_maxNumSurfaceMaterialDescriptors);

        m_optixContext["VLR::pv_materialDescriptorBuffer"]->set(m_optixSurfaceMaterialDescriptorBuffer);

        SurfaceNode::initialize(*this);
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
        SurfaceNode::finalize(*this);

        m_surfMatDescSlotManager.finalize();
        m_optixSurfaceMaterialDescriptorBuffer->destroy();

        unsetEDFProcedureSet(m_nullEDFProcedureSetIndex);
        m_optixCallableProgramNullEDF_evaluateEDFInternal->destroy();
        m_optixCallableProgramNullEDF_evaluateEmittanceInternal->destroy();
        m_optixCallableProgramNullEDF_setupEDF->destroy();

        unsetBSDFProcedureSet(m_nullBSDFProcedureSetIndex);
        m_optixCallableProgramNullBSDF_weightInternal->destroy();
        m_optixCallableProgramNullBSDF_evaluateBSDF_PDFInternal->destroy();
        m_optixCallableProgramNullBSDF_evaluateBSDFInternal->destroy();
        m_optixCallableProgramNullBSDF_sampleBSDFInternal->destroy();
        m_optixCallableProgramNullBSDF_matches->destroy();
        m_optixCallableProgramNullBSDF_getBaseColor->destroy();
        m_optixCallableProgramNullBSDF_setupBSDF->destroy();

        m_edfProcSetSlotManager.finalize();
        m_optixEDFProcedureSetBuffer->destroy();

        m_bsdfProcSetSlotManager.finalize();
        m_optixBSDFProcedureSetBuffer->destroy();

        m_optixMaterialWithAlpha->destroy();
        m_optixMaterialDefault->destroy();

        m_optixProgramException->destroy();
        m_optixProgramPathTracingMiss->destroy();
        m_optixProgramPathTracing->destroy();

        m_optixProgramPathTracingIteration->destroy();
        m_optixProgramShadowAnyHitWithAlpha->destroy();
        m_optixProgramAnyHitWithAlpha->destroy();

        m_optixCallableProgramFetchNormal->destroy();
        m_optixCallableProgramFetchAlpha->destroy();
        m_optixCallableProgramNullFetchNormal->destroy();
        m_optixCallableProgramNullFetchAlpha->destroy();

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

    uint32_t Context::setBSDFProcedureSet(const Shared::BSDFProcedureSet &procSet) {
        uint32_t index = m_bsdfProcSetSlotManager.getFirstAvailableSlot();
        {
            auto procSets = (Shared::BSDFProcedureSet*)m_optixBSDFProcedureSetBuffer->map();
            procSets[index] = procSet;
            m_optixBSDFProcedureSetBuffer->unmap();
        }
        m_bsdfProcSetSlotManager.setInUse(index);

        return index;
    }

    void Context::unsetBSDFProcedureSet(uint32_t index) {
        m_bsdfProcSetSlotManager.setNotInUse(index);
    }

    uint32_t Context::setEDFProcedureSet(const Shared::EDFProcedureSet &procSet) {
        uint32_t index = m_edfProcSetSlotManager.getFirstAvailableSlot();
        {
            auto procSets = (Shared::EDFProcedureSet*)m_optixEDFProcedureSetBuffer->map();
            procSets[index] = procSet;
            m_optixEDFProcedureSetBuffer->unmap();
        }
        m_edfProcSetSlotManager.setInUse(index);

        return index;
    }

    void Context::unsetEDFProcedureSet(uint32_t index) {
        m_edfProcSetSlotManager.setNotInUse(index);
    }

    uint32_t Context::setSurfaceMaterialDescriptor(const Shared::SurfaceMaterialDescriptor &matDesc) {
        uint32_t index = m_surfMatDescSlotManager.getFirstAvailableSlot();
        {
            auto matDescs = (Shared::SurfaceMaterialDescriptor*)m_optixSurfaceMaterialDescriptorBuffer->map();
            matDescs[index] = matDesc;
            m_optixSurfaceMaterialDescriptorBuffer->unmap();
        }
        m_surfMatDescSlotManager.setInUse(index);

        return index;
    }

    void Context::unsetSurfaceMaterialDescriptor(uint32_t index) {
        m_surfMatDescSlotManager.setNotInUse(index);
    }
}
