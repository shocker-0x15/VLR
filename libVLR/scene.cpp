#include "scene.h"

#include <random>

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

    defineClassID(Object, Node);
    defineClassID(Node, SurfaceNode);
    defineClassID(SurfaceNode, TriangleMeshSurfaceNode);
    defineClassID(Node, ParentNode);
    defineClassID(ParentNode, InternalNode);
    defineClassID(ParentNode, RootNode);
    defineClassID(Object, Scene);

    defineClassID(Object, Camera);
    defineClassID(Camera, PerspectiveCamera);
    defineClassID(Camera, EquirectangularCamera);

#undef defineClassID
    
    
    
    uint32_t Context::NextID = 0;
    
    Context::Context() {
        m_ID = getInstanceID();

        m_optixContext = optix::Context::create();

        m_optixContext->setRayTypeCount(Shared::RayType::NumTypes);

        {
            std::string ptx = readTxtFile("resources/ptxes/path_tracing.ptx");

            m_optixCallableProgramNullFetchAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::Null_NormalAlphaModifier_fetchAlpha");
            m_optixCallableProgramNullFetchNormal = m_optixContext->createProgramFromPTXString(ptx, "VLR::Null_NormalAlphaModifier_fetchNormal");
            m_optixCallableProgramFetchAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::NormalAlphaModifier_fetchAlpha");
            m_optixCallableProgramFetchNormal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NormalAlphaModifier_fetchNormal");

            m_optixProgramStochasticAlphaAnyHit = m_optixContext->createProgramFromPTXString(ptx, "VLR::stochasticAlphaAnyHit");
            m_optixProgramAlphaAnyHit = m_optixContext->createProgramFromPTXString(ptx, "VLR::alphaAnyHit");
            m_optixProgramPathTracingIteration = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingIteration");

            m_optixProgramPathTracing = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracing");
            m_optixProgramPathTracingMiss = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingMiss");
            m_optixProgramException = m_optixContext->createProgramFromPTXString(ptx, "VLR::exception");
        }



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



        m_optixContext->setEntryPointCount(1);
        m_optixContext->setRayGenerationProgram(0, m_optixProgramPathTracing);
        m_optixContext->setMissProgram(0, m_optixProgramPathTracingMiss);
        m_optixContext->setExceptionProgram(0, m_optixProgramException);

        SurfaceNode::initialize(*this);
        SurfaceMaterial::initialize(*this);
        Camera::initialize(*this);

        //m_optixContext->setPrintEnabled(true);
        //m_optixContext->setPrintBufferSize(4096);
        //m_optixContext->setExceptionEnabled(RT_EXCEPTION_BUFFER_ID_INVALID, true);
        //m_optixContext->setExceptionEnabled(RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true);
        //m_optixContext->setExceptionEnabled(RT_EXCEPTION_INTERNAL_ERROR, true);

        RTsize stackSize = m_optixContext->getStackSize();
        VLRDebugPrintf("Default Stack Size: %u\n", stackSize);
        m_optixContext->setStackSize(1280);
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

        m_optixProgramException->destroy();
        m_optixProgramPathTracingMiss->destroy();
        m_optixProgramPathTracing->destroy();

        m_optixProgramPathTracingIteration->destroy();
        m_optixProgramAlphaAnyHit->destroy();
        m_optixProgramStochasticAlphaAnyHit->destroy();

        m_optixCallableProgramFetchNormal->destroy();
        m_optixCallableProgramFetchAlpha->destroy();
        m_optixCallableProgramNullFetchNormal->destroy();
        m_optixCallableProgramNullFetchAlpha->destroy();

        m_optixContext->destroy();
    }

    void Context::bindOpenGLBuffer(uint32_t bufferID, uint32_t width, uint32_t height) {
        if (m_outputBuffer)
            m_outputBuffer->destroy();
        if (m_rngBuffer)
            m_rngBuffer->destroy();

        m_width = width;
        m_height = height;

        m_outputBuffer = m_optixContext->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, bufferID);
        m_outputBuffer->setFormat(RT_FORMAT_USER);
        m_outputBuffer->setElementSize(sizeof(RGBSpectrum));
        m_outputBuffer->setSize(m_width, m_height);
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

        optixContext->setTimeoutCallback([]() { return 1; }, 0.1);

        //uint32_t deviceIdx = 0;

        //int32_t numMaxTexs = optixContext->getMaxTextureCount();
        //int32_t numCPUThreads = optixContext->getCPUNumThreads();
        //RTsize usedHostMem = optixContext->getUsedHostMemory();
        //RTsize availMem = optixContext->getAvailableDeviceMemory(deviceIdx);

        //char name[256];
        //char pciBusId[16];
        //int computeCaps[2];
        //RTsize total_mem;
        //int clock_rate;
        //int threads_per_block;
        //int sm_count;
        //int execution_timeout_enabled;
        //int texture_count;
        //int tcc_driver;
        //int cuda_device_ordinal;
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(pciBusId), pciBusId);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clock_rate), &clock_rate);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(threads_per_block), &threads_per_block);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(sm_count), &sm_count);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(execution_timeout_enabled), &execution_timeout_enabled);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(texture_count), &texture_count);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tcc_driver), &tcc_driver);
        //optixContext->getDeviceAttribute(deviceIdx, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(cuda_device_ordinal), &cuda_device_ordinal);

        optixContext->validate();

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



    // ----------------------------------------------------------------
    // Shallow Hierarchy

    void SHGroup::addChild(SHTransform* transform) {
        TransformStatus status;
        SHGeometryGroup* descendant;
        status.hasGeometryDescendant = transform->hasGeometryDescendant(&descendant);
        m_transforms[transform] = status;
        if (status.hasGeometryDescendant) {
            optix::Transform optixTransform = transform->getOptiXObject();
            optixTransform->setChild(descendant->getOptiXObject());

            RTobject trChild;
            rtTransformGetChild(optixTransform->get(), &trChild);
            VLRAssert(trChild, "Transform must have a child.");

            m_optixGroup->addChild(optixTransform);
            m_optixAcceleration->markDirty();
            ++m_numValidTransforms;
        }
    }

    void SHGroup::removeChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        const TransformStatus status = m_transforms.at(transform);
        m_transforms.erase(transform);
        if (status.hasGeometryDescendant) {
            m_optixGroup->removeChild(transform->getOptiXObject());
            m_optixAcceleration->markDirty();
            --m_numValidTransforms;
        }
    }

    void SHGroup::updateChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);
        SHGeometryGroup* descendant;
        optix::Transform optixTransform = transform->getOptiXObject();
        if (status.hasGeometryDescendant) {
            if (!transform->hasGeometryDescendant()) {
                m_optixGroup->removeChild(optixTransform);
                m_optixAcceleration->markDirty();
                status.hasGeometryDescendant = false;
                --m_numValidTransforms;
            }
        }
        else {
            if (transform->hasGeometryDescendant(&descendant)) {
                optixTransform->setChild(descendant->getOptiXObject());

                RTobject trChild;
                rtTransformGetChild(optixTransform->get(), &trChild);
                VLRAssert(trChild, "Transform must have a child.");

                m_optixGroup->addChild(optixTransform);
                m_optixAcceleration->markDirty();
                status.hasGeometryDescendant = true;
                ++m_numValidTransforms;
            }
        }
    }

    void SHGroup::printOptiXHierarchy() {
        std::stack<RTobject> stackRTObjects;
        std::stack<RTobjecttype> stackRTObjectTypes;

        std::set<RTgroup> groupList;
        std::set<RTtransform> transformList;
        std::set<RTgeometrygroup> geometryGroupList;
        std::set<RTgeometryinstance> geometryInstanceList;

        stackRTObjects.push(m_optixGroup->get());
        stackRTObjectTypes.push(RT_OBJECTTYPE_GROUP);
        while (!stackRTObjects.empty()) {
            RTobject object = stackRTObjects.top();
            RTobjecttype objType = stackRTObjectTypes.top();
            stackRTObjects.pop();
            stackRTObjectTypes.pop();

            VLRDebugPrintf("0x%p: ", object);

            switch (objType) {
            case RT_OBJECTTYPE_GROUP: {
                auto group = (RTgroup)object;
                VLRDebugPrintf("Group\n");

                groupList.insert(group);

                uint32_t numChildren;
                rtGroupGetChildCount(group, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTobject childObject = nullptr;
                    RTobjecttype childObjType;
                    rtGroupGetChild(group, i, &childObject);
                    rtGroupGetChildType(group, i, &childObjType);

                    VLRDebugPrintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(childObjType);
                }

                break;
            }
            case RT_OBJECTTYPE_TRANSFORM: {
                auto transform = (RTtransform)object;
                VLRDebugPrintf("Transform\n");

                transformList.insert(transform);

                RTobject childObject = nullptr;
                RTobjecttype childObjType;
                rtTransformGetChild(transform, &childObject);
                rtTransformGetChildType(transform, &childObjType);

                VLRDebugPrintf("- 0x%p\n", childObject);

                stackRTObjects.push(childObject);
                stackRTObjectTypes.push(childObjType);

                break;
            }
            case RT_OBJECTTYPE_SELECTOR: {
                VLRAssert_NotImplemented();
                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_GROUP: {
                auto geometryGroup = (RTgeometrygroup)object;
                VLRDebugPrintf("GeometryGroup\n");

                geometryGroupList.insert(geometryGroup);

                uint32_t numChildren;
                rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTgeometryinstance childObject = nullptr;
                    rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                    VLRDebugPrintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(RT_OBJECTTYPE_GEOMETRY_INSTANCE);
                }

                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_INSTANCE: {
                auto geometryInstance = (RTgeometryinstance)object;
                VLRDebugPrintf("GeometryInstance\n");

                geometryInstanceList.insert(geometryInstance);

                break;
            }
            default:
                VLRDebugPrintf("\n");
                VLRAssert_ShouldNotBeCalled();
                break;
            }

            VLRDebugPrintf("\n");
        }



        VLRDebugPrintf("Groups:\n");
        for (auto group : groupList) {
            VLRDebugPrintf("  0x%p:\n", group);
            uint32_t numChildren;
            rtGroupGetChildCount(group, &numChildren);
            RTacceleration acceleration;
            rtGroupGetAcceleration(group, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            VLRDebugPrintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTobject childObject = nullptr;
                rtGroupGetChild(group, i, &childObject);

                VLRDebugPrintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        VLRDebugPrintf("Transforms:\n");
        for (auto transform : transformList) {
            VLRDebugPrintf("  0x%p:\n", transform);
            RTobject childObject = nullptr;
            rtTransformGetChild(transform, &childObject);
            float mat[16];
            float invMat[16];
            rtTransformGetMatrix(transform, true, mat, invMat);
            VLRDebugPrintf("    Matrix\n");
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
            VLRDebugPrintf("    Inverse Matrix\n");
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[0], invMat[4], invMat[8], invMat[12]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[1], invMat[5], invMat[9], invMat[13]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[2], invMat[6], invMat[10], invMat[14]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[3], invMat[7], invMat[11], invMat[15]);

            VLRDebugPrintf("  - 0x%p\n", childObject);
        }

        VLRDebugPrintf("GeometryGroups:\n");
        for (auto geometryGroup : geometryGroupList) {
            VLRDebugPrintf("  0x%p:\n", geometryGroup);
            uint32_t numChildren;
            rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
            RTacceleration acceleration;
            rtGeometryGroupGetAcceleration(geometryGroup, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            VLRDebugPrintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTgeometryinstance childObject = nullptr;
                rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                VLRDebugPrintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        VLRDebugPrintf("GeometryInstances:\n");
        for (auto geometryInstance : geometryInstanceList) {
            VLRDebugPrintf("  0x%p:\n", geometryInstance);
        }
    }



    void SHTransform::resolveTransform() {
        int32_t stackIdx = 0;
        const SHTransform* stack[5];
        std::fill_n(stack, lengthof(stack), nullptr);
        const SHTransform* nextSHTr = m_childIsTransform ? m_childTransform : nullptr;
        while (nextSHTr) {
            stack[stackIdx++] = nextSHTr;
            nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
        }

        StaticTransform res;
        std::string concatenatedName = "";
        --stackIdx;
        while (stackIdx >= 0) {
            const SHTransform* shtr = stack[stackIdx--];
            res = shtr->m_transform * res;
            concatenatedName = "-" + shtr->getName() + concatenatedName;
        }
        res = m_transform * res;
        concatenatedName = m_name + concatenatedName;

        float mat[16], invMat[16];
        res.getArrays(mat, invMat);
        m_optixTransform->setMatrix(true, mat, invMat);

        //if (true/*m_parent*/) {
        //    VLRDebugPrintf("%s:\n", concatenatedName.c_str());
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
        //    VLRDebugPrintf("\n");
        //}
    }

    void SHTransform::setTransform(const StaticTransform &transform) {
        m_transform = transform;
        resolveTransform();
    }

    void SHTransform::update() {
        resolveTransform();
    }

    bool SHTransform::isStatic() const {
        // TODO: implement
        return true;
    }

    StaticTransform SHTransform::getStaticTransform() const {
        if (isStatic()) {
            float mat[16], invMat[16];
            m_optixTransform->getMatrix(true, mat, invMat);
            return StaticTransform(Matrix4x4(mat));
        }
        else {
            return StaticTransform();
        }
    }

    void SHTransform::setChild(SHGeometryGroup* geomGroup) {
        VLRAssert(!m_childIsTransform, "Transform which doesn't have a child transform can have a geometry group as a child.");
        m_childGeometryGroup = geomGroup;
    }

    bool SHTransform::hasGeometryDescendant(SHGeometryGroup** descendant) const {
        if (descendant)
            *descendant = nullptr;

        const SHTransform* nextSHTr = this;
        while (nextSHTr) {
            if (!nextSHTr->m_childIsTransform && nextSHTr->m_childGeometryGroup != nullptr) {
                if (descendant)
                    *descendant = nextSHTr->m_childGeometryGroup;
                return true;
            }
            else {
                nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
            }
        }

        return false;
    }



    void SHGeometryGroup::addGeometryInstance(const SHGeometryInstance* instance) {
        m_instances.insert(instance);
        m_optixGeometryGroup->addChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGeometryGroup::removeGeometryInstance(const SHGeometryInstance* instance) {
        m_instances.erase(instance);
        m_optixGeometryGroup->removeChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



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
    void DiscreteDistribution1DTemplate<RealType>::getInternalType(Shared::DiscreteDistribution1DTemplate<RealType>* instance) {
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
        m_CDF->destroy();
        m_PDF->destroy();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution1DTemplate<RealType>::getInternalType(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) {
        new (instance) Shared::RegularConstantContinuousDistribution1DTemplate<RealType>(m_PDF->getId(), m_CDF->getId(), m_integral, m_numValues);
    }

    template class RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::initialize(Context &context, const RealType* values, size_t numD1, size_t numD2) {
        optix::Context optixContext = context.getOptiXContext();

        m_num1DDists = numD2;

        // JP: まず各行に関するDistribution1Dを作成する。
        // EN: First, create Distribution1D's for every rows.
        m_1DDists = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_num1DDists);
        m_1DDists->setElementSize(sizeof(RegularConstantContinuousDistribution1DTemplate<RealType>));

        RegularConstantContinuousDistribution1DTemplate<RealType>* dists = (RegularConstantContinuousDistribution1DTemplate<RealType>*)m_1DDists->map();

        CompensatedSum<RealType> sum(0);
        for (int i = 0; i < m_num1DDists; ++i) {
            dists[i].initialize(context, values + i * numD1, numD1);
            sum += dists[i].getIntegral();
        }
        m_integral = sum;

        // JP: 各行の積分値を用いてDistribution1Dを作成する。
        // EN: create a Distribution1D using integral values of each row.
        RealType* integrals = new RealType[m_num1DDists];
        for (int i = 0; i < m_num1DDists; ++i)
            integrals[i] = dists[i].getIntegral();
        m_top1DDist.initialize(context, integrals, m_num1DDists);
        delete[] integrals;
        
        VLRAssert(std::isfinite(m_integral), "invalid integral value.");

        m_1DDists->unmap();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::finalize(Context &context) {
        m_top1DDist.finalize(context);

        RegularConstantContinuousDistribution1DTemplate<RealType>* dists = (RegularConstantContinuousDistribution1DTemplate<RealType>*)m_1DDists->map();
        for (int i = m_num1DDists - 1; i >= 0; --i) {
            dists[i].finalize(context);
        }
        m_1DDists->unmap();

        m_1DDists->destroy();
    }

    template <typename RealType>
    void RegularConstantContinuousDistribution2DTemplate<RealType>::getInternalType(Shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) {
        Shared::RegularConstantContinuousDistribution1DTemplate<RealType> top1DDist;
        m_top1DDist.getInternalType(&top1DDist);
        new (instance) Shared::RegularConstantContinuousDistribution2DTemplate<RealType>(m_1DDists->getId(), m_num1DDists, m_integral, top1DDist);
    }

    template class RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
    
    
    
    // ----------------------------------------------------------------
    // Material
    
    const size_t sizesOfDataFormats[(uint32_t)DataFormat::Num] = {
        sizeof(RGB8x3),
        sizeof(RGB_8x4),
        sizeof(RGBA8x4),
        sizeof(RGBA16Fx4),
        sizeof(RGBA32Fx4),
        sizeof(Gray8),
    };

    DataFormat Image2D::getInternalFormat(DataFormat inputFormat) {
        switch (inputFormat) {
        case DataFormat::RGB8x3:
            return DataFormat::RGBA8x4;
        case DataFormat::RGB_8x4:
            return DataFormat::RGBA8x4;
        case DataFormat::RGBA8x4:
            return DataFormat::RGBA8x4;
        case DataFormat::RGBA16Fx4:
            return DataFormat::RGBA16Fx4;
        case DataFormat::RGBA32Fx4:
            return DataFormat::RGBA32Fx4;
        case DataFormat::Gray8:
            return DataFormat::Gray8;
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }
        return DataFormat::RGBA8x4;
    }

    Image2D::Image2D(Context &context, uint32_t width, uint32_t height, DataFormat dataFormat) :
        Object(context), m_width(width), m_height(height), m_dataFormat(dataFormat) {
        optix::Context optixContext = context.getOptiXContext();
        switch (m_dataFormat) {
        case VLR::DataFormat::RGB8x3:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE3, width, height);
            break;
        case VLR::DataFormat::RGB_8x4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
            break;
        case VLR::DataFormat::RGBA8x4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
            break;
        case VLR::DataFormat::RGBA16Fx4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_HALF4, width, height);
            break;
        case VLR::DataFormat::RGBA32Fx4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, width, height);
            break;
        case VLR::DataFormat::Gray8:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, width, height);
            break;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    Image2D::~Image2D() {
        m_optixDataBuffer->destroy();
    }



    LinearImage2D::LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat) :
        Image2D(context, width, height, Image2D::getInternalFormat(dataFormat)) {
        m_data.resize(getStride() * getWidth() * getHeight());

        switch (dataFormat) {
        case DataFormat::RGB8x3: {
            auto srcHead = (const RGB8x3*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    const RGB8x3 &src = *(srcLineHead + x);
                    RGBA8x4 &dst = *(dstLineHead + x);
                    dst.r = src.r;
                    dst.g = src.g;
                    dst.b = src.b;
                    dst.a = 255;
                }
            }
            break;
        }
        case DataFormat::RGB_8x4: {
            auto srcHead = (const RGB_8x4*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    const RGB_8x4 &src = *(srcLineHead + x);
                    RGBA8x4 &dst = *(dstLineHead + x);
                    dst.r = src.r;
                    dst.g = src.g;
                    dst.b = src.b;
                    dst.a = 255;
                }
            }
            break;
        }
        case DataFormat::RGBA8x4: {
            auto srcHead = (const RGBA8x4*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            std::copy_n(srcHead, width * height, dstHead);
            break;
        }
        case DataFormat::RGBA16Fx4: {
            auto srcHead = (const RGBA16Fx4*)linearData;
            auto dstHead = (RGBA16Fx4*)m_data.data();
            std::copy_n(srcHead, width * height, dstHead);
            break;
        }
        case DataFormat::RGBA32Fx4: {
            auto srcHead = (const RGBA32Fx4*)linearData;
            auto dstHead = (RGBA32Fx4*)m_data.data();
            std::copy_n(srcHead, width * height, dstHead);
            break;
        }
        case DataFormat::Gray8: {
            auto srcHead = (const Gray8*)linearData;
            auto dstHead = (Gray8*)m_data.data();
            std::copy_n(srcHead, width * height, dstHead);
            break;
        }
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }

        optix::Buffer buffer = getOptiXObject();
        auto dstData = (uint8_t*)buffer->map();
        {
            std::copy(m_data.cbegin(), m_data.cend(), dstData);
        }
        buffer->unmap();
    }



    FloatTexture::FloatTexture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    FloatTexture::~FloatTexture() {
        m_optixTextureSampler->destroy();
    }

    void FloatTexture::setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    Float2Texture::Float2Texture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    Float2Texture::~Float2Texture() {
        m_optixTextureSampler->destroy();
    }

    void Float2Texture::setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    Float3Texture::Float3Texture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    Float3Texture::~Float3Texture() {
        m_optixTextureSampler->destroy();
    }

    void Float3Texture::setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    ConstantFloat3Texture::ConstantFloat3Texture(Context &context, const float value[3]) :
        Float3Texture(context) {
        float value4[] = { value[0], value[1], value[2], 0 };
        m_image = new LinearImage2D(context, (const uint8_t*)value4, 1, 1, DataFormat::RGBA32Fx4);
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    ConstantFloat3Texture::~ConstantFloat3Texture() {
        delete m_image;
    }



    ImageFloat3Texture::ImageFloat3Texture(Context &context, const Image2D* image) :
        Float3Texture(context), m_image(image) {
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }



    Float4Texture::Float4Texture(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_REPEAT);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_REPEAT);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }

    Float4Texture::~Float4Texture() {
        m_optixTextureSampler->destroy();
    }

    void Float4Texture::setTextureFilterMode(TextureFilter minification, TextureFilter magnification, TextureFilter mipmapping) {
        m_optixTextureSampler->setFilteringModes((RTfiltermode)minification, (RTfiltermode)magnification, (RTfiltermode)mipmapping);
    }



    ConstantFloat4Texture::ConstantFloat4Texture(Context &context, const float value[4]) :
        Float4Texture(context) {
        m_image = new LinearImage2D(context, (const uint8_t*)value, 1, 1, DataFormat::RGBA32Fx4);
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    ConstantFloat4Texture::~ConstantFloat4Texture() {
        delete m_image;
    }
    
    
    
    ImageFloat4Texture::ImageFloat4Texture(Context &context, const Image2D* image) :
        Float4Texture(context), m_image(image) {
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }



    // static
    void SurfaceMaterial::commonInitializeProcedure(Context &context, const char* identifiers[10], OptiXProgramSet* programSet) {
        std::string ptx = readTxtFile("resources/ptxes/materials.ptx");

        optix::Context optixContext = context.getOptiXContext();

        if (identifiers[0] && identifiers[1] && identifiers[2] && identifiers[3] && identifiers[4] && identifiers[5] && identifiers[6]) {
            programSet->callableProgramSetupBSDF = optixContext->createProgramFromPTXString(ptx, identifiers[0]);

            programSet->callableProgramGetBaseColor = optixContext->createProgramFromPTXString(ptx, identifiers[1]);
            programSet->callableProgramBSDFmatches = optixContext->createProgramFromPTXString(ptx, identifiers[2]);
            programSet->callableProgramSampleBSDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[3]);
            programSet->callableProgramEvaluateBSDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[4]);
            programSet->callableProgramEvaluateBSDF_PDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[5]);
            programSet->callableProgramBSDFWeightInternal = optixContext->createProgramFromPTXString(ptx, identifiers[6]);

            Shared::BSDFProcedureSet bsdfProcSet;
            {
                bsdfProcSet.progGetBaseColor = programSet->callableProgramGetBaseColor->getId();
                bsdfProcSet.progBSDFmatches = programSet->callableProgramBSDFmatches->getId();
                bsdfProcSet.progSampleBSDFInternal = programSet->callableProgramSampleBSDFInternal->getId();
                bsdfProcSet.progEvaluateBSDFInternal = programSet->callableProgramEvaluateBSDFInternal->getId();
                bsdfProcSet.progEvaluateBSDF_PDFInternal = programSet->callableProgramEvaluateBSDF_PDFInternal->getId();
                bsdfProcSet.progWeightInternal = programSet->callableProgramBSDFWeightInternal->getId();
            }
            programSet->bsdfProcedureSetIndex = context.setBSDFProcedureSet(bsdfProcSet);
        }

        if (identifiers[7] && identifiers[8] && identifiers[9]) {
            programSet->callableProgramSetupEDF = optixContext->createProgramFromPTXString(ptx, identifiers[7]);

            programSet->callableProgramEvaluateEmittanceInternal = optixContext->createProgramFromPTXString(ptx, identifiers[8]);
            programSet->callableProgramEvaluateEDFInternal = optixContext->createProgramFromPTXString(ptx, identifiers[9]);

            Shared::EDFProcedureSet edfProcSet;
            {
                edfProcSet.progEvaluateEmittanceInternal = programSet->callableProgramEvaluateEmittanceInternal->getId();
                edfProcSet.progEvaluateEDFInternal = programSet->callableProgramEvaluateEDFInternal->getId();
            }
            programSet->edfProcedureSetIndex = context.setEDFProcedureSet(edfProcSet);
        }
    }

    // static
    void SurfaceMaterial::commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet) {
        if (programSet.callableProgramSetupEDF) {
            context.unsetEDFProcedureSet(programSet.edfProcedureSetIndex);

            programSet.callableProgramEvaluateEDFInternal->destroy();
            programSet.callableProgramEvaluateEmittanceInternal->destroy();

            programSet.callableProgramSetupEDF->destroy();
        }

        if (programSet.callableProgramSetupBSDF) {
            context.unsetBSDFProcedureSet(programSet.bsdfProcedureSetIndex);

            programSet.callableProgramBSDFWeightInternal->destroy();
            programSet.callableProgramEvaluateBSDF_PDFInternal->destroy();
            programSet.callableProgramEvaluateBSDFInternal->destroy();
            programSet.callableProgramSampleBSDFInternal->destroy();
            programSet.callableProgramBSDFmatches->destroy();
            programSet.callableProgramGetBaseColor->destroy();

            programSet.callableProgramSetupBSDF->destroy();
        }
    }

    // static
    uint32_t SurfaceMaterial::setupMaterialDescriptorHead(Context &context, const OptiXProgramSet &progSet, Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) {
        Shared::SurfaceMaterialHead &head = *(Shared::SurfaceMaterialHead*)&matDesc->i1[baseIndex];

        if (progSet.callableProgramSetupBSDF) {
            head.progSetupBSDF = progSet.callableProgramSetupBSDF->getId();
            head.bsdfProcedureSetIndex = progSet.bsdfProcedureSetIndex;
        }
        else {
            head.progSetupBSDF = context.getOptixCallableProgramNullBSDF_setupBSDF()->getId();
            head.bsdfProcedureSetIndex = context.getNullBSDFProcedureSetIndex();
        }

        if (progSet.callableProgramSetupEDF) {
            head.progSetupEDF = progSet.callableProgramSetupEDF->getId();
            head.edfProcedureSetIndex = progSet.edfProcedureSetIndex;
        }
        else {
            head.progSetupEDF = context.getOptixCallableProgramNullEDF_setupEDF()->getId();
            head.edfProcedureSetIndex = context.getNullEDFProcedureSetIndex();
        }

        return baseIndex + sizeof(Shared::SurfaceMaterialHead) / 4;
    }

    // static
    void SurfaceMaterial::initialize(Context &context) {
        MatteSurfaceMaterial::initialize(context);
        SpecularReflectionSurfaceMaterial::initialize(context);
        SpecularScatteringSurfaceMaterial::initialize(context);
        UE4SurfaceMaterial::initialize(context);
        DiffuseEmitterSurfaceMaterial::initialize(context);
        MultiSurfaceMaterial::initialize(context);
    }

    // static
    void SurfaceMaterial::finalize(Context &context) {
        MultiSurfaceMaterial::finalize(context);
        DiffuseEmitterSurfaceMaterial::finalize(context);
        UE4SurfaceMaterial::finalize(context);
        SpecularScatteringSurfaceMaterial::finalize(context);
        SpecularReflectionSurfaceMaterial::finalize(context);
        MatteSurfaceMaterial::finalize(context);
    }

    SurfaceMaterial::SurfaceMaterial(Context &context) : Object(context) {
        optix::Context optixContext = context.getOptiXContext();
        m_optixMaterial = optixContext->createMaterial();

        m_optixMaterial->setClosestHitProgram(Shared::RayType::Primary, context.getOptiXProgramPathTracingIteration());
        m_optixMaterial->setClosestHitProgram(Shared::RayType::Scattered, context.getOptiXProgramPathTracingIteration());
        m_optixMaterial->setAnyHitProgram(Shared::RayType::Primary, context.getOptiXProgramStochasticAlphaAnyHit());
        m_optixMaterial->setAnyHitProgram(Shared::RayType::Scattered, context.getOptiXProgramStochasticAlphaAnyHit());
        m_optixMaterial->setAnyHitProgram(Shared::RayType::Shadow, context.getOptiXProgramAlphaAnyHit());

        m_matIndex = 0xFFFFFFFF;
    }
    
    SurfaceMaterial::~SurfaceMaterial() {
        if (m_matIndex != 0xFFFFFFFF)
            m_context.unsetSurfaceMaterialDescriptor(m_matIndex);
        m_matIndex = 0xFFFFFFFF;

        m_optixMaterial->destroy();
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MatteSurfaceMaterial::OptiXProgramSets;

    // static
    void MatteSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MatteSurfaceMaterial_setupBSDF",
            "VLR::MatteBRDF_getBaseColor",
            "VLR::MatteBRDF_matches",
            "VLR::MatteBRDF_sampleBSDFInternal",
            "VLR::MatteBRDF_evaluateBSDFInternal",
            "VLR::MatteBRDF_evaluateBSDF_PDFInternal",
            "VLR::MatteBRDF_weightInternal",
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MatteSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    MatteSurfaceMaterial::MatteSurfaceMaterial(Context &context, const Float4Texture* texAlbedoRoughness) :
        SurfaceMaterial(context), m_texAlbedoRoughness(texAlbedoRoughness) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
        //m_optixMaterial["VLR::pv_materialIndex"]->setUint(m_matIndex); // 何故かvalidate()でエラーになる。
        m_optixMaterial["VLR::pv_materialIndex"]->setUserData(sizeof(m_matIndex), &m_matIndex);
    }

    MatteSurfaceMaterial::~MatteSurfaceMaterial() {
    }

    uint32_t MatteSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::MatteSurfaceMaterial &mat = *(Shared::MatteSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texAlbedoRoughness = m_texAlbedoRoughness->getOptiXObject()->getId();

        return baseIndex + sizeof(Shared::MatteSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularReflectionSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularReflectionSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::SpecularReflectionSurfaceMaterial_setupBSDF",
            "VLR::SpecularBRDF_getBaseColor",
            "VLR::SpecularBRDF_matches",
            "VLR::SpecularBRDF_sampleBSDFInternal",
            "VLR::SpecularBRDF_evaluateBSDFInternal",
            "VLR::SpecularBRDF_evaluateBSDF_PDFInternal",
            "VLR::SpecularBRDF_weightInternal",
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void SpecularReflectionSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    SpecularReflectionSurfaceMaterial::SpecularReflectionSurfaceMaterial(Context &context, const Float3Texture* texCoeffR, const Float3Texture* texEta, const Float3Texture* tex_k) :
        SurfaceMaterial(context), m_texCoeffR(texCoeffR), m_texEta(texEta), m_tex_k(tex_k) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
        //m_optixMaterial["VLR::pv_materialIndex"]->setUint(m_matIndex); // 何故かvalidate()でエラーになる。
        m_optixMaterial["VLR::pv_materialIndex"]->setUserData(sizeof(m_matIndex), &m_matIndex);
    }

    SpecularReflectionSurfaceMaterial::~SpecularReflectionSurfaceMaterial() {
    }

    uint32_t SpecularReflectionSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::SpecularReflectionSurfaceMaterial &mat = *(Shared::SpecularReflectionSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texCoeffR = m_texCoeffR->getOptiXObject()->getId();
        mat.texEta = m_texEta->getOptiXObject()->getId();
        mat.tex_k = m_tex_k->getOptiXObject()->getId();

        return baseIndex + sizeof(Shared::SpecularReflectionSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> SpecularScatteringSurfaceMaterial::OptiXProgramSets;

    // static
    void SpecularScatteringSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::SpecularScatteringSurfaceMaterial_setupBSDF",
            "VLR::SpecularBSDF_getBaseColor",
            "VLR::SpecularBSDF_matches",
            "VLR::SpecularBSDF_sampleBSDFInternal",
            "VLR::SpecularBSDF_evaluateBSDFInternal",
            "VLR::SpecularBSDF_evaluateBSDF_PDFInternal",
            "VLR::SpecularBSDF_weightInternal",
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void SpecularScatteringSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    SpecularScatteringSurfaceMaterial::SpecularScatteringSurfaceMaterial(Context &context, const Float3Texture* texCoeff, const Float3Texture* texEtaExt, const Float3Texture* texEtaInt) :
        SurfaceMaterial(context), m_texCoeff(texCoeff), m_texEtaExt(texEtaExt), m_texEtaInt(texEtaInt) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
        //m_optixMaterial["VLR::pv_materialIndex"]->setUint(m_matIndex); // 何故かvalidate()でエラーになる。
        m_optixMaterial["VLR::pv_materialIndex"]->setUserData(sizeof(m_matIndex), &m_matIndex);
    }

    SpecularScatteringSurfaceMaterial::~SpecularScatteringSurfaceMaterial() {
    }

    uint32_t SpecularScatteringSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::SpecularScatteringSurfaceMaterial &mat = *(Shared::SpecularScatteringSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texCoeff = m_texCoeff->getOptiXObject()->getId();
        mat.texEtaExt = m_texEtaExt->getOptiXObject()->getId();
        mat.texEtaInt = m_texEtaInt->getOptiXObject()->getId();

        return baseIndex + sizeof(Shared::SpecularScatteringSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> UE4SurfaceMaterial::OptiXProgramSets;

    // static
    void UE4SurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::UE4SurfaceMaterial_setupBSDF",
            "VLR::UE4BRDF_getBaseColor",
            "VLR::UE4BRDF_matches",
            "VLR::UE4BRDF_sampleBSDFInternal",
            "VLR::UE4BRDF_evaluateBSDFInternal",
            "VLR::UE4BRDF_evaluateBSDF_PDFInternal",
            "VLR::UE4BRDF_weightInternal",
            nullptr,
            nullptr,
            nullptr
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void UE4SurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    UE4SurfaceMaterial::UE4SurfaceMaterial(Context &context, const Float3Texture* texBaseColor, const Float2Texture* texRoughnessMetallic) :
        SurfaceMaterial(context), m_texBaseColor(texBaseColor), m_texRoughnessMetallic(texRoughnessMetallic) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
        m_optixMaterial["VLR::pv_materialIndex"]->setUserData(sizeof(m_matIndex), &m_matIndex);
    }

    UE4SurfaceMaterial::~UE4SurfaceMaterial() {
    }

    uint32_t UE4SurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::UE4SurfaceMaterial &mat = *(Shared::UE4SurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texBaseColor = m_texBaseColor->getOptiXObject()->getId();
        mat.texRoughnessMetallic = m_texRoughnessMetallic->getOptiXObject()->getId();

        return baseIndex + sizeof(Shared::UE4SurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> DiffuseEmitterSurfaceMaterial::OptiXProgramSets;

    // static
    void DiffuseEmitterSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            "VLR::DiffuseEmitterSurfaceMaterial_setupEDF",
            "VLR::DiffuseEDF_evaluateEmittanceInternal",
            "VLR::DiffuseEDF_evaluateEDFInternal"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void DiffuseEmitterSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    DiffuseEmitterSurfaceMaterial::DiffuseEmitterSurfaceMaterial(Context &context, const Float3Texture* texEmittance) :
    SurfaceMaterial(context), m_texEmittance(texEmittance) {
        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
        //m_optixMaterial["VLR::pv_materialIndex"]->setUint(m_matIndex); // 何故かvalidate()でエラーになる。
        m_optixMaterial["VLR::pv_materialIndex"]->setUserData(sizeof(m_matIndex), &m_matIndex);
    }

    DiffuseEmitterSurfaceMaterial::~DiffuseEmitterSurfaceMaterial() {
    }

    uint32_t DiffuseEmitterSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::DiffuseEmitterSurfaceMaterial &mat = *(Shared::DiffuseEmitterSurfaceMaterial*)&matDesc->i1[baseIndex];
        mat.texEmittance = m_texEmittance->getOptiXObject()->getId();

        return baseIndex + sizeof(Shared::DiffuseEmitterSurfaceMaterial) / 4;
    }



    std::map<uint32_t, SurfaceMaterial::OptiXProgramSet> MultiSurfaceMaterial::OptiXProgramSets;

    // static
    void MultiSurfaceMaterial::initialize(Context &context) {
        const char* identifiers[] = {
            "VLR::MultiSurfaceMaterial_setupBSDF",
            "VLR::MultiBSDF_getBaseColor",
            "VLR::MultiBSDF_matches",
            "VLR::MultiBSDF_sampleBSDFInternal",
            "VLR::MultiBSDF_evaluateBSDFInternal",
            "VLR::MultiBSDF_evaluateBSDF_PDFInternal",
            "VLR::MultiBSDF_weightInternal",
            "VLR::MultiSurfaceMaterial_setupEDF",
            "VLR::MultiEDF_evaluateEmittanceInternal",
            "VLR::MultiEDF_evaluateEDFInternal"
        };
        OptiXProgramSet programSet;
        commonInitializeProcedure(context, identifiers, &programSet);

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void MultiSurfaceMaterial::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());
        commonFinalizeProcedure(context, programSet);
    }

    MultiSurfaceMaterial::MultiSurfaceMaterial(Context &context, const SurfaceMaterial** materials, uint32_t numMaterials) :
        SurfaceMaterial(context) {
        VLRAssert(numMaterials <= lengthof(m_materials), "numMaterials should be less than or equal to %u", lengthof(m_materials));
        std::copy_n(materials, numMaterials, m_materials);
        m_numMaterials = numMaterials;

        Shared::SurfaceMaterialDescriptor matDesc;
        setupMaterialDescriptor(&matDesc, 0);

        m_matIndex = m_context.setSurfaceMaterialDescriptor(matDesc);
        //m_optixMaterial["VLR::pv_materialIndex"]->setUint(m_matIndex); // 何故かvalidate()でエラーになる。
        m_optixMaterial["VLR::pv_materialIndex"]->setUserData(sizeof(m_matIndex), &m_matIndex);
    }

    MultiSurfaceMaterial::~MultiSurfaceMaterial() {
    }

    uint32_t MultiSurfaceMaterial::setupMaterialDescriptor(Shared::SurfaceMaterialDescriptor* matDesc, uint32_t baseIndex) const {
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        baseIndex = setupMaterialDescriptorHead(m_context, progSet, matDesc, baseIndex);
        Shared::MultiSurfaceMaterial &mat = *(Shared::MultiSurfaceMaterial*)&matDesc->i1[baseIndex];
        baseIndex += sizeof(Shared::MultiSurfaceMaterial) / 4;

        uint32_t matOffsets[4] = { 0, 0, 0, 0 };
        VLRAssert(lengthof(matOffsets) == lengthof(m_materials), "Two sizes must match.");
        for (int i = 0; i < m_numMaterials; ++i) {
            const SurfaceMaterial* mat = m_materials[i];
            matOffsets[i] = baseIndex;
            baseIndex = mat->setupMaterialDescriptor(matDesc, baseIndex);
        }
        VLRAssert(baseIndex <= VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS, "exceeds the size of SurfaceMaterialDescriptor.");

        mat.matOffset0 = matOffsets[0];
        mat.matOffset1 = matOffsets[1];
        mat.matOffset2 = matOffsets[2];
        mat.matOffset3 = matOffsets[3];
        mat.numMaterials = m_numMaterials;

        return baseIndex;
    }

    bool MultiSurfaceMaterial::isEmitting() const {
        for (int i = 0; i < m_numMaterials; ++i) {
            if (m_materials[i]->isEmitting())
                return true;
        }
        return false;
    }

    // END: Material
    // ----------------------------------------------------------------
    
    
    
    // static
    void SurfaceNode::initialize(Context &context) {
        TriangleMeshSurfaceNode::initialize(context);
    }

    // static
    void SurfaceNode::finalize(Context &context) {
        TriangleMeshSurfaceNode::finalize(context);
    }

    void SurfaceNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);
    }

    void SurfaceNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);
    }



    std::map<uint32_t, TriangleMeshSurfaceNode::OptiXProgramSet> TriangleMeshSurfaceNode::OptiXProgramSets;

    // static
    void TriangleMeshSurfaceNode::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/triangle_intersection.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.programIntersectTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::intersectTriangle");
        programSet.programCalcBBoxForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::calcBBoxForTriangle");

        programSet.callableProgramDecodeHitPointForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::decodeHitPointForTriangle");
        programSet.callableProgramDecodeTexCoordForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::decodeTexCoordForTriangle");

        programSet.callableProgramSampleTriangleMesh = optixContext->createProgramFromPTXString(ptx, "VLR::sampleTriangleMesh");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void TriangleMeshSurfaceNode::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleTriangleMesh->destroy();

        programSet.callableProgramDecodeTexCoordForTriangle->destroy();
        programSet.callableProgramDecodeHitPointForTriangle->destroy();

        programSet.programCalcBBoxForTriangle->destroy();
        programSet.programIntersectTriangle->destroy();

        OptiXProgramSets.erase(context.getID());
    }

    TriangleMeshSurfaceNode::TriangleMeshSurfaceNode(Context &context, const std::string &name) : SurfaceNode(context, name) {
    }

    TriangleMeshSurfaceNode::~TriangleMeshSurfaceNode() {
        for (auto it = m_shGeometryInstances.crbegin(); it != m_shGeometryInstances.crend(); ++it)
            delete *it;
        m_shGeometryInstances.clear();

        for (auto it = m_optixGeometries.begin(); it != m_optixGeometries.end(); ++it) {
            OptiXGeometry &geom = *it;
            geom.primDist.finalize(m_context);
            geom.optixIndexBuffer->destroy();
            geom.optixGeometry->destroy();
        }
        m_optixVertexBuffer->destroy();
    }

    void TriangleMeshSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
    }

    void TriangleMeshSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 削除した親に対してジオメトリインスタンスの削除を行わせる。
        std::set<SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryRemoved, delta);
    }

    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        optix::Context optixContext = m_context.getOptiXContext();
        m_optixVertexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_vertices.size());
        m_optixVertexBuffer->setElementSize(sizeof(Shared::Vertex));
        {
            auto dstVertices = (Shared::Vertex*)m_optixVertexBuffer->map();
            static_assert(sizeof(Vertex) == sizeof(Shared::Vertex), "These two types must match in size (and structure).");
            std::copy_n((Shared::Vertex*)m_vertices.data(), m_vertices.size(), dstVertices);
            m_optixVertexBuffer->unmap();
        }

        // TODO: 頂点情報更新時の処理。(IndexBufferとの整合性など)
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(std::vector<uint32_t> &&indices, SurfaceMaterial* material) {
        optix::Context optixContext = m_context.getOptiXContext();
        const OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        OptiXGeometry geom;
        CompensatedSum<float> sumImportances(0.0f);
        {
            geom.indices = std::move(indices);
            uint32_t numTriangles = (uint32_t)geom.indices.size() / 3;

            geom.optixGeometry = optixContext->createGeometry();
            geom.optixGeometry->setPrimitiveCount(numTriangles);
            geom.optixGeometry->setIntersectionProgram(progSet.programIntersectTriangle);
            geom.optixGeometry->setBoundingBoxProgram(progSet.programCalcBBoxForTriangle);

            geom.optixIndexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, numTriangles);
            geom.optixIndexBuffer->setElementSize(sizeof(Shared::Triangle));

            std::vector<float> areas;
            areas.resize(numTriangles);
            {
                auto dstTriangles = (Shared::Triangle*)geom.optixIndexBuffer->map();
                for (auto i = 0; i < numTriangles; ++i) {
                    uint32_t i0 = geom.indices[3 * i + 0];
                    uint32_t i1 = geom.indices[3 * i + 1];
                    uint32_t i2 = geom.indices[3 * i + 2];

                    dstTriangles[i] = Shared::Triangle{ i0, i1, i2 };

                    const Vertex (&v)[3] = { m_vertices[i0], m_vertices[i1], m_vertices[i2] };
                    areas[i] = std::max<float>(0.0f, 0.5f * cross(v[1].position - v[0].position, v[2].position - v[0].position).length());
                    sumImportances += areas[i];
                }
                geom.optixIndexBuffer->unmap();
            }

            if (material->isEmitting())
                geom.primDist.initialize(m_context, areas.data(), areas.size());
        }
        m_optixGeometries.push_back(geom);

        m_materials.push_back(material);

        Shared::SurfaceLightDescriptor lightDesc;
        lightDesc.body.vertexBuffer = m_optixVertexBuffer->getId();
        lightDesc.body.triangleBuffer = geom.optixIndexBuffer->getId();
        geom.primDist.getInternalType(&lightDesc.body.primDistribution);
        lightDesc.body.materialIndex = material->getMaterialIndex();
        lightDesc.sampleFunc = progSet.callableProgramSampleTriangleMesh->getId();
        lightDesc.importance = material->isEmitting() ? 1.0f : 0.0f; // TODO:

        SHGeometryInstance* geomInst = new SHGeometryInstance(m_context, lightDesc);
        {
            optix::GeometryInstance optixGeomInst = geomInst->getOptiXObject();
            optixGeomInst->setGeometry(geom.optixGeometry);
            optixGeomInst->setMaterialCount(1);
            optixGeomInst->setMaterial(0, material->getOptiXObject());
            optixGeomInst["VLR::pv_vertexBuffer"]->set(m_optixVertexBuffer);
            optixGeomInst["VLR::pv_triangleBuffer"]->set(geom.optixIndexBuffer);
            optixGeomInst["VLR::pv_sumImportances"]->setFloat(sumImportances.result);
            optixGeomInst["VLR::pv_progDecodeTexCoord"]->set(progSet.callableProgramDecodeTexCoordForTriangle);
            optixGeomInst["VLR::pv_progDecodeHitPoint"]->set(progSet.callableProgramDecodeHitPointForTriangle);
            optixGeomInst["VLR::pv_progFetchAlpha"]->set(m_context.getOptiXCallableProgramNullFetchAlpha());
            optixGeomInst["VLR::pv_progFetchNormal"]->set(m_context.getOptiXCallableProgramNullFetchNormal());
            optixGeomInst["VLR::pv_importance"]->setFloat(lightDesc.importance);
        }
        m_shGeometryInstances.push_back(geomInst);

        // JP: 親にジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        delta.insert(geomInst);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
        }
    }
    
    
    
    ParentNode::ParentNode(Context &context, const std::string &name, const Transform* localToWorld) :
        Node(context, name), m_localToWorld(localToWorld), m_shGeomGroup(context) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: 
        if (m_localToWorld->isStatic()) {
            StaticTransform* tr = (StaticTransform*)m_localToWorld;
            m_shTransforms[nullptr] = new SHTransform(name, m_context, *tr, nullptr);
        }
        else {
            VLRAssert_NotImplemented();
        }
    }

    ParentNode::~ParentNode() {
        for (auto it = m_shTransforms.crbegin(); it != m_shTransforms.crend(); ++it)
            delete it->second;
        m_shTransforms.clear();
    }

    void ParentNode::setName(const std::string &name) {
        Node::setName(name);
        for (auto it = m_shTransforms.begin(); it != m_shTransforms.end(); ++it)
            it->second->setName(name);
    }
    
    void ParentNode::addChild(InternalNode* child) {
        m_children.insert(child);
        child->addParent(this);
    }

    void ParentNode::addChild(SurfaceNode* child) {
        m_children.insert(child);
        child->addParent(this);
    }

    void ParentNode::removeChild(InternalNode* child) {
        m_children.erase(child);
        child->removeParent(this);
    }

    void ParentNode::removeChild(SurfaceNode* child) {
        m_children.erase(child);
        child->removeParent(this);
    }

    void ParentNode::setTransform(const Transform* localToWorld) {
        m_localToWorld = localToWorld;

        // JP: 管理中のSHTransformを更新する。
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                StaticTransform* tr = (StaticTransform*)m_localToWorld;
                SHTransform* shtr = it->second;
                shtr->setTransform(*tr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
    }



    void InternalNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) {
        switch (eventType) {
        case UpdateEvent::TransformAdded: {
            // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
            //     子のSHTransformをキーとして辞書に保存する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_localToWorld->isStatic()) {
                    StaticTransform* tr = (StaticTransform*)m_localToWorld;
                    SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                    m_shTransforms[*it] = shtr;
                    delta.insert(shtr);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 親に自分が保持するSHTransformが増えたことを通知(増分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            break;
        }
        case UpdateEvent::TransformRemoved: {
            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを削除。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shTransforms.erase(*it);
                delta.insert(shtr);
            }

            // JP: 親に自分が保持するSHTransformが減ったことを通知(減分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            for (auto it = delta.cbegin(); it != delta.cend(); ++it)
                delete *it;

            break;
        }
        case UpdateEvent::TransformUpdated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
                delta.insert(shtr);
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 親に自分が保持するSHTransformが更新されたことを通知(更新分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            break;
        }
        case UpdateEvent::GeometryAdded:
        case UpdateEvent::GeometryRemoved: {
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                delta.insert(shtr);
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: 親に自分が保持するSHTransformが更新されたことを通知(更新分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta, geomInstDelta);
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    void InternalNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) {
        switch (eventType) {
        case UpdateEvent::GeometryAdded: {
            // JP: このInternalNodeが管理するGeometryGroupにGeometryInstanceを追加する。
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                m_shGeomGroup.addGeometryInstance(*it);
                geomInstDelta.push_back(TransformAndGeometryInstance{ selfTransform, *it });
            }

            // JP: このInternalNodeのTransformにGeometryGroupをセットする。
            //     このInternalNodeのTransformを末尾に持つTransformに変更があったことを親に知らせる。
            if (m_shGeomGroup.getNumInstances() > 0) {
                selfTransform->setChild(&m_shGeomGroup);
                
                std::set<SHTransform*> delta;
                delta.insert(selfTransform);
                for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                    ParentNode* parent = *it;
                    parent->childUpdateEvent(eventType, delta, geomInstDelta);
                }
            }

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: 
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                m_shGeomGroup.removeGeometryInstance(*it);
                geomInstDelta.push_back(TransformAndGeometryInstance{ selfTransform, *it });
            }

            if (m_shGeomGroup.getNumInstances() == 0) {
                selfTransform->setChild(nullptr);

                std::set<SHTransform*> delta;
                delta.insert(selfTransform);
                for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                    ParentNode* parent = *it;
                    parent->childUpdateEvent(eventType, delta, geomInstDelta);
                }
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    InternalNode::InternalNode(Context &context, const std::string &name, const Transform* localToWorld) :
        ParentNode(context, name, localToWorld) {
    }

    void InternalNode::setTransform(const Transform* localToWorld) {
        ParentNode::setTransform(localToWorld);

        // JP: 親に変形情報が更新されたことを通知する。
        std::set<SHTransform*> delta;
        std::vector<TransformAndGeometryInstance> geomInstDelta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            delta.insert(it->second);

            SHGeometryGroup* shGeomGroup = nullptr;
            if (it->second->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumInstances(); ++i) {
                    geomInstDelta.push_back(TransformAndGeometryInstance{ it->second, shGeomGroup->getGeometryInstanceAt(i) });
                }
            }
        }
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(UpdateEvent::TransformUpdated, delta, geomInstDelta);
        }
    }

    void InternalNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);

        // JP: 追加した親に対して変形情報の追加を行わせる。
        std::set<SHTransform*> delta;
        std::vector<TransformAndGeometryInstance> geomInstDelta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            delta.insert(it->second);

            SHGeometryGroup* shGeomGroup = nullptr;
            if (it->second->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumInstances(); ++i) {
                    geomInstDelta.push_back(TransformAndGeometryInstance{ it->second, shGeomGroup->getGeometryInstanceAt(i) });
                }
            }
        }
        parent->childUpdateEvent(UpdateEvent::TransformAdded, delta, geomInstDelta);
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 削除した親に対して変形情報の削除を行わせる。
        std::set<SHTransform*> delta;
        std::vector<TransformAndGeometryInstance> geomInstDelta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            delta.insert(it->second);

            SHGeometryGroup* shGeomGroup = nullptr;
            if (it->second->hasGeometryDescendant(&shGeomGroup)) {
                for (int i = 0; i < shGeomGroup->getNumInstances(); ++i) {
                    geomInstDelta.push_back(TransformAndGeometryInstance{ it->second, shGeomGroup->getGeometryInstanceAt(i) });
                }
            }
        }
        parent->childUpdateEvent(UpdateEvent::TransformRemoved, delta, geomInstDelta);
    }



    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta, const std::vector<TransformAndGeometryInstance> &childGeomInstDelta) {
        switch (eventType) {
        case UpdateEvent::TransformAdded: {
            // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
            //     子のSHTransformをキーとして辞書に保存する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_localToWorld->isStatic()) {
                    StaticTransform* tr = (StaticTransform*)m_localToWorld;
                    SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                    m_shTransforms[*it] = shtr;
                    delta.insert(shtr);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    VLRDebugPrintf("Surface light cannot be instanced.");
                    VLRAssert_ShouldNotBeCalled();
                }
                else {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[it->geomInstance];
                    it->geomInstance->getSurfaceLightDescriptor(&lightDesc);
                    if (it->transform->isStatic()) {
                        StaticTransform tr = it->transform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
            }
            m_surfaceLightsAreSetup = false;

            // JP: SHGroupにもSHTransformを追加する。
            for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
                SHTransform* shtr = *it;
                m_shGroup.addChild(shtr);
            }

            break;
        }
        case UpdateEvent::TransformRemoved: {
            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    m_surfaceLights.erase(it->geomInstance);
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            // JP: 子InternalNodeが持つSHTransformがつながっているSHTransformを削除。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shTransforms.erase(*it);
                delta.insert(shtr);
            }

            // JP: SHGroupからもSHTransformを削除する。
            for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
                SHTransform* shtr = *it;
                m_shGroup.removeChild(shtr);
            }

            for (auto it = delta.cbegin(); it != delta.cend(); ++it)
                delete *it;

            break;
        }
        case UpdateEvent::TransformUpdated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights.at(it->geomInstance);
                    if (it->transform->isStatic()) {
                        StaticTransform tr = it->transform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        case UpdateEvent::GeometryAdded: {
            // JP: SHGroupに対してSHTransformの末尾のジオメトリ状態に変化があったことを通知する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shGroup.updateChild(shtr);
            }

            std::vector<TransformAndGeometryInstance> geomInstDelta;
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(it->transform);
                geomInstDelta.push_back(TransformAndGeometryInstance{ shtr, it->geomInstance });
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = geomInstDelta.cbegin(); it != geomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    VLRDebugPrintf("Surface light cannot be instanced.");
                    VLRAssert_ShouldNotBeCalled();
                }
                else {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[it->geomInstance];
                    it->geomInstance->getSurfaceLightDescriptor(&lightDesc);
                    if (it->transform->isStatic()) {
                        StaticTransform tr = it->transform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: SHGroupに対してSHTransformの末尾のジオメトリ状態に変化があったことを通知する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shGroup.updateChild(shtr);
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childGeomInstDelta.cbegin(); it != childGeomInstDelta.cend(); ++it) {
                if (m_surfaceLights.count(it->geomInstance)) {
                    m_surfaceLights.erase(it->geomInstance);
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) {
        switch (eventType) {
        case UpdateEvent::GeometryAdded: {
            // JP: 
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.addGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() > 0) {
                selfTransform->setChild(&m_shGeomGroup);
                m_shGroup.updateChild(selfTransform);
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_surfaceLights.count(*it)) {
                    VLRDebugPrintf("Surface light cannot be instanced.");
                    VLRAssert_ShouldNotBeCalled();
                }
                else {
                    Shared::SurfaceLightDescriptor &lightDesc = m_surfaceLights[*it];
                    (*it)->getSurfaceLightDescriptor(&lightDesc);
                    if (selfTransform->isStatic()) {
                        StaticTransform tr = selfTransform->getStaticTransform();
                        float mat[16], invMat[16];
                        tr.getArrays(mat, invMat);
                        lightDesc.body.transform = Shared::StaticTransform(Matrix4x4(mat));
                    }
                    else {
                        VLRAssert_NotImplemented();
                    }
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: 
            SHTransform* selfTransform = m_shTransforms.at(nullptr);
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.removeGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() == 0) {
                selfTransform->setChild(nullptr);
                m_shGroup.updateChild(selfTransform);
            }

            // JP: SurfaceLightDescriptorのマップを構築する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_surfaceLights.count(*it)) {
                    m_surfaceLights.erase(*it);
                }
                else {
                    VLRAssert_ShouldNotBeCalled();
                }
            }
            m_surfaceLightsAreSetup = false;

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    RootNode::RootNode(Context &context, const Transform* localToWorld) :
        ParentNode(context, "Root", localToWorld), m_shGroup(context), m_surfaceLightsAreSetup(false) {
        SHTransform* shtr = m_shTransforms[0];
        m_shGroup.addChild(shtr);
    }

    RootNode::~RootNode() {
        if (m_surfaceLightsAreSetup) {
            m_surfaceLightImpDist.finalize(m_context);

            m_optixSurfaceLightDescriptorBuffer->destroy();

            m_surfaceLightsAreSetup = false;
        }
    }

    void RootNode::set() {
        optix::Context optixContext = m_context.getOptiXContext();

        optixContext["VLR::pv_topGroup"]->set(m_shGroup.getOptiXObject());

        if (!m_surfaceLightsAreSetup) {
            if (m_optixSurfaceLightDescriptorBuffer)
                m_optixSurfaceLightDescriptorBuffer->destroy();
            m_optixSurfaceLightDescriptorBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_surfaceLights.size());
            m_optixSurfaceLightDescriptorBuffer->setElementSize(sizeof(Shared::SurfaceLightDescriptor));

            std::vector<float> importances;
            importances.resize(m_surfaceLights.size());

            {
                Shared::SurfaceLightDescriptor* descs = (Shared::SurfaceLightDescriptor*)m_optixSurfaceLightDescriptorBuffer->map();
                for (auto it = m_surfaceLights.cbegin(); it != m_surfaceLights.cend(); ++it) {
                    uint32_t index = std::distance(m_surfaceLights.cbegin(), it);
                    const Shared::SurfaceLightDescriptor &lightDesc = it->second;
                    descs[index] = lightDesc;
                    importances[index] = lightDesc.importance;
                }
                m_optixSurfaceLightDescriptorBuffer->unmap();
            }

            m_surfaceLightImpDist.finalize(m_context);
            m_surfaceLightImpDist.initialize(m_context, importances.data(), importances.size());

            m_surfaceLightsAreSetup = true;
        }

        Shared::DiscreteDistribution1D lightImpDist;
        m_surfaceLightImpDist.getInternalType(&lightImpDist);
        optixContext["VLR::pv_lightImpDist"]->setUserData(sizeof(lightImpDist), &lightImpDist);

        optixContext["VLR::pv_surfaceLightDescriptorBuffer"]->set(m_optixSurfaceLightDescriptorBuffer);
    }



    Scene::Scene(Context &context, const Transform* localToWorld) : 
    Object(context), m_rootNode(context, localToWorld) {

    }

    void Scene::set() {
        m_rootNode.set();
    }



    // static
    void Camera::initialize(Context &context) {
        PerspectiveCamera::initialize(context);
        EquirectangularCamera::initialize(context);
    }

    // static
    void Camera::finalize(Context &context) {
        EquirectangularCamera::finalize(context);
        PerspectiveCamera::finalize(context);
    }
    
    
    
    std::map<uint32_t, PerspectiveCamera::OptiXProgramSet> PerspectiveCamera::OptiXProgramSets;
    
    // static
    void PerspectiveCamera::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/cameras.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.callableProgramSampleLensPosition = optixContext->createProgramFromPTXString(ptx, "VLR::PerspectiveCamera_sampleLensPosition");
        programSet.callableProgramSampleIDF = optixContext->createProgramFromPTXString(ptx, "VLR::PerspectiveCamera_sampleIDF");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void PerspectiveCamera::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleIDF->destroy();
        programSet.callableProgramSampleLensPosition->destroy();

        OptiXProgramSets.erase(context.getID());
    }
    
    PerspectiveCamera::PerspectiveCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                                         float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) :
        Camera(context), m_data(sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist) {
        m_data.position = position;
        m_data.orientation = orientation;
    }

    void PerspectiveCamera::set() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_perspectiveCamera"]->setUserData(sizeof(Shared::PerspectiveCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }



    std::map<uint32_t, EquirectangularCamera::OptiXProgramSet> EquirectangularCamera::OptiXProgramSets;

    // static
    void EquirectangularCamera::initialize(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/cameras.ptx");

        OptiXProgramSet programSet;

        optix::Context optixContext = context.getOptiXContext();

        programSet.callableProgramSampleLensPosition = optixContext->createProgramFromPTXString(ptx, "VLR::EquirectangularCamera_sampleLensPosition");
        programSet.callableProgramSampleIDF = optixContext->createProgramFromPTXString(ptx, "VLR::EquirectangularCamera_sampleIDF");

        OptiXProgramSets[context.getID()] = programSet;
    }

    // static
    void EquirectangularCamera::finalize(Context &context) {
        OptiXProgramSet &programSet = OptiXProgramSets.at(context.getID());

        programSet.callableProgramSampleIDF->destroy();
        programSet.callableProgramSampleLensPosition->destroy();

        OptiXProgramSets.erase(context.getID());
    }

    EquirectangularCamera::EquirectangularCamera(Context &context, const Point3D &position, const Quaternion &orientation,
                                                 float sensitivity, float phiAngle, float thetaAngle) :
        Camera(context), m_data(sensitivity, phiAngle, thetaAngle) {
        m_data.position = position;
        m_data.orientation = orientation;
    }

    void EquirectangularCamera::set() const {
        optix::Context optixContext = m_context.getOptiXContext();
        OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        optixContext["VLR::pv_equirectangularCamera"]->setUserData(sizeof(Shared::EquirectangularCamera), &m_data);
        optixContext["VLR::pv_progSampleLensPosition"]->set(progSet.callableProgramSampleLensPosition);
        optixContext["VLR::pv_progSampleIDF"]->set(progSet.callableProgramSampleIDF);
    }
}
