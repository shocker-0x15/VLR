#include "optix_util_private.h"

namespace optixu {
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        va_end(args);
        OutputDebugString(str);
    }



    Context Context::create(CUcontext cudaContext) {
        return (new _Context(cudaContext))->getPublicType();
    }

    void Context::destroy() {
        delete m;
        m = nullptr;
    }



    Material Context::createMaterial() const {
        return (new _Material(m))->getPublicType();
    }
    
    Scene Context::createScene() const {
        return (new _Scene(m))->getPublicType();
    }

    Pipeline Context::createPipeline() const {
        return (new _Pipeline(m))->getPublicType();
    }



    void Material::Priv::setRecordData(const _Pipeline* pipeline, uint32_t rayType, HitGroupSBTRecord* record) const {
        Key key{ pipeline, rayType };
        const _ProgramGroup* hitGroup = programs.at(key);
        hitGroup->packHeader(record->header);
        record->data.materialData = userData;
    }
    
    void Material::destroy() {
        delete m;
        m = nullptr;
    }

    void Material::setHitGroup(uint32_t rayType, ProgramGroup hitGroup) {
        auto _pipeline = extract(hitGroup)->getPipeline();
        THROW_RUNTIME_ERROR(_pipeline, "Invalid pipeline: %p.", _pipeline);

        _Material::Key key{ _pipeline, rayType };
        m->programs[key] = extract(hitGroup);
    }
    
    void Material::setUserData(uint32_t data) const {
        m->userData = data;
    }



    
    void Scene::Priv::markSBTLayoutDirty() {
        sbtLayoutIsUpToDate = false;

        for (_InstanceAccelerationStructure* _ias : instASs)
            _ias->markDirty();
    }

    void Scene::Priv::setupHitGroupSBT(const _Pipeline* pipeline, Buffer* sbt) {
        THROW_RUNTIME_ERROR(sbt->sizeInBytes() >= sizeof(HitGroupSBTRecord) * numSBTRecords,
                            "Shader binding table size is not enough.");

        auto records = sbt->map<HitGroupSBTRecord>();

        for (_GeometryAccelerationStructure* gas : geomASs) {
            uint32_t numMatSets = gas->getNumMaterialSets();
            for (int j = 0; j < numMatSets; ++j) {
                uint32_t numRecords = gas->fillSBTRecords(pipeline, j, records);
                records += numRecords;
            }
        }

        sbt->unmap();
    }

    bool Scene::Priv::isReady() {
        for (_GeometryAccelerationStructure* _gas : geomASs) {
            if (!_gas->isReady()) {
                return false;
            }
        }

        for (_InstanceAccelerationStructure* _ias : instASs) {
            if (!_ias->isReady()) {
                return false;
            }
        }

        return true;
    }

    void Scene::destroy() {
        delete m;
        m = nullptr;
    }
    
    GeometryInstance Scene::createGeometryInstance() const {
        return (new _GeometryInstance(m))->getPublicType();
    }

    GeometryAccelerationStructure Scene::createGeometryAccelerationStructure() const {
        return (new _GeometryAccelerationStructure(m))->getPublicType();
    }

    Instance Scene::createInstance() const {
        return (new _Instance(m))->getPublicType();
    }

    InstanceAccelerationStructure Scene::createInstanceAccelerationStructure() const {
        return (new _InstanceAccelerationStructure(m))->getPublicType();
    }

    void Scene::generateShaderBindingTableLayout(size_t* memorySize) const {
        if (m->sbtLayoutIsUpToDate) {
            *memorySize = sizeof(HitGroupSBTRecord) * m->numSBTRecords;
            return;
        }

        uint32_t sbtOffset = 0;
        m->sbtOffsets.clear();
        for (_GeometryAccelerationStructure* gas : m->geomASs) {
            uint32_t numMatSets = gas->getNumMaterialSets();
            for (int matSetIdx = 0; matSetIdx < numMatSets; ++matSetIdx) {
                uint32_t gasNumSBTRecords = gas->calcNumSBTRecords(matSetIdx);
                _Scene::SBTOffsetKey key = { gas, matSetIdx };
                m->sbtOffsets[key] = sbtOffset;
                sbtOffset += gasNumSBTRecords;
            }
        }
        m->numSBTRecords = sbtOffset;
        m->sbtLayoutIsUpToDate = true;

        *memorySize = sizeof(HitGroupSBTRecord) * m->numSBTRecords;
    }



    void GeometryInstance::Priv::fillBuildInput(OptixBuildInput* input) const {
        *input = OptixBuildInput{};

        input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        OptixBuildInputTriangleArray &triArray = input->triangleArray;

        triArray.vertexBuffers = vertexBufferArray;
        triArray.numVertices = vertexBuffer->numElements();
        triArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triArray.vertexStrideInBytes = vertexBuffer->stride();

        triArray.indexBuffer = triangleBuffer->getCUdeviceptr();
        triArray.numIndexTriplets = triangleBuffer->numElements();
        triArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triArray.indexStrideInBytes = triangleBuffer->stride();
        triArray.primitiveIndexOffset = 0;

        triArray.numSbtRecords = buildInputFlags.size();
        if (triArray.numSbtRecords > 1) {
            optixAssert_NotImplemented();
            triArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer->getCUdeviceptr();
            triArray.sbtIndexOffsetSizeInBytes = 4;
            triArray.sbtIndexOffsetStrideInBytes = materialIndexOffsetBuffer->stride();
        }
        else {
            triArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
            triArray.sbtIndexOffsetSizeInBytes = 0; // No effect
            triArray.sbtIndexOffsetStrideInBytes = 0; // No effect
        }

        triArray.preTransform = 0;

        triArray.flags = buildInputFlags.data();
    }

    void GeometryInstance::Priv::updateBuildInput(OptixBuildInput* input) const {
        OptixBuildInputTriangleArray &triArray = input->triangleArray;

        triArray.vertexBuffers = vertexBufferArray;

        triArray.indexBuffer = triangleBuffer->getCUdeviceptr();

        if (triArray.numSbtRecords > 1) {
            optixAssert_NotImplemented();
            triArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer->getCUdeviceptr();
        }
    }

    uint32_t GeometryInstance::Priv::getNumSBTRecords() const {
        return static_cast<uint32_t>(buildInputFlags.size());
    }

    uint32_t GeometryInstance::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes,
                                                    HitGroupSBTRecord* records) const {
        THROW_RUNTIME_ERROR(matSetIdx < materialSets.size(),
                            "Out of material set bound: [0, %u)", static_cast<uint32_t>(materialSets.size()));

        const std::vector<const _Material*> &materialSet = materialSets[matSetIdx];
        HitGroupSBTRecord* recordPtr = records;
        uint32_t numMaterials = buildInputFlags.size();
        for (int matIdx = 0; matIdx < numMaterials; ++matIdx) {
            const _Material* mat = materialSet[matIdx];
            THROW_RUNTIME_ERROR(mat, "No material set for %u-%u.", matSetIdx, matIdx);
            for (int rIdx = 0; rIdx < numRayTypes; ++rIdx) {
                mat->setRecordData(pipeline, rIdx, recordPtr);
                recordPtr->data.geomInstData = userData;
                ++recordPtr;
            }
        }

        return numMaterials * numRayTypes;
    }

    void GeometryInstance::destroy() {
        delete m;
        m = nullptr;
    }

    void GeometryInstance::setVertexBuffer(Buffer* vertexBuffer) const {
        m->vertexBuffer = vertexBuffer;
        m->vertexBufferArray[0] = vertexBuffer->getCUdeviceptr();
    }

    void GeometryInstance::setTriangleBuffer(Buffer* triangleBuffer) const {
        m->triangleBuffer = triangleBuffer;
    }

    void GeometryInstance::setNumMaterials(uint32_t numMaterials, TypedBuffer<uint32_t>* matIdxOffsetBuffer) const {
        THROW_RUNTIME_ERROR(numMaterials > 0, "Invalid number of materials %u.", numMaterials);
        THROW_RUNTIME_ERROR((numMaterials == 1) != (matIdxOffsetBuffer != nullptr),
                            "Material index offset buffer must be provided when multiple materials are used.");
        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
        m->materialIndexOffsetBuffer = matIdxOffsetBuffer;
    }

    void GeometryInstance::setUserData(uint32_t data) const {
        m->userData = data;
    }

    void GeometryInstance::setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const {
        size_t numMaterials = m->buildInputFlags.size();
        THROW_RUNTIME_ERROR(matIdx < numMaterials,
                            "Out of material bounds [0, %u).", (uint32_t)numMaterials);

        m->buildInputFlags[matIdx] = flags;
    }

    void GeometryInstance::setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const {
        size_t numMaterials = m->buildInputFlags.size();
        THROW_RUNTIME_ERROR(matIdx < numMaterials,
                            "Out of material bounds [0, %u).", (uint32_t)numMaterials);

        uint32_t prevNumMatSets = m->materialSets.size();
        if (matSetIdx >= prevNumMatSets) {
            m->materialSets.resize(matSetIdx + 1);
            for (int i = prevNumMatSets; i < m->materialSets.size(); ++i)
                m->materialSets[i].resize(numMaterials, nullptr);
        }
        m->materialSets[matSetIdx][matIdx] = extract(mat);
    }



    uint32_t GeometryAccelerationStructure::Priv::calcNumSBTRecords(uint32_t matSetIdx) const {
        uint32_t numSBTRecords = 0;
        for (const _GeometryInstance* child : children)
            numSBTRecords += child->getNumSBTRecords();
        numSBTRecords *= numRayTypesPerMaterialSet[matSetIdx];

        return numSBTRecords;
    }

    uint32_t GeometryAccelerationStructure::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, HitGroupSBTRecord* records) const {
        THROW_RUNTIME_ERROR(matSetIdx < numRayTypesPerMaterialSet.size(),
                            "Material set index %u is out of bound [0, %u).",
                            matSetIdx, static_cast<uint32_t>(numRayTypesPerMaterialSet.size()));

        uint32_t numRayTypes = numRayTypesPerMaterialSet[matSetIdx];
        uint32_t sumRecords = 0;
        for (const _GeometryInstance* child : children) {
            uint32_t numRecords = child->fillSBTRecords(pipeline, matSetIdx, numRayTypes, records);
            records += numRecords;
            sumRecords += numRecords;
        }

        return sumRecords;
    }

    void GeometryAccelerationStructure::Priv::markDirty() {
        readyToBuild = false;
        available = false;
        readyToCompact = false;
        compactedAvailable = false;

        scene->markSBTLayoutDirty();
    }
    
    void GeometryAccelerationStructure::destroy() {
        delete m;
        m = nullptr;
    }

    void GeometryAccelerationStructure::setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) const {
        bool changed = false;
        changed |= m->preferFastTrace != preferFastTrace;
        m->preferFastTrace = preferFastTrace;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;

        if (changed)
            m->markDirty();
    }

    void GeometryAccelerationStructure::setNumMaterialSets(uint32_t numMatSets) const {
        m->numRayTypesPerMaterialSet.resize(numMatSets, 0);

        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const {
        THROW_RUNTIME_ERROR(matSetIdx < m->numRayTypesPerMaterialSet.size(),
                            "Material set index %u is out of bounds [0, %u).",
                            matSetIdx, static_cast<uint32_t>(m->numRayTypesPerMaterialSet.size()));
        m->numRayTypesPerMaterialSet[matSetIdx] = numRayTypes;

        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::addChild(GeometryInstance geomInst) const {
        auto _geomInst = extract(geomInst);
        THROW_RUNTIME_ERROR(_geomInst, "Invalid geometry instance %p.", _geomInst);
        THROW_RUNTIME_ERROR(_geomInst->getScene() == m->scene, "Scene mismatch for the given geometry instance.");
        THROW_RUNTIME_ERROR(m->children.count(_geomInst) == 0, "Geometry instance %p has been already added.", _geomInst);

        m->children.insert(_geomInst);

        m->markDirty();
    }

    void GeometryAccelerationStructure::removeChild(GeometryInstance geomInst) const {
        auto _geomInst = extract(geomInst);
        THROW_RUNTIME_ERROR(_geomInst, "Invalid geometry instance %p.", _geomInst);
        THROW_RUNTIME_ERROR(_geomInst->getScene() == m->scene, "Scene mismatch for the given geometry instance.");
        THROW_RUNTIME_ERROR(m->children.count(_geomInst) > 0, "Geometry instance %p has not been added.", _geomInst);

        m->children.erase(_geomInst);

        m->markDirty();
    }

    void GeometryAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const {
        m->buildInputs.resize(m->children.size(), OptixBuildInput{});
        uint32_t childIdx = 0;
        for (const _GeometryInstance* child : m->children)
            child->fillBuildInput(&m->buildInputs[childIdx++]);

        m->buildOptions = {};
        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        m->buildOptions.buildFlags = ((m->preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD) |
                                      (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                      (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
        //buildOptions.motionOptions

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 m->buildInputs.data(), m->buildInputs.size(),
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;

        m->readyToBuild = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::rebuild(CUstream stream, const Buffer &accelBuffer, const Buffer &scratchBuffer) const {
        THROW_RUNTIME_ERROR(m->readyToBuild, "You need to call prepareForBuild() before rebuild.");
        THROW_RUNTIME_ERROR(accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
                            "Size of the given buffer is not enough.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        // JP: アップデートの意味でリビルドするときはprepareForBuild()を呼ばないため
        //     ビルド入力を更新する処理をここにも書いておく必要がある。
        // EN: User is not required to call prepareForBuild() when performing rebuild
        //     for purpose of update so updating build inputs should be here.
        uint32_t childIdx = 0;
        for (const _GeometryInstance* child : m->children)
            child->updateBuildInput(&m->buildInputs[childIdx++]);

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr,
                                    compactionEnabled ? 1 : 0));

        m->accelBuffer = &accelBuffer;
        m->available = true;
        m->readyToCompact = false;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void GeometryAccelerationStructure::prepareForCompact(CUstream rebuildOrUpdateStream, size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        CUDADRV_CHECK(cuStreamSynchronize(rebuildOrUpdateStream));
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::compact(CUstream stream, const Buffer &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->readyToCompact, "You need to call prepareForCompact() before compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");
        THROW_RUNTIME_ERROR(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
                                      m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                                      &m->compactedHandle));

        m->compactedAccelBuffer = &compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void GeometryAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuStreamSynchronize(compactionStream));

        m->handle = 0;
        m->available = false;
    }

    OptixTraversableHandle GeometryAccelerationStructure::update(CUstream stream, const Buffer &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        THROW_RUNTIME_ERROR(updateEnabled, "This AS does not allow update.");
        THROW_RUNTIME_ERROR(m->available || m->compactedAvailable, "AS has not been built yet.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const _GeometryInstance* child : m->children)
            child->updateBuildInput(&m->buildInputs[childIdx++]);

        const Buffer* accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle &handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer->getCUdeviceptr(), accelBuffer->sizeInBytes(),
                                    &handle,
                                    nullptr, 0));

        return handle;
    }

    bool GeometryAccelerationStructure::isReady() const {
        return m->isReady();
    }



    void Instance::Priv::fillInstance(OptixInstance* instance) const {
        if (type == InstanceType::GAS) {
            THROW_RUNTIME_ERROR(gas->isReady(), "GAS %p is not ready.", gas);

            *instance = {};
            instance->instanceId = 0;
            instance->visibilityMask = 0xFF;
            std::copy_n(transform, 12, instance->transform);
            instance->flags = OPTIX_INSTANCE_FLAG_NONE;
            instance->traversableHandle = gas->getHandle();
            instance->sbtOffset = scene->getSBTOffset(gas, matSetIndex);
        }
        else {
            optixAssert_NotImplemented();
        }
    }

    void Instance::Priv::updateInstance(OptixInstance* instance) const {
        instance->instanceId = 0;
        instance->visibilityMask = 0xFF;
        std::copy_n(transform, 12, instance->transform);
        //instance->flags = OPTIX_INSTANCE_FLAG_NONE; これは変えられない？
        //instance->sbtOffset = scene->getSBTOffset(gas, matSetIndex);
    }

    void Instance::destroy() {
        delete m;
        m = nullptr;
    }

    void Instance::setGAS(GeometryAccelerationStructure gas, uint32_t matSetIdx) const {
        m->type = InstanceType::GAS;
        m->gas = extract(gas);
        m->matSetIndex = matSetIdx;
    }

    void Instance::setTransform(const float transform[12]) const {
        std::copy_n(transform, 12, m->transform);
    }



    void InstanceAccelerationStructure::Priv::markDirty() {
        readyToBuild = false;
        available = false;
        readyToCompact = false;
        compactedAvailable = false;
    }
    
    void InstanceAccelerationStructure::destroy() {
        delete m;
        m = nullptr;
    }

    void InstanceAccelerationStructure::setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) const {
        bool changed = false;
        changed |= m->preferFastTrace != preferFastTrace;
        m->preferFastTrace = preferFastTrace;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;

        if (changed)
            m->markDirty();
    }

    void InstanceAccelerationStructure::addChild(Instance instance) const {
        _Instance* _inst = extract(instance);
        THROW_RUNTIME_ERROR(_inst, "Invalid instance %p.", _inst);
        THROW_RUNTIME_ERROR(_inst->getScene() == m->scene, "Scene mismatch for the given instance.");
        THROW_RUNTIME_ERROR(m->children.count(_inst) == 0, "Instance %p has been already added.", _inst);

        m->children.insert(_inst);

        m->markDirty();
    }

    void InstanceAccelerationStructure::removeChild(Instance instance) const {
        _Instance* _inst = extract(instance);
        THROW_RUNTIME_ERROR(_inst, "Invalid instance %p.", _inst);
        THROW_RUNTIME_ERROR(_inst->getScene() == m->scene, "Scene mismatch for the given instance.");
        THROW_RUNTIME_ERROR(m->children.count(_inst) > 0, "Instance %p has not been added.", _inst);

        m->children.erase(_inst);

        m->markDirty();
    }

    void InstanceAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement, uint32_t* numInstances) const {
        m->instances.resize(m->children.size());
        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->fillInstance(&m->instances[childIdx++]);

        // Fill the build input.
        {
            m->buildInput = OptixBuildInput{};
            m->buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            OptixBuildInputInstanceArray &instArray = m->buildInput.instanceArray;
            instArray.instances = 0;
            instArray.numInstances = static_cast<uint32_t>(m->children.size());
        }

        m->buildOptions = {};
        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        m->buildOptions.buildFlags = ((m->preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD) |
                                      (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                      (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
        //buildOptions.motionOptions

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 &m->buildInput, 1,
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;
        *numInstances = m->instances.size();

        m->readyToBuild = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::rebuild(CUstream stream, const TypedBuffer<OptixInstance> &instanceBuffer,
                                                                  const Buffer &accelBuffer, const Buffer &scratchBuffer) const {
        THROW_RUNTIME_ERROR(m->readyToBuild, "You need to call prepareForBuild() before rebuild.");
        THROW_RUNTIME_ERROR(accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
                            "Size of the given buffer is not enough.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
                            "Size of the given scratch buffer is not enough.");
        THROW_RUNTIME_ERROR(instanceBuffer.numElements() >= m->instances.size(),
                            "Size of the given instance buffer is not enough.");
        THROW_RUNTIME_ERROR(m->scene->sbtLayoutGenerationDone(),
                            "Shader binding table layout generation has not been done.");

        // JP: アップデートの意味でリビルドするときはprepareForBuild()を呼ばないため
        //     インスタンス情報を更新する処理をここにも書いておく必要がある。
        // EN: User is not required to call prepareForBuild() when performing rebuild
        //     for purpose of update so updating instance information should be here.
        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->updateInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(instanceBuffer.getCUdeviceptr(), m->instances.data(),
                                        instanceBuffer.sizeInBytes(),
                                        stream));
        m->buildInput.instanceArray.instances = instanceBuffer.getCUdeviceptr();

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream, &m->buildOptions, &m->buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr,
                                    compactionEnabled ? 1 : 0));

        m->instanceBuffer = &instanceBuffer;
        m->accelBuffer = &accelBuffer;
        m->available = true;
        m->readyToCompact = false;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void InstanceAccelerationStructure::prepareForCompact(CUstream rebuildOrUpdateStream, size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        CUDADRV_CHECK(cuStreamSynchronize(rebuildOrUpdateStream));
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::compact(CUstream stream, const Buffer &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->readyToCompact, "You need to call prepareForCompact() before compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");
        THROW_RUNTIME_ERROR(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
                                      m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                                      &m->compactedHandle));

        m->compactedAccelBuffer = &compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void InstanceAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuStreamSynchronize(compactionStream));

        m->handle = 0;
        m->available = false;
    }

    OptixTraversableHandle InstanceAccelerationStructure::update(CUstream stream, const Buffer &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        THROW_RUNTIME_ERROR(updateEnabled, "This AS does not allow update.");
        THROW_RUNTIME_ERROR(m->available || m->compactedAvailable, "AS has not been built yet.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->updateInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(m->instanceBuffer->getCUdeviceptr(), m->instances.data(),
                                        m->instanceBuffer->sizeInBytes(),
                                        stream));

        const Buffer* accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle &handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, &m->buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer->getCUdeviceptr(), accelBuffer->sizeInBytes(),
                                    &handle,
                                    nullptr, 0));

        return handle;
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m->isReady();
    }



    void Pipeline::Priv::createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group) {
        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context->getRawContext(),
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                group));
        programGroups.insert(*group);
    }

    void Pipeline::Priv::destroyProgram(OptixProgramGroup group) {
        optixAssert(programGroups.count(group) > 0, "This program group has not been registered.");
        programGroups.erase(group);
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }
    
    void Pipeline::Priv::setupShaderBindingTable() {
        if (!sbtAllocDone) {
            missRecords.resize(numMissRayTypes, missRecords.stride());
            callableRecords.resize(callablePrograms.size(), callableRecords.stride());

            sbtAllocDone = true;
            sbtIsUpToDate = false;
        }

        if (!sbtIsUpToDate) {
            THROW_RUNTIME_ERROR(rayGenProgram, "Ray generation program is not set.");

            for (int i = 0; i < numMissRayTypes; ++i)
                THROW_RUNTIME_ERROR(missPrograms[i], "Miss program is not set for ray type %d.", i);

            sbt = {};
            {
                auto rayGenRecordOnHost = rayGenRecord.map<uint8_t>();
                rayGenProgram->packHeader(rayGenRecordOnHost);
                rayGenRecord.unmap();

                if (exceptionProgram) {
                    auto exceptionRecordOnHost = exceptionRecord.map<uint8_t>();
                    exceptionProgram->packHeader(exceptionRecordOnHost);
                    exceptionRecord.unmap();
                }

                auto missRecordsOnHost = missRecords.map<uint8_t>();
                for (int i = 0; i < numMissRayTypes; ++i)
                    missPrograms[i]->packHeader(missRecordsOnHost + OPTIX_SBT_RECORD_HEADER_SIZE * i);
                missRecords.unmap();

                scene->setupHitGroupSBT(this, hitGroupSbt);

                auto callableRecordsOnHost = callableRecords.map<uint8_t>();
                for (int i = 0; i < callablePrograms.size(); ++i)
                    callablePrograms[i]->packHeader(callableRecordsOnHost + OPTIX_SBT_RECORD_HEADER_SIZE * i);
                callableRecords.unmap();



                sbt.raygenRecord = rayGenRecord.getCUdeviceptr();

                sbt.exceptionRecord = exceptionProgram ? exceptionRecord.getCUdeviceptr() : 0;

                sbt.missRecordBase = missRecords.getCUdeviceptr();
                sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
                sbt.missRecordCount = numMissRayTypes;

                sbt.hitgroupRecordBase = hitGroupSbt->getCUdeviceptr();
                sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSBTRecord);
                sbt.hitgroupRecordCount = hitGroupSbt->sizeInBytes() / sizeof(HitGroupSBTRecord);

                sbt.callablesRecordBase = callableRecords.getCUdeviceptr();
                sbt.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
                sbt.callablesRecordCount = callableRecords.numElements();
            }

            sbtIsUpToDate = true;
        }
    }

    void Pipeline::destroy() {
        delete m;
        m = nullptr;
    }
       
    void Pipeline::setMaxTraceDepth(uint32_t maxTraceDepth) const {
        m->maxTraceDepth = maxTraceDepth;
    }

    void Pipeline::setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues, const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                      bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags) const {
        // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
        // EN: Set pipeline compile options common among modules in the pipeline and the pipeline itself.
        m->pipelineCompileOptions = {};
        m->pipelineCompileOptions.numPayloadValues = numPayloadValues;
        m->pipelineCompileOptions.numAttributeValues = numAttributeValues;
        m->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m->pipelineCompileOptions.exceptionFlags = exceptionFlags;

        m->sizeOfPipelineLaunchParams = sizeOfLaunchParams;
    }



    Module Pipeline::createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount, OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) const {
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = maxRegisterCount;
        moduleCompileOptions.optLevel = optLevel;
        moduleCompileOptions.debugLevel = debugLevel;

        OptixModule rawModule;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m->context->getRawContext(),
                                                 &moduleCompileOptions,
                                                 &m->pipelineCompileOptions,
                                                 ptxString.c_str(), ptxString.size(),
                                                 log, &logSize,
                                                 &rawModule));

        return (new _Module(m, rawModule))->getPublicType();
    }



    ProgramGroup Pipeline::createRayGenProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        THROW_RUNTIME_ERROR((_module != nullptr) == (entryFunctionName != nullptr),
                            "Either of Miss module or entry function name is not provided.");
        if (_module)
            THROW_RUNTIME_ERROR(_module->getPipeline() == m,
                                "Pipeline mismatch for the given module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = _module->getRawModule();
        desc.raygen.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createExceptionProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        THROW_RUNTIME_ERROR((_module != nullptr) == (entryFunctionName != nullptr),
                            "Either of Miss module or entry function name is not provided.");
        if (_module)
            THROW_RUNTIME_ERROR(_module->getPipeline() == m,
                                "Pipeline mismatch for the given module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        desc.exception.module = _module->getRawModule();
        desc.exception.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createMissProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
        THROW_RUNTIME_ERROR((_module != nullptr) == (entryFunctionName != nullptr),
                            "Either of Miss module or entry function name is not provided.");
        if (_module)
            THROW_RUNTIME_ERROR(_module->getPipeline() == m,
                                "Pipeline mismatch for the given module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (_module)
            desc.miss.module = _module->getRawModule();
        desc.miss.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createHitProgramGroup(Module module_CH, const char* entryFunctionNameCH,
                                                 Module module_AH, const char* entryFunctionNameAH,
                                                 Module module_IS, const char* entryFunctionNameIS) const {
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        _Module* _module_IS = extract(module_IS);
        THROW_RUNTIME_ERROR((_module_CH != nullptr) == (entryFunctionNameCH != nullptr),
                            "Either of CH module or entry function name is not provided.");
        THROW_RUNTIME_ERROR((_module_AH != nullptr) == (entryFunctionNameAH != nullptr),
                            "Either of AH module or entry function name is not provided.");
        THROW_RUNTIME_ERROR((_module_IS != nullptr) == (entryFunctionNameIS != nullptr),
                            "Either of IS module or entry function name is not provided.");
        THROW_RUNTIME_ERROR(entryFunctionNameCH || entryFunctionNameAH || entryFunctionNameIS,
                            "Either of CH/AH/IS entry function name must be provided.");
        if (_module_CH)
            THROW_RUNTIME_ERROR(_module_CH->getPipeline() == m,
                                "Pipeline mismatch for the given CH module.");
        if (_module_AH)
            THROW_RUNTIME_ERROR(_module_AH->getPipeline() == m,
                                "Pipeline mismatch for the given AH module.");
        if (_module_IS)
            THROW_RUNTIME_ERROR(_module_IS->getPipeline() == m,
                                "Pipeline mismatch for the given IS module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH && _module_CH) {
            desc.hitgroup.moduleCH = _module_CH->getRawModule();
            desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        }
        if (entryFunctionNameAH && _module_AH) {
            desc.hitgroup.moduleAH = _module_AH->getRawModule();
            desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        }
        if (entryFunctionNameIS && _module_IS) {
            desc.hitgroup.moduleIS = _module_IS->getRawModule();
            desc.hitgroup.entryFunctionNameIS = entryFunctionNameIS;
        }

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createCallableGroup(Module module_DC, const char* entryFunctionNameDC,
                                               Module module_CC, const char* entryFunctionNameCC) const {
        _Module* _module_DC = extract(module_DC);
        _Module* _module_CC = extract(module_CC);
        THROW_RUNTIME_ERROR((_module_DC != nullptr) == (entryFunctionNameDC != nullptr),
                            "Either of DC module or entry function name is not provided.");
        THROW_RUNTIME_ERROR((_module_CC != nullptr) == (entryFunctionNameCC != nullptr),
                            "Either of CC module or entry function name is not provided.");
        THROW_RUNTIME_ERROR(entryFunctionNameDC || entryFunctionNameCC,
                            "Either of CC/DC entry function name must be provided.");
        if (_module_DC)
            THROW_RUNTIME_ERROR(_module_DC->getPipeline() == m,
                                "Pipeline mismatch for the given DC module.");
        if (_module_CC)
            THROW_RUNTIME_ERROR(_module_CC->getPipeline() == m,
                                "Pipeline mismatch for the given CC module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        if (entryFunctionNameDC && _module_DC) {
            desc.callables.moduleDC = _module_DC->getRawModule();
            desc.callables.entryFunctionNameDC = entryFunctionNameDC;
        }
        if (entryFunctionNameCC && _module_CC) {
            desc.callables.moduleCC = _module_CC->getRawModule();
            desc.callables.entryFunctionNameCC = entryFunctionNameCC;
        }

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }



    void Pipeline::link(OptixCompileDebugLevel debugLevel, bool overrideUseMotionBlur) const {
        THROW_RUNTIME_ERROR(!m->pipelineLinked, "This pipeline has been already linked.");

        if (!m->pipelineLinked) {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = m->maxTraceDepth;
            pipelineLinkOptions.debugLevel = debugLevel;
            pipelineLinkOptions.overrideUsesMotionBlur = overrideUseMotionBlur;

            std::vector<OptixProgramGroup> groups;
            groups.resize(m->programGroups.size());
            std::copy(m->programGroups.cbegin(), m->programGroups.cend(), groups.begin());

            char log[4096];
            size_t logSize = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(m->context->getRawContext(),
                                                &m->pipelineCompileOptions,
                                                &pipelineLinkOptions,
                                                groups.data(), static_cast<uint32_t>(groups.size()),
                                                log, &logSize,
                                                &m->rawPipeline));

            m->pipelineLinked = true;
        }
    }



    void Pipeline::setNumMissRayTypes(uint32_t numMissRayTypes) const {
        m->numMissRayTypes = numMissRayTypes;
        m->missPrograms.resize(m->numMissRayTypes);
        m->sbtAllocDone = false;
    }
    
    void Pipeline::setRayGenerationProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->rayGenProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setExceptionProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->exceptionProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setMissProgram(uint32_t rayType, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(rayType < m->numMissRayTypes, "Invalid ray type.");
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->missPrograms[rayType] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setCallableProgram(uint32_t index, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        if (index >= m->callablePrograms.size()) {
            m->callablePrograms.resize(index + 1);
            m->sbtIsUpToDate = false;
        }
        m->callablePrograms[index] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setScene(const Scene &scene) const {
        m->scene = extract(scene);
    }

    void Pipeline::setHitGroupShaderBindingTable(Buffer* shaderBindingTable) const {
        m->hitGroupSbt = shaderBindingTable;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::markHitGroupShaderBindingTableDirty() const {
        m->sbtIsUpToDate = false;
    }

    void Pipeline::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const {
        THROW_RUNTIME_ERROR(m->scene, "Scene is not set.");
        THROW_RUNTIME_ERROR(m->scene->isReady(), "Scene is not ready.");
        THROW_RUNTIME_ERROR(m->hitGroupSbt, "Hitgroup shader binding table is not set.");

        m->setupShaderBindingTable();

        OPTIX_CHECK(optixLaunch(m->rawPipeline, stream, plpOnDevice, m->sizeOfPipelineLaunchParams,
                                &m->sbt, dimX, dimY, dimZ));
    }

    void Pipeline::setStackSize(uint32_t directCallableStackSizeFromTraversal,
                                uint32_t directCallableStackSizeFromState,
                                uint32_t continuationStackSize) const {
        OPTIX_CHECK(optixPipelineSetStackSize(m->rawPipeline,
                                              directCallableStackSizeFromTraversal,
                                              directCallableStackSizeFromState,
                                              continuationStackSize,
                                              m->maxTraceDepth));
    }



    void Module::destroy() {
        OPTIX_CHECK(optixModuleDestroy(m->rawModule));

        delete m;
        m = nullptr;
    }



    void ProgramGroup::destroy() {
        m->pipeline->destroyProgram(m->rawGroup);

        delete m;
        m = nullptr;
    }

    void ProgramGroup::getStackSize(OptixStackSizes* sizes) const {
        OPTIX_CHECK(optixProgramGroupGetStackSize(m->rawGroup, sizes));
    }
}
