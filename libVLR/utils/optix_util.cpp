/*

   Copyright 2020 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

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

    Denoiser Context::createDenoiser(OptixDenoiserInputKind inputKind) const {
        return (new _Denoiser(m, inputKind))->getPublicType();
    }

    CUcontext Context::getCUcontext() const {
        return m->cudaContext;
    }



    void Material::Priv::setRecordData(const _Pipeline* pipeline, uint32_t rayType, uint8_t* record, SizeAlign* curSizeAlign) const {
        Key key{ pipeline, rayType };
        THROW_RUNTIME_ERROR(programs.count(key), "Pipeline: %p: No hit group is set to ray type: %u", pipeline, rayType);
        const _ProgramGroup* hitGroup = programs.at(key);
        *curSizeAlign = SizeAlign(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
        hitGroup->packHeader(record);
        uint32_t offset;
        curSizeAlign->add(userDataSizeAlign, &offset);
        std::memcpy(record + offset, userData.data(), userDataSizeAlign.size);
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
    
    void Material::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        THROW_RUNTIME_ERROR(size <= s_maxMaterialUserDataSize,
                            "Maximum user data size for Material is %u bytes.", s_maxMaterialUserDataSize);
        THROW_RUNTIME_ERROR(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                            "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }



    
    void Scene::Priv::markSBTLayoutDirty() {
        sbtLayoutIsUpToDate = false;

        for (_InstanceAccelerationStructure* _ias : instASs)
            _ias->markDirty();
    }

    void Scene::Priv::setupHitGroupSBT(CUstream stream, const _Pipeline* pipeline, const BufferView &sbt, void* hostMem) {
        THROW_RUNTIME_ERROR(sbt.sizeInBytes() >= singleRecordSize * numSBTRecords,
                            "Hit group shader binding table size is not enough.");

        auto records = reinterpret_cast<uint8_t*>(hostMem);

        for (_GeometryAccelerationStructure* gas : geomASs) {
            uint32_t numMatSets = gas->getNumMaterialSets();
            for (int matSetIdx = 0; matSetIdx < numMatSets; ++matSetIdx) {
                uint32_t numRecords = gas->fillSBTRecords(pipeline, matSetIdx, records);
                records += numRecords * singleRecordSize;
            }
        }

        CUDADRV_CHECK(cuMemcpyHtoDAsync(sbt.getCUdeviceptr(), hostMem, sbt.sizeInBytes(), stream));
    }

    bool Scene::Priv::isReady(bool* hasMotionAS) {
        *hasMotionAS = false;
        for (_GeometryAccelerationStructure* _gas : geomASs) {
            *hasMotionAS |= _gas->hasMotion();
            if (!_gas->isReady())
                return false;
        }

        for (_Transform* _tr : transforms) {
            if (!_tr->isReady())
                return false;
        }

        for (_InstanceAccelerationStructure* _ias : instASs) {
            *hasMotionAS |= _ias->hasMotion();
            if (!_ias->isReady())
                return false;
        }

        if (!sbtLayoutIsUpToDate)
            return false;

        return true;
    }

    void Scene::destroy() {
        delete m;
        m = nullptr;
    }
    
    GeometryInstance Scene::createGeometryInstance(bool forCustomPrimitives) const {
        return (new _GeometryInstance(m, forCustomPrimitives))->getPublicType();
    }

    GeometryAccelerationStructure Scene::createGeometryAccelerationStructure(bool forCustomPrimitives) const {
        return (new _GeometryAccelerationStructure(m, forCustomPrimitives))->getPublicType();
    }

    Transform Scene::createTransform() const {
        return (new _Transform(m))->getPublicType();
    }

    Instance Scene::createInstance() const {
        return (new _Instance(m))->getPublicType();
    }

    InstanceAccelerationStructure Scene::createInstanceAccelerationStructure() const {
        return (new _InstanceAccelerationStructure(m))->getPublicType();
    }

    void Scene::generateShaderBindingTableLayout(size_t* memorySize) const {
        if (m->sbtLayoutIsUpToDate) {
            *memorySize = m->singleRecordSize * std::max(m->numSBTRecords, 1u);
            return;
        }

        uint32_t sbtOffset = 0;
        m->sbtOffsets.clear();
        m->singleRecordSize = OPTIX_SBT_RECORD_HEADER_SIZE;
        for (_GeometryAccelerationStructure* gas : m->geomASs) {
            uint32_t numMatSets = gas->getNumMaterialSets();
            SizeAlign maxRecordSizeAlign;
            for (int matSetIdx = 0; matSetIdx < numMatSets; ++matSetIdx)
                maxRecordSizeAlign = max(maxRecordSizeAlign, gas->calcMaxRecordSizeAlign(matSetIdx));
            maxRecordSizeAlign.alignUp();
            m->singleRecordSize = std::max(m->singleRecordSize, maxRecordSizeAlign.size);
        }
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

        *memorySize = m->singleRecordSize * std::max(m->numSBTRecords, 1u);
    }



    void GeometryInstance::Priv::fillBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const {
        *input = OptixBuildInput{};

        if (forCustomPrimitives) {
            THROW_RUNTIME_ERROR(primitiveAABBBuffer.isValid(), "Custom Primitive AABB buffer is not set.");

            input->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            OptixBuildInputCustomPrimitiveArray &customPrimArray = input->customPrimitiveArray;

            primitiveAabbBufferArray[0] = primitiveAABBBuffer.getCUdeviceptr();

            customPrimArray.aabbBuffers = primitiveAabbBufferArray;
            customPrimArray.numPrimitives = primitiveAABBBuffer.numElements();
            customPrimArray.strideInBytes = primitiveAABBBuffer.stride();
            customPrimArray.primitiveIndexOffset = primitiveIndexOffset;

            customPrimArray.numSbtRecords = buildInputFlags.size();
            if (customPrimArray.numSbtRecords > 1) {
                customPrimArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer.getCUdeviceptr();
                customPrimArray.sbtIndexOffsetSizeInBytes = materialIndexOffsetSize;
                customPrimArray.sbtIndexOffsetStrideInBytes = materialIndexOffsetBuffer.stride();
            }
            else {
                customPrimArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
                customPrimArray.sbtIndexOffsetSizeInBytes = 0; // No effect
                customPrimArray.sbtIndexOffsetStrideInBytes = 0; // No effect
            }

            customPrimArray.flags = buildInputFlags.data();
        }
        else {
            THROW_RUNTIME_ERROR(vertexBuffer.isValid(), "Vertex buffer is not set.");
            THROW_RUNTIME_ERROR((indexFormat != OPTIX_INDICES_FORMAT_NONE) == triangleBuffer.isValid(),
                                "Triangle buffer must be provided if using a index format other than None, otherwise must not be provided.");

            input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            vertexBufferArray[0] = vertexBuffer.getCUdeviceptr();

            triArray.vertexBuffers = vertexBufferArray;
            triArray.numVertices = vertexBuffer.numElements();
            triArray.vertexFormat = vertexFormat;
            triArray.vertexStrideInBytes = vertexBuffer.stride();

            if (indexFormat != OPTIX_INDICES_FORMAT_NONE) {
                triArray.indexBuffer = triangleBuffer.getCUdeviceptr();
                triArray.indexStrideInBytes = triangleBuffer.stride();
                triArray.numIndexTriplets = triangleBuffer.numElements();
            }
            else {
                triArray.indexBuffer = 0;
                triArray.indexStrideInBytes = 0;
                triArray.numIndexTriplets = 0;
            }
            triArray.indexFormat = indexFormat;
            triArray.primitiveIndexOffset = primitiveIndexOffset;

            triArray.numSbtRecords = buildInputFlags.size();
            if (triArray.numSbtRecords > 1) {
                triArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer.getCUdeviceptr();
                triArray.sbtIndexOffsetSizeInBytes = materialIndexOffsetSize;
                triArray.sbtIndexOffsetStrideInBytes = materialIndexOffsetBuffer.stride();
            }
            else {
                triArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
                triArray.sbtIndexOffsetSizeInBytes = 0; // No effect
                triArray.sbtIndexOffsetStrideInBytes = 0; // No effect
            }

            triArray.preTransform = preTransform;
            triArray.transformFormat = preTransform ? OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 : OPTIX_TRANSFORM_FORMAT_NONE;

            triArray.flags = buildInputFlags.data();
        }
    }

    void GeometryInstance::Priv::updateBuildInput(OptixBuildInput* input, CUdeviceptr preTransform) const {
        if (forCustomPrimitives) {
            OptixBuildInputCustomPrimitiveArray &customPrimArray = input->customPrimitiveArray;

            primitiveAabbBufferArray[0] = primitiveAABBBuffer.getCUdeviceptr();
            customPrimArray.aabbBuffers = primitiveAabbBufferArray;

            if (customPrimArray.numSbtRecords > 1)
                customPrimArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer.getCUdeviceptr();
        }
        else {
            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            vertexBufferArray[0] = vertexBuffer.getCUdeviceptr();
            triArray.vertexBuffers = vertexBufferArray;

            if (indexFormat != OPTIX_INDICES_FORMAT_NONE)
                triArray.indexBuffer = triangleBuffer.getCUdeviceptr();

            if (triArray.numSbtRecords > 1)
                triArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer.getCUdeviceptr();

            triArray.preTransform = preTransform;
            triArray.transformFormat = preTransform ? OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 : OPTIX_TRANSFORM_FORMAT_NONE;
        }
    }

    SizeAlign GeometryInstance::Priv::calcMaxRecordSizeAlign(uint32_t gasMatSetIdx) const {
        SizeAlign maxRecordSizeAlign;
        for (int matIdx = 0; matIdx < materials.size(); ++matIdx) {
            THROW_RUNTIME_ERROR(materials[matIdx][0], "Default material (== material set 0) is not set for material %u.", matIdx);
            uint32_t matSetIdx = gasMatSetIdx < materials[matIdx].size() ? gasMatSetIdx : 0;
            const _Material* mat = materials[matIdx][matSetIdx];
            if (!mat)
                mat = materials[matIdx][0];
            SizeAlign recordSizeAlign(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
            recordSizeAlign += mat->getUserDataSizeAlign();
            maxRecordSizeAlign = max(maxRecordSizeAlign, recordSizeAlign);
        }
        maxRecordSizeAlign += userDataSizeAlign;
        return maxRecordSizeAlign;
    }

    uint32_t GeometryInstance::Priv::getNumSBTRecords() const {
        return static_cast<uint32_t>(buildInputFlags.size());
    }

    uint32_t GeometryInstance::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t gasMatSetIdx,
                                                    const void* gasUserData, const SizeAlign gasUserDataSizeAlign,
                                                    uint32_t numRayTypes, uint8_t* records) const {
        uint32_t numMaterials = materials.size();
        for (int matIdx = 0; matIdx < numMaterials; ++matIdx) {
            THROW_RUNTIME_ERROR(materials[matIdx][0], "Default material (== material set 0) is not set for material %u.", matIdx);
            uint32_t matSetIdx = gasMatSetIdx < materials[matIdx].size() ? gasMatSetIdx : 0;
            const _Material* mat = materials[matIdx][matSetIdx];
            if (!mat)
                mat = materials[matIdx][0];
            for (int rIdx = 0; rIdx < numRayTypes; ++rIdx) {
                SizeAlign curSizeAlign;
                mat->setRecordData(pipeline, rIdx, records, &curSizeAlign);
                uint32_t offset;
                curSizeAlign.add(userDataSizeAlign, &offset);
                std::memcpy(records + offset, userData.data(), userDataSizeAlign.size);
                curSizeAlign.add(gasUserDataSizeAlign, &offset);
                std::memcpy(records + offset, gasUserData, gasUserDataSizeAlign.size);
                records += scene->getSingleRecordSize();
            }
        }

        return numMaterials * numRayTypes;
    }

    void GeometryInstance::destroy() {
        delete m;
        m = nullptr;
    }

    void GeometryInstance::setVertexBuffer(const BufferView &vertexBuffer, OptixVertexFormat format) const {
        THROW_RUNTIME_ERROR(!m->forCustomPrimitives, "This geometry instance was created for custom primitives.");
        m->vertexBuffer = vertexBuffer;
        m->vertexFormat = format;
    }

    void GeometryInstance::setTriangleBuffer(const BufferView &triangleBuffer, OptixIndicesFormat format) const {
        THROW_RUNTIME_ERROR(!m->forCustomPrimitives, "This geometry instance was created for custom primitives.");
        m->triangleBuffer = triangleBuffer;
        m->indexFormat = format;
    }

    void GeometryInstance::setCustomPrimitiveAABBBuffer(const BufferView &primitiveAABBBuffer) const {
        THROW_RUNTIME_ERROR(m->forCustomPrimitives, "This geometry instance was created for triangles.");
        m->primitiveAABBBuffer = primitiveAABBBuffer;
    }

    void GeometryInstance::setPrimitiveIndexOffset(uint32_t offset) const {
        m->primitiveIndexOffset = offset;
    }

    void GeometryInstance::setNumMaterials(uint32_t numMaterials, const BufferView &matIndexOffsetBuffer, uint32_t indexOffsetSize) const {
        THROW_RUNTIME_ERROR(numMaterials > 0, "Invalid number of materials %u.", numMaterials);
        THROW_RUNTIME_ERROR((numMaterials == 1) != matIndexOffsetBuffer.isValid(),
                            "Material index offset buffer must be provided when multiple materials are used.");
        THROW_RUNTIME_ERROR(indexOffsetSize >= 1 && indexOffsetSize <= 4, "Invalid index offset size.");
        if (matIndexOffsetBuffer.isValid())
            THROW_RUNTIME_ERROR(matIndexOffsetBuffer.stride() >= indexOffsetSize, "Buffer's stride is smaller than the given index offset size.");
        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
        m->materialIndexOffsetBuffer = matIndexOffsetBuffer;
        m->materialIndexOffsetSize = indexOffsetSize;
        m->materials.resize(numMaterials);
    }

    void GeometryInstance::setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const {
        size_t numMaterials = m->materials.size();
        THROW_RUNTIME_ERROR(matIdx < numMaterials,
                            "Out of material bounds [0, %u).", static_cast<uint32_t>(numMaterials));

        m->buildInputFlags[matIdx] = flags;
    }

    void GeometryInstance::setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const {
        size_t numMaterials = m->materials.size();
        THROW_RUNTIME_ERROR(matIdx < numMaterials,
                            "Out of material bounds [0, %u).", static_cast<uint32_t>(numMaterials));

        uint32_t prevNumMatSets = m->materials[matIdx].size();
        if (matSetIdx >= prevNumMatSets)
            m->materials[matIdx].resize(matSetIdx + 1, nullptr);
        m->materials[matIdx][matSetIdx] = extract(mat);
    }

    void GeometryInstance::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        THROW_RUNTIME_ERROR(size <= s_maxGeometryInstanceUserDataSize,
                            "Maximum user data size for Material is %u bytes.", s_maxGeometryInstanceUserDataSize);
        THROW_RUNTIME_ERROR(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                            "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }



    SizeAlign GeometryAccelerationStructure::Priv::calcMaxRecordSizeAlign(uint32_t matSetIdx) const {
        SizeAlign maxRecordSizeAlign;
        for (const Child &child : children)
            maxRecordSizeAlign = max(maxRecordSizeAlign, child.geomInst->calcMaxRecordSizeAlign(matSetIdx));
        maxRecordSizeAlign += userDataSizeAlign;
        return maxRecordSizeAlign;
    }

    uint32_t GeometryAccelerationStructure::Priv::calcNumSBTRecords(uint32_t matSetIdx) const {
        uint32_t numSBTRecords = 0;
        for (const Child &child : children)
            numSBTRecords += child.geomInst->getNumSBTRecords();
        numSBTRecords *= numRayTypesPerMaterialSet[matSetIdx];

        return numSBTRecords;
    }

    uint32_t GeometryAccelerationStructure::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint8_t* records) const {
        THROW_RUNTIME_ERROR(matSetIdx < numRayTypesPerMaterialSet.size(),
                            "Material set index %u is out of bounds [0, %u).",
                            matSetIdx, static_cast<uint32_t>(numRayTypesPerMaterialSet.size()));

        uint32_t numRayTypes = numRayTypesPerMaterialSet[matSetIdx];
        uint32_t sumRecords = 0;
        for (uint32_t sbtGasIdx = 0; sbtGasIdx < children.size(); ++sbtGasIdx) {
            const Child &child = children[sbtGasIdx];
            uint32_t numRecords = child.geomInst->fillSBTRecords(pipeline, matSetIdx,
                                                                 userData.data(), userDataSizeAlign,
                                                                 numRayTypes, records);
            records += numRecords * scene->getSingleRecordSize();
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

    void GeometryAccelerationStructure::setConfiguration(ASTradeoff tradeoff,
                                                         bool allowUpdate,
                                                         bool allowCompaction,
                                                         bool allowRandomVertexAccess) const {
        THROW_RUNTIME_ERROR(!(m->forCustomPrimitives && allowRandomVertexAccess),
                            "Random vertex access is the feature only for triangle GAS.");
        bool changed = false;
        changed |= m->tradeoff != tradeoff;
        m->tradeoff = tradeoff;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;
        changed |= m->allowRandomVertexAccess != allowRandomVertexAccess;
        m->allowRandomVertexAccess = allowRandomVertexAccess;

        if (changed)
            m->markDirty();
    }

    void GeometryAccelerationStructure::addChild(GeometryInstance geomInst, CUdeviceptr preTransform) const {
        auto _geomInst = extract(geomInst);
        THROW_RUNTIME_ERROR(_geomInst, "Invalid geometry instance %p.", _geomInst);
        THROW_RUNTIME_ERROR(_geomInst->getScene() == m->scene, "Scene mismatch for the given geometry instance.");
        THROW_RUNTIME_ERROR(_geomInst->isCustomPrimitiveInstance() == m->forCustomPrimitives,
                            "This GAS was created for %s.", m->forCustomPrimitives ? "custom primitives" : "triangles");
        Priv::Child child;
        child.geomInst = _geomInst;
        child.preTransform = preTransform;
        auto idx = std::find(m->children.cbegin(), m->children.cend(), child);
        THROW_RUNTIME_ERROR(idx == m->children.cend(), "Geometry instance %p with transform %p has been already added.", _geomInst, preTransform);

        m->children.push_back(child);

        m->markDirty();
    }

    void GeometryAccelerationStructure::removeChild(GeometryInstance geomInst, CUdeviceptr preTransform) const {
        auto _geomInst = extract(geomInst);
        THROW_RUNTIME_ERROR(_geomInst, "Invalid geometry instance %p.", _geomInst);
        THROW_RUNTIME_ERROR(_geomInst->getScene() == m->scene, "Scene mismatch for the given geometry instance.");
        Priv::Child child;
        child.geomInst = _geomInst;
        child.preTransform = preTransform;
        auto idx = std::find(m->children.cbegin(), m->children.cend(), child);
        THROW_RUNTIME_ERROR(idx != m->children.cend(), "Geometry instance %p with transform %p has not been added.", _geomInst, preTransform);

        m->children.erase(idx);

        m->markDirty();
    }

    void GeometryAccelerationStructure::markDirty() const {
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

    void GeometryAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const {
        m->buildInputs.resize(m->children.size(), OptixBuildInput{});
        uint32_t childIdx = 0;
        for (const Priv::Child &child : m->children)
            child.geomInst->fillBuildInput(&m->buildInputs[childIdx++], child.preTransform);

        m->buildOptions = {};
        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        if (m->tradeoff == ASTradeoff::PreferFastTrace)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        else if (m->tradeoff == ASTradeoff::PreferFastBuild)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        m->buildOptions.buildFlags |= ((m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                       (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0) |
                                       (m->allowRandomVertexAccess ? OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS : 0));
        //m->buildOptions.motionOptions

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 m->buildInputs.data(), m->buildInputs.size(),
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;

        m->readyToBuild = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::rebuild(CUstream stream, const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
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
        for (const Priv::Child &child : m->children)
            child.geomInst->updateBuildInput(&m->buildInputs[childIdx++], child.preTransform);

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr,
                                    compactionEnabled ? 1 : 0));
        CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));

        m->accelBuffer = accelBuffer;
        m->available = true;
        m->readyToCompact = false;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void GeometryAccelerationStructure::prepareForCompact(size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        // TODO: ? stream
        CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle GeometryAccelerationStructure::compact(CUstream stream, const BufferView &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->readyToCompact, "You need to call prepareForCompact() before compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");
        THROW_RUNTIME_ERROR(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
                                      m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                                      &m->compactedHandle));
        CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));

        m->compactedAccelBuffer = compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void GeometryAccelerationStructure::removeUncompacted() const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));

        m->handle = 0;
        m->available = false;
    }

    void GeometryAccelerationStructure::update(CUstream stream, const BufferView &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        THROW_RUNTIME_ERROR(updateEnabled, "This AS does not allow update.");
        THROW_RUNTIME_ERROR(m->available || m->compactedAvailable, "AS has not been built yet.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const Priv::Child &child : m->children)
            child.geomInst->updateBuildInput(&m->buildInputs[childIdx++], child.preTransform);

        const BufferView &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OptixTraversableHandle tempHandle = handle;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &tempHandle,
                                    nullptr, 0));
        optixAssert(tempHandle == handle, "Update should not change the handle itself, what's going on?");
    }

    void GeometryAccelerationStructure::setUserData(const void* data, uint32_t size, uint32_t alignment) const {
        THROW_RUNTIME_ERROR(size <= s_maxGASUserDataSize,
                            "Maximum user data size for Material is %u bytes.", s_maxGASUserDataSize);
        THROW_RUNTIME_ERROR(alignment > 0 && alignment <= OPTIX_SBT_RECORD_ALIGNMENT,
                            "Valid alignment range is [1, %u].", OPTIX_SBT_RECORD_ALIGNMENT);
        m->userDataSizeAlign = SizeAlign(size, alignment);
        m->userData.resize(size);
        std::memcpy(m->userData.data(), data, size);
    }

    bool GeometryAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle GeometryAccelerationStructure::getHandle() const {
        return m->getHandle();
    }



    _GeometryAccelerationStructure* Transform::Priv::getDescendantGAS() const {
        if (childType == ChildType::GAS)
            return childGas;
        if (childType == ChildType::IAS)
            return nullptr;
        else if (childType == ChildType::Transform)
            return childXfm->getDescendantGAS();
        optixAssert_ShouldNotBeCalled();
        return nullptr;
    }

    void Transform::Priv::markDirty() {
        available = false;
    }

    void Transform::destroy() {
        delete m;
        m = nullptr;
    }

    void Transform::setConfiguration(TransformType type, uint32_t numKeys,
                                     size_t* transformSize) {
        m->type = type;
        numKeys = std::max(numKeys, 2u);
        if (m->type == TransformType::MatrixMotion) {
            m->dataSize = sizeof(OptixMatrixMotionTransform) +
                (numKeys - 2) * 12 * sizeof(float);
            m->options.numKeys = numKeys;
            m->data = new uint8_t[m->dataSize];
            std::memset(m->data, 0, m->dataSize);
            auto motionData = reinterpret_cast<float*>(m->data + offsetof(OptixMatrixMotionTransform, transform));
            for (int i = 0; i < numKeys; ++i) {
                float* dataPerKey = motionData + 12 * i;
                dataPerKey[0] = 1.0f; dataPerKey[1] = 0.0f; dataPerKey[2] = 0.0f; dataPerKey[3] = 0.0f;
                dataPerKey[4] = 1.0f; dataPerKey[5] = 0.0f; dataPerKey[6] = 0.0f; dataPerKey[7] = 0.0f;
                dataPerKey[8] = 1.0f; dataPerKey[9] = 0.0f; dataPerKey[10] = 0.0f; dataPerKey[11] = 0.0f;
            }
        }
        else if (m->type == TransformType::SRTMotion) {
            m->dataSize = sizeof(OptixSRTMotionTransform) +
                (numKeys - 2) * sizeof(OptixSRTData);
            m->options.numKeys = numKeys;
            m->data = new uint8_t[m->dataSize];
            std::memset(m->data, 0, m->dataSize);
            auto motionData = reinterpret_cast<OptixSRTData*>(m->data + offsetof(OptixSRTMotionTransform, srtData));
            for (int i = 0; i < numKeys; ++i) {
                OptixSRTData* dataPerKey = motionData + i;
                dataPerKey->sx = dataPerKey->sy = dataPerKey->sz = 1.0f;
                dataPerKey->a = dataPerKey->b = dataPerKey->c = 0.0f;
                dataPerKey->pvx = dataPerKey->pvy = dataPerKey->pvz = 0.0f;
                dataPerKey->qx = dataPerKey->qy = dataPerKey->qz = 0.0f;
                dataPerKey->qw = 1.0f;
                dataPerKey->tx = dataPerKey->ty = dataPerKey->tz = 0.0f;
            }
        }
        else if (m->type == TransformType::Static) {
            m->dataSize = sizeof(OptixStaticTransform);
            m->data = new uint8_t[m->dataSize];
            std::memset(m->data, 0, m->dataSize);
            auto xfm = reinterpret_cast<OptixStaticTransform*>(m->data);
            xfm->transform[0] = 1.0f; xfm->transform[1] = 0.0f; xfm->transform[2] = 0.0f; xfm->transform[3] = 0.0f;
            xfm->transform[4] = 1.0f; xfm->transform[5] = 0.0f; xfm->transform[6] = 0.0f; xfm->transform[7] = 0.0f;
            xfm->transform[8] = 1.0f; xfm->transform[9] = 0.0f; xfm->transform[10] = 0.0f; xfm->transform[11] = 0.0f;
            xfm->invTransform[0] = 1.0f; xfm->invTransform[1] = 0.0f; xfm->invTransform[2] = 0.0f; xfm->invTransform[3] = 0.0f;
            xfm->invTransform[4] = 1.0f; xfm->invTransform[5] = 0.0f; xfm->invTransform[6] = 0.0f; xfm->invTransform[7] = 0.0f;
            xfm->invTransform[8] = 1.0f; xfm->invTransform[9] = 0.0f; xfm->invTransform[10] = 0.0f; xfm->invTransform[11] = 0.0f;
        }

        *transformSize = m->dataSize;

        markDirty();
    }

    void Transform::setMotionOptions(float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->options.timeBegin = timeBegin;
        m->options.timeEnd = timeEnd;
        m->options.flags = flags;

        markDirty();
    }

    void Transform::setMatrixMotionKey(uint32_t keyIdx, const float matrix[12]) const {
        THROW_RUNTIME_ERROR(m->type == TransformType::MatrixMotion,
                            "This transform has been configured as matrix motion transform.");
        THROW_RUNTIME_ERROR(keyIdx <= m->options.numKeys,
                            "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<float*>(m->data + offsetof(OptixMatrixMotionTransform, transform));
        float* dataPerKey = motionData + 12 * keyIdx;

        std::copy_n(matrix, 12, dataPerKey);

        markDirty();
    }

    void Transform::setSRTMotionKey(uint32_t keyIdx, const float scale[3], const float orientation[4], const float translation[3]) const {
        THROW_RUNTIME_ERROR(m->type == TransformType::SRTMotion,
                            "This transform has been configured as SRT motion transform.");
        THROW_RUNTIME_ERROR(keyIdx <= m->options.numKeys,
                            "Number of motion keys was set to %u", m->options.numKeys);
        auto motionData = reinterpret_cast<OptixSRTData*>(m->data + offsetof(OptixSRTMotionTransform, srtData));
        OptixSRTData* dataPerKey = motionData + keyIdx;

        dataPerKey->sx = scale[0];
        dataPerKey->sy = scale[1];
        dataPerKey->sz = scale[2];
        dataPerKey->a = dataPerKey->b = dataPerKey->c = 0.0f;
        dataPerKey->pvx = dataPerKey->pvy = dataPerKey->pvz = 0.0f;
        std::copy_n(orientation, 4, &dataPerKey->qx);
        std::copy_n(translation, 3, &dataPerKey->tx);

        markDirty();
    }

    void Transform::setStaticTransform(const float matrix[12]) const {
        THROW_RUNTIME_ERROR(m->type == TransformType::Static,
                            "This transform has been configured as static transform.");
        float invDet = 1.0f / (matrix[ 0] * matrix[ 5] * matrix[10] +
                               matrix[ 1] * matrix[ 6] * matrix[ 8] +
                               matrix[ 2] * matrix[ 4] * matrix[ 9] -
                               matrix[ 2] * matrix[ 5] * matrix[ 8] -
                               matrix[ 1] * matrix[ 4] * matrix[10] -
                               matrix[ 0] * matrix[ 6] * matrix[ 9]);
        THROW_RUNTIME_ERROR(invDet != 0.0f, "Given matrix is not invertible.");

        auto xfm = reinterpret_cast<OptixStaticTransform*>(m->data);

        std::copy_n(matrix, 12, xfm->transform);

        float invMat[12];
        invMat[ 0] = invDet * (matrix[ 5] * matrix[10] - matrix[ 6] * matrix[ 9]);
        invMat[ 1] = invDet * (matrix[ 2] * matrix[ 9] - matrix[ 1] * matrix[10]);
        invMat[ 2] = invDet * (matrix[ 1] * matrix[ 6] - matrix[ 2] * matrix[ 5]);
        invMat[ 3] = -matrix[3];
        invMat[ 4] = invDet * (matrix[ 6] * matrix[ 8] - matrix[ 4] * matrix[10]);
        invMat[ 5] = invDet * (matrix[ 0] * matrix[10] - matrix[ 2] * matrix[ 8]);
        invMat[ 6] = invDet * (matrix[ 2] * matrix[ 4] - matrix[ 0] * matrix[ 6]);
        invMat[ 7] = -matrix[7];
        invMat[ 8] = invDet * (matrix[ 4] * matrix[ 9] - matrix[ 5] * matrix[ 8]);
        invMat[ 9] = invDet * (matrix[ 1] * matrix[ 8] - matrix[ 0] * matrix[ 9]);
        invMat[10] = invDet * (matrix[ 0] * matrix[ 5] - matrix[ 1] * matrix[ 4]);
        invMat[11] = -matrix[11];
        std::copy_n(invMat, 12, xfm->invTransform);

        markDirty();
    }

    void Transform::setChild(GeometryAccelerationStructure child) const {
        m->childType = ChildType::GAS;
        m->childGas = extract(child);

        markDirty();
    }

    void Transform::setChild(InstanceAccelerationStructure child) const {
        m->childType = ChildType::IAS;
        m->childIas = extract(child);

        markDirty();
    }

    void Transform::setChild(Transform child) const {
        m->childType = ChildType::Transform;
        m->childXfm = extract(child);

        markDirty();
    }

    void Transform::markDirty() const {
        return m->markDirty();
    }

    OptixTraversableHandle Transform::rebuild(CUstream stream, const BufferView &trDeviceMem) {
        THROW_RUNTIME_ERROR(m->type != TransformType::Invalid, "Transform type is invalid.");
        THROW_RUNTIME_ERROR(trDeviceMem.sizeInBytes() >= m->dataSize,
                            "Size of the given buffer is not enough.");
        THROW_RUNTIME_ERROR(m->childType != ChildType::Invalid, "Child is invalid.");

        OptixTraversableHandle childHandle = 0;
        if (m->childType == ChildType::GAS)
            childHandle = m->childGas->getHandle();
        else if (m->childType == ChildType::IAS)
            childHandle = m->childIas->getHandle();
        else if (m->childType == ChildType::Transform)
            childHandle = m->childXfm->getHandle();

        OptixTraversableType travType;
        if (m->type == TransformType::MatrixMotion) {
            auto tr = reinterpret_cast<OptixMatrixMotionTransform*>(m->data);
            tr->child = childHandle;
            tr->motionOptions = m->options;
            travType = OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM;
        }
        else if (m->type == TransformType::SRTMotion) {
            auto tr = reinterpret_cast<OptixSRTMotionTransform*>(m->data);
            tr->child = childHandle;
            tr->motionOptions = m->options;
            travType = OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
        }
        else if (m->type == TransformType::Static) {
            auto tr = reinterpret_cast<OptixStaticTransform*>(m->data);
            tr->child = childHandle;
            travType = OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
        }

        CUDADRV_CHECK(cuMemcpyHtoDAsync(trDeviceMem.getCUdeviceptr(), m->data, m->dataSize, stream));
        OPTIX_CHECK(optixConvertPointerToTraversableHandle(m->getRawContext(), trDeviceMem.getCUdeviceptr(),
                                                           travType,
                                                           &m->handle));
        m->available = true;

        return m->handle;
    }

    bool Transform::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle Transform::getHandle() const {
        return m->getHandle();
    }



    void Instance::Priv::fillInstance(OptixInstance* instance) const {
        *instance = {};
        instance->instanceId = id;
        instance->visibilityMask = visibilityMask;
        std::copy_n(instTransform, 12, instance->transform);
        instance->flags = flags;

        if (type == ChildType::GAS) {
            THROW_RUNTIME_ERROR(childGas->isReady(), "GAS %p is not ready.", childGas);
            instance->traversableHandle = childGas->getHandle();
            instance->sbtOffset = scene->getSBTOffset(childGas, matSetIndex);
        }
        else if (type == ChildType::IAS) {
            THROW_RUNTIME_ERROR(childIas->isReady(), "IAS %p is not ready.", childGas);
            instance->traversableHandle = childIas->getHandle();
            instance->sbtOffset = 0;
        }
        else if (type == ChildType::Transform) {
            THROW_RUNTIME_ERROR(childXfm->isReady(), "Transform %p is not ready.", childXfm);
            instance->traversableHandle = childXfm->getHandle();
            _GeometryAccelerationStructure* desGas = childXfm->getDescendantGAS();
            if (desGas)
                instance->sbtOffset = scene->getSBTOffset(desGas, matSetIndex);
            else
                instance->sbtOffset = 0;
        }
        else {
            optixAssert_ShouldNotBeCalled();
        }
    }

    void Instance::Priv::updateInstance(OptixInstance* instance) const {
        instance->instanceId = id;
        instance->visibilityMask = visibilityMask;
        std::copy_n(instTransform, 12, instance->transform);
        //instance->flags = flags; これは変えられない？
        //instance->sbtOffset = scene->getSBTOffset(childGas, matSetIndex);
    }

    bool Instance::Priv::isMotionAS() const {
        if (type == ChildType::GAS)
            childGas->hasMotion();
        else if (type == ChildType::IAS)
            childIas->hasMotion();
        return false;
    }

    bool Instance::Priv::isTransform() const {
        return type == ChildType::Transform;
    }

    void Instance::destroy() {
        delete m;
        m = nullptr;
    }

    void Instance::setChild(GeometryAccelerationStructure child, uint32_t matSetIdx) const {
        m->type = ChildType::GAS;
        m->childGas = extract(child);
        m->matSetIndex = matSetIdx;
    }

    void Instance::setChild(InstanceAccelerationStructure child) const {
        m->type = ChildType::IAS;
        m->childIas = extract(child);
        m->matSetIndex = 0;
    }

    void Instance::setChild(Transform child, uint32_t matSetIdx) const {
        m->type = ChildType::Transform;
        m->childXfm = extract(child);
        m->matSetIndex = matSetIdx;
    }

    void Instance::setTransform(const float transform[12]) const {
        std::copy_n(transform, 12, m->instTransform);
    }

    void Instance::setID(uint32_t value) const {
        uint32_t maxInstanceID = m->scene->getContext()->getMaxInstanceID();
        THROW_RUNTIME_ERROR(value <= maxInstanceID,
                            "Max instance ID value is 0x%08x.", maxInstanceID);
        m->id = value;
    }

    void Instance::setVisibilityMask(uint32_t mask) const {
        uint32_t numVisibilityMaskBits = m->scene->getContext()->getNumVisibilityMaskBits();
        THROW_RUNTIME_ERROR((mask >> numVisibilityMaskBits) == 0,
                            "Number of visibility mask bits is %u.", numVisibilityMaskBits);
        m->visibilityMask = mask;
    }

    void Instance::setFlags(OptixInstanceFlags flags) const {
        m->flags = flags;
    }



    void InstanceAccelerationStructure::Priv::markDirty() {
        readyToBuild = false;
        available = false;
        readyToCompact = false;
        compactedAvailable = false;
    }

    OptixTraversableHandle InstanceAccelerationStructure::Priv::rebuild(CUstream stream, const BufferView &instanceBuffer, const BufferView &aabbBuffer,
                                                                        const BufferView &accelBuffer, const BufferView &scratchBuffer) {
        THROW_RUNTIME_ERROR(readyToBuild, "You need to call prepareForBuild() before rebuild.");
        THROW_RUNTIME_ERROR(accelBuffer.sizeInBytes() >= memoryRequirement.outputSizeInBytes,
                            "Size of the given buffer is not enough.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= memoryRequirement.tempSizeInBytes,
                            "Size of the given scratch buffer is not enough.");
        THROW_RUNTIME_ERROR(instanceBuffer.sizeInBytes() >= instances.size() * sizeof(OptixInstance),
                            "Size of the given instance buffer is not enough.");

        // JP: アップデートの意味でリビルドするときはprepareForBuild()を呼ばないため
        //     インスタンス情報を更新する処理をここにも書いておく必要がある。
        // EN: User is not required to call prepareForBuild() when performing rebuild
        //     for purpose of update so updating instance information should be here.
        uint32_t childIdx = 0;
        for (const _Instance* child : children)
            child->updateInstance(&instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(instanceBuffer.getCUdeviceptr(), instances.data(),
                                        instanceBuffer.sizeInBytes(),
                                        stream));
        buildInput.instanceArray.instances = instanceBuffer.getCUdeviceptr();
        if (aabbBuffer.isValid()) {
            uint32_t numAABBs = motionOptions.numKeys * children.size();
            THROW_RUNTIME_ERROR(aabbBuffer.sizeInBytes() >= numAABBs * sizeof(OptixAabb),
                                "Size of the given AABB buffer is not enough.");
            buildInput.instanceArray.aabbs = aabbBuffer.getCUdeviceptr();
            buildInput.instanceArray.numAabbs = numAABBs;
        }

        bool compactionEnabled = (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(getRawContext(), stream, &buildOptions, &buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &handle,
                                    compactionEnabled ? &propertyCompactedSize : nullptr,
                                    compactionEnabled ? 1 : 0));
        CUDADRV_CHECK(cuEventRecord(finishEvent, stream));

        this->instanceBuffer = instanceBuffer;
        this->accelBuffer = accelBuffer;
        this->aabbBuffer = aabbBuffer;
        available = true;
        readyToCompact = false;
        compactedHandle = 0;
        compactedAvailable = false;

        return handle;
    }
    
    void InstanceAccelerationStructure::destroy() {
        delete m;
        m = nullptr;
    }

    void InstanceAccelerationStructure::setConfiguration(ASTradeoff tradeoff, bool allowUpdate, bool allowCompaction) const {
        bool changed = false;
        changed |= m->tradeoff != tradeoff;
        m->tradeoff = tradeoff;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;

        if (changed)
            m->markDirty();
    }

    void InstanceAccelerationStructure::setMotionOptions(uint32_t numKeys, float timeBegin, float timeEnd, OptixMotionFlags flags) const {
        m->motionOptions.numKeys = numKeys;
        m->motionOptions.timeBegin = timeBegin;
        m->motionOptions.timeEnd = timeEnd;
        m->motionOptions.flags = flags;

        markDirty();
    }

    void InstanceAccelerationStructure::addChild(Instance instance) const {
        _Instance* _inst = extract(instance);
        THROW_RUNTIME_ERROR(_inst, "Invalid instance %p.", _inst);
        THROW_RUNTIME_ERROR(_inst->getScene() == m->scene, "Scene mismatch for the given instance.");
        auto idx = std::find(m->children.cbegin(), m->children.cend(), _inst);
        THROW_RUNTIME_ERROR(idx == m->children.cend(), "Instance %p has been already added.", _inst);

        m->children.push_back(_inst);

        m->markDirty();
    }

    void InstanceAccelerationStructure::removeChild(Instance instance) const {
        _Instance* _inst = extract(instance);
        THROW_RUNTIME_ERROR(_inst, "Invalid instance %p.", _inst);
        THROW_RUNTIME_ERROR(_inst->getScene() == m->scene, "Scene mismatch for the given instance.");
        auto idx = std::find(m->children.cbegin(), m->children.cend(), _inst);
        THROW_RUNTIME_ERROR(idx != m->children.cend(), "Instance %p has not been added.", _inst);

        m->children.erase(idx);

        m->markDirty();
    }

    void InstanceAccelerationStructure::markDirty() const {
        m->markDirty();
    }

    void InstanceAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement, uint32_t* numInstances,
                                                        uint32_t* numAABBs) const {
        THROW_RUNTIME_ERROR(m->scene->sbtLayoutGenerationDone(),
                            "Shader binding table layout generation has not been done.");
        m->instances.resize(m->children.size());
        uint32_t childIdx = 0;
        bool transformExists = false;
        bool motionASExists = false;
        for (const _Instance* child : m->children) {
            child->fillInstance(&m->instances[childIdx++]);
            transformExists |= child->isTransform();
            motionASExists |= child->isMotionAS();
        }
        m->aabbsRequired = transformExists || (motionASExists && m->motionOptions.numKeys >= 2);
        if (m->aabbsRequired) {
            THROW_RUNTIME_ERROR(numAABBs, "This IAS requires AABB buffer, but numAABBs is not provided.");
            *numAABBs = m->children.size() * m->motionOptions.numKeys;
        }

        // Fill the build input.
        {
            m->buildInput = OptixBuildInput{};
            m->buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            OptixBuildInputInstanceArray &instArray = m->buildInput.instanceArray;
            instArray.instances = 0;
            instArray.numInstances = static_cast<uint32_t>(m->children.size());
            instArray.aabbs = 0;
            if (m->aabbsRequired)
                instArray.numAabbs = *numAABBs;
        }

        m->buildOptions = {};
        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        if (m->tradeoff == ASTradeoff::PreferFastTrace)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        else if (m->tradeoff == ASTradeoff::PreferFastBuild)
            m->buildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        m->buildOptions.buildFlags |= ((m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                       (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
        m->buildOptions.motionOptions = m->motionOptions;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 &m->buildInput, 1,
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;
        *numInstances = m->instances.size();

        m->readyToBuild = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::rebuild(CUstream stream, const BufferView &instanceBuffer,
                                                                  const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
        THROW_RUNTIME_ERROR(!m->aabbsRequired, "You need to call the motion-enabled variant of this function for this IAS.");
        return m->rebuild(stream, instanceBuffer, BufferView(), accelBuffer, scratchBuffer);
    }

    OptixTraversableHandle InstanceAccelerationStructure::rebuild(CUstream stream, const BufferView &instanceBuffer, const BufferView &aabbBuffer,
                                                                  const BufferView &accelBuffer, const BufferView &scratchBuffer) const {
        THROW_RUNTIME_ERROR(m->aabbsRequired, "You need to call the motion-disabled variant of this function for this IAS.");
        return m->rebuild(stream, instanceBuffer, aabbBuffer, accelBuffer, scratchBuffer);
    }

    void InstanceAccelerationStructure::prepareForCompact(size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        // EN: Wait the completion of rebuild/update then obtain the size after coompaction.
        // TODO: ? stream
        CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));

        *compactedAccelBufferSize = m->compactedSize;

        m->readyToCompact = true;
    }

    OptixTraversableHandle InstanceAccelerationStructure::compact(CUstream stream, const BufferView &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->readyToCompact, "You need to call prepareForCompact() before compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");
        THROW_RUNTIME_ERROR(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
                                      m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                                      &m->compactedHandle));
        CUDADRV_CHECK(cuEventRecord(m->finishEvent, stream));

        m->compactedAccelBuffer = compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void InstanceAccelerationStructure::removeUncompacted() const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuEventSynchronize(m->finishEvent));

        m->handle = 0;
        m->available = false;
    }

    void InstanceAccelerationStructure::update(CUstream stream, const BufferView &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        THROW_RUNTIME_ERROR(updateEnabled, "This AS does not allow update.");
        THROW_RUNTIME_ERROR(m->available || m->compactedAvailable, "AS has not been built yet.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        uint32_t childIdx = 0;
        for (const _Instance* child : m->children)
            child->updateInstance(&m->instances[childIdx++]);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(m->instanceBuffer.getCUdeviceptr(), m->instances.data(),
                                        m->instanceBuffer.sizeInBytes(),
                                        stream));

        const BufferView &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OptixTraversableHandle tempHandle = handle;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, &m->buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &tempHandle,
                                    nullptr, 0));
        optixAssert(tempHandle == handle, "Update should not change the handle itself, what's going on?");
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle InstanceAccelerationStructure::getHandle() const {
        return m->getHandle();
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
    
    void Pipeline::Priv::setupShaderBindingTable(CUstream stream) {
        if (!sbtIsUpToDate) {
            THROW_RUNTIME_ERROR(rayGenProgram, "Ray generation program is not set.");
            for (int i = 0; i < numMissRayTypes; ++i)
                THROW_RUNTIME_ERROR(missPrograms[i], "Miss program is not set for ray type %d.", i);
            for (int i = 0; i < numCallablePrograms; ++i)
                THROW_RUNTIME_ERROR(callablePrograms[i], "Callable program is not set for index %d.", i);

            auto records = reinterpret_cast<uint8_t*>(sbtHostMem);
            size_t offset = 0;

            size_t rayGenRecordOffset = offset;
            rayGenProgram->packHeader(records + offset);
            offset += OPTIX_SBT_RECORD_HEADER_SIZE;

            size_t exceptionRecordOffset = offset;
            if (exceptionProgram)
                exceptionProgram->packHeader(records + offset);
            offset += OPTIX_SBT_RECORD_HEADER_SIZE;

            CUdeviceptr missRecordOffset = offset;
            for (int i = 0; i < numMissRayTypes; ++i) {
                missPrograms[i]->packHeader(records + offset);
                offset += OPTIX_SBT_RECORD_HEADER_SIZE;
            }

            CUdeviceptr callableRecordOffset = offset;
            for (int i = 0; i < numCallablePrograms; ++i) {
                callablePrograms[i]->packHeader(records + offset);
                offset += OPTIX_SBT_RECORD_HEADER_SIZE;
            }

            CUDADRV_CHECK(cuMemcpyHtoDAsync(sbt.getCUdeviceptr(), sbtHostMem, sbt.sizeInBytes(), stream));

            CUdeviceptr baseAddress = sbt.getCUdeviceptr();
            sbtParams.raygenRecord = baseAddress + rayGenRecordOffset;
            sbtParams.exceptionRecord = exceptionProgram ? baseAddress + exceptionRecordOffset : 0;
            sbtParams.missRecordBase = baseAddress + missRecordOffset;
            sbtParams.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
            sbtParams.missRecordCount = numMissRayTypes;
            sbtParams.callablesRecordBase = numCallablePrograms ? baseAddress + callableRecordOffset : 0;
            sbtParams.callablesRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
            sbtParams.callablesRecordCount = numCallablePrograms;

            sbtIsUpToDate = true;
        }

        if (!hitGroupSbtIsUpToDate) {
            scene->setupHitGroupSBT(stream, this, hitGroupSbt, hitGroupSbtHostMem);

            sbtParams.hitgroupRecordBase = hitGroupSbt.getCUdeviceptr();
            sbtParams.hitgroupRecordStrideInBytes = scene->getSingleRecordSize();
            sbtParams.hitgroupRecordCount = hitGroupSbt.sizeInBytes() / scene->getSingleRecordSize();

            hitGroupSbtIsUpToDate = true;
        }
    }

    void Pipeline::destroy() {
        delete m;
        m = nullptr;
    }

    void Pipeline::setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues,
                                      const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                      bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags,
                                      uint32_t supportedPrimitiveTypeFlags) const {
        // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
        // EN: Set pipeline compile options common among modules in the pipeline and the pipeline itself.
        m->pipelineCompileOptions = {};
        m->pipelineCompileOptions.numPayloadValues = numPayloadValues;
        m->pipelineCompileOptions.numAttributeValues = numAttributeValues;
        m->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m->pipelineCompileOptions.exceptionFlags = exceptionFlags;
        m->pipelineCompileOptions.usesPrimitiveTypeFlags = supportedPrimitiveTypeFlags;

        m->sizeOfPipelineLaunchParams = sizeOfLaunchParams;
    }



    Module Pipeline::createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount,
                                               OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) const {
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
        THROW_RUNTIME_ERROR(_module && entryFunctionName,
                            "Either of RayGen module or entry function name is not provided.");
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
        THROW_RUNTIME_ERROR(_module && entryFunctionName,
                            "Either of Exception module or entry function name is not provided.");
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

    ProgramGroup Pipeline::createCallableProgramGroup(Module module_DC, const char* entryFunctionNameDC,
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



    void Pipeline::link(uint32_t maxTraceDepth, OptixCompileDebugLevel debugLevel) const {
        THROW_RUNTIME_ERROR(!m->pipelineLinked, "This pipeline has been already linked.");

        if (!m->pipelineLinked) {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
            pipelineLinkOptions.debugLevel = debugLevel;

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
        m->sbtLayoutIsUpToDate = false;
    }

    void Pipeline::setNumCallablePrograms(uint32_t numCallablePrograms) const {
        m->numCallablePrograms = numCallablePrograms;
        m->callablePrograms.resize(m->numCallablePrograms);
        m->sbtLayoutIsUpToDate = false;
    }

    void Pipeline::generateShaderBindingTableLayout(size_t* memorySize) const {
        if (m->sbtLayoutIsUpToDate) {
            *memorySize = m->sbtSize;
            return;
        }

        m->sbtSize = 0;
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE; // RayGen
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE; // Exception
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE * m->numMissRayTypes; // Miss
        m->sbtSize += OPTIX_SBT_RECORD_HEADER_SIZE * m->numCallablePrograms; // Callable
        m->sbtLayoutIsUpToDate = true;

        *memorySize = m->sbtSize;
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
        THROW_RUNTIME_ERROR(index < m->numCallablePrograms, "Invalid callable program index.");
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->callablePrograms[index] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const {
        THROW_RUNTIME_ERROR(shaderBindingTable.sizeInBytes() >= m->sbtSize,
                            "Hit group shader binding table size is not enough.");
        THROW_RUNTIME_ERROR(hostMem,
                            "Host-side SBT counterpart must be provided.");
        m->sbt = shaderBindingTable;
        m->sbtHostMem = hostMem;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setScene(const Scene &scene) const {
        m->scene = extract(scene);
        m->hitGroupSbt = BufferView();
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::setHitGroupShaderBindingTable(const BufferView &shaderBindingTable, void* hostMem) const {
        THROW_RUNTIME_ERROR(hostMem,
                            "Host-side hit group SBT counterpart must be provided.");
        m->hitGroupSbt = shaderBindingTable;
        m->hitGroupSbtHostMem = hostMem;
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::markHitGroupShaderBindingTableDirty() const {
        m->hitGroupSbtIsUpToDate = false;
    }

    void Pipeline::setStackSize(uint32_t directCallableStackSizeFromTraversal,
                                uint32_t directCallableStackSizeFromState,
                                uint32_t continuationStackSize,
                                uint32_t maxTraversableGraphDepth) const {
        if (m->pipelineCompileOptions.traversableGraphFlags & OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING)
            maxTraversableGraphDepth = 2;
        else if (m->pipelineCompileOptions.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)
            maxTraversableGraphDepth = 1;
        OPTIX_CHECK(optixPipelineSetStackSize(m->rawPipeline,
                                              directCallableStackSizeFromTraversal,
                                              directCallableStackSizeFromState,
                                              continuationStackSize,
                                              maxTraversableGraphDepth));
    }

    void Pipeline::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const {
        THROW_RUNTIME_ERROR(m->sbtLayoutIsUpToDate, "Shader binding table layout is outdated.");
        THROW_RUNTIME_ERROR(m->sbt.isValid(), "Shader binding table is not set.");
        THROW_RUNTIME_ERROR(m->sbt.sizeInBytes() >= m->sbtSize,
                            "Shader binding table size is not enough.");
        THROW_RUNTIME_ERROR(m->scene, "Scene is not set.");
        bool hasMotionAS;
        THROW_RUNTIME_ERROR(m->scene->isReady(&hasMotionAS), "Scene is not ready.");
        THROW_RUNTIME_ERROR(m->pipelineCompileOptions.usesMotionBlur || !hasMotionAS,
                            "Scene has a motion AS but the pipeline has not been configured for motion.");
        THROW_RUNTIME_ERROR(m->hitGroupSbt.isValid(), "Hitgroup shader binding table is not set.");

        m->setupShaderBindingTable(stream);

        OPTIX_CHECK(optixLaunch(m->rawPipeline, stream, plpOnDevice, m->sizeOfPipelineLaunchParams,
                                &m->sbtParams, dimX, dimY, dimZ));
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



    void Denoiser::destroy() {
        delete m;
        m = nullptr;
    }

    void Denoiser::setModel(OptixDenoiserModelKind kind, void* data, size_t sizeInBytes) const {
        THROW_RUNTIME_ERROR((kind != OPTIX_DENOISER_MODEL_KIND_USER) != (data != nullptr),
                            "When a user model is used, data must be provided, otherwise data must be null.");
        OPTIX_CHECK(optixDenoiserSetModel(m->rawDenoiser, kind, data, sizeInBytes));

        m->stateIsReady = false;
        m->imageSizeSet = false;

        m->modelSet = true;
    }

    void Denoiser::prepare(uint32_t imageWidth, uint32_t imageHeight, uint32_t tileWidth, uint32_t tileHeight,
                           size_t* stateBufferSize, size_t* scratchBufferSize, size_t* scratchBufferSizeForComputeIntensity,
                           uint32_t* numTasks) const {
        THROW_RUNTIME_ERROR(m->modelSet, "Model has not been set.");
        THROW_RUNTIME_ERROR(tileWidth <= imageWidth && tileHeight <= imageHeight,
                            "Tile width/height must be equal to or smaller than the image size.");

        if (tileWidth == 0)
            tileWidth = imageWidth;
        if (tileHeight == 0)
            tileHeight = imageHeight;

        m->useTiling = tileWidth < imageWidth || tileHeight < imageHeight;

        m->imageWidth = imageWidth;
        m->imageHeight = imageHeight;
        m->tileWidth = tileWidth;
        m->tileHeight = tileHeight;
        OptixDenoiserSizes sizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(m->rawDenoiser, tileWidth, tileHeight, &sizes));
        m->stateSize = sizes.stateSizeInBytes;
        m->scratchSize = m->useTiling ?
            sizes.withOverlapScratchSizeInBytes : sizes.withoutOverlapScratchSizeInBytes;
        m->scratchSizeForComputeIntensity = sizeof(int32_t) * (2 + m->imageWidth * m->imageHeight);
        m->overlapWidth = sizes.overlapWindowSizeInPixels;
        m->maxInputWidth = std::min(tileWidth + 2 * m->overlapWidth, imageWidth);
        m->maxInputHeight = std::min(tileHeight + 2 * m->overlapWidth, imageHeight);

        *stateBufferSize = m->stateSize;
        *scratchBufferSize = m->scratchSize;
        *scratchBufferSizeForComputeIntensity = m->scratchSizeForComputeIntensity;

        *numTasks = 0;
        for (int32_t outputOffsetY = 0; outputOffsetY < imageHeight;) {
            int32_t outputHeight = tileHeight;
            if (outputOffsetY == 0)
                outputHeight += m->overlapWidth;

            for (int32_t outputOffsetX = 0; outputOffsetX < imageWidth;) {
                uint32_t outputWidth = tileWidth;
                if (outputOffsetX == 0)
                    outputWidth += m->overlapWidth;

                ++*numTasks;

                outputOffsetX += outputWidth;
            }

            outputOffsetY += outputHeight;
        }

        m->stateIsReady = false;
        m->imageSizeSet = true;
    }

    void Denoiser::getTasks(DenoisingTask* tasks) const {
        THROW_RUNTIME_ERROR(m->imageSizeSet,
                            "Call prepare() before this function.");

        uint32_t taskIdx = 0;
        for (int32_t outputOffsetY = 0; outputOffsetY < m->imageHeight;) {
            int32_t outputHeight = m->tileHeight;
            if (outputOffsetY == 0)
                outputHeight += m->overlapWidth;
            if (outputOffsetY + outputHeight > m->imageHeight)
                outputHeight = m->imageHeight - outputOffsetY;

            int32_t inputOffsetY = std::max(outputOffsetY - m->overlapWidth, 0);
            if (inputOffsetY + m->maxInputHeight > m->imageHeight)
                inputOffsetY = m->imageHeight - m->maxInputHeight;

            for (int32_t outputOffsetX = 0; outputOffsetX < m->imageWidth;) {
                uint32_t outputWidth = m->tileWidth;
                if (outputOffsetX == 0)
                    outputWidth += m->overlapWidth;
                if (outputOffsetX + outputWidth > m->imageWidth)
                    outputWidth = m->imageWidth - outputOffsetX;

                int32_t inputOffsetX = std::max(outputOffsetX - m->overlapWidth, 0);
                if (inputOffsetX + m->maxInputWidth > m->imageWidth)
                    inputOffsetX = m->imageWidth - m->maxInputWidth;

                _DenoisingTask task;
                task.inputOffsetX = inputOffsetX;
                task.inputOffsetY = inputOffsetY;
                task.outputOffsetX = outputOffsetX;
                task.outputOffsetY = outputOffsetY;
                task.outputWidth = outputWidth;
                task.outputHeight = outputHeight;
                tasks[taskIdx++] = task;

                outputOffsetX += outputWidth;
            }

            outputOffsetY += outputHeight;
        }
    }

    void Denoiser::setLayers(const BufferView &color, const BufferView &albedo, const BufferView &normal, const BufferView &denoisedColor,
                             OptixPixelFormat colorFormat, OptixPixelFormat albedoFormat, OptixPixelFormat normalFormat) const {
        THROW_RUNTIME_ERROR(m->imageSizeSet,
                            "Call prepare() before this function.");
        THROW_RUNTIME_ERROR(color.isValid(), "Input color buffer must be set.");
        if (m->inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO || m->inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
            THROW_RUNTIME_ERROR(albedo.isValid(), "Denoiser requires albedo buffer.");
        if (m->inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
            THROW_RUNTIME_ERROR(normal.isValid(), "Denoiser requires normal buffer.");

        m->colorBuffer = color;
        m->albedoBuffer = albedo;
        m->normalBuffer = normal;
        m->outputBuffer = denoisedColor;
        m->colorFormat = colorFormat;
        m->albedoFormat = albedoFormat;
        m->normalFormat = normalFormat;

        m->imageLayersSet = true;
    }

    void Denoiser::setupState(CUstream stream, const BufferView &stateBuffer, const BufferView &scratchBuffer) const {
        THROW_RUNTIME_ERROR(m->imageSizeSet,
                            "Call setImageSizes() before this function.");
        THROW_RUNTIME_ERROR(stateBuffer.sizeInBytes() >= m->stateSize,
                            "Size of the given state buffer is not enough.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->scratchSize,
                            "Size of the given scratch buffer is not enough.");
        uint32_t maxInputWidth = m->useTiling ? (m->tileWidth + 2 * m->overlapWidth) : m->imageWidth;
        uint32_t maxInputHeight = m->useTiling ? (m->tileHeight + 2 * m->overlapWidth) : m->imageHeight;
        OPTIX_CHECK(optixDenoiserSetup(m->rawDenoiser, stream,
                                       maxInputWidth, maxInputHeight,
                                       stateBuffer.getCUdeviceptr(), stateBuffer.sizeInBytes(),
                                       scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));

        m->stateBuffer = stateBuffer;
        m->scratchBuffer = scratchBuffer;
        m->stateIsReady = true;
    }

    void Denoiser::computeIntensity(CUstream stream, const BufferView &scratchBuffer, CUdeviceptr outputIntensity) {
        THROW_RUNTIME_ERROR(m->imageLayersSet, "You need to set image layers and formats before invoke.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->scratchSizeForComputeIntensity,
                            "Size of the given scratch buffer is not enough.");

        OptixImage2D colorLayer = {};
        colorLayer.data = m->colorBuffer.getCUdeviceptr();
        colorLayer.width = m->imageWidth;
        colorLayer.height = m->imageHeight;
        colorLayer.format = m->colorFormat;
        colorLayer.pixelStrideInBytes = getPixelSize(m->colorFormat);
        colorLayer.rowStrideInBytes = colorLayer.pixelStrideInBytes * m->imageWidth;

        OPTIX_CHECK(optixDenoiserComputeIntensity(m->rawDenoiser, stream,
                                                  &colorLayer, outputIntensity,
                                                  scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes()));
    }

    void Denoiser::invoke(CUstream stream, bool denoiseAlpha, CUdeviceptr hdrIntensity, float blendFactor,
                          const DenoisingTask &task) {
        THROW_RUNTIME_ERROR(m->stateIsReady, "You need to call setupState() before invoke.");
        THROW_RUNTIME_ERROR(m->imageLayersSet, "You need to set image layers and formats before invoke.");
        OptixDenoiserParams params = {};
        params.denoiseAlpha = denoiseAlpha;
        params.hdrIntensity = hdrIntensity;
        params.blendFactor = blendFactor;

        _DenoisingTask _task(task);

        uint32_t numInputLayers = 0;

        const auto setupInputLayer = [&]
        (OptixPixelFormat format, CUdeviceptr baseAddress, OptixImage2D* layer) {
            uint32_t pixelStride = getPixelSize(format);
            *layer = {};
            layer->rowStrideInBytes = m->imageWidth * pixelStride;
            layer->pixelStrideInBytes = pixelStride;
            uint32_t addressOffset = _task.inputOffsetY * layer->rowStrideInBytes + _task.inputOffsetX * pixelStride;
            layer->data = baseAddress + addressOffset;
            layer->width = m->maxInputWidth;
            layer->height = m->maxInputHeight;
            layer->format = format;

            ++numInputLayers;
        };

        // TODO: 入出力画像のrowStrideを指定できるようにする。

        OptixImage2D denoiserInputs[3];
        setupInputLayer(m->colorFormat, m->colorBuffer.getCUdeviceptr(), &denoiserInputs[0]);
        if (m->inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO ||
            m->inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
            setupInputLayer(m->albedoFormat, m->albedoBuffer.getCUdeviceptr(), &denoiserInputs[1]);
        if (m->inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
            setupInputLayer(m->normalFormat, m->normalBuffer.getCUdeviceptr(), &denoiserInputs[2]);

        OptixImage2D denoiserOutput = {};
        {
            OptixImage2D &layer = denoiserOutput;
            OptixPixelFormat format = m->colorFormat;
            uint32_t pixelStride = getPixelSize(format);
            layer.rowStrideInBytes = m->imageWidth * pixelStride;
            layer.pixelStrideInBytes = pixelStride;
            uint32_t addressOffset = _task.outputOffsetY * layer.rowStrideInBytes + _task.outputOffsetX * pixelStride;
            layer.data = m->outputBuffer.getCUdeviceptr() + addressOffset;
            layer.width = _task.outputWidth;
            layer.height = _task.outputHeight;
            layer.format = format;
        }

        int32_t offsetX = _task.outputOffsetX - _task.inputOffsetX;
        int32_t offsetY = _task.outputOffsetY - _task.inputOffsetY;
        OPTIX_CHECK(optixDenoiserInvoke(m->rawDenoiser, stream,
                                        &params,
                                        m->stateBuffer.getCUdeviceptr(), m->stateBuffer.sizeInBytes(),
                                        denoiserInputs, numInputLayers,
                                        offsetX, offsetY,
                                        &denoiserOutput,
                                        m->scratchBuffer.getCUdeviceptr(), m->scratchBuffer.sizeInBytes()));
    }
}
