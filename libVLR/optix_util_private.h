#pragma once

#include "optix_util.h"

#if defined(OPTIX_Platform_Windows_MSVC)
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef min
#   undef max
#   undef near
#   undef far
#   undef RGB
#endif

#include <optix_function_table_definition.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include <intrin.h>

#include <stdexcept>

#define OPTIX_CHECK(call) \
    do { \
        OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK_LOG(call) \
    do { \
        OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n" \
               << "Log: " << log << (logSize > sizeof(log) ? "<TRUNCATED>" : "") \
               << "\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



namespace optixu {
    static std::runtime_error make_runtime_error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        va_end(args);

        return std::runtime_error(str);
    }

#define THROW_RUNTIME_ERROR(expr, fmt, ...) do { if (!(expr)) throw make_runtime_error(fmt, ##__VA_ARGS__); } while (0)

    static void logCallBack(uint32_t level, const char* tag, const char* message, void* cbdata) {
        optixPrintf("[%2u][%12s]: %s\n", level, tag, message);
    }

    static BufferType s_BufferType = BufferType::Device;



#define OPTIX_ALIAS_PIMPL(Name) using _ ## Name = Name::Priv

    OPTIX_ALIAS_PIMPL(Context);
    OPTIX_ALIAS_PIMPL(Material);
    OPTIX_ALIAS_PIMPL(Scene);
    OPTIX_ALIAS_PIMPL(GeometryInstance);
    OPTIX_ALIAS_PIMPL(GeometryAccelerationStructure);
    OPTIX_ALIAS_PIMPL(Instance);
    OPTIX_ALIAS_PIMPL(InstanceAccelerationStructure);
    OPTIX_ALIAS_PIMPL(Pipeline);
    OPTIX_ALIAS_PIMPL(Module);
    OPTIX_ALIAS_PIMPL(ProgramGroup);



#define OPTIX_OPAQUE_BRIDGE(BaseName) \
    friend class BaseName; \
\
    BaseName getPublicType() { \
        BaseName ret; \
        ret.m = this; \
        return ret; \
    } \
\
    static BaseName::Priv* extract(BaseName publicType) { \
        return publicType.m; \
    }

    template <typename PublicType>
    static typename PublicType::Priv* extract(const PublicType &obj) {
        return PublicType::Priv::extract(obj);
    }



    struct SizeAlign {
        uint32_t size;
        uint32_t alignment;

        constexpr SizeAlign() : size(0), alignment(0) {}
        constexpr SizeAlign(uint32_t s, uint32_t a) : size(s), alignment(a) {}

        SizeAlign &add(const SizeAlign &sa, uint32_t* offset) {
            uint32_t mask = sa.alignment - 1;
            alignment = std::max(alignment, sa.alignment);
            size = (size + mask) & ~mask;
            if (offset)
                *offset = size;
            size += sa.size;
            return *this;
        }
        SizeAlign &operator+=(const SizeAlign &sa) {
            return add(sa, nullptr);
        }
        SizeAlign &alignUp() {
            uint32_t mask = alignment - 1;
            size = (size + mask) & ~mask;
            return *this;
        }
    };

    SizeAlign max(const SizeAlign &sa0, const SizeAlign &sa1) {
        return SizeAlign{ std::max(sa0.size, sa1.size), std::max(sa0.alignment, sa1.alignment) };
    }



    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupSBTRecord {
        uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
        HitGroupSBTRecordData data;
    };



    class Context::Priv {
        CUcontext cudaContext;
        OptixDeviceContext rawContext;

    public:
        OPTIX_OPAQUE_BRIDGE(Context);

        Priv(CUcontext cuContext) : cudaContext(cuContext) {
            OPTIX_CHECK(optixInit());

            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &logCallBack;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &rawContext));
        }
        ~Priv() {
            optixDeviceContextDestroy(rawContext);
        }

        CUcontext getCUDAContext() const {
            return cudaContext;
        }
        OptixDeviceContext getRawContext() const {
            return rawContext;
        }
    };



    class Material::Priv {
        struct Key {
            const _Pipeline* pipeline;
            uint32_t rayType;

            bool operator<(const Key &rKey) const {
                if (pipeline < rKey.pipeline) {
                    return true;
                }
                else if (pipeline == rKey.pipeline) {
                    if (rayType < rKey.rayType)
                        return true;
                }
                return false;
            }

            struct Hash {
                typedef std::size_t result_type;

                std::size_t operator()(const Key& key) const {
                    size_t seed = 0;
                    auto hash0 = std::hash<const _Pipeline*>()(key.pipeline);
                    auto hash1 = std::hash<uint32_t>()(key.rayType);
                    seed ^= hash0 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };
            bool operator==(const Key &rKey) const {
                return pipeline == rKey.pipeline && rayType == rKey.rayType;
            }
        };

        _Context* context;
        uint32_t userData;

        std::unordered_map<Key, const _ProgramGroup*, Key::Hash> programs;

    public:
        OPTIX_OPAQUE_BRIDGE(Material);

        Priv(_Context* ctxt) :
            context(ctxt), userData(0) {}
        ~Priv() {}

        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }

        void setRecordData(const _Pipeline* pipeline, uint32_t rayType, HitGroupSBTRecord* record) const;
    };


    
    class Scene::Priv {
        struct SBTOffsetKey {
            const _GeometryAccelerationStructure* gas;
            uint32_t matSetIndex;

            bool operator<(const SBTOffsetKey &rKey) const {
                if (gas < rKey.gas) {
                    return true;
                }
                else if (gas == rKey.gas) {
                    if (matSetIndex < rKey.matSetIndex)
                        return true;
                }
                return false;
            }

            struct Hash {
                typedef std::size_t result_type;

                std::size_t operator()(const SBTOffsetKey& key) const {
                    size_t seed = 0;
                    auto hash0 = std::hash<const _GeometryAccelerationStructure*>()(key.gas);
                    auto hash1 = std::hash<uint32_t>()(key.matSetIndex);
                    seed ^= hash0 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };
            bool operator==(const SBTOffsetKey &rKey) const {
                return gas == rKey.gas && matSetIndex == rKey.matSetIndex;
            }
        };

        const _Context* context;
        std::unordered_set<_GeometryAccelerationStructure*> geomASs;
        std::unordered_map<SBTOffsetKey, uint32_t, SBTOffsetKey::Hash> sbtOffsets;
        uint32_t numSBTRecords;
        std::unordered_set<_InstanceAccelerationStructure*> instASs;
        struct {
            unsigned int sbtLayoutIsUpToDate : 1;
        };

    public:
        OPTIX_OPAQUE_BRIDGE(Scene);

        Priv(const _Context* ctxt) : context(ctxt), sbtLayoutIsUpToDate(false) {}
        ~Priv() {}

        CUcontext getCUDAContext() const {
            return context->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        void addGAS(_GeometryAccelerationStructure* gas) {
            geomASs.insert(gas);
        }
        void removeGAS(_GeometryAccelerationStructure* gas) {
            geomASs.erase(gas);
        }
        void addIAS(_InstanceAccelerationStructure* ias) {
            instASs.insert(ias);
        }
        void removeIAS(_InstanceAccelerationStructure* ias) {
            instASs.erase(ias);
        }

        bool sbtLayoutGenerationDone() const {
            return sbtLayoutIsUpToDate;
        }
        void markSBTLayoutDirty();
        uint32_t getSBTOffset(_GeometryAccelerationStructure* gas, uint32_t matSetIdx) {
            return sbtOffsets.at(SBTOffsetKey{ gas, matSetIdx });
        }

        void setupHitGroupSBT(const _Pipeline* pipeline, Buffer* sbt);

        bool isReady();
    };



    class GeometryInstance::Priv {
        _Scene* scene;
        uint32_t userData;

        // TODO: support deformation blur (multiple vertex buffers)
        CUdeviceptr vertexBufferArray[1];
        Buffer* vertexBuffer;
        Buffer* triangleBuffer;
        TypedBuffer<uint32_t>* materialIndexOffsetBuffer;
        std::vector<uint32_t> buildInputFlags; // per SBT record

        std::vector<std::vector<const _Material*>> materialSets;

    public:
        OPTIX_OPAQUE_BRIDGE(GeometryInstance);

        Priv(_Scene* _scene) :
            scene(_scene),
            userData(0),
            vertexBuffer(nullptr), triangleBuffer(nullptr), materialIndexOffsetBuffer(nullptr) {
        }
        ~Priv() {}

        const _Scene* getScene() const {
            return scene;
        }
        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        void fillBuildInput(OptixBuildInput* input) const;
        void updateBuildInput(OptixBuildInput* input) const;

        uint32_t getNumSBTRecords() const;
        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes,
                                HitGroupSBTRecord* records) const;
    };



    class GeometryAccelerationStructure::Priv {
        _Scene* scene;

        std::vector<uint32_t> numRayTypesPerMaterialSet;

        std::unordered_set<_GeometryInstance*> children;
        std::vector<OptixBuildInput> buildInputs;

        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes memoryRequirement;

        TypedBuffer<size_t> compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        const Buffer* accelBuffer;
        const Buffer* compactedAccelBuffer;
        struct {
            unsigned int preferFastTrace : 1;
            unsigned int allowUpdate : 1;
            unsigned int allowCompaction : 1;
            unsigned int readyToBuild : 1;
            unsigned int available : 1;
            unsigned int readyToCompact : 1;
            unsigned int compactedAvailable : 1;
        };

    public:
        OPTIX_OPAQUE_BRIDGE(GeometryAccelerationStructure);

        Priv(_Scene* _scene) : scene(_scene) {
            scene->addGAS(this);

            compactedSizeOnDevice.initialize(scene->getCUDAContext(), s_BufferType, 1);

            propertyCompactedSize = OptixAccelEmitDesc{};
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getCUdeviceptr();

            preferFastTrace = true;
            allowUpdate = false;
            allowCompaction = false;

            readyToBuild = false;
            available = false;
            readyToCompact = false;
            compactedAvailable = false;
        }
        ~Priv() {
            compactedSizeOnDevice.finalize();

            scene->removeGAS(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        CUcontext getCUDAContext() const {
            return scene->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        uint32_t getNumMaterialSets() const {
            return static_cast<uint32_t>(numRayTypesPerMaterialSet.size());
        }
        uint32_t getNumRayTypes(uint32_t matSetIdx) const {
            return numRayTypesPerMaterialSet[matSetIdx];
        }

        uint32_t calcNumSBTRecords(uint32_t matSetIdx) const;
        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, HitGroupSBTRecord* records) const;
        
        void markDirty();
        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            THROW_RUNTIME_ERROR(isReady(), "Traversable handle is not ready.");
            if (compactedAvailable)
                return compactedHandle;
            if (available)
                return handle;
            return 0;
        }
    };



    enum class InstanceType {
        GAS = 0,
        //MatrixMotionTransform,
        //SRTMotionTransform,
        //StaticTransform,
        Invalid
    };

    class Instance::Priv {
        _Scene* scene;
        InstanceType type;
        union {
            struct {
                _GeometryAccelerationStructure* gas;
                uint32_t matSetIndex;
            };
        };
        float transform[12];

    public:
        OPTIX_OPAQUE_BRIDGE(Instance);

        Priv(_Scene* _scene) :
            scene(_scene),
            matSetIndex(0xFFFFFFFF), type(InstanceType::Invalid), gas(nullptr) {
            float identity[] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            };
            std::copy_n(identity, 12, transform);
        }
        ~Priv() {}

        const _Scene* getScene() const {
            return scene;
        }



        void fillInstance(OptixInstance* instance) const;
        void updateInstance(OptixInstance* instance) const;
    };



    class InstanceAccelerationStructure::Priv {
        _Scene* scene;

        std::unordered_set<_Instance*> children;
        OptixBuildInput buildInput;
        std::vector<OptixInstance> instances;

        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes memoryRequirement;

        TypedBuffer<size_t> compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        const TypedBuffer<OptixInstance>* instanceBuffer;
        const Buffer* accelBuffer;
        const Buffer* compactedAccelBuffer;
        struct {
            unsigned int preferFastTrace : 1;
            unsigned int allowUpdate : 1;
            unsigned int allowCompaction : 1;
            unsigned int readyToBuild : 1;
            unsigned int available : 1;
            unsigned int readyToCompact : 1;
            unsigned int compactedAvailable : 1;
        };

    public:
        OPTIX_OPAQUE_BRIDGE(InstanceAccelerationStructure);

        Priv(_Scene* _scene) : scene(_scene) {
            scene->addIAS(this);

            compactedSizeOnDevice.initialize(scene->getCUDAContext(), s_BufferType, 1);

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getCUdeviceptr();

            preferFastTrace = true;
            allowUpdate = false;
            allowCompaction = false;

            readyToBuild = false;
            available = false;
            readyToCompact = false;
            compactedAvailable = false;
        }
        ~Priv() {
            compactedSizeOnDevice.finalize();

            scene->removeIAS(this);
        }

        const _Scene* getScene() const {
            return scene;
        }
        CUcontext getCUDAContext() const {
            return scene->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        void markDirty();
        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            THROW_RUNTIME_ERROR(isReady(), "Traversable handle is not ready.");
            if (compactedAvailable)
                return compactedHandle;
            if (available)
                return handle;
            optixAssert_ShouldNotBeCalled();
            return 0;
        }
    };



    class Pipeline::Priv {
        const _Context* context;
        OptixPipeline rawPipeline;

        uint32_t maxTraceDepth;
        OptixPipelineCompileOptions pipelineCompileOptions;
        size_t sizeOfPipelineLaunchParams;
        std::unordered_set<OptixProgramGroup> programGroups;

        _Scene* scene;
        uint32_t numMissRayTypes;

        _ProgramGroup* rayGenProgram;
        _ProgramGroup* exceptionProgram;
        std::vector<_ProgramGroup*> missPrograms;
        std::vector<_ProgramGroup*> callablePrograms;
        Buffer rayGenRecord;
        Buffer exceptionRecord;
        Buffer missRecords;
        Buffer callableRecords;

        Buffer* hitGroupSbt;
        OptixShaderBindingTable sbt;

        struct {
            unsigned int pipelineLinked : 1;
            unsigned int sbtAllocDone : 1;
            unsigned int sbtIsUpToDate : 1;
        };

        void setupShaderBindingTable();

    public:
        OPTIX_OPAQUE_BRIDGE(Pipeline);

        Priv(const _Context* ctxt) : context(ctxt),
            maxTraceDepth(0),
            scene(nullptr), numMissRayTypes(0),
            rayGenProgram(nullptr), exceptionProgram(nullptr), hitGroupSbt(nullptr),
            pipelineLinked(false), sbtAllocDone(false), sbtIsUpToDate(false) {
            rayGenRecord.initialize(context->getCUDAContext(), s_BufferType, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
            exceptionRecord.initialize(context->getCUDAContext(), s_BufferType, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
            missRecords.initialize(context->getCUDAContext(), s_BufferType, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
            callableRecords.initialize(context->getCUDAContext(), s_BufferType, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
        }
        ~Priv() {
            if (pipelineLinked)
                optixPipelineDestroy(rawPipeline);

            callableRecords.finalize();
            missRecords.finalize();
            exceptionRecord.finalize();
            rayGenRecord.finalize();
        }

        CUcontext getCUDAContext() const {
            return context->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        void createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group);
        void destroyProgram(OptixProgramGroup group);
    };



    class Module::Priv {
        const _Pipeline* pipeline;
        OptixModule rawModule;

    public:
        OPTIX_OPAQUE_BRIDGE(Module);

        Priv(const _Pipeline* pl, OptixModule _rawModule) : pipeline(pl), rawModule(_rawModule) {}



        const _Pipeline* getPipeline() const {
            return pipeline;
        }

        OptixModule getRawModule() const {
            return rawModule;
        }
    };



    class ProgramGroup::Priv {
        _Pipeline* pipeline;
        OptixProgramGroup rawGroup;

    public:
        OPTIX_OPAQUE_BRIDGE(ProgramGroup);

        Priv(_Pipeline* pl, OptixProgramGroup _rawGroup) : pipeline(pl), rawGroup(_rawGroup) {}



        const _Pipeline* getPipeline() const {
            return pipeline;
        }

        OptixProgramGroup getRawProgramGroup() const {
            return rawGroup;
        }

        void packHeader(uint8_t* record) const {
            OPTIX_CHECK(optixSbtRecordPackHeader(rawGroup, record));
        }
    };
}
