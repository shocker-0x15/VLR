#define RT_PIPELINE_LAUNCH_PARAMETERS extern "C" __constant__
#include "../shared/shared.h"

namespace vlr {
    using namespace shared;

    CUDA_DEVICE_KERNEL void computeInstanceAABBs(
        const uint32_t* instIndices, const uint32_t* itemOffsets,
        Instance* instances, const GeometryInstance* geomInsts, uint32_t numItems) {
        uint32_t globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if (globalIndex >= numItems)
            return;

        uint32_t instIndex = instIndices[globalIndex];
        Instance &inst = instances[instIndex];
        VLRAssert(inst.isActive, "This instance is inactive and should not be assigned to a thread.");
        uint32_t geomInstIndexInInst = globalIndex - itemOffsets[globalIndex];
        uint32_t geomInstIndex = inst.geomInstIndices[geomInstIndexInInst];
        const GeometryInstance &geomInst = geomInsts[geomInstIndex];
        if (geomInst.geomType != GeometryType_TriangleMesh)
            return;
        //BoundingBox3D aabb = geomInst.asTriMesh.aabb;
        //printf("%u: Inst %u-%u: GeomInst %u: (%g, %g, %g) - (%g, %g, %g)\n",
        //       globalIndex, instIndex, geomInstIndexInInst, geomInstIndex,
        //       aabb.minP.x, aabb.minP.y, aabb.minP.z,
        //       aabb.maxP.x, aabb.maxP.y, aabb.maxP.z);
        BoundingBox3DAsOrderedInt geomInstAabbAsInt(geomInst.asTriMesh.aabb);

        auto &instAabbAsInt = reinterpret_cast<BoundingBox3DAsOrderedInt &>(inst.childAabb);
        atomicUnifyBoundingBox3D(&instAabbAsInt, geomInstAabbAsInt);
    }

    CUDA_DEVICE_KERNEL void finalizeInstanceAABBs(Instance* instances, uint32_t numInstances) {
        uint32_t globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if (globalIndex >= numInstances)
            return;

        uint32_t instIndex = globalIndex;
        Instance &inst = instances[instIndex];
        if (inst.isActive && inst.aabbIsDirty) {
            auto aabbAsInt = reinterpret_cast<BoundingBox3DAsOrderedInt &>(inst.childAabb);
            inst.childAabb = static_cast<BoundingBox3D>(aabbAsInt);
            inst.aabbIsDirty = false;
            //printf("Inst %u: (%g, %g, %g) - (%g, %g, %g)\n",
            //       instIndex,
            //       inst.childAabb.minP.x, inst.childAabb.minP.y, inst.childAabb.minP.z,
            //       inst.childAabb.maxP.x, inst.childAabb.maxP.y, inst.childAabb.maxP.z);
        }
    }

    CUDA_DEVICE_KERNEL void computeSceneAABB(const Instance* instances, uint32_t numInstances,
                                             SceneBounds* sceneBounds) {
        uint32_t globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if (globalIndex >= numInstances)
            return;

        CUDA_SHARED_MEM uint32_t b_mem[sizeof(BoundingBox3DAsOrderedInt) / 4];
        auto &b_AabbAsInt = reinterpret_cast<BoundingBox3DAsOrderedInt &>(b_mem);
        if (threadIdx.x == 0)
            b_AabbAsInt = BoundingBox3DAsOrderedInt();

        uint32_t instIndex = globalIndex;
        const Instance &inst = instances[instIndex];
        BoundingBox3DAsOrderedInt aabbAsInt;
        if (inst.isActive) {
            //printf("Pre Xfm Inst %u: (%g, %g, %g) - (%g, %g, %g)\n",
            //       instIndex,
            //       inst.childAabb.minP.x, inst.childAabb.minP.y, inst.childAabb.minP.z,
            //       inst.childAabb.maxP.x, inst.childAabb.maxP.y, inst.childAabb.maxP.z);
            BoundingBox3D aabb = inst.transform * inst.childAabb;
            aabbAsInt = aabb;
        }

        __syncthreads();
        atomicUnifyBoundingBox3D_block(&b_AabbAsInt, aabbAsInt);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicUnifyBoundingBox3D(&sceneBounds->aabbAsInt, b_AabbAsInt);
        }
    }

    CUDA_DEVICE_KERNEL void finalizeSceneBounds(SceneBounds* sceneBounds) {
        uint32_t globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
        if (globalIndex >= 1)
            return;

        BoundingBox3D sceneAabb = static_cast<BoundingBox3D>(sceneBounds->aabbAsInt);
        sceneBounds->aabb = sceneAabb;
        Point3D worldCenter = sceneAabb.centroid();
        sceneBounds->center = worldCenter;
        float worldRadius = (sceneAabb.maxP - worldCenter).length();
        sceneBounds->worldDiscArea = VLR_M_PI * pow2(worldRadius);
    }
}
