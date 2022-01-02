#pragma once

#include "renderer_common.h"

namespace vlr::shared {
    struct PTReadOnlyPayload {
        float initImportance;
        WavelengthSamples wls;
        float prevDirPDF;
        DirectionType prevSampledType;
        unsigned int pathLength : 16;
        unsigned int maxLengthTerminate : 1;
    };

    struct PTWriteOnlyPayload {
        Point3D nextOrigin;
        Vector3D nextDirection;
        float dirPDF;
        DirectionType sampledType;
        unsigned int terminate : 1;
    };

    struct PTReadWritePayload {
        KernelRNG rng;
        SampledSpectrum alpha;
        SampledSpectrum contribution;
    };

    struct PTExtraPayload {
        SampledSpectrum firstHitAlbedo;
        Normal3D firstHitNormal;
    };

    using PTPayloadSignature = optixu::PayloadSignature<
        PTReadOnlyPayload*, PTWriteOnlyPayload*, PTReadWritePayload*, PTExtraPayload*>;

    struct LTReadOnlyPayload {
        WavelengthSamples wls;
        float prevDirPDF;
        DirectionType prevSampledType;
        unsigned int pathLength : 16;
        unsigned int maxLengthTerminate : 1;
    };

    struct LTWriteOnlyPayload {
        Point3D nextOrigin;
        Vector3D nextDirection;
        float dirPDF;
        DirectionType sampledType;
        unsigned int terminate : 1;
    };

    struct LTReadWritePayload {
        KernelRNG rng;
        SampledSpectrum alpha;
    };

    using LTPayloadSignature = optixu::PayloadSignature<
        LTReadOnlyPayload*, LTWriteOnlyPayload*, LTReadWritePayload*>;

    struct LVCBPTLightPathReadOnlyPayload {
        float prevDirPDF;
        float prevCosTerm;
        float prevRevAreaPDF;
        DirectionType prevSampledType;
    };

    struct LVCBPTLightPathWriteOnlyPayload {
        Point3D nextOrigin;
        Vector3D nextDirection;
        float dirPDF;
        float cosTerm;
        float revAreaPDF;
        DirectionType sampledType;
    };

    struct LVCBPTLightPathReadWritePayload {
        KernelRNG rng;
        SampledSpectrum alpha;
        float totalPowerProbDensity;
        float prevTotalPowerProbDensity;
        float prevSumPowerProbDensities;
        unsigned int singleIsSelected : 1;
        unsigned int pathLength : 16;
        unsigned int maxLengthTerminate : 1;
        unsigned int terminate : 1;
    };

    struct LVCBPTEyePathReadOnlyPayload {
        float prevDirPDF;
        float prevCosTerm;
        float prevRevAreaPDF;
        DirectionType prevSampledType;
    };

    struct LVCBPTEyePathWriteOnlyPayload {
        Point3D nextOrigin;
        Vector3D nextDirection;
        float dirPDF;
        float cosTerm;
        float revAreaPDF;
        DirectionType sampledType;
    };

    struct LVCBPTEyePathReadWritePayload {
        KernelRNG rng;
        SampledSpectrum alpha;
        SampledSpectrum contribution;
        float totalPowerProbDensity;
        float prevTotalPowerProbDensity;
        float prevSumPowerProbDensities;
        unsigned int singleIsSelected : 1;
        unsigned int pathLength : 16;
        unsigned int maxLengthTerminate : 1;
        unsigned int terminate : 1;
    };

    struct LVCBPTEyePathExtraPayload {
        SampledSpectrum firstHitAlbedo;
        Normal3D firstHitNormal;
    };

    using LVCBPTLightPathPayloadSignature = optixu::PayloadSignature<
        LVCBPTLightPathReadOnlyPayload*, LVCBPTLightPathWriteOnlyPayload*, LVCBPTLightPathReadWritePayload*>;
    using LVCBPTEyePathPayloadSignature = optixu::PayloadSignature<
        LVCBPTEyePathReadOnlyPayload*, LVCBPTEyePathWriteOnlyPayload*, LVCBPTEyePathReadWritePayload*, LVCBPTEyePathExtraPayload*>;

    using AuxBufGenPayloadSignature = optixu::PayloadSignature<
        WavelengthSamples, KernelRNG, SampledSpectrum, Normal3D>;

    using ShadowPayloadSignature = optixu::PayloadSignature<
        WavelengthSamples, float>;



#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)

    // ----------------------------------------------------------------
    // Light

    template <typename RayType>
    CUDA_DEVICE_FUNCTION bool testVisibility(
        const SurfacePoint &shadingSurfacePoint, const SurfacePoint &lightSurfacePoint, const WavelengthSamples &wls,
        Vector3D* shadowRayDir, float* squaredDistance, float* fractionalVisibility) {
        VLRAssert(shadingSurfacePoint.atInfinity == false, "Shading point must be in finite region.");

        *shadowRayDir = lightSurfacePoint.calcDirectionFrom(shadingSurfacePoint.position, squaredDistance);

        const Normal3D &geomNormal = shadingSurfacePoint.geometricNormal;
        bool isFrontSide = dot(geomNormal, *shadowRayDir) > 0;
        Point3D shadingPoint = offsetRayOrigin(shadingSurfacePoint.position, isFrontSide ? geomNormal : -geomNormal);

        float tmax = FLT_MAX;
        if (!lightSurfacePoint.atInfinity)
            tmax = std::sqrt(*squaredDistance) * 0.9999f;

        WavelengthSamples plWls = wls;
        float plFracVis = 1.0f;
        optixu::trace<ShadowPayloadSignature>(
            plp.topGroup, asOptiXType(shadingPoint), asOptiXType(*shadowRayDir), 0.0f, tmax, 0.0f,
            VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE, RayType::Shadow, MaxNumRayTypes, RayType::Shadow,
            plWls, plFracVis);

        *fractionalVisibility = plFracVis;

        return *fractionalVisibility > 0;
    }

    CUDA_DEVICE_FUNCTION void selectSurfaceLight(
        float uLight, SurfaceLight* light, float* lightProb, float* uPrim) {
        float uInst;
        float instProb;
        uint32_t instIndex = plp.instIndices[plp.lightInstDist.sample(uLight, &instProb, &uInst)];
        const Instance &inst = plp.instBuffer[instIndex];
        float geomInstProb;
        uint32_t geomInstIndex = inst.geomInstIndices[inst.lightGeomInstDistribution.sample(uInst, &geomInstProb, uPrim)];
        //printf("%u, %g, %u, %g\n", instIndex, instProb, geomInstIndex, geomInstProb);
        const GeometryInstance &geomInst = plp.geomInstBuffer[geomInstIndex];
        *light = SurfaceLight(instIndex, geomInstIndex,
                              ProgSigSurfaceLight_sample(geomInst.progSample));
        *lightProb = instProb * geomInstProb;
    }

    // END: Light
    // ----------------------------------------------------------------



    CUDA_DEVICE_KERNEL void RT_AH_NAME(shadowAnyHitDefault)() {
        float fractionalVisibility = 0.0f;
        ShadowPayloadSignature::set(nullptr, &fractionalVisibility);
        optixTerminateRay();
    }

    // Common Any Hit Program for All Primitive Types and Materials for shadow rays
    CUDA_DEVICE_KERNEL void RT_AH_NAME(shadowAnyHitWithAlpha)() {
        WavelengthSamples wls;
        float fractionalVisibility;
        ShadowPayloadSignature::get(&wls, &fractionalVisibility);

        float alpha = getAlpha(wls);

        fractionalVisibility *= (1 - alpha);
        ShadowPayloadSignature::set(nullptr, &fractionalVisibility);
        if (fractionalVisibility == 0.0f)
            optixTerminateRay();
        else
            optixIgnoreIntersection();
    }

#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
}
