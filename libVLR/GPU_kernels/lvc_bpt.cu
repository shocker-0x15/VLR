#include "../shared/light_transport_common.h"

namespace vlr {
    using namespace shared;

    CUDA_DEVICE_KERNEL void RT_AH_NAME(lvcbptAnyHitWithAlpha)() {
        LTReadOnlyPayload* roPayload;
        LTReadWritePayload* rwPayload;
        LTPayloadSignature::get(&roPayload, nullptr, &rwPayload);

        float alpha = getAlpha(roPayload->wls);

        // Stochastic Alpha Test
        if (rwPayload->rng.getFloat0cTo1o() >= alpha)
            optixIgnoreIntersection();
    }

    CUDA_DEVICE_FUNCTION void atomicAddToBuffer(
        const WavelengthSamples &wls, const SampledSpectrum &contribution,
        const float2 &pixel) {
        uint32_t ipx = static_cast<uint32_t>(pixel.x);
        uint32_t ipy = static_cast<uint32_t>(pixel.y);
        if (ipx < plp.imageSize.x && ipy < plp.imageSize.y) {
            if (!contribution.allFinite()) {
                vlrprintf("Pass %u, (%u - %u, %u): Not a finite value.\n",
                          plp.numAccumFrames, optixGetLaunchIndex().x, ipx, ipy);
                return;
            }
            plp.atomicAccumBuffer[ipy * plp.imageStrideInPixels + ipx].atomicAdd(wls, contribution);
        }
    }

    CUDA_DEVICE_FUNCTION void storeLightVertex(
        const SampledSpectrum &flux, const Vector3D &dirIn, DirectionType sampledType, const SurfacePoint &surfPt) {
        LightPathVertex lightVertex = {};
        lightVertex.flux = flux;
        lightVertex.dirIn = dirIn;
        lightVertex.sampledType = sampledType;
        lightVertex.instIndex = surfPt.instIndex;
        lightVertex.geomInstIndex = surfPt.geomInstIndex;
        lightVertex.primIndex = surfPt.primIndex;
        lightVertex.u = surfPt.u;
        lightVertex.v = surfPt.v;
        uint32_t cacheIndex = atomicAdd(plp.numLightVertices, 1u);
        plp.lightVertexCache[cacheIndex] = lightVertex;
    }



    CUDA_DEVICE_KERNEL void RT_RG_NAME(lvcbptLightPath)() {
        uint32_t launchIndex = optixGetLaunchIndex().x;

        KernelRNG rng = plp.linearRngBuffer[launchIndex];

        float selectWLPDF;
        WavelengthSamples wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &selectWLPDF);

        float uLight = rng.getFloat0cTo1o();
        SurfaceLight light;
        float lightProb;
        float uPrim;
        selectSurfaceLight(uLight, &light, &lightProb, &uPrim);

        SurfaceLightPosSample Le0Sample(uPrim, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        SurfaceLightPosQueryResult Le0Result;
        light.sample(Le0Sample, Point3D(NAN), &Le0Result);

        const SurfaceMaterialDescriptor lightMatDesc = plp.materialDescriptorBuffer[Le0Result.materialIndex];
        EDF edf(lightMatDesc, Le0Result.surfPt, wls);

        SampledSpectrum Le0 = edf.evaluateEmittance();
        SampledSpectrum alpha = Le0 / (plp.numLightPaths * lightProb * Le0Result.areaPDF * selectWLPDF);
        alpha *= (plp.imageSize.x * plp.imageSize.y); // TODO: マテリアル側を修正してこの補正項を無くす。

        storeLightVertex(alpha, Vector3D(NAN, NAN, NAN), Le0Result.posType, Le0Result.surfPt);

        EDFQuery feQuery(DirectionType::All(), wls);
        EDFSample Le1Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        EDFQueryResult Le1Result;
        SampledSpectrum Le1 = edf.sample(feQuery, Le1Sample, &Le1Result);

        Point3D rayOrg = offsetRayOrigin(Le0Result.surfPt.position, Le0Result.surfPt.geometricNormal);
        if (Le0Result.surfPt.atInfinity) {
            rayOrg = plp.sceneBounds->center +
                1.1f * plp.sceneBounds->worldRadius * Le0Result.surfPt.position +
                Le0Result.surfPt.shadingFrame.x * Le1Result.dirLocal.x +
                Le0Result.surfPt.shadingFrame.y * Le1Result.dirLocal.y;
            Le1Result.dirLocal.x = 0;
            Le1Result.dirLocal.y = 0;
        }
        Vector3D rayDir = Le0Result.surfPt.fromLocal(Le1Result.dirLocal);
        alpha *= Le1 * (Le0Result.surfPt.calcCosTerm(rayDir) / Le1Result.dirPDF);

        LVCBPTLightPathReadOnlyPayload roPayload = {};
        roPayload.wls = wls;
        roPayload.prevDirPDF = Le1Result.dirPDF;
        roPayload.prevSampledType = Le1Result.sampledType;
        roPayload.pathLength = 0;
        roPayload.maxLengthTerminate = false;
        LVCBPTLightPathWriteOnlyPayload woPayload = {};
        LVCBPTLightPathReadWritePayload rwPayload = {};
        rwPayload.rng = rng;
        rwPayload.alpha = alpha;
        LVCBPTLightPathReadOnlyPayload* roPayloadPtr = &roPayload;
        LVCBPTLightPathWriteOnlyPayload* woPayloadPtr = &woPayload;
        LVCBPTLightPathReadWritePayload* rwPayloadPtr = &rwPayload;

        const uint32_t MaxPathLength = 25;
        while (true) {
            woPayload.terminate = true;
            ++roPayload.pathLength;
            if (roPayload.pathLength >= MaxPathLength)
                break;

            optixu::trace<LVCBPTLightPathPayloadSignature>(
                plp.topGroup, asOptiXType(rayOrg), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
                shared::VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE,
                LVCBPTRayType::LightPath, MaxNumRayTypes, LVCBPTRayType::LightPath,
                roPayloadPtr, woPayloadPtr, rwPayloadPtr);

            if (woPayload.terminate)
                break;
            VLRAssert(roPayload.pathLength < MaxPathLength, "Path should be terminated... Something went wrong...");

            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
            roPayload.prevDirPDF = woPayload.dirPDF;
            roPayload.prevSampledType = woPayload.sampledType;
        }
        plp.linearRngBuffer[launchIndex] = rwPayload.rng;
    }



    CUDA_DEVICE_KERNEL void RT_CH_NAME(lvcbptLightPath)() {
        const auto hp = HitPointParameter::get();

        LTReadOnlyPayload* roPayload;
        LTWriteOnlyPayload* woPayload;
        LTReadWritePayload* rwPayload;
        LTPayloadSignature::get(&roPayload, &woPayload, &rwPayload);

        KernelRNG &rng = rwPayload->rng;
        WavelengthSamples &wls = roPayload->wls;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(hp, wls, &surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[hp.sbtr->geomInst.materialIndex];
        constexpr TransportMode transportMode = TransportMode::Importance;
        BSDF<transportMode> bsdf(matDesc, surfPt, wls);

        Vector3D dirInLocal = surfPt.shadingFrame.toLocal(-asVector3D(optixGetWorldRayDirection()));

        Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        BSDFQuery fsQuery(dirInLocal, geomNormalLocal, transportMode, DirectionType::All(), wls);

        storeLightVertex(rwPayload->alpha, -asVector3D(optixGetWorldRayDirection()),
                         roPayload->prevSampledType, surfPt);

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        SampledSpectrum fs = bsdf.sample(fsQuery, sample, &fsResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !wls.singleIsSelected()) {
            fsResult.dirPDF /= SampledSpectrum::NumComponents();
            wls.setSingleIsSelected();
        }

        float cosFactor = dot(fsResult.dirLocal, geomNormalLocal);
        SampledSpectrum throughput = fs * (std::fabs(cosFactor) / fsResult.dirPDF);
        rwPayload->alpha *= throughput;

        // Russian roulette
        float continueProb = std::fmin(throughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        rwPayload->alpha /= continueProb;

        Vector3D dirOut = surfPt.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPt.position, cosFactor > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        woPayload->nextDirection = dirOut;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->sampledType = fsResult.sampledType;
        woPayload->terminate = false;
    }
}
