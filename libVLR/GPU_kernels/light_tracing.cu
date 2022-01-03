#include "../shared/light_transport_common.h"

namespace vlr {
    using namespace shared;

    // Common Any Hit Program for All Primitive Types and Materials for non-shadow rays
    CUDA_DEVICE_KERNEL void RT_AH_NAME(lightTracingAnyHitWithAlpha)() {
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



    CUDA_DEVICE_KERNEL void RT_RG_NAME(lightTracing)() {
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

        // Next Event Estimation (explicit lens sampling)
        EDFQuery feQuery(DirectionType::All(), wls);
        {
            Camera camera(static_cast<ProgSigCamera_sample>(plp.progSampleLensPosition));
            LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
            LensPosQueryResult We0Result;
            camera.sample(We0Sample, &We0Result);

            IDF idf(plp.cameraDescriptor, We0Result.surfPt, wls);
            SampledSpectrum We0 = idf.evaluateSpatialImportance();

            Vector3D shadowRayDir;
            float squaredDistance;
            float fractionalVisibility;
            if (We0.hasNonZero() &&
                testVisibility<LTRayType::Shadow>(
                    We0Result.surfPt, Le0Result.surfPt,
                    wls, &shadowRayDir, &squaredDistance, &fractionalVisibility)) {
                Vector3D shadowRayDir_lens = We0Result.surfPt.toLocal(shadowRayDir);
                Vector3D shadowRayDir_sn = Le0Result.surfPt.toLocal(-shadowRayDir);

                IDFQuery fiQuery;
                SampledSpectrum We = We0 * idf.evaluateDirectionalImportance(fiQuery, shadowRayDir_lens);
                float2 posInScreen = idf.backProjectDirection(fiQuery, shadowRayDir_lens);
                float2 pixel = make_float2(posInScreen.x * plp.imageSize.x, posInScreen.y * plp.imageSize.y);
                float lensPDF = We0Result.areaPDF;

                SampledSpectrum Le1 = edf.evaluate(feQuery, shadowRayDir_sn);

                float cosLens = We0Result.surfPt.calcCosTerm(shadowRayDir);
                float cosLight = Le0Result.surfPt.calcCosTerm(-shadowRayDir);
                float G = fractionalVisibility * cosLight * cosLens / squaredDistance;
                float scalarCoeff = G / lensPDF;
                SampledSpectrum contribution = alpha * We * Le1 * scalarCoeff;
                atomicAddToBuffer(wls, contribution, pixel);
            }
        }

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

        float initImportance = alpha.importance(wls.selectedLambdaIndex());

        LTReadOnlyPayload roPayload = {};
        roPayload.wls = wls;
        roPayload.prevDirPDF = Le1Result.dirPDF;
        roPayload.prevSampledType = Le1Result.sampledType;
        roPayload.pathLength = 0;
        roPayload.maxLengthTerminate = false;
        LTWriteOnlyPayload woPayload = {};
        LTReadWritePayload rwPayload = {};
        rwPayload.rng = rng;
        rwPayload.alpha = alpha;
        LTReadOnlyPayload* roPayloadPtr = &roPayload;
        LTWriteOnlyPayload* woPayloadPtr = &woPayload;
        LTReadWritePayload* rwPayloadPtr = &rwPayload;

        const uint32_t MaxPathLength = 25;
        while (true) {
            woPayload.terminate = true;
            ++roPayload.pathLength;
            if (roPayload.pathLength >= MaxPathLength)
                break;

            optixu::trace<LTPayloadSignature>(
                plp.topGroup, asOptiXType(rayOrg), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
                shared::VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE,
                LTRayType::Closest, MaxNumRayTypes, LTRayType::Closest,
                roPayloadPtr, woPayloadPtr, rwPayloadPtr);

            if (woPayload.terminate)
                break;
            VLRAssert(roPayload.pathLength < MaxPathLength, "Path should be terminated... Something went wrong...");

            // Russian roulette
            float continueProb = std::fmin(rwPayload.alpha.importance(wls.selectedLambdaIndex()) / initImportance, 1.0f);
            if (rng.getFloat0cTo1o() >= continueProb)
                break;
            rwPayload.alpha /= continueProb;

            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
            roPayload.prevDirPDF = woPayload.dirPDF;
            roPayload.prevSampledType = woPayload.sampledType;
        }
        plp.linearRngBuffer[launchIndex] = rwPayload.rng;
    }



    // Common Closest Hit Program for All Primitive Types and Materials
    CUDA_DEVICE_KERNEL void RT_CH_NAME(lightTracingIteration)() {
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

        // Next Event Estimation (explicit lens sampling)
        if (bsdf.hasNonDelta()) {
            Camera camera(static_cast<ProgSigCamera_sample>(plp.progSampleLensPosition));
            LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
            LensPosQueryResult We0Result;
            camera.sample(We0Sample, &We0Result);

            IDF idf(plp.cameraDescriptor, We0Result.surfPt, wls);
            SampledSpectrum We0 = idf.evaluateSpatialImportance();

            Vector3D shadowRayDir;
            float squaredDistance;
            float fractionalVisibility;
            if (We0.hasNonZero() &&
                testVisibility<LTRayType::Shadow>(
                    We0Result.surfPt, surfPt, wls, &shadowRayDir, &squaredDistance,
                    &fractionalVisibility)) {
                Vector3D shadowRayDir_lens = We0Result.surfPt.toLocal(shadowRayDir);
                Vector3D shadowRayDir_sn = surfPt.toLocal(-shadowRayDir);

                IDFQuery fiQuery;
                SampledSpectrum We = We0 * idf.evaluateDirectionalImportance(fiQuery, shadowRayDir_lens);
                float2 posInScreen = idf.backProjectDirection(fiQuery, shadowRayDir_lens);
                float2 pixel = make_float2(posInScreen.x * plp.imageSize.x, posInScreen.y * plp.imageSize.y);
                float lensPDF = We0Result.areaPDF;

                SampledSpectrum fs = bsdf.evaluate(fsQuery, shadowRayDir_sn);
                float cosLens = We0Result.surfPt.calcCosTerm(shadowRayDir);

                float G = fractionalVisibility * absDot(shadowRayDir_sn, geomNormalLocal) * cosLens / squaredDistance;
                float scalarCoeff = G / lensPDF; // 直接contributionの計算式に入れるとCUDAのバグなのかおかしな結果になる。
                SampledSpectrum contribution = rwPayload->alpha * We * fs * scalarCoeff;
                atomicAddToBuffer(wls, contribution, pixel);
            }
        }

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
        rwPayload->alpha *= fs * (std::fabs(cosFactor) / fsResult.dirPDF);

        Vector3D dirOut = surfPt.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPt.position, cosFactor > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        woPayload->nextDirection = dirOut;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->sampledType = fsResult.sampledType;
        woPayload->terminate = false;
    }
}
