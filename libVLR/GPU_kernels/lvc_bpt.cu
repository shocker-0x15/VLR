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
        const WavelengthSamples &wls, SampledSpectrum contribution,
        const float2 &pixel) {
        uint32_t ipx = static_cast<uint32_t>(pixel.x);
        uint32_t ipy = static_cast<uint32_t>(pixel.y);
        if (ipx < plp.imageSize.x && ipy < plp.imageSize.y) {
            if (!contribution.allFinite()) {
                vlrprintf("Pass %u, (%u - %u, %u): Not a finite value.\n",
                          plp.numAccumFrames, optixGetLaunchIndex().x, ipx, ipy);
                return;
            }
            contribution *= (plp.imageSize.x * plp.imageSize.y); // TODO: マテリアル側を修正してこの補正項を無くす。
            plp.atomicAccumBuffer[ipy * plp.imageStrideInPixels + ipx].atomicAdd(wls, contribution);
        }
    }

    CUDA_DEVICE_FUNCTION void storeLightVertex(
        float powerProbDensity, float prevPowerProbDensity, float prevSumPowerProbDensities,
        const SampledSpectrum &flux, const Vector3D &dirIn, DirectionType sampledType, bool wlSelected,
        const SurfacePoint &surfPt) {
        LightPathVertex lightVertex = {};
        lightVertex.instIndex = surfPt.instIndex;
        lightVertex.geomInstIndex = surfPt.geomInstIndex;
        lightVertex.primIndex = surfPt.primIndex;
        lightVertex.u = surfPt.u;
        lightVertex.v = surfPt.v;
        lightVertex.powerProbDensity = powerProbDensity;
        lightVertex.prevPowerProbDensity = prevPowerProbDensity;
        lightVertex.prevSumPowerProbDensities = prevSumPowerProbDensities;
        lightVertex.flux = flux;
        lightVertex.dirIn = dirIn;
        lightVertex.sampledType = sampledType;
        lightVertex.wlSelected = wlSelected;
        uint32_t cacheIndex = atomicAdd(plp.numLightVertices, 1u);
        plp.lightVertexCache[cacheIndex] = lightVertex;
    }



    CUDA_DEVICE_KERNEL void RT_RG_NAME(lvcbptLightPath)() {
        uint32_t launchIndex = optixGetLaunchIndex().x;

        KernelRNG rng = plp.linearRngBuffer[launchIndex];

        float uLight = rng.getFloat0cTo1o();
        SurfaceLight light;
        float lightProb;
        float uPrim;
        selectSurfaceLight(uLight, &light, &lightProb, &uPrim);

        SurfaceLightPosSample Le0Sample(uPrim, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        SurfaceLightPosQueryResult Le0Result;
        light.sample(Le0Sample, Point3D(NAN), &Le0Result);

        const SurfaceMaterialDescriptor lightMatDesc = plp.materialDescriptorBuffer[Le0Result.materialIndex];
        EDF edf(lightMatDesc, Le0Result.surfPt, plp.commonWavelengthSamples);

        float probDensity0 = plp.numLightPaths * lightProb * Le0Result.areaPDF;
        SampledSpectrum Le0 = edf.evaluateEmittance();
        SampledSpectrum alpha = Le0 / probDensity0;

        float powerProbDensities0 = pow2(probDensity0);
        float prevPowerProbDensity0 = 1;
        float prevSumPowerProbDensities0 = 0;
        storeLightVertex(powerProbDensities0, prevPowerProbDensity0, prevSumPowerProbDensities0,
                         alpha, Vector3D(NAN, NAN, NAN), Le0Result.posType, false,
                         Le0Result.surfPt);

        EDFQuery feQuery(DirectionType::All(), plp.commonWavelengthSamples);
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
        float cosTerm = Le0Result.surfPt.calcCosTerm(rayDir);
        alpha *= Le1 * (cosTerm / Le1Result.dirPDF);

        LVCBPTLightPathReadOnlyPayload roPayload = {};
        roPayload.prevDirPDF = Le1Result.dirPDF;
        roPayload.prevCosTerm = cosTerm;
        roPayload.prevRevAreaPDF = 0;
        roPayload.prevSampledType = Le1Result.sampledType;
        LVCBPTLightPathWriteOnlyPayload woPayload = {};
        LVCBPTLightPathReadWritePayload rwPayload = {};
        rwPayload.rng = rng;
        rwPayload.alpha = alpha;
        rwPayload.totalPowerProbDensity = powerProbDensities0;
        rwPayload.prevTotalPowerProbDensity = prevPowerProbDensity0;
        rwPayload.prevSumPowerProbDensities = prevSumPowerProbDensities0;
        rwPayload.singleIsSelected = false;
        rwPayload.pathLength = 0;
        rwPayload.maxLengthTerminate = false;
        LVCBPTLightPathReadOnlyPayload* roPayloadPtr = &roPayload;
        LVCBPTLightPathWriteOnlyPayload* woPayloadPtr = &woPayload;
        LVCBPTLightPathReadWritePayload* rwPayloadPtr = &rwPayload;

        const uint32_t MaxPathLength = 25;
        while (true) {
            rwPayload.terminate = true;
            ++rwPayload.pathLength;
            if (rwPayload.pathLength >= MaxPathLength)
                break;

            optixu::trace<LVCBPTLightPathPayloadSignature>(
                plp.topGroup, asOptiXType(rayOrg), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
                shared::VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE,
                LVCBPTRayType::LightPath, MaxNumRayTypes, LVCBPTRayType::LightPath,
                roPayloadPtr, woPayloadPtr, rwPayloadPtr);

            if (rwPayload.terminate)
                break;
            VLRAssert(rwPayload.pathLength < MaxPathLength, "Path should be terminated... Something went wrong...");

            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
            roPayload.prevDirPDF = woPayload.dirPDF;
            roPayload.prevCosTerm = woPayload.cosTerm;
            roPayload.prevRevAreaPDF = woPayload.revAreaPDF;
            roPayload.prevSampledType = woPayload.sampledType;
        }
        plp.linearRngBuffer[launchIndex] = rwPayload.rng;
    }



    CUDA_DEVICE_KERNEL void RT_CH_NAME(lvcbptLightPath)() {
        const auto hp = HitPointParameter::get();

        LVCBPTLightPathReadOnlyPayload* roPayload;
        LVCBPTLightPathWriteOnlyPayload* woPayload;
        LVCBPTLightPathReadWritePayload* rwPayload;
        LVCBPTLightPathPayloadSignature::get(&roPayload, &woPayload, &rwPayload);

        KernelRNG &rng = rwPayload->rng;
        WavelengthSamples wls = plp.commonWavelengthSamples;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(hp, wls, &surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[hp.sbtr->geomInst.materialIndex];
        constexpr TransportMode transportMode = TransportMode::Importance;
        BSDF<transportMode, BSDFTier::Bidirectional> bsdf(matDesc, surfPt, wls);

        Vector3D dirIn = -asVector3D(optixGetWorldRayDirection());
        Vector3D dirInLocal = surfPt.shadingFrame.toLocal(dirIn);

        Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        BSDFQuery fsQuery(dirInLocal, geomNormalLocal, transportMode, DirectionType::All(), wls);

        rwPayload->prevSumPowerProbDensities =
            rwPayload->prevTotalPowerProbDensity +
            roPayload->prevRevAreaPDF * rwPayload->prevSumPowerProbDensities;
        rwPayload->prevTotalPowerProbDensity = rwPayload->totalPowerProbDensity;

        float lastDist2 = surfPt.atInfinity ?
            1.0f :
            sqDistance(asPoint3D(optixGetWorldRayOrigin()), surfPt.position);
        float probDensity = roPayload->prevDirPDF * absDot(dirInLocal, geomNormalLocal) / lastDist2;
        rwPayload->totalPowerProbDensity *= pow2(probDensity);

        storeLightVertex(rwPayload->totalPowerProbDensity,
                         rwPayload->prevTotalPowerProbDensity, rwPayload->prevSumPowerProbDensities,
                         rwPayload->alpha, dirIn,
                         roPayload->prevSampledType, rwPayload->singleIsSelected, surfPt);

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        BSDFQueryReverseResult fsRevResult;
        SampledSpectrum fs = bsdf.sample(fsQuery, sample, &fsResult, &fsRevResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !wls.singleIsSelected())
            rwPayload->singleIsSelected = true;

        float cosTerm = dot(fsResult.dirLocal, geomNormalLocal);
        SampledSpectrum throughput = fs * (std::fabs(cosTerm) / fsResult.dirPDF);
        rwPayload->alpha *= throughput;

        // Russian roulette
        float continueProb = std::fmin(throughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        rwPayload->alpha /= continueProb;
        rwPayload->terminate = false;

        Vector3D dirOut = surfPt.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPt.position, cosTerm > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        woPayload->nextDirection = dirOut;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->cosTerm = std::fabs(cosTerm);
        woPayload->revAreaPDF = fsRevResult.dirPDF * roPayload->prevCosTerm / lastDist2;
        woPayload->sampledType = fsResult.sampledType;
    }



    CUDA_DEVICE_KERNEL void RT_RG_NAME(lvcbptEyePath)() {
        uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

        KernelRNG rng = plp.rngBuffer.read(launchIndex);

        float2 p = make_float2(launchIndex.x + rng.getFloat0cTo1o(),
                               launchIndex.y + rng.getFloat0cTo1o());

        WavelengthSamples wls = plp.commonWavelengthSamples;

        Camera camera(static_cast<ProgSigCamera_sample>(plp.progSampleLensPosition));
        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        camera.sample(We0Sample, &We0Result);

        IDF idf(plp.cameraDescriptor, We0Result.surfPt, wls);

        float probDensity0 = We0Result.areaPDF;
        SampledSpectrum We0 = idf.evaluateSpatialImportance();
        SampledSpectrum alpha = We0 / We0Result.areaPDF;

        float powerProbDensities0 = pow2(probDensity0);
        float prevPowerProbDensity0 = 1;
        float prevSumPowerProbDensities0 = 0;
        //storeLightVertex(powerProbDensities0, prevPowerProbDensity0, prevSumPowerProbDensities0,
        //                 alpha, Vector3D(NAN, NAN, NAN), Le0Result.posType, false,
        //                 Le0Result.surfPt);

        IDFSample We1Sample(p.x / plp.imageSize.x, p.y / plp.imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = idf.sample(IDFQuery(), We1Sample, &We1Result);

        Point3D rayOrg = We0Result.surfPt.position;
        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        float cosTerm = We0Result.surfPt.calcCosTerm(rayDir);
        alpha *= We1 * (cosTerm / We1Result.dirPDF);

        LVCBPTEyePathReadOnlyPayload roPayload = {};
        roPayload.prevDirPDF = We1Result.dirPDF;
        roPayload.prevCosTerm = cosTerm;
        roPayload.prevRevAreaPDF = 0;
        roPayload.prevSampledType = We1Result.sampledType;
        LVCBPTEyePathWriteOnlyPayload woPayload = {};
        LVCBPTEyePathReadWritePayload rwPayload = {};
        rwPayload.rng = rng;
        rwPayload.alpha = alpha;
        rwPayload.contribution = SampledSpectrum::Zero();
        rwPayload.totalPowerProbDensity = powerProbDensities0;
        rwPayload.prevTotalPowerProbDensity = prevPowerProbDensity0;
        rwPayload.prevSumPowerProbDensities = prevSumPowerProbDensities0;
        rwPayload.singleIsSelected = false;
        rwPayload.pathLength = 0;
        rwPayload.maxLengthTerminate = false;
        LVCBPTEyePathExtraPayload exPayload = {};
        LVCBPTEyePathReadOnlyPayload* roPayloadPtr = &roPayload;
        LVCBPTEyePathWriteOnlyPayload* woPayloadPtr = &woPayload;
        LVCBPTEyePathReadWritePayload* rwPayloadPtr = &rwPayload;
        LVCBPTEyePathExtraPayload* exPayloadPtr = &exPayload;

        const uint32_t MaxPathLength = 25;
        while (true) {
            rwPayload.terminate = true;
            ++rwPayload.pathLength;
            if (rwPayload.pathLength >= MaxPathLength)
                break;

            optixu::trace<LVCBPTEyePathPayloadSignature>(
                plp.topGroup, asOptiXType(rayOrg), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
                shared::VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE,
                LVCBPTRayType::EyePath, MaxNumRayTypes, LVCBPTRayType::EyePath,
                roPayloadPtr, woPayloadPtr, rwPayloadPtr, exPayloadPtr);

            if (rwPayload.pathLength == 1) {
                uint32_t linearIndex = launchIndex.y * plp.imageStrideInPixels + launchIndex.x;
                DiscretizedSpectrum &accumAlbedo = plp.accumAlbedoBuffer[linearIndex];
                Normal3D &accumNormal = plp.accumNormalBuffer[linearIndex];
                if (plp.numAccumFrames == 1) {
                    accumAlbedo = DiscretizedSpectrum::Zero();
                    accumNormal = Normal3D(0.0f, 0.0f, 0.0f);
                }
                TripletSpectrum whitePoint = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                                                   1, 1, 1);
                accumAlbedo += DiscretizedSpectrum(wls, exPayload.firstHitAlbedo * whitePoint.evaluate(wls) / plp.wavelengthProbability);
                accumNormal += exPayload.firstHitNormal;
                exPayloadPtr = nullptr;
            }
        }
        plp.rngBuffer.write(launchIndex, rwPayload.rng);
        if (!rwPayload.contribution.allFinite()) {
            vlrprintf("Pass %u, (%u, %u): Not a finite value.\n", plp.numAccumFrames, launchIndex.x, launchIndex.y);
            return;
        }

        if (plp.numAccumFrames == 1)
            plp.accumBuffer[launchIndex].reset();
        plp.accumBuffer[launchIndex].add(wls, rwPayload.contribution);
    }



    CUDA_DEVICE_KERNEL void RT_CH_NAME(lvcbptEyePath)() {
        const auto hp = HitPointParameter::get();

        LVCBPTEyePathReadOnlyPayload* roPayload;
        LVCBPTEyePathWriteOnlyPayload* woPayload;
        LVCBPTEyePathReadWritePayload* rwPayload;
        LVCBPTEyePathExtraPayload* exPayload;
        LVCBPTEyePathPayloadSignature::get(&roPayload, &woPayload, &rwPayload, &exPayload);

        KernelRNG &rng = rwPayload->rng;
        WavelengthSamples wls = plp.commonWavelengthSamples;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(hp, wls, &surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[hp.sbtr->geomInst.materialIndex];
        constexpr TransportMode transportMode = TransportMode::Radiance;
        BSDF<transportMode, BSDFTier::Bidirectional> bsdf(matDesc, surfPt, wls);
        EDF edf(matDesc, surfPt, wls);

        if (exPayload) {
            exPayload->firstHitAlbedo = bsdf.getBaseColor();
            exPayload->firstHitNormal = surfPt.shadingFrame.z;
        }

        Vector3D dirOut = -asVector3D(optixGetWorldRayDirection());
        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(dirOut);

        // implicit light sampling
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            EDFQuery feQuery(DirectionType::All(), wls);
            SampledSpectrum Le = spEmittance * edf.evaluate(feQuery, dirOutLocal);

            //float MISWeight = 1.0f;
            //if (!roPayload->prevSampledType.isDelta() && roPayload->pathLength > 1) {
            //    const Instance &inst = plp.instBuffer[surfPt.instIndex];
            //    float instProb = inst.lightGeomInstDistribution.integral() / plp.lightInstDist.integral();
            //    float geomInstProb = hp.sbtr->geomInst.importance / inst.lightGeomInstDistribution.integral();

            //    float bsdfPDF = roPayload->prevDirPDF;
            //    float dist2 = surfPt.calcSquaredDistance(asPoint3D(optixGetWorldRayOrigin()));
            //    float lightPDF = instProb * geomInstProb * hypAreaPDF * dist2 / std::fabs(dirOutLocal.z);
            //    MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            //}

            //rwPayload->contribution += rwPayload->alpha * Le * MISWeight;
        }

        Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        BSDFQuery fsQuery(dirOutLocal, geomNormalLocal, transportMode, DirectionType::All(), wls);

        rwPayload->prevSumPowerProbDensities =
            rwPayload->prevTotalPowerProbDensity +
            roPayload->prevRevAreaPDF * rwPayload->prevSumPowerProbDensities;
        rwPayload->prevTotalPowerProbDensity = rwPayload->totalPowerProbDensity;

        float lastDist2 = sqDistance(asPoint3D(optixGetWorldRayOrigin()), surfPt.position);
        float probDensity = roPayload->prevDirPDF * absDot(dirOutLocal, geomNormalLocal) / lastDist2;
        rwPayload->totalPowerProbDensity *= pow2(probDensity);

        {
            uint32_t lightVertexIndex = vlr::min<uint32_t>(
                *plp.numLightVertices * rng.getFloat0cTo1o(),
                *plp.numLightVertices - 1);
            const LightPathVertex &vertex = plp.lightVertexCache[lightVertexIndex];

            SurfacePoint lightPt;
            {
                ProgSigDecodeHitPoint decodeHitPoint(vertex.geomInstIndex);
                decodeHitPoint(vertex.instIndex, vertex.geomInstIndex, vertex.primIndex,
                               vertex.u, vertex.v, &lightPt);

                const GeometryInstance &geomInst = plp.geomInstBuffer[vertex.geomInstIndex];
                Normal3D localNormal = calcNode(geomInst.nodeNormal, Normal3D(0.0f, 0.0f, 1.0f), lightPt, wls);
                applyBumpMapping(localNormal, &lightPt);

                Vector3D newTangent = calcNode(geomInst.nodeTangent, lightPt.shadingFrame.x, lightPt, wls);
                modifyTangent(newTangent, &lightPt);
            }

            SampledSpectrum unweightedContribution = rwPayload->contribution;
        }

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        BSDFQueryReverseResult fsRevResult;
        SampledSpectrum fs = bsdf.sample(fsQuery, sample, &fsResult, &fsRevResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !wls.singleIsSelected())
            rwPayload->singleIsSelected = true;

        float cosTerm = dot(fsResult.dirLocal, geomNormalLocal);
        SampledSpectrum throughput = fs * (std::fabs(cosTerm) / fsResult.dirPDF);
        rwPayload->alpha *= throughput;

        // Russian roulette
        float continueProb = std::fmin(throughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        rwPayload->alpha /= continueProb;
        rwPayload->terminate = false;

        Vector3D dirIn = surfPt.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPt.position, cosTerm > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        woPayload->nextDirection = dirIn;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->cosTerm = std::fabs(cosTerm);
        woPayload->revAreaPDF = fsRevResult.dirPDF * roPayload->prevCosTerm / lastDist2;
        woPayload->sampledType = fsResult.sampledType;
    }



    CUDA_DEVICE_KERNEL void RT_MS_NAME(lvcbptEyePath)() {

    }
}
