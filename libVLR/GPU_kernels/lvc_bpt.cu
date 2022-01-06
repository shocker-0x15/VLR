#include "../shared/light_transport_common.h"

// Reference
// Progressive Light Transport Simulation on the GPU: Survey and Improvements

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
            plp.atomicAccumBuffer[ipy * plp.imageStrideInPixels + ipx].atomicAdd(wls, contribution);
        }
    }

    CUDA_DEVICE_FUNCTION void storeLightVertex(
        float totalPowerProbDensity, float prevTotalPowerProbDensity, float prevSumPowerProbDensities,
        float backwardConversionFactor,
        const SampledSpectrum &flux, const Vector3D &dirInLocal, DirectionType sampledType, bool wlSelected,
        const SurfacePoint &surfPt, uint32_t pathLength) {
        LightPathVertex lightVertex = {};
        lightVertex.instIndex = surfPt.instIndex;
        lightVertex.geomInstIndex = surfPt.geomInstIndex;
        lightVertex.primIndex = surfPt.primIndex;
        lightVertex.u = surfPt.u;
        lightVertex.v = surfPt.v;
        lightVertex.totalPowerProbDensity = totalPowerProbDensity;
        lightVertex.prevTotalPowerProbDensity = prevTotalPowerProbDensity;
        lightVertex.prevSumPowerProbDensities = prevSumPowerProbDensities;
        lightVertex.backwardConversionFactor = backwardConversionFactor;
        lightVertex.flux = flux;
        lightVertex.dirInLocal = dirInLocal;
        lightVertex.sampledType = sampledType;
        lightVertex.wlSelected = wlSelected;
        lightVertex.pathLength = pathLength;
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
        float prevPowerProbDensity0 = pow2(1);
        float prevSumPowerProbDensities0 = 0;
        storeLightVertex(powerProbDensities0, prevPowerProbDensity0, prevSumPowerProbDensities0, 0,
                         alpha, Vector3D(0, 0, 1), Le0Result.posType, false,
                         Le0Result.surfPt, 0);

        EDFQuery edfQuery(DirectionType::All(), plp.commonWavelengthSamples);
        EDFSample Le1Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        EDFQueryResult Le1Result;
        SampledSpectrum Le1 = edf.sample(edfQuery, Le1Sample, &Le1Result);

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
        LVCBPTLightPathReadOnlyPayload* roPayloadPtr = &roPayload;
        LVCBPTLightPathWriteOnlyPayload* woPayloadPtr = &woPayload;
        LVCBPTLightPathReadWritePayload* rwPayloadPtr = &rwPayload;

        const uint32_t MaxPathLength = 25;
        while (true) {
            rwPayload.terminate = true;
            ++rwPayload.pathLength;

            optixu::trace<LVCBPTLightPathPayloadSignature>(
                plp.topGroup, asOptiXType(rayOrg), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
                shared::VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE,
                LVCBPTRayType::LightPath, MaxNumRayTypes, LVCBPTRayType::LightPath,
                roPayloadPtr, woPayloadPtr, rwPayloadPtr);

            if (rwPayload.pathLength >= MaxPathLength)
                rwPayload.terminate = true;
            if (rwPayload.terminate)
                break;

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
            pow2(roPayload->prevRevAreaPDF) * rwPayload->prevSumPowerProbDensities;
        rwPayload->prevTotalPowerProbDensity = rwPayload->totalPowerProbDensity;

        float lastDist2 = sqDistance(asPoint3D(optixGetWorldRayOrigin()), surfPt.position);
        float probDensity = roPayload->prevDirPDF * absDot(dirInLocal, geomNormalLocal) / lastDist2;
        //if (!vlr::isinf(rwPayload->totalPowerProbDensity) &&
        //    vlr::isinf(rwPayload->totalPowerProbDensity * pow2(probDensity))) {
        //    printf("LightPath: %g, %g\n", rwPayload->totalPowerProbDensity, pow2(probDensity));
        //}
        rwPayload->totalPowerProbDensity *= pow2(probDensity);

        storeLightVertex(rwPayload->totalPowerProbDensity,
                         rwPayload->prevTotalPowerProbDensity, rwPayload->prevSumPowerProbDensities,
                         roPayload->prevCosTerm / lastDist2,
                         rwPayload->alpha, dirInLocal,
                         roPayload->prevSampledType, rwPayload->singleIsSelected, surfPt, rwPayload->pathLength);

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



    static constexpr int32_t debugPathLength = 0;

    CUDA_DEVICE_KERNEL void RT_RG_NAME(lvcbptEyePath)() {
        uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

        KernelRNG rng = plp.rngBuffer.read(launchIndex);

        float2 p = make_float2(launchIndex.x + rng.getFloat0cTo1o(),
                               launchIndex.y + rng.getFloat0cTo1o());

        float resCorrection = plp.imageSize.x * plp.imageSize.y;
        WavelengthSamples wls = plp.commonWavelengthSamples;

        Camera camera(static_cast<ProgSigCamera_sample>(plp.progSampleLensPosition));
        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        camera.sample(We0Sample, &We0Result);

        IDF idf(plp.cameraDescriptor, We0Result.surfPt, wls);

        SampledSpectrum We0 = idf.evaluateSpatialImportance();
        SampledSpectrum alpha = We0 / We0Result.areaPDF;

        float powerProbDensities0 = pow2(We0Result.areaPDF);
        float prevPowerProbDensity0 = pow2(1);
        float prevSumPowerProbDensities0 = 0;

        IDFQuery idfQuery;

        // Connect with a randomly chosen light vertex.
        {
            const SurfacePoint &surfPtE = We0Result.surfPt;
            Normal3D geomNormalLocalE = surfPtE.shadingFrame.toLocal(surfPtE.geometricNormal);

            uint32_t lightVertexIndex = vlr::min<uint32_t>(
                *plp.numLightVertices * rng.getFloat0cTo1o(),
                *plp.numLightVertices - 1);
            const LightPathVertex &vertex = plp.lightVertexCache[lightVertexIndex];
            float vertexProb = 1.0f / *plp.numLightVertices;

            SurfacePoint surfPtL;
            uint32_t matIndexL;
            {
                const GeometryInstance &geomInst = plp.geomInstBuffer[vertex.geomInstIndex];
                ProgSigDecodeHitPoint decodeHitPoint(geomInst.progDecodeHitPoint);
                decodeHitPoint(vertex.instIndex, vertex.geomInstIndex, vertex.primIndex,
                               vertex.u, vertex.v, &surfPtL);

                Normal3D localNormal = calcNode(geomInst.nodeNormal, Normal3D(0.0f, 0.0f, 1.0f), surfPtL, wls);
                applyBumpMapping(localNormal, &surfPtL);

                Vector3D newTangent = calcNode(geomInst.nodeTangent, surfPtL.shadingFrame.x, surfPtL, wls);
                modifyTangent(newTangent, &surfPtL);

                matIndexL = geomInst.materialIndex;
            }

            Vector3D conRayDir;
            float squaredConDist;
            float fractionalVisibility;
            if ((debugPathLength == 0 || (vertex.pathLength + 1) == debugPathLength) &&
                testVisibility<LVCBPTRayType::Connection>(
                    surfPtE, surfPtL, wls, &conRayDir, &squaredConDist, &fractionalVisibility)) {
                float recSquaredConDist = 1.0f / squaredConDist;

                const SurfaceMaterialDescriptor matDescL = plp.materialDescriptorBuffer[matIndexL];
                constexpr TransportMode transportModeL = TransportMode::Importance;
                BSDF<transportModeL, BSDFTier::Bidirectional> bsdfL(matDescL, surfPtL, wls, vertex.pathLength == 0);
                Vector3D dirInLocalL = vertex.dirInLocal;
                Normal3D geomNormalLocalL = surfPtL.shadingFrame.toLocal(surfPtL.geometricNormal);
                BSDFQuery bsdfLQuery(dirInLocalL, geomNormalLocalL, transportModeL, DirectionType::All(), wls);

                Vector3D conRayDirLocalL = surfPtL.toLocal(-conRayDir);
                Vector3D conRayDirLocalE = surfPtE.toLocal(conRayDir);

                float cosL = absDot(conRayDirLocalL, geomNormalLocalL);
                float cosE = absDot(conRayDirLocalE, geomNormalLocalE);
                float G = cosL * cosE * recSquaredConDist;

                SampledSpectrum backwardFsL;
                SampledSpectrum forwardFsL = bsdfL.evaluate(bsdfLQuery, conRayDirLocalL, &backwardFsL);
                float backwardDirDensityL;
                /*float forwardDirDensityL = */bsdfL.evaluatePDF(bsdfLQuery, conRayDirLocalL, &backwardDirDensityL);
                //float forwardAreaDensityL = forwardDirDensityL * cosE * recSquaredConDist;
                float backwardAreaDensityL = backwardDirDensityL * vertex.backwardConversionFactor;
                float partialDenomMisWeightL = vertex.prevTotalPowerProbDensity +
                    pow2(backwardAreaDensityL) * vertex.prevSumPowerProbDensities; // extend eye subpath, shorten light subpath.

                SampledSpectrum backwardFsE;
                SampledSpectrum forwardFsE = idf.evaluateDirectionalImportance(idfQuery, conRayDirLocalE);
                float forwardDirDensityE = idf.evaluatePDF(idfQuery, conRayDirLocalE);
                forwardDirDensityE *= resCorrection;
                float forwardAreaDensityE = forwardDirDensityE * cosL * recSquaredConDist;
                float2 posInScreen = idf.backProjectDirection(idfQuery, conRayDirLocalE);
                float2 pixel = make_float2(posInScreen.x * plp.imageSize.x, posInScreen.y * plp.imageSize.y);

                // JP: ライトトレーシングをピクセル数実行することに等しいので確率密度がピクセル数倍になる。
                float scalarTerm = G * fractionalVisibility /
                    (vertexProb * resCorrection * plp.wavelengthProbability);
                if (vertex.wlSelected)
                    scalarTerm *= SampledSpectrum::NumComponents();
                SampledSpectrum conTerm = forwardFsL * scalarTerm * forwardFsE;
                SampledSpectrum unweightedContribution = vertex.flux * conTerm * alpha;

                float numMisWeight = powerProbDensities0 * pow2(resCorrection) * vertex.totalPowerProbDensity;
                float denomMisWeight = numMisWeight;
                // extend eye subpath, shorten light subpath.
                denomMisWeight += powerProbDensities0 * pow2(forwardAreaDensityE) * partialDenomMisWeightL;

                float misWeight = 10 * numMisWeight / denomMisWeight;
                SampledSpectrum contribution = misWeight * unweightedContribution;
                atomicAddToBuffer(wls, contribution, pixel);
            }
        }

        IDFSample We1Sample(p.x / plp.imageSize.x, p.y / plp.imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = idf.sample(idfQuery, We1Sample, &We1Result);
        We1Result.dirPDF *= resCorrection;

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
        LVCBPTEyePathExtraPayload exPayload = {};
        LVCBPTEyePathReadOnlyPayload* roPayloadPtr = &roPayload;
        LVCBPTEyePathWriteOnlyPayload* woPayloadPtr = &woPayload;
        LVCBPTEyePathReadWritePayload* rwPayloadPtr = &rwPayload;
        LVCBPTEyePathExtraPayload* exPayloadPtr = &exPayload;

        const uint32_t MaxPathLength = 25;
        while (true) {
            rwPayload.terminate = true;
            ++rwPayload.pathLength;

            if (debugPathLength != 0 &&
                rwPayload.pathLength > debugPathLength)
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

            if (rwPayload.pathLength >= MaxPathLength)
                rwPayload.terminate = true;
            if (rwPayload.terminate)
                break;

            rayOrg = woPayload.nextOrigin;
            rayDir = woPayload.nextDirection;
            roPayload.prevDirPDF = woPayload.dirPDF;
            roPayload.prevCosTerm = woPayload.cosTerm;
            roPayload.prevRevAreaPDF = woPayload.revAreaPDF;
            roPayload.prevSampledType = woPayload.sampledType;
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

        SurfacePoint surfPtE;
        float hypAreaPDF;
        calcSurfacePoint(hp, wls, &surfPtE, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDescE = plp.materialDescriptorBuffer[hp.sbtr->geomInst.materialIndex];
        constexpr TransportMode transportModeE = TransportMode::Radiance;
        BSDF<transportModeE, BSDFTier::Bidirectional> bsdfE(matDescE, surfPtE, wls);
        EDF edf(matDescE, surfPtE, wls);

        if (exPayload) {
            exPayload->firstHitAlbedo = bsdfE.getBaseColor();
            exPayload->firstHitNormal = surfPtE.shadingFrame.z;
        }

        Vector3D dirOutE = -asVector3D(optixGetWorldRayDirection());
        Vector3D dirOutLocalE = surfPtE.shadingFrame.toLocal(dirOutE);

        Normal3D geomNormalLocalE = surfPtE.shadingFrame.toLocal(surfPtE.geometricNormal);
        BSDFQuery bsdfEQuery(dirOutLocalE, geomNormalLocalE, transportModeE, DirectionType::All(), wls);

        rwPayload->prevSumPowerProbDensities =
            rwPayload->prevTotalPowerProbDensity +
            pow2(roPayload->prevRevAreaPDF) * rwPayload->prevSumPowerProbDensities;
        rwPayload->prevTotalPowerProbDensity = rwPayload->totalPowerProbDensity;
        if (rwPayload->pathLength == 1) {
            // Ignore the strategy with zero eye vertices.
            rwPayload->prevSumPowerProbDensities = 0;
            // 
            float resCorrection = plp.imageSize.x * plp.imageSize.y;
            rwPayload->prevTotalPowerProbDensity *= pow2(resCorrection);
        }

        float lastDist2 = sqDistance(asPoint3D(optixGetWorldRayOrigin()), surfPtE.position);
        float probDensity = roPayload->prevDirPDF * absDot(dirOutLocalE, geomNormalLocalE) / lastDist2;
        //if (!vlr::isinf(rwPayload->totalPowerProbDensity) &&
        //    vlr::isinf(rwPayload->totalPowerProbDensity * pow2(probDensity))) {
        //    printf("EyePath: %g, %g\n", rwPayload->totalPowerProbDensity, pow2(probDensity));
        //}
        rwPayload->totalPowerProbDensity *= pow2(probDensity);

        // implicit light sampling
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if ((debugPathLength == 0 || rwPayload->pathLength == debugPathLength) &&
            spEmittance.hasNonZero()) {
            EDFQuery edfQuery(DirectionType::All(), wls);
            SampledSpectrum Le = spEmittance * edf.evaluate(edfQuery, dirOutLocalE);
            SampledSpectrum unweightedContribution = rwPayload->alpha * Le;

            const Instance &inst = plp.instBuffer[surfPtE.instIndex];
            float instProb = inst.lightGeomInstDistribution.integral() / plp.lightInstDist.integral();
            float geomInstProb = hp.sbtr->geomInst.importance / inst.lightGeomInstDistribution.integral();
            float forwardAreaDensityL = plp.numLightPaths * instProb * geomInstProb * hypAreaPDF;

            float backwardDirDensityE = edf.evaluatePDF(edfQuery, dirOutLocalE);
            float backwardAreaDensityE = backwardDirDensityE * roPayload->prevCosTerm / lastDist2;
            float partialDenomMisWeightE = rwPayload->prevTotalPowerProbDensity +
                pow2(backwardAreaDensityE) * rwPayload->prevSumPowerProbDensities; // extend light subpath, shorten eye subpath.

            float numMisWeight = rwPayload->totalPowerProbDensity;
            float denomMisWeight = numMisWeight;
            // extend light subpath, shorten eye subpath.
            denomMisWeight += pow2(forwardAreaDensityL) * partialDenomMisWeightE;

            float misWeight = 10 * numMisWeight / denomMisWeight;
            rwPayload->contribution += misWeight * unweightedContribution;
        }

        // Connect with a randomly chosen light vertex.
        if (bsdfE.hasNonDelta()) {
            uint32_t lightVertexIndex = vlr::min<uint32_t>(
                *plp.numLightVertices * rng.getFloat0cTo1o(),
                *plp.numLightVertices - 1);
            const LightPathVertex &vertex = plp.lightVertexCache[lightVertexIndex];
            float vertexProb = 1.0f / *plp.numLightVertices;

            SurfacePoint surfPtL;
            uint32_t matIndexL;
            {
                const GeometryInstance &geomInst = plp.geomInstBuffer[vertex.geomInstIndex];
                ProgSigDecodeHitPoint decodeHitPoint(geomInst.progDecodeHitPoint);
                decodeHitPoint(vertex.instIndex, vertex.geomInstIndex, vertex.primIndex,
                               vertex.u, vertex.v, &surfPtL);

                Normal3D localNormal = calcNode(geomInst.nodeNormal, Normal3D(0.0f, 0.0f, 1.0f), surfPtL, wls);
                applyBumpMapping(localNormal, &surfPtL);

                Vector3D newTangent = calcNode(geomInst.nodeTangent, surfPtL.shadingFrame.x, surfPtL, wls);
                modifyTangent(newTangent, &surfPtL);

                matIndexL = geomInst.materialIndex;
            }

            //printf("SurfPtL: p: (%g, %g, %g), frame: (%g, %g, %g), (%g, %g, %g), (%g, %g, %g), "
            //       "gn: (%g, %g, %g), %u - %u - %u - %g, %g, tc: %g, %g, inf: %u, point: %u\n",
            //       VLR3DPrint(surfPtL.position),
            //       VLR3DPrint(surfPtL.shadingFrame.x), VLR3DPrint(surfPtL.shadingFrame.y), VLR3DPrint(surfPtL.shadingFrame.z),
            //       VLR3DPrint(surfPtL.geometricNormal),
            //       surfPtL.instIndex, surfPtL.geomInstIndex, surfPtL.primIndex, surfPtL.u, surfPtL.v,
            //       surfPtL.texCoord.u, surfPtL.texCoord.v,
            //       surfPtL.atInfinity, surfPtL.isPoint);

            Vector3D conRayDir;
            float squaredConDist;
            float fractionalVisibility;
            if ((debugPathLength == 0 || (rwPayload->pathLength + vertex.pathLength + 1) == debugPathLength) &&
                testVisibility<LVCBPTRayType::Connection>(
                    surfPtE, surfPtL, wls, &conRayDir, &squaredConDist, &fractionalVisibility)) {
                float recSquaredConDist = 1.0f / squaredConDist;

                const SurfaceMaterialDescriptor matDescL = plp.materialDescriptorBuffer[matIndexL];
                constexpr TransportMode transportModeL = TransportMode::Importance;
                BSDF<transportModeL, BSDFTier::Bidirectional> bsdfL(matDescL, surfPtL, wls, vertex.pathLength == 0);
                Vector3D dirInLocalL = vertex.dirInLocal;
                Normal3D geomNormalLocalL = surfPtL.shadingFrame.toLocal(surfPtL.geometricNormal);
                BSDFQuery bsdfLQuery(dirInLocalL, geomNormalLocalL, transportModeL, DirectionType::All(), wls);

                Vector3D conRayDirLocalL = surfPtL.toLocal(-conRayDir);
                Vector3D conRayDirLocalE = surfPtE.toLocal(conRayDir);

                float cosL = absDot(conRayDirLocalL, geomNormalLocalL);
                float cosE = absDot(conRayDirLocalE, geomNormalLocalE);
                float G = cosL * cosE * recSquaredConDist;

                SampledSpectrum backwardFsL;
                SampledSpectrum forwardFsL = bsdfL.evaluate(bsdfLQuery, conRayDirLocalL, &backwardFsL);
                float backwardDirDensityL;
                float forwardDirDensityL = bsdfL.evaluatePDF(bsdfLQuery, conRayDirLocalL, &backwardDirDensityL);
                float forwardAreaDensityL = forwardDirDensityL * cosE * recSquaredConDist;
                float backwardAreaDensityL = backwardDirDensityL * vertex.backwardConversionFactor;
                float partialDenomMisWeightL = vertex.prevTotalPowerProbDensity +
                    pow2(backwardAreaDensityL) * vertex.prevSumPowerProbDensities; // extend eye subpath, shorten light subpath.

                SampledSpectrum backwardFsE;
                SampledSpectrum forwardFsE = bsdfE.evaluate(bsdfEQuery, conRayDirLocalE, &backwardFsE);
                float backwardDirDensityE;
                float forwardDirDensityE = bsdfE.evaluatePDF(bsdfEQuery, conRayDirLocalE, &backwardDirDensityE);
                float forwardAreaDensityE = forwardDirDensityE * cosL * recSquaredConDist;
                float backwardAreaDensityE = backwardDirDensityE * roPayload->prevCosTerm / lastDist2;
                float partialDenomMisWeightE = rwPayload->prevTotalPowerProbDensity +
                    pow2(backwardAreaDensityE) * rwPayload->prevSumPowerProbDensities; // extend light subpath, shorten eye subpath.

                float scalarTerm = G * fractionalVisibility /
                    (vertexProb * plp.wavelengthProbability);
                if (vertex.wlSelected || rwPayload->singleIsSelected)
                    scalarTerm *= SampledSpectrum::NumComponents();
                SampledSpectrum conTerm = forwardFsL * scalarTerm * forwardFsE;
                SampledSpectrum unweightedContribution = vertex.flux * conTerm * rwPayload->alpha;

                float numMisWeight = rwPayload->totalPowerProbDensity * vertex.totalPowerProbDensity;
                float denomMisWeight = numMisWeight;
                // extend eye subpath, shorten light subpath.
                denomMisWeight += rwPayload->totalPowerProbDensity * pow2(forwardAreaDensityE) * partialDenomMisWeightL;
                // extend light subpath, shorten eye subpath.
                denomMisWeight += vertex.totalPowerProbDensity * pow2(forwardAreaDensityL) * partialDenomMisWeightE;

                float misWeight = 10 * numMisWeight / denomMisWeight;
                rwPayload->contribution += misWeight * unweightedContribution;

                //if (!vlr::isfinite(misWeight) || !unweightedContribution.allFinite()) {
                //    printf("%g (%g, %g), (%g, %g), (%g, %g, %g, %g), (%g, %g, %g)\n",
                //           misWeight, numMisWeight, denomMisWeight,
                //           rwPayload->totalPowerProbDensity, vertex.totalPowerProbDensity,
                //           probDensity, roPayload->prevDirPDF, absDot(dirOutLocalE, geomNormalLocalE), lastDist2,
                //           unweightedContribution.r, unweightedContribution.g, unweightedContribution.b);
                //}
            }
        }

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        BSDFQueryReverseResult fsRevResult;
        SampledSpectrum fs = bsdfE.sample(bsdfEQuery, sample, &fsResult, &fsRevResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !wls.singleIsSelected())
            rwPayload->singleIsSelected = true;

        float cosTerm = dot(fsResult.dirLocal, geomNormalLocalE);
        SampledSpectrum throughput = fs * (std::fabs(cosTerm) / fsResult.dirPDF);
        rwPayload->alpha *= throughput;

        // Russian roulette
        float continueProb = std::fmin(throughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        rwPayload->alpha /= continueProb;
        rwPayload->terminate = false;

        Vector3D dirInE = surfPtE.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPtE.position, cosTerm > 0.0f ? surfPtE.geometricNormal : -surfPtE.geometricNormal);
        woPayload->nextDirection = dirInE;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->cosTerm = std::fabs(cosTerm);
        woPayload->revAreaPDF = fsRevResult.dirPDF * roPayload->prevCosTerm / lastDist2;
        woPayload->sampledType = fsResult.sampledType;
    }



    CUDA_DEVICE_KERNEL void RT_MS_NAME(lvcbptEyePath)() {
        LVCBPTEyePathReadOnlyPayload* roPayload;
        LVCBPTEyePathReadWritePayload* rwPayload;
        LVCBPTEyePathExtraPayload* exPayload;
        LVCBPTEyePathPayloadSignature::get(&roPayload, nullptr, &rwPayload, &exPayload);

        if (exPayload) {
            exPayload->firstHitAlbedo = SampledSpectrum::Zero();
            exPayload->firstHitNormal = Normal3D(0.0f, 0.0f, 0.0f);
        }
    }
}
