#include "../shared/light_transport_common.h"

// Reference
// Progressive Light Transport Simulation on the GPU: Survey and Improvements

namespace vlr {
    using namespace shared;

    static constexpr bool includeRRProbability = true;
    static constexpr int32_t debugPathLength = 0;

    CUDA_DEVICE_FUNCTION bool onProbePixel() {
        return optixGetLaunchIndex().x == plp.probePixX && optixGetLaunchIndex().y == plp.probePixY;
    }

    CUDA_DEVICE_FUNCTION bool onProbePixel(const float2 &projPixel) {
        return static_cast<int32_t>(projPixel.x) == plp.probePixX &&
            static_cast<int32_t>(projPixel.y) == plp.probePixY;
    }



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
        if (ipx < plp.imageSize.x && ipy < plp.imageSize.y)
            plp.atomicAccumBuffer[ipy * plp.imageStrideInPixels + ipx].atomicAdd(wls, contribution);
    }

    CUDA_DEVICE_FUNCTION void storeLightVertex(
        const SurfacePoint &surfPt, const SampledSpectrum &flux,
        const Vector3D &dirInLocal, float backwardConversionFactor,
        float powerProbDensity, float prevPowerProbDensity,
        float secondPrevPartialDenomMisWeight, float secondPrevPowerProbRatioToFirst,
        bool deltaSampled, bool prevDeltaSampled, bool wlSelected, uint32_t pathLength) {
        LightPathVertex lightVertex = {};
        lightVertex.instIndex = surfPt.instIndex;
        lightVertex.geomInstIndex = surfPt.geomInstIndex;
        lightVertex.primIndex = surfPt.primIndex;
        lightVertex.u = surfPt.u;
        lightVertex.v = surfPt.v;
        lightVertex.powerProbDensity = powerProbDensity;
        lightVertex.prevPowerProbDensity = prevPowerProbDensity;
        lightVertex.secondPrevPartialDenomMisWeight = secondPrevPartialDenomMisWeight;
        lightVertex.secondPrevPowerProbRatioToFirst = secondPrevPowerProbRatioToFirst;
        lightVertex.backwardConversionFactor = backwardConversionFactor;
        lightVertex.flux = flux;
        lightVertex.dirInLocal = dirInLocal;
        lightVertex.deltaSampled = deltaSampled;
        lightVertex.prevDeltaSampled = prevDeltaSampled;
        lightVertex.wlSelected = wlSelected;
        lightVertex.pathLength = pathLength;
        uint32_t cacheIndex = atomicAdd(plp.numLightVertices, 1u);
        plp.lightVertexCache[cacheIndex] = lightVertex;
    }

    CUDA_DEVICE_FUNCTION void decodeHitPoint(
        const LightPathVertex &vertex, const WavelengthSamples &wls,
        SurfacePoint* surfPt, uint32_t* materialIndex) {
        const GeometryInstance &geomInst = plp.geomInstBuffer[vertex.geomInstIndex];
        ProgSigDecodeHitPoint decodeHitPoint(geomInst.progDecodeHitPoint);
        decodeHitPoint(vertex.instIndex, vertex.geomInstIndex, vertex.primIndex,
                       vertex.u, vertex.v, surfPt);

        Normal3D localNormal = calcNode(geomInst.nodeNormal, Normal3D(0.0f, 0.0f, 1.0f), *surfPt, wls);
        if (localNormal != Normal3D(0.0f, 0.0f, 1.0f))
            applyBumpMapping(localNormal, surfPt);

        Vector3D newTangent = calcNode(geomInst.nodeTangent, surfPt->shadingFrame.x, *surfPt, wls);
        if (newTangent != surfPt->shadingFrame.x)
            modifyTangent(newTangent, surfPt);

        *materialIndex = geomInst.materialIndex;
    }

    CUDA_DEVICE_FUNCTION float computeRRProbability(
        const SampledSpectrum &fs, const Vector3D &dirLocal, float dirDensity, const Normal3D &geomNormalLocal) {
        SampledSpectrum localThroughput = fs * absDot(dirLocal, geomNormalLocal) / dirDensity;
        return std::fmin(localThroughput.importance(plp.commonWavelengthSamples.selectedLambdaIndex()), 1.0f);
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
        float secondPrevPartialDenomMisWeight0 = 0;
        float secondPrevPowerProbRatioToFirst0 = 1;
        storeLightVertex(Le0Result.surfPt, alpha,
                         Vector3D(0, 0, 1), 0,
                         powerProbDensities0, prevPowerProbDensity0,
                         secondPrevPartialDenomMisWeight0, secondPrevPowerProbRatioToFirst0,
                         Le0Result.posType.isDelta(), false, false, 0);

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
        roPayload.dirPDF = Le1Result.dirPDF;
        roPayload.cosTerm = cosTerm;
        roPayload.prevRevAreaPDF = 1;
        roPayload.secondPrevDeltaSampled = false;
        roPayload.prevDeltaSampled = Le0Result.posType.isDelta();
        roPayload.sampledType = Le1Result.sampledType;
        LVCBPTLightPathWriteOnlyPayload woPayload = {};
        LVCBPTLightPathReadWritePayload rwPayload = {};
        rwPayload.rng = rng;
        rwPayload.alpha = alpha;
        rwPayload.powerProbDensity = powerProbDensities0;
        rwPayload.prevPowerProbDensity = prevPowerProbDensity0;
        rwPayload.secondPrevPartialDenomMisWeight = secondPrevPartialDenomMisWeight0;
        rwPayload.secondPrevPowerProbRatioToFirst = secondPrevPowerProbRatioToFirst0;
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
            roPayload.dirPDF = woPayload.dirPDF;
            roPayload.cosTerm = woPayload.cosTerm;
            roPayload.prevRevAreaPDF = woPayload.revAreaPDF;
            roPayload.secondPrevDeltaSampled = roPayload.prevDeltaSampled;
            roPayload.prevDeltaSampled = roPayload.sampledType.isDelta();
            roPayload.sampledType = woPayload.sampledType;
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
        if (rwPayload->singleIsSelected)
            wls.setSingleIsSelected();

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

        bool thirdLastSegIsValidStrategy =
            (!roPayload->prevDeltaSampled && !roPayload->secondPrevDeltaSampled) &&
            rwPayload->pathLength > 2; // separately accumulate the ratio for the strategy with zero light vertices.
        float powerProbRatio = pow2(roPayload->prevRevAreaPDF) / rwPayload->prevPowerProbDensity;
        rwPayload->secondPrevPartialDenomMisWeight =
            powerProbRatio *
            (rwPayload->secondPrevPartialDenomMisWeight + (thirdLastSegIsValidStrategy ? 1 : 0));
        rwPayload->secondPrevPowerProbRatioToFirst *= powerProbRatio;
        rwPayload->prevPowerProbDensity = rwPayload->powerProbDensity;

        float lastDist2 = sqDistance(asPoint3D(optixGetWorldRayOrigin()), surfPt.position);
        float probDensity = roPayload->dirPDF * absDot(dirInLocal, geomNormalLocal) / lastDist2;
        rwPayload->powerProbDensity = pow2(probDensity);

        storeLightVertex(surfPt, rwPayload->alpha,
                         dirInLocal, roPayload->cosTerm / lastDist2,
                         rwPayload->powerProbDensity, rwPayload->prevPowerProbDensity,
                         rwPayload->secondPrevPartialDenomMisWeight, rwPayload->secondPrevPowerProbRatioToFirst,
                         roPayload->sampledType.isDelta(), roPayload->prevDeltaSampled,
                         rwPayload->singleIsSelected, rwPayload->pathLength);

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        BSDFQueryReverseResult fsRevResult;
        SampledSpectrum fs = bsdf.sample(fsQuery, sample, &fsResult, &fsRevResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !rwPayload->singleIsSelected)
            rwPayload->singleIsSelected = true;

        float cosTerm = dot(fsResult.dirLocal, geomNormalLocal);
        SampledSpectrum localThroughput = fs * (std::fabs(cosTerm) / fsResult.dirPDF);
        rwPayload->alpha *= localThroughput;

        // Russian roulette
        float continueProb = std::fmin(localThroughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        rwPayload->alpha /= continueProb;
        rwPayload->terminate = false;

        Vector3D dirOut = surfPt.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPt.position, cosTerm > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        woPayload->nextDirection = dirOut;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->cosTerm = std::fabs(cosTerm);
        woPayload->revAreaPDF = fsRevResult.dirPDF * roPayload->cosTerm / lastDist2;
        woPayload->sampledType = fsResult.sampledType;
        if constexpr (includeRRProbability) {
            woPayload->dirPDF *= continueProb;
            woPayload->revAreaPDF *= computeRRProbability(
                fsRevResult.value, fsQuery.dirLocal, fsRevResult.dirPDF, geomNormalLocal);
        }
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

        // JP: 
        We0Result.areaPDF *= plp.imageSize.x * plp.imageSize.y;

        IDF idf(plp.cameraDescriptor, We0Result.surfPt, wls);

        SampledSpectrum We0 = idf.evaluateSpatialImportance();
        SampledSpectrum alpha = We0 / We0Result.areaPDF;

        float powerProbDensities0 = pow2(We0Result.areaPDF);
        float prevPowerProbDensity0 = pow2(1);
        float secondPrevPartialDenomMisWeight0 = 0;

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
            decodeHitPoint(vertex, wls, &surfPtL, &matIndexL);

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
                if (bsdfL.hasNonDelta()) {
                    Vector3D dirInLocalL = vertex.dirInLocal;
                    Normal3D geomNormalLocalL = surfPtL.shadingFrame.toLocal(surfPtL.geometricNormal);
                    BSDFQuery bsdfLQuery(dirInLocalL, geomNormalLocalL, transportModeL, DirectionType::All(), wls);

                    Vector3D conRayDirLocalL = surfPtL.toLocal(-conRayDir);
                    Vector3D conRayDirLocalE = surfPtE.toLocal(conRayDir);

                    float cosL = absDot(conRayDirLocalL, geomNormalLocalL);
                    float cosE = absDot(conRayDirLocalE, geomNormalLocalE);
                    float G = cosL * cosE * recSquaredConDist;
                    float scalarConTerm = G * fractionalVisibility / (vertexProb * plp.wavelengthProbability);
                    if (vertex.wlSelected)
                        scalarConTerm *= SampledSpectrum::NumComponents();

                    // on the light vertex
                    // JP: Implicit Lens Sampling戦略は考えない。
                    // EN: Don't consider the implicit lens sampling strategy.
                    SampledSpectrum backwardFsL;
                    SampledSpectrum forwardFsL = bsdfL.evaluate(bsdfLQuery, conRayDirLocalL, &backwardFsL);
                    float backwardDirDensityL;
                    /*float forwardDirDensityL = */bsdfL.evaluatePDF(bsdfLQuery, conRayDirLocalL, &backwardDirDensityL);
                    //float forwardAreaDensityL = forwardDirDensityL * cosE * recSquaredConDist;
                    if constexpr (includeRRProbability)
                        backwardDirDensityL *= computeRRProbability(
                            backwardFsL, bsdfLQuery.dirLocal, backwardDirDensityL, geomNormalLocalL);
                    float backwardAreaDensityL = backwardDirDensityL * vertex.backwardConversionFactor;

                    // on the eye vertex
                    SampledSpectrum backwardFsE;
                    SampledSpectrum forwardFsE = idf.evaluateDirectionalImportance(idfQuery, conRayDirLocalE);
                    float forwardDirDensityE = idf.evaluatePDF(idfQuery, conRayDirLocalE);
                    float forwardAreaDensityE = forwardDirDensityE * cosL * recSquaredConDist;
                    float2 posInScreen = idf.backProjectDirection(idfQuery, conRayDirLocalE);
                    float2 pixel = make_float2(posInScreen.x * plp.imageSize.x, posInScreen.y * plp.imageSize.y);

                    // extend eye subpath, shorten light subpath.
                    float partialDenomMisWeightL;
                    {
                        bool secondLastSegIsValidStrategyL =
                            (!vertex.deltaSampled && !vertex.prevDeltaSampled) &&
                            vertex.pathLength > 1; // separately accumulate the ratio for the strategy with zero light vertices.
                        float lastToSecondLastPowerProbRatio = pow2(backwardAreaDensityL) / vertex.prevPowerProbDensity;
                        partialDenomMisWeightL =
                            lastToSecondLastPowerProbRatio *
                            (vertex.secondPrevPartialDenomMisWeight + (secondLastSegIsValidStrategyL ? 1 : 0));
                        float powerProbRatioToFirst = vertex.secondPrevPowerProbRatioToFirst;
                        if (vertex.pathLength > 0)
                            powerProbRatioToFirst *= lastToSecondLastPowerProbRatio;

                        bool lastSegIsValidStrategyL =
                            (/*bsdfL.hasNonDelta() &&*/ !vertex.deltaSampled) &&
                            vertex.pathLength > 0; // separately accumulate the ratio for the strategy with zero light vertices.
                        float curToLastPowerProbRatio = pow2(forwardAreaDensityE) / (vertex.powerProbDensity);
                        partialDenomMisWeightL =
                            curToLastPowerProbRatio *
                            (partialDenomMisWeightL + (lastSegIsValidStrategyL ? 1 : 0));
                        powerProbRatioToFirst *= curToLastPowerProbRatio;

                        // JP: Implicit Light Sampling戦略にはLight Vertex Cacheからのランダムな選択確率は含まれない。
                        partialDenomMisWeightL += powerProbRatioToFirst / pow2(vertexProb);
                    }

                    SampledSpectrum conTerm = forwardFsL * scalarConTerm * forwardFsE;
                    SampledSpectrum unweightedContribution = vertex.flux * conTerm * alpha;

                    float recMisWeight = 1.0f + partialDenomMisWeightL;
                    float misWeight = 1.0f / recMisWeight;
                    SampledSpectrum contribution = misWeight * unweightedContribution;
                    if (contribution.allFinite())
                        atomicAddToBuffer(wls, contribution, pixel);
                    else
                        vlrprintf("Pass %u, (%u, %u - %u, %u): Not a finite value.\n",
                                  plp.numAccumFrames, optixGetLaunchIndex().x, optixGetLaunchIndex().y,
                                  static_cast<int32_t>(pixel.x), static_cast<int32_t>(pixel.y));
                }
            }
        }

        IDFSample We1Sample(p.x / plp.imageSize.x, p.y / plp.imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = idf.sample(idfQuery, We1Sample, &We1Result);

        Point3D rayOrg = We0Result.surfPt.position;
        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        float cosTerm = We0Result.surfPt.calcCosTerm(rayDir);
        alpha *= We1 * (cosTerm / We1Result.dirPDF);

        LVCBPTEyePathReadOnlyPayload roPayload = {};
        roPayload.dirPDF = We1Result.dirPDF;
        roPayload.cosTerm = cosTerm;
        roPayload.prevRevAreaPDF = 1;
        roPayload.secondPrevDeltaSampled = false;
        roPayload.prevDeltaSampled = We0Result.posType.isDelta();
        roPayload.sampledType = We1Result.sampledType;
        LVCBPTEyePathWriteOnlyPayload woPayload = {};
        LVCBPTEyePathReadWritePayload rwPayload = {};
        rwPayload.rng = rng;
        rwPayload.alpha = alpha;
        rwPayload.contribution = SampledSpectrum::Zero();
        rwPayload.powerProbDensity = powerProbDensities0;
        rwPayload.prevPowerProbDensity = prevPowerProbDensity0;
        rwPayload.secondPrevPartialDenomMisWeight = secondPrevPartialDenomMisWeight0;
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
            roPayload.dirPDF = woPayload.dirPDF;
            roPayload.cosTerm = woPayload.cosTerm;
            roPayload.prevRevAreaPDF = woPayload.revAreaPDF;
            roPayload.secondPrevDeltaSampled = roPayload.prevDeltaSampled;
            roPayload.prevDeltaSampled = roPayload.sampledType.isDelta();
            roPayload.sampledType = woPayload.sampledType;
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
        if (rwPayload->singleIsSelected)
            wls.setSingleIsSelected();

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

        bool thirdLastSegIsValidStrategyE =
            (!roPayload->prevDeltaSampled && !roPayload->secondPrevDeltaSampled) &&
            rwPayload->pathLength > 2; // Ignore the strategy with zero eye path vertices.
        rwPayload->secondPrevPartialDenomMisWeight =
            pow2(roPayload->prevRevAreaPDF) / rwPayload->prevPowerProbDensity *
            (rwPayload->secondPrevPartialDenomMisWeight + (thirdLastSegIsValidStrategyE ? 1 : 0));
        rwPayload->prevPowerProbDensity = rwPayload->powerProbDensity;

        float lastDist2 = sqDistance(asPoint3D(optixGetWorldRayOrigin()), surfPtE.position);
        float probDensity = roPayload->dirPDF * absDot(dirOutLocalE, geomNormalLocalE) / lastDist2;
        rwPayload->powerProbDensity = pow2(probDensity);

        float vertexProb = 1.0f / *plp.numLightVertices;

        // implicit light sampling (zero light path vertices)
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if ((debugPathLength == 0 || rwPayload->pathLength == debugPathLength) &&
            spEmittance.hasNonZero() && edf.hasNonDelta()) {
            EDFQuery edfQuery(DirectionType::All(), wls);
            SampledSpectrum Le = spEmittance * edf.evaluate(edfQuery, dirOutLocalE);
            SampledSpectrum unweightedContribution = rwPayload->alpha * Le;
            if (rwPayload->singleIsSelected)
                unweightedContribution *= SampledSpectrum::NumComponents();

            const Instance &inst = plp.instBuffer[surfPtE.instIndex];
            float instProb = inst.lightGeomInstDistribution.integral() / plp.lightInstDist.integral();
            float geomInstProb = hp.sbtr->geomInst.importance / inst.lightGeomInstDistribution.integral();
            float forwardAreaDensityL = plp.numLightPaths * instProb * geomInstProb * hypAreaPDF;

            float backwardDirDensityE = edf.evaluatePDF(edfQuery, dirOutLocalE);
            float backwardAreaDensityE = backwardDirDensityE * roPayload->cosTerm / lastDist2;

            // extend light subpath, shorten eye subpath.
            float partialDenomMisWeightE;
            {
                bool secondLastSegIsValidStrategyE =
                    (!roPayload->sampledType.isDelta() && !roPayload->prevDeltaSampled) &&
                    rwPayload->pathLength > 1; // Ignore the strategy with zero eye vertices.
                float lastToSecondLastPowerProbRatio = pow2(backwardAreaDensityE) / rwPayload->prevPowerProbDensity;
                partialDenomMisWeightE =
                    lastToSecondLastPowerProbRatio *
                    (rwPayload->secondPrevPartialDenomMisWeight + (secondLastSegIsValidStrategyE ? 1 : 0));

                bool lastSegIsValidStrategyE = /*edf.hasNonDelta() &&*/ !roPayload->sampledType.isDelta();
                float curToLastPowerProbRatio = pow2(forwardAreaDensityL) / rwPayload->powerProbDensity;
                partialDenomMisWeightE =
                    curToLastPowerProbRatio *
                    (partialDenomMisWeightE + (lastSegIsValidStrategyE ? 1 : 0));

                // JP: Implicit Light Sampling戦略以外にはLight Vertex Cacheからのランダムな選択確率が含まれる。
                partialDenomMisWeightE *= pow2(vertexProb);
            }

            float recMisWeight = 1.0f + partialDenomMisWeightE;
            float misWeight = 1.0f / recMisWeight;
            rwPayload->contribution += misWeight * unweightedContribution;
        }

        // Connect with a randomly chosen light vertex.
        if (bsdfE.hasNonDelta()) {
            uint32_t lightVertexIndex = vlr::min<uint32_t>(
                *plp.numLightVertices * rng.getFloat0cTo1o(),
                *plp.numLightVertices - 1);
            const LightPathVertex &vertex = plp.lightVertexCache[lightVertexIndex];

            SurfacePoint surfPtL;
            uint32_t matIndexL;
            decodeHitPoint(vertex, wls, &surfPtL, &matIndexL);

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
                if (bsdfL.hasNonDelta()) {
                    Vector3D dirInLocalL = vertex.dirInLocal;
                    Normal3D geomNormalLocalL = surfPtL.shadingFrame.toLocal(surfPtL.geometricNormal);
                    BSDFQuery bsdfLQuery(dirInLocalL, geomNormalLocalL, transportModeL, DirectionType::All(), wls);

                    Vector3D conRayDirLocalL = surfPtL.toLocal(-conRayDir);
                    Vector3D conRayDirLocalE = surfPtE.toLocal(conRayDir);

                    float cosL = absDot(conRayDirLocalL, geomNormalLocalL);
                    float cosE = absDot(conRayDirLocalE, geomNormalLocalE);
                    float G = cosL * cosE * recSquaredConDist;
                    float scalarConTerm = G * fractionalVisibility / (vertexProb * plp.wavelengthProbability);
                    if (vertex.wlSelected || rwPayload->singleIsSelected)
                        scalarConTerm *= SampledSpectrum::NumComponents();

                    // on the light vertex
                    SampledSpectrum backwardFsL;
                    SampledSpectrum forwardFsL = bsdfL.evaluate(bsdfLQuery, conRayDirLocalL, &backwardFsL);
                    float backwardDirDensityL;
                    float forwardDirDensityL = bsdfL.evaluatePDF(bsdfLQuery, conRayDirLocalL, &backwardDirDensityL);
                    if constexpr (includeRRProbability) {
                        if (vertex.pathLength > 0)
                            forwardDirDensityL *= computeRRProbability(
                                forwardFsL, conRayDirLocalL, forwardDirDensityL, geomNormalLocalL);
                        backwardDirDensityL *= computeRRProbability(
                            backwardFsL, bsdfLQuery.dirLocal, backwardDirDensityL, geomNormalLocalL);
                    }
                    float forwardAreaDensityL = forwardDirDensityL * cosE * recSquaredConDist;
                    float backwardAreaDensityL = backwardDirDensityL * vertex.backwardConversionFactor;

                    // on the eye vertex
                    SampledSpectrum backwardFsE;
                    SampledSpectrum forwardFsE = bsdfE.evaluate(bsdfEQuery, conRayDirLocalE, &backwardFsE);
                    float backwardDirDensityE;
                    float forwardDirDensityE = bsdfE.evaluatePDF(bsdfEQuery, conRayDirLocalE, &backwardDirDensityE);
                    if constexpr (includeRRProbability) {
                        forwardDirDensityE *= computeRRProbability(
                            forwardFsE, conRayDirLocalE, forwardDirDensityE, geomNormalLocalE);
                        backwardDirDensityE *= computeRRProbability(
                            backwardFsE, bsdfEQuery.dirLocal, backwardDirDensityE, geomNormalLocalE);
                    }
                    float forwardAreaDensityE = forwardDirDensityE * cosL * recSquaredConDist;
                    float backwardAreaDensityE = backwardDirDensityE * roPayload->cosTerm / lastDist2;

                    // extend eye subpath, shorten light subpath.
                    float partialDenomMisWeightL;
                    {
                        bool secondLastSegIsValidStrategyL =
                            (!vertex.deltaSampled && !vertex.prevDeltaSampled) &&
                            vertex.pathLength > 1; // separately accumulate the ratio for the strategy with zero light vertices.
                        float lastToSecondLastPowerProbRatio = pow2(backwardAreaDensityL) / vertex.prevPowerProbDensity;
                        partialDenomMisWeightL =
                            lastToSecondLastPowerProbRatio *
                            (vertex.secondPrevPartialDenomMisWeight + (secondLastSegIsValidStrategyL ? 1 : 0));
                        float powerProbRatioToFirst = vertex.secondPrevPowerProbRatioToFirst;
                        if (vertex.pathLength > 0)
                            powerProbRatioToFirst *= lastToSecondLastPowerProbRatio;

                        bool lastSegIsValidStrategyL =
                            (/*bsdfL.hasNonDelta() &&*/ !vertex.deltaSampled) &&
                            vertex.pathLength > 0; // separately accumulate the ratio for the strategy with zero light vertices.
                        float curToLastPowerProbRatio = pow2(forwardAreaDensityE) / vertex.powerProbDensity;
                        partialDenomMisWeightL =
                            curToLastPowerProbRatio *
                            (partialDenomMisWeightL + (lastSegIsValidStrategyL ? 1 : 0));
                        powerProbRatioToFirst *= curToLastPowerProbRatio;

                        // JP: Implicit Light Sampling戦略にはLight Vertex Cacheからのランダムな選択確率は含まれない。
                        partialDenomMisWeightL += powerProbRatioToFirst / pow2(vertexProb);
                    }

                    // extend light subpath, shorten eye subpath.
                    float partialDenomMisWeightE;
                    {
                        bool secondLastSegIsValidStrategyE =
                            !roPayload->sampledType.isDelta() && !roPayload->prevDeltaSampled;
                        float lastToSecondLastPowerProbRatio = pow2(backwardAreaDensityE) / rwPayload->prevPowerProbDensity;
                        partialDenomMisWeightE =
                            lastToSecondLastPowerProbRatio *
                            (rwPayload->secondPrevPartialDenomMisWeight + (secondLastSegIsValidStrategyE ? 1 : 0));

                        bool lastSegIsValidStrategyE = /*bsdfE.hasNonDelta() &&*/ !roPayload->sampledType.isDelta();
                        float curToLastPowerProbRatio = pow2(forwardAreaDensityL) / rwPayload->powerProbDensity;
                        partialDenomMisWeightE =
                            curToLastPowerProbRatio *
                            (partialDenomMisWeightE + (lastSegIsValidStrategyE ? 1 : 0));
                    }

                    SampledSpectrum conTerm = forwardFsL * scalarConTerm * forwardFsE;
                    SampledSpectrum unweightedContribution = vertex.flux * conTerm * rwPayload->alpha;

                    float recMisWeight = 1.0f + partialDenomMisWeightL + partialDenomMisWeightE;
                    float misWeight = 1.0f / recMisWeight;
                    rwPayload->contribution += misWeight * unweightedContribution;
                }
            }
        }

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        BSDFQueryReverseResult fsRevResult;
        SampledSpectrum fs = bsdfE.sample(bsdfEQuery, sample, &fsResult, &fsRevResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !rwPayload->singleIsSelected)
            rwPayload->singleIsSelected = true;

        float cosTerm = dot(fsResult.dirLocal, geomNormalLocalE);
        SampledSpectrum localThroughput = fs * (std::fabs(cosTerm) / fsResult.dirPDF);
        rwPayload->alpha *= localThroughput;

        // Russian roulette
        float continueProb = std::fmin(localThroughput.importance(wls.selectedLambdaIndex()), 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        rwPayload->alpha /= continueProb;
        rwPayload->terminate = false;

        Vector3D dirInE = surfPtE.fromLocal(fsResult.dirLocal);
        woPayload->nextOrigin = offsetRayOrigin(surfPtE.position, cosTerm > 0.0f ? surfPtE.geometricNormal : -surfPtE.geometricNormal);
        woPayload->nextDirection = dirInE;
        woPayload->dirPDF = fsResult.dirPDF;
        woPayload->cosTerm = std::fabs(cosTerm);
        woPayload->revAreaPDF = fsRevResult.dirPDF * roPayload->cosTerm / lastDist2;
        woPayload->sampledType = fsResult.sampledType;
        if constexpr (includeRRProbability) {
            woPayload->dirPDF *= continueProb;
            woPayload->revAreaPDF *= computeRRProbability(
                fsRevResult.value, bsdfEQuery.dirLocal, fsRevResult.dirPDF, geomNormalLocalE);
        }
    }



    CUDA_DEVICE_KERNEL void RT_MS_NAME(lvcbptEyePath)() {
        LVCBPTEyePathReadOnlyPayload* roPayload;
        LVCBPTEyePathReadWritePayload* rwPayload;
        LVCBPTEyePathExtraPayload* exPayload;
        LVCBPTEyePathPayloadSignature::get(&roPayload, nullptr, &rwPayload, &exPayload);

        WavelengthSamples wls = plp.commonWavelengthSamples;
        if (rwPayload->singleIsSelected)
            wls.setSingleIsSelected();

        if (exPayload) {
            exPayload->firstHitAlbedo = SampledSpectrum::Zero();
            exPayload->firstHitNormal = Normal3D(0.0f, 0.0f, 0.0f);
        }

        const Instance &inst = plp.instBuffer[plp.envLightInstIndex];
        uint32_t geomInstindex = inst.geomInstIndices[0];
        const GeometryInstance &geomInst = plp.geomInstBuffer[geomInstindex];

        if (geomInst.importance == 0)
            return;

        SurfacePoint surfPtE;
        Vector3D direction = asVector3D(optixGetWorldRayDirection());
        float hypAreaPDF;
        {
            float posPhi, theta;
            direction.toPolarYUp(&theta, &posPhi);

            float sinPhi, cosPhi;
            ::vlr::sincos(posPhi, &sinPhi, &cosPhi);
            Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));
            ReferenceFrame shadingFrame;
            shadingFrame.x = texCoord0Dir;
            shadingFrame.z = -direction;
            shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

            surfPtE.instIndex = plp.envLightInstIndex;
            surfPtE.geomInstIndex = geomInstindex;
            surfPtE.primIndex = 0;

            surfPtE.position = Point3D(direction.x, direction.y, direction.z);
            surfPtE.shadingFrame = shadingFrame;
            surfPtE.isPoint = false;
            surfPtE.atInfinity = true;
            surfPtE.geometricNormal = -direction;
            surfPtE.u = posPhi;
            surfPtE.v = theta;
            float phi = posPhi + inst.rotationPhi;
            phi = phi - ::vlr::floor(phi / (2 * VLR_M_PI)) * 2 * VLR_M_PI;
            surfPtE.texCoord = TexCoord2D(phi / (2 * VLR_M_PI), theta / VLR_M_PI);

            VLRAssert(vlr::isfinite(phi) && vlr::isfinite(theta), "\"phi\", \"theta\": Not finite values %g, %g.", phi, theta);
            float uvPDF = geomInst.asInfSphere.importanceMap.evaluatePDF(phi / (2 * VLR_M_PI), theta / VLR_M_PI);
            // The true value is: lim_{l to inf} uvPDF / (2 * M_PI * M_PI * std::sin(theta)) / l^2
            hypAreaPDF = uvPDF / (2 * VLR_M_PI * VLR_M_PI * std::sin(theta));
        }

        const SurfaceMaterialDescriptor &matDescE = plp.materialDescriptorBuffer[geomInst.materialIndex];
        EDF edf(matDescE, surfPtE, wls);

        Vector3D dirOutLocalE = surfPtE.shadingFrame.toLocal(-direction);

        bool thirdLastSegIsValidStrategyE =
            (!roPayload->prevDeltaSampled && !roPayload->secondPrevDeltaSampled) &&
            rwPayload->pathLength > 2; // Ignore the strategy with zero eye vertices.
        rwPayload->secondPrevPartialDenomMisWeight =
            pow2(roPayload->prevRevAreaPDF) / rwPayload->prevPowerProbDensity *
            (rwPayload->secondPrevPartialDenomMisWeight + (thirdLastSegIsValidStrategyE ? 1 : 0));
        rwPayload->prevPowerProbDensity = rwPayload->powerProbDensity;

        float lastDist2 = 1.0f;
        float probDensity = roPayload->dirPDF / lastDist2;
        rwPayload->powerProbDensity = pow2(probDensity);

        float vertexProb = 1.0f / *plp.numLightVertices;

        // implicit light sampling (zero light path vertices)
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if ((debugPathLength == 0 || rwPayload->pathLength == debugPathLength) &&
            spEmittance.hasNonZero() && edf.hasNonDelta()) {
            EDFQuery edfQuery(DirectionType::All(), wls);
            SampledSpectrum Le = spEmittance * edf.evaluate(edfQuery, dirOutLocalE);
            SampledSpectrum unweightedContribution = rwPayload->alpha * Le;
            unweightedContribution /= plp.wavelengthProbability;
            if (rwPayload->singleIsSelected)
                unweightedContribution *= SampledSpectrum::NumComponents();

            const Instance &inst = plp.instBuffer[surfPtE.instIndex];
            float instProb = inst.lightGeomInstDistribution.integral() / plp.lightInstDist.integral();
            float geomInstProb = geomInst.importance / inst.lightGeomInstDistribution.integral();
            float forwardAreaDensityL = plp.numLightPaths * instProb * geomInstProb * hypAreaPDF;

            float backwardDirDensityE = edf.evaluatePDF(edfQuery, dirOutLocalE);
            float backwardAreaDensityE = backwardDirDensityE * roPayload->cosTerm / lastDist2;

            // extend light subpath, shorten eye subpath.
            float partialDenomMisWeightE;
            {
                bool secondLastSegIsValidStrategyE =
                    (!roPayload->sampledType.isDelta() && !roPayload->prevDeltaSampled) &&
                    rwPayload->pathLength > 1; // Ignore the strategy with zero eye vertices.
                float lastToSecondLastPowerProbRatio = pow2(backwardAreaDensityE) / rwPayload->prevPowerProbDensity;
                partialDenomMisWeightE =
                    lastToSecondLastPowerProbRatio *
                    (rwPayload->secondPrevPartialDenomMisWeight + (secondLastSegIsValidStrategyE ? 1 : 0));

                bool lastSegIsValidStrategyE = /*edf.hasNonDelta() &&*/ !roPayload->sampledType.isDelta();
                float curToLastPowerProbRatio = pow2(forwardAreaDensityL) / rwPayload->powerProbDensity;
                partialDenomMisWeightE =
                    curToLastPowerProbRatio *
                    (partialDenomMisWeightE + (lastSegIsValidStrategyE ? 1 : 0));

                // JP: Implicit Light Sampling戦略以外にはLight Vertex Cacheからのランダムな選択確率が含まれる。
                partialDenomMisWeightE *= pow2(vertexProb);
            }

            float recMisWeight = 1.0f + partialDenomMisWeightE;
            float misWeight = 1.0f / recMisWeight;
            rwPayload->contribution += misWeight * unweightedContribution;
        }
    }
}
