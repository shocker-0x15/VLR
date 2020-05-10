#include "light_transport_common.cuh"

namespace VLR {
    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void pathTracingIteration() {
        auto sbtr = optixu::getHitGroupSBTRecordData();
        const GeometryInstanceData &geomInst = plp.geomInstData[sbtr.geomInstData];

        Payload* payload;
        optixu::getPayloads<PayloadSignature>(&payload);

        KernelRNG &rng = payload->rng;
        WavelengthSamples &wls = payload->wls;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(geomInst, wls, &surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[geomInst.materialIndex];
        BSDF bsdf(matDesc, surfPt, wls);
        EDF edf(matDesc, surfPt, wls);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(optixGetWorldRayDirection()));

        // implicit light sampling
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            SampledSpectrum Le = spEmittance * edf.evaluate(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!payload->prevSampledType.isDelta() && payload->rayType != RayType::Primary) {
                float bsdfPDF = payload->prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(optixGetWorldRayOrigin()));
                float lightPDF = geomInst.importance / getSumLightImportances() * hypAreaPDF * dist2 / std::fabs(dirOutLocal.z);
                MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            }

            payload->contribution += payload->alpha * Le * MISWeight;
        }
        if (surfPt.atInfinity || payload->maxLengthTerminate)
            return;

        // Russian roulette
        float continueProb = std::fmin(payload->alpha.importance(wls.selectedLambdaIndex()) / payload->initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        payload->alpha /= continueProb;

        Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        BSDFQuery fsQuery(dirOutLocal, geomNormalLocal, DirectionType::All(), wls);

        // Next Event Estimation (explicit light sampling)
        if (bsdf.hasNonDelta()) {
            SurfaceLight light;
            float lightProb;
            float uPrim;
            selectSurfaceLight(rng.getFloat0cTo1o(), &light, &lightProb, &uPrim);

            SurfaceLightPosSample lpSample(uPrim, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
            SurfaceLightPosQueryResult lpResult;
            light.sample(lpSample, &lpResult);

            const SurfaceMaterialDescriptor lightMatDesc = plp.materialDescriptorBuffer[lpResult.materialIndex];
            EDF ledf(lightMatDesc, lpResult.surfPt, wls);
            SampledSpectrum M = ledf.evaluateEmittance();

            Vector3D shadowRayDir;
            float squaredDistance;
            float fractionalVisibility;
            if (M.hasNonZero() && testVisibility(surfPt, lpResult.surfPt, wls, &shadowRayDir, &squaredDistance, &fractionalVisibility)) {
                Vector3D shadowRayDir_l = lpResult.surfPt.toLocal(-shadowRayDir);
                Vector3D shadowRayDir_sn = surfPt.toLocal(shadowRayDir);

                SampledSpectrum Le = M * ledf.evaluate(EDFQuery(), shadowRayDir_l);
                float lightPDF = lightProb * lpResult.areaPDF;

                SampledSpectrum fs = bsdf.evaluate(fsQuery, shadowRayDir_sn);
                float cosLight = lpResult.surfPt.calcCosTerm(-shadowRayDir);
                float bsdfPDF = bsdf.evaluatePDF(fsQuery, shadowRayDir_sn) * cosLight / squaredDistance;

                float MISWeight = 1.0f;
                if (!lpResult.posType.isDelta() && !std::isinf(lightPDF))
                    MISWeight = (lightPDF * lightPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);

                float G = fractionalVisibility * absDot(shadowRayDir_sn, geomNormalLocal) * cosLight / squaredDistance;
                float scalarCoeff = G * MISWeight / lightPDF; // 直接contributionの計算式に入れるとCUDAのバグなのかおかしな結果になる。
                payload->contribution += payload->alpha * Le * fs * scalarCoeff;
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
        payload->alpha *= fs * (std::fabs(cosFactor) / fsResult.dirPDF);

        Vector3D dirIn = surfPt.fromLocal(fsResult.dirLocal);
        payload->origin = offsetRayOrigin(surfPt.position, cosFactor > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        payload->direction = dirIn;
        payload->prevDirPDF = fsResult.dirPDF;
        payload->prevSampledType = fsResult.sampledType;
        payload->terminate = false;
    }



    // JP: 本当は無限大の球のIntersection/Bounding Box Programを使用して環境光に関する処理もClosest Hit Programで統一的に行いたい。
    //     が、OptiXのBVHビルダーがLBVHベースなので無限大のAABBを生成するのは危険。
    //     仕方なくMiss Programで環境光を処理する。
    RT_PROGRAM void pathTracingMiss() {
        Payload* payload;
        optixu::getPayloads<PayloadSignature>(&payload);

        if (plp.envLightDescriptor.importance == 0)
            return;

        Vector3D direction = asVector3D(optixGetWorldRayDirection());
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        float sinPhi, cosPhi;
        VLR::sincos(phi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));
        ReferenceFrame shadingFrame;
        shadingFrame.x = texCoord0Dir;
        shadingFrame.z = -static_cast<Normal3D>(direction);
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint surfPt;
        surfPt.position = Point3D(direction.x, direction.y, direction.z);
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = false;
        surfPt.atInfinity = true;

        surfPt.geometricNormal = shadingFrame.z;
        surfPt.u = phi;
        surfPt.v = theta;
        phi += plp.envLightDescriptor.body.asInfSphere.rotationPhi;
        phi = phi - std::floor(phi / (2 * VLR_M_PI)) * 2 * VLR_M_PI;
        surfPt.texCoord = TexCoord2D(phi / (2 * VLR_M_PI), theta / VLR_M_PI);

        float hypAreaPDF = evaluateEnvironmentAreaPDF(phi, theta);

        const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[plp.envLightDescriptor.materialIndex];
        EDF edf(matDesc, surfPt, payload->wls);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-direction);

        // implicit light sampling
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            SampledSpectrum Le = spEmittance * edf.evaluate(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!payload->prevSampledType.isDelta() && payload->rayType != RayType::Primary) {
                float bsdfPDF = payload->prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(optixGetWorldRayOrigin()));
                float lightPDF = plp.envLightDescriptor.importance / getSumLightImportances() * hypAreaPDF * dist2 / std::fabs(dirOutLocal.z);
                MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            }

            payload->contribution += payload->alpha * Le * MISWeight;
        }
    }



    // Common Ray Generation Program for All Camera Types
    RT_PROGRAM void pathTracing() {
        uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

        KernelRNG rng = plp.rngBuffer[launchIndex];

        float2 p = make_float2(launchIndex.x + rng.getFloat0cTo1o(),
                               launchIndex.y + rng.getFloat0cTo1o());

        float selectWLPDF;
        WavelengthSamples wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &selectWLPDF);

        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        SampledSpectrum We0 = plp.progSampleLensPosition(wls, We0Sample, &We0Result);

        IDFSample We1Sample(p.x / plp.imageSize.x, p.y / plp.imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = plp.progSampleIDF(We0Result.surfPt, wls, We1Sample, &We1Result);

        Point3D rayOrg = We0Result.surfPt.position;
        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        RayType rayType = RayType::Primary;
        SampledSpectrum alpha = (We0 * We1) * (We0Result.surfPt.calcCosTerm(rayDir) / (We0Result.areaPDF * We1Result.dirPDF * selectWLPDF));

        Payload payload;
        payload.maxLengthTerminate = false;
        payload.rng = rng;
        payload.initImportance = alpha.importance(wls.selectedLambdaIndex());
        payload.wls = wls;
        payload.alpha = alpha;
        payload.contribution = SampledSpectrum::Zero();
        Payload* payloadPtr = &payload;

        const uint32_t MaxPathLength = 25;
        uint32_t pathLength = 0;
        while (true) {
            payload.rayType = rayType;
            payload.terminate = true;
            ++pathLength;
            if (pathLength >= MaxPathLength)
                payload.maxLengthTerminate = true;
            optixu::trace<PayloadSignature>(
                plp.topGroup,
                asOptiXType(rayOrg), asOptiXType(rayDir),
                0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                rayType, RayType::NumTypes, rayType,
                payloadPtr);

            if (payload.terminate)
                break;
            VLRAssert(pathLength < MaxPathLength, "Path should be terminated... Something went wrong...");

            rayOrg = payload.origin;
            rayDir = payload.direction;
            rayType = RayType::Scattered;
        }
        plp.rngBuffer[launchIndex] = payload.rng;
        if (!payload.contribution.allFinite()) {
            vlrprintf("Pass %u, (%u, %u): Not a finite value.\n", plp.numAccumFrames, launchIndex.x, launchIndex.y);
            return;
        }

        if (plp.numAccumFrames == 1)
            plp.outputBuffer[launchIndex].reset();
        plp.outputBuffer[launchIndex].add(wls, payload.contribution);
    }



    //// Exception Program
    //RT_PROGRAM void exception() {
    //    //uint32_t code = rtGetExceptionCode();
    //    rtPrintExceptionDetails();
    //}
}
