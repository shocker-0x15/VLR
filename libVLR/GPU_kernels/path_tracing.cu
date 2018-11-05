#include "light_transport_common.cuh"

namespace VLR {
    // Context-scope Variables
    rtDeclareVariable(optix::uint2, pv_imageSize, , );
    rtDeclareVariable(uint32_t, pv_numAccumFrames, , );
    rtDeclareVariable(ProgSigSampleLensPosition, pv_progSampleLensPosition, , );
    rtDeclareVariable(ProgSigSampleIDF, pv_progSampleIDF, , );
    rtBuffer<KernelRNG, 2> pv_rngBuffer;
    rtBuffer<RGBSpectrum, 2> pv_outputBuffer;



    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void pathTracingIteration() {
        KernelRNG &rng = sm_payload.rng;

        SurfacePoint surfPt;
        float hypAreaPDF;
        HitPointParameter hitPointParam = a_hitPointParam;
        pv_progDecodeHitPoint(hitPointParam, &surfPt, &hypAreaPDF);

        applyBumpMapping(fetchNormal(surfPt), &surfPt);

        const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_materialIndex];
        BSDF bsdf(matDesc, surfPt, sm_payload.wavelengthSelected);
        EDF edf(matDesc, surfPt);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(sm_ray.direction));

        // implicit light sampling
        RGBSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            RGBSpectrum Le = spEmittance * edf.evaluate(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!sm_payload.prevSampledType.isDelta() && sm_ray.ray_type != RayType::Primary) {
                float bsdfPDF = sm_payload.prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(sm_ray.origin));
                float lightPDF = pv_importance / getSumLightImportances() * hypAreaPDF * dist2 / std::abs(dirOutLocal.z);
                MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            }

            sm_payload.contribution += sm_payload.alpha * Le * MISWeight;
        }
        if (surfPt.atInfinity || sm_payload.maxLengthTerminate)
            return;

        // Russian roulette
        float continueProb = std::min(sm_payload.alpha.importance(sm_payload.wlHint) / sm_payload.initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        sm_payload.alpha /= continueProb;

        Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        BSDFQuery fsQuery(dirOutLocal, geomNormalLocal, DirectionType::All(), sm_payload.wlHint);

        // Next Event Estimation (explicit light sampling)
        if (bsdf.hasNonDelta()) {
            SurfaceLight light;
            float lightProb;
            float uPrim;
            selectSurfaceLight(rng.getFloat0cTo1o(), &light, &lightProb, &uPrim);

            SurfaceLightPosSample lpSample(uPrim, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
            SurfaceLightPosQueryResult lpResult;
            light.sample(lpSample, &lpResult);

            const SurfaceMaterialDescriptor lightMatDesc = pv_materialDescriptorBuffer[lpResult.materialIndex];
            EDF ledf(lightMatDesc, lpResult.surfPt);
            RGBSpectrum M = ledf.evaluateEmittance();

            Vector3D shadowRayDir;
            float squaredDistance;
            float fractionalVisibility;
            if (M.hasNonZero() && testVisibility(surfPt, lpResult.surfPt, &shadowRayDir, &squaredDistance, &fractionalVisibility)) {
                Vector3D shadowRayDir_l = lpResult.surfPt.toLocal(-shadowRayDir);
                Vector3D shadowRayDir_sn = surfPt.toLocal(shadowRayDir);

                RGBSpectrum Le = M * ledf.evaluate(EDFQuery(), shadowRayDir_l);
                float lightPDF = lightProb * lpResult.areaPDF;

                RGBSpectrum fs = bsdf.evaluate(fsQuery, shadowRayDir_sn);
                float cosLight = lpResult.surfPt.calcCosTerm(-shadowRayDir);
                float bsdfPDF = bsdf.evaluatePDF(fsQuery, shadowRayDir_sn) * cosLight / squaredDistance;

                float MISWeight = 1.0f;
                if (!lpResult.posType.isDelta() && !std::isinf(lightPDF))
                    MISWeight = (lightPDF * lightPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);

                float G = fractionalVisibility * absDot(shadowRayDir_sn, geomNormalLocal) * cosLight / squaredDistance;
                float scalarCoeff = G * MISWeight / lightPDF; // 直接contributionの計算式に入れるとCUDAのバグなのかおかしな結果になる。
                sm_payload.contribution += sm_payload.alpha * Le * fs * scalarCoeff;
            }
        }

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        RGBSpectrum fs = bsdf.sample(fsQuery, sample, &fsResult);
        if (fs == RGBSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !sm_payload.wavelengthSelected) {
            fsResult.dirPDF /= RGBSpectrum::NumComponents();
            sm_payload.wavelengthSelected = true;
        }

        sm_payload.alpha *= fs * (absDot(fsResult.dirLocal, geomNormalLocal) / fsResult.dirPDF);

        Vector3D dirIn = surfPt.fromLocal(fsResult.dirLocal);
        sm_payload.origin = surfPt.position;
        sm_payload.direction = dirIn;
        sm_payload.prevDirPDF = fsResult.dirPDF;
        sm_payload.prevSampledType = fsResult.sampledType;
        sm_payload.terminate = false;
    }



    // JP: 本当は無限大の球のIntersection/Bounding Box Programを使用して環境光に関する処理もClosest Hit Programで統一的に行いたい。
    //     が、OptiXのBVHビルダーがLBVHベースなので無限大のAABBを生成するのは危険。
    //     仕方なくMiss Programで環境光を処理する。
    RT_PROGRAM void pathTracingMiss() {
        if (pv_envLightDescriptor.importance == 0)
            return;

        Vector3D direction = asVector3D(sm_ray.direction);
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        Vector3D texCoord0Dir = Vector3D(-std::cos(theta), 0.0f, -std::sin(theta));
        ReferenceFrame shadingFrame;
        shadingFrame.x = texCoord0Dir;
        shadingFrame.z = -direction;
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint surfPt;
        surfPt.position = Point3D(direction.x, direction.y, direction.z);
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = false;
        surfPt.atInfinity = true;

        surfPt.geometricNormal = -direction;
        surfPt.u = phi;
        surfPt.v = theta;
        surfPt.texCoord = TexCoord2D(phi / (2 * M_PIf), theta / M_PIf);

        float hypAreaPDF = evaluateEnvironmentAreaPDF(phi, theta);

        const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_envLightDescriptor.body.asEnvironmentLight.materialIndex];
        EDF edf(matDesc, surfPt);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(sm_ray.direction));

        // implicit light sampling
        RGBSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            RGBSpectrum Le = spEmittance * edf.evaluate(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!sm_payload.prevSampledType.isDelta() && sm_ray.ray_type != RayType::Primary) {
                float bsdfPDF = sm_payload.prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(sm_ray.origin));
                float lightPDF = pv_envLightDescriptor.importance / getSumLightImportances() * hypAreaPDF * dist2 / std::abs(dirOutLocal.z);
                MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            }

            sm_payload.contribution += sm_payload.alpha * Le * MISWeight;
        }
    }



    // Common Ray Generation Program for All Camera Types
    RT_PROGRAM void pathTracing() {
        KernelRNG rng = pv_rngBuffer[sm_launchIndex];

        optix::float2 p = make_float2(sm_launchIndex.x + rng.getFloat0cTo1o(), sm_launchIndex.y + rng.getFloat0cTo1o());

        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        RGBSpectrum We0 = pv_progSampleLensPosition(We0Sample, &We0Result);

        IDFSample We1Sample(p.x / pv_imageSize.x, p.y / pv_imageSize.y);
        IDFQueryResult We1Result;
        RGBSpectrum We1 = pv_progSampleIDF(We0Result.surfPt, We1Sample, &We1Result);

        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        RGBSpectrum alpha = (We0 * We1) * (We0Result.surfPt.calcCosTerm(rayDir) / (We0Result.areaPDF * We1Result.dirPDF));

        optix::Ray ray = optix::make_Ray(asOptiXType(We0Result.surfPt.position), asOptiXType(rayDir), RayType::Primary, 0.0f, FLT_MAX);

        Payload payload;
        payload.wavelengthSelected = false;
        payload.maxLengthTerminate = false;
        payload.wlHint = std::min<uint32_t>(RGBSpectrum::NumComponents() - 1, RGBSpectrum::NumComponents() * rng.getFloat0cTo1o());
        payload.rng = rng;
        payload.initImportance = alpha.importance(payload.wlHint);
        payload.alpha = alpha;
        payload.contribution = RGBSpectrum::Zero();

        const uint32_t MaxPathLength = 25;
        uint32_t pathLength = 0;
        while (true) {
            payload.terminate = true;
            ++pathLength;
            if (pathLength >= MaxPathLength)
                payload.maxLengthTerminate = true;
            rtTrace(pv_topGroup, ray, payload);

            if (payload.terminate)
                break;
            VLRAssert(pathLength < MaxPathLength, "Path should be terminated... Something went wrong...");

            ray = optix::make_Ray(asOptiXType(payload.origin), asOptiXType(payload.direction), RayType::Scattered, 1e-4f, FLT_MAX);
        }
        pv_rngBuffer[sm_launchIndex] = payload.rng;
        if (!payload.contribution.allFinite()) {
            rtPrintf("Pass %u, (%u, %u): Not a finite value.\n", pv_numAccumFrames, sm_launchIndex.x, sm_launchIndex.y);
            return;
        }

        RGBSpectrum &contribution = pv_outputBuffer[sm_launchIndex];
        if (pv_numAccumFrames == 1)
            contribution = payload.contribution;
        else
            contribution = (contribution * (pv_numAccumFrames - 1) + payload.contribution) / pv_numAccumFrames;
    }



    // Exception Program
    RT_PROGRAM void exception() {
        //uint32_t code = rtGetExceptionCode();
        rtPrintExceptionDetails();
    }
}
