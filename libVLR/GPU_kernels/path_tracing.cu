#include "kernel_common.cuh"

namespace VLR {
    // Context-scope Variables
    rtDeclareVariable(rtObject, pv_topGroup, , );
    rtDeclareVariable(optix::uint2, pv_imageSize, , );
    rtDeclareVariable(uint32_t, pv_numAccumFrames, , );
    rtBuffer<KernelRNG, 2> pv_rngBuffer;
    rtBuffer<RGBSpectrum, 2> pv_outputBuffer;
    rtBuffer<SurfaceMaterialDescriptor, 1> pv_materialDescriptorBuffer;

    rtDeclareVariable(DiscreteDistribution1D, pv_lightImpDist, , );
    rtBuffer<SurfaceLightDescriptor> pv_surfaceLightDescriptorBuffer;



    // ----------------------------------------------------------------
    // Light
    
    RT_FUNCTION bool testVisibility(const SurfacePoint &shadingSurfacePoint, const SurfacePoint &lightSurfacePoint, 
                                    Vector3D* shadowRayDir, float* squaredDistance, float* fractionalVisibility) {
        VLRAssert(shadingSurfacePoint.atInfinity == false, "Shading point must be in finite region.");

        *shadowRayDir = lightSurfacePoint.calcDirectionFrom(shadingSurfacePoint.position, squaredDistance);
        optix::Ray shadowRay = optix::make_Ray(asOptiXType(shadingSurfacePoint.position), asOptiXType(*shadowRayDir), RayType::Shadow, 1e-4f, FLT_MAX);
        if (!lightSurfacePoint.atInfinity)
            shadowRay.tmax = std::sqrt(*squaredDistance) * 0.9999f;

        ShadowRayPayload shadowPayload;
        shadowPayload.fractionalVisibility = 1.0f;
        rtTrace(pv_topGroup, shadowRay, shadowPayload);

        *fractionalVisibility = shadowPayload.fractionalVisibility;

        return *fractionalVisibility > 0;
    }

    RT_FUNCTION void selectSurfaceLight(float lightSample, SurfaceLight* light, float* lightProb, float* remapped) {
        uint32_t lightIdx = pv_lightImpDist.sample(lightSample, lightProb, remapped);
        *light = SurfaceLight(pv_surfaceLightDescriptorBuffer[lightIdx]);
    }

    // END: Light
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // NormalAlphaModifier

    // bound
    RT_CALLABLE_PROGRAM float Null_NormalAlphaModifier_fetchAlpha(const TexCoord2D &texCoord) {
        return 1.0f;
    }

    // bound
    RT_CALLABLE_PROGRAM Normal3D Null_NormalAlphaModifier_fetchNormal(const TexCoord2D &texCoord) {
        return Normal3D(0, 0, 1);
    }

    // per GeometryInstance
    rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> pv_texNormalAlpha;

    // bound
    RT_CALLABLE_PROGRAM float NormalAlphaModifier_fetchAlpha(const TexCoord2D &texCoord) {
        float alpha = tex2D(pv_texNormalAlpha, texCoord.u, texCoord.v).w;
        return alpha;
    }

    // bound
    RT_CALLABLE_PROGRAM Normal3D NormalAlphaModifier_fetchNormal(const TexCoord2D &texCoord) {
        float4 texValue = tex2D(pv_texNormalAlpha, texCoord.u, texCoord.v);
        Normal3D normalLocal = 2 * Normal3D(texValue.x, texValue.y, texValue.z) - 1.0f;
        return normalLocal;
    }

    // JP: 法線マップに従ってシェーディングフレームを変更する。
    // EN: perturb the shading frame according to the normal map.
    RT_FUNCTION void applyBumpMapping(const Normal3D &normalLocal, SurfacePoint* surfPt) {
        const ReferenceFrame &originalFrame = surfPt->shadingFrame;

        Vector3D nLocal = normalLocal;
        Vector3D tLocal = Vector3D::Ex() - dot(nLocal, Vector3D::Ex()) * nLocal;
        Vector3D bLocal = Vector3D::Ey() - dot(nLocal, Vector3D::Ey()) * nLocal;
        Vector3D t = normalize(originalFrame.fromLocal(tLocal));
        Vector3D b = normalize(originalFrame.fromLocal(bLocal));
        Vector3D n = normalize(originalFrame.fromLocal(nLocal));
        ReferenceFrame bumpFrame(t, b, n);

        surfPt->shadingFrame = bumpFrame;
    }

    // END: NormalAlphaModifier
    // ----------------------------------------------------------------



    rtDeclareVariable(optix::uint2, sm_launchIndex, rtLaunchIndex, );
    rtDeclareVariable(optix::Ray, sm_ray, rtCurrentRay, );
    rtDeclareVariable(Payload, sm_payload, rtPayload, );
    rtDeclareVariable(ShadowRayPayload, sm_shadowRayPayload, rtPayload, );
    rtDeclareVariable(HitPointParameter, a_hitPointParam, attribute hitPointParam, );

    typedef rtCallableProgramX<RGBSpectrum(const LensPosSample &, LensPosQueryResult*)> progSigSampleLensPosition;
    typedef rtCallableProgramX<RGBSpectrum(const SurfacePoint &, const IDFSample &, IDFQueryResult*)> progSigSampleIDF;

    typedef rtCallableProgramX<TexCoord2D(const HitPointParameter &)> progSigDecodeTexCoord;
    typedef rtCallableProgramX<void(const HitPointParameter &, SurfacePoint*, float*)> progSigDecodeHitPoint;
    typedef rtCallableProgramX<float(const TexCoord2D &)> progSigFetchAlpha;
    typedef rtCallableProgramX<Normal3D(const TexCoord2D &)> progSigFetchNormal;

    // per Camera
    rtDeclareVariable(progSigSampleLensPosition, pv_progSampleLensPosition, , );
    rtDeclareVariable(progSigSampleIDF, pv_progSampleIDF, , );

    // per Material
    rtDeclareVariable(uint32_t, pv_materialIndex, , );

    // per GeometryInstance
    rtDeclareVariable(progSigDecodeTexCoord, pv_progDecodeTexCoord, , );
    rtDeclareVariable(progSigDecodeHitPoint, pv_progDecodeHitPoint, , );
    rtDeclareVariable(progSigFetchAlpha, pv_progFetchAlpha, , );
    rtDeclareVariable(progSigFetchNormal, pv_progFetchNormal, , );
    rtDeclareVariable(float, pv_importance, , );

    // Common Any Hit Program for All Primitive Types and Materials for non-shadow rays
    RT_PROGRAM void stochasticAlphaAnyHit() {
        HitPointParameter hitPointParam = a_hitPointParam;
        TexCoord2D texCoord = pv_progDecodeTexCoord(hitPointParam);

        float alpha = pv_progFetchAlpha(texCoord);

        if (sm_payload.rng.getFloat0cTo1o() >= alpha)
            rtIgnoreIntersection();
    }

    // Common Any Hit Program for All Primitive Types and Materials for shadow rays
    RT_PROGRAM void alphaAnyHit() {
        HitPointParameter hitPointParam = a_hitPointParam;
        TexCoord2D texCoord = pv_progDecodeTexCoord(hitPointParam);

        float alpha = pv_progFetchAlpha(texCoord);

        sm_shadowRayPayload.fractionalVisibility *= (1 - alpha);
        if (sm_shadowRayPayload.fractionalVisibility == 0.0f)
            rtTerminateRay();
    }

    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void pathTracingIteration() {
        KernelRNG &rng = sm_payload.rng;

        SurfacePoint surfPt;
        float areaPDF;
        HitPointParameter hitPointParam = a_hitPointParam;
        pv_progDecodeHitPoint(hitPointParam, &surfPt, &areaPDF);

        applyBumpMapping(pv_progFetchNormal(surfPt.texCoord), &surfPt);

        const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_materialIndex];
        BSDF bsdf(matDesc, surfPt, sm_payload.wavelengthSelected);
        EDF edf(matDesc, surfPt);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(sm_ray.direction));

        // implicit light sampling
        RGBSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            RGBSpectrum Le = spEmittance * edf.evaluateEDF(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!sm_payload.prevSampledType.isDelta() && sm_ray.ray_type != RayType::Primary) {
                float bsdfPDF = sm_payload.prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(sm_ray.origin));
                float lightPDF = pv_importance / pv_lightImpDist.integral() * areaPDF * dist2 / std::abs(dirOutLocal.z);
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

                RGBSpectrum Le = M * ledf.evaluateEDF(EDFQuery(), shadowRayDir_l);
                float lightPDF = lightProb * lpResult.areaPDF;

                RGBSpectrum fs = bsdf.evaluateBSDF(fsQuery, shadowRayDir_sn);
                float cosLight = lpResult.surfPt.calcCosTerm(-shadowRayDir);
                float bsdfPDF = bsdf.evaluateBSDF_PDF(fsQuery, shadowRayDir_sn) * cosLight / squaredDistance;

                float MISWeight = 1.0f;
                if (!lpResult.posType.isDelta() && !std::isinf(lpResult.areaPDF))
                    MISWeight = (lightPDF * lightPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);

                float G = fractionalVisibility * absDot(shadowRayDir_sn, geomNormalLocal) * cosLight / squaredDistance;
                float scalarCoeff = G * MISWeight / lightPDF; // 直接contributionの計算式に入れるとCUDAのバグなのかおかしな結果になる。
                sm_payload.contribution += sm_payload.alpha * Le * fs * scalarCoeff;
            }
        }

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        RGBSpectrum fs = bsdf.sampleBSDF(fsQuery, sample, &fsResult);
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

    RT_PROGRAM void pathTracingMiss() {
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

        const uint32_t MaxPathLength = 5;
        uint32_t pathLength = 0;
        while (true) {
            payload.terminate = true;
            ++pathLength;
            if (pathLength >= MaxPathLength)
                payload.maxLengthTerminate = true;
            rtTrace(pv_topGroup, ray, payload);

            if (payload.terminate)
                break;

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
