#include "kernel_common.cuh"
#include "random_distributions.cuh"

namespace VLR {
    // Context-scope Variables
    rtDeclareVariable(rtObject, pv_topGroup, , );
    rtBuffer<PCG32RNG, 2> pv_rngBuffer;
    rtBuffer<RGBSpectrum, 2> pv_outputBuffer;



    // ----------------------------------------------------------------
    // Light

    //rtDeclareVariable(DiscreteDistribution1D, pv_lightImpDist, , );
    //rtBuffer<SurfaceLightDescriptor> pv_surfaceLightDescriptors;
    
    RT_FUNCTION bool testVisibility(const SurfacePoint &shadingSurfacePoint, const SurfacePoint &lightSurfacePoint, 
                                      Vector3D* shadowRayDir, float* squaredDistance, float* fractionalVisibility) {
        VLRAssert(shadingSurfacePoint.atInfinity == false, "Shading point must be in finite region.");

        *shadowRayDir = lightSurfacePoint.calcDirectionFrom(shadingSurfacePoint.position, squaredDistance);
        optix::Ray shadowRay(asOptiXType(shadingSurfacePoint.position), asOptiXType(*shadowRayDir), RayType::Shadow, 1e-4f);
        if (lightSurfacePoint.atInfinity)
            shadowRay.tmax = FLT_MAX;
        else
            shadowRay.tmax = std::sqrt(*squaredDistance) * 0.9999f;

        ShadowRayPayload shadowPayload;
        rtTrace(pv_topGroup, shadowRay, shadowPayload);

        *fractionalVisibility = shadowPayload.fractionalVisibility;

        return *fractionalVisibility > 0;
    }

    RT_FUNCTION void selectSurfaceLight(float lightSample, SurfaceLight* light, float* lightProb, float* remapped) {
        //uint32_t lightIdx = pv_lightImpDist.sample(lightSample, lightProb, remapped);
        //*light = SurfaceLight(pv_surfaceLightDescriptors[lightIdx]);
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
    typedef rtCallableProgramX<void(const HitPointParameter &, SurfacePoint*)> progSigDecodeHitPoint;
    typedef rtCallableProgramX<float(const TexCoord2D &)> progSigFetchAlpha;
    typedef rtCallableProgramX<Normal3D(const TexCoord2D &)> progSigFetchNormal;

    // per Camera
    rtDeclareVariable(progSigSampleLensPosition, pv_progSampleLensPosition, , );
    rtDeclareVariable(progSigSampleIDF, pv_progSampleIDF, , );

    // per Material
    rtDeclareVariable(MaterialDescriptor, pv_materialDescriptor, , );

    // per GeometryInstance
    rtDeclareVariable(progSigDecodeTexCoord, pv_progDecodeTexCoord, , );
    rtDeclareVariable(progSigDecodeHitPoint, pv_progDecodeHitPoint, , );
    rtDeclareVariable(progSigFetchAlpha, pv_progFetchAlpha, , );
    rtDeclareVariable(progSigFetchNormal, pv_progFetchNormal, , );

    // Common Any Hit Program for All Primitive Types and Materials for non-shadow rays
    RT_PROGRAM void stochasticAlphaAnyHit() {
        HitPointParameter hitPointParam = a_hitPointParam;
        TexCoord2D texCoord = pv_progDecodeTexCoord(hitPointParam);

        float alpha = pv_progFetchAlpha(texCoord);

        PCG32RNG &rng = pv_rngBuffer[sm_launchIndex];
        if (rng.getFloat0cTo1o() >= alpha)
            rtIgnoreIntersection();
    }

    // Common Any Hit Program for All Primitive Types and Materials for shadow rays
    RT_PROGRAM void alphaAnyHit() {
        HitPointParameter hitPointParam = a_hitPointParam;
        TexCoord2D texCoord = pv_progDecodeTexCoord(hitPointParam);

        float alpha = pv_progFetchAlpha(texCoord);

        sm_shadowRayPayload.fractionalVisibility *= alpha;
        if (sm_shadowRayPayload.fractionalVisibility == 0.0f)
            rtTerminateRay();
    }

    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void pathTracingIteration() {
        PCG32RNG &rng = pv_rngBuffer[sm_launchIndex];

        SurfacePoint surfPt;
        HitPointParameter hitPointParam = a_hitPointParam;
        pv_progDecodeHitPoint(hitPointParam, &surfPt);

        const MaterialDescriptor matDesc = pv_materialDescriptor;
        MaterialParameters matParams(matDesc, surfPt);

        // DELETE ME
        sm_payload.contribution = matParams.getBaseColor();
        sm_payload.terminate = true;
        return;

        //applyBumpMapping(pv_progFetchNormal(surfPt.texCoord), &surfPt);

        //Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(sm_ray.direction));

        //// implicit light sampling
        ///*if (surfPt.isEmitting())*/ {
        //    float bsdfPDF = sm_payload.prevDirPDF;

        //    RGBSpectrum Le = matParams.evaluateEmittance * matParams.evaluateEDF(EDFQuery(), dirOutLocal);
        //    float dist2 = surfPt.calcSquaredDistance(asPoint3D(sm_ray.origin));
        //    float lightPDF = 1.0f;// = si.getLightProb() * surfPt.evaluateAreaPDF() * dist2 / surfPt.calcCosTerm(asVector3D(ray.direction));

        //    float MISWeight = 1.0f;
        //    if (!sm_payload.prevSampledType.isDelta() && sm_ray.ray_type != RayType::Primary)
        //        MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);

        //    sm_payload.contribution += sm_payload.alpha * Le * MISWeight;
        //}
        //if (surfPt.atInfinity) {
        //    sm_payload.terminate = true;
        //    return;
        //}

        //// Russian roulette
        //float continueProb = std::min(sm_payload.alpha.importance() / sm_payload.initY, 1.0f);
        //if (rng.getFloat0cTo1o() >= continueProb) {
        //    sm_payload.terminate = true;
        //    return;
        //}

        //Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        //BSDFQuery fsQuery(dirOutLocal, geomNormalLocal, DirectionType::All());

        //// Next Event Estimation (explicit light sampling)
        ///*if (bsdf->hasNonDelta())*/ {
        //    float lightSample = rng.getFloat0cTo1o();
        //    SurfaceLight light;
        //    float lightProb;
        //    selectSurfaceLight(lightSample, &light, &lightProb, &lightSample);

        //    SurfaceLightPosSample lpSample(lightSample, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        //    SurfaceLightPosQueryResult lpResult;
        //    RGBSpectrum M = light.sample(lpSample, &lpResult);

        //    Vector3D shadowRayDir;
        //    float squaredDistance;
        //    float fractionalVisibility;
        //    if (testVisibility(surfPt, lpResult.surfPt, &shadowRayDir, &squaredDistance, &fractionalVisibility)) {
        //        Vector3D shadowRayDir_l = lpResult.surfPt.toLocal(-shadowRayDir);
        //        Vector3D shadowRayDir_sn = surfPt.toLocal(shadowRayDir);

        //        RGBSpectrum Le = M * evaluateEDF(lpResult.surfPt.texCoord, EDFQuery(), shadowRayDir_l);
        //        float lightPDF = lightProb * lpResult.areaPDF;

        //        RGBSpectrum fs = matParams.evaluateBSDF(fsQuery, shadowRayDir_sn);
        //        float cosLight = lpResult.surfPt.calcCosTerm(-shadowRayDir);
        //        float bsdfPDF = matParams.evaluateBSDF_PDF(fsQuery, shadowRayDir_sn) * cosLight / squaredDistance;

        //        float MISWeight = 1.0f;
        //        if (!lpResult.posType.isDelta() && !std::isinf(lpResult.areaPDF))
        //            MISWeight = (lightPDF * lightPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);

        //        float G = fractionalVisibility * absDot(shadowRayDir_sn, geomNormalLocal) * cosLight / squaredDistance;
        //        sm_payload.contribution += sm_payload.alpha * Le * fs * (G * MISWeight / lightPDF);
        //    }
        //}

        //BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        //BSDFQueryResult fsResult;
        //RGBSpectrum fs = matParams.sampleBSDF(fsQuery, sample, &fsResult);
        //if (fs == RGBSpectrum::Zero() || fsResult.dirPDF == 0.0f) {
        //    sm_payload.terminate = true;
        //    return;
        //}

        //sm_payload.alpha *= fs * absDot(fsResult.dirLocal, geomNormalLocal) / fsResult.dirPDF;

        //Vector3D dirIn = surfPt.fromLocal(fsResult.dirLocal);
        //sm_payload.origin = surfPt.position;
        //sm_payload.direction = dirIn;
        //sm_payload.prevDirPDF = fsResult.dirPDF;
        //sm_payload.prevSampledType = fsResult.sampledType;
        //sm_payload.terminate = false;
    }

    RT_PROGRAM void pathTracingMiss() {
        sm_payload.terminate = true;
    }

    // Common Ray Generation Program for All Camera Types
    RT_PROGRAM void pathTracing() {
        PCG32RNG &rng = pv_rngBuffer[sm_launchIndex];

        optix::float2 p = make_float2(sm_launchIndex.x + rng.getFloat0cTo1o(), sm_launchIndex.y + rng.getFloat0cTo1o());
        optix::size_t2 bufferSize = pv_outputBuffer.size();
        optix::uint2 imageSize = make_uint2(bufferSize.x, bufferSize.y);

        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        RGBSpectrum We0 = pv_progSampleLensPosition(We0Sample, &We0Result);

        IDFSample We1Sample(p.x / imageSize.x, p.y / imageSize.y);
        IDFQueryResult We1Result;
        RGBSpectrum We1 = pv_progSampleIDF(We0Result.surfPt, We1Sample, &We1Result);

        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        RGBSpectrum alpha = (We0 * We1) * (We0Result.surfPt.calcCosTerm(rayDir) / (We0Result.areaPDF * We1Result.dirPDF));

        optix::Ray ray = optix::make_Ray(asOptiXType(We0Result.surfPt.position), asOptiXType(rayDir), RayType::Primary, 0.0f, INFINITY);
        Payload payload;
        payload.initY = alpha.importance();
        payload.alpha = alpha;
        payload.contribution = RGBSpectrum::Zero();

        while (true) {
            rtTrace(pv_topGroup, ray, payload);

            if (payload.terminate)
                break;

            ray = optix::make_Ray(asOptiXType(payload.origin), asOptiXType(payload.direction), RayType::Scattered, 1e-4f, INFINITY);
        }

        RGBSpectrum &contribution = pv_outputBuffer[sm_launchIndex];
        //contribution += payload.contribution;
        contribution = payload.contribution;
    }

    // Exception Program
    RT_PROGRAM void exception() {
        //uint32_t code = rtGetExceptionCode();
        rtPrintExceptionDetails();
    }
}
