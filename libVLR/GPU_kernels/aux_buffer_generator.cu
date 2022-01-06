#include "../shared/light_transport_common.h"

namespace vlr {
    using namespace shared;

    CUDA_DEVICE_KERNEL void RT_AH_NAME(auxBufferGeneratorAnyHitWithAlpha)() {
        WavelengthSamples wls;
        KernelRNG rng;
        AuxBufGenPayloadSignature::get(&wls, &rng, nullptr, nullptr);

        float alpha = getAlpha(wls);

        // Stochastic Alpha Test
        if (rng.getFloat0cTo1o() >= alpha)
            optixIgnoreIntersection();

        AuxBufGenPayloadSignature::set(nullptr, &rng, nullptr, nullptr);
    }



    CUDA_DEVICE_KERNEL void RT_RG_NAME(auxBufferGenerator)() {
        uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

        KernelRNG rng = plp.rngBuffer.read(launchIndex);

        float2 p = make_float2(launchIndex.x + rng.getFloat0cTo1o(),
                               launchIndex.y + rng.getFloat0cTo1o());

        float selectWLPDF;
        WavelengthSamples wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &selectWLPDF);

        Camera camera(static_cast<ProgSigCamera_sample>(plp.progSampleLensPosition));
        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        camera.sample(We0Sample, &We0Result);

        IDF idf(plp.cameraDescriptor, We0Result.surfPt, wls);

        idf.evaluateSpatialImportance();

        IDFSample We1Sample(p.x / plp.imageSize.x, p.y / plp.imageSize.y);
        IDFQueryResult We1Result;
        idf.sample(IDFQuery(), We1Sample, &We1Result);

        Point3D rayOrg = We0Result.surfPt.position;
        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);

        SampledSpectrum firstHitAlbedo;
        Normal3D firstHitNormal;
        optixu::trace<AuxBufGenPayloadSignature>(
            plp.topGroup, asOptiXType(rayOrg), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
            VisibilityGroup_Everything, OPTIX_RAY_FLAG_NONE,
            AuxBufGenRayType::Primary, MaxNumRayTypes, AuxBufGenRayType::Primary,
            wls, rng, firstHitAlbedo, firstHitNormal);

        uint32_t linearIndex = launchIndex.y * plp.imageStrideInPixels + launchIndex.x;
        DiscretizedSpectrum &accumAlbedo = plp.accumAlbedoBuffer[linearIndex];
        Normal3D &accumNormal = plp.accumNormalBuffer[linearIndex];
        if (plp.numAccumFrames == 1) {
            accumAlbedo = DiscretizedSpectrum::Zero();
            accumNormal = Normal3D(0.0f, 0.0f, 0.0f);
        }
        TripletSpectrum whitePoint = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                                           1, 1, 1);
        accumAlbedo += DiscretizedSpectrum(wls, firstHitAlbedo * whitePoint.evaluate(wls) / selectWLPDF);
        accumNormal += firstHitNormal;

        plp.rngBuffer.write(launchIndex, rng);
    }



    CUDA_DEVICE_KERNEL void RT_CH_NAME(auxBufferGeneratorFirstHit)() {
        const auto hp = HitPointParameter::get();

        WavelengthSamples wls;
        AuxBufGenPayloadSignature::get(&wls, nullptr, nullptr, nullptr);

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(hp, wls, &surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[hp.sbtr->geomInst.materialIndex];
        constexpr TransportMode transportMode = TransportMode::Radiance;
        BSDF<transportMode> bsdf(matDesc, surfPt, wls);
        
        SampledSpectrum firstHitAlbedo = bsdf.getBaseColor();
        Normal3D firstHitNormal = surfPt.shadingFrame.z;
        AuxBufGenPayloadSignature::set(nullptr, nullptr, &firstHitAlbedo, &firstHitNormal);
    }



    CUDA_DEVICE_KERNEL void RT_MS_NAME(auxBufferGeneratorMiss)() {
        SampledSpectrum firstHitAlbedo = SampledSpectrum::Zero();
        Normal3D firstHitNormal = Normal3D(0.0f, 0.0f, 0.0f);
        AuxBufGenPayloadSignature::set(nullptr, nullptr, &firstHitAlbedo, &firstHitNormal);
    }
}
