#include "light_transport_common.cuh"

namespace VLR {
    struct DebugRenderingPayload {
        KernelRNG rng;
        WavelengthSamples wls;
        SampledSpectrum value;
    };

#define DebugPayloadSignature DebugRenderingPayload*



    // for debug rendering
    CUDA_DEVICE_FUNCTION TripletSpectrum debugRenderingAttributeToSpectrum(const SurfacePoint &surfPt, DebugRenderingAttribute attribute) {
        TripletSpectrum value;

        switch (attribute) {
        case DebugRenderingAttribute::GeometricNormal:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.geometricNormal.x),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.geometricNormal.y),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.geometricNormal.z));
            break;
        case DebugRenderingAttribute::ShadingTangent:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.x.x),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.x.y),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.x.z));
            break;
        case DebugRenderingAttribute::ShadingBitangent:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.y.x),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.y.y),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.y.z));
            break;
        case DebugRenderingAttribute::ShadingNormal:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.z.x),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.z.y),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.shadingFrame.z.z));
            break;
        case DebugRenderingAttribute::TextureCoordinates:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          surfPt.texCoord.u - std::floor(surfPt.texCoord.u),
                                          surfPt.texCoord.v - std::floor(surfPt.texCoord.v),
                                          0.0f);
            break;
        case DebugRenderingAttribute::GeometricVsShadingNormal: {
            float sim = dot(surfPt.geometricNormal, surfPt.shadingFrame.z);
            bool opposite = sim < 0.0f;
            sim = std::fabs(sim);
            const float coeff = 5.0f;
            float sValue = 0.5f + coeff * (sim - 1);
            sValue = clamp(sValue, 0.0f, 1.0f);
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65, sValue, opposite ? 0 : sValue, opposite ? 0 : sValue);
            break;
        }
        case DebugRenderingAttribute::ShadingFrameLengths:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          clamp(0.5f + 10 * (surfPt.shadingFrame.x.length() - 1), 0.0f, 1.0f),
                                          clamp(0.5f + 10 * (surfPt.shadingFrame.y.length() - 1), 0.0f, 1.0f),
                                          clamp(0.5f + 10 * (surfPt.shadingFrame.z.length() - 1), 0.0f, 1.0f));
            break;
        case DebugRenderingAttribute::ShadingFrameOrthogonality:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          clamp(0.5f + 100 * dot(surfPt.shadingFrame.x, surfPt.shadingFrame.y), 0.0f, 1.0f),
                                          clamp(0.5f + 100 * dot(surfPt.shadingFrame.y, surfPt.shadingFrame.z), 0.0f, 1.0f),
                                          clamp(0.5f + 100 * dot(surfPt.shadingFrame.z, surfPt.shadingFrame.x), 0.0f, 1.0f));
            break;
        default:
            break;
        }

        return value;
    }




    // Common Any Hit Program for All Primitive Types and Materials
    CUDA_DEVICE_KERNEL void RT_AH_NAME(debugRenderingAnyHitWithAlpha)() {
        DebugRenderingPayload* payload;
        optixu::getPayloads<DebugPayloadSignature>(&payload);

        float alpha = getAlpha(payload->wls);

        // Stochastic Alpha Test
        if (payload->rng.getFloat0cTo1o() >= alpha)
            optixIgnoreIntersection();
    }



    // Common Closest Hit Program for All Primitive Types and Materials
    CUDA_DEVICE_KERNEL void RT_CH_NAME(debugRenderingClosestHit)() {
        const auto &sbtr = HitGroupSBTRecordData::get();

        DebugRenderingPayload* payload;
        optixu::getPayloads<DebugPayloadSignature>(&payload);

        WavelengthSamples &wls = payload->wls;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(wls, &surfPt, &hypAreaPDF);

        //if (!surfPt.shadingFrame.x.allFinite() || !surfPt.shadingFrame.y.allFinite() || !surfPt.shadingFrame.z.allFinite())
        //    vlrprintf("(%g, %g, %g), (%g, %g, %g), (%g, %g, %g)\n",
        //              surfPt.shadingFrame.x.x, surfPt.shadingFrame.x.y, surfPt.shadingFrame.x.z,
        //              surfPt.shadingFrame.y.x, surfPt.shadingFrame.y.y, surfPt.shadingFrame.y.z,
        //              surfPt.shadingFrame.z.x, surfPt.shadingFrame.z.y, surfPt.shadingFrame.z.z);

        if (plp.debugRenderingAttribute == DebugRenderingAttribute::BaseColor) {
            const SurfaceMaterialDescriptor matDesc = plp.materialDescriptorBuffer[sbtr.geomInst.materialIndex];
            BSDF bsdf(matDesc, surfPt, wls);

            const BSDFProcedureSet procSet = plp.bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];
            auto progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;

            payload->value = progGetBaseColor((const uint32_t*)&bsdf);
        }
        else {
            payload->value = debugRenderingAttributeToSpectrum(surfPt, plp.debugRenderingAttribute).evaluate(wls);
        }
    }



    // JP: 本当は無限大の球のIntersection/Bounding Box Programを使用して環境光に関する処理もClosest Hit Programで統一的に行いたい。
    //     が、OptiXのBVHビルダーがLBVHベースなので無限大のAABBを生成するのは危険。
    //     仕方なくMiss Programで環境光を処理する。
    CUDA_DEVICE_KERNEL void RT_MS_NAME(debugRenderingMiss)() {
        DebugRenderingPayload* payload;
        optixu::getPayloads<DebugPayloadSignature>(&payload);

        WavelengthSamples &wls = payload->wls;

        Vector3D direction = asVector3D(optixGetWorldRayDirection());
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        float sinPhi, cosPhi;
        VLR::sincos(phi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));
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
        phi += plp.envLightDescriptor.rotationPhi;
        phi = phi - std::floor(phi / (2 * VLR_M_PI)) * 2 * VLR_M_PI;
        surfPt.texCoord = TexCoord2D(phi / (2 * VLR_M_PI), theta / VLR_M_PI);

        if (plp.debugRenderingAttribute == DebugRenderingAttribute::BaseColor) {
            payload->value = SampledSpectrum::Zero();
        }
        else {
            payload->value = debugRenderingAttributeToSpectrum(surfPt, plp.debugRenderingAttribute).evaluate(wls);
        }
    }



    // Common Ray Generation Program for All Camera Types
    CUDA_DEVICE_KERNEL void RT_RG_NAME(debugRenderingRayGeneration)() {
        uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

        KernelRNG rng = plp.rngBuffer[launchIndex];

        float2 p = make_float2(launchIndex.x + rng.getFloat0cTo1o(),
                               launchIndex.y + rng.getFloat0cTo1o());

        float selectWLPDF;
        WavelengthSamples wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &selectWLPDF);

        ProgSigSampleLensPosition sampleLensPosition(plp.progSampleLensPosition);
        ProgSigSampleIDF sampleIDF(plp.progSampleIDF);

        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        SampledSpectrum We0 = sampleLensPosition(wls, We0Sample, &We0Result);

        IDFSample We1Sample(p.x / plp.imageSize.x, p.y / plp.imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = sampleIDF(We0Result.surfPt, wls, We1Sample, &We1Result);

        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        SampledSpectrum alpha = (We0 * We1) * (We0Result.surfPt.calcCosTerm(rayDir) / (We0Result.areaPDF * We1Result.dirPDF * selectWLPDF));

        DebugRenderingPayload payload;
        payload.rng = rng;
        payload.wls = wls;
        DebugRenderingPayload* payloadPtr = &payload;
        optixu::trace<DebugPayloadSignature>(
            plp.topGroup, asOptiXType(We0Result.surfPt.position), asOptiXType(rayDir), 0.0f, FLT_MAX, 0.0f,
            0xFF, OPTIX_RAY_FLAG_NONE,
            RayType::DebugPrimary, RayType::NumTypes, RayType::DebugPrimary,
            payloadPtr);

        plp.rngBuffer[launchIndex] = payload.rng;

        if (!payload.value.allFinite()) {
            vlrprintf("Pass %u, (%u, %u): Not a finite value.\n", plp.numAccumFrames, launchIndex.x, launchIndex.y);
            return;
        }

        if (plp.numAccumFrames == 1)
            plp.outputBuffer[launchIndex].reset();
        plp.outputBuffer[launchIndex].add(wls, payload.value / selectWLPDF);
    }
}
