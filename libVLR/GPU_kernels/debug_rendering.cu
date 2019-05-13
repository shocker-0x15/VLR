#include "light_transport_common.cuh"

namespace VLR {
    struct DebugRenderingPayload {
        WavelengthSamples wls;
        SampledSpectrum value;
    };

    rtDeclareVariable(DebugRenderingPayload, sm_debugPayload, rtPayload, );
    rtDeclareVariable(DebugRenderingAttribute, pv_debugRenderingAttribute, , );

    // Context-scope Variables
    rtDeclareVariable(optix::uint2, pv_imageSize, , );
    rtDeclareVariable(uint32_t, pv_numAccumFrames, , );
    rtDeclareVariable(ProgSigSampleLensPosition, pv_progSampleLensPosition, , );
    rtDeclareVariable(ProgSigSampleIDF, pv_progSampleIDF, , );
    rtBuffer<KernelRNG, 2> pv_rngBuffer;
    rtBuffer<SpectrumStorage, 2> pv_outputBuffer;



    // for debug rendering
    RT_FUNCTION TripletSpectrum DebugRenderingAttributeToSpectrum(const SurfacePoint &surfPt, DebugRenderingAttribute attribute) {
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
        case DebugRenderingAttribute::TC0Direction:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.tc0Direction.x),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.tc0Direction.y),
                                          std::fmax(0.0f, 0.5f + 0.5f * surfPt.tc0Direction.z));
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
            sim = clamp(0.5f * std::fabs(sim), 0.0f, 1.0f);
            //float gLength = clamp(0.5f + 100 * (surfPt.geometricNormal.length() - 1), 0.0f, 1.0f);
            //float sLength = clamp(0.5f + 100 * (surfPt.shadingFrame.z.length() - 1), 0.0f, 1.0f);
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65, sim, opposite ? 0 : sim, opposite ? 0 : sim);
            break;
        }
        case DebugRenderingAttribute::ShadingFrameLengths:
            value = createTripletSpectrum(SpectrumType::LightSource, ColorSpace::Rec709_D65,
                                          clamp(0.5f + 100 * (surfPt.shadingFrame.x.length() - 1), 0.0f, 1.0f),
                                          clamp(0.5f + 100 * (surfPt.shadingFrame.y.length() - 1), 0.0f, 1.0f),
                                          clamp(0.5f + 100 * (surfPt.shadingFrame.z.length() - 1), 0.0f, 1.0f));
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



    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void debugRenderingClosestHit() {
        WavelengthSamples &wls = sm_payload.wls;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(&surfPt, &hypAreaPDF);

        if (pv_debugRenderingAttribute == DebugRenderingAttribute::BaseColor) {
            const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_materialIndex];
            BSDF bsdf(matDesc, surfPt, wls);

            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];
            auto progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;

            sm_debugPayload.value = progGetBaseColor((const uint32_t*)&bsdf);
        }
        else {
            sm_debugPayload.value = DebugRenderingAttributeToSpectrum(surfPt, pv_debugRenderingAttribute).evaluate(wls);
        }
    }



    // JP: 本当は無限大の球のIntersection/Bounding Box Programを使用して環境光に関する処理もClosest Hit Programで統一的に行いたい。
    //     が、OptiXのBVHビルダーがLBVHベースなので無限大のAABBを生成するのは危険。
    //     仕方なくMiss Programで環境光を処理する。
    RT_PROGRAM void debugRenderingMiss() {
        WavelengthSamples &wls = sm_payload.wls;

        Vector3D direction = asVector3D(sm_ray.direction);
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
        phi += pv_envLightDescriptor.body.asEnvironmentLight.rotationPhi;
        phi = phi - std::floor(phi / (2 * M_PIf)) * 2 * M_PIf;
        surfPt.texCoord = TexCoord2D(phi / (2 * M_PIf), theta / M_PIf);

        if (pv_debugRenderingAttribute == DebugRenderingAttribute::BaseColor) {
            sm_debugPayload.value = SampledSpectrum::Zero();
        }
        else {
            sm_debugPayload.value = DebugRenderingAttributeToSpectrum(surfPt, pv_debugRenderingAttribute).evaluate(wls);
        }
    }



    // Common Ray Generation Program for All Camera Types
    RT_PROGRAM void debugRenderingRayGeneration() {
        KernelRNG rng = pv_rngBuffer[sm_launchIndex];

        optix::float2 p = make_float2(sm_launchIndex.x + rng.getFloat0cTo1o(), sm_launchIndex.y + rng.getFloat0cTo1o());

        float selectWLPDF;
        WavelengthSamples wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &selectWLPDF);

        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        SampledSpectrum We0 = pv_progSampleLensPosition(wls, We0Sample, &We0Result);

        IDFSample We1Sample(p.x / pv_imageSize.x, p.y / pv_imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = pv_progSampleIDF(We0Result.surfPt, wls, We1Sample, &We1Result);

        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        SampledSpectrum alpha = (We0 * We1) * (We0Result.surfPt.calcCosTerm(rayDir) / (We0Result.areaPDF * We1Result.dirPDF * selectWLPDF));

        optix::Ray ray = optix::make_Ray(asOptiXType(We0Result.surfPt.position), asOptiXType(rayDir), RayType::DebugPrimary, 0.0f, FLT_MAX);

        DebugRenderingPayload payload;
        payload.wls = wls;
        rtTrace(pv_topGroup, ray, payload);
        pv_rngBuffer[sm_launchIndex] = rng;

        if (pv_numAccumFrames == 1)
            pv_outputBuffer[sm_launchIndex].reset();
        pv_outputBuffer[sm_launchIndex].add(wls, payload.value / selectWLPDF);
    }



    // Exception Program
    RT_PROGRAM void debugRenderingException() {
        //uint32_t code = rtGetExceptionCode();
        rtPrintExceptionDetails();
    }
}
