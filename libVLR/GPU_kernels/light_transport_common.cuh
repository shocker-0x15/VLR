#pragma once

#include "kernel_common.cuh"

namespace VLR {
    // Context-scope Variables
    rtDeclareVariable(rtObject, pv_topGroup, , );

    rtDeclareVariable(DiscreteDistribution1D, pv_lightImpDist, , );
    rtBuffer<SurfaceLightDescriptor> pv_surfaceLightDescriptorBuffer;
    rtDeclareVariable(SurfaceLightDescriptor, pv_envLightDescriptor, , );



    class BSDF {
#define VLR_MAX_NUM_BSDF_PARAMETER_SLOTS (32)
        union {
            int32_t i1[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];
            uint32_t ui1[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];
            float f1[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];

            optix::int2 i2[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS >> 1];
            optix::float2 f2[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS >> 1];

            optix::float4 f4[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS >> 2];
        };

        //ProgSigBSDFGetBaseColor progGetBaseColor;
        ProgSigBSDFmatches progMatches;
        ProgSigBSDFSampleInternal progSampleInternal;
        ProgSigBSDFEvaluateInternal progEvaluateInternal;
        ProgSigBSDFEvaluatePDFInternal progEvaluatePDFInternal;

        RT_FUNCTION bool matches(DirectionType dirType) {
            return progMatches((const uint32_t*)this, dirType);
        }
        RT_FUNCTION RGBSpectrum sampleInternal(const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
            return progSampleInternal((const uint32_t*)this, query, uComponent, uDir, result);
        }
        RT_FUNCTION RGBSpectrum evaluateInternal(const BSDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluateInternal((const uint32_t*)this, query, dirLocal);
        }
        RT_FUNCTION float evaluatePDFInternal(const BSDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluatePDFInternal((const uint32_t*)this, query, dirLocal);
        }

    public:
        RT_FUNCTION BSDF(const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, bool wavelengthSelected) {
            SurfaceMaterialHead &head = *(SurfaceMaterialHead*)&matDesc.i1[0];

            ProgSigSetupBSDF setupBSDF = (ProgSigSetupBSDF)head.progSetupBSDF;
            setupBSDF((const uint32_t*)&matDesc, surfPt, wavelengthSelected, (uint32_t*)this);

            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[head.bsdfProcedureSetIndex];

            //progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;
            progMatches = (ProgSigBSDFmatches)procSet.progMatches;
            progSampleInternal = (ProgSigBSDFSampleInternal)procSet.progSampleInternal;
            progEvaluateInternal = (ProgSigBSDFEvaluateInternal)procSet.progEvaluateInternal;
            progEvaluatePDFInternal = (ProgSigBSDFEvaluatePDFInternal)procSet.progEvaluatePDFInternal;
        }

        //RT_FUNCTION RGBSpectrum getBaseColor() {
        //    return progGetBaseColor((const uint32_t*)this);
        //}

        RT_FUNCTION bool hasNonDelta() {
            return matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
        }

        RT_FUNCTION RGBSpectrum sample(const BSDFQuery &query, const BSDFSample &sample, BSDFQueryResult* result) {
            if (!matches(query.dirTypeFilter)) {
                result->dirPDF = 0.0f;
                result->sampledType = DirectionType();
                return RGBSpectrum::Zero();
            }
            RGBSpectrum fs_sn = sampleInternal(query, sample.uComponent, sample.uDir, result);
            VLRAssert((result->dirPDF > 0 && fs_sn.allPositiveFinite()) || result->dirPDF == 0,
                      "Invalid BSDF value.\ndirPDF: %g", result->dirPDF);
            float snCorrection = std::fabs(result->dirLocal.z / dot(result->dirLocal, query.geometricNormalLocal));
            return fs_sn * snCorrection;
        }

        RT_FUNCTION RGBSpectrum evaluate(const BSDFQuery &query, const Vector3D &dirLocal) {
            RGBSpectrum fs_sn = evaluateInternal(query, dirLocal);
            float snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
            return fs_sn * snCorrection;
        }

        RT_FUNCTION float evaluatePDF(const BSDFQuery &query, const Vector3D &dirLocal) {
            if (!matches(query.dirTypeFilter)) {
                return 0;
            }
            float ret = evaluatePDFInternal(query, dirLocal);
            return ret;
        }
    };



    class EDF {
#define VLR_MAX_NUM_EDF_PARAMETER_SLOTS (8)
        union {
            int32_t i1[VLR_MAX_NUM_EDF_PARAMETER_SLOTS];
            uint32_t ui1[VLR_MAX_NUM_EDF_PARAMETER_SLOTS];
            float f1[VLR_MAX_NUM_EDF_PARAMETER_SLOTS];

            optix::int2 i2[VLR_MAX_NUM_EDF_PARAMETER_SLOTS >> 1];
            optix::float2 f2[VLR_MAX_NUM_EDF_PARAMETER_SLOTS >> 1];

            optix::float4 f4[VLR_MAX_NUM_EDF_PARAMETER_SLOTS >> 2];
        };

        ProgSigEDFEvaluateEmittanceInternal progEvaluateEmittanceInternal;
        ProgSigEDFEvaluateInternal progEvaluateInternal;

        RT_FUNCTION RGBSpectrum evaluateEmittanceInternal() {
            return progEvaluateEmittanceInternal((const uint32_t*)this);
        }
        RT_FUNCTION RGBSpectrum evaluateInternal(const EDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluateInternal((const uint32_t*)this, query, dirLocal);
        }

    public:
        RT_FUNCTION EDF(const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt) {
            SurfaceMaterialHead &head = *(SurfaceMaterialHead*)&matDesc.i1[0];

            ProgSigSetupEDF setupEDF = (ProgSigSetupEDF)head.progSetupEDF;
            setupEDF((const uint32_t*)&matDesc, surfPt, (uint32_t*)this);

            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[head.edfProcedureSetIndex];

            progEvaluateEmittanceInternal = (ProgSigEDFEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;
            progEvaluateInternal = (ProgSigEDFEvaluateInternal)procSet.progEvaluateInternal;
        }

        RT_FUNCTION RGBSpectrum evaluateEmittance() {
            RGBSpectrum Le0 = evaluateEmittanceInternal();
            return Le0;
        }

        RT_FUNCTION RGBSpectrum evaluate(const EDFQuery &query, const Vector3D &dirLocal) {
            RGBSpectrum Le1 = evaluateInternal(query, dirLocal);
            return Le1;
        }
    };



    struct Payload {
        struct {
            unsigned int wlHint : 29;
            bool wavelengthSelected : 1;
            bool terminate : 1;
            bool maxLengthTerminate : 1;
        };
        KernelRNG rng;
        float initImportance;
        RGBSpectrum alpha;
        RGBSpectrum contribution;
        Point3D origin;
        Vector3D direction;
        float prevDirPDF;
        DirectionType prevSampledType;
    };

    struct ShadowPayload {
        float fractionalVisibility;
    };



    // ----------------------------------------------------------------
    // Light

    RT_FUNCTION bool testVisibility(const SurfacePoint &shadingSurfacePoint, const SurfacePoint &lightSurfacePoint,
                                    Vector3D* shadowRayDir, float* squaredDistance, float* fractionalVisibility) {
        VLRAssert(shadingSurfacePoint.atInfinity == false, "Shading point must be in finite region.");

        *shadowRayDir = lightSurfacePoint.calcDirectionFrom(shadingSurfacePoint.position, squaredDistance);
        optix::Ray shadowRay = optix::make_Ray(asOptiXType(shadingSurfacePoint.position), asOptiXType(*shadowRayDir), RayType::Shadow, 1e-4f, FLT_MAX);
        if (!lightSurfacePoint.atInfinity)
            shadowRay.tmax = std::sqrt(*squaredDistance) * 0.9999f;

        ShadowPayload shadowPayload;
        shadowPayload.fractionalVisibility = 1.0f;
        rtTrace(pv_topGroup, shadowRay, shadowPayload);

        *fractionalVisibility = shadowPayload.fractionalVisibility;

        return *fractionalVisibility > 0;
    }

    RT_FUNCTION void selectSurfaceLight(float lightSample, SurfaceLight* light, float* lightProb, float* remapped) {
        float sumImps = pv_envLightDescriptor.importance + pv_lightImpDist.integral();
        float su = sumImps * lightSample;
        if (su < pv_envLightDescriptor.importance) {
            *light = SurfaceLight(pv_envLightDescriptor);
            *lightProb = pv_envLightDescriptor.importance / sumImps;
        }
        else {
            lightSample = (su - pv_envLightDescriptor.importance) / pv_lightImpDist.integral();
            uint32_t lightIdx = pv_lightImpDist.sample(lightSample, lightProb, remapped);
            *light = SurfaceLight(pv_surfaceLightDescriptorBuffer[lightIdx]);
            *lightProb *= pv_lightImpDist.integral() / sumImps;
        }
    }

    RT_FUNCTION float getSumLightImportances() {
        return pv_envLightDescriptor.importance + pv_lightImpDist.integral();
    }

    RT_FUNCTION float evaluateEnvironmentAreaPDF(float phi, float theta) {
        float uvPDF = pv_envLightDescriptor.body.asEnvironmentLight.importanceMap.evaluatePDF(phi / (2 * M_PIf), theta / M_PIf);
        return uvPDF / (2 * M_PIf * M_PIf * std::sin(theta));
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
        normalLocal.y *= -1; // for DirectX format normal map
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
    rtDeclareVariable(Payload, sm_payload, rtPayload, );
    rtDeclareVariable(ShadowPayload, sm_shadowPayload, rtPayload, );

    typedef rtCallableProgramX<RGBSpectrum(const LensPosSample &, LensPosQueryResult*)> ProgSigSampleLensPosition;
    typedef rtCallableProgramX<RGBSpectrum(const SurfacePoint &, const IDFSample &, IDFQueryResult*)> ProgSigSampleIDF;

    typedef rtCallableProgramX<TexCoord2D(const HitPointParameter &)> ProgSigDecodeTexCoord;
    typedef rtCallableProgramX<void(const HitPointParameter &, SurfacePoint*, float*)> ProgSigDecodeHitPoint;
    typedef rtCallableProgramX<float(const TexCoord2D &)> ProgSigFetchAlpha;
    typedef rtCallableProgramX<Normal3D(const TexCoord2D &)> ProgSigFetchNormal;

    // per GeometryInstance
    rtDeclareVariable(ProgSigDecodeTexCoord, pv_progDecodeTexCoord, , );
    rtDeclareVariable(ProgSigFetchAlpha, pv_progFetchAlpha, , );
    rtDeclareVariable(ProgSigFetchNormal, pv_progFetchNormal, , );

    RT_PROGRAM void shadowAnyHitDefault() {
        sm_shadowPayload.fractionalVisibility = 0.0f;
        rtTerminateRay();
    }

    // Common Any Hit Program for All Primitive Types and Materials for non-shadow rays
    RT_PROGRAM void anyHitWithAlpha() {
        HitPointParameter hitPointParam = a_hitPointParam;
        TexCoord2D texCoord = pv_progDecodeTexCoord(hitPointParam);

        float alpha = pv_progFetchAlpha(texCoord);

        // Stochastic Alpha Test
        if (sm_payload.rng.getFloat0cTo1o() >= alpha)
            rtIgnoreIntersection();
    }

    // Common Any Hit Program for All Primitive Types and Materials for shadow rays
    RT_PROGRAM void shadowAnyHitWithAlpha() {
        HitPointParameter hitPointParam = a_hitPointParam;
        TexCoord2D texCoord = pv_progDecodeTexCoord(hitPointParam);

        float alpha = pv_progFetchAlpha(texCoord);

        sm_shadowPayload.fractionalVisibility *= (1 - alpha);
        if (sm_shadowPayload.fractionalVisibility == 0.0f)
            rtTerminateRay();
    }
}
