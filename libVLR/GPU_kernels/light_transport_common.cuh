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
        uint32_t data[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];

        //ProgSigBSDFGetBaseColor progGetBaseColor;
        ProgSigBSDFmatches progMatches;
        ProgSigBSDFSampleInternal progSampleInternal;
        ProgSigBSDFEvaluateInternal progEvaluateInternal;
        ProgSigBSDFEvaluatePDFInternal progEvaluatePDFInternal;

        RT_FUNCTION bool matches(DirectionType dirType) {
            return progMatches((const uint32_t*)this, dirType);
        }
        RT_FUNCTION SampledSpectrum sampleInternal(const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
            return progSampleInternal((const uint32_t*)this, query, uComponent, uDir, result);
        }
        RT_FUNCTION SampledSpectrum evaluateInternal(const BSDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluateInternal((const uint32_t*)this, query, dirLocal);
        }
        RT_FUNCTION float evaluatePDFInternal(const BSDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluatePDFInternal((const uint32_t*)this, query, dirLocal);
        }

    public:
        RT_FUNCTION BSDF(const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            ProgSigSetupBSDF setupBSDF = (ProgSigSetupBSDF)matDesc.progSetupBSDF;
            setupBSDF(matDesc.data, surfPt, wls, (uint32_t*)this);

            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];

            //progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;
            progMatches = (ProgSigBSDFmatches)procSet.progMatches;
            progSampleInternal = (ProgSigBSDFSampleInternal)procSet.progSampleInternal;
            progEvaluateInternal = (ProgSigBSDFEvaluateInternal)procSet.progEvaluateInternal;
            progEvaluatePDFInternal = (ProgSigBSDFEvaluatePDFInternal)procSet.progEvaluatePDFInternal;
        }

        //RT_FUNCTION SampledSpectrum getBaseColor() {
        //    return progGetBaseColor((const uint32_t*)this);
        //}

        RT_FUNCTION bool hasNonDelta() {
            return matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
        }

        RT_FUNCTION SampledSpectrum sample(const BSDFQuery &query, const BSDFSample &sample, BSDFQueryResult* result) {
            if (!matches(query.dirTypeFilter)) {
                result->dirPDF = 0.0f;
                result->sampledType = DirectionType();
                return SampledSpectrum::Zero();
            }
            SampledSpectrum fs_sn = sampleInternal(query, sample.uComponent, sample.uDir, result);
            VLRAssert((result->dirPDF > 0 && fs_sn.allPositiveFinite()) || result->dirPDF == 0,
                      "Invalid BSDF value.\ndirV: (%g, %g, %g), sample: (%g, %g, %g), dirPDF: %g", 
                      query.dirLocal.x, query.dirLocal.y, query.dirLocal.z, sample.uComponent, sample.uDir[0], sample.uDir[1],
                      result->dirPDF);
            float snCorrection = std::fabs(result->dirLocal.z / dot(result->dirLocal, query.geometricNormalLocal));
            return fs_sn * snCorrection;
        }

        RT_FUNCTION SampledSpectrum evaluate(const BSDFQuery &query, const Vector3D &dirLocal) {
            SampledSpectrum fs_sn = evaluateInternal(query, dirLocal);
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
        uint32_t data[VLR_MAX_NUM_EDF_PARAMETER_SLOTS];

        ProgSigEDFEvaluateEmittanceInternal progEvaluateEmittanceInternal;
        ProgSigEDFEvaluateInternal progEvaluateInternal;

        RT_FUNCTION SampledSpectrum evaluateEmittanceInternal() {
            return progEvaluateEmittanceInternal((const uint32_t*)this);
        }
        RT_FUNCTION SampledSpectrum evaluateInternal(const EDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluateInternal((const uint32_t*)this, query, dirLocal);
        }

    public:
        RT_FUNCTION EDF(const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            ProgSigSetupEDF setupEDF = (ProgSigSetupEDF)matDesc.progSetupEDF;
            setupEDF(matDesc.data, surfPt, wls, (uint32_t*)this);

            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[matDesc.edfProcedureSetIndex];

            progEvaluateEmittanceInternal = (ProgSigEDFEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;
            progEvaluateInternal = (ProgSigEDFEvaluateInternal)procSet.progEvaluateInternal;
        }

        RT_FUNCTION SampledSpectrum evaluateEmittance() {
            SampledSpectrum Le0 = evaluateEmittanceInternal();
            return Le0;
        }

        RT_FUNCTION SampledSpectrum evaluate(const EDFQuery &query, const Vector3D &dirLocal) {
            SampledSpectrum Le1 = evaluateInternal(query, dirLocal);
            return Le1;
        }
    };



    struct Payload {
        struct {
            bool terminate : 1;
            bool maxLengthTerminate : 1;
        };
        KernelRNG rng;
        float initImportance;
        WavelengthSamples wls;
        SampledSpectrum alpha;
        SampledSpectrum contribution;
        Point3D origin;
        Vector3D direction;
        float prevDirPDF;
        DirectionType prevSampledType;
    };

    struct ShadowPayload {
        WavelengthSamples wls;
        float fractionalVisibility;
    };



    rtDeclareVariable(optix::uint2, sm_launchIndex, rtLaunchIndex, );
    rtDeclareVariable(Payload, sm_payload, rtPayload, );
    rtDeclareVariable(ShadowPayload, sm_shadowPayload, rtPayload, );

    typedef rtCallableProgramX<SampledSpectrum(const WavelengthSamples &, const LensPosSample &, LensPosQueryResult*)> ProgSigSampleLensPosition;
    typedef rtCallableProgramX<SampledSpectrum(const SurfacePoint &, const WavelengthSamples &, const IDFSample &, IDFQueryResult*)> ProgSigSampleIDF;

    typedef rtCallableProgramX<TexCoord2D(const HitPointParameter &)> ProgSigDecodeTexCoord;
    typedef rtCallableProgramX<void(const HitPointParameter &, SurfacePoint*, float*)> ProgSigDecodeHitPoint;
    typedef rtCallableProgramX<float(const TexCoord2D &)> ProgSigFetchAlpha;
    typedef rtCallableProgramX<Normal3D(const TexCoord2D &)> ProgSigFetchNormal;

    // per GeometryInstance
    rtDeclareVariable(ProgSigDecodeTexCoord, pv_progDecodeTexCoord, , );
    rtDeclareVariable(ProgSigDecodeHitPoint, pv_progDecodeHitPoint, , );
    rtDeclareVariable(TangentType, pv_tangentType, , ) = TangentType::TC0Direction;
    rtDeclareVariable(ShaderNodeSocketID, pv_nodeNormal, , );
    rtDeclareVariable(ShaderNodeSocketID, pv_nodeAlpha, , );
    rtDeclareVariable(uint32_t, pv_materialIndex, , );
    rtDeclareVariable(float, pv_importance, , );



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
        shadowPayload.wls = sm_payload.wls;
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
        VLRAssert(std::isfinite(phi) && std::isfinite(theta), "\"phi\", \"theta\": Not finite values %g, %g.", phi, theta);
        float uvPDF = pv_envLightDescriptor.body.asEnvironmentLight.importanceMap.evaluatePDF(phi / (2 * M_PIf), theta / M_PIf);
        return uvPDF / (2 * M_PIf * M_PIf * std::sin(theta));
    }

    // END: Light
    // ----------------------------------------------------------------



    RT_PROGRAM void shadowAnyHitDefault() {
        sm_shadowPayload.fractionalVisibility = 0.0f;
        rtTerminateRay();
    }

    // Common Any Hit Program for All Primitive Types and Materials for non-shadow rays
    RT_PROGRAM void anyHitWithAlpha() {
        HitPointParameter hitPointParam = a_hitPointParam;
        SurfacePoint surfPt;
        float hypAreaPDF;
        pv_progDecodeHitPoint(hitPointParam, &surfPt, &hypAreaPDF);

        float alpha = calcNode(pv_nodeAlpha, 1.0f, surfPt, sm_payload.wls);

        // Stochastic Alpha Test
        if (sm_payload.rng.getFloat0cTo1o() >= alpha)
            rtIgnoreIntersection();
    }

    // Common Any Hit Program for All Primitive Types and Materials for shadow rays
    RT_PROGRAM void shadowAnyHitWithAlpha() {
        HitPointParameter hitPointParam = a_hitPointParam;
        SurfacePoint surfPt;
        float hypAreaPDF;
        pv_progDecodeHitPoint(hitPointParam, &surfPt, &hypAreaPDF);

        float alpha = calcNode(pv_nodeAlpha, 1.0f, surfPt, sm_shadowPayload.wls);

        sm_shadowPayload.fractionalVisibility *= (1 - alpha);
        if (sm_shadowPayload.fractionalVisibility == 0.0f)
            rtTerminateRay();
    }



    RT_FUNCTION void modifyTangent(SurfacePoint* surfPt) {
        if (pv_tangentType == TangentType::TC0Direction)
            return;

        Point3D localPosition = transform(RT_WORLD_TO_OBJECT, surfPt->position);

        Vector3D localTangent;
        if (pv_tangentType == TangentType::RadialX) {
            localTangent = Vector3D(0, -localPosition.z, localPosition.y);
        }
        else if (pv_tangentType == TangentType::RadialY) {
            localTangent = Vector3D(localPosition.z, 0, -localPosition.x);
        }
        else {
            localTangent = Vector3D(-localPosition.y, localPosition.x, 0);
        }

        Vector3D newTangent = normalize(transform(RT_OBJECT_TO_WORLD, localTangent));

        float dotNT = dot(surfPt->shadingFrame.z, newTangent);
        surfPt->shadingFrame.x = normalize(newTangent - dotNT * surfPt->shadingFrame.z);
        surfPt->shadingFrame.y = cross(surfPt->shadingFrame.z, surfPt->shadingFrame.x);
    }



    RT_FUNCTION Normal3D fetchNormal(const SurfacePoint &surfPt) {
        optix::float3 value = calcNode(pv_nodeNormal, optix::make_float3(0.5f, 0.5f, 1.0f), surfPt, sm_payload.wls);
        Normal3D normalLocal = 2 * Normal3D(value.x, value.y, value.z) - 1.0f;
        normalLocal.y *= -1; // for DirectX format normal map
        return normalLocal;
    }

    // JP: 法線マップに従ってシェーディングフレームを変更する。
    // EN: perturb the shading frame according to the normal map.
    RT_FUNCTION void applyBumpMapping(const Normal3D &modNormalInTF, SurfacePoint* surfPt) {
        if (modNormalInTF.x == 0.0f && modNormalInTF.y == 0.0f)
            return;

        const ReferenceFrame &originalFrame = surfPt->shadingFrame;

        // JP: テクスチャーフレームもシェーディングフレームもこの時点ではz軸は共通。
        //     後者に対する前者の角度を求める。
        // EN: z axes of the texture frame and the shading frame are the same at this moment.
        //     calculate the angle of the latter to the former.
        Vector3D tc0Direction = surfPt->tc0Direction;
        Vector3D tc1Direction = cross(originalFrame.z, tc0Direction);
        float tlx = dot(originalFrame.x, tc0Direction);
        float tly = dot(originalFrame.x, tc1Direction);
        float angleFromTexFrame = std::atan2(tly, tlx);

        // JP: 法線マップの値はテクスチャーフレーム内で定義されているためシェーディングフレーム内に変換。
        // EN: convert a normal map value to that in the shading frame because the value is defined in the texture frame.
        float cosTFtoSF, sinTFtoSF;
        VLR::sincos(angleFromTexFrame, &sinTFtoSF, &cosTFtoSF);
        Normal3D modNormalInSF = Normal3D(cosTFtoSF * modNormalInTF.x + sinTFtoSF * modNormalInTF.y,
                                          -sinTFtoSF * modNormalInTF.x + cosTFtoSF * modNormalInTF.y,
                                          modNormalInTF.z);

        // JP: 法線から回転軸と回転角(、Quaternion)を求めて対応する接平面ベクトルを求める。
        // EN: calculate a rotating axis and an angle (and quaternion) from the normal then calculate corresponding tangential vectors.
        float projLength = std::sqrt(modNormalInSF.x * modNormalInSF.x + modNormalInSF.y * modNormalInSF.y);
        float tiltAngle = std::atan(projLength / modNormalInSF.z);
        float qSin, qCos;
        VLR::sincos(tiltAngle / 2, &qSin, &qCos);
        float qX = (-modNormalInSF.y / projLength) * qSin;
        float qY = (modNormalInSF.x / projLength) * qSin;
        float qW = qCos;
        Vector3D modTangentInSF = Vector3D(1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
        Vector3D modBitangentInSF = Vector3D(2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

        Matrix3x3 matSFtoW = Matrix3x3(originalFrame.x, originalFrame.y, originalFrame.z);
        ReferenceFrame bumpShadingFrame(matSFtoW * modTangentInSF, matSFtoW * modBitangentInSF, matSFtoW * modNormalInSF);

        surfPt->shadingFrame = bumpShadingFrame;
    }



    RT_FUNCTION void calcSurfacePoint(SurfacePoint* surfPt, float* hypAreaPDF) {
        HitPointParameter hitPointParam = a_hitPointParam;
        pv_progDecodeHitPoint(hitPointParam, surfPt, hypAreaPDF);

        modifyTangent(surfPt);
        applyBumpMapping(fetchNormal(*surfPt), surfPt);
    }
}
