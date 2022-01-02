#pragma once

#include "kernel_common.h"

namespace vlr::shared {
    using DebugPayloadSignature = optixu::PayloadSignature<KernelRNG, WavelengthSamples, SampledSpectrum>;



    enum class BSDFTier {
        Debug = 0,
        Unidirectional,
        Bidirectional,
    };

    template <TransportMode transportMode, BSDFTier tier = BSDFTier::Unidirectional>
    class BSDF;

    template <TransportMode transportMode>
    class BSDF<transportMode, BSDFTier::Debug> {
#define VLR_MAX_NUM_BSDF_PARAMETER_SLOTS (32)
        uint32_t data[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];

        ProgSigBSDFGetBaseColor progGetBaseColor;

    public:
#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION BSDF(
            const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            auto setupBSDF = static_cast<ProgSigSetupBSDF>(matDesc.progSetupBSDF);
            setupBSDF(matDesc.data, surfPt, wls, reinterpret_cast<uint32_t*>(data));

            const BSDFProcedureSet procSet = plp.bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];

            progGetBaseColor = static_cast<ProgSigBSDFGetBaseColor>(procSet.progGetBaseColor);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return progGetBaseColor(data);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };
    
    template <TransportMode transportMode>
    class BSDF<transportMode, BSDFTier::Unidirectional> {
#define VLR_MAX_NUM_BSDF_PARAMETER_SLOTS (32)
        uint32_t data[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];

        ProgSigBSDFGetBaseColor progGetBaseColor;
        ProgSigBSDFmatches progMatches;
        ProgSigBSDFSampleInternal progSampleInternal;
        ProgSigBSDFEvaluateInternal progEvaluateInternal;
        ProgSigBSDFEvaluatePDFInternal progEvaluatePDFInternal;

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION bool matches(DirectionType dirType) const {
            return progMatches(data, dirType);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) const {
            return progSampleInternal(data, query, uComponent, uDir, result);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal) const {
            return progEvaluateInternal(data, query, dirLocal);
        }
        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(const BSDFQuery &query, const Vector3D &dirLocal) const {
            return progEvaluatePDFInternal(data, query, dirLocal);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)

    public:
#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION BSDF(
            const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            auto setupBSDF = static_cast<ProgSigSetupBSDF>(matDesc.progSetupBSDF);
            setupBSDF(matDesc.data, surfPt, wls, reinterpret_cast<uint32_t*>(data));

            const BSDFProcedureSet procSet = plp.bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];

            progGetBaseColor = static_cast<ProgSigBSDFGetBaseColor>(procSet.progGetBaseColor);
            progMatches = static_cast<ProgSigBSDFmatches>(procSet.progMatches);
            progSampleInternal = static_cast<ProgSigBSDFSampleInternal>(procSet.progSampleInternal);
            progEvaluateInternal = static_cast<ProgSigBSDFEvaluateInternal>(procSet.progEvaluateInternal);
            progEvaluatePDFInternal = static_cast<ProgSigBSDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return progGetBaseColor(data);
        }

        CUDA_DEVICE_FUNCTION bool hasNonDelta() const {
            return matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sample(
            const BSDFQuery &query, const BSDFSample &sample, BSDFQueryResult* result) const {
            if (!matches(query.dirTypeFilter)) {
                result->dirPDF = 0.0f;
                result->sampledType = DirectionType();
                return SampledSpectrum::Zero();
            }
            SampledSpectrum fs_sn = sampleInternal(query, sample.uComponent, sample.uDir, result);
            float snCorrection;
            if constexpr (transportMode == TransportMode::Radiance) {
                snCorrection = std::fabs(result->dirLocal.z / dot(result->dirLocal, query.geometricNormalLocal));
                if (result->dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            else {
                snCorrection = std::fabs(query.dirLocal.z / dot(query.dirLocal, query.geometricNormalLocal));
                if (query.dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            SampledSpectrum ret = fs_sn * snCorrection;
            VLRAssert((result->dirPDF > 0 && ret.allFinite() && !ret.hasNegative()) || result->dirPDF == 0,
                      "sample: (%g, %g, %g), qDir: (%g, %g, %g), gNormal: (%g, %g, %g), wlIdx: %u, "
                      "rDir: (%g, %g, %g), dirPDF: %g, "
                      "snCorrection: %g",
                      sample.uComponent, sample.uDir[0], sample.uDir[1],
                      VLR3DPrint(query.dirLocal), VLR3DPrint(query.geometricNormalLocal), query.wlHint,
                      VLR3DPrint(result->dirLocal), result->dirPDF,
                      snCorrection);
            return ret;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(const BSDFQuery &query, const Vector3D &dirLocal) const {
            SampledSpectrum fs_sn = evaluateInternal(query, dirLocal);
            float snCorrection;
            if constexpr (transportMode == TransportMode::Radiance) {
                snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
                if (dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            else {
                snCorrection = std::fabs(query.dirLocal.z / dot(query.dirLocal, query.geometricNormalLocal));
                if (query.dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            SampledSpectrum ret = fs_sn * snCorrection;
            VLRAssert(ret.allFinite() && !ret.hasNegative(),
                      "qDir: (%g, %g, %g), gNormal: (%g, %g, %g), wlIdx: %u, "
                      "rDir: (%g, %g, %g), "
                      "snCorrection: %g",
                      VLR3DPrint(query.dirLocal), VLR3DPrint(query.geometricNormalLocal), query.wlHint,
                      VLR3DPrint(dirLocal),
                      snCorrection);
            return ret;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDF(const BSDFQuery &query, const Vector3D &dirLocal) const {
            if (!matches(query.dirTypeFilter)) {
                return 0;
            }
            float ret = evaluatePDFInternal(query, dirLocal);
            VLRAssert(vlr::isfinite(ret) && ret >= 0,
                      "qDir: (%g, %g, %g), gNormal: (%g, %g, %g), wlIdx: %u, "
                      "rDir: (%g, %g, %g), dirPDF: %g",
                      VLR3DPrint(query.dirLocal), VLR3DPrint(query.geometricNormalLocal), query.wlHint,
                      VLR3DPrint(dirLocal), ret);
            return ret;
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };

    template <TransportMode transportMode>
    class BSDF<transportMode, BSDFTier::Bidirectional> {
#define VLR_MAX_NUM_BSDF_PARAMETER_SLOTS (32)
        uint32_t data[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];

        ProgSigBSDFGetBaseColor progGetBaseColor;
        ProgSigBSDFmatches progMatches;
        ProgSigBSDFSampleWithRevInternal progSampleInternal;
        ProgSigBSDFEvaluateWithRevInternal progEvaluateInternal;
        ProgSigBSDFEvaluatePDFWithRevInternal progEvaluatePDFInternal;

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION bool matches(DirectionType dirType) const {
            return progMatches(data, dirType);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult) const {
            return progSampleInternal(data, query, uComponent, uDir, result, revResult);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue) const {
            return progEvaluateInternal(data, query, dirLocal, revValue);
        }
        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue) const {
            return progEvaluatePDFInternal(data, query, dirLocal, revValue);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)

    public:
#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION BSDF(
            const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            auto setupBSDF = static_cast<ProgSigSetupBSDF>(matDesc.progSetupBSDF);
            setupBSDF(matDesc.data, surfPt, wls, reinterpret_cast<uint32_t*>(data));

            const BSDFProcedureSet procSet = plp.bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];

            progGetBaseColor = static_cast<ProgSigBSDFGetBaseColor>(procSet.progGetBaseColor);
            progMatches = static_cast<ProgSigBSDFmatches>(procSet.progMatches);
            progSampleInternal = static_cast<ProgSigBSDFSampleWithRevInternal>(procSet.progSampleWithRevInternal);
            progEvaluateInternal = static_cast<ProgSigBSDFEvaluateWithRevInternal>(procSet.progEvaluateWithRevInternal);
            progEvaluatePDFInternal = static_cast<ProgSigBSDFEvaluatePDFWithRevInternal>(procSet.progEvaluatePDFWithRevInternal);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return progGetBaseColor(data);
        }

        CUDA_DEVICE_FUNCTION bool hasNonDelta() const {
            return matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sample(
            const BSDFQuery &query, const BSDFSample &sample,
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult) const {
            if (!matches(query.dirTypeFilter)) {
                result->dirPDF = 0.0f;
                result->sampledType = DirectionType();
                return SampledSpectrum::Zero();
            }
            SampledSpectrum fs_sn = sampleInternal(query, sample.uComponent, sample.uDir, result, revResult);
            float snCorrection;
            if constexpr (transportMode == TransportMode::Radiance) {
                snCorrection = std::fabs(result->dirLocal.z / dot(result->dirLocal, query.geometricNormalLocal));
                if (result->dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            else {
                snCorrection = std::fabs(query.dirLocal.z / dot(query.dirLocal, query.geometricNormalLocal));
                if (query.dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            SampledSpectrum ret = fs_sn * snCorrection;
            VLRAssert((result->dirPDF > 0 && ret.allFinite() && !ret.hasNegative()) || result->dirPDF == 0,
                      "sample: (%g, %g, %g), qDir: (%g, %g, %g), gNormal: (%g, %g, %g), wlIdx: %u, "
                      "rDir: (%g, %g, %g), dirPDF: %g, "
                      "snCorrection: %g",
                      sample.uComponent, sample.uDir[0], sample.uDir[1],
                      VLR3DPrint(query.dirLocal), VLR3DPrint(query.geometricNormalLocal), query.wlHint,
                      VLR3DPrint(result->dirLocal), result->dirPDF,
                      snCorrection);
            return ret;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue) const {
            SampledSpectrum fs_sn = evaluateInternal(query, dirLocal, revValue);
            float snCorrection;
            if constexpr (transportMode == TransportMode::Radiance) {
                snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
                if (dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            else {
                snCorrection = std::fabs(query.dirLocal.z / dot(query.dirLocal, query.geometricNormalLocal));
                if (query.dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            SampledSpectrum ret = fs_sn * snCorrection;
            VLRAssert(ret.allFinite() && !ret.hasNegative(),
                      "qDir: (%g, %g, %g), gNormal: (%g, %g, %g), wlIdx: %u, "
                      "rDir: (%g, %g, %g), "
                      "snCorrection: %g",
                      VLR3DPrint(query.dirLocal), VLR3DPrint(query.geometricNormalLocal), query.wlHint,
                      VLR3DPrint(dirLocal),
                      snCorrection);
            return ret;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDF(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue) const {
            if (!matches(query.dirTypeFilter)) {
                return 0;
            }
            float ret = evaluatePDFInternal(query, dirLocal, revValue);
            VLRAssert(vlr::isfinite(ret) && ret >= 0,
                      "qDir: (%g, %g, %g), gNormal: (%g, %g, %g), wlIdx: %u, "
                      "rDir: (%g, %g, %g), dirPDF: %g",
                      VLR3DPrint(query.dirLocal), VLR3DPrint(query.geometricNormalLocal), query.wlHint,
                      VLR3DPrint(dirLocal), ret);
            return ret;
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };



    class EDF {
#define VLR_MAX_NUM_EDF_PARAMETER_SLOTS (8)
        uint32_t data[VLR_MAX_NUM_EDF_PARAMETER_SLOTS];

        ProgSigEDFmatches progMatches;
        ProgSigEDFSampleInternal progSampleInternal;
        ProgSigEDFEvaluateEmittanceInternal progEvaluateEmittanceInternal;
        ProgSigEDFEvaluateInternal progEvaluateInternal;
        ProgSigEDFEvaluatePDFInternal progEvaluatePDFInternal;

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION bool matches(DirectionType dirType) const {
            return progMatches(data, dirType);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            return progSampleInternal(data, query, uComponent, uDir, result);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            return progEvaluateEmittanceInternal(data);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(const EDFQuery &query, const Vector3D &dirLocal) const {
            return progEvaluateInternal(data, query, dirLocal);
        }
        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(const EDFQuery &query, const Vector3D &dirLocal) const {
            return progEvaluatePDFInternal(data, query, dirLocal);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)

    public:
#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION EDF(
            const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            auto setupEDF = static_cast<ProgSigSetupEDF>(matDesc.progSetupEDF);
            setupEDF(matDesc.data, surfPt, wls, reinterpret_cast<uint32_t*>(data));

            const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[matDesc.edfProcedureSetIndex];

            progMatches = static_cast<ProgSigEDFmatches>(procSet.progMatches);
            progSampleInternal = static_cast<ProgSigEDFSampleInternal>(procSet.progSampleInternal);
            progEvaluateEmittanceInternal = static_cast<ProgSigEDFEvaluateEmittanceInternal>(procSet.progEvaluateEmittanceInternal);
            progEvaluateInternal = static_cast<ProgSigEDFEvaluateInternal>(procSet.progEvaluateInternal);
            progEvaluatePDFInternal = static_cast<ProgSigEDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sample(
            const EDFQuery &query, const EDFSample &sample, EDFQueryResult* result) const {
            if (!matches(query.dirTypeFilter)) {
                result->dirPDF = 0.0f;
                result->sampledType = DirectionType();
                return SampledSpectrum::Zero();
            }
            SampledSpectrum fe = sampleInternal(query, sample.uComponent, sample.uDir, result);
            VLRAssert((result->dirPDF > 0 && fe.allPositiveFinite()) || result->dirPDF == 0,
                      "Invalid EDF value.\nsample: (%g, %g, %g), dirPDF: %g",
                      sample.uComponent, sample.uDir[0], sample.uDir[1],
                      result->dirPDF);
            return fe;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittance() const {
            SampledSpectrum Le0 = evaluateEmittanceInternal();
            return Le0;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(const EDFQuery &query, const Vector3D &dirLocal) const {
            SampledSpectrum Le1 = evaluateInternal(query, dirLocal);
            return Le1;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDF(const EDFQuery &query, const Vector3D &dirLocal) const {
            if (!matches(query.dirTypeFilter)) {
                return 0;
            }
            float ret = evaluatePDFInternal(query, dirLocal);
            return ret;
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };



    class IDF {
#define VLR_MAX_NUM_IDF_PARAMETER_SLOTS (8)
        uint32_t data[VLR_MAX_NUM_IDF_PARAMETER_SLOTS];

        ProgSigIDFSampleInternal progSampleInternal;
        ProgSigIDFEvaluateSpatialImportanceInternal progEvaluateSpatialImportanceInternal;
        ProgSigIDFEvaluateDirectionalImportanceInternal progEvaluateDirectionalImportanceInternal;
        ProgSigIDFEvaluatePDFInternal progEvaluatePDFInternal;
        ProgSigIDFBackProjectDirectionInternal progBackProjectDirectionInternal;

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const IDFQuery &query, const float uDir[2], IDFQueryResult* result) const {
            return progSampleInternal(data, query, uDir, result);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateSpatialImportanceInternal() const {
            return progEvaluateSpatialImportanceInternal(data);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateDirectionalImportanceInternal(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            return progEvaluateDirectionalImportanceInternal(data, query, dirLocal);
        }
        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(const IDFQuery &query, const Vector3D &dirLocal) const {
            return progEvaluatePDFInternal(data, query, dirLocal);
        }
        CUDA_DEVICE_FUNCTION float2 backProjectDirectionInternal(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            return progBackProjectDirectionInternal(data, query, dirLocal);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)

    public:
#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION IDF(
            const CameraDescriptor &camDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            auto setupIDF = static_cast<ProgSigSetupIDF>(camDesc.progSetupIDF);
            setupIDF(camDesc.data, surfPt, wls, reinterpret_cast<uint32_t*>(data));

            const IDFProcedureSet procSet = plp.idfProcedureSetBuffer[camDesc.idfProcedureSetIndex];

            progSampleInternal = static_cast<ProgSigIDFSampleInternal>(procSet.progSampleInternal);
            progEvaluateSpatialImportanceInternal = static_cast<ProgSigIDFEvaluateSpatialImportanceInternal>(procSet.progEvaluateSpatialImportanceInternal);
            progEvaluateDirectionalImportanceInternal = static_cast<ProgSigIDFEvaluateDirectionalImportanceInternal>(procSet.progEvaluateDirectionalImportanceInternal);
            progEvaluatePDFInternal = static_cast<ProgSigIDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);
            progBackProjectDirectionInternal = static_cast<ProgSigIDFBackProjectDirectionInternal>(procSet.progBackProjectDirectionInternal);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sample(
            const IDFQuery &query, const IDFSample &sample, IDFQueryResult* result) const {
            SampledSpectrum fi = sampleInternal(query, sample.uDir, result);
            VLRAssert((result->dirPDF > 0 && fi.allPositiveFinite()) || result->dirPDF == 0,
                      "Invalid IDF value.\nsample: (%g, %g), dirPDF: %g",
                      sample.uDir[0], sample.uDir[1],
                      result->dirPDF);
            return fi;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateSpatialImportance() const {
            SampledSpectrum We0 = evaluateSpatialImportanceInternal();
            return We0;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateDirectionalImportance(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            SampledSpectrum We1 = evaluateDirectionalImportanceInternal(query, dirLocal);
            return We1;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDF(const IDFQuery &query, const Vector3D &dirLocal) const {
            float ret = evaluatePDFInternal(query, dirLocal);
            return ret;
        }

        CUDA_DEVICE_FUNCTION float2 backProjectDirection(const IDFQuery &query, const Vector3D &dirLocal) const {
            float2 ret = backProjectDirectionInternal(query, dirLocal);
            return ret;
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };



    using ProgSigDecodeLocalHitPoint = optixu::DirectCallableProgramID<
        void(const HitPointParameter &, SurfacePoint*, float*)>;
    using ProgSigDecodeHitPoint = optixu::DirectCallableProgramID<
        void(uint32_t instIndex, uint32_t geomInstIndex, uint32_t primIndex,
             float u, float v,
             SurfacePoint* surfPt)>;
    using ProgSigFetchAlpha = optixu::DirectCallableProgramID<
        float(const TexCoord2D &)>;
    using ProgSigFetchNormal = optixu::DirectCallableProgramID<
        Normal3D(const TexCoord2D &)>;

    static constexpr uint32_t VisibilityGroup_Everything = 0b1111'1111;



#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)

    // Reference:
    // Chapter 6. A Fast and Robust Method for Avoiding Self-Intersection, Ray Tracing Gems, 2019
    CUDA_DEVICE_FUNCTION Point3D offsetRayOrigin(const Point3D &p, const Normal3D &geometricNormal) {
        constexpr float kOrigin = 1.0f / 32.0f;
        constexpr float kFloatScale = 1.0f / 65536.0f;
        constexpr float kIntScale = 256.0f;

        int32_t offsetInInt[] = {
            static_cast<int32_t>(kIntScale * geometricNormal.x),
            static_cast<int32_t>(kIntScale * geometricNormal.y),
            static_cast<int32_t>(kIntScale * geometricNormal.z)
        };

        // JP: ���w�I�ȏՓ˓_�̍��W�ƁA���ۂ̍��W�̌덷�͌��_����̋����ɔ�Ⴗ��B
        //     int�Ƃ��ăI�t�Z�b�g�������邱�ƂŃX�P�[����ˑ��ɓK�؂ȃI�t�Z�b�g�������邱�Ƃ��ł���B
        // EN: The error of the actual coorinates of the intersection point to the mathematical one is proportional to the distance to the origin.
        //     Applying the offset as int makes applying appropriate scale invariant amount of offset possible.
        Point3D newP1 = Point3D(__int_as_float(__float_as_int(p.x) + (p.x < 0 ? -1 : 1) * offsetInInt[0]),
                                __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -1 : 1) * offsetInInt[1]),
                                __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -1 : 1) * offsetInInt[2]));

        // JP: ���_�ɋ߂��ꏊ�ł́A���_����̋����Ɉˑ��������̌덷���c�邽�ߕʏ������K�v�B
        // EN: A constant amount of error remains near the origin independent of the distance to the origin so we need handle it separately.
        Point3D newP2 = p + kFloatScale * geometricNormal;

        return Point3D(std::fabs(p.x) < kOrigin ? newP2.x : newP1.x,
                       std::fabs(p.y) < kOrigin ? newP2.y : newP1.y,
                       std::fabs(p.z) < kOrigin ? newP2.z : newP1.z);
    }



    CUDA_DEVICE_FUNCTION float getAlpha(const WavelengthSamples &wls) {
        const auto hp = HitPointParameter::get();

        SurfacePoint surfPt;
        float hypAreaPDF;
        ProgSigDecodeLocalHitPoint decodeHitPoint(hp.sbtr->geomInst.progDecodeLocalHitPoint);
        surfPt.instIndex = optixGetInstanceId();
        surfPt.geomInstIndex = hp.sbtr->geomInst.geomInstIndex;
        surfPt.geomInstIndex = hp.primIndex;
        decodeHitPoint(hp, &surfPt, &hypAreaPDF);
        surfPt.position = transform<TransformKind::ObjectToWorld>(surfPt.position);
        surfPt.shadingFrame = ReferenceFrame(normalize(transform<TransformKind::ObjectToWorld>(surfPt.shadingFrame.x)),
                                             normalize(transform<TransformKind::ObjectToWorld>(surfPt.shadingFrame.z)));
        surfPt.geometricNormal = normalize(transform<TransformKind::ObjectToWorld>(surfPt.geometricNormal));

        return calcNode(hp.sbtr->geomInst.nodeAlpha, 1.0f, surfPt, wls);
    }



    // JP: �ψق��ꂽ�@���ɏ]���ăV�F�[�f�B���O�t���[����ύX����B
    // EN: perturb the shading frame according to the modified normal.
    CUDA_DEVICE_FUNCTION void applyBumpMapping(const Normal3D &modNormalInTF, SurfacePoint* surfPt) {
        if (modNormalInTF.x == 0.0f && modNormalInTF.y == 0.0f)
            return;

        // JP: �@�������]���Ɖ�]�p(�AQuaternion)�����߂đΉ�����ڕ��ʃx�N�g�������߂�B
        // EN: calculate a rotating axis and an angle (and quaternion) from the normal then calculate corresponding tangential vectors.
        float projLength = std::sqrt(modNormalInTF.x * modNormalInTF.x + modNormalInTF.y * modNormalInTF.y);
        float tiltAngle = std::atan(projLength / modNormalInTF.z);
        float qSin, qCos;
        ::vlr::sincos(tiltAngle / 2, &qSin, &qCos);
        float qX = (-modNormalInTF.y / projLength) * qSin;
        float qY = (modNormalInTF.x / projLength) * qSin;
        float qW = qCos;
        Vector3D modTangentInTF = Vector3D(1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
        Vector3D modBitangentInTF = Vector3D(2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

        Matrix3x3 matTFtoW = Matrix3x3(surfPt->shadingFrame.x, surfPt->shadingFrame.y, surfPt->shadingFrame.z);
        ReferenceFrame bumpShadingFrame(matTFtoW * modTangentInTF,
                                        matTFtoW * modBitangentInTF,
                                        matTFtoW * modNormalInTF);

        surfPt->shadingFrame = bumpShadingFrame;
    }



    // JP: �ψق��ꂽ�ڐ��ɏ]���ăV�F�[�f�B���O�t���[����ύX����B
    // EN: perturb the shading frame according to the modified tangent.
    CUDA_DEVICE_FUNCTION void modifyTangent(const Vector3D& modTangent, SurfacePoint* surfPt) {
        if (modTangent == surfPt->shadingFrame.x)
            return;

        float dotNT = dot(surfPt->shadingFrame.z, modTangent);
        Vector3D projModTangent = modTangent - dotNT * surfPt->shadingFrame.z;

        float lx = dot(surfPt->shadingFrame.x, projModTangent);
        float ly = dot(surfPt->shadingFrame.y, projModTangent);

        float tangentAngle = std::atan2(ly, lx);

        float s, c;
        ::vlr::sincos(tangentAngle, &s, &c);
        Vector3D modTangentInTF = Vector3D(c, s, 0);
        Vector3D modBitangentInTF = Vector3D(-s, c, 0);

        Matrix3x3 matTFtoW = Matrix3x3(surfPt->shadingFrame.x, surfPt->shadingFrame.y, surfPt->shadingFrame.z);
        ReferenceFrame newShadingFrame(normalize(matTFtoW * modTangentInTF),
                                       normalize(matTFtoW * modBitangentInTF),
                                       surfPt->shadingFrame.z);

        surfPt->shadingFrame = newShadingFrame;
    }



    CUDA_DEVICE_FUNCTION void calcSurfacePoint(
        const HitPointParameter &hp, const WavelengthSamples &wls, SurfacePoint* surfPt, float* hypAreaPDF) {
        ProgSigDecodeLocalHitPoint decodeHitPoint(hp.sbtr->geomInst.progDecodeLocalHitPoint);
        surfPt->instIndex = optixGetInstanceId();
        surfPt->geomInstIndex = hp.sbtr->geomInst.geomInstIndex;
        surfPt->primIndex = hp.primIndex;
        decodeHitPoint(hp, surfPt, hypAreaPDF);
        surfPt->position = transform<TransformKind::ObjectToWorld>(surfPt->position);
        surfPt->shadingFrame = ReferenceFrame(normalize(transform<TransformKind::ObjectToWorld>(surfPt->shadingFrame.x)),
                                              normalize(transform<TransformKind::ObjectToWorld>(surfPt->shadingFrame.z)));
        surfPt->geometricNormal = normalize(transform<TransformKind::ObjectToWorld>(surfPt->geometricNormal));

        Normal3D localNormal = calcNode(hp.sbtr->geomInst.nodeNormal, Normal3D(0.0f, 0.0f, 1.0f), *surfPt, wls);
        applyBumpMapping(localNormal, surfPt);

        Vector3D newTangent = calcNode(hp.sbtr->geomInst.nodeTangent, surfPt->shadingFrame.x, *surfPt, wls);
        modifyTangent(newTangent, surfPt);
    }

#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
}
