#pragma once

#include "shared.h"

namespace vlr::shared {
    enum class TransformKind {
        ObjectToWorld = 0,
        WorldToObject
    };

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    template <TransformKind kind>
    CUDA_DEVICE_FUNCTION Point3D transform(const Point3D &p) {
        if constexpr (kind == TransformKind::ObjectToWorld)
            return asPoint3D(optixTransformPointFromObjectToWorldSpace(asOptiXType(p)));
        else
            return asPoint3D(optixTransformPointFromWorldToObjectSpace(asOptiXType(p)));
        VLRAssert_ShouldNotBeCalled();
        return Point3D();
    }

    template <TransformKind kind>
    CUDA_DEVICE_FUNCTION Vector3D transform(const Vector3D &v) {
        if constexpr (kind == TransformKind::ObjectToWorld)
            return asVector3D(optixTransformVectorFromObjectToWorldSpace(asOptiXType(v)));
        else
            return asVector3D(optixTransformVectorFromWorldToObjectSpace(asOptiXType(v)));
        VLRAssert_ShouldNotBeCalled();
        return Vector3D();
    }

    template <TransformKind kind>
    CUDA_DEVICE_FUNCTION Normal3D transform(const Normal3D &n) {
        if constexpr (kind == TransformKind::ObjectToWorld)
            return asNormal3D(optixTransformNormalFromObjectToWorldSpace(asOptiXType(n)));
        else
            return asNormal3D(optixTransformNormalFromWorldToObjectSpace(asOptiXType(n)));
        VLRAssert_ShouldNotBeCalled();
        return Normal3D();
    }
#endif



    struct HitGroupSBTRecordData {
        GeometryInstance geomInst;
    };

    using InfiniteSphereAttributeSignature = optixu::AttributeSignature<float, float>;
    
    struct HitPointParameter {
        const HitGroupSBTRecordData* sbtr;
        union {
            struct {
                float b1, b2;
            };
            struct {
                float phi, theta;
            };
        };
        int32_t primIndex;

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION static HitPointParameter get() {
            HitPointParameter ret;
            ret.sbtr = reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
            float2 bc = optixGetTriangleBarycentrics();
            ret.b1 = bc.x;
            ret.b2 = bc.y;
            ret.primIndex = optixGetPrimitiveIndex();
            return ret;
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };

    struct ReferenceFrame {
        Vector3D x, y;
        Normal3D z;

        CUDA_DEVICE_FUNCTION ReferenceFrame() {}
        CUDA_DEVICE_FUNCTION ReferenceFrame(const Vector3D &t, const Normal3D &n) :
            x(t), y(cross(n, t)), z(n) { }
        CUDA_DEVICE_FUNCTION ReferenceFrame(const Vector3D &t, const Vector3D &b, const Normal3D &n) :
            x(t), y(b), z(n) { }
        CUDA_DEVICE_FUNCTION ReferenceFrame(const Normal3D &zz) : z(zz) {
            z.makeCoordinateSystem(&x, &y);
        }

        CUDA_DEVICE_FUNCTION Vector3D toLocal(const Vector3D &v) const {
            return Vector3D(dot(x, v), dot(y, v), dot(z, v));
        }
        CUDA_DEVICE_FUNCTION Vector3D fromLocal(const Vector3D &v) const {
            // assume orthonormal basis
            return Vector3D(dot(Vector3D(x.x, y.x, z.x), v),
                            dot(Vector3D(x.y, y.y, z.y), v),
                            dot(Vector3D(x.z, y.z, z.z), v));
        }
    };

    struct SurfacePoint {
        uint32_t instanceIndex;
        Point3D position;
        Normal3D geometricNormal;
        ReferenceFrame shadingFrame;
        float u, v; // Parameters used to identify the point on a surface, not texture coordinates.
        TexCoord2D texCoord;
        struct {
            bool isPoint : 1;
            bool atInfinity : 1;
        };

        CUDA_DEVICE_FUNCTION float calcSquaredDistance(const Point3D &shadingPoint) const {
            return atInfinity ? 1.0f : sqDistance(position, shadingPoint);
        }
        CUDA_DEVICE_FUNCTION Vector3D calcDirectionFrom(const Point3D &shadingPoint, float* dist2) const {
            if (atInfinity) {
                *dist2 = 1.0f;
                return normalize(position - Point3D::Zero());
            }
            else {
                Vector3D ret(position - shadingPoint);
                *dist2 = ret.sqLength();
                return ret / std::sqrt(*dist2);
            }
        }

        CUDA_DEVICE_FUNCTION Vector3D toLocal(const Vector3D &vecWorld) const {
            return isPoint ? Vector3D(0, 0, 1) : shadingFrame.toLocal(vecWorld);
        }
        CUDA_DEVICE_FUNCTION Vector3D fromLocal(const Vector3D &vecLocal) const {
            return shadingFrame.fromLocal(vecLocal);
        }
        CUDA_DEVICE_FUNCTION float calcCosTerm(const Vector3D &vecWorld) const {
            return isPoint ? 1 : absDot(vecWorld, geometricNormal);
        }
    };



    struct DirectionType {
        enum InternalEnum : uint32_t {
            IE_LowFreq = 1 << 0,
            IE_HighFreq = 1 << 1,
            IE_Delta0D = 1 << 2,
            IE_Delta1D = 1 << 3,
            IE_NonDelta = IE_LowFreq | IE_HighFreq,
            IE_Delta = IE_Delta0D | IE_Delta1D,
            IE_AllFreq = IE_NonDelta | IE_Delta,

            IE_Reflection = 1 << 4,
            IE_Transmission = 1 << 5,
            IE_Emission = IE_Reflection,
            IE_Acquisition = IE_Reflection,
            IE_WholeSphere = IE_Reflection | IE_Transmission,

            IE_All = IE_AllFreq | IE_WholeSphere,

            IE_Dispersive = 1 << 6,

            IE_LowFreqReflection = IE_LowFreq | IE_Reflection,
            IE_LowFreqTransmission = IE_LowFreq | IE_Transmission,
            IE_LowFreqScattering = IE_LowFreqReflection | IE_LowFreqTransmission,
            IE_HighFreqReflection = IE_HighFreq | IE_Reflection,
            IE_HighFreqTransmission = IE_HighFreq | IE_Transmission,
            IE_HighFreqScattering = IE_HighFreqReflection | IE_HighFreqTransmission,
            IE_Delta0DReflection = IE_Delta0D | IE_Reflection,
            IE_Delta0DTransmission = IE_Delta0D | IE_Transmission,
            IE_Delta0DScattering = IE_Delta0DReflection | IE_Delta0DTransmission,
        };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreq() { return IE_LowFreq; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreq() { return IE_HighFreq; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0D() { return IE_Delta0D; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta1D() { return IE_Delta1D; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType NonDelta() { return IE_NonDelta; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta() { return IE_Delta; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType AllFreq() { return IE_AllFreq; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Reflection() { return IE_Reflection; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Transmission() { return IE_Transmission; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Emission() { return IE_Emission; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Acquisition() { return IE_Acquisition; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType WholeSphere() { return IE_WholeSphere; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType All() { return IE_All; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Dispersive() { return IE_Dispersive; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreqReflection() { return IE_LowFreqReflection; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreqTransmission() { return IE_LowFreqTransmission; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreqScattering() { return IE_LowFreqScattering; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreqReflection() { return IE_HighFreqReflection; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreqTransmission() { return IE_HighFreqTransmission; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreqScattering() { return IE_HighFreqScattering; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0DReflection() { return IE_Delta0DReflection; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0DTransmission() { return IE_Delta0DTransmission; };
        CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0DScattering() { return IE_Delta0DScattering; };

        InternalEnum value;

        CUDA_DEVICE_FUNCTION DirectionType() { }
        CUDA_DEVICE_FUNCTION constexpr DirectionType(InternalEnum v) : value(v) { }
        CUDA_DEVICE_FUNCTION DirectionType operator&(const DirectionType &r) const { return (InternalEnum)(value & r.value); }
        CUDA_DEVICE_FUNCTION DirectionType operator|(const DirectionType &r) const { return (InternalEnum)(value | r.value); }
        CUDA_DEVICE_FUNCTION DirectionType &operator&=(const DirectionType &r) { value = (InternalEnum)(value & r.value); return *this; }
        CUDA_DEVICE_FUNCTION DirectionType &operator|=(const DirectionType &r) { value = (InternalEnum)(value | r.value); return *this; }
        CUDA_DEVICE_FUNCTION DirectionType flip() const { return InternalEnum(value ^ IE_WholeSphere); }
        CUDA_DEVICE_FUNCTION operator bool() const { return value; }
        CUDA_DEVICE_FUNCTION bool operator==(const DirectionType &r) const { return value == r.value; }
        CUDA_DEVICE_FUNCTION bool operator!=(const DirectionType &r) const { return value != r.value; }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType t) const { uint32_t res = value & t.value; return (res & IE_WholeSphere) && (res & IE_AllFreq); }
        CUDA_DEVICE_FUNCTION bool hasNonDelta() const { return value & IE_NonDelta; }
        CUDA_DEVICE_FUNCTION bool hasDelta() const { return value & IE_Delta; }
        CUDA_DEVICE_FUNCTION bool isDelta() const { return (value & IE_Delta) && !(value & IE_NonDelta); }
        CUDA_DEVICE_FUNCTION bool isReflection() const { return (value & IE_Reflection) && !(value & IE_Transmission); }
        CUDA_DEVICE_FUNCTION bool isTransmission() const { return !(value & IE_Reflection) && (value & IE_Transmission); }
        CUDA_DEVICE_FUNCTION bool isDispersive() const { return value & IE_Dispersive; }
    };



    struct SurfaceLightPosSample {
        float uElem;
        float uPos[2];

        CUDA_DEVICE_FUNCTION SurfaceLightPosSample() {}
        CUDA_DEVICE_FUNCTION SurfaceLightPosSample(float uEl, float uPos0, float uPos1) :
            uElem(uEl), uPos{ uPos0, uPos1 } {}
    };

    struct SurfaceLightPosQueryResult {
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
        uint32_t materialIndex;
    };
    
    using ProgSigSurfaceLight_sample = optixu::DirectCallableProgramID<
        void(uint32_t, uint32_t,
             const SurfaceLightPosSample &, const Point3D &,
             SurfaceLightPosQueryResult*)>;

    class SurfaceLight {
        uint32_t m_instIndex;
        uint32_t m_geomInstIndex;
        ProgSigSurfaceLight_sample m_progSample;

    public:
        CUDA_DEVICE_FUNCTION SurfaceLight() {}
        CUDA_DEVICE_FUNCTION SurfaceLight(
            uint32_t instIndex, uint32_t geomInstIndex, ProgSigSurfaceLight_sample progSample) :
            m_instIndex(instIndex),
            m_geomInstIndex(geomInstIndex),
            m_progSample(progSample) {
        }

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void sample(
            const SurfaceLightPosSample &posSample, const Point3D &shadingPoint,
            SurfaceLightPosQueryResult* lpResult) const {
            m_progSample(m_instIndex, m_geomInstIndex, posSample, shadingPoint, lpResult);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };


    
    struct EDFQuery {
        DirectionType dirTypeFilter;
        struct {
            unsigned int wlHint : 6;
        };

        CUDA_DEVICE_FUNCTION EDFQuery(DirectionType filter, const WavelengthSamples &wls) :
            dirTypeFilter(filter), wlHint(wls.selectedLambdaIndex()) {}
    };

    struct EDFSample {
        float uComponent;
        float uDir[2];

        CUDA_DEVICE_FUNCTION EDFSample() {}
        CUDA_DEVICE_FUNCTION EDFSample(float uComp, float uDir0, float uDir1) :
            uComponent(uComp), uDir{ uDir0, uDir1 } {}
    };

    struct EDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;

        CUDA_DEVICE_FUNCTION EDFQueryResult() {}
    };



    struct BSDFQuery {
        Vector3D dirLocal;
        Normal3D geometricNormalLocal;
        DirectionType dirTypeFilter;
        struct {
            unsigned int wlHint : 6;
        };

        CUDA_DEVICE_FUNCTION BSDFQuery(
            const Vector3D &dirL, const Normal3D &gNormL, DirectionType filter, const WavelengthSamples &wls) : 
            dirLocal(dirL), geometricNormalLocal(gNormL), dirTypeFilter(filter), wlHint(wls.selectedLambdaIndex()) {}
    };

    struct BSDFSample {
        float uComponent;
        float uDir[2];

        CUDA_DEVICE_FUNCTION BSDFSample() {}
        CUDA_DEVICE_FUNCTION BSDFSample(float uComp, float uDir0, float uDir1) :
            uComponent(uComp), uDir{ uDir0, uDir1 } {}
    };

    struct BSDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;

        CUDA_DEVICE_FUNCTION BSDFQueryResult() {}
    };



    struct LensPosSample {
        float uPos[2];

        CUDA_DEVICE_FUNCTION LensPosSample() {}
        CUDA_DEVICE_FUNCTION LensPosSample(float uPos0, float uPos1) :
            uPos{ uPos0, uPos1 } {}
    };

    struct LensPosQueryResult {
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
    };

    using ProgSigCamera_sample = optixu::DirectCallableProgramID<
        void(const LensPosSample &, LensPosQueryResult*)>;

    using ProgSigCamera_testLensIntersection = optixu::DirectCallableProgramID<
        bool(const Point3D &, const Vector3D &, SurfacePoint*, float*)>;

    struct Camera {
        ProgSigCamera_sample m_progSample;

    public:
        CUDA_DEVICE_FUNCTION Camera() {}
        CUDA_DEVICE_FUNCTION Camera(ProgSigCamera_sample progSample) :
            m_progSample(progSample) {
        }

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void sample(
            const LensPosSample &posSample, LensPosQueryResult* lpResult) const {
            m_progSample(posSample, lpResult);
        }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    };


    
    struct IDFQuery {

    };
    
    struct IDFSample {
        float uDir[2];

        CUDA_DEVICE_FUNCTION IDFSample(float uDir0, float uDir1) :
            uDir{ uDir0, uDir1 } {}
    };

    struct IDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;
    };



    using ProgSigSetupBSDF = optixu::DirectCallableProgramID<
        uint32_t(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)>;
    using ProgSigSetupEDF = optixu::DirectCallableProgramID<
        uint32_t(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)>;
    using ProgSigSetupIDF = optixu::DirectCallableProgramID<
        void(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)>;

    // BSDF callables
    using ProgSigBSDFGetBaseColor = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*)>;
    using ProgSigBSDFmatches = optixu::DirectCallableProgramID<
        bool(const uint32_t*, DirectionType)>;
    using ProgSigBSDFSampleInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*, const BSDFQuery &, float, const float[2], BSDFQueryResult*)>;
    using ProgSigBSDFEvaluateInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*, const BSDFQuery &, const Vector3D &)>;
    using ProgSigBSDFEvaluatePDFInternal = optixu::DirectCallableProgramID<
        float(const uint32_t*, const BSDFQuery &, const Vector3D &)>;
    using ProgSigBSDFWeightInternal = optixu::DirectCallableProgramID<
        float(const uint32_t*, const BSDFQuery &)>;

    // Emitter/EDF callables
    using ProgSigEDFmatches = optixu::DirectCallableProgramID<
        bool(const uint32_t*, DirectionType)>;
    using ProgSigEDFSampleInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*, const EDFQuery &, float, const float[2], EDFQueryResult*)>;
    using ProgSigEDFEvaluateEmittanceInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*)>;
    using ProgSigEDFEvaluateInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*, const EDFQuery &, const Vector3D &)>;
    using ProgSigEDFEvaluatePDFInternal = optixu::DirectCallableProgramID<
        float(const uint32_t*, const EDFQuery &, const Vector3D &)>;
    using ProgSigEDFWeightInternal = optixu::DirectCallableProgramID<
        float(const uint32_t*, const EDFQuery &)>;

    // Lens/IDF callables
    using ProgSigIDFSampleInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*, const IDFQuery &, const float[2], IDFQueryResult*)>;
    using ProgSigIDFEvaluateSpatialImportanceInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*)>;
    using ProgSigIDFEvaluateDirectionalImportanceInternal = optixu::DirectCallableProgramID<
        SampledSpectrum(const uint32_t*, const IDFQuery &, const Vector3D &)>;
    using ProgSigIDFEvaluatePDFInternal = optixu::DirectCallableProgramID<
        float(const uint32_t*, const IDFQuery &, const Vector3D &)>;
    using ProgSigIDFBackProjectDirectionInternal = optixu::DirectCallableProgramID<
        float2(const uint32_t*, const IDFQuery &, const Vector3D &)>;



#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
    template <typename T>
    CUDA_DEVICE_FUNCTION T calcNode(ShaderNodePlug plug, const T &defaultValue,
                                    const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.isValid()) {
            int32_t programID = plp.nodeProcedureSetBuffer[plug.nodeType].progs[plug.plugType];

            bool conversionDefined = false;
            T ret = T();

#define VLR_DEFINE_CASE(ReturnType, EnumName) \
    case EnumName: { \
        using ProgSigT = optixu::DirectCallableProgramID<ReturnType(const ShaderNodePlug &, const SurfacePoint &, const WavelengthSamples &)>; \
        auto program = static_cast<ProgSigT>(programID); \
        conversionDefined = NodeTypeInfo<T>::ConversionIsDefinedFrom<ReturnType>(); \
        ret = NodeTypeInfo<T>::convertFrom<ReturnType>(program(plug, surfPt, wls)); \
        break; \
    }
            switch (static_cast<ShaderNodePlugType>(plug.plugType)) {
                VLR_DEFINE_CASE(float, ShaderNodePlugType::float1);
                VLR_DEFINE_CASE(float2, ShaderNodePlugType::float2);
                VLR_DEFINE_CASE(float3, ShaderNodePlugType::float3);
                VLR_DEFINE_CASE(float4, ShaderNodePlugType::float4);
                VLR_DEFINE_CASE(Point3D, ShaderNodePlugType::Point3D);
                VLR_DEFINE_CASE(Vector3D, ShaderNodePlugType::Vector3D);
                VLR_DEFINE_CASE(Normal3D, ShaderNodePlugType::Normal3D);
                VLR_DEFINE_CASE(float, ShaderNodePlugType::Alpha);
                VLR_DEFINE_CASE(Point3D, ShaderNodePlugType::TextureCoordinates);
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }

#undef VLR_DEFINE_CASE

            if (conversionDefined)
                return ret;
        }
        return defaultValue;
    }

    CUDA_DEVICE_FUNCTION SampledSpectrum calcNode(ShaderNodePlug plug, const TripletSpectrum &defaultValue,
                                                  const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.isValid()) {
            int32_t programID = plp.nodeProcedureSetBuffer[plug.nodeType].progs[plug.plugType];

            bool conversionDefined = false;
            SampledSpectrum ret = SampledSpectrum::Zero();

#define VLR_DEFINE_CASE(ReturnType, EnumName) \
    case EnumName: { \
        using ProgSigT = optixu::DirectCallableProgramID<ReturnType(const ShaderNodePlug &, const SurfacePoint &, const WavelengthSamples &)>; \
        auto program = static_cast<ProgSigT>(programID); \
        conversionDefined = NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom<ReturnType>(); \
        ret = NodeTypeInfo<SampledSpectrum>::convertFrom<ReturnType>(program(plug, surfPt, wls)); \
        break; \
    }
            switch (static_cast<ShaderNodePlugType>(plug.plugType)) {
                VLR_DEFINE_CASE(float, ShaderNodePlugType::float1);
                VLR_DEFINE_CASE(float2, ShaderNodePlugType::float2);
                VLR_DEFINE_CASE(float3, ShaderNodePlugType::float3);
                VLR_DEFINE_CASE(float4, ShaderNodePlugType::float4);
                VLR_DEFINE_CASE(Point3D, ShaderNodePlugType::Point3D);
                VLR_DEFINE_CASE(Vector3D, ShaderNodePlugType::Vector3D);
                VLR_DEFINE_CASE(Normal3D, ShaderNodePlugType::Normal3D);
                VLR_DEFINE_CASE(SampledSpectrum, ShaderNodePlugType::Spectrum);
                VLR_DEFINE_CASE(float, ShaderNodePlugType::Alpha);
                VLR_DEFINE_CASE(Point3D, ShaderNodePlugType::TextureCoordinates);
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }

#undef VLR_DEFINE_CASE

            if (conversionDefined)
                return ret;
        }
        return defaultValue.evaluate(wls);
    }
#endif // #if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
}
