#pragma once

#include "../shared/shared.h"
#include "random_distributions.cuh"

namespace VLR {
    using namespace Shared;



    enum class TransformKind {
        WorldToObject = 0,
        ObjectToWorld
    };
    
    template <TransformKind kind>
    RT_FUNCTION Point3D transform(const Point3D &p) {
        if constexpr (kind == TransformKind::WorldToObject)
            return asPoint3D(optixTransformPointFromWorldToObjectSpace(asOptiXType(p)));
        else
            return asPoint3D(optixTransformPointFromObjectToWorldSpace(asOptiXType(p)));
        return Point3D(0, 0, 0); // JP: これが無いと何故か"missing return state..."警告が出る。
    }

    template <TransformKind kind>
    RT_FUNCTION Vector3D transform(const Vector3D &v) {
        if constexpr (kind == TransformKind::WorldToObject)
            return asVector3D(optixTransformVectorFromWorldToObjectSpace(asOptiXType(v)));
        else
            return asVector3D(optixTransformVectorFromObjectToWorldSpace(asOptiXType(v)));
        return Vector3D(0, 0, 0); // JP: これが無いと何故か"missing return state..."警告が出る。
    }

    template <TransformKind kind>
    RT_FUNCTION Normal3D transform(const Normal3D &n) {
        if constexpr (kind == TransformKind::WorldToObject)
            return asNormal3D(optixTransformVectorFromWorldToObjectSpace(asOptiXType(n)));
        else
            return asNormal3D(optixTransformVectorFromObjectToWorldSpace(asOptiXType(n)));
        return Normal3D(0, 0, 0); // JP: これが無いと何故か"missing return state..."警告が出る。
    }



    struct HitPointParameter {
        float b0, b1;
        int32_t primIndex;

        RT_FUNCTION static HitPointParameter get() {
            HitPointParameter ret;
            if (optixIsTriangleHit()) {
                float2 bc = optixGetTriangleBarycentrics();
                ret.b0 = 1 - bc.x - bc.y;
                ret.b1 = bc.x;
            }
            else {
                optixu::getAttributes(&ret.b0, &ret.b1);
            }
            ret.primIndex = optixGetPrimitiveIndex();

            return ret;
        }
    };

    struct ReferenceFrame {
        Vector3D x, y;
        Normal3D z;

        RT_FUNCTION ReferenceFrame() { }
        RT_FUNCTION ReferenceFrame(const Vector3D &t, const Normal3D &n) : x(t), y(cross(n, t)), z(n) { }
        RT_FUNCTION ReferenceFrame(const Vector3D &t, const Vector3D &b, const Normal3D &n) : x(t), y(b), z(n) { }
        RT_FUNCTION ReferenceFrame(const Normal3D &zz) : z(zz) {
            z.makeCoordinateSystem(&x, &y);
        }

        RT_FUNCTION Vector3D toLocal(const Vector3D &v) const { return Vector3D(dot(x, v), dot(y, v), dot(z, v)); }
        RT_FUNCTION Normal3D toLocal(const Normal3D &n) const { return Normal3D(dot(x, n), dot(y, n), dot(z, n)); }
        RT_FUNCTION Vector3D fromLocal(const Vector3D &v) const {
            // assume orthonormal basis
            return Vector3D(dot(Vector3D(x.x, y.x, z.x), v),
                            dot(Vector3D(x.y, y.y, z.y), v),
                            dot(Vector3D(x.z, y.z, z.z), v));
        }
        RT_FUNCTION Normal3D fromLocal(const Normal3D &n) const {
            // assume orthonormal basis
            return Normal3D(dot(Normal3D(x.x, y.x, z.x), n),
                            dot(Normal3D(x.y, y.y, z.y), n),
                            dot(Normal3D(x.z, y.z, z.z), n));
        }
    };

    struct SurfacePoint {
        uint32_t geometryInstanceIndex;
        Point3D position;
        Normal3D geometricNormal;
        ReferenceFrame shadingFrame;
        float u, v; // Parameters used to identify the point on a surface, not texture coordinates.
        TexCoord2D texCoord;
        struct {
            bool isPoint : 1;
            bool atInfinity : 1;
        };

        RT_FUNCTION float calcSquaredDistance(const Point3D &shadingPoint) const {
            return atInfinity ? 1.0f : sqDistance(position, shadingPoint);
        }
        RT_FUNCTION Vector3D calcDirectionFrom(const Point3D &shadingPoint, float* dist2) const {
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

        RT_FUNCTION Vector3D toLocal(const Vector3D &vecWorld) const { return shadingFrame.toLocal(vecWorld); }
        RT_FUNCTION Vector3D fromLocal(const Vector3D &vecLocal) const { return shadingFrame.fromLocal(vecLocal); }
        RT_FUNCTION float calcCosTerm(const Vector3D &vecWorld) const {
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
        RT_FUNCTION static constexpr DirectionType LowFreq() { return IE_LowFreq; };
        RT_FUNCTION static constexpr DirectionType HighFreq() { return IE_HighFreq; };
        RT_FUNCTION static constexpr DirectionType Delta0D() { return IE_Delta0D; };
        RT_FUNCTION static constexpr DirectionType Delta1D() { return IE_Delta1D; };
        RT_FUNCTION static constexpr DirectionType NonDelta() { return IE_NonDelta; };
        RT_FUNCTION static constexpr DirectionType Delta() { return IE_Delta; };
        RT_FUNCTION static constexpr DirectionType AllFreq() { return IE_AllFreq; };
        RT_FUNCTION static constexpr DirectionType Reflection() { return IE_Reflection; };
        RT_FUNCTION static constexpr DirectionType Transmission() { return IE_Transmission; };
        RT_FUNCTION static constexpr DirectionType Emission() { return IE_Emission; };
        RT_FUNCTION static constexpr DirectionType Acquisition() { return IE_Acquisition; };
        RT_FUNCTION static constexpr DirectionType WholeSphere() { return IE_WholeSphere; };
        RT_FUNCTION static constexpr DirectionType All() { return IE_All; };
        RT_FUNCTION static constexpr DirectionType Dispersive() { return IE_Dispersive; };
        RT_FUNCTION static constexpr DirectionType LowFreqReflection() { return IE_LowFreqReflection; };
        RT_FUNCTION static constexpr DirectionType LowFreqTransmission() { return IE_LowFreqTransmission; };
        RT_FUNCTION static constexpr DirectionType LowFreqScattering() { return IE_LowFreqScattering; };
        RT_FUNCTION static constexpr DirectionType HighFreqReflection() { return IE_HighFreqReflection; };
        RT_FUNCTION static constexpr DirectionType HighFreqTransmission() { return IE_HighFreqTransmission; };
        RT_FUNCTION static constexpr DirectionType HighFreqScattering() { return IE_HighFreqScattering; };
        RT_FUNCTION static constexpr DirectionType Delta0DReflection() { return IE_Delta0DReflection; };
        RT_FUNCTION static constexpr DirectionType Delta0DTransmission() { return IE_Delta0DTransmission; };
        RT_FUNCTION static constexpr DirectionType Delta0DScattering() { return IE_Delta0DScattering; };

        InternalEnum value;

        RT_FUNCTION DirectionType() { }
        RT_FUNCTION constexpr DirectionType(InternalEnum v) : value(v) { }
        RT_FUNCTION DirectionType operator&(const DirectionType &r) const { return (InternalEnum)(value & r.value); }
        RT_FUNCTION DirectionType operator|(const DirectionType &r) const { return (InternalEnum)(value | r.value); }
        RT_FUNCTION DirectionType &operator&=(const DirectionType &r) { value = (InternalEnum)(value & r.value); return *this; }
        RT_FUNCTION DirectionType &operator|=(const DirectionType &r) { value = (InternalEnum)(value | r.value); return *this; }
        RT_FUNCTION DirectionType flip() const { return InternalEnum(value ^ IE_WholeSphere); }
        RT_FUNCTION operator bool() const { return value; }
        RT_FUNCTION bool operator==(const DirectionType &r) const { return value == r.value; }
        RT_FUNCTION bool operator!=(const DirectionType &r) const { return value != r.value; }

        RT_FUNCTION bool matches(DirectionType t) const { uint32_t res = value & t.value; return (res & IE_WholeSphere) && (res & IE_AllFreq); }
        RT_FUNCTION bool hasNonDelta() const { return value & IE_NonDelta; }
        RT_FUNCTION bool hasDelta() const { return value & IE_Delta; }
        RT_FUNCTION bool isDelta() const { return (value & IE_Delta) && !(value & IE_NonDelta); }
        RT_FUNCTION bool isReflection() const { return (value & IE_Reflection) && !(value & IE_Transmission); }
        RT_FUNCTION bool isTransmission() const { return !(value & IE_Reflection) && (value & IE_Transmission); }
        RT_FUNCTION bool isDispersive() const { return value & IE_Dispersive; }
    };


    struct EDFQuery {
        DirectionType flags;

        RT_FUNCTION EDFQuery(DirectionType f = DirectionType::All()) : flags(f) {}
    };



    struct BSDFQuery {
        Vector3D dirLocal;
        Normal3D geometricNormalLocal;
        DirectionType dirTypeFilter;
        struct {
            unsigned int wlHint : 6;
        };

        RT_FUNCTION BSDFQuery(const Vector3D &dirL, const Normal3D &gNormL, DirectionType filter, const WavelengthSamples &wls) : 
            dirLocal(dirL), geometricNormalLocal(gNormL), dirTypeFilter(filter), wlHint(wls.selectedLambdaIndex()) {}
    };

    struct BSDFSample {
        float uComponent;
        float uDir[2];

        RT_FUNCTION BSDFSample() {}
        RT_FUNCTION BSDFSample(float uComp, float uDir0, float uDir1) : uComponent(uComp), uDir{ uDir0, uDir1 } {}
    };

    struct BSDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;

        RT_FUNCTION BSDFQueryResult() {}
    };



    struct IDFSample {
        float uDir[2];

        RT_FUNCTION IDFSample(float uDir0, float uDir1) : uDir{ uDir0, uDir1 } {}
    };

    struct IDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;
    };



    struct SurfaceLightPosSample {
        float uElem;
        float uPos[2];

        RT_FUNCTION SurfaceLightPosSample() {}
        RT_FUNCTION SurfaceLightPosSample(float uEl, float uPos0, float uPos1) : uElem(uEl), uPos{ uPos0, uPos1 } {}
    };

    struct SurfaceLightPosQueryResult {
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
        uint32_t materialIndex;
    };

    using ProgSigSurfaceLight_sample = optixu::DirectCallableProgramID<void(const GeometryInstanceDescriptor::Body &, const SurfaceLightPosSample &, SurfaceLightPosQueryResult*)>;

    class SurfaceLight {
        GeometryInstanceDescriptor::Body m_desc;
        uint32_t m_materialIndex;
        ProgSigSurfaceLight_sample m_progSurfaceLight_sample;

    public:
        RT_FUNCTION SurfaceLight() {}
        RT_FUNCTION SurfaceLight(const GeometryInstanceDescriptor &desc) :
            m_desc(desc.body), 
            m_materialIndex(desc.materialIndex),
            m_progSurfaceLight_sample((ProgSigSurfaceLight_sample)desc.sampleFunc) {
        }

        RT_FUNCTION void sample(const SurfaceLightPosSample &posSample, SurfaceLightPosQueryResult* lpResult) /*const*/ {
            lpResult->materialIndex = m_materialIndex;
            m_progSurfaceLight_sample(m_desc, posSample, lpResult);
        }
    };



    struct LensPosSample {
        float uPos[2];

        RT_FUNCTION LensPosSample() {}
        RT_FUNCTION LensPosSample(float uPos0, float uPos1) : uPos{ uPos0, uPos1 } {}
    };

    struct LensPosQueryResult {
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
    };



    using ProgSigSetupBSDF = optixu::DirectCallableProgramID<uint32_t(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)>;
    using ProgSigSetupEDF = optixu::DirectCallableProgramID<uint32_t(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)>;

    using ProgSigBSDFGetBaseColor = optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*)>;
    using ProgSigBSDFmatches = optixu::DirectCallableProgramID<bool(const uint32_t*, DirectionType)>;
    using ProgSigBSDFSampleInternal = optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*, const BSDFQuery &, float, const float[2], BSDFQueryResult*)>;
    using ProgSigBSDFEvaluateInternal = optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*, const BSDFQuery &, const Vector3D &)>;
    using ProgSigBSDFEvaluatePDFInternal = optixu::DirectCallableProgramID<float(const uint32_t*, const BSDFQuery &, const Vector3D &)>;
    using ProgSigBSDFWeightInternal = optixu::DirectCallableProgramID<float(const uint32_t*, const BSDFQuery &)>;

    using ProgSigEDFEvaluateEmittanceInternal = optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*)>;
    using ProgSigEDFEvaluateInternal = optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*, const EDFQuery &, const Vector3D &)>;

    using ProgSigSampleLensPosition = optixu::DirectCallableProgramID<SampledSpectrum(const WavelengthSamples &, const LensPosSample &, LensPosQueryResult*)>;
    using ProgSigSampleIDF = optixu::DirectCallableProgramID<SampledSpectrum(const SurfacePoint &, const WavelengthSamples &, const IDFSample &, IDFQueryResult*)>;

    using ProgSigDecodeHitPoint = optixu::DirectCallableProgramID<void(const HitPointParameter &, SurfacePoint*, float*)>;
    using ProgSigFetchAlpha = optixu::DirectCallableProgramID<float(const TexCoord2D &)>;
    using ProgSigFetchNormal = optixu::DirectCallableProgramID<Normal3D(const TexCoord2D &)>;



    extern "C" __constant__ PipelineLaunchParameters plp;
    
    //rtDeclareVariable(optix::Ray, sm_ray, rtCurrentRay, );
    //rtDeclareVariable(HitPointParameter, a_hitPointParam, attribute hitPointParam, );


    
    template <typename T>
    RT_FUNCTION T calcNode(ShaderNodePlug plug, const T &defaultValue,
                           const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.isValid()) {
            int32_t programID = plp.nodeProcedureSetBuffer[plug.nodeType].progs[plug.plugType];

            bool conversionDefined = false;
            T ret = T();

#define VLR_DEFINE_CASE(ReturnType, EnumName) \
    case EnumName: { \
        using ProgSigT = optixu::DirectCallableProgramID<ReturnType(const ShaderNodePlug &, const SurfacePoint &, const WavelengthSamples &)>; \
        ProgSigT program(programID); \
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

    RT_FUNCTION SampledSpectrum calcNode(ShaderNodePlug plug, const TripletSpectrum &defaultValue,
                                         const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.isValid()) {
            int32_t programID = plp.nodeProcedureSetBuffer[plug.nodeType].progs[plug.plugType];

            bool conversionDefined = false;
            SampledSpectrum ret = SampledSpectrum::Zero();

#define VLR_DEFINE_CASE(ReturnType, EnumName) \
    case EnumName: { \
        using ProgSigT = optixu::DirectCallableProgramID<ReturnType(const ShaderNodePlug &, const SurfacePoint &, const WavelengthSamples &)>; \
        ProgSigT program(programID); \
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
}

#if defined(VLR_Device)
#include "../shared/spectrum_types.cpp"
#endif
