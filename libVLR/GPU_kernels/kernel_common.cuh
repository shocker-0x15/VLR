#pragma once

#include "../shared/shared.h"

namespace VLR {
    using namespace Shared;



    enum class TransformKind {
        ObjectToWorld = 0,
        WorldToObject
    };

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



    struct HitPointParameter {
        union {
            struct {
                float b1, b2;
            };
            struct {
                float phi, theta;
            };
        };
        int32_t primIndex;

        CUDA_DEVICE_FUNCTION static HitPointParameter get() {
            HitPointParameter ret;
            float2 bc = optixGetTriangleBarycentrics();
            ret.b1 = bc.x;
            ret.b2 = bc.y;
            ret.primIndex = optixGetPrimitiveIndex();
            return ret;
        }
    };

    struct HitGroupSBTRecordData {
        GeometryInstance geomInst;

        CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
            return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
        }
    };

    struct ReferenceFrame {
        Vector3D x, y;
        Normal3D z;

        CUDA_DEVICE_FUNCTION ReferenceFrame() { }
        CUDA_DEVICE_FUNCTION ReferenceFrame(const Vector3D &t, const Normal3D &n) : x(t), y(cross(n, t)), z(n) { }
        CUDA_DEVICE_FUNCTION ReferenceFrame(const Vector3D &t, const Vector3D &b, const Normal3D &n) : x(t), y(b), z(n) { }
        CUDA_DEVICE_FUNCTION ReferenceFrame(const Normal3D &zz) : z(zz) {
            z.makeCoordinateSystem(&x, &y);
        }

        CUDA_DEVICE_FUNCTION Vector3D toLocal(const Vector3D &v) const { return Vector3D(dot(x, v), dot(y, v), dot(z, v)); }
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

        CUDA_DEVICE_FUNCTION Vector3D toLocal(const Vector3D &vecWorld) const { return shadingFrame.toLocal(vecWorld); }
        CUDA_DEVICE_FUNCTION Vector3D fromLocal(const Vector3D &vecLocal) const { return shadingFrame.fromLocal(vecLocal); }
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


    struct EDFQuery {
        DirectionType flags;

        CUDA_DEVICE_FUNCTION EDFQuery(DirectionType f = DirectionType::All()) : flags(f) {}
    };



    struct BSDFQuery {
        Vector3D dirLocal;
        Normal3D geometricNormalLocal;
        DirectionType dirTypeFilter;
        struct {
            unsigned int wlHint : 6;
        };

        CUDA_DEVICE_FUNCTION BSDFQuery(const Vector3D &dirL, const Normal3D &gNormL, DirectionType filter, const WavelengthSamples &wls) : 
            dirLocal(dirL), geometricNormalLocal(gNormL), dirTypeFilter(filter), wlHint(wls.selectedLambdaIndex()) {}
    };

    struct BSDFSample {
        float uComponent;
        float uDir[2];

        CUDA_DEVICE_FUNCTION BSDFSample() {}
        CUDA_DEVICE_FUNCTION BSDFSample(float uComp, float uDir0, float uDir1) : uComponent(uComp), uDir{ uDir0, uDir1 } {}
    };

    struct BSDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;

        CUDA_DEVICE_FUNCTION BSDFQueryResult() {}
    };



    struct IDFSample {
        float uDir[2];

        CUDA_DEVICE_FUNCTION IDFSample(float uDir0, float uDir1) : uDir{ uDir0, uDir1 } {}
    };

    struct IDFQueryResult {
        Vector3D dirLocal;
        float dirPDF;
        DirectionType sampledType;
    };



    struct SurfaceLightPosSample {
        float uElem;
        float uPos[2];

        CUDA_DEVICE_FUNCTION SurfaceLightPosSample() {}
        CUDA_DEVICE_FUNCTION SurfaceLightPosSample(float uEl, float uPos0, float uPos1) : uElem(uEl), uPos{ uPos0, uPos1 } {}
    };

    struct SurfaceLightPosQueryResult {
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
        uint32_t materialIndex;
    };

    typedef optixu::DirectCallableProgramID<void(const GeometryInstanceDescriptor &, const SurfaceLightPosSample &, SurfaceLightPosQueryResult*)> ProgSigSurfaceLight_sample;

    class SurfaceLight {
        GeometryInstanceDescriptor m_desc;
        ProgSigSurfaceLight_sample m_progSurfaceLight_sample;

    public:
        CUDA_DEVICE_FUNCTION SurfaceLight() {}
        CUDA_DEVICE_FUNCTION SurfaceLight(const GeometryInstanceDescriptor &desc) :
            m_desc(desc), 
            m_progSurfaceLight_sample(desc.sampleFunc) {
        }

        CUDA_DEVICE_FUNCTION void sample(const SurfaceLightPosSample &posSample, SurfaceLightPosQueryResult* lpResult) const {
            m_progSurfaceLight_sample(m_desc, posSample, lpResult);
        }
    };



    struct LensPosSample {
        float uPos[2];

        CUDA_DEVICE_FUNCTION LensPosSample() {}
        CUDA_DEVICE_FUNCTION LensPosSample(float uPos0, float uPos1) : uPos{ uPos0, uPos1 } {}
    };

    struct LensPosQueryResult {
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
    };



    typedef optixu::DirectCallableProgramID<uint32_t(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)> ProgSigSetupBSDF;
    typedef optixu::DirectCallableProgramID<uint32_t(const uint32_t*, const SurfacePoint &, const WavelengthSamples &, uint32_t*)> ProgSigSetupEDF;

    typedef optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*)> ProgSigBSDFGetBaseColor;
    typedef optixu::DirectCallableProgramID<bool(const uint32_t*, DirectionType)> ProgSigBSDFmatches;
    typedef optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*, const BSDFQuery &, float, const float[2], BSDFQueryResult*)> ProgSigBSDFSampleInternal;
    typedef optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*, const BSDFQuery &, const Vector3D &)> ProgSigBSDFEvaluateInternal;
    typedef optixu::DirectCallableProgramID<float(const uint32_t*, const BSDFQuery &, const Vector3D &)> ProgSigBSDFEvaluatePDFInternal;
    typedef optixu::DirectCallableProgramID<float(const uint32_t*, const BSDFQuery &)> ProgSigBSDFWeightInternal;

    typedef optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*)> ProgSigEDFEvaluateEmittanceInternal;
    typedef optixu::DirectCallableProgramID<SampledSpectrum(const uint32_t*, const EDFQuery &, const Vector3D &)> ProgSigEDFEvaluateInternal;


    
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
        ProgSigT program = (ProgSigT)programID; \
        conversionDefined = NodeTypeInfo<T>::ConversionIsDefinedFrom<ReturnType>(); \
        ret = NodeTypeInfo<T>::convertFrom<ReturnType>(program(plug, surfPt, wls)); \
        break; \
    }
            switch ((ShaderNodePlugType)plug.plugType) {
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
        ProgSigT program = (ProgSigT)programID; \
        conversionDefined = NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom<ReturnType>(); \
        ret = NodeTypeInfo<SampledSpectrum>::convertFrom<ReturnType>(program(plug, surfPt, wls)); \
        break; \
    }
            switch ((ShaderNodePlugType)plug.plugType) {
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
