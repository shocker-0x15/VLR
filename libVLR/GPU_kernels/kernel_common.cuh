#pragma once

#include "../shared/shared.h"
#include "random_distributions.cuh"

namespace VLR {
    using namespace Shared;



    RT_FUNCTION Point3D transform(RTtransformkind kind, const Point3D &p) {
        return asPoint3D(rtTransformPoint(kind, asOptiXType(p)));
    }

    RT_FUNCTION Vector3D transform(RTtransformkind kind, const Vector3D &v) {
        return asVector3D(rtTransformVector(kind, asOptiXType(v)));
    }

    RT_FUNCTION Normal3D transform(RTtransformkind kind, const Normal3D &n) {
        return asNormal3D(rtTransformNormal(kind, asOptiXType(n)));
    }

    struct ObjectInfo {
        StaticTransform transform;

        RT_FUNCTION ObjectInfo() {}
        RT_FUNCTION ObjectInfo(const StaticTransform& tr) : transform(tr) {}
    };



    struct HitPointParameter {
        float b0, b1;
        int32_t primIndex;
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
        RT_FUNCTION Vector3D fromLocal(const Vector3D &v) const {
            // assume orthonormal basis
            return Vector3D(dot(Vector3D(x.x, y.x, z.x), v),
                            dot(Vector3D(x.y, y.y, z.y), v),
                            dot(Vector3D(x.z, y.z, z.z), v));
        }
    };

    struct SurfacePoint {
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
        ObjectInfo objInfo;
        SurfacePoint surfPt;
        float areaPDF;
        DirectionType posType;
        uint32_t materialIndex;
    };

    typedef rtCallableProgramId<void(const SurfaceLightDescriptor::Body &, const SurfaceLightPosSample &, SurfaceLightPosQueryResult*)> ProgSigSurfaceLight_sample;

    class SurfaceLight {
        SurfaceLightDescriptor::Body m_desc;
        ProgSigSurfaceLight_sample m_progSurfaceLight_sample;

    public:
        RT_FUNCTION SurfaceLight() {}
        RT_FUNCTION SurfaceLight(const SurfaceLightDescriptor &desc) :
            m_desc(desc.body), 
            m_progSurfaceLight_sample((ProgSigSurfaceLight_sample)desc.sampleFunc) {
        }

        RT_FUNCTION void sample(const SurfaceLightPosSample &posSample, SurfaceLightPosQueryResult* lpResult) /*const*/ {
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



    typedef rtCallableProgramId<uint32_t(const uint32_t*, const ObjectInfo &, const SurfacePoint &, const WavelengthSamples &, uint32_t*)> ProgSigSetupBSDF;
    typedef rtCallableProgramId<uint32_t(const uint32_t*, const ObjectInfo &, const SurfacePoint &, const WavelengthSamples &, uint32_t*)> ProgSigSetupEDF;

    typedef rtCallableProgramId<SampledSpectrum(const uint32_t*)> ProgSigBSDFGetBaseColor;
    typedef rtCallableProgramId<bool(const uint32_t*, DirectionType)> ProgSigBSDFmatches;
    typedef rtCallableProgramId<SampledSpectrum(const uint32_t*, const BSDFQuery &, float, const float[2], BSDFQueryResult*)> ProgSigBSDFSampleInternal;
    typedef rtCallableProgramId<SampledSpectrum(const uint32_t*, const BSDFQuery &, const Vector3D &)> ProgSigBSDFEvaluateInternal;
    typedef rtCallableProgramId<float(const uint32_t*, const BSDFQuery &, const Vector3D &)> ProgSigBSDFEvaluatePDFInternal;
    typedef rtCallableProgramId<float(const uint32_t*, const BSDFQuery &)> ProgSigBSDFWeightInternal;

    typedef rtCallableProgramId<SampledSpectrum(const uint32_t*)> ProgSigEDFEvaluateEmittanceInternal;
    typedef rtCallableProgramId<SampledSpectrum(const uint32_t*, const EDFQuery &, const Vector3D &)> ProgSigEDFEvaluateInternal;



    rtDeclareVariable(optix::Ray, sm_ray, rtCurrentRay, );
    rtDeclareVariable(HitPointParameter, a_hitPointParam, attribute hitPointParam, );

    // Context-scope Variables
    rtBuffer<NodeProcedureSet, 1> pv_nodeProcedureSetBuffer;
    rtBuffer<SmallNodeDescriptor, 1> pv_smallNodeDescriptorBuffer;
    rtBuffer<MediumNodeDescriptor, 1> pv_mediumNodeDescriptorBuffer;
    rtBuffer<LargeNodeDescriptor, 1> pv_largeNodeDescriptorBuffer;
    rtBuffer<BSDFProcedureSet, 1> pv_bsdfProcedureSetBuffer;
    rtBuffer<EDFProcedureSet, 1> pv_edfProcedureSetBuffer;
    rtBuffer<SurfaceMaterialDescriptor, 1> pv_materialDescriptorBuffer;


    
    template <typename T>
    RT_FUNCTION T calcNode(ShaderNodePlug plug, const T &defaultValue,
                           const ObjectInfo &objInfo, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.isValid()) {
            int32_t programID = pv_nodeProcedureSetBuffer[plug.nodeType].progs[plug.plugType];

            bool conversionDefined = false;
            T ret = T();

#define VLR_DEFINE_CASE(ReturnType, EnumName) \
    case EnumName: { \
        using ProgSigT = rtCallableProgramId<ReturnType(const ShaderNodePlug &, const ObjectInfo &, const SurfacePoint &, const WavelengthSamples &)>; \
        ProgSigT program = (ProgSigT)programID; \
        conversionDefined = NodeTypeInfo<T>::ConversionIsDefinedFrom<ReturnType>(); \
        ret = NodeTypeInfo<T>::convertFrom<ReturnType>(program(plug, objInfo, surfPt, wls)); \
        break; \
    }
            switch ((ShaderNodePlugType)plug.plugType) {
                VLR_DEFINE_CASE(float, ShaderNodePlugType::float1);
                VLR_DEFINE_CASE(optix::float2, ShaderNodePlugType::float2);
                VLR_DEFINE_CASE(optix::float3, ShaderNodePlugType::float3);
                VLR_DEFINE_CASE(optix::float4, ShaderNodePlugType::float4);
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
                                         const ObjectInfo &objInfo, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.isValid()) {
            int32_t programID = pv_nodeProcedureSetBuffer[plug.nodeType].progs[plug.plugType];

            bool conversionDefined = false;
            SampledSpectrum ret = SampledSpectrum::Zero();

#define VLR_DEFINE_CASE(ReturnType, EnumName) \
    case EnumName: { \
        using ProgSigT = rtCallableProgramId<ReturnType(const ShaderNodePlug &, const ObjectInfo &, const SurfacePoint &, const WavelengthSamples &)>; \
        ProgSigT program = (ProgSigT)programID; \
        conversionDefined = NodeTypeInfo<SampledSpectrum>::ConversionIsDefinedFrom<ReturnType>(); \
        ret = NodeTypeInfo<SampledSpectrum>::convertFrom<ReturnType>(program(plug, objInfo, surfPt, wls)); \
        break; \
    }
            switch ((ShaderNodePlugType)plug.plugType) {
                VLR_DEFINE_CASE(float, ShaderNodePlugType::float1);
                VLR_DEFINE_CASE(optix::float2, ShaderNodePlugType::float2);
                VLR_DEFINE_CASE(optix::float3, ShaderNodePlugType::float3);
                VLR_DEFINE_CASE(optix::float4, ShaderNodePlugType::float4);
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
