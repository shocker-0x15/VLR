#pragma once

#include "../basic_types_internal.h"
#include "../shared.h"

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
        //Vector3D tc0Direction;
        bool atInfinity;

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
            return absDot(vecWorld, geometricNormal);
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

        RT_FUNCTION BSDFQuery(const Vector3D &dirL, const Normal3D &gNormL, DirectionType filter) : 
            dirLocal(dirL), geometricNormalLocal(gNormL), dirTypeFilter(filter) {}
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
    };

    typedef rtCallableProgramId<void(const SurfaceLightDescriptor::Body &, const SurfaceLightPosSample &, SurfaceLightPosQueryResult*)> ProgSigSurfaceLight_sample;
    typedef rtCallableProgramId<RGBSpectrum(const SurfaceLightDescriptor::Body &, const TexCoord2D &)> ProgSigSurfaceLight_evaluate;

    class SurfaceLight {
        SurfaceLightDescriptor m_desc;
        //SurfaceLightDescriptor::Body m_desc;
        //ProgSigSurfaceLight_sample m_progSurfaceLight_sample;
        //ProgSigSurfaceLight_evaluate m_progSurfaceLight_evaluate;

    public:
        RT_FUNCTION SurfaceLight() {}
        RT_FUNCTION SurfaceLight(const SurfaceLightDescriptor &desc) : 
            m_desc(desc)
            //m_desc(desc.body), 
            //m_progSurfaceLight_sample((ProgSigSurfaceLight_sample)desc.sampleFunc), 
            //m_progSurfaceLight_evaluate((ProgSigSurfaceLight_evaluate)desc.evaluateFunc) 
        {}

        RT_FUNCTION RGBSpectrum sample(const SurfaceLightPosSample &posSample, SurfaceLightPosQueryResult* lpResult) const {
            ProgSigSurfaceLight_sample m_progSurfaceLight_sample = (ProgSigSurfaceLight_sample)m_desc.sampleFunc;
            ProgSigSurfaceLight_evaluate m_progSurfaceLight_evaluate = (ProgSigSurfaceLight_evaluate)m_desc.evaluateFunc;
            m_progSurfaceLight_sample(m_desc.body, posSample, lpResult);
            return m_progSurfaceLight_evaluate(m_desc.body, lpResult->surfPt.texCoord);

            //m_progSurfaceLight_sample(m_desc, posSample, lpResult);
            //return m_progSurfaceLight_evaluate(m_desc, lpResult->surfPt.texCoord);
        }
    };

    //class MyClass {
    //    typedef rtCallableProgramId<void()> ProgSig;
    //    ProgSig m_callableProgram;
    //public:
    //    RT_FUNCTION MyClass(int32_t cpId) : m_callableProgram((ProgSig)cpId) {}

    //    RT_FUNCTION void func() const {
    //        m_callableProgram();
    //    }
    //};

    //class MyClass {
    //    typedef rtCallableProgramId<void()> ProgSig;
    //    int32_t m_cpId;
    //public:
    //    RT_FUNCTION MyClass(int32_t cpId) : m_cpId(cpId) {}

    //    RT_FUNCTION void func() const {
    //        ProgSig callableProgram = (ProgSig)m_cpId;
    //        callableProgram();
    //    }
    //};



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



    struct Payload {
        float initY;
        RGBSpectrum alpha;
        RGBSpectrum contribution;
        Point3D origin;
        Vector3D direction;
        float prevDirPDF;
        DirectionType prevSampledType;
        bool terminate;
    };

    struct ShadowRayPayload {
        float fractionalVisibility;
    };



    // per GeometryInstance
    typedef rtCallableProgramX<TexCoord2D(const HitPointParameter &)> progSigDecodeTexCoord;
    typedef rtCallableProgramX<void(const HitPointParameter &, SurfacePoint*)> progSigDecodeHitPoint;

    // per Material
    typedef rtCallableProgramX<float(const TexCoord2D &)> progSigFetchAlpha;
    typedef rtCallableProgramX<Normal3D(const TexCoord2D &)> progSigFetchNormal;

    // per Material
    typedef rtCallableProgramX<bool(DirectionType)> progSigBSDFMatches;
    typedef rtCallableProgramX<RGBSpectrum(const TexCoord2D &, const BSDFQuery &, float, const float[2], BSDFQueryResult*)> progSigSampleBSDFInternal;
    typedef rtCallableProgramX<RGBSpectrum(const TexCoord2D &, const BSDFQuery &, const Vector3D &)> progSigEvaluateBSDFInternal;
    typedef rtCallableProgramX<float(const TexCoord2D &, const BSDFQuery &, const Vector3D &)> progSigEvaluateBSDF_PDFInternal;

    // per Material
    typedef rtCallableProgramX<RGBSpectrum(const TexCoord2D &)> progSigEvaluateEmittance;
    typedef rtCallableProgramX<RGBSpectrum(const TexCoord2D &, const EDFQuery &, const Vector3D &)> progSigEvaluateEDFInternal;
}
