#include "../shared/kernel_common.h"

namespace vlr {
    using namespace shared;

    CUDA_DEVICE_FUNCTION DirectionType sideTest(const Normal3D &ng, const Vector3D &d0, const Vector3D &d1) {
        bool reflect = dot(Vector3D(ng), d0) * dot(Vector3D(ng), d1) > 0;
        return DirectionType::AllFreq() | (reflect ? DirectionType::Reflection() : DirectionType::Transmission());
    }



    class FresnelConductor {
        SampledSpectrum m_eta;
        SampledSpectrum m_k;

    public:
        CUDA_DEVICE_FUNCTION FresnelConductor(const SampledSpectrum &eta, const SampledSpectrum &k) :
            m_eta(eta), m_k(k) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(float cosEnter) const {
            cosEnter = std::fabs(cosEnter);
            float cosEnter2 = cosEnter * cosEnter;
            SampledSpectrum _2EtaCosEnter = 2.0f * m_eta * cosEnter;
            SampledSpectrum tmp_f = m_eta * m_eta + m_k * m_k;
            SampledSpectrum tmp = tmp_f * cosEnter2;
            SampledSpectrum Rparl2 = (tmp - _2EtaCosEnter + 1) / (tmp + _2EtaCosEnter + 1);
            SampledSpectrum Rperp2 = (tmp_f - _2EtaCosEnter + cosEnter2) / (tmp_f + _2EtaCosEnter + cosEnter2);

            return (Rparl2 + Rperp2) / 2.0f;
        }
        CUDA_DEVICE_FUNCTION float evaluate(float cosEnter, uint32_t wlIdx) const {
            cosEnter = std::fabs(cosEnter);
            float cosEnter2 = cosEnter * cosEnter;
            float _2EtaCosEnter = 2.0f * m_eta[wlIdx] * cosEnter;
            float tmp_f = m_eta[wlIdx] * m_eta[wlIdx] + m_k[wlIdx] * m_k[wlIdx];
            float tmp = tmp_f * cosEnter2;
            float Rparl2 = (tmp - _2EtaCosEnter + 1) / (tmp + _2EtaCosEnter + 1);
            float Rperp2 = (tmp_f - _2EtaCosEnter + cosEnter2) / (tmp_f + _2EtaCosEnter + cosEnter2);

            return (Rparl2 + Rperp2) / 2.0f;
        }
    };



    class FresnelDielectric {
        SampledSpectrum m_etaExt;
        SampledSpectrum m_etaInt;

    public:
        CUDA_DEVICE_FUNCTION FresnelDielectric(const SampledSpectrum &etaExt, const SampledSpectrum &etaInt) : m_etaExt(etaExt), m_etaInt(etaInt) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum etaExt() const { return m_etaExt; }
        CUDA_DEVICE_FUNCTION SampledSpectrum etaInt() const { return m_etaInt; }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(float cosEnter) const {
            cosEnter = clamp(cosEnter, -1.0f, 1.0f);

            bool entering = cosEnter > 0.0f;
            const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
            const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;

            SampledSpectrum sinExit = eEnter / eExit * std::sqrt(std::fmax(0.0f, 1.0f - cosEnter * cosEnter));
            SampledSpectrum ret = SampledSpectrum::Zero();
            cosEnter = std::fabs(cosEnter);
            for (int i = 0; i < SampledSpectrum::NumComponents(); ++i) {
                if (sinExit[i] >= 1.0f) {
                    ret[i] = 1.0f;
                }
                else {
                    float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit[i] * sinExit[i]));
                    ret[i] = evalF(eEnter[i], eExit[i], cosEnter, cosExit);
                }
            }
            return ret;
        }
        CUDA_DEVICE_FUNCTION float evaluate(float cosEnter, uint32_t wlIdx) const {
            cosEnter = clamp(cosEnter, -1.0f, 1.0f);

            bool entering = cosEnter > 0.0f;
            const float &eEnter = entering ? m_etaExt[wlIdx] : m_etaInt[wlIdx];
            const float &eExit = entering ? m_etaInt[wlIdx] : m_etaExt[wlIdx];

            float sinExit = eEnter / eExit * std::sqrt(std::fmax(0.0f, 1.0f - cosEnter * cosEnter));
            cosEnter = std::fabs(cosEnter);
            if (sinExit >= 1.0f) {
                return 1.0f;
            }
            else {
                float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit * sinExit));
                return evalF(eEnter, eExit, cosEnter, cosExit);
            }
        }

        CUDA_DEVICE_FUNCTION static float evalF(float etaEnter, float etaExit, float cosEnter, float cosExit);
    };

    CUDA_DEVICE_FUNCTION float FresnelDielectric::evalF(float etaEnter, float etaExit, float cosEnter, float cosExit) {
        float Rparl = ((etaExit * cosEnter) - (etaEnter * cosExit)) / ((etaExit * cosEnter) + (etaEnter * cosExit));
        float Rperp = ((etaEnter * cosEnter) - (etaExit * cosExit)) / ((etaEnter * cosEnter) + (etaExit * cosExit));
        return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
    }



    class FresnelSchlick {
        // assume vacuum-dielectric interface
        float m_F0;

    public:
        CUDA_DEVICE_FUNCTION FresnelSchlick(float F0) : m_F0(F0) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluate(float cosEnter) const {
            bool entering = cosEnter >= 0;
            float cosEval = cosEnter;
            if (!entering) {
                float sqrtF0 = std::sqrt(m_F0);
                float etaExit = (1 + sqrtF0) / (1 - sqrtF0);
                float invRelIOR = 1.0f / etaExit;
                float sinExit2 = invRelIOR * invRelIOR * std::fmax(0.0f, 1.0f - cosEnter * cosEnter);
                if (sinExit2 > 1.0f) {
                    return SampledSpectrum::One();
                }
                cosEval = std::sqrt(1 - sinExit2);
            }
            return SampledSpectrum(m_F0 + (1.0f - m_F0) * pow5(1 - cosEval));
        }
    };



    class GGXMicrofacetDistribution {
        float m_alpha_gx;
        float m_alpha_gy;
        float m_cosRt;
        float m_sinRt;

    public:
        CUDA_DEVICE_FUNCTION GGXMicrofacetDistribution(float alpha_gx, float alpha_gy, float rotation) :
            m_alpha_gx(alpha_gx), m_alpha_gy(alpha_gy) {
            ::vlr::sincos(rotation, &m_sinRt, &m_cosRt);
        }

        CUDA_DEVICE_FUNCTION float evaluate(const Normal3D &m) {
            Normal3D mr = Normal3D(m_cosRt * m.x + m_sinRt * m.y,
                                   -m_sinRt * m.x + m_cosRt * m.y,
                                   m.z);

            if (mr.z <= 0)
                return 0.0f;
            float temp = pow2(mr.x / m_alpha_gx) + pow2(mr.y / m_alpha_gy) + pow2(mr.z);
            return 1.0f / (VLR_M_PI * m_alpha_gx * m_alpha_gy * pow2(temp));
        }

        CUDA_DEVICE_FUNCTION float evaluateSmithG1(const Vector3D &v, const Normal3D &m) {
            Vector3D vr = Vector3D(m_cosRt * v.x + m_sinRt * v.y,
                                   -m_sinRt * v.x + m_cosRt * v.y,
                                   v.z);

            float alpha_g2_tanTheta2 = (pow2(vr.x * m_alpha_gx) + pow2(vr.y * m_alpha_gy)) / pow2(vr.z);
            float Lambda = (-1 + std::sqrt(1 + alpha_g2_tanTheta2)) / 2;
            float chi = (dot(v, m) / v.z) > 0 ? 1 : 0;
            return chi / (1 + Lambda);
        }

        CUDA_DEVICE_FUNCTION float evaluateHeightCorrelatedSmithG(const Vector3D &v1, const Vector3D &v2, const Normal3D &m) {
            Vector3D v1r = Vector3D(m_cosRt * v1.x + m_sinRt * v1.y,
                                    -m_sinRt * v1.x + m_cosRt * v1.y,
                                    v1.z);
            Vector3D v2r = Vector3D(m_cosRt * v2.x + m_sinRt * v2.y,
                                    -m_sinRt * v2.x + m_cosRt * v2.y,
                                    v2.z);

            float alpha_g2_tanTheta2_1 = (pow2(v1r.x * m_alpha_gx) + pow2(v1r.y * m_alpha_gy)) / pow2(v1r.z);
            float alpha_g2_tanTheta2_2 = (pow2(v2r.x * m_alpha_gx) + pow2(v2r.y * m_alpha_gy)) / pow2(v2r.z);
            float Lambda1 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_1)) / 2;
            float Lambda2 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_2)) / 2;
            float chi1 = (dot(v1, m) / v1.z) > 0 ? 1 : 0;
            float chi2 = (dot(v2, m) / v2.z) > 0 ? 1 : 0;
            return chi1 * chi2 / (1 + Lambda1 + Lambda2);
        }

        CUDA_DEVICE_FUNCTION float sample(const Vector3D &v, float u0, float u1, Normal3D* m, float* normalPDF) {
            Vector3D vr = Vector3D(m_cosRt * v.x + m_sinRt * v.y,
                                   -m_sinRt * v.x + m_cosRt * v.y,
                                   v.z);

            // stretch view
            Vector3D sv = normalize(Vector3D(m_alpha_gx * vr.x, m_alpha_gy * vr.y, vr.z));

            // orthonormal basis
            //        Vector3D T1 = (sv.z < 0.9999f) ? normalize(cross(sv, Vector3D::Ez)) : Vector3D::Ex;
            //        Vector3D T2 = cross(T1, sv);
            float distIn2D = std::sqrt(sv.x * sv.x + sv.y * sv.y);
            float recDistIn2D = 1.0f / distIn2D;
            Vector3D T1 = (sv.z < 0.9999f) ? Vector3D(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0) : Vector3D::Ex();
            Vector3D T2 = Vector3D(T1.y * sv.z, -T1.x * sv.z, distIn2D);

            // sample point with polar coordinates (r, phi)
            float a = 1.0f / (1.0f + sv.z);
            float r = std::sqrt(u0);
            float phi = VLR_M_PI * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            ::vlr::sincos(phi, &sinPhi, &cosPhi);
            float P1 = r * cosPhi;
            float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);

            // compute normal
            Normal3D mr = P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv;

            // unstretch
            mr = normalize(Normal3D(m_alpha_gx * mr.x, m_alpha_gy * mr.y, mr.z));

            float D = evaluate(mr);
            *normalPDF = evaluateSmithG1(vr, mr) * absDot(vr, mr) * D / std::fabs(vr.z);

            *m = Normal3D(m_cosRt * mr.x - m_sinRt * mr.y,
                          m_sinRt * mr.x + m_cosRt * mr.y,
                          mr.z);

            return D;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &v, const Normal3D &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };



    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum BSDF_getBaseColor(
        const uint32_t* params) {
        auto &p = *reinterpret_cast<const BSDFType*>(params);
        return p.getBaseColor();
    }

    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION bool BSDF_matches(
        const uint32_t* params,
        DirectionType flags) {
        auto &p = *reinterpret_cast<const BSDFType*>(params);
        return p.matches(flags);
    }

    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum BSDF_sampleInternal(
        const uint32_t* params,
        const BSDFQuery &query, float uComponent, const float uDir[2],
        BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) {
        auto &p = *reinterpret_cast<const BSDFType*>(params);
        return p.sampleInternal(query, uComponent, uDir, result, revResult);
    }

    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum BSDF_evaluateInternal(
        const uint32_t* params,
        const BSDFQuery &query, const Vector3D &dirLocal, SampledSpectrum* revValue = nullptr) {
        auto &p = *reinterpret_cast<const BSDFType*>(params);
        return p.evaluateInternal(query, dirLocal, revValue);
    }

    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION float BSDF_evaluatePDFInternal(
        const uint32_t* params,
        const BSDFQuery &query, const Vector3D &dirLocal, float* revValue = nullptr) {
        auto &p = *reinterpret_cast<const BSDFType*>(params);
        return p.evaluatePDFInternal(query, dirLocal, revValue);
    }

    template <typename BSDFType>
    CUDA_DEVICE_FUNCTION float BSDF_weightInternal(
        const uint32_t* params,
        const BSDFQuery &query) {
        auto &p = *reinterpret_cast<const BSDFType*>(params);
        return p.weightInternal(query);
    }



#define DEFINE_BSDF_CALLABLE_PROGRAMS(BSDF)\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDF ## _getBaseColor)(\
        const uint32_t* params) {\
        return BSDF_getBaseColor<BSDF>(params);\
    }\
    RT_CALLABLE_PROGRAM bool RT_DC_NAME(BSDF ## _matches)(\
        const uint32_t* params,\
        DirectionType flags) {\
        return BSDF_matches<BSDF>(params, flags);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDF ## _sampleInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, float uComponent, const float uDir[2],\
        BSDFQueryResult* result) {\
        return BSDF_sampleInternal<BSDF>(params, query, uComponent, uDir, result);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDF ## _sampleWithRevInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, float uComponent, const float uDir[2],\
        BSDFQueryResult* result, BSDFQueryReverseResult* revResult) {\
        return BSDF_sampleInternal<BSDF>(params, query, uComponent, uDir, result, revResult);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDF ## _evaluateInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal) {\
        return BSDF_evaluateInternal<BSDF>(params, query, dirLocal);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(BSDF ## _evaluateWithRevInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal, SampledSpectrum* revValue) {\
        return BSDF_evaluateInternal<BSDF>(params, query, dirLocal, revValue);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDF ## _evaluatePDFInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal) {\
        return BSDF_evaluatePDFInternal<BSDF>(params, query, dirLocal);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDF ## _evaluatePDFWithRevInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal, float* revValue) {\
        return BSDF_evaluatePDFInternal<BSDF>(params, query, dirLocal, revValue);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDF ## _weightInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query) {\
        return BSDF_weightInternal<BSDF>(params, query);\
    }




    template <typename EDFType>
    CUDA_DEVICE_FUNCTION bool EDF_matches(
        const uint32_t* params,
        DirectionType flags) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.matches(flags);
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum EDF_sampleInternal(
        const uint32_t* params,
        const EDFQuery &query, float uComponent, const float uDir[2],
        EDFQueryResult* result) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.sampleInternal(query, uComponent, uDir, result);
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum EDF_evaluateEmittanceInternal(
        const uint32_t* params) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.evaluateEmittanceInternal();
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum EDF_evaluateInternal(
        const uint32_t* params,
        const EDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.evaluateInternal(query, dirLocal);
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION float EDF_evaluatePDFInternal(
        const uint32_t* params,
        const EDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.evaluatePDFInternal(query, dirLocal);
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION float EDF_weightInternal(
        const uint32_t* params,
        const EDFQuery &query) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.weightInternal(query);
    }



#define DEFINE_EDF_CALLABLE_PROGRAMS(EDF)\
    RT_CALLABLE_PROGRAM bool RT_DC_NAME(EDF ## _matches)(\
        const uint32_t* params,\
        DirectionType flags) {\
        return EDF_matches<EDF>(params, flags);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _sampleInternal)(\
        const uint32_t* params,\
        const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) {\
        return EDF_sampleInternal<EDF>(params, query, uComponent, uDir, result);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _evaluateEmittanceInternal)(\
        const uint32_t* params) {\
        return EDF_evaluateEmittanceInternal<EDF>(params);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _evaluateInternal)(\
        const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {\
        return EDF_evaluateInternal<EDF>(params, query, dirLocal);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(EDF ## _evaluatePDFInternal)(\
        const uint32_t* params,\
        const EDFQuery &query, const Vector3D &dirLocal) {\
        return EDF_evaluatePDFInternal<EDF>(params, query, dirLocal);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(EDF ## _weightInternal)(\
        const uint32_t* params,\
        const EDFQuery &query) {\
        return EDF_weightInternal<EDF>(params, query);\
    }



    template <typename EDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum EDF_as_BSDF_getBaseColor(
        const uint32_t* params) {
        return SampledSpectrum::Zero();
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION bool EDF_as_BSDF_matches(
        const uint32_t* params,
        DirectionType flags) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        return p.matches(flags);
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum EDF_as_BSDF_sampleInternal(
        const uint32_t* params,
        const BSDFQuery &query, float uComponent, const float uDir[2],
        BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        EDFQuery edfQuery(query.dirTypeFilter, query.wlHint);
        EDFQueryResult edfResult;
        SampledSpectrum ret = p.sampleInternal(edfQuery, uComponent, uDir, &edfResult);
        result->dirLocal = edfResult.dirLocal;
        result->dirPDF = edfResult.dirPDF;
        result->sampledType = edfResult.sampledType;
        if (revResult) {
            revResult->value = SampledSpectrum::Zero();
            revResult->dirPDF = 0.0f;
        }
        return ret;
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION SampledSpectrum EDF_as_BSDF_evaluateInternal(
        const uint32_t* params,
        const BSDFQuery &query, const Vector3D &dirLocal, SampledSpectrum* revValue = nullptr) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        EDFQuery edfQuery(query.dirTypeFilter, query.wlHint);
        SampledSpectrum ret = p.evaluateInternal(edfQuery, dirLocal);
        if (revValue)
            *revValue = SampledSpectrum::Zero();
        return ret;
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION float EDF_as_BSDF_evaluatePDFInternal(
        const uint32_t* params,
        const BSDFQuery &query, const Vector3D &dirLocal, float* revValue = nullptr) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        EDFQuery edfQuery(query.dirTypeFilter, query.wlHint);
        float ret = p.evaluatePDFInternal(edfQuery, dirLocal);
        if (revValue)
            *revValue = 0.0f;
        return ret;
    }

    template <typename EDFType>
    CUDA_DEVICE_FUNCTION float EDF_as_BSDF_weightInternal(
        const uint32_t* params,
        const BSDFQuery &query) {
        auto &p = *reinterpret_cast<const EDFType*>(params);
        EDFQuery edfQuery(query.dirTypeFilter, query.wlHint);
        return p.weightInternal(edfQuery);
    }



#define DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(EDF)\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _as_BSDF_getBaseColor)(\
        const uint32_t* params) {\
        return EDF_as_BSDF_getBaseColor<EDF>(params);\
    }\
    RT_CALLABLE_PROGRAM bool RT_DC_NAME(EDF ## _as_BSDF_matches)(\
        const uint32_t* params,\
        DirectionType flags) {\
        return EDF_as_BSDF_matches<EDF>(params, flags);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _as_BSDF_sampleInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, float uComponent, const float uDir[2],\
        BSDFQueryResult* result) {\
        return EDF_as_BSDF_sampleInternal<EDF>(params, query, uComponent, uDir, result);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _as_BSDF_sampleWithRevInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, float uComponent, const float uDir[2],\
        BSDFQueryResult* result, BSDFQueryReverseResult* revResult) {\
        return EDF_as_BSDF_sampleInternal<EDF>(params, query, uComponent, uDir, result, revResult);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _as_BSDF_evaluateInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal) {\
        return EDF_as_BSDF_evaluateInternal<EDF>(params, query, dirLocal);\
    }\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EDF ## _as_BSDF_evaluateWithRevInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal, SampledSpectrum* revValue) {\
        return EDF_as_BSDF_evaluateInternal<EDF>(params, query, dirLocal, revValue);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(EDF ## _as_BSDF_evaluatePDFInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal) {\
        return EDF_as_BSDF_evaluatePDFInternal<EDF>(params, query, dirLocal);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(EDF ## _as_BSDF_evaluatePDFWithRevInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query, const Vector3D &dirLocal, float* revValue) {\
        return EDF_as_BSDF_evaluatePDFInternal<EDF>(params, query, dirLocal, revValue);\
    }\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(EDF ## _as_BSDF_weightInternal)(\
        const uint32_t* params,\
        const BSDFQuery &query) {\
        return EDF_as_BSDF_weightInternal<EDF>(params, query);\
    }



    class NullBSDF {
    public:
        CUDA_DEVICE_FUNCTION NullBSDF() {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            return false;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            result->dirPDF = 0.0f;
            if (revResult)
                revResult->dirPDF = 0.0f;
            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            if (revValue)
                *revValue = SampledSpectrum::Zero();
            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (*revValue)
                *revValue = 0.0f;
            return 0.0f;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            return 0.0f;
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(NullBSDF_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        return 0;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(NullBSDF)



    class MatteBRDF {
        SampledSpectrum m_albedo;
        float m_roughness;

    public:
        CUDA_DEVICE_FUNCTION MatteBRDF(const SampledSpectrum &albedo, float roughness) :
            m_albedo(albedo), m_roughness(roughness) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return m_albedo;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq();
            return m_type.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            result->dirLocal = cosineSampleHemisphere(uDir[0], uDir[1]);
            result->dirPDF = result->dirLocal.z / VLR_M_PI;
            result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();
            result->dirLocal.z *= query.dirLocal.z >= 0 ? 1 : -1;
            SampledSpectrum fsValue = m_albedo / VLR_M_PI;
            if (revResult) {
                revResult->value = fsValue;
                revResult->dirPDF = std::fabs(query.dirLocal.z) / VLR_M_PI;
            }

            return fsValue;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            if (query.dirLocal.z * dirLocal.z <= 0.0f) {
                SampledSpectrum fs = SampledSpectrum::Zero();
                if (revValue)
                    *revValue = fs;
                return fs;
            }
            SampledSpectrum fsValue = m_albedo / VLR_M_PI;
            if (revValue)
                *revValue = fsValue;

            return fsValue;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (query.dirLocal.z * dirLocal.z <= 0.0f) {
                if (revValue)
                    *revValue = 0.0f;
                return 0.0f;
            }
            float pdfValue = std::fabs(dirLocal.z) / VLR_M_PI;
            if (revValue)
                *revValue = std::fabs(query.dirLocal.z) / VLR_M_PI;

            return pdfValue;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            return m_albedo.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(MatteSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<MatteBRDF*>(params);
        auto &mat = *reinterpret_cast<const MatteSurfaceMaterial*>(matDesc);

        p = MatteBRDF(calcNode(mat.nodeAlbedo, mat.immAlbedo, surfPt, wls), 0.0f);

        return sizeof(MatteBRDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(MatteBRDF)



    class SpecularBRDF {
        SampledSpectrum m_coeffR;
        SampledSpectrum m_eta;
        SampledSpectrum m_k;

    public:
        CUDA_DEVICE_FUNCTION SpecularBRDF(
            const SampledSpectrum &coeffR, const SampledSpectrum &eta, const SampledSpectrum &k) :
        m_coeffR(coeffR), m_eta(eta), m_k(k) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return m_coeffR;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType dirType;
            if constexpr (usePathSpaceRegularization)
                dirType = DirectionType::Reflection() | DirectionType::HighFreq();
            else
                dirType = DirectionType::Reflection() | DirectionType::Delta0D();
            return dirType.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            float regFactor = 1.0f;
            DirectionType dirType = DirectionType::Delta0D();
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                regFactor = computeRegularizationFactor(&cosEpsilon);
                dirType = DirectionType::HighFreq();
            }

            FresnelConductor fresnel(m_eta, m_k);

            result->dirLocal = Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z);
            result->dirPDF = regFactor;
            result->sampledType = DirectionType::Reflection() | dirType;
            SampledSpectrum ret = m_coeffR * fresnel.evaluate(query.dirLocal.z) *
                (regFactor / std::fabs(query.dirLocal.z));

            if (revResult) {
                revResult->value = ret;
                revResult->dirPDF = regFactor;
            }

            return ret;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                float regFactor = computeRegularizationFactor(&cosEpsilon);

                SampledSpectrum ret = SampledSpectrum::Zero();
                Vector3D dirReflected = Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z);
                if (dot(dirLocal, dirReflected) >= cosEpsilon) {
                    FresnelConductor fresnel(m_eta, m_k);
                    ret = m_coeffR * fresnel.evaluate(query.dirLocal.z) *
                        (regFactor / std::fabs(query.dirLocal.z));
                }

                if (revValue)
                    *revValue = ret;

                return ret;
            }
            else {
                if (revValue)
                    *revValue = SampledSpectrum::Zero();
                return SampledSpectrum::Zero();
            }
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                float regFactor = computeRegularizationFactor(&cosEpsilon);

                float ret = 0.0f;
                Vector3D dirReflected = Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z);
                if (dot(dirLocal, dirReflected) >= cosEpsilon)
                    ret = regFactor;

                if (revValue)
                    *revValue = ret;

                return ret;
            }
            else {
                if (revValue)
                    *revValue = 0.0f;
                return 0.0f;
            }
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            FresnelDielectric fresnel(m_eta, m_k);
            float weight = (m_coeffR * fresnel.evaluate(query.dirLocal.z)).importance(query.wlHint);

            return weight;
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(SpecularReflectionSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<SpecularBRDF*>(params);
        auto &mat = *reinterpret_cast<const SpecularReflectionSurfaceMaterial*>(matDesc);

        p = SpecularBRDF(
            calcNode(mat.nodeCoeffR, mat.immCoeffR, surfPt, wls),
            calcNode(mat.nodeEta, mat.immEta, surfPt, wls),
            calcNode(mat.node_k, mat.imm_k, surfPt, wls));

        return sizeof(SpecularBRDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(SpecularBRDF)



    class SpecularBSDF {
        SampledSpectrum m_coeff;
        SampledSpectrum m_etaExt;
        SampledSpectrum m_etaInt;
        bool m_dispersive;

    public:
        CUDA_DEVICE_FUNCTION SpecularBSDF(
            const SampledSpectrum &coeff, const SampledSpectrum &etaExt, const SampledSpectrum &etaInt,
            bool dispersive) :
            m_coeff(coeff), m_etaExt(etaExt), m_etaInt(etaInt), m_dispersive(dispersive) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return m_coeff;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType dirType;
            if constexpr (usePathSpaceRegularization)
                dirType = DirectionType::WholeSphere() | DirectionType::HighFreq();
            else
                dirType = DirectionType::WholeSphere() | DirectionType::Delta0D();
            return dirType.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            if (revResult)
                revResult->dirPDF = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
            const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
            FresnelDielectric fresnel(eEnter, eExit);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

            float regFactor = 1.0f;
            DirectionType dirType = DirectionType::Delta0D();
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                regFactor = computeRegularizationFactor(&cosEpsilon);
                dirType = DirectionType::HighFreq();
            }

            SampledSpectrum F = fresnel.evaluate(dirV.z);
            float reflectProb = F.importance(query.wlHint);
            if (query.dirTypeFilter.isReflection())
                reflectProb = 1.0f;
            if (query.dirTypeFilter.isTransmission())
                reflectProb = 0.0f;
            if (uComponent < reflectProb) {
                if (dirV.z == 0.0f) {
                    result->dirPDF = 0.0f;
                    return SampledSpectrum::Zero();
                }
                Vector3D dirL = Vector3D(-dirV.x, -dirV.y, dirV.z);
                result->dirLocal = entering ? dirL : -dirL;
                result->dirPDF = reflectProb * regFactor;
                result->sampledType = DirectionType::Reflection() | dirType;
                SampledSpectrum ret = m_coeff * F * (regFactor / std::fabs(dirV.z));

                if (revResult) {
                    revResult->value = ret;
                    revResult->dirPDF = result->dirPDF;
                }

                return ret;
            }
            else {
                float sinEnter2 = 1.0f - dirV.z * dirV.z;
                float recRelIOR = eEnter[query.wlHint] / eExit[query.wlHint];// reciprocal of relative IOR.
                float sinExit2 = recRelIOR * recRelIOR * sinEnter2;

                if (sinExit2 >= 1.0f) {
                    result->dirPDF = 0.0f;
                    return SampledSpectrum::Zero();
                }
                float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit2));
                Vector3D dirL = Vector3D(recRelIOR * -dirV.x, recRelIOR * -dirV.y, -cosExit);
                result->dirLocal = entering ? dirL : -dirL;
                result->dirPDF = (1.0f - reflectProb) * regFactor;
                result->sampledType =
                    DirectionType::Transmission() | dirType |
                    (m_dispersive ? DirectionType::Dispersive() : DirectionType());

                SampledSpectrum ret = SampledSpectrum::Zero();
                ret[query.wlHint] = m_coeff[query.wlHint] * (1.0f - F[query.wlHint]) * regFactor;

                if (revResult) {
                    float revSqueezeFactor = 1.0f;
                    if (static_cast<TransportMode>(query.transportMode) == TransportMode::Importance)
                        revSqueezeFactor = pow2(eExit[query.wlHint] / eEnter[query.wlHint]);
                    SampledSpectrum revRet = ret;
                    revRet[query.wlHint] *= revSqueezeFactor / std::fabs(query.dirLocal.z);
                    revResult->value = revRet;
                    revResult->dirPDF = revSqueezeFactor * result->dirPDF;
                }

                float squeezeFactor = 1.0f;
                if (static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance)
                    squeezeFactor *= pow2(eEnter[query.wlHint] / eExit[query.wlHint]);
                ret[query.wlHint] *= squeezeFactor / std::fabs(cosExit);
                result->dirPDF *= squeezeFactor;

                return ret;
            }
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            if constexpr (usePathSpaceRegularization) {
                bool entering = query.dirLocal.z >= 0.0f;

                const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
                const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
                FresnelDielectric fresnel(eEnter, eExit);

                Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

                float cosEpsilon;
                float regFactor = computeRegularizationFactor(&cosEpsilon);

                SampledSpectrum F = fresnel.evaluate(dirV.z);

                Vector3D dirReflected = Vector3D(-dirV.x, -dirV.y, dirV.z);
                if (!entering)
                    dirReflected = -dirReflected;
                if (dot(dirLocal, dirReflected) >= cosEpsilon) {
                    SampledSpectrum ret = m_coeff * F * (regFactor / std::fabs(dirV.z));
                    if (revValue)
                        *revValue = ret;
                    return ret;
                }

                float sinEnter2 = 1.0f - dirV.z * dirV.z;
                float recRelIOR = eEnter[query.wlHint] / eExit[query.wlHint];// reciprocal of relative IOR.
                float sinExit2 = recRelIOR * recRelIOR * sinEnter2;

                SampledSpectrum ret = SampledSpectrum::Zero();
                float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit2));
                Vector3D dirRefracted = Vector3D(recRelIOR * -dirV.x, recRelIOR * -dirV.y, -cosExit);
                if (!entering)
                    dirRefracted = -dirRefracted;
                if (dot(dirLocal, dirRefracted) >= cosEpsilon && cosExit != 0.0f) {
                    ret[query.wlHint] = m_coeff[query.wlHint] * (1.0f - F[query.wlHint]) * regFactor;

                    if (revValue) {
                        float revSqueezeFactor = 1.0f;
                        if (static_cast<TransportMode>(query.transportMode) == TransportMode::Importance)
                            revSqueezeFactor = pow2(eExit[query.wlHint] / eEnter[query.wlHint]);
                        SampledSpectrum revRet = ret;
                        revRet[query.wlHint] *= revSqueezeFactor / std::fabs(query.dirLocal.z);
                        *revValue = revRet;
                    }

                    float squeezeFactor = 1.0f;
                    if (static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance)
                        squeezeFactor *= pow2(eEnter[query.wlHint] / eExit[query.wlHint]);
                    ret[query.wlHint] *= squeezeFactor / std::fabs(cosExit);
                }
                else {
                    if (revValue)
                        *revValue = SampledSpectrum::Zero();
                }

                return ret;
            }
            else {
                if (revValue)
                    *revValue = SampledSpectrum::Zero();
                return SampledSpectrum::Zero();
            }
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (revValue)
                *revValue = 0.0f;

            if constexpr (usePathSpaceRegularization) {
                bool entering = query.dirLocal.z >= 0.0f;

                const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
                const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
                FresnelDielectric fresnel(eEnter, eExit);

                Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

                float cosEpsilon;
                float regFactor = computeRegularizationFactor(&cosEpsilon);

                SampledSpectrum F = fresnel.evaluate(dirV.z);
                float reflectProb = F.importance(query.wlHint);
                if (query.dirTypeFilter.isReflection())
                    reflectProb = 1.0f;
                if (query.dirTypeFilter.isTransmission())
                    reflectProb = 0.0f;

                if (dirV.z == 0.0f)
                    return 0.0f;
                Vector3D dirReflected = Vector3D(-dirV.x, -dirV.y, dirV.z);
                if (!entering)
                    dirReflected = -dirReflected;
                if (dot(dirLocal, dirReflected) >= cosEpsilon) {
                    float ret = reflectProb * regFactor;
                    if (revValue)
                        *revValue = ret;
                    return ret;
                }

                float sinEnter2 = 1.0f - dirV.z * dirV.z;
                float recRelIOR = eEnter[query.wlHint] / eExit[query.wlHint];// reciprocal of relative IOR.
                float sinExit2 = recRelIOR * recRelIOR * sinEnter2;

                if (sinExit2 >= 1.0f)
                    return 0.0f;
                float ret = 0.0f;
                float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit2));
                Vector3D dirRefracted = Vector3D(recRelIOR * -dirV.x, recRelIOR * -dirV.y, -cosExit);
                if (!entering)
                    dirRefracted = -dirRefracted;
                if (dot(dirLocal, dirRefracted) >= cosEpsilon)
                    ret = (1.0f - reflectProb) * regFactor;

                if (revValue) {
                    float revSqueezeFactor = 1.0f;
                    if (static_cast<TransportMode>(query.transportMode) == TransportMode::Importance)
                        revSqueezeFactor = pow2(eExit[query.wlHint] / eEnter[query.wlHint]);
                    *revValue = revSqueezeFactor * ret;
                }

                float squeezeFactor = 1.0f;
                if (static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance)
                    squeezeFactor = pow2(eEnter[query.wlHint] / eExit[query.wlHint]);
                ret *= squeezeFactor;

                return ret;
            }
            else {
                return 0.0f;
            }
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            return m_coeff.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(SpecularScatteringSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<SpecularBSDF*>(params);
        auto &mat = *reinterpret_cast<const SpecularScatteringSurfaceMaterial*>(matDesc);

        p = SpecularBSDF(
            calcNode(mat.nodeCoeff, mat.immCoeff, surfPt, wls),
            calcNode(mat.nodeEtaExt, mat.immEtaExt, surfPt, wls),
            calcNode(mat.nodeEtaInt, mat.immEtaInt, surfPt, wls),
            !wls.singleIsSelected());

        return sizeof(SpecularBSDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(SpecularBSDF)



    class MicrofacetBRDF {
        SampledSpectrum m_eta;
        SampledSpectrum m_k;
        float m_alphaX;
        float m_alphaY;
        float m_rotation;

    public:
        CUDA_DEVICE_FUNCTION MicrofacetBRDF(
            const SampledSpectrum &eta, const SampledSpectrum &k,
            float alphaX, float alphaY, float rotation) :
        m_eta(eta), m_k(k), m_alphaX(alphaX), m_alphaY(alphaY), m_rotation(rotation) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            FresnelConductor fresnel(m_eta, m_k);

            return fresnel.evaluate(1.0f);
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType m_type = DirectionType::Reflection() | DirectionType::HighFreq();
            return m_type.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            if (revResult)
                revResult->dirPDF = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            FresnelConductor fresnel(m_eta, m_k);

            GGXMicrofacetDistribution ggx(m_alphaX, m_alphaY, m_rotation);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

            // JP: ハーフベクトルをサンプルして、最終的な方向サンプルを生成する。
            // EN: sample a half vector, then generate a resulting direction sample based on it.
            Normal3D m;
            float mPDF;
            float D = ggx.sample(dirV, uDir[0], uDir[1], &m, &mPDF);
            float dotHV = dot(dirV, m);
            if (dotHV <= 0) {
                result->dirPDF = 0.0f;
                return SampledSpectrum::Zero();
            }

            Vector3D dirL = 2 * dotHV * m - dirV;
            result->dirLocal = entering ? dirL : -dirL;
            if (dirL.z * dirV.z <= 0) {
                result->dirPDF = 0.0f;
                return SampledSpectrum::Zero();
            }

            float commonPDFTerm = 1.0f / (4 * dotHV);
            result->dirPDF = commonPDFTerm * mPDF;
            result->sampledType = DirectionType::Reflection() | DirectionType::HighFreq();

            SampledSpectrum F = fresnel.evaluate(dotHV);
            float G = ggx.evaluateSmithG1(dirV, m) * ggx.evaluateSmithG1(dirL, m);
            SampledSpectrum fs = F * D * G / (4 * dirV.z * dirL.z);

            if (revResult) {
                revResult->value = fs;
                revResult->dirPDF = commonPDFTerm * ggx.evaluatePDF(dirL, m);
            }

            //VLRAssert(fs.allFinite(), "fs: %s, F: %s, G, %g, D: %g, wlIdx: %u, qDir: %s, rDir: %s",
            //          fs.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return fs;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            bool entering = query.dirLocal.z >= 0.0f;

            FresnelConductor fresnel(m_eta, m_k);

            GGXMicrofacetDistribution ggx(m_alphaX, m_alphaY, m_rotation);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;
            float dotNVdotNL = dirL.z * dirV.z;

            if (dotNVdotNL <= 0)
                return SampledSpectrum::Zero();

            Normal3D m = halfVector(dirV, dirL);
            float dotHV = dot(dirV, m);
            float D = ggx.evaluate(m);

            SampledSpectrum F = fresnel.evaluate(dotHV);
            float G = ggx.evaluateSmithG1(dirV, m) * ggx.evaluateSmithG1(dirL, m);
            SampledSpectrum fs = F * D * G / (4 * dotNVdotNL);

            if (revValue)
                *revValue = fs;

            //VLRAssert(fs.allFinite(), "fs: %s, F: %s, G, %g, D: %g, wlIdx: %u, qDir: %s, dir: %s",
            //          fs.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return fs;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (revValue)
                *revValue = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            FresnelConductor fresnel(m_eta, m_k);

            GGXMicrofacetDistribution ggx(m_alphaX, m_alphaY, m_rotation);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;
            float dotNVdotNL = dirL.z * dirV.z;

            if (dotNVdotNL <= 0.0f)
                return 0.0f;

            Normal3D m = halfVector(dirV, dirL);
            float dotHV = dot(dirV, m);
            if (dotHV <= 0)
                return 0.0f;

            float mPDF = ggx.evaluatePDF(dirV, m);
            float commonPDFTerm = 1.0f / (4 * dotHV);
            float ret = commonPDFTerm * mPDF;

            if (revValue)
                *revValue = commonPDFTerm * ggx.evaluatePDF(dirL, m);

            //VLRAssert(std::isfinite(commonPDFTerm) && std::isfinite(mPDF),
            //          "commonPDFTerm: %g, mPDF: %g, wlIdx: %u, qDir: %s, dir: %s",
            //          commonPDFTerm, mPDF, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return ret;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(
            const BSDFQuery &query) const {
            FresnelConductor fresnel(m_eta, m_k);

            float expectedDotHV = query.dirLocal.z;

            return fresnel.evaluate(expectedDotHV).importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(MicrofacetReflectionSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<MicrofacetBRDF*>(params);
        auto &mat = *reinterpret_cast<const MicrofacetReflectionSurfaceMaterial*>(matDesc);

        SampledSpectrum eta = calcNode(mat.nodeEta, mat.immEta, surfPt, wls);
        SampledSpectrum k = calcNode(mat.node_k, mat.imm_k, surfPt, wls);
        float3 roughnessAnisotropyRotation = 
            calcNode(mat.nodeRoughnessAnisotropyRotation,
                     make_float3(mat.immRoughness, mat.immAnisotropy, mat.immRotation),
                     surfPt, wls);
        float alpha = pow2(roughnessAnisotropyRotation.x);
        float aspect = std::sqrt(1.0f - 0.9f * roughnessAnisotropyRotation.y);
        float alphaX = std::fmax(0.001f, alpha / aspect);
        float alphaY = std::fmax(0.001f, alpha * aspect);
        float rotation = 2 * VLR_M_PI * roughnessAnisotropyRotation.z;

        p = MicrofacetBRDF(eta, k, alphaX, alphaY, rotation);

        return sizeof(MicrofacetBRDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(MicrofacetBRDF)



    class MicrofacetBSDF {
        SampledSpectrum m_coeff;
        SampledSpectrum m_etaExt;
        SampledSpectrum m_etaInt;
        float m_alphaX;
        float m_alphaY;
        float m_rotation;

    public:
        CUDA_DEVICE_FUNCTION MicrofacetBSDF(
            const SampledSpectrum &coeff, const SampledSpectrum &etaExt, const SampledSpectrum &etaInt,
            float alphaX, float alphaY, float rotation) :
        m_coeff(coeff), m_etaExt(etaExt), m_etaInt(etaInt),
        m_alphaX(alphaX), m_alphaY(alphaY), m_rotation(rotation) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return m_coeff;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType m_type = DirectionType::WholeSphere() | DirectionType::HighFreq();
            return m_type.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            if (revResult)
                revResult->dirPDF = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
            const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
            FresnelDielectric fresnel(eEnter, eExit);

            GGXMicrofacetDistribution ggx(m_alphaX, m_alphaY, m_rotation);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

            // JP: ハーフベクトルをサンプルする。
            // EN: sample a half vector.
            Normal3D m;
            float mPDF;
            float D = ggx.sample(dirV, uDir[0], uDir[1], &m, &mPDF);
            float dotHV = dot(dirV, m);
            if (dotHV <= 0 || ::vlr::isnan(D)) {
                result->dirPDF = 0.0f;
                return SampledSpectrum::Zero();
            }

            // JP: サンプルしたハーフベクトルからフレネル項の値を計算して、反射か透過を選択する。
            // EN: calculate the Fresnel term using the sampled half vector, then select reflection or transmission.
            SampledSpectrum F = fresnel.evaluate(dotHV);
            float reflectProb = F.importance(query.wlHint);
            if (query.dirTypeFilter.isReflection())
                reflectProb = 1.0f;
            if (query.dirTypeFilter.isTransmission())
                reflectProb = 0.0f;
            if (uComponent < reflectProb) {
                // JP: 最終的な方向サンプルを生成する。
                // EN: calculate a resulting direction.
                Vector3D dirL = 2 * dotHV * m - dirV;
                result->dirLocal = entering ? dirL : -dirL;
                if (dirL.z * dirV.z <= 0) {
                    result->dirPDF = 0.0f;
                    return SampledSpectrum::Zero();
                }
                float commonPDFTerm = reflectProb / (4 * dotHV);
                result->dirPDF = commonPDFTerm * mPDF;
                result->sampledType = DirectionType::Reflection() | DirectionType::HighFreq();

                float G = ggx.evaluateSmithG1(dirV, m) * ggx.evaluateSmithG1(dirL, m);
                SampledSpectrum ret = m_coeff * F * D * G / (4 * dirV.z * dirL.z);

                if (revResult) {
                    revResult->value = ret;
                    revResult->dirPDF = commonPDFTerm * ggx.evaluatePDF(dirL, m);
                }

                //VLRAssert(ret.allFinite(), "ret: %s, F: %g, %g, %g, G, %g, D: %g, wlIdx: %u, qDir: (%g, %g, %g), rDir: (%g, %g, %g)",
                //          ret.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, 
                //          dirV.x, dirV.y, dirV.z, dirL.x, dirL.y, dirL.z);

                return ret;
            }
            else {
                // JP: 最終的な方向サンプルを生成する。
                // EN: calculate a resulting direction.
                float recRelIOR = eEnter[query.wlHint] / eExit[query.wlHint];
                float innerRoot = 1 + recRelIOR * recRelIOR * (dotHV * dotHV - 1);
                if (innerRoot < 0) {
                    result->dirPDF = 0.0f;
                    return SampledSpectrum::Zero();
                }
                Vector3D dirL = (recRelIOR * dotHV - std::sqrt(innerRoot)) * m - recRelIOR * dirV;
                result->dirLocal = entering ? dirL : -dirL;
                if (dirL.z * dirV.z >= 0) {
                    result->dirPDF = 0.0f;
                    return SampledSpectrum::Zero();
                }
                float dotHL = dot(dirL, m);
                float commonPDFTerm = (1 - reflectProb) / pow2(eEnter[query.wlHint] * dotHV + eExit[query.wlHint] * dotHL);
                result->dirPDF = commonPDFTerm * mPDF * pow2(eExit[query.wlHint]) * std::fabs(dotHL);
                result->sampledType = DirectionType::Transmission() | DirectionType::HighFreq();

                // JP: マイクロファセットBSDFの各項の値を波長成分ごとに計算する。
                // EN: calculate the value of each term of the microfacet BSDF for each wavelength component.
                SampledSpectrum ret = SampledSpectrum::Zero();
                for (int wlIdx = 0; wlIdx < SampledSpectrum::NumComponents(); ++wlIdx) {
                    Normal3D m_wl = normalize(-(eEnter[wlIdx] * dirV + eExit[wlIdx] * dirL) * (entering ? 1 : -1));
                    float dotHV_wl = dot(dirV, m_wl);
                    float dotHL_wl = dot(dirL, m_wl);
                    float F_wl = fresnel.evaluate(dotHV_wl, wlIdx);
                    float G_wl = ggx.evaluateSmithG1(dirV, m_wl) * ggx.evaluateSmithG1(dirL, m_wl);
                    float D_wl = ggx.evaluate(m_wl);
                    ret[wlIdx] = std::fabs(dotHV_wl * dotHL_wl) * (1 - F_wl) * G_wl * D_wl / pow2(eEnter[wlIdx] * dotHV_wl + eExit[wlIdx] * dotHL_wl);

                    //VLRAssert(std::isfinite(ret[wlIdx]), "fs: %g, F: %g, G, %g, D: %g, wlIdx: %u, qDir: %s",
                    //          ret[wlIdx], F_wl, G_wl, D_wl, query.wlHint, dirV.toString().c_str());
                }
                ret /= std::fabs(dirV.z * dirL.z);
                ret *= m_coeff;
                ret *= static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance ?
                    pow2(eEnter) : pow2(eExit); // adjoint: need to cancel eEnter^2 / eExit^2 => eEnter^2 * (eExit^2 / eEnter^2)

                if (revResult) {
                    SampledSpectrum revRet = ret;
                    revResult->value = revRet;
                    revResult->dirPDF = commonPDFTerm * ggx.evaluatePDF(dirL, m) * pow2(eEnter[query.wlHint]) * std::fabs(dotHV);
                }

                //VLRAssert(ret.allFinite(), "fs: %s, wlIdx: %u, qDir: %s, rDir: %s",
                //          ret.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

                return ret;
            }
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            bool entering = query.dirLocal.z >= 0.0f;

            const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
            const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
            FresnelDielectric fresnel(eEnter, eExit);

            GGXMicrofacetDistribution ggx(m_alphaX, m_alphaY, m_rotation);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;
            float dotNVdotNL = dirL.z * dirV.z;

            if (dotNVdotNL > 0 && query.dirTypeFilter.matches(DirectionType::Reflection() | DirectionType::AllFreq())) {
                Normal3D m = halfVector(dirV, dirL);
                float dotHV = dot(dirV, m);
                float D = ggx.evaluate(m);

                SampledSpectrum F = fresnel.evaluate(dotHV);
                float G = ggx.evaluateSmithG1(dirV, m) * ggx.evaluateSmithG1(dirL, m);
                SampledSpectrum ret = m_coeff * F * D * G / (4 * dotNVdotNL);

                if (revValue)
                    *revValue = ret;

                //VLRAssert(ret.allFinite(), "ret: %s, F: %s, G, %g, D: %g, wlIdx: %u, qDir: %s, dir: %s",
                //          ret.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

                return ret;
            }
            else if (dotNVdotNL < 0 && query.dirTypeFilter.matches(DirectionType::Transmission() | DirectionType::AllFreq())) {
                SampledSpectrum ret = SampledSpectrum::Zero();
                for (int wlIdx = 0; wlIdx < SampledSpectrum::NumComponents(); ++wlIdx) {
                    Normal3D m_wl = normalize(-(eEnter[wlIdx] * dirV + eExit[wlIdx] * dirL) * (entering ? 1 : -1));
                    float dotHV_wl = dot(dirV, m_wl);
                    float dotHL_wl = dot(dirL, m_wl);
                    float F_wl = fresnel.evaluate(dotHV_wl, wlIdx);
                    float G_wl = ggx.evaluateSmithG1(dirV, m_wl) * ggx.evaluateSmithG1(dirL, m_wl);
                    float D_wl = ggx.evaluate(m_wl);
                    ret[wlIdx] = std::fabs(dotHV_wl * dotHL_wl) * (1 - F_wl) * G_wl * D_wl / pow2(eEnter[wlIdx] * dotHV_wl + eExit[wlIdx] * dotHL_wl);

                    //VLRAssert(std::isfinite(ret[wlIdx]), "fs: %g, F: %g, G, %g, D: %g, wlIdx: %u, qDir: %s, dir: %s",
                    //          ret[wlIdx], F_wl, G_wl, D_wl, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());
                }
                ret /= std::fabs(dotNVdotNL);
                ret *= m_coeff;
                ret *= static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance ?
                    pow2(eEnter) : pow2(eExit); // !adjoint: eExit^2 * (eEnter / eExit)^2

                if (revValue)
                    *revValue = ret;

                //VLRAssert(ret.allFinite(), "fs: %s, wlIdx: %u, qDir: %s, dir: %s",
                //          ret.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

                return ret;
            }

            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (revValue)
                *revValue = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            const SampledSpectrum &eEnter = entering ? m_etaExt : m_etaInt;
            const SampledSpectrum &eExit = entering ? m_etaInt : m_etaExt;
            FresnelDielectric fresnel(eEnter, eExit);

            GGXMicrofacetDistribution ggx(m_alphaX, m_alphaY, m_rotation);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;
            float dotNVdotNL = dirL.z * dirV.z;
            if (dotNVdotNL == 0)
                return 0.0f;

            Normal3D m;
            if (dotNVdotNL > 0)
                m = halfVector(dirV, dirL);
            else
                m = normalize(-(eEnter[query.wlHint] * dirV + eExit[query.wlHint] * dirL));
            float dotHV = dot(dirV, m);
            if (dotHV <= 0)
                return 0.0f;
            float mPDF = ggx.evaluatePDF(dirV, m);

            SampledSpectrum F = fresnel.evaluate(dotHV);
            float reflectProb = F.importance(query.wlHint);
            if (query.dirTypeFilter.isReflection())
                reflectProb = 1.0f;
            if (query.dirTypeFilter.isTransmission())
                reflectProb = 0.0f;
            if (dotNVdotNL > 0) {
                float commonPDFTerm = reflectProb / (4 * dotHV);

                if (revValue)
                    *revValue = commonPDFTerm * ggx.evaluatePDF(dirL, m);

                //VLRAssert(std::isfinite(commonPDFTerm) && std::isfinite(mPDF),
                //          "commonPDFTerm: %g, mPDF: %g, F: %s, wlIdx: %u, qDir: %s, dir: %s",
                //          commonPDFTerm, mPDF, F.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

                return commonPDFTerm * mPDF;
            }
            else {
                float dotHL = dot(dirL, m);
                float commonPDFTerm = (1 - reflectProb) / pow2(eEnter[query.wlHint] * dotHV + eExit[query.wlHint] * dotHL);

                if (revValue)
                    *revValue = commonPDFTerm * ggx.evaluatePDF(dirL, m) * pow2(eEnter[query.wlHint]) * std::fabs(dotHV);

                //VLRAssert(std::isfinite(commonPDFTerm) && std::isfinite(mPDF),
                //          "commonPDFTerm: %g, mPDF: %g, F: %s, wlIdx: %u, qDir: %s, dir: %s",
                //          commonPDFTerm, mPDF, F.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

                return commonPDFTerm * mPDF * pow2(eExit[query.wlHint]) * std::fabs(dotHL);
            }
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            return m_coeff.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(MicrofacetScatteringSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<MicrofacetBSDF*>(params);
        auto &mat = *reinterpret_cast<const MicrofacetScatteringSurfaceMaterial*>(matDesc);

        SampledSpectrum coeff = calcNode(mat.nodeCoeff, mat.immCoeff, surfPt, wls);
        SampledSpectrum etaExt = calcNode(mat.nodeEtaExt, mat.immEtaExt, surfPt, wls);
        SampledSpectrum etaInt = calcNode(mat.nodeEtaInt, mat.immEtaInt, surfPt, wls);
        float3 roughnessAnisotropyRotation = calcNode(mat.nodeRoughnessAnisotropyRotation,
                                                      make_float3(mat.immRoughness, mat.immAnisotropy, mat.immRotation),
                                                      surfPt, wls);
        float alpha = pow2(roughnessAnisotropyRotation.x);
        float aspect = std::sqrt(1 - 0.9f * roughnessAnisotropyRotation.y);
        float alphaX = std::fmax(0.001f, alpha / aspect);
        float alphaY = std::fmax(0.001f, alpha * aspect);
        float rotation = 2 * VLR_M_PI * roughnessAnisotropyRotation.z;

        p = MicrofacetBSDF(coeff, etaExt, etaInt, alphaX, alphaY, rotation);

        return sizeof(MicrofacetBSDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(MicrofacetBSDF)



    class LambertianBSDF {
        SampledSpectrum m_coeff;
        float m_F0;

    public:
        CUDA_DEVICE_FUNCTION LambertianBSDF(
            const SampledSpectrum &coeff, float F0) :
        m_coeff(coeff), m_F0(F0) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return m_coeff;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType m_type = DirectionType::WholeSphere() | DirectionType::LowFreq();
            return m_type.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            if (revResult)
                revResult->dirPDF = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            FresnelSchlick fresnel(m_F0);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = cosineSampleHemisphere(uDir[0], uDir[1]);
            result->dirPDF = dirL.z / VLR_M_PI;

            SampledSpectrum F = fresnel.evaluate(query.dirLocal.z);
            float reflectProb = F.importance(query.wlHint);
            if (query.dirTypeFilter.isReflection())
                reflectProb = 1.0f;
            if (query.dirTypeFilter.isTransmission())
                reflectProb = 0.0f;

            if (uComponent < reflectProb) {
                result->dirLocal = entering ? dirL : -dirL;
                result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();
                SampledSpectrum ret = F * m_coeff / VLR_M_PI;
                result->dirPDF *= reflectProb;

                if (revResult) {
                    revResult->value = ret;
                    revResult->dirPDF = reflectProb * dirV.z / VLR_M_PI;
                }

                return ret;
            }
            else {
                result->dirLocal = entering ? -dirL : dirL;
                result->sampledType = DirectionType::Transmission() | DirectionType::LowFreq();
                SampledSpectrum ret = (SampledSpectrum::One() - F) * m_coeff / VLR_M_PI;
                result->dirPDF *= (1 - reflectProb);

                if (revResult) {
                    revResult->value = ret;
                    revResult->dirPDF = (1 - reflectProb) * dirV.z / VLR_M_PI;
                }

                return ret;
            }
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            bool entering = query.dirLocal.z >= 0.0f;

            FresnelSchlick fresnel(m_F0);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;

            SampledSpectrum F = fresnel.evaluate(query.dirLocal.z);

            if (dirV.z * dirL.z > 0.0f) {
                SampledSpectrum ret = F * m_coeff / VLR_M_PI;
                if (revValue)
                    *revValue = ret;
                return ret;
            }
            else {
                SampledSpectrum ret = (SampledSpectrum::One() - F) * m_coeff / VLR_M_PI;
                if (revValue)
                    *revValue = ret;
                return ret;
            }
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (revValue)
                *revValue = 0.0f;

            bool entering = query.dirLocal.z >= 0.0f;

            FresnelSchlick fresnel(m_F0);

            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;

            SampledSpectrum F = fresnel.evaluate(query.dirLocal.z);
            float reflectProb = F.importance(query.wlHint);
            if (query.dirTypeFilter.isReflection())
                reflectProb = 1.0f;
            if (query.dirTypeFilter.isTransmission())
                reflectProb = 0.0f;

            if (dirV.z * dirL.z > 0.0f) {
                float dirPDF = reflectProb * dirL.z / VLR_M_PI;
                if (revValue)
                    *revValue = reflectProb * dirV.z / VLR_M_PI;
                return dirPDF;
            }
            else {
                float dirPDF = (1 - reflectProb) * std::fabs(dirL.z) / VLR_M_PI;
                if (revValue)
                    *revValue = (1 - reflectProb) * dirV.z / VLR_M_PI;
                return dirPDF;
            }
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            return m_coeff.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(LambertianScatteringSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<LambertianBSDF*>(params);
        auto &mat = *reinterpret_cast<const LambertianScatteringSurfaceMaterial*>(matDesc);

        p = LambertianBSDF(
            calcNode(mat.nodeCoeff, mat.immCoeff, surfPt, wls),
            calcNode(mat.nodeF0, mat.immF0, surfPt, wls));

        return sizeof(LambertianBSDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(LambertianBSDF)



#define USE_HEIGHT_CORRELATED_SMITH

    class DiffuseAndSpecularBRDF {
        SampledSpectrum m_diffuseColor;
        SampledSpectrum m_specularF0Color;
        float m_roughness;

    public:
        CUDA_DEVICE_FUNCTION DiffuseAndSpecularBRDF(
            const SampledSpectrum &diffuseColor, const SampledSpectrum &specularF0Color, float roughness) :
            m_diffuseColor(diffuseColor), m_specularF0Color(specularF0Color), m_roughness(roughness) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            return m_diffuseColor + m_specularF0Color;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq() | DirectionType::HighFreq();
            return m_type.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            if (revResult)
                revResult->dirPDF = 0.0f;

            float alpha = pow2(m_roughness);
            GGXMicrofacetDistribution ggx(alpha, alpha, 0.0f);

            bool entering = query.dirLocal.z >= 0.0f;
            Vector3D dirL;
            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

            float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(query.dirLocal.z);
            float oneMinusDotVN5 = pow5(1 - dirV.z);
            float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            float iBaseColor = m_diffuseColor.importance(query.wlHint) * pow2(expectedDiffuseFresnel) * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            float iSpecularF0 = m_specularF0Color.importance(query.wlHint);

            float diffuseWeight = iBaseColor;
            float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

            float weights[] = { diffuseWeight, specularWeight };
            float probSelection;
            float sumWeights = 0.0f;
            uint32_t component = sampleDiscrete(weights, 2, uComponent, &probSelection, &sumWeights, &uComponent);

            float diffuseDirPDF, specularDirPDF;
            SampledSpectrum fs;
            Normal3D m;
            float dotLH;
            float D;
            if (component == 0) {
                result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();

                // JP: コサイン分布からサンプルする。
                // EN: sample based on cosine distribution.
                dirL = cosineSampleHemisphere(uDir[0], uDir[1]);
                diffuseDirPDF = dirL.z / VLR_M_PI;

                // JP: 同じ方向サンプルを別の要素からサンプルする確率密度を求める。
                // EN: calculate PDFs to generate the sampled direction from the other distributions.
                m = halfVector(dirL, dirV);
                dotLH = dot(dirL, m);
                float commonPDFTerm = 1.0f / (4 * dotLH);
                specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

                D = ggx.evaluate(m);
            }
            else if (component == 1) {
                result->sampledType = DirectionType::Reflection() | DirectionType::HighFreq();

                // ----------------------------------------------------------------
                // JP: ベーススペキュラー層のマイクロファセット分布からサンプルする。
                // EN: sample based on the base specular microfacet distribution.
                float mPDF;
                D = ggx.sample(dirV, uDir[0], uDir[1], &m, &mPDF);
                float dotVH = dot(dirV, m);
                dotLH = dotVH;
                dirL = 2 * dotVH * m - dirV;
                if (dirL.z * dirV.z <= 0) {
                    result->dirPDF = 0.0f;
                    return SampledSpectrum::Zero();
                }
                float commonPDFTerm = 1.0f / (4 * dotLH);
                specularDirPDF = commonPDFTerm * mPDF;
                // ----------------------------------------------------------------

                // JP: 同じ方向サンプルを別の要素からサンプルする確率密度を求める。
                // EN: calculate PDFs to generate the sampled direction from the other distributions.
                diffuseDirPDF = dirL.z / VLR_M_PI;
            }

            float oneMinusDotLH5 = pow5(1 - dotLH);

#if defined(USE_HEIGHT_CORRELATED_SMITH)
            float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
            float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
            SampledSpectrum F = lerp(m_specularF0Color, SampledSpectrum::One(), oneMinusDotLH5);

            float microfacetDenom = 4 * dirL.z * dirV.z;
            SampledSpectrum specularValue = F * ((D * G) / microfacetDenom);

            float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
            float oneMinusDotLN5 = pow5(1 - dirL.z);
            float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
            float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);
            SampledSpectrum diffuseValue = m_diffuseColor * (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / VLR_M_PI);

            SampledSpectrum ret = diffuseValue + specularValue;

            result->dirLocal = entering ? dirL : -dirL;

            // PDF based on the single-sample model MIS.
            result->dirPDF = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

            if (revResult) {
                float revDiffuseDirPDF = dirV.z / VLR_M_PI;
                float commonPDFTerm = 1.0f / (4 * dotLH);
                float revSpecularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirL, m);

                float revExpectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(dirL.z);
                float revOneMinusDotVN5 = pow5(1 - dirL.z);
                float revExpectedDiffuseFresnel = lerp(1.0f, revExpectedF_D90, revOneMinusDotVN5);
                float revIBaseColor = m_diffuseColor.importance(query.wlHint) * pow2(revExpectedDiffuseFresnel) * lerp(1.0f, 1.0f / 1.51f, m_roughness);

                float revExpectedOneMinusDotVH5 = pow5(1 - dirL.z);
                float revISpecularF0 = m_specularF0Color.importance(query.wlHint);

                float revDiffuseWeight = revIBaseColor;
                float revSpecularWeight = lerp(revISpecularF0, 1.0f, revExpectedOneMinusDotVH5);

                revResult->value = ret;
                revResult->dirPDF =
                    (revDiffuseDirPDF * revDiffuseWeight + revSpecularDirPDF * revSpecularWeight) /
                    (revDiffuseWeight + revSpecularWeight);
            }

            return ret;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            float alpha = pow2(m_roughness);
            GGXMicrofacetDistribution ggx(alpha, alpha, 0.0f);

            if (dirLocal.z * query.dirLocal.z <= 0)
                return SampledSpectrum::Zero();

            bool entering = query.dirLocal.z >= 0.0f;
            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;

            Normal3D m = halfVector(dirL, dirV);
            float dotLH = dot(dirL, m);

            float oneMinusDotLH5 = pow5(1 - dotLH);

            float D = ggx.evaluate(m);
#if defined(USE_HEIGHT_CORRELATED_SMITH)
            float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
            float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
            SampledSpectrum F = lerp(m_specularF0Color, SampledSpectrum::One(), oneMinusDotLH5);

            float microfacetDenom = 4 * dirL.z * dirV.z;
            SampledSpectrum specularValue = F * ((D * G) / microfacetDenom);

            float F_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(dotLH);
            float oneMinusDotVN5 = pow5(1 - dirV.z);
            float oneMinusDotLN5 = pow5(1 - dirL.z);
            float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
            float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

            SampledSpectrum diffuseValue = m_diffuseColor * (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / VLR_M_PI);

            SampledSpectrum ret = diffuseValue + specularValue;

            if (revValue)
                *revValue = ret;

            return ret;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (revValue)
                *revValue = 0.0f;

            float alpha = pow2(m_roughness);
            GGXMicrofacetDistribution ggx(alpha, alpha, 0.0f);

            bool entering = query.dirLocal.z >= 0.0f;
            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
            Vector3D dirL = entering ? dirLocal : -dirLocal;

            Normal3D m = halfVector(dirL, dirV);
            float dotLH = dot(dirL, m);
            float commonPDFTerm = 1.0f / (4 * dotLH);

            float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * query.dirLocal.z * query.dirLocal.z;
            float oneMinusDotVN5 = pow5(1 - dirV.z);
            float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            float iBaseColor = m_diffuseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            float iSpecularF0 = m_specularF0Color.importance(query.wlHint);

            float diffuseWeight = iBaseColor;
            float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

            float sumWeights = diffuseWeight + specularWeight;

            float diffuseDirPDF = vlr::max(dirL.z / VLR_M_PI, 0.0f);
            float specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

            float ret = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

            if (revValue) {
                float revDiffuseDirPDF = dirV.z / VLR_M_PI;
                float commonPDFTerm = 1.0f / (4 * dotLH);
                float revSpecularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirL, m);

                float revExpectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(dirL.z);
                float revOneMinusDotVN5 = pow5(1 - dirL.z);
                float revExpectedDiffuseFresnel = lerp(1.0f, revExpectedF_D90, revOneMinusDotVN5);
                float revIBaseColor = m_diffuseColor.importance(query.wlHint) * pow2(revExpectedDiffuseFresnel) * lerp(1.0f, 1.0f / 1.51f, m_roughness);

                float revExpectedOneMinusDotVH5 = pow5(1 - dirL.z);
                float revISpecularF0 = m_specularF0Color.importance(query.wlHint);

                float revDiffuseWeight = revIBaseColor;
                float revSpecularWeight = lerp(revISpecularF0, 1.0f, revExpectedOneMinusDotVH5);

                *revValue =
                    (revDiffuseDirPDF * revDiffuseWeight + revSpecularDirPDF * revSpecularWeight) /
                    (revDiffuseWeight + revSpecularWeight);
            }

            return ret;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            bool entering = query.dirLocal.z >= 0.0f;
            Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

            float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * query.dirLocal.z * query.dirLocal.z;
            float oneMinusDotVN5 = pow5(1 - dirV.z);
            float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            float iBaseColor = m_diffuseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            float iSpecularF0 = m_specularF0Color.importance(query.wlHint);

            float diffuseWeight = iBaseColor;
            float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

            return diffuseWeight + specularWeight;
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(UE4SurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<DiffuseAndSpecularBRDF*>(params);
        auto &mat = *reinterpret_cast<const UE4SurfaceMaterial*>(matDesc);

        SampledSpectrum baseColor = calcNode(mat.nodeBaseColor, mat.immBaseColor, surfPt, wls);
        float3 occlusionRoughnessMetallic = calcNode(mat.nodeOcclusionRoughnessMetallic,
                                                     make_float3(mat.immOcclusion, mat.immRoughness, mat.immMetallic),
                                                     surfPt, wls);
        float roughness = std::fmax(0.01f, occlusionRoughnessMetallic.y);
        float metallic = occlusionRoughnessMetallic.z;

        const float specular = 0.5f;
        SampledSpectrum diffuseColor = baseColor * (1 - metallic);
        SampledSpectrum specularF0Color = lerp(0.08f * specular * SampledSpectrum::One(), baseColor, metallic);

        p = DiffuseAndSpecularBRDF(diffuseColor, specularF0Color, roughness);

        return sizeof(DiffuseAndSpecularBRDF) / 4;
    }

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(OldStyleSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<DiffuseAndSpecularBRDF*>(params);
        auto &mat = *reinterpret_cast<const OldStyleSurfaceMaterial*>(matDesc);

        SampledSpectrum diffuseColor = calcNode(mat.nodeDiffuseColor, mat.immDiffuseColor, surfPt, wls);
        SampledSpectrum specularF0Color = calcNode(mat.nodeSpecularColor, mat.immSpecularColor, surfPt, wls);
        float roughness = std::fmax(0.01f, 1.0f - calcNode(mat.nodeGlossiness, mat.immGlossiness, surfPt, wls));

        p = DiffuseAndSpecularBRDF(diffuseColor, specularF0Color, roughness);

        return sizeof(DiffuseAndSpecularBRDF) / 4;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(DiffuseAndSpecularBRDF)



    class NullEDF {
    public:
        CUDA_DEVICE_FUNCTION NullEDF() {}

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            return false;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            return SampledSpectrum::Zero();
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            return 0.0f;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const EDFQuery &query) const {
            return 0.0f;
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(NullEDF_setupEDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        return 0;
    }

    DEFINE_EDF_CALLABLE_PROGRAMS(NullEDF)
    DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(NullEDF)



    class DiffuseEDF {
        SampledSpectrum m_emittance;

    public:
        CUDA_DEVICE_FUNCTION DiffuseEDF(const SampledSpectrum &emittance) :
            m_emittance(emittance) {}

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType dirType = DirectionType::Emission() | DirectionType::LowFreq();
            return dirType.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            result->dirLocal = cosineSampleHemisphere(uDir[0], uDir[1]);
            result->dirPDF = result->dirLocal.z / VLR_M_PI;
            result->sampledType = DirectionType::Emission() | DirectionType::LowFreq();
            SampledSpectrum feValue(1.0f / VLR_M_PI);

            return feValue;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            return m_emittance;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            return SampledSpectrum(dirLocal.z > 0.0f ? 1.0f / VLR_M_PI : 0.0f);
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            if (dirLocal.z <= 0.0f)
                return 0.0f;
            float pdfValue = dirLocal.z / VLR_M_PI;

            return pdfValue;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const EDFQuery &query) const {
            return m_emittance.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(DiffuseEmitterSurfaceMaterial_setupEDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<DiffuseEDF*>(params);
        auto &mat = *reinterpret_cast<const DiffuseEmitterSurfaceMaterial*>(matDesc);

        p = DiffuseEDF(calcNode(mat.nodeEmittance, mat.immEmittance, surfPt, wls) * mat.immScale);

        return sizeof(DiffuseEDF) / 4;
    }

    DEFINE_EDF_CALLABLE_PROGRAMS(DiffuseEDF)
    DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(DiffuseEDF)



    class DirectionalEDF {
        SampledSpectrum m_emittance;
        Vector3D m_direction;

    public:
        CUDA_DEVICE_FUNCTION DirectionalEDF(const SampledSpectrum &emittance, const Vector3D &direction) :
            m_emittance(emittance), m_direction(direction) {}

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType dirType;
            if constexpr (usePathSpaceRegularization)
                dirType = DirectionType::Emission() | DirectionType::HighFreq();
            else
                dirType = DirectionType::Emission() | DirectionType::Delta0D();
            return dirType.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            float regFactor = 1.0f;
            DirectionType dirType = DirectionType::Delta0D();
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                regFactor = computeRegularizationFactor(&cosEpsilon);
                dirType = DirectionType::HighFreq();
            }

            result->dirLocal = Vector3D(0, 0, 1);
            result->dirPDF = regFactor;
            result->sampledType = DirectionType::Emission() | dirType;
            SampledSpectrum feValue(regFactor);

            return feValue;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            return m_emittance;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                float regFactor = computeRegularizationFactor(&cosEpsilon);

                if (dot(dirLocal, m_direction) >= cosEpsilon) {
                    SampledSpectrum ret = SampledSpectrum(regFactor);
                    return ret;
                }

                return SampledSpectrum::Zero();
            }
            else {
                return SampledSpectrum::Zero();
            }
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            if constexpr (usePathSpaceRegularization) {
                float cosEpsilon;
                float regFactor = computeRegularizationFactor(&cosEpsilon);

                if (dot(dirLocal, m_direction) >= cosEpsilon)
                    return regFactor;

                return 0.0f;
            }
            else {
                return 0.0f;
            }
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const EDFQuery &query) const {
            return m_emittance.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(DirectionalEmitterSurfaceMaterial_setupEDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<DirectionalEDF*>(params);
        auto &mat = *reinterpret_cast<const DirectionalEmitterSurfaceMaterial*>(matDesc);

        p = DirectionalEDF(calcNode(mat.nodeEmittance, mat.immEmittance, surfPt, wls) * mat.immScale,
                           calcNode(mat.nodeDirection, mat.immDirection, surfPt, wls));

        return sizeof(DirectionalEDF) / 4;
    }

    DEFINE_EDF_CALLABLE_PROGRAMS(DirectionalEDF)
    DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(DirectionalEDF)



    class PointEDF {
        SampledSpectrum m_intensity;

    public:
        CUDA_DEVICE_FUNCTION PointEDF(const SampledSpectrum &intensity) :
            m_intensity(intensity) {}

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType dirType = DirectionType::Emission() | DirectionType::LowFreq();
            return dirType.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            result->dirLocal = uniformSampleSphere(uDir[0], uDir[1]);
            result->dirPDF = 1.0f / (4 * VLR_M_PI);
            result->sampledType = DirectionType::Emission() | DirectionType::LowFreq();

            return 1.0f / (4 * VLR_M_PI);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            // power = intensity * (4 * Pi)
            // area = 4 * pi * r^2
            // emittance = power / area = intensity / r^2
            return m_intensity;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            return 1.0f / (4 * VLR_M_PI);
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            float pdfValue = 1.0f / (4 * VLR_M_PI);

            return pdfValue;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const EDFQuery &query) const {
            return m_intensity.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(PointEmitterSurfaceMaterial_setupEDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<PointEDF*>(params);
        auto &mat = *reinterpret_cast<const PointEmitterSurfaceMaterial*>(matDesc);

        p = PointEDF(calcNode(mat.nodeIntensity, mat.immIntensity, surfPt, wls) * mat.immScale);

        return sizeof(PointEDF) / 4;
    }

    DEFINE_EDF_CALLABLE_PROGRAMS(PointEDF)
    DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(PointEDF)



    // ----------------------------------------------------------------
    // MultiBSDF / MultiEDF

    // bsdf0-3: param offsets
    // numBSDFs
    // --------------------------------
    // BSDF0 procedure set index
    // BSDF0 params
    // ...
    // BSDF3 procedure set index
    // BSDF3 params
    class MultiBSDF {
        unsigned int m_bsdf0 : 6;
        unsigned int m_bsdf1 : 6;
        unsigned int m_bsdf2 : 6;
        unsigned int m_bsdf3 : 6;
        unsigned int m_numBSDFs : 8;

        CUDA_DEVICE_FUNCTION BSDFProcedureSet getBSDFProcSet(uint32_t offset, const uint32_t** body) const {
            const uint32_t* bsdf = reinterpret_cast<const uint32_t*>(this) + offset;
            uint32_t procIdx = *reinterpret_cast<const uint32_t*>(bsdf);
            *body = bsdf + 1;
            return plp.bsdfProcedureSetBuffer[procIdx];
        }

        CUDA_DEVICE_FUNCTION float BSDFWeight(uint32_t offset, const BSDFQuery &query) const {
            const uint32_t* body;
            BSDFProcedureSet procSet = getBSDFProcSet(offset, &body);
            auto matches = static_cast<ProgSigBSDFmatches>(procSet.progMatches);
            auto weightInternal = static_cast<ProgSigBSDFWeightInternal>(procSet.progWeightInternal);
            if (!matches(body, query.dirTypeFilter))
                return 0.0f;
            float weight_sn = weightInternal(body, query);
            float snCorrection;
            if (static_cast<TransportMode>(query.transportMode) == TransportMode::Radiance) {
                snCorrection = 1.0f;
            }
            else {
                snCorrection = std::fabs(query.dirLocal.z / dot(query.dirLocal, query.geometricNormalLocal));
                if (query.dirLocal.z == 0.0f)
                    snCorrection = 0.0f;
            }
            float ret = weight_sn * snCorrection;
            return ret;
        }

    public:
        CUDA_DEVICE_FUNCTION MultiBSDF(
            uint32_t bsdf0, uint32_t bsdf1, uint32_t bsdf2, uint32_t bsdf3,
            uint32_t numBSDFs) :
        m_bsdf0(bsdf0), m_bsdf1(bsdf1), m_bsdf2(bsdf2), m_bsdf3(bsdf3),
        m_numBSDFs(numBSDFs) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum getBaseColor() const {
            uint32_t bsdfOffsets[4] = { m_bsdf0, m_bsdf1, m_bsdf2, m_bsdf3 };

            SampledSpectrum ret;
            for (int i = 0; i < m_numBSDFs; ++i) {
                const uint32_t* bsdfBody;
                BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                auto getBaseColor = static_cast<ProgSigBSDFGetBaseColor>(procSet.progGetBaseColor);
                ret += getBaseColor(bsdfBody);
            }

            return ret;
        }

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            uint32_t bsdfOffsets[4] = { m_bsdf0, m_bsdf1, m_bsdf2, m_bsdf3 };

            for (int i = 0; i < m_numBSDFs; ++i) {
                const uint32_t* bsdfBody;
                BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                auto matches = static_cast<ProgSigBSDFmatches>(procSet.progMatches);
                if (matches(bsdfBody, flags))
                    return true;
            }

            return false;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const BSDFQuery &query, float uComponent, const float uDir[2],
            BSDFQueryResult* result, BSDFQueryReverseResult* revResult = nullptr) const {
            if (revResult)
                revResult->dirPDF = 0.0f;

            uint32_t bsdfOffsets[4] = { m_bsdf0, m_bsdf1, m_bsdf2, m_bsdf3 };

            float weights[4];
            for (int i = 0; i < m_numBSDFs; ++i)
                weights[i] = BSDFWeight(bsdfOffsets[i], query);

            // JP: 各BSDFのウェイトに基づいて方向のサンプルを行うBSDFを選択する。
            // EN: Based on the weight of each BSDF, select a BSDF from which direction sampling.
            float tempProb;
            float sumWeights;
            uint32_t idx = sampleDiscrete(weights, m_numBSDFs, uComponent, &tempProb, &sumWeights, &uComponent);
            if (sumWeights == 0.0f) {
                result->dirPDF = 0.0f;
                return SampledSpectrum::Zero();
            }

            // JP: 選択したBSDFから方向をサンプリングする。
            // EN: sample a direction from the selected BSDF.
            SampledSpectrum value;
            const uint32_t* selectedBsdfBody;
            BSDFProcedureSet selProcSet = getBSDFProcSet(bsdfOffsets[idx], &selectedBsdfBody);
            if (revResult) {
                auto sampleInternal = static_cast<ProgSigBSDFSampleWithRevInternal>(selProcSet.progSampleWithRevInternal);
                value = sampleInternal(selectedBsdfBody, query, uComponent, uDir, result, revResult);
            }
            else {
                auto sampleInternal = static_cast<ProgSigBSDFSampleInternal>(selProcSet.progSampleInternal);
                value = sampleInternal(selectedBsdfBody, query, uComponent, uDir, result);
            }

            // JP: 逆方向の確率密度を求めるための諸量を計算する。
            // EN: calculate quantities for reverse probability density.
            float revWeights[4];
            float sumRevWeights = 0;
            if (revResult) {
                BSDFQuery revQuery = query;// mQuery?
                Vector3D revDirIn = result->dirLocal;
                vlr::_swap(revQuery.dirLocal, revDirIn);
                revQuery.transportMode ^= 0b1;
                for (int i = 0; i < m_numBSDFs; ++i) {
                    revWeights[i] = BSDFWeight(bsdfOffsets[i], revQuery);
                    sumRevWeights += revWeights[i];
                }
            }

            result->dirPDF *= weights[idx];
            if (revResult)
                revResult->value *= revWeights[idx];
            if (result->dirPDF == 0.0f)
                return SampledSpectrum::Zero();

            // JP: サンプルした方向に関するBSDFの値の合計と、single-sample model MISに基づいた確率密度を計算する。
            // EN: calculate the total of BSDF values and a PDF based on the single-sample model MIS for the sampled direction.
            if (!result->sampledType.isDelta()) {
                for (int i = 0; i < m_numBSDFs; ++i) {
                    const uint32_t* bsdfBody;
                    BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                    if (revResult) {
                        auto evaluatePDFInternal = static_cast<ProgSigBSDFEvaluatePDFWithRevInternal>(procSet.progEvaluatePDFWithRevInternal);
                        if (i != idx && weights[i] > 0.0f) {
                            float revDirPDF;
                            float dirPDF = evaluatePDFInternal(bsdfBody, query, result->dirLocal, &revDirPDF);
                            result->dirPDF += dirPDF * weights[i];
                            revResult->dirPDF += revDirPDF * revWeights[i];
                        }
                    }
                    else {
                        auto evaluatePDFInternal = static_cast<ProgSigBSDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);
                        if (i != idx && weights[i] > 0.0f)
                            result->dirPDF += evaluatePDFInternal(bsdfBody, query, result->dirLocal) * weights[i];
                    }
                }

                BSDFQuery mQuery = query;
                mQuery.dirTypeFilter &= sideTest(query.geometricNormalLocal, query.dirLocal, result->dirLocal);
                value = SampledSpectrum::Zero();
                if (revResult)
                    revResult->value = SampledSpectrum::Zero();
                for (int i = 0; i < m_numBSDFs; ++i) {
                    const uint32_t* bsdfBody;
                    BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                    if (revResult) {
                        auto evaluateInternal = static_cast<ProgSigBSDFEvaluateWithRevInternal>(procSet.progEvaluateWithRevInternal);
                        if (weights[i] == 0.0f)
                            continue;
                        SampledSpectrum revValue;
                        value += evaluateInternal(bsdfBody, mQuery, result->dirLocal, &revValue);
                        revResult->value += revValue;
                    }
                    else {
                        auto evaluateInternal = static_cast<ProgSigBSDFEvaluateInternal>(procSet.progEvaluateInternal);
                        if (weights[i] == 0.0f)
                            continue;
                        value += evaluateInternal(bsdfBody, mQuery, result->dirLocal);
                    }
                }
            }
            result->dirPDF /= sumWeights;
            if (revResult)
                revResult->dirPDF /= sumRevWeights;

            return value;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            SampledSpectrum* revValue = nullptr) const {
            uint32_t bsdfOffsets[4] = { m_bsdf0, m_bsdf1, m_bsdf2, m_bsdf3 };

            SampledSpectrum retValue = SampledSpectrum::Zero();
            if (revValue)
                *revValue = SampledSpectrum::Zero();
            for (int i = 0; i < m_numBSDFs; ++i) {
                const uint32_t* bsdfBody;
                BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                auto matches = static_cast<ProgSigBSDFmatches>(procSet.progMatches);
                if (revValue) {
                    auto evaluateInternal = static_cast<ProgSigBSDFEvaluateWithRevInternal>(procSet.progEvaluateWithRevInternal);
                    if (!matches(bsdfBody, query.dirTypeFilter))
                        continue;
                    SampledSpectrum eRevValue;
                    retValue += evaluateInternal(bsdfBody, query, dirLocal, &eRevValue);
                    *revValue += eRevValue;
                }
                else {
                    auto evaluateInternal = static_cast<ProgSigBSDFEvaluateInternal>(procSet.progEvaluateInternal);
                    if (!matches(bsdfBody, query.dirTypeFilter))
                        continue;
                    retValue += evaluateInternal(bsdfBody, query, dirLocal);
                }
            }
            return retValue;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const BSDFQuery &query, const Vector3D &dirLocal,
            float* revValue = nullptr) const {
            if (revValue)
                *revValue = 0.0f;

            uint32_t bsdfOffsets[4] = { m_bsdf0, m_bsdf1, m_bsdf2, m_bsdf3 };

            float sumWeights = 0.0f;
            float weights[4];
            for (int i = 0; i < m_numBSDFs; ++i) {
                weights[i] = BSDFWeight(bsdfOffsets[i], query);
                sumWeights += weights[i];
            }
            if (sumWeights == 0.0f)
                return 0.0f;

            float revWeights[4];
            float sumRevWeights = 0;
            if (revValue) {
                BSDFQuery revQuery = query;// mQuery?
                Vector3D revDirIn = dirLocal;
                vlr::_swap(revQuery.dirLocal, revDirIn);
                revQuery.transportMode ^= 0b1;
                for (int i = 0; i < m_numBSDFs; ++i) {
                    revWeights[i] = BSDFWeight(bsdfOffsets[i], revQuery);
                    sumRevWeights += revWeights[i];
                }
            }

            float retPDF = 0.0f;
            if (revValue)
                *revValue = 0.0f;
            for (int i = 0; i < m_numBSDFs; ++i) {
                const uint32_t* bsdfBody;
                BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                if (revValue) {
                    auto evaluatePDFInternal = static_cast<ProgSigBSDFEvaluatePDFWithRevInternal>(procSet.progEvaluatePDFWithRevInternal);
                    if (weights[i] > 0) {
                        float revDirPDF;
                        float dirPDF = evaluatePDFInternal(bsdfBody, query, dirLocal, &revDirPDF);
                        retPDF += dirPDF * weights[i];
                        *revValue += revDirPDF * revWeights[i];
                    }
                }
                else {
                    auto evaluatePDFInternal = static_cast<ProgSigBSDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);
                    if (weights[i] > 0)
                        retPDF += evaluatePDFInternal(bsdfBody, query, dirLocal) * weights[i];
                }
            }
            retPDF /= sumWeights;
            if (revValue)
                *revValue /= sumRevWeights;

            return retPDF;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const BSDFQuery &query) const {
            uint32_t bsdfOffsets[4] = { m_bsdf0, m_bsdf1, m_bsdf2, m_bsdf3 };

            float ret = 0.0f;
            for (int i = 0; i < m_numBSDFs; ++i) {
                const uint32_t* bsdfBody;
                BSDFProcedureSet procSet = getBSDFProcSet(bsdfOffsets[i], &bsdfBody);
                auto matches = static_cast<ProgSigBSDFmatches>(procSet.progMatches);
                auto weightInternal = static_cast<ProgSigBSDFWeightInternal>(procSet.progWeightInternal);
                if (!matches(bsdfBody, query.dirTypeFilter))
                    continue;
                ret += weightInternal(bsdfBody, query);
            }

            return ret;
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(MultiSurfaceMaterial_setupBSDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<MultiBSDF*>(params);
        auto &mat = *reinterpret_cast<const MultiSurfaceMaterial*>(matDesc);

        /*
        MultiBSDF
        ---- <-- MultiBSDF::bsdf0
        ProcedureSetIndex0
        Data0
        ---- <-- MultiBSDF::bsdf1
        ProcedureSetIndex1
        Data1
        ---- <-- MultiBSDF::bsdf2
        ProcedureSetIndex2
        Data2
        ---- <-- MultiBSDF::bsdf3
        ProcedureSetIndex3
        Data3
        */
        uint32_t baseIndex = sizeof(MultiBSDF) / 4;
        uint32_t bsdfOffsets[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < mat.numSubMaterials; ++i) {
            bsdfOffsets[i] = baseIndex;

            const SurfaceMaterialDescriptor subMatDesc = plp.materialDescriptorBuffer[mat.subMatIndices[i]];
            auto setupBSDF = static_cast<ProgSigSetupBSDF>(subMatDesc.progSetupBSDF);
            *(params + baseIndex++) = subMatDesc.bsdfProcedureSetIndex;
            baseIndex += setupBSDF(subMatDesc.data, surfPt, wls, params + baseIndex);
        }

        p = MultiBSDF(
            bsdfOffsets[0],
            bsdfOffsets[1],
            bsdfOffsets[2],
            bsdfOffsets[3],
            mat.numSubMaterials);

        //vlrDevPrintf("%u, %u, %u, %u, %u mats\n", p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3, p.numBSDFs);

        return baseIndex;
    }

    DEFINE_BSDF_CALLABLE_PROGRAMS(MultiBSDF)

    // edf0-3: param offsets
    // numEDFs
    // --------------------------------
    // EDF0 procedure set index
    // EDF0 params
    // ...
    // EDF3 procedure set index
    // EDF3 params
    class MultiEDF {
        unsigned int m_edf0 : 6;
        unsigned int m_edf1 : 6;
        unsigned int m_edf2 : 6;
        unsigned int m_edf3 : 6;
        unsigned int m_numEDFs : 8;

        CUDA_DEVICE_FUNCTION const uint32_t* getEDF(uint32_t offset) const {
            return reinterpret_cast<const uint32_t*>(this) + offset;
        }

    public:
        CUDA_DEVICE_FUNCTION MultiEDF(
            uint32_t edf0, uint32_t edf1, uint32_t edf2, uint32_t edf3,
            uint32_t numEDFs) :
            m_edf0(edf0), m_edf1(edf1), m_edf2(edf2), m_edf3(edf3),
            m_numEDFs(numEDFs) {}

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            uint32_t edfOffsets[4] = { m_edf0, m_edf1, m_edf2, m_edf3 };

            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto matches = static_cast<ProgSigEDFmatches>(procSet.progMatches);

                if (matches(edf + 1, flags))
                    return true;
            }

            return false;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            uint32_t edfOffsets[4] = { m_edf0, m_edf1, m_edf2, m_edf3 };

            float weights[4];
            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto weightInternal = static_cast<ProgSigEDFWeightInternal>(procSet.progWeightInternal);

                weights[i] = weightInternal(edf + 1, query);
            }

            // JP: 各EDFのウェイトに基づいて方向のサンプルを行うEDFを選択する。
            // EN: Based on the weight of each EDF, select a EDF from which direction sampling.
            float tempProb;
            float sumWeights;
            uint32_t idx = sampleDiscrete(weights, m_numEDFs, uComponent, &tempProb, &sumWeights, &uComponent);
            if (sumWeights == 0.0f) {
                result->dirPDF = 0.0f;
                return SampledSpectrum::Zero();
            }

            const uint32_t* selectedEDF = getEDF(edfOffsets[idx]);
            uint32_t selProcIdx = *reinterpret_cast<const uint32_t*>(selectedEDF);
            const EDFProcedureSet selProcSet = plp.edfProcedureSetBuffer[selProcIdx];
            auto sampleInternal = static_cast<ProgSigEDFSampleInternal>(selProcSet.progSampleInternal);

            // JP: 選択したEDFから方向をサンプリングする。
            // EN: sample a direction from the selected EDF.
            SampledSpectrum value = sampleInternal(selectedEDF + 1, query, uComponent, uDir, result);
            result->dirPDF *= weights[idx];
            if (result->dirPDF == 0.0f) {
                result->dirPDF = 0.0f;
                return SampledSpectrum::Zero();
            }

            // JP: サンプルした方向に関するEDFの値の合計と、single-sample model MISに基づいた確率密度を計算する。
            // EN: calculate the total of EDF values and a PDF based on the single-sample model MIS for the sampled direction.
            if (!result->sampledType.isDelta()) {
                for (int i = 0; i < m_numEDFs; ++i) {
                    const uint32_t* edf = getEDF(edfOffsets[i]);
                    uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);
                    const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                    auto matches = static_cast<ProgSigEDFmatches>(procSet.progMatches);
                    auto evaluatePDFInternal = static_cast<ProgSigEDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);

                    if (i != idx && matches(edf + 1, query.dirTypeFilter))
                        result->dirPDF += evaluatePDFInternal(edf + 1, query, result->dirLocal) * weights[i];
                }

                value = SampledSpectrum::Zero();
                for (int i = 0; i < m_numEDFs; ++i) {
                    const uint32_t* edf = getEDF(edfOffsets[i]);
                    uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);
                    const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                    auto matches = static_cast<ProgSigEDFmatches>(procSet.progMatches);
                    auto evaluateInternal = static_cast<ProgSigEDFEvaluateInternal>(procSet.progEvaluateInternal);

                    if (!matches(edf + 1, query.dirTypeFilter))
                        continue;
                    value += evaluateInternal(edf + 1, query, result->dirLocal);
                }
            }
            result->dirPDF /= sumWeights;

            return value;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            uint32_t edfOffsets[4] = { m_edf0, m_edf1, m_edf2, m_edf3 };

            SampledSpectrum ret = SampledSpectrum::Zero();
            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto evaluateEmittanceInternal = static_cast<ProgSigEDFEvaluateEmittanceInternal>(procSet.progEvaluateEmittanceInternal);

                ret += evaluateEmittanceInternal(edf + 1);
            }

            return ret;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            uint32_t edfOffsets[4] = { m_edf0, m_edf1, m_edf2, m_edf3 };

            SampledSpectrum ret = SampledSpectrum::Zero();
            SampledSpectrum sumEmittance = SampledSpectrum::Zero();
            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto evaluateEmittanceInternal = static_cast<ProgSigEDFEvaluateEmittanceInternal>(procSet.progEvaluateEmittanceInternal);
                auto evaluateInternal = static_cast<ProgSigEDFEvaluateInternal>(procSet.progEvaluateInternal);

                SampledSpectrum emittance = evaluateEmittanceInternal(edf + 1);
                sumEmittance += emittance;
                ret += emittance * evaluateInternal(edf + 1, query, dirLocal);
            }
            ret.safeDivide(sumEmittance);

            return ret;
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            uint32_t edfOffsets[4] = { m_edf0, m_edf1, m_edf2, m_edf3 };

            float sumWeights = 0.0f;
            float weights[4];
            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);;
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto weightInternal = static_cast<ProgSigEDFWeightInternal>(procSet.progWeightInternal);

                weights[i] = weightInternal(edf + 1, query);
                sumWeights += weights[i];
            }
            if (sumWeights == 0.0f)
                return 0.0f;

            float retPDF = 0.0f;
            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);;
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto evaluatePDFInternal = static_cast<ProgSigEDFEvaluatePDFInternal>(procSet.progEvaluatePDFInternal);

                if (weights[i] > 0)
                    retPDF += evaluatePDFInternal(edf + 1, query, dirLocal) * weights[i];
            }
            retPDF /= sumWeights;

            return retPDF;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const EDFQuery &query) const {
            uint32_t edfOffsets[4] = { m_edf0, m_edf1, m_edf2, m_edf3 };

            float ret = 0.0f;
            for (int i = 0; i < m_numEDFs; ++i) {
                const uint32_t* edf = getEDF(edfOffsets[i]);
                uint32_t procIdx = *reinterpret_cast<const uint32_t*>(edf);;
                const EDFProcedureSet procSet = plp.edfProcedureSetBuffer[procIdx];
                auto weightInternal = static_cast<ProgSigEDFWeightInternal>(procSet.progWeightInternal);

                ret += weightInternal(edf + 1, query);
            }

            return ret;
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(MultiSurfaceMaterial_setupEDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<MultiEDF*>(params);
        auto &mat = *reinterpret_cast<const MultiSurfaceMaterial*>(matDesc);

        uint32_t baseIndex = sizeof(MultiEDF) / 4;
        uint32_t edfOffsets[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < mat.numSubMaterials; ++i) {
            edfOffsets[i] = baseIndex;

            const SurfaceMaterialDescriptor subMatDesc = plp.materialDescriptorBuffer[mat.subMatIndices[i]];
            ProgSigSetupEDF setupEDF = (ProgSigSetupEDF)subMatDesc.progSetupEDF;
            *(params + baseIndex++) = subMatDesc.edfProcedureSetIndex;
            baseIndex += setupEDF(subMatDesc.data, surfPt, wls, params + baseIndex);
        }

        p = MultiEDF(
            edfOffsets[0],
            edfOffsets[1],
            edfOffsets[2],
            edfOffsets[3],
            mat.numSubMaterials);

        return baseIndex;
    }

    DEFINE_EDF_CALLABLE_PROGRAMS(MultiEDF)
    DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(MultiEDF)

    // END: MultiBSDF / MultiEDF
    // ----------------------------------------------------------------



    class EnvironmentEDF {
        SampledSpectrum m_emittance;
        float m_worldRadius;
        float m_worldDiscArea;

    public:
        CUDA_DEVICE_FUNCTION EnvironmentEDF(
            const SampledSpectrum &emittance, float worldRadius, float worldDiscArea) :
            m_emittance(emittance), m_worldRadius(worldRadius), m_worldDiscArea(worldDiscArea) {}

        CUDA_DEVICE_FUNCTION bool matches(DirectionType flags) const {
            DirectionType dirType = DirectionType::Emission() | DirectionType::LowFreq();
            return dirType.matches(flags);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const EDFQuery &query, float uComponent, const float uDir[2], EDFQueryResult* result) const {
            float dx, dy;
            concentricSampleDisk(uDir[0], uDir[1], &dx, &dy);

            // encode the projected position as the xy components.
            result->dirLocal = Vector3D(m_worldRadius * dx, m_worldRadius * dy, 1);
            // The true value is: lim_{l to inf} l^2 / m_worldDiscArea
            result->dirPDF = 1.0f / m_worldDiscArea;
            result->sampledType = DirectionType::Emission() | DirectionType::LowFreq();
            SampledSpectrum feValue(1 / VLR_M_PI);

            return feValue;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateEmittanceInternal() const {
            return VLR_M_PI * m_emittance;
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            return SampledSpectrum(dirLocal.z > 0.0f ? 1.0f / VLR_M_PI : 0.0f);
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const EDFQuery &query, const Vector3D &dirLocal) const {
            if (dirLocal.z <= 0.0f)
                return 0.0f;
            // The true value is: lim_{l to inf} l^2 / m_worldDiscArea
            float pdfValue = 1.0f / m_worldDiscArea;

            return pdfValue;
        }

        CUDA_DEVICE_FUNCTION float weightInternal(const EDFQuery &query) const {
            return m_emittance.importance(query.wlHint);
        }
    };

    RT_CALLABLE_PROGRAM uint32_t RT_DC_NAME(EnvironmentEmitterSurfaceMaterial_setupEDF)(
        const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<EnvironmentEDF*>(params);
        auto &mat = *reinterpret_cast<const EnvironmentEmitterSurfaceMaterial*>(matDesc);

        p = EnvironmentEDF(calcNode(mat.nodeEmittance, mat.immEmittance, surfPt, wls) * mat.immScale,
                           plp.sceneBounds->worldRadius, plp.sceneBounds->worldDiscArea);

        return sizeof(EnvironmentEDF) / 4;
    }

    DEFINE_EDF_CALLABLE_PROGRAMS(EnvironmentEDF)
    DEFINE_EDF_AS_BSDF_CALLABLE_PROGRAMS(EnvironmentEDF)
}
