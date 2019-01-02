#include "kernel_common.cuh"

namespace VLR {
    RT_FUNCTION DirectionType sideTest(const Normal3D &ng, const Vector3D &d0, const Vector3D &d1) {
        bool reflect = dot(Vector3D(ng), d0) * dot(Vector3D(ng), d1) > 0;
        return DirectionType::AllFreq() | (reflect ? DirectionType::Reflection() : DirectionType::Transmission());
    }



    class FresnelConductor {
        SampledSpectrum m_eta;
        SampledSpectrum m_k;

    public:
        RT_FUNCTION FresnelConductor(const SampledSpectrum &eta, const SampledSpectrum &k) : m_eta(eta), m_k(k) {}

        RT_FUNCTION SampledSpectrum evaluate(float cosEnter) const {
            cosEnter = std::fabs(cosEnter);
            float cosEnter2 = cosEnter * cosEnter;
            SampledSpectrum _2EtaCosEnter = 2.0f * m_eta * cosEnter;
            SampledSpectrum tmp_f = m_eta * m_eta + m_k * m_k;
            SampledSpectrum tmp = tmp_f * cosEnter2;
            SampledSpectrum Rparl2 = (tmp - _2EtaCosEnter + 1) / (tmp + _2EtaCosEnter + 1);
            SampledSpectrum Rperp2 = (tmp_f - _2EtaCosEnter + cosEnter2) / (tmp_f + _2EtaCosEnter + cosEnter2);

            return (Rparl2 + Rperp2) / 2.0f;
        }
        RT_FUNCTION float evaluate(float cosEnter, uint32_t wlIdx) const {
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
        RT_FUNCTION FresnelDielectric(const SampledSpectrum &etaExt, const SampledSpectrum &etaInt) : m_etaExt(etaExt), m_etaInt(etaInt) {}

        RT_FUNCTION SampledSpectrum etaExt() const { return m_etaExt; }
        RT_FUNCTION SampledSpectrum etaInt() const { return m_etaInt; }

        RT_FUNCTION SampledSpectrum evaluate(float cosEnter) const {
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
        RT_FUNCTION float evaluate(float cosEnter, uint32_t wlIdx) const {
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

        RT_FUNCTION static float evalF(float etaEnter, float etaExit, float cosEnter, float cosExit);
    };

    RT_FUNCTION float FresnelDielectric::evalF(float etaEnter, float etaExit, float cosEnter, float cosExit) {
        float Rparl = ((etaExit * cosEnter) - (etaEnter * cosExit)) / ((etaExit * cosEnter) + (etaEnter * cosExit));
        float Rperp = ((etaEnter * cosEnter) - (etaExit * cosExit)) / ((etaEnter * cosEnter) + (etaExit * cosExit));
        return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
    }



    class FresnelSchlick {
        // assume vacuum-dielectric interface
        float m_F0;

    public:
        RT_FUNCTION FresnelSchlick(float F0) : m_F0(F0) {}

        RT_FUNCTION SampledSpectrum evaluate(float cosEnter) const {
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
        RT_FUNCTION GGXMicrofacetDistribution(float alpha_gx, float alpha_gy, float rotation) :
            m_alpha_gx(alpha_gx), m_alpha_gy(alpha_gy) {
            VLR::sincos(rotation, &m_sinRt, &m_cosRt);
        }

        RT_FUNCTION float evaluate(const Normal3D &m) {
            Normal3D mr = Normal3D(m_cosRt * m.x + m_sinRt * m.y,
                                   -m_sinRt * m.x + m_cosRt * m.y,
                                   m.z);

            if (mr.z <= 0)
                return 0.0f;
            float temp = pow2(mr.x / m_alpha_gx) + pow2(mr.y / m_alpha_gy) + pow2(mr.z);
            return 1.0f / (M_PIf * m_alpha_gx * m_alpha_gy * pow2(temp));
        }

        RT_FUNCTION float evaluateSmithG1(const Vector3D &v, const Normal3D &m) {
            Vector3D vr = Vector3D(m_cosRt * v.x + m_sinRt * v.y,
                                   -m_sinRt * v.x + m_cosRt * v.y,
                                   v.z);

            float alpha_g2_tanTheta2 = (pow2(vr.x * m_alpha_gx) + pow2(vr.y * m_alpha_gy)) / pow2(vr.z);
            float Lambda = (-1 + std::sqrt(1 + alpha_g2_tanTheta2)) / 2;
            float chi = (dot(v, m) / v.z) > 0 ? 1 : 0;
            return chi / (1 + Lambda);
        }

        RT_FUNCTION float evaluateHeightCorrelatedSmithG(const Vector3D &v1, const Vector3D &v2, const Normal3D &m) {
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

        RT_FUNCTION float sample(const Vector3D &v, float u0, float u1, Normal3D* m, float* normalPDF) {
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
            float phi = M_PIf * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            VLR::sincos(phi, &sinPhi, &cosPhi);
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

        RT_FUNCTION float evaluatePDF(const Vector3D &v, const Normal3D &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };



    // ----------------------------------------------------------------
    // NullBSDF

    RT_CALLABLE_PROGRAM uint32_t NullBSDF_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        return 0;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum NullBSDF_getBaseColor(const uint32_t* params) {
        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM bool NullBSDF_matches(const uint32_t* params, DirectionType flags) {
        return false;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum NullBSDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM SampledSpectrum NullBSDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float NullBSDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM float NullBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        return 0.0f;
    }

    // END: NullBSDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // MatteBRDF

    struct MatteBRDF {
        SampledSpectrum albedo;
        float roughness;
    };

    RT_CALLABLE_PROGRAM uint32_t MatteSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        MatteBRDF &p = *(MatteBRDF*)params;
        auto &mat = *(const MatteSurfaceMaterial*)matDesc;

        p.albedo = calcNode(mat.nodeAlbedo, mat.immAlbedo, surfPt, wls);
        p.roughness = 0.0f;

        return sizeof(MatteBRDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MatteBRDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const MatteBRDF*)params;

        return p.albedo;
    }

    RT_CALLABLE_PROGRAM bool MatteBRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MatteBRDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const MatteBRDF*)params;

        result->dirLocal = cosineSampleHemisphere(uDir[0], uDir[1]);
        result->dirPDF = result->dirLocal.z / M_PIf;
        result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();
        result->dirLocal.z *= query.dirLocal.z >= 0 ? 1 : -1;

        return p.albedo / M_PIf;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MatteBRDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MatteBRDF*)params;

        if (query.dirLocal.z * dirLocal.z <= 0.0f) {
            SampledSpectrum fs = SampledSpectrum::Zero();
            return fs;
        }
        SampledSpectrum fs = p.albedo / M_PIf;

        return fs;
    }

    RT_CALLABLE_PROGRAM float MatteBRDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        if (query.dirLocal.z * dirLocal.z <= 0.0f) {
            return 0.0f;
        }

        return std::fabs(dirLocal.z) / M_PIf;
    }

    RT_CALLABLE_PROGRAM float MatteBRDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const MatteBRDF*)params;
        return p.albedo.importance(query.wlHint);
    }

    // END: MatteBRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // SpecularBRDF

    struct SpecularBRDF {
        SampledSpectrum coeffR;
        SampledSpectrum eta;
        SampledSpectrum k;
    };

    RT_CALLABLE_PROGRAM uint32_t SpecularReflectionSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(SpecularBRDF*)params;
        auto &mat = *(const SpecularReflectionSurfaceMaterial*)matDesc;

        p.coeffR = calcNode(mat.nodeCoeffR, mat.immCoeffR, surfPt, wls);
        p.eta = calcNode(mat.nodeEta, mat.immEta, surfPt, wls);
        p.k = calcNode(mat.node_k, mat.imm_k, surfPt, wls);

        return sizeof(SpecularBRDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum SpecularBRDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const SpecularBRDF*)params;

        return p.coeffR;
    }

    RT_CALLABLE_PROGRAM bool SpecularBRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::Delta0D();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum SpecularBRDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const SpecularBRDF*)params;

        FresnelConductor fresnel(p.eta, p.k);

        result->dirLocal = Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z);
        result->dirPDF = 1.0f;
        result->sampledType = DirectionType::Reflection() | DirectionType::Delta0D();
        SampledSpectrum fs = p.coeffR * fresnel.evaluate(query.dirLocal.z) / std::fabs(query.dirLocal.z);

        return fs;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum SpecularBRDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float SpecularBRDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM float SpecularBRDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const SpecularBRDF*)params;

        FresnelDielectric fresnel(p.eta, p.k);

        return (p.coeffR * fresnel.evaluate(query.dirLocal.z)).importance(query.wlHint);
    }

    // END: SpecularBRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // SpecularBSDF

    struct SpecularBSDF {
        SampledSpectrum coeff;
        SampledSpectrum etaExt;
        SampledSpectrum etaInt;
        bool dispersive;
    };

    RT_CALLABLE_PROGRAM uint32_t SpecularScatteringSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(SpecularBSDF*)params;
        auto &mat = *(const SpecularScatteringSurfaceMaterial*)matDesc;

        p.coeff = calcNode(mat.nodeCoeff, mat.immCoeff, surfPt, wls);
        p.etaExt = calcNode(mat.nodeEtaExt, mat.immEtaExt, surfPt, wls);
        p.etaInt = calcNode(mat.nodeEtaInt, mat.immEtaInt, surfPt, wls);
        p.dispersive = !wls.singleIsSelected();

        return sizeof(SpecularBSDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum SpecularBSDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const SpecularBSDF*)params;

        return p.coeff;
    }

    RT_CALLABLE_PROGRAM bool SpecularBSDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::WholeSphere() | DirectionType::Delta0D();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum SpecularBSDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const SpecularBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        const SampledSpectrum &eEnter = entering ? p.etaExt : p.etaInt;
        const SampledSpectrum &eExit = entering ? p.etaInt : p.etaExt;
        FresnelDielectric fresnel(eEnter, eExit);

        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

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
            result->dirPDF = reflectProb;
            result->sampledType = DirectionType::Reflection() | DirectionType::Delta0D();
            SampledSpectrum fs = p.coeff * F / std::fabs(dirV.z);

            return fs;
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
            result->dirPDF = 1.0f - reflectProb;
            result->sampledType = DirectionType::Transmission() | DirectionType::Delta0D() | (p.dispersive ? DirectionType::Dispersive() : DirectionType());

            SampledSpectrum ret = SampledSpectrum::Zero();
            ret[query.wlHint] = p.coeff[query.wlHint] * (1.0f - F[query.wlHint]);
            SampledSpectrum fs = ret / std::fabs(cosExit);

            return fs;
        }
    }

    RT_CALLABLE_PROGRAM SampledSpectrum SpecularBSDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float SpecularBSDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM float SpecularBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const SpecularBSDF*)params;
        return p.coeff.importance(query.wlHint);
    }

    // END: SpecularBSDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // MicrofacetBRDF

    struct MicrofacetBRDF {
        SampledSpectrum eta;
        SampledSpectrum k;
        float alphaX;
        float alphaY;
        float rotation;
    };

    RT_CALLABLE_PROGRAM uint32_t MicrofacetReflectionSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(MicrofacetBRDF*)params;
        auto &mat = *(const MicrofacetReflectionSurfaceMaterial*)matDesc;

        p.eta = calcNode(mat.nodeEta, mat.immEta, surfPt, wls);
        p.k = calcNode(mat.node_k, mat.imm_k, surfPt, wls);
        optix::float3 roughnessAnisotropyRotation = calcNode(mat.nodeRoughnessAnisotropyRotation, 
                                                             optix::make_float3(mat.immRoughness, mat.immAnisotropy, mat.immRotation), 
                                                             surfPt, wls);
        float alpha = pow2(roughnessAnisotropyRotation.x);
        float aspect = std::sqrt(1.0f - 0.9f * roughnessAnisotropyRotation.y);
        p.alphaX = std::fmax(0.001f, alpha / aspect);
        p.alphaY = std::fmax(0.001f, alpha * aspect);
        p.rotation = 2 * M_PIf * roughnessAnisotropyRotation.z;

        return sizeof(MicrofacetBRDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MicrofacetBRDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const MicrofacetBRDF*)params;

        FresnelConductor fresnel(p.eta, p.k);

        return fresnel.evaluate(1.0f);
    }

    RT_CALLABLE_PROGRAM bool MicrofacetBRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::HighFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MicrofacetBRDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const MicrofacetBRDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        FresnelConductor fresnel(p.eta, p.k);

        GGXMicrofacetDistribution ggx(p.alphaX, p.alphaY, p.rotation);

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

        //VLRAssert(fs.allFinite(), "fs: %s, F: %s, G, %g, D: %g, wlIdx: %u, qDir: %s, rDir: %s",
        //          fs.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

        return fs;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MicrofacetBRDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MicrofacetBRDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        FresnelConductor fresnel(p.eta, p.k);

        GGXMicrofacetDistribution ggx(p.alphaX, p.alphaY, p.rotation);

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

        //VLRAssert(fs.allFinite(), "fs: %s, F: %s, G, %g, D: %g, wlIdx: %u, qDir: %s, dir: %s",
        //          fs.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

        return fs;
    }

    RT_CALLABLE_PROGRAM float MicrofacetBRDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MicrofacetBRDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        FresnelConductor fresnel(p.eta, p.k);

        GGXMicrofacetDistribution ggx(p.alphaX, p.alphaY, p.rotation);

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

        //VLRAssert(std::isfinite(commonPDFTerm) && std::isfinite(mPDF),
        //          "commonPDFTerm: %g, mPDF: %g, wlIdx: %u, qDir: %s, dir: %s",
        //          commonPDFTerm, mPDF, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

        return ret;
    }

    RT_CALLABLE_PROGRAM float MicrofacetBRDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const MicrofacetBRDF*)params;

        FresnelConductor fresnel(p.eta, p.k);

        float expectedDotHV = query.dirLocal.z;

        return fresnel.evaluate(expectedDotHV).importance(query.wlHint);
    }

    // END: MicrofacetBRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // MicrofacetBSDF

    struct MicrofacetBSDF {
        SampledSpectrum coeff;
        SampledSpectrum etaExt;
        SampledSpectrum etaInt;
        float alphaX;
        float alphaY;
        float rotation;
    };

    RT_CALLABLE_PROGRAM uint32_t MicrofacetScatteringSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(MicrofacetBSDF*)params;
        auto &mat = *(const MicrofacetScatteringSurfaceMaterial*)matDesc;

        p.coeff = calcNode(mat.nodeCoeff, mat.immCoeff, surfPt, wls);
        p.etaExt = calcNode(mat.nodeEtaExt, mat.immEtaExt, surfPt, wls);
        p.etaInt = calcNode(mat.nodeEtaInt, mat.immEtaInt, surfPt, wls);
        optix::float3 roughnessAnisotropyRotation = calcNode(mat.nodeRoughnessAnisotropyRotation,
                                                             optix::make_float3(mat.immRoughness, mat.immAnisotropy, mat.immRotation), 
                                                             surfPt, wls);
        float alpha = pow2(roughnessAnisotropyRotation.x);
        float aspect = std::sqrt(1 - 0.9f * roughnessAnisotropyRotation.y);
        p.alphaX = std::fmax(0.001f, alpha / aspect);
        p.alphaY = std::fmax(0.001f, alpha * aspect);
        p.rotation = 2 * M_PIf * roughnessAnisotropyRotation.z;

        return sizeof(MicrofacetBSDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MicrofacetBSDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const MicrofacetBSDF*)params;

        return p.coeff;
    }

    RT_CALLABLE_PROGRAM bool MicrofacetBSDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::WholeSphere() | DirectionType::HighFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MicrofacetBSDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const MicrofacetBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        const SampledSpectrum &eEnter = entering ? p.etaExt : p.etaInt;
        const SampledSpectrum &eExit = entering ? p.etaInt : p.etaExt;
        FresnelDielectric fresnel(eEnter, eExit);

        GGXMicrofacetDistribution ggx(p.alphaX, p.alphaY, p.rotation);

        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        // JP: ハーフベクトルをサンプルする。
        // EN: sample a half vector.
        Normal3D m;
        float mPDF;
        float D = ggx.sample(dirV, uDir[0], uDir[1], &m, &mPDF);
        float dotHV = dot(dirV, m);
        if (dotHV <= 0 || std::isnan(D)) {
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
            SampledSpectrum fs = F * D * G / (4 * dirV.z * dirL.z);

            //VLRAssert(fs.allFinite(), "fs: %s, F: %g, %g, %g, G, %g, D: %g, wlIdx: %u, qDir: (%g, %g, %g), rDir: (%g, %g, %g)",
            //          fs.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, 
            //          dirV.x, dirV.y, dirV.z, dirL.x, dirL.y, dirL.z);

            return fs;
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
            float commonPDFTerm = (1 - reflectProb) / std::pow(eEnter[query.wlHint] * dotHV + eExit[query.wlHint] * dotHL, 2);
            result->dirPDF = commonPDFTerm * mPDF * eExit[query.wlHint] * eExit[query.wlHint] * std::fabs(dotHL);
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
                ret[wlIdx] = std::fabs(dotHV_wl * dotHL_wl) * (1 - F_wl) * G_wl * D_wl / std::pow(eEnter[wlIdx] * dotHV_wl + eExit[wlIdx] * dotHL_wl, 2);

                //VLRAssert(std::isfinite(ret[wlIdx]), "fs: %g, F: %g, G, %g, D: %g, wlIdx: %u, qDir: %s",
                //          ret[wlIdx], F_wl, G_wl, D_wl, query.wlHint, dirV.toString().c_str());
            }
            ret /= std::fabs(dirV.z * dirL.z);
            ret *= eEnter * eEnter;
            //ret *= query.adjoint ? (eExit * eExit) : (eEnter * eEnter);// adjoint: need to cancel eEnter^2 / eExit^2 => eEnter^2 * (eExit^2 / eEnter^2)

            //VLRAssert(ret.allFinite(), "fs: %s, wlIdx: %u, qDir: %s, rDir: %s",
            //          ret.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return ret;
        }
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MicrofacetBSDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MicrofacetBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        const SampledSpectrum &eEnter = entering ? p.etaExt : p.etaInt;
        const SampledSpectrum &eExit = entering ? p.etaInt : p.etaExt;
        FresnelDielectric fresnel(eEnter, eExit);

        GGXMicrofacetDistribution ggx(p.alphaX, p.alphaY, p.rotation);

        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? dirLocal : -dirLocal;
        float dotNVdotNL = dirL.z * dirV.z;

        if (dotNVdotNL > 0 && query.dirTypeFilter.matches(DirectionType::Reflection() | DirectionType::AllFreq())) {
            Normal3D m = halfVector(dirV, dirL);
            float dotHV = dot(dirV, m);
            float D = ggx.evaluate(m);

            SampledSpectrum F = fresnel.evaluate(dotHV);
            float G = ggx.evaluateSmithG1(dirV, m) * ggx.evaluateSmithG1(dirL, m);
            SampledSpectrum fs = F * D * G / (4 * dotNVdotNL);

            //VLRAssert(fs.allFinite(), "fs: %s, F: %s, G, %g, D: %g, wlIdx: %u, qDir: %s, dir: %s",
            //          fs.toString().c_str(), F.toString().c_str(), G, D, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return fs;
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
                ret[wlIdx] = std::fabs(dotHV_wl * dotHL_wl) * (1 - F_wl) * G_wl * D_wl / std::pow(eEnter[wlIdx] * dotHV_wl + eExit[wlIdx] * dotHL_wl, 2);

                //VLRAssert(std::isfinite(ret[wlIdx]), "fs: %g, F: %g, G, %g, D: %g, wlIdx: %u, qDir: %s, dir: %s",
                //          ret[wlIdx], F_wl, G_wl, D_wl, query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());
            }
            ret /= std::fabs(dotNVdotNL);
            ret *= eEnter * eEnter;
            //ret *= query.adjoint ? (eExit * eExit) : (eEnter * eEnter);// !adjoint: eExit^2 * (eEnter / eExit)^2

            //VLRAssert(ret.allFinite(), "fs: %s, wlIdx: %u, qDir: %s, dir: %s",
            //          ret.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return ret;
        }

        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float MicrofacetBSDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MicrofacetBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        const SampledSpectrum &eEnter = entering ? p.etaExt : p.etaInt;
        const SampledSpectrum &eExit = entering ? p.etaInt : p.etaExt;
        FresnelDielectric fresnel(eEnter, eExit);

        GGXMicrofacetDistribution ggx(p.alphaX, p.alphaY, p.rotation);

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

            //VLRAssert(std::isfinite(commonPDFTerm) && std::isfinite(mPDF),
            //          "commonPDFTerm: %g, mPDF: %g, F: %s, wlIdx: %u, qDir: %s, dir: %s",
            //          commonPDFTerm, mPDF, F.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return commonPDFTerm * mPDF;
        }
        else {
            float dotHL = dot(dirL, m);
            float commonPDFTerm = (1 - reflectProb) / std::pow(eEnter[query.wlHint] * dotHV + eExit[query.wlHint] * dotHL, 2);

            //VLRAssert(std::isfinite(commonPDFTerm) && std::isfinite(mPDF),
            //          "commonPDFTerm: %g, mPDF: %g, F: %s, wlIdx: %u, qDir: %s, dir: %s",
            //          commonPDFTerm, mPDF, F.toString().c_str(), query.wlHint, dirV.toString().c_str(), dirL.toString().c_str());

            return commonPDFTerm * mPDF * eExit[query.wlHint] * eExit[query.wlHint] * std::fabs(dotHL);
        }
    }

    RT_CALLABLE_PROGRAM float MicrofacetBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const MicrofacetBSDF*)params;
        return p.coeff.importance(query.wlHint);
    }

    // END: MicrofacetBSDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // LambertianBSDF

    struct LambertianBSDF {
        SampledSpectrum coeff;
        float F0;
    };

    RT_CALLABLE_PROGRAM uint32_t LambertianScatteringSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(LambertianBSDF*)params;
        auto &mat = *(const LambertianScatteringSurfaceMaterial*)matDesc;

        p.coeff = calcNode(mat.nodeCoeff, mat.immCoeff, surfPt, wls);
        p.F0 = calcNode(mat.nodeF0, mat.immF0, surfPt, wls);

        return sizeof(LambertianBSDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum LambertianBSDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const LambertianBSDF*)params;

        return p.coeff;
    }

    RT_CALLABLE_PROGRAM bool LambertianBSDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::WholeSphere() | DirectionType::LowFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum LambertianBSDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const LambertianBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        FresnelSchlick fresnel(p.F0);

        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = cosineSampleHemisphere(uDir[0], uDir[1]);
        result->dirPDF = dirL.z / M_PIf;

        SampledSpectrum F = fresnel.evaluate(query.dirLocal.z);
        float reflectProb = F.importance(query.wlHint);
        if (query.dirTypeFilter.isReflection())
            reflectProb = 1.0f;
        if (query.dirTypeFilter.isTransmission())
            reflectProb = 0.0f;

        if (uComponent < reflectProb) {
            result->dirLocal = entering ? dirL : -dirL;
            result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();
            SampledSpectrum fs = F * p.coeff / M_PIf;
            result->dirPDF *= reflectProb;

            return fs;
        }
        else {
            result->dirLocal = entering ? -dirL : dirL;
            result->sampledType = DirectionType::Transmission() | DirectionType::LowFreq();
            SampledSpectrum fs = (SampledSpectrum::One() - F) * p.coeff / M_PIf;
            result->dirPDF *= (1 - reflectProb);

            return fs;
        }
    }

    RT_CALLABLE_PROGRAM SampledSpectrum LambertianBSDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const LambertianBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        FresnelSchlick fresnel(p.F0);

        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? dirLocal : -dirLocal;

        SampledSpectrum F = fresnel.evaluate(query.dirLocal.z);

        if (dirV.z * dirL.z > 0.0f) {
            SampledSpectrum fs = F * p.coeff / M_PIf;
            return fs;
        }
        else {
            SampledSpectrum fs = (SampledSpectrum::One() - F) * p.coeff / M_PIf;
            return fs;
        }
    }

    RT_CALLABLE_PROGRAM float LambertianBSDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const LambertianBSDF*)params;

        bool entering = query.dirLocal.z >= 0.0f;

        FresnelSchlick fresnel(p.F0);

        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? dirLocal : -dirLocal;

        SampledSpectrum F = fresnel.evaluate(query.dirLocal.z);
        float reflectProb = F.importance(query.wlHint);
        if (query.dirTypeFilter.isReflection())
            reflectProb = 1.0f;
        if (query.dirTypeFilter.isTransmission())
            reflectProb = 0.0f;

        if (dirV.z * dirL.z > 0.0f) {
            float dirPDF = reflectProb * dirL.z / M_PIf;
            return dirPDF;
        }
        else {
            float dirPDF = (1 - reflectProb) * std::fabs(dirL.z) / M_PIf;
            return dirPDF;
        }
    }

    RT_CALLABLE_PROGRAM float LambertianBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const LambertianBSDF*)params;
        return p.coeff.importance(query.wlHint);
    }

    // END: LambertianBSDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // UE4 (Modified) BRDF

#define USE_HEIGHT_CORRELATED_SMITH

    struct UE4BRDF {
        SampledSpectrum baseColor;
        float roughness;
        float metallic;
    };

    RT_CALLABLE_PROGRAM uint32_t UE4SurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(UE4BRDF*)params;
        auto &mat = *(const UE4SurfaceMaterial*)matDesc;

        p.baseColor = calcNode(mat.nodeBaseColor, mat.immBaseColor, surfPt, wls);
        optix::float3 occlusionRoughnessMetallic = calcNode(mat.nodeOcclusionRoughnessMetallic, 
                                                            optix::make_float3(mat.immOcclusion, mat.immRoughness, mat.immMetallic), 
                                                            surfPt, wls);
        p.roughness = std::fmax(0.01f, occlusionRoughnessMetallic.y);
        p.metallic = occlusionRoughnessMetallic.z;

        return sizeof(UE4BRDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum UE4BRDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const UE4BRDF*)params;

        return p.baseColor;
    }

    RT_CALLABLE_PROGRAM bool UE4BRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq() | DirectionType::HighFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum UE4BRDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const UE4BRDF*)params;

        const float specular = 0.5f;
        float alpha = p.roughness * p.roughness;
        GGXMicrofacetDistribution ggx(alpha, alpha, 0.0f);

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirL;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        float expectedF_D90 = 0.5f * p.roughness + 2 * p.roughness * query.dirLocal.z * query.dirLocal.z;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = p.baseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, p.roughness);

        SampledSpectrum specularF0Color = lerp(0.08f * specular * SampledSpectrum::One(), p.baseColor, p.metallic);
        float expectedOneMinusDotVH5 = std::pow(1 - dirV.z, 5);
        float iSpecularF0 = specularF0Color.importance(query.wlHint);

        float diffuseWeight = iBaseColor * (1 - p.metallic);
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
            diffuseDirPDF = dirL.z / M_PIf;

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
            diffuseDirPDF = dirL.z / M_PIf;
        }

        float oneMinusDotLH5 = std::pow(1 - dotLH, 5);

#if defined(USE_HEIGHT_CORRELATED_SMITH)
        float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
        SampledSpectrum F = lerp(specularF0Color, SampledSpectrum::One(), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        SampledSpectrum specularValue = F * ((D * G) / microfacetDenom);

        float F_D90 = 0.5f * p.roughness + 2 * p.roughness * dotLH * dotLH;
        float oneMinusDotLN5 = std::pow(1 - dirL.z, 5);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);
        SampledSpectrum diffuseValue = p.baseColor * ((diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, p.roughness) / M_PIf) * (1 - p.metallic));

        SampledSpectrum ret = diffuseValue + specularValue;

        result->dirLocal = entering ? dirL : -dirL;

        // PDF based on the single-sample model MIS.
        result->dirPDF = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        return ret;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum UE4BRDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const UE4BRDF*)params;

        const float specular = 0.5f;
        float alpha = p.roughness * p.roughness;
        GGXMicrofacetDistribution ggx(alpha, alpha, 0.0f);

        if (dirLocal.z * query.dirLocal.z <= 0) {
            return SampledSpectrum::Zero();
        }

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? dirLocal : -dirLocal;

        Normal3D m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);

        float oneMinusDotLH5 = std::pow(1 - dotLH, 5);

        SampledSpectrum specularF0Color = lerp(0.08f * specular * SampledSpectrum::One(), p.baseColor, p.metallic);

        float D = ggx.evaluate(m);
#if defined(USE_HEIGHT_CORRELATED_SMITH)
        float G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
#else
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
#endif
        SampledSpectrum F = lerp(specularF0Color, SampledSpectrum::One(), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        SampledSpectrum specularValue = F * ((D * G) / microfacetDenom);

        float F_D90 = 0.5f * p.roughness + 2 * p.roughness * dotLH * dotLH;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float oneMinusDotLN5 = std::pow(1 - dirL.z, 5);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        SampledSpectrum diffuseValue = p.baseColor * ((diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, p.roughness) / M_PIf) * (1 - p.metallic));

        SampledSpectrum ret = diffuseValue + specularValue;

        return ret;
    }

    RT_CALLABLE_PROGRAM float UE4BRDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const UE4BRDF*)params;

        const float specular = 0.5f;
        float alpha = p.roughness * p.roughness;
        GGXMicrofacetDistribution ggx(alpha, alpha, 0.0f);

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? dirLocal : -dirLocal;

        Normal3D m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);
        float commonPDFTerm = 1.0f / (4 * dotLH);

        float expectedF_D90 = 0.5f * p.roughness + 2 * p.roughness * query.dirLocal.z * query.dirLocal.z;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = p.baseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, p.roughness);

        SampledSpectrum specularF0Color = lerp(0.08f * specular * SampledSpectrum::One(), p.baseColor, p.metallic);
        float expectedOneMinusDotVH5 = std::pow(1 - dirV.z, 5);
        float iSpecularF0 = specularF0Color.importance(query.wlHint);

        float diffuseWeight = iBaseColor * (1 - p.metallic);
        float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

        float sumWeights = diffuseWeight + specularWeight;

        float diffuseDirPDF = dirL.z / M_PIf;
        float specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

        float ret = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        return ret;
    }

    RT_CALLABLE_PROGRAM float UE4BRDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const UE4BRDF*)params;

        const float specular = 0.5f;

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        float expectedF_D90 = 0.5f * p.roughness + 2 * p.roughness * query.dirLocal.z * query.dirLocal.z;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = p.baseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, p.roughness);

        SampledSpectrum specularF0Color = lerp(0.08f * specular * SampledSpectrum::One(), p.baseColor, p.metallic);
        float expectedOneMinusDotVH5 = std::pow(1 - dirV.z, 5);
        float iSpecularF0 = specularF0Color.importance(query.wlHint);

        float diffuseWeight = iBaseColor * (1 - p.metallic);
        float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

        return diffuseWeight + specularWeight;
    }

    // END: UE4 (Modified) BRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // NullEDF

    RT_CALLABLE_PROGRAM uint32_t NullEDF_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        return 0;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum NullEDF_evaluateEmittanceInternal(const uint32_t* params) {
        return SampledSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM SampledSpectrum NullEDF_evaluateInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        return SampledSpectrum::Zero();
    }

    // END: NullEDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // DiffuseEDF

    struct DiffuseEDF {
        SampledSpectrum emittance;
    };

    RT_CALLABLE_PROGRAM uint32_t DiffuseEmitterSurfaceMaterial_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(DiffuseEDF*)params;
        auto &mat = *(const DiffuseEmitterSurfaceMaterial*)matDesc;

        p.emittance = calcNode(mat.nodeEmittance, mat.immEmittance, surfPt, wls);

        return sizeof(DiffuseEDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum DiffuseEDF_evaluateEmittanceInternal(const uint32_t* params) {
        auto &p = *(const DiffuseEDF*)params;
        return p.emittance;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum DiffuseEDF_evaluateInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        return SampledSpectrum(dirLocal.z > 0.0f ? 1.0f / M_PIf : 0.0f);
    }

    // END: DiffuseEDF
    // ----------------------------------------------------------------



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
    struct MultiBSDF {
        struct {
            unsigned int bsdf0 : 6;
            unsigned int bsdf1 : 6;
            unsigned int bsdf2 : 6;
            unsigned int bsdf3 : 6;
            unsigned int numBSDFs : 8;
        };
    };

    RT_CALLABLE_PROGRAM uint32_t MultiSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(MultiBSDF*)params;
        auto &mat = *(const MultiSurfaceMaterial*)matDesc;

        uint32_t baseIndex = sizeof(MultiBSDF) / 4;
        uint32_t bsdfOffsets[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < mat.numSubMaterials; ++i) {
            bsdfOffsets[i] = baseIndex;

            const SurfaceMaterialDescriptor subMatDesc = pv_materialDescriptorBuffer[mat.subMatIndices[i]];
            ProgSigSetupBSDF setupBSDF = (ProgSigSetupBSDF)subMatDesc.progSetupBSDF;
            *(params + baseIndex++) = subMatDesc.bsdfProcedureSetIndex;
            baseIndex += setupBSDF(subMatDesc.data, surfPt, wls, params + baseIndex);
        }

        p.bsdf0 = bsdfOffsets[0];
        p.bsdf1 = bsdfOffsets[1];
        p.bsdf2 = bsdfOffsets[2];
        p.bsdf3 = bsdfOffsets[3];
        p.numBSDFs = mat.numSubMaterials;

        //vlrDevPrintf("%u, %u, %u, %u, %u mats\n", p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3, p.numBSDFs);

        return baseIndex;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MultiBSDF_getBaseColor(const uint32_t* params) {
        auto &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        SampledSpectrum ret;
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFGetBaseColor getBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;

            ret += getBaseColor(bsdf + 1);
        }

        return ret;
    }

    RT_CALLABLE_PROGRAM bool MultiBSDF_matches(const uint32_t* params, DirectionType flags) {
        auto &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFmatches matches = (ProgSigBSDFmatches)procSet.progMatches;

            if (matches(bsdf + 1, flags))
                return true;
        }

        return false;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MultiBSDF_sampleInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        auto &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        float weights[4];
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFWeightInternal weightInternal = (ProgSigBSDFWeightInternal)procSet.progWeightInternal;

            weights[i] = weightInternal(bsdf + 1, query);
        }

        // JP: 各BSDFのウェイトに基づいて方向のサンプルを行うBSDFを選択する。
        // EN: Based on the weight of each BSDF, select a BSDF from which direction sampling.
        float tempProb;
        float sumWeights;
        uint32_t idx = sampleDiscrete(weights, p.numBSDFs, uComponent, &tempProb, &sumWeights, &uComponent);
        if (sumWeights == 0.0f) {
            result->dirPDF = 0.0f;
            return SampledSpectrum::Zero();
        }

        const uint32_t* selectedBSDF = params + bsdfOffsets[idx];
        uint32_t selProcIdx = *(const uint32_t*)selectedBSDF;
        const BSDFProcedureSet selProcSet = pv_bsdfProcedureSetBuffer[selProcIdx];
        ProgSigBSDFSampleInternal sampleInternal = (ProgSigBSDFSampleInternal)selProcSet.progSampleInternal;

        // JP: 選択したBSDFから方向をサンプリングする。
        // EN: sample a direction from the selected BSDF.
        SampledSpectrum value = sampleInternal(selectedBSDF + 1, query, uComponent, uDir, result);
        result->dirPDF *= weights[idx];
        if (result->dirPDF == 0.0f) {
            result->dirPDF = 0.0f;
            return SampledSpectrum::Zero();
        }

        // JP: サンプルした方向に関するBSDFの値の合計と、single-sample model MISに基づいた確率密度を計算する。
        // EN: calculate the total of BSDF values and a PDF based on the single-sample model MIS for the sampled direction.
        if (!result->sampledType.isDelta()) {
            for (int i = 0; i < p.numBSDFs; ++i) {
                const uint32_t* bsdf = params + bsdfOffsets[i];
                uint32_t procIdx = *(const uint32_t*)bsdf;
                const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
                ProgSigBSDFmatches matches = (ProgSigBSDFmatches)procSet.progMatches;
                ProgSigBSDFEvaluatePDFInternal evaluatePDFInternal = (ProgSigBSDFEvaluatePDFInternal)procSet.progEvaluatePDFInternal;

                if (i != idx && matches(bsdf + 1, query.dirTypeFilter))
                    result->dirPDF += evaluatePDFInternal(bsdf + 1, query, result->dirLocal) * weights[i];
            }

            BSDFQuery mQuery = query;
            mQuery.dirTypeFilter &= sideTest(query.geometricNormalLocal, query.dirLocal, result->dirLocal);
            value = SampledSpectrum::Zero();
            for (int i = 0; i < p.numBSDFs; ++i) {
                const uint32_t* bsdf = params + bsdfOffsets[i];
                uint32_t procIdx = *(const uint32_t*)bsdf;
                const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
                ProgSigBSDFmatches matches = (ProgSigBSDFmatches)procSet.progMatches;
                ProgSigBSDFEvaluateInternal evaluateInternal = (ProgSigBSDFEvaluateInternal)procSet.progEvaluateInternal;

                if (!matches(bsdf + 1, mQuery.dirTypeFilter))
                    continue;
                value += evaluateInternal(bsdf + 1, mQuery, result->dirLocal);
            }
        }
        result->dirPDF /= sumWeights;

        return value;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MultiBSDF_evaluateInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        SampledSpectrum retValue = SampledSpectrum::Zero();
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFmatches matches = (ProgSigBSDFmatches)procSet.progMatches;
            ProgSigBSDFEvaluateInternal evaluateInternal = (ProgSigBSDFEvaluateInternal)procSet.progEvaluateInternal;

            if (!matches(bsdf + 1, query.dirTypeFilter))
                continue;
            retValue += evaluateInternal(bsdf + 1, query, dirLocal);
        }
        return retValue;
    }

    RT_CALLABLE_PROGRAM float MultiBSDF_evaluatePDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        float sumWeights = 0.0f;
        float weights[4];
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFWeightInternal weightInternal = (ProgSigBSDFWeightInternal)procSet.progWeightInternal;

            weights[i] = weightInternal(bsdf + 1, query);
            sumWeights += weights[i];
        }
        if (sumWeights == 0.0f)
            return 0.0f;

        float retPDF = 0.0f;
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFEvaluatePDFInternal evaluatePDFInternal = (ProgSigBSDFEvaluatePDFInternal)procSet.progEvaluatePDFInternal;

            if (weights[i] > 0)
                retPDF += evaluatePDFInternal(bsdf + 1, query, dirLocal) * weights[i];
        }
        retPDF /= sumWeights;

        return retPDF;
    }

    RT_CALLABLE_PROGRAM float MultiBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        auto &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        float ret = 0.0f;
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            ProgSigBSDFWeightInternal weightInternal = (ProgSigBSDFWeightInternal)procSet.progWeightInternal;

            ret += weightInternal(bsdf + 1, query);
        }

        return ret;
    }

    // edf0-3: param offsets
    // numEDFs
    // --------------------------------
    // EDF0 procedure set index
    // EDF0 params
    // ...
    // EDF3 procedure set index
    // EDF3 params
    struct MultiEDF {
        struct {
            unsigned int edf0 : 6;
            unsigned int edf1 : 6;
            unsigned int edf2 : 6;
            unsigned int edf3 : 6;
            unsigned int numEDFs : 8;
        };
    };

    RT_CALLABLE_PROGRAM uint32_t MultiSurfaceMaterial_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(MultiEDF*)params;
        auto &mat = *(const MultiSurfaceMaterial*)matDesc;

        uint32_t baseIndex = sizeof(MultiEDF) / 4;
        uint32_t edfOffsets[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < mat.numSubMaterials; ++i) {
            edfOffsets[i] = baseIndex;

            const SurfaceMaterialDescriptor subMatDesc = pv_materialDescriptorBuffer[mat.subMatIndices[i]];
            ProgSigSetupEDF setupEDF = (ProgSigSetupEDF)subMatDesc.progSetupEDF;
            *(params + baseIndex++) = subMatDesc.edfProcedureSetIndex;
            baseIndex += setupEDF(subMatDesc.data, surfPt, wls, params + baseIndex);
        }

        p.edf0 = edfOffsets[0];
        p.edf1 = edfOffsets[1];
        p.edf2 = edfOffsets[2];
        p.edf3 = edfOffsets[3];
        p.numEDFs = mat.numSubMaterials;

        return baseIndex;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MultiEDF_evaluateEmittanceInternal(const uint32_t* params) {
        auto &p = *(const MultiEDF*)params;

        uint32_t edfOffsets[4] = { p.edf0, p.edf1, p.edf2, p.edf3 };

        SampledSpectrum ret = SampledSpectrum::Zero();
        for (int i = 0; i < p.numEDFs; ++i) {
            const uint32_t* edf = params + edfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)edf;
            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[procIdx];
            ProgSigEDFEvaluateEmittanceInternal evaluateEmittanceInternal = (ProgSigEDFEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;

            ret += evaluateEmittanceInternal(edf + 1);
        }

        return ret;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum MultiEDF_evaluateInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        auto &p = *(const MultiEDF*)params;

        uint32_t edfOffsets[4] = { p.edf0, p.edf1, p.edf2, p.edf3 };

        SampledSpectrum ret = SampledSpectrum::Zero();
        SampledSpectrum sumEmittance = SampledSpectrum::Zero();
        for (int i = 0; i < p.numEDFs; ++i) {
            const uint32_t* edf = params + edfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)edf;
            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[procIdx];
            ProgSigEDFEvaluateEmittanceInternal evaluateEmittanceInternal = (ProgSigEDFEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;
            ProgSigEDFEvaluateInternal evaluateInternal = (ProgSigEDFEvaluateInternal)procSet.progEvaluateInternal;

            SampledSpectrum emittance = evaluateEmittanceInternal(edf + 1);
            sumEmittance += emittance;
            ret += emittance * evaluateInternal(edf + 1, query, dirLocal);
        }
        ret.safeDivide(sumEmittance);

        return ret;
    }

    // END: MultiBSDF / MultiEDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // EnvironmentEDF

    struct EnvironmentEDF {
        SampledSpectrum emittance;
    };

    RT_CALLABLE_PROGRAM uint32_t EnvironmentEmitterSurfaceMaterial_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *(EnvironmentEDF*)params;
        auto &mat = *(const EnvironmentEmitterSurfaceMaterial*)matDesc;

        p.emittance = calcNode(mat.nodeEmittance, mat.immEmittance, surfPt, wls) * mat.immScale;

        return sizeof(EnvironmentEDF) / 4;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum EnvironmentEDF_evaluateEmittanceInternal(const uint32_t* params) {
        auto &p = *(const EnvironmentEDF*)params;
        return M_PIf * p.emittance;
    }

    RT_CALLABLE_PROGRAM SampledSpectrum EnvironmentEDF_evaluateInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        return SampledSpectrum(dirLocal.z > 0.0f ? 1.0f / M_PIf : 0.0f);
    }

    // END: EnvironmentEDF
    // ----------------------------------------------------------------
}
