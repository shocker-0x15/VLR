#include "kernel_common.cuh"

namespace VLR {
    RT_FUNCTION DirectionType sideTest(const Normal3D &ng, const Vector3D &d0, const Vector3D &d1) {
        bool reflect = dot(Vector3D(ng), d0) * dot(Vector3D(ng), d1) > 0;
        return DirectionType::AllFreq() | (reflect ? DirectionType::Reflection() : DirectionType::Transmission());
    }



    // ----------------------------------------------------------------
    // NullBSDF

    RT_CALLABLE_PROGRAM uint32_t NullBSDF_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, bool wavelengthSelected, uint32_t* params) {
        return 0;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum NullBSDF_getBaseColor(const uint32_t* params) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM bool NullBSDF_matches(const uint32_t* params, DirectionType flags) {
        return false;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum NullBSDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM RGBSpectrum NullBSDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float NullBSDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
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
        RGBSpectrum albedo;
        float roughness;
    };

    RT_CALLABLE_PROGRAM uint32_t MatteSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, bool wavelengthSelected, uint32_t* params) {
        MatteBRDF &p = *(MatteBRDF*)params;
        const MatteSurfaceMaterial &mat = *(const MatteSurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        optix::float4 texValue = optix::rtTex2D<optix::float4>(mat.texAlbedoRoughness, surfPt.texCoord.u, surfPt.texCoord.v);
        p.albedo = sRGB_degamma(RGBSpectrum(texValue.x, texValue.y, texValue.z));
        p.roughness = texValue.w;

        return sizeof(MatteBRDF) / 4;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MatteBRDF_getBaseColor(const uint32_t* params) {
        MatteBRDF &p = *(MatteBRDF*)params;

        return p.albedo;
    }

    RT_CALLABLE_PROGRAM bool MatteBRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MatteBRDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        MatteBRDF &p = *(MatteBRDF*)params;

        result->dirLocal = cosineSampleHemisphere(uDir[0], uDir[1]);
        result->dirPDF = result->dirLocal.z / M_PIf;
        result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();
        result->dirLocal.z *= query.dirLocal.z > 0 ? 1 : -1;

        return p.albedo / M_PIf;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MatteBRDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        MatteBRDF &p = *(MatteBRDF*)params;

        if (query.dirLocal.z * dirLocal.z <= 0.0f) {
            RGBSpectrum fs = RGBSpectrum::Zero();
            return fs;
        }
        RGBSpectrum fs = p.albedo / M_PIf;

        return fs;
    }

    RT_CALLABLE_PROGRAM float MatteBRDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        if (query.dirLocal.z * dirLocal.z <= 0.0f) {
            return 0.0f;
        }

        return std::abs(dirLocal.z) / M_PIf;
    }

    RT_CALLABLE_PROGRAM float MatteBRDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        MatteBRDF &p = *(MatteBRDF*)params;
        return p.albedo.importance(query.wlHint);
    }

    // END: MatteBRDF
    // ----------------------------------------------------------------



    RT_FUNCTION RGBSpectrum evaluateFresnelConductor(const RGBSpectrum &eta, const RGBSpectrum &k, float cosEnter) {
        cosEnter = std::fabs(cosEnter);
        float cosEnter2 = cosEnter * cosEnter;
        RGBSpectrum _2EtaCosEnter = 2.0f * eta * cosEnter;
        RGBSpectrum tmp_f = eta * eta + k * k;
        RGBSpectrum tmp = tmp_f * cosEnter2;
        RGBSpectrum Rparl2 = (tmp - _2EtaCosEnter + 1) / (tmp + _2EtaCosEnter + 1);
        RGBSpectrum Rperp2 = (tmp_f - _2EtaCosEnter + cosEnter2) / (tmp_f + _2EtaCosEnter + cosEnter2);
        return (Rparl2 + Rperp2) / 2.0f;
    }

    RT_FUNCTION float evalF(float etaEnter, float etaExit, float cosEnter, float cosExit) {
        float Rparl = ((etaExit * cosEnter) - (etaEnter * cosExit)) / ((etaExit * cosEnter) + (etaEnter * cosExit));
        float Rperp = ((etaEnter * cosEnter) - (etaExit * cosExit)) / ((etaEnter * cosEnter) + (etaExit * cosExit));
        return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
    }

    RT_FUNCTION RGBSpectrum evaluateFresnelDielectric(const RGBSpectrum &etaExt, const RGBSpectrum &etaInt, float cosEnter) {
        cosEnter = std::fmin(std::fmax(cosEnter, -1.0f), 1.0f);

        bool entering = cosEnter > 0.0f;
        const RGBSpectrum &eEnter = entering ? etaExt : etaInt;
        const RGBSpectrum &eExit = entering ? etaInt : etaExt;

        RGBSpectrum sinExit = eEnter / eExit * std::sqrt(std::fmax(0.0f, 1.0f - cosEnter * cosEnter));
        RGBSpectrum ret = RGBSpectrum::Zero();
        cosEnter = std::fabs(cosEnter);
        for (int i = 0; i < RGBSpectrum::NumComponents(); ++i) {
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



    // ----------------------------------------------------------------
    // SpecularBRDF

    struct SpecularBRDF {
        RGBSpectrum coeffR;
        RGBSpectrum eta;
        RGBSpectrum k;
    };

    RT_CALLABLE_PROGRAM uint32_t SpecularReflectionSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, bool wavelengthSelected, uint32_t* params) {
        SpecularBRDF &p = *(SpecularBRDF*)params;
        const SpecularReflectionSurfaceMaterial &mat = *(const SpecularReflectionSurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        optix::float4 texValue;
        texValue = optix::rtTex2D<optix::float4>(mat.texCoeffR, surfPt.texCoord.u, surfPt.texCoord.v);
        p.coeffR = sRGB_degamma(RGBSpectrum(texValue.x, texValue.y, texValue.z));
        texValue = optix::rtTex2D<optix::float4>(mat.texEta, surfPt.texCoord.u, surfPt.texCoord.v);
        p.eta = RGBSpectrum(texValue.x, texValue.y, texValue.z);
        texValue = optix::rtTex2D<optix::float4>(mat.tex_k, surfPt.texCoord.u, surfPt.texCoord.v);
        p.k = RGBSpectrum(texValue.x, texValue.y, texValue.z);

        return sizeof(SpecularBRDF) / 4;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum SpecularBRDF_getBaseColor(const uint32_t* params) {
        SpecularBRDF &p = *(SpecularBRDF*)params;

        return p.coeffR;
    }

    RT_CALLABLE_PROGRAM bool SpecularBRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::Delta0D();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM RGBSpectrum SpecularBRDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        SpecularBRDF &p = *(SpecularBRDF*)params;

        result->dirLocal = Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z);
        result->dirPDF = 1.0f;
        result->sampledType = DirectionType::Reflection() | DirectionType::Delta0D();
        RGBSpectrum fs = p.coeffR * evaluateFresnelConductor(p.eta, p.k, query.dirLocal.z) / std::fabs(query.dirLocal.z);

        return fs;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum SpecularBRDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float SpecularBRDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM float SpecularBRDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        SpecularBRDF &p = *(SpecularBRDF*)params;
        float ret = (p.coeffR * evaluateFresnelConductor(p.eta, p.k, query.dirLocal.z)).importance(query.wlHint);
        //float snCorrection = query.adjoint ? std::fabs(dot(Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z), query.gNormalLocal) /
        //                                               query.dirLocal.z) : 1;
        return ret;
    }

    // END: SpecularBRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // SpecularBSDF

    struct SpecularBSDF {
        RGBSpectrum coeff;
        RGBSpectrum etaExt;
        RGBSpectrum etaInt;
        bool dispersive;
    };

    RT_CALLABLE_PROGRAM uint32_t SpecularScatteringSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, bool wavelengthSelected, uint32_t* params) {
        SpecularBSDF &p = *(SpecularBSDF*)params;
        const SpecularScatteringSurfaceMaterial &mat = *(const SpecularScatteringSurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        optix::float4 texValue;
        texValue = optix::rtTex2D<optix::float4>(mat.texCoeff, surfPt.texCoord.u, surfPt.texCoord.v);
        p.coeff = sRGB_degamma(RGBSpectrum(texValue.x, texValue.y, texValue.z));
        texValue = optix::rtTex2D<optix::float4>(mat.texEtaExt, surfPt.texCoord.u, surfPt.texCoord.v);
        p.etaExt = RGBSpectrum(texValue.x, texValue.y, texValue.z);
        texValue = optix::rtTex2D<optix::float4>(mat.texEtaInt, surfPt.texCoord.u, surfPt.texCoord.v);
        p.etaInt = RGBSpectrum(texValue.x, texValue.y, texValue.z);
        p.dispersive = !wavelengthSelected;

        return sizeof(SpecularBSDF) / 4;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum SpecularBSDF_getBaseColor(const uint32_t* params) {
        SpecularBSDF &p = *(SpecularBSDF*)params;

        return p.coeff;
    }

    RT_CALLABLE_PROGRAM bool SpecularBSDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::WholeSphere() | DirectionType::Delta0D();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM RGBSpectrum SpecularBSDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        SpecularBSDF &p = *(SpecularBSDF*)params;

        RGBSpectrum F = evaluateFresnelDielectric(p.etaExt, p.etaInt, query.dirLocal.z);
        float reflectProb = F.importance(query.wlHint);
        if (query.dirTypeFilter.isReflection())
            reflectProb = 1.0f;
        if (query.dirTypeFilter.isTransmission())
            reflectProb = 0.0f;
        if (uComponent < reflectProb) {
            if (query.dirLocal.z == 0.0f) {
                result->dirPDF = 0.0f;
                return RGBSpectrum::Zero();
            }
            result->dirLocal = Vector3D(-query.dirLocal.x, -query.dirLocal.y, query.dirLocal.z);
            result->dirPDF = reflectProb;
            result->sampledType = DirectionType::Reflection() | DirectionType::Delta0D();
            RGBSpectrum fs = p.coeff * F / std::fabs(query.dirLocal.z);

            return fs;
        }
        else {
            bool entering = query.dirLocal.z > 0.0f;
            float eEnter = entering ? p.etaExt[query.wlHint] : p.etaInt[query.wlHint];
            float eExit = entering ? p.etaInt[query.wlHint] : p.etaExt[query.wlHint];

            float sinEnter2 = 1.0f - query.dirLocal.z * query.dirLocal.z;
            float rrEta = eEnter / eExit;// reciprocal of relative IOR.
            float sinExit2 = rrEta * rrEta * sinEnter2;

            if (sinExit2 >= 1.0f) {
                result->dirPDF = 0.0f;
                return RGBSpectrum::Zero();
            }
            float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit2));
            if (entering)
                cosExit = -cosExit;
            result->dirLocal = Vector3D(rrEta * -query.dirLocal.x, rrEta * -query.dirLocal.y, cosExit);
            result->dirPDF = 1.0f - reflectProb;
            result->sampledType = DirectionType::Transmission() | DirectionType::Delta0D() | (p.dispersive ? DirectionType::Dispersive() : DirectionType());
            RGBSpectrum ret = RGBSpectrum::Zero();
            ret[query.wlHint] = p.coeff[query.wlHint] * (1.0f - F[query.wlHint]);
            RGBSpectrum fs = ret / std::fabs(cosExit);

            return fs;
        }
    }

    RT_CALLABLE_PROGRAM RGBSpectrum SpecularBSDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float SpecularBSDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM float SpecularBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        SpecularBSDF &p = *(SpecularBSDF*)params;
        return p.coeff.importance(query.wlHint);
    }

    // END: SpecularBSDF
    // ----------------------------------------------------------------



    class GGXMicrofacetDistribution {
        float m_alpha_gx;
        float m_alpha_gy;

    public:
        RT_FUNCTION GGXMicrofacetDistribution(float alpha_gx, float alpha_gy) :
            m_alpha_gx(alpha_gx), m_alpha_gy(alpha_gy) {}

        RT_FUNCTION float evaluate(const Normal3D &m) {
            if (m.z <= 0)
                return 0.0f;
            float temp = m.x * m.x / (m_alpha_gx * m_alpha_gx) + m.y * m.y / (m_alpha_gy * m_alpha_gy) + m.z * m.z;
            return 1.0f / (M_PIf * m_alpha_gx * m_alpha_gy * temp * temp);
        }

        RT_FUNCTION float evaluateSmithG1(const Vector3D &v, const Normal3D &m) {
            float chi = (dot(v, m) / v.z) > 0 ? 1 : 0;
            float tanTheta_v_alpha_go_2 = (v.x * v.x * m_alpha_gx * m_alpha_gx + v.y * v.y * m_alpha_gy * m_alpha_gy) / (v.z * v.z);
            return chi * 2 / (1 + std::sqrt(1 + tanTheta_v_alpha_go_2));
        }

        RT_FUNCTION float sample(const Vector3D &v, float u0, float u1, Normal3D* m, float* normalPDF) {
            // stretch view
            Vector3D sv = normalize(Vector3D(m_alpha_gx * v.x, m_alpha_gy * v.y, v.z));

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
            float P1 = r * std::cos(phi);
            float P2 = r * std::sin(phi) * ((u1 < a) ? 1.0 : sv.z);

            // compute normal
            *m = P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv;

            // unstretch
            *m = normalize(Normal3D(m_alpha_gx * m->x, m_alpha_gy * m->y, m->z));

            float D = evaluate(*m);
            *normalPDF = evaluateSmithG1(v, *m) * absDot(v, *m) * D / std::abs(v.z);

            return D;
        }

        RT_FUNCTION float evaluatePDF(const Vector3D &v, const Normal3D &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::abs(v.z);
        }
    };



    // ----------------------------------------------------------------
    // UE4 BRDF

    struct UE4BRDF {
        RGBSpectrum baseColor;
        float roughness;
        float metallic;
    };

    RT_CALLABLE_PROGRAM uint32_t UE4SurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, bool wavelengthSelected, uint32_t* params) {
        UE4BRDF &p = *(UE4BRDF*)params;
        const UE4SurfaceMaterial &mat = *(const UE4SurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        optix::float4 texValue;
        texValue = optix::rtTex2D<optix::float4>(mat.texBaseColor, surfPt.texCoord.u, surfPt.texCoord.v);
        p.baseColor = sRGB_degamma(RGBSpectrum(texValue.x, texValue.y, texValue.z));
        texValue = optix::rtTex2D<optix::float4>(mat.texOcclusionRoughnessMetallic, surfPt.texCoord.u, surfPt.texCoord.v);
        p.roughness = texValue.y;
        p.metallic = texValue.z;

        return sizeof(UE4BRDF) / 4;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_getBaseColor(const uint32_t* params) {
        UE4BRDF &p = *(UE4BRDF*)params;

        return p.baseColor;
    }
    
    RT_CALLABLE_PROGRAM bool UE4BRDF_matches(const uint32_t* params, DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq() | DirectionType::HighFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        UE4BRDF &p = *(UE4BRDF*)params;

        const float specular = 0.5f;
        float alpha = p.roughness * p.roughness;
        GGXMicrofacetDistribution ggx(alpha, alpha);

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirL;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        float expectedF_D90 = 0.5f * p.roughness + 2 * p.roughness * query.dirLocal.z * query.dirLocal.z;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = p.baseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, p.roughness);

        RGBSpectrum specularF0Color = lerp(0.08f * specular * RGBSpectrum::One(), p.baseColor, p.metallic);
        float expectedOneMinusDotVH5 = std::pow(1 - dirV.z, 5);
        float iSpecularF0 = specularF0Color.importance(query.wlHint);

        float diffuseWeight = iBaseColor * (1 - p.metallic);
        float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

        float weights[] = { diffuseWeight, specularWeight };
        float probSelection;
        float sumWeights = 0.0f;
        uint32_t component = sampleDiscrete(weights, 2, uComponent, &probSelection, &sumWeights, &uComponent);

        float diffuseDirPDF, specularDirPDF;
        RGBSpectrum fs;
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
                return RGBSpectrum::Zero();
            }
            float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDF = commonPDFTerm * mPDF;
            // ----------------------------------------------------------------

            // JP: 同じ方向サンプルを別の要素からサンプルする確率密度を求める。
            // EN: calculate PDFs to generate the sampled direction from the other distributions.
            diffuseDirPDF = dirL.z / M_PIf;
        }

        float oneMinusDotLH5 = std::pow(1 - dotLH, 5);

        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
        RGBSpectrum F = lerp(specularF0Color, RGBSpectrum::One(), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        RGBSpectrum specularValue = F * ((D * G) / microfacetDenom);

        float F_D90 = 0.5f * p.roughness + 2 * p.roughness * dotLH * dotLH;
        float oneMinusDotLN5 = std::pow(1 - dirL.z, 5);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);
        RGBSpectrum diffuseValue = p.baseColor * ((diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, p.roughness) / M_PIf) * (1 - p.metallic));

        RGBSpectrum ret = diffuseValue + specularValue;

        result->dirLocal = entering ? dirL : -dirL;

        // PDF based on the single-sample model MIS.
        result->dirPDF = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        return ret;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        UE4BRDF &p = *(UE4BRDF*)params;

        const float specular = 0.5f;
        float alpha = p.roughness * p.roughness;
        GGXMicrofacetDistribution ggx(alpha, alpha);

        if (dirLocal.z * query.dirLocal.z <= 0) {
            return RGBSpectrum::Zero();
        }

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirL = entering ? dirLocal : -dirLocal;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        Normal3D m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);

        float oneMinusDotLH5 = std::pow(1 - dotLH, 5);

        RGBSpectrum specularF0Color = lerp(0.08f * specular * RGBSpectrum::One(), p.baseColor, p.metallic);

        float D = ggx.evaluate(m);
        float G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
        RGBSpectrum F = lerp(specularF0Color, RGBSpectrum::One(), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        RGBSpectrum specularValue = F * ((D * G) / microfacetDenom);

        float F_D90 = 0.5f * p.roughness + 2 * p.roughness * dotLH * dotLH;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float oneMinusDotLN5 = std::pow(1 - dirL.z, 5);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        RGBSpectrum diffuseValue = p.baseColor * ((diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, p.roughness) / M_PIf) * (1 - p.metallic));

        RGBSpectrum ret = diffuseValue + specularValue;

        return ret;
    }

    RT_CALLABLE_PROGRAM float UE4BRDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        UE4BRDF &p = *(UE4BRDF*)params;

        const float specular = 0.5f;
        float alpha = p.roughness * p.roughness;
        GGXMicrofacetDistribution ggx(alpha, alpha);

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirL = entering ? dirLocal : -dirLocal;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        Normal3D m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);
        float commonPDFTerm = 1.0f / (4 * dotLH);

        float expectedF_D90 = 0.5f * p.roughness + 2 * p.roughness * query.dirLocal.z * query.dirLocal.z;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = p.baseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, p.roughness);

        RGBSpectrum specularF0Color = lerp(0.08f * specular * RGBSpectrum::One(), p.baseColor, p.metallic);
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
        UE4BRDF &p = *(UE4BRDF*)params;

        const float specular = 0.5f;

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        float expectedF_D90 = 0.5f * p.roughness + 2 * p.roughness * query.dirLocal.z * query.dirLocal.z;
        float oneMinusDotVN5 = std::pow(1 - dirV.z, 5);
        float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
        float iBaseColor = p.baseColor.importance(query.wlHint) * expectedDiffuseFresnel * expectedDiffuseFresnel * lerp(1.0f, 1.0f / 1.51f, p.roughness);

        RGBSpectrum specularF0Color = lerp(0.08f * specular * RGBSpectrum::One(), p.baseColor, p.metallic);
        float expectedOneMinusDotVH5 = std::pow(1 - dirV.z, 5);
        float iSpecularF0 = specularF0Color.importance(query.wlHint);

        float diffuseWeight = iBaseColor * (1 - p.metallic);
        float specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);

        return diffuseWeight + specularWeight;
    }

    // END: UE4 BRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // NullEDF

    RT_CALLABLE_PROGRAM uint32_t NullEDF_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        return 0;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum NullEDF_evaluateEmittanceInternal(const uint32_t* params) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM RGBSpectrum NullEDF_evaluateEDFInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        return RGBSpectrum::Zero();
    }

    // END: NullEDF
    // ----------------------------------------------------------------


    
    // ----------------------------------------------------------------
    // DiffuseEDF

    struct DiffuseEDF {
        RGBSpectrum emittance;
    };

    RT_CALLABLE_PROGRAM uint32_t DiffuseEmitterSurfaceMaterial_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        DiffuseEDF &p = *(DiffuseEDF*)params;
        const DiffuseEmitterSurfaceMaterial &mat = *(const DiffuseEmitterSurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        optix::float4 texValue = optix::rtTex2D<optix::float4>(mat.texEmittance, surfPt.texCoord.u, surfPt.texCoord.v);
        p.emittance = RGBSpectrum(texValue.x, texValue.y, texValue.z);

        return sizeof(DiffuseEDF) / 4;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum DiffuseEDF_evaluateEmittanceInternal(const uint32_t* params) {
        DiffuseEDF &p = *(DiffuseEDF*)params;
        return p.emittance;
    }
    
    RT_CALLABLE_PROGRAM RGBSpectrum DiffuseEDF_evaluateEDFInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        return RGBSpectrum(dirLocal.z > 0.0f ? 1.0f / M_PIf : 0.0f);
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

    RT_CALLABLE_PROGRAM uint32_t MultiSurfaceMaterial_setupBSDF(const uint32_t* matDesc, const SurfacePoint &surfPt, bool wavelengthSelected, uint32_t* params) {
        MultiBSDF &p = *(MultiBSDF*)params;
        const MultiSurfaceMaterial &mat = *(const MultiSurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        uint32_t baseIndex = sizeof(MultiBSDF) / 4;
        const uint32_t matOffsets[] = { mat.matOffset0, mat.matOffset1, mat.matOffset2, mat.matOffset3 };
        uint32_t bsdfOffsets[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < mat.numMaterials; ++i) {
            bsdfOffsets[i] = baseIndex;

            const SurfaceMaterialHead &matHead = *(const SurfaceMaterialHead*)(matDesc + matOffsets[i]);
            //rtPrintf("%d: %u, %u, %u, %u\n", i, matHead.progSetupBSDF, matHead.bsdfProcedureSetIndex, matHead.progSetupEDF, matHead.edfProcedureSetIndex);
            progSigSetupBSDF setupBSDF = (progSigSetupBSDF)matHead.progSetupBSDF;
            *(uint32_t*)(params + baseIndex++) = matHead.bsdfProcedureSetIndex;
            baseIndex += setupBSDF((const uint32_t*)&matHead, surfPt, wavelengthSelected, params + baseIndex);
        }

        p.bsdf0 = bsdfOffsets[0];
        p.bsdf1 = bsdfOffsets[1];
        p.bsdf2 = bsdfOffsets[2];
        p.bsdf3 = bsdfOffsets[3];
        p.numBSDFs = mat.numMaterials;

        //rtPrintf("%u, %u, %u, %u, %u mats\n", p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3, p.numBSDFs);

        return baseIndex;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MultiBSDF_getBaseColor(const uint32_t* params) {
        const MultiBSDF &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        RGBSpectrum ret;
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            progSigGetBaseColor getBaseColor = (progSigGetBaseColor)procSet.progGetBaseColor;

            ret += getBaseColor(bsdf + 1);
        }

        return ret;
    }

    RT_CALLABLE_PROGRAM bool MultiBSDF_matches(const uint32_t* params, DirectionType flags) {
        const MultiBSDF &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            progSigBSDFmatches matches = (progSigBSDFmatches)procSet.progBSDFmatches;

            if (matches(bsdf + 1, flags))
                return true;
        }

        return false;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MultiBSDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        const MultiBSDF &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        float weights[4];
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            progSigBSDFWeightInternal weightInternal = (progSigBSDFWeightInternal)procSet.progWeightInternal;

            weights[i] = weightInternal(bsdf + 1, query);
        }

        // JP: 各BSDFのウェイトに基づいて方向のサンプルを行うBSDFを選択する。
        // EN: Based on the weight of each BSDF, select a BSDF from which direction sampling.
        float tempProb;
        float sumWeights;
        uint32_t idx = sampleDiscrete(weights, p.numBSDFs, uComponent, &tempProb, &sumWeights, &uComponent);
        if (sumWeights == 0.0f) {
            result->dirPDF = 0.0f;
            return RGBSpectrum::Zero();
        }

        const uint32_t* selectedBSDF = params + bsdfOffsets[idx];
        uint32_t selProcIdx = *(const uint32_t*)selectedBSDF;
        const BSDFProcedureSet selProcSet = pv_bsdfProcedureSetBuffer[selProcIdx];
        progSigSampleBSDFInternal sampleInternal = (progSigSampleBSDFInternal)selProcSet.progSampleBSDFInternal;

        // JP: 選択したBSDFから方向をサンプリングする。
        // EN: sample a direction from the selected BSDF.
        RGBSpectrum value = sampleInternal(selectedBSDF + 1, query, uComponent, uDir, result);
        result->dirPDF *= weights[idx];
        if (result->dirPDF == 0.0f) {
            result->dirPDF = 0.0f;
            return RGBSpectrum::Zero();
        }

        // JP: サンプルした方向に関するBSDFの値の合計と、single-sample model MISに基づいた確率密度を計算する。
        // EN: calculate the total of BSDF values and a PDF based on the single-sample model MIS for the sampled direction.
        if (!result->sampledType.isDelta()) {
            for (int i = 0; i < p.numBSDFs; ++i) {
                const uint32_t* bsdf = params + bsdfOffsets[i];
                uint32_t procIdx = *(const uint32_t*)bsdf;
                const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
                progSigBSDFmatches matches = (progSigBSDFmatches)procSet.progBSDFmatches;
                progSigEvaluateBSDF_PDFInternal evaluatePDFInternal = (progSigEvaluateBSDF_PDFInternal)procSet.progEvaluateBSDF_PDFInternal;

                if (i != idx && matches(bsdf + 1, query.dirTypeFilter))
                    result->dirPDF += evaluatePDFInternal(bsdf + 1, query, result->dirLocal) * weights[i];
            }

            BSDFQuery mQuery = query;
            mQuery.dirTypeFilter &= sideTest(query.geometricNormalLocal, query.dirLocal, result->dirLocal);
            value = RGBSpectrum::Zero();
            for (int i = 0; i < p.numBSDFs; ++i) {
                const uint32_t* bsdf = params + bsdfOffsets[i];
                uint32_t procIdx = *(const uint32_t*)bsdf;
                const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
                progSigBSDFmatches matches = (progSigBSDFmatches)procSet.progBSDFmatches;
                progSigEvaluateBSDFInternal evaluateBSDFInternal = (progSigEvaluateBSDFInternal)procSet.progEvaluateBSDFInternal;

                if (!matches(bsdf + 1, mQuery.dirTypeFilter))
                    continue;
                value += evaluateBSDFInternal(bsdf + 1, mQuery, result->dirLocal);
            }
        }
        result->dirPDF /= sumWeights;

        return value;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MultiBSDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        const MultiBSDF &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        RGBSpectrum retValue = RGBSpectrum::Zero();
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            progSigBSDFmatches matches = (progSigBSDFmatches)procSet.progBSDFmatches;
            progSigEvaluateBSDFInternal evaluateBSDFInternal = (progSigEvaluateBSDFInternal)procSet.progEvaluateBSDFInternal;

            if (!matches(bsdf + 1, query.dirTypeFilter))
                continue;
            retValue += evaluateBSDFInternal(bsdf + 1, query, dirLocal);
        }
        return retValue;
    }

    RT_CALLABLE_PROGRAM float MultiBSDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        const MultiBSDF &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        float sumWeights = 0.0f;
        float weights[4];
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            progSigBSDFWeightInternal weightInternal = (progSigBSDFWeightInternal)procSet.progWeightInternal;

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
            progSigEvaluateBSDF_PDFInternal evaluatePDFInternal = (progSigEvaluateBSDF_PDFInternal)procSet.progEvaluateBSDF_PDFInternal;

            if (weights[i] > 0)
                retPDF += evaluatePDFInternal(bsdf + 1, query, dirLocal) * weights[i];
        }
        retPDF /= sumWeights;

        return retPDF;
    }

    RT_CALLABLE_PROGRAM float MultiBSDF_weightInternal(const uint32_t* params, const BSDFQuery &query) {
        const MultiBSDF &p = *(const MultiBSDF*)params;

        uint32_t bsdfOffsets[4] = { p.bsdf0, p.bsdf1, p.bsdf2, p.bsdf3 };

        float ret = 0.0f;
        for (int i = 0; i < p.numBSDFs; ++i) {
            const uint32_t* bsdf = params + bsdfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)bsdf;
            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[procIdx];
            progSigBSDFWeightInternal weightInternal = (progSigBSDFWeightInternal)procSet.progWeightInternal;

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

    RT_CALLABLE_PROGRAM uint32_t MultiSurfaceMaterial_setupEDF(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        MultiEDF &p = *(MultiEDF*)params;
        const MultiSurfaceMaterial &mat = *(const MultiSurfaceMaterial*)(matDesc + sizeof(SurfaceMaterialHead) / 4);

        uint32_t baseIndex = sizeof(MultiEDF) / 4;
        const uint32_t matOffsets[] = { mat.matOffset0, mat.matOffset1, mat.matOffset2, mat.matOffset3 };
        uint32_t edfOffsets[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < mat.numMaterials; ++i) {
            edfOffsets[i] = baseIndex;

            const SurfaceMaterialHead &matHead = *(const SurfaceMaterialHead*)(matDesc + matOffsets[i]);
            progSigSetupEDF setupEDF = (progSigSetupEDF)matHead.progSetupEDF;
            *(uint32_t*)(params + baseIndex++) = matHead.edfProcedureSetIndex;
            baseIndex += setupEDF((const uint32_t*)&matHead, surfPt, params + baseIndex);
        }

        p.edf0 = edfOffsets[0];
        p.edf1 = edfOffsets[1];
        p.edf2 = edfOffsets[2];
        p.edf3 = edfOffsets[3];
        p.numEDFs = mat.numMaterials;

        return baseIndex;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MultiEDF_evaluateEmittanceInternal(const uint32_t* params) {
        const MultiEDF &p = *(const MultiEDF*)params;

        uint32_t edfOffsets[4] = { p.edf0, p.edf1, p.edf2, p.edf3 };

        RGBSpectrum ret = RGBSpectrum::Zero();
        for (int i = 0; i < p.numEDFs; ++i) {
            const uint32_t* edf = params + edfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)edf;
            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[procIdx];
            progSigEvaluateEmittanceInternal evaluateEmittanceInternal = (progSigEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;

            ret += evaluateEmittanceInternal(edf + 1);
        }

        return ret;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MultiEDF_evaluateEDFInternal(const uint32_t* params, const EDFQuery &query, const Vector3D &dirLocal) {
        const MultiEDF &p = *(const MultiEDF*)params;

        uint32_t edfOffsets[4] = { p.edf0, p.edf1, p.edf2, p.edf3 };

        RGBSpectrum ret = RGBSpectrum::Zero();
        RGBSpectrum sumEmittance = RGBSpectrum::Zero();
        for (int i = 0; i < p.numEDFs; ++i) {
            const uint32_t* edf = params + edfOffsets[i];
            uint32_t procIdx = *(const uint32_t*)edf;
            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[procIdx];
            progSigEvaluateEmittanceInternal evaluateEmittanceInternal = (progSigEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;
            progSigEvaluateEDFInternal evaluateEDFInternal = (progSigEvaluateEDFInternal)procSet.progEvaluateEDFInternal;

            RGBSpectrum emittance = evaluateEmittanceInternal(edf + 1);
            sumEmittance += emittance;
            ret += emittance * evaluateEDFInternal(edf + 1, query, dirLocal);
        }
        ret.safeDivide(sumEmittance);

        return ret;
    }

    // END: MultiBSDF / MultiEDF
    // ----------------------------------------------------------------
}
