#include "kernel_common.cuh"
#include "random_distributions.cuh"

namespace VLR {
    // ----------------------------------------------------------------
    // NormalAlphaModifier

    // per GeometryInstance
    rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> pv_texNormalAlpha;

    RT_CALLABLE_PROGRAM float NormalAlphaModifier_fetchAlpha(const TexCoord2D &texCoord) {
        float alpha = tex2D(pv_texNormalAlpha, texCoord.u, texCoord.v).w;
        return alpha;
    }

    RT_CALLABLE_PROGRAM Normal3D NormalAlphaModifier_fetchNormal(const TexCoord2D &texCoord) {
        float4 texValue = tex2D(pv_texNormalAlpha, texCoord.u, texCoord.v);
        Normal3D normalLocal = 2 * Normal3D(texValue.x, texValue.y, texValue.z) - 1.0f;
        return normalLocal;
    }

    // END: NormalAlphaModifier
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // LambertianBRDF

    // per GeometryInstance
    rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> texAlbedoRoughness;

    RT_CALLABLE_PROGRAM bool LambertianBRDF_matches(DirectionType flags) {
        DirectionType m_type = DirectionType::Reflection() | DirectionType::LowFreq();
        return m_type.matches(flags);
    }

    RT_CALLABLE_PROGRAM RGBSpectrum LambertianBRDF_sampleBSDFInternal(const TexCoord2D &texCoord, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        // setup BSDF parameters.
        float4 texValue = tex2D(texAlbedoRoughness, texCoord.u, texCoord.v);
        RGBSpectrum albedo(texValue.x, texValue.y, texValue.z);
        //float roughness = texValue.w;

        result->dirLocal = cosineSampleHemisphere(uDir[0], uDir[1]);
        result->dirPDF = result->dirLocal.z / M_PIf;
        result->sampledType = DirectionType::Reflection() | DirectionType::LowFreq();
        result->dirLocal.z *= query.dirLocal.z > 0 ? 1 : -1;

        return albedo / M_PIf;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum LambertianBRDF_evaluateBSDFInternal(const TexCoord2D &texCoord, const BSDFQuery &query, const Vector3D &dirLocal) {
        // setup BSDF parameters.
        float4 texValue = tex2D(texAlbedoRoughness, texCoord.u, texCoord.v);
        RGBSpectrum albedo(texValue.x, texValue.y, texValue.z);
        //float roughness = texValue.w;

        if (query.dirLocal.z * dirLocal.z <= 0.0f) {
            RGBSpectrum fs = RGBSpectrum::Zero();
            return fs;
        }
        RGBSpectrum fs = albedo / M_PIf;

        return fs;
    }

    RT_CALLABLE_PROGRAM float LambertianBRDF_evaluateBSDF_PDFInternal(const TexCoord2D &texCoord, const BSDFQuery &query, const Vector3D &dirLocal) {
        if (query.dirLocal.z * dirLocal.z <= 0.0f) {
            return 0.0f;
        }

        return std::abs(dirLocal.z) / M_PIf;
    }

    // END: LambertianBRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // DiffuseEDF
    
    RT_CALLABLE_PROGRAM RGBSpectrum DiffuseEDF_evaluateEDFInternal(const TexCoord2D &texCoord, const EDFQuery &query, const Vector3D &dirLocal) {
        //if (!matches(query.flags))
        //    return RGBSpectrum::Zero();
        return RGBSpectrum(dirLocal.z > 0.0f ? 1.0f / M_PIf : 0.0f);
    }

    // END: DiffuseEDF
    // ----------------------------------------------------------------
}
