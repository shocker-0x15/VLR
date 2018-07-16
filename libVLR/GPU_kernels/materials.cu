#include "kernel_common.cuh"
#include "random_distributions.cuh"

namespace VLR {
    // ----------------------------------------------------------------
    // LambertianBRDF

    // per Material
    rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> texAlbedoRoughness;

    RT_CALLABLE_PROGRAM RGBSpectrum LambertianBRDF_getBaseColor(const TexCoord2D &texCoord) {
        // setup BSDF parameters.
        float4 texValue = tex2D(texAlbedoRoughness, texCoord.u, texCoord.v);
        RGBSpectrum albedo(texValue.x, texValue.y, texValue.z);
        //float roughness = texValue.w;

        return albedo;
    }

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
    // UE4 BRDF

    // per Material
    rtTextureSampler<uchar3, 2, cudaReadModeNormalizedFloat> texBaseColor;
    rtTextureSampler<uchar2, 2, cudaReadModeNormalizedFloat> texRoughnessMetallic;

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_getBaseColor(const TexCoord2D &texCoord) {
        VLRAssert_NotImplemented();
        return RGBSpectrum::Zero();
    }
    
    RT_CALLABLE_PROGRAM bool UE4BRDF_matches(DirectionType flags) {
        VLRAssert_NotImplemented();
        return true;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_sampleBSDFInternal(const TexCoord2D &texCoord, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        VLRAssert_NotImplemented();
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_evaluateBSDFInternal(const TexCoord2D &texCoord, const BSDFQuery &query, const Vector3D &dirLocal) {
        VLRAssert_NotImplemented();
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float UE4BRDF_evaluateBSDF_PDFInternal(const TexCoord2D &texCoord, const BSDFQuery &query, const Vector3D &dirLocal) {
        VLRAssert_NotImplemented();
        return 0.0f;
    }

    // END: UE4 BRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // NullEDF

    RT_CALLABLE_PROGRAM RGBSpectrum NullEDF_evaluateEmittance(const TexCoord2D &texCoord) {
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM RGBSpectrum NullEDF_evaluateEDFInternal(const TexCoord2D &texCoord, const EDFQuery &query, const Vector3D &dirLocal) {
        return RGBSpectrum::Zero();
    }

    // END: NullEDF
    // ----------------------------------------------------------------
    
    // ----------------------------------------------------------------
    // DiffuseEDF

    // per Material
    rtTextureSampler<float4, 2, cudaReadModeElementType> texEmittance;

    RT_CALLABLE_PROGRAM RGBSpectrum DiffuseEDF_evaluateEmittance(const TexCoord2D &texCoord) {
        float4 texValue = tex2D(texEmittance, texCoord.u, texCoord.v);
        return RGBSpectrum(texValue.x, texValue.y, texValue.z);
    }
    
    RT_CALLABLE_PROGRAM RGBSpectrum DiffuseEDF_evaluateEDFInternal(const TexCoord2D &texCoord, const EDFQuery &query, const Vector3D &dirLocal) {
        //if (!matches(query.flags))
        //    return RGBSpectrum::Zero();
        return RGBSpectrum(dirLocal.z > 0.0f ? 1.0f / M_PIf : 0.0f);
    }

    // END: DiffuseEDF
    // ----------------------------------------------------------------
}
