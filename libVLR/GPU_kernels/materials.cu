#include "kernel_common.cuh"
#include "random_distributions.cuh"

namespace VLR {
    // ----------------------------------------------------------------
    // MatteBRDF

    struct MatteBRDF {
        RGBSpectrum albedo;
        float roughness;
    };

    RT_CALLABLE_PROGRAM void MatteSurfaceMaterial_setup(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        MatteBRDF &p = *(MatteBRDF*)params;
        MatteSurfaceMaterial &mat = *(MatteSurfaceMaterial*)matDesc;

        optix::float4 texValue = optix::rtTex2D<optix::float4>(mat.texAlbedoRoughness, surfPt.texCoord.u, surfPt.texCoord.v);
        p.albedo = RGBSpectrum(texValue.x, texValue.y, texValue.z);
        p.roughness = texValue.w;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum MatteBRDF_getBaseColor(const uint32_t* params) {
        MatteBRDF &p = *(MatteBRDF*)params;

        return p.albedo;
    }

    RT_CALLABLE_PROGRAM bool MatteBRDF_matches(DirectionType flags) {
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

    // END: MatteBRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // UE4 BRDF

    struct UE4BRDF {
        RGBSpectrum baseColor;
        float roughenss;
        float metallic;
    };

    RT_CALLABLE_PROGRAM void UE4SurfaceMaterial_setup(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        UE4BRDF &p = *(UE4BRDF*)params;
        UE4SurfaceMaterial &mat = *(UE4SurfaceMaterial*)matDesc;

        VLRAssert_NotImplemented();
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_getBaseColor(const uint32_t* params) {
        VLRAssert_NotImplemented();
        return RGBSpectrum::Zero();
    }
    
    RT_CALLABLE_PROGRAM bool UE4BRDF_matches(DirectionType flags) {
        VLRAssert_NotImplemented();
        return true;
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_sampleBSDFInternal(const uint32_t* params, const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
        VLRAssert_NotImplemented();
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM RGBSpectrum UE4BRDF_evaluateBSDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        VLRAssert_NotImplemented();
        return RGBSpectrum::Zero();
    }

    RT_CALLABLE_PROGRAM float UE4BRDF_evaluateBSDF_PDFInternal(const uint32_t* params, const BSDFQuery &query, const Vector3D &dirLocal) {
        VLRAssert_NotImplemented();
        return 0.0f;
    }

    // END: UE4 BRDF
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // NullEDF

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

    RT_CALLABLE_PROGRAM void DiffuseEmitterMaterial_setup(const uint32_t* matDesc, const SurfacePoint &surfPt, uint32_t* params) {
        DiffuseEDF &p = *(DiffuseEDF*)params;
        DiffuseEmitterMaterial &mat = *(DiffuseEmitterMaterial*)matDesc;

        optix::float4 texValue = optix::rtTex2D<optix::float4>(mat.texEmittance, surfPt.texCoord.u, surfPt.texCoord.v);
        p.emittance = RGBSpectrum(texValue.x, texValue.y, texValue.z);
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
}
