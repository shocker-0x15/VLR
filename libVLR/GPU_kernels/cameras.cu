#include "kernel_common.cuh"

namespace VLR {
    // ----------------------------------------------------------------
    // PerspectiveCamera

    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(PerspectiveCamera_sampleLensPosition)(const WavelengthSamples &wls, const LensPosSample &sample, LensPosQueryResult* result) {
        Matrix3x3 rotMat = plp.perspectiveCamera.orientation.toMatrix3x3();

        float lensRadius = plp.perspectiveCamera.lensRadius;

        float lx, ly;
        concentricSampleDisk(sample.uPos[0], sample.uPos[1], &lx, &ly);
        Point3D orgLocal = Point3D(lensRadius * lx, lensRadius * ly, 0.0f);

        Normal3D geometricNormal = normalize(rotMat * Normal3D(0, 0, 1));

        ReferenceFrame shadingFrame;
        shadingFrame.z = geometricNormal;
        shadingFrame.x = normalize(rotMat * Vector3D(1, 0, 0));
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint &surfPt = result->surfPt;
        surfPt.position = rotMat * orgLocal + plp.perspectiveCamera.position;
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = false;
        surfPt.atInfinity = false;

        surfPt.geometricNormal = geometricNormal;
        surfPt.u = lx;
        surfPt.v = ly;
        surfPt.texCoord = TexCoord2D::Zero();
        //surfPt.tc0Direction = Vector3D::Zero();

        result->areaPDF = lensRadius > 0.0f ? 1.0f / (VLR_M_PI * lensRadius * lensRadius) : 1.0f;
        result->posType = lensRadius > 0.0f ? DirectionType::LowFreq() : DirectionType::Delta0D();

        return SampledSpectrum(plp.perspectiveCamera.sensitivity);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(PerspectiveCamera_sampleIDF)(const SurfacePoint &surfPt, const WavelengthSamples &wls, const IDFSample &sample, IDFQueryResult* result) {
        Point3D orgLocal = Point3D(plp.perspectiveCamera.lensRadius * surfPt.u,
                                   plp.perspectiveCamera.lensRadius * surfPt.v,
                                   0.0f);

        Point3D pFocus = Point3D(plp.perspectiveCamera.opWidth * (0.5f - sample.uDir[0]),
                                 plp.perspectiveCamera.opHeight * (0.5f - sample.uDir[1]),
                                 plp.perspectiveCamera.objPlaneDistance);

        Vector3D dirLocal = normalize(pFocus - orgLocal);
        result->dirLocal = dirLocal;
        result->dirPDF = plp.perspectiveCamera.imgPlaneDistance * plp.perspectiveCamera.imgPlaneDistance / ((dirLocal.z * dirLocal.z * dirLocal.z) * plp.perspectiveCamera.imgPlaneArea);
        result->sampledType = DirectionType::Acquisition() | DirectionType::LowFreq();

        return SampledSpectrum::One();
    }

    // END: PerspectiveCamera
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // EquirectangularCamera

    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EquirectangularCamera_sampleLensPosition)(const WavelengthSamples &wls, const LensPosSample &sample, LensPosQueryResult* result) {
        Matrix3x3 rotMat = plp.equirectangularCamera.orientation.toMatrix3x3();

        Normal3D geometricNormal = normalize(rotMat * Normal3D(0, 0, 1));
        VLRAssert(geometricNormal.length() < 1.01f, "Transform applied to camera can not include scaling.");

        ReferenceFrame shadingFrame;
        shadingFrame.z = geometricNormal;
        shadingFrame.x = normalize(rotMat * Vector3D(1, 0, 0));
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint &surfPt = result->surfPt;
        surfPt.position = rotMat * Point3D::Zero() + plp.equirectangularCamera.position;
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = true;
        surfPt.atInfinity = false;

        surfPt.geometricNormal = geometricNormal;
        surfPt.u = 0;
        surfPt.v = 0;
        surfPt.texCoord = TexCoord2D::Zero();
        //surfPt.tc0Direction = Vector3D::Zero();

        result->areaPDF = 1.0f;
        result->posType = DirectionType::Delta0D();

        return SampledSpectrum(plp.equirectangularCamera.sensitivity);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(EquirectangularCamera_sampleIDF)(const SurfacePoint &surfPt, const WavelengthSamples &wls, const IDFSample &sample, IDFQueryResult* result) {
        float phi = plp.equirectangularCamera.phiAngle * (sample.uDir[0] - 0.5f);
        float theta = 0.5f * VLR_M_PI + plp.equirectangularCamera.thetaAngle * (sample.uDir[1] - 0.5f);
        result->dirLocal = Vector3D::fromPolarYUp(phi, theta);
        float sinTheta = std::sqrt(1.0f - result->dirLocal.y * result->dirLocal.y);
        result->dirPDF = 1.0f / (plp.equirectangularCamera.phiAngle * plp.equirectangularCamera.thetaAngle * sinTheta);
        result->sampledType = DirectionType::Acquisition() | DirectionType::LowFreq();

        return SampledSpectrum::One();
    }

    // END: EquirectangularCamera
    // ----------------------------------------------------------------
}
