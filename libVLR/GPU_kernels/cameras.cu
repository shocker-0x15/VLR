#include "kernel_common.cuh"

namespace VLR {
    // ----------------------------------------------------------------
    // PerspectiveCamera

    // per Context
    rtDeclareVariable(PerspectiveCamera, pv_perspectiveCamera, , );

    RT_CALLABLE_PROGRAM SampledSpectrum PerspectiveCamera_sampleLensPosition(const WavelengthSamples &wls, const LensPosSample &sample, LensPosQueryResult* result) {
        Matrix3x3 rotMat = pv_perspectiveCamera.orientation.toMatrix3x3();

        float lensRadius = pv_perspectiveCamera.lensRadius;

        float lx, ly;
        concentricSampleDisk(sample.uPos[0], sample.uPos[1], &lx, &ly);
        Point3D orgLocal = Point3D(lensRadius * lx, lensRadius * ly, 0.0f);

        Normal3D geometricNormal = normalize(rotMat * Normal3D(0, 0, 1));

        ReferenceFrame shadingFrame;
        shadingFrame.z = (Vector3D)geometricNormal;
        shadingFrame.x = normalize(rotMat * Vector3D(1, 0, 0));
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint &surfPt = result->surfPt;
        surfPt.position = rotMat * orgLocal + pv_perspectiveCamera.position;
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = false;
        surfPt.atInfinity = false;

        surfPt.geometricNormal = geometricNormal;
        surfPt.u = lx;
        surfPt.v = ly;
        surfPt.texCoord = TexCoord2D::Zero();
        //surfPt.tc0Direction = Vector3D::Zero();

        result->areaPDF = lensRadius > 0.0f ? 1.0f / (M_PIf * lensRadius * lensRadius) : 1.0f;
        result->posType = lensRadius > 0.0f ? DirectionType::LowFreq() : DirectionType::Delta0D();

        return SampledSpectrum(pv_perspectiveCamera.sensitivity);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum PerspectiveCamera_sampleIDF(const SurfacePoint &surfPt, const WavelengthSamples &wls, const IDFSample &sample, IDFQueryResult* result) {
        Point3D orgLocal = Point3D(pv_perspectiveCamera.lensRadius * surfPt.u,
                                   pv_perspectiveCamera.lensRadius * surfPt.v,
                                   0.0f);

        Point3D pFocus = Point3D(pv_perspectiveCamera.opWidth * (0.5f - sample.uDir[0]),
                                 pv_perspectiveCamera.opHeight * (0.5f - sample.uDir[1]),
                                 pv_perspectiveCamera.objPlaneDistance);

        Vector3D dirLocal = normalize(pFocus - orgLocal);
        result->dirLocal = dirLocal;
        result->dirPDF = pv_perspectiveCamera.imgPlaneDistance * pv_perspectiveCamera.imgPlaneDistance / ((dirLocal.z * dirLocal.z * dirLocal.z) * pv_perspectiveCamera.imgPlaneArea);
        result->sampledType = DirectionType::Acquisition() | DirectionType::LowFreq();

        return SampledSpectrum::One();
    }

    // END: PerspectiveCamera
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // EquirectangularCamera

    // per Context
    rtDeclareVariable(EquirectangularCamera, pv_equirectangularCamera, , );

    RT_CALLABLE_PROGRAM SampledSpectrum EquirectangularCamera_sampleLensPosition(const LensPosSample &sample, LensPosQueryResult* result) {
        Matrix3x3 rotMat = pv_equirectangularCamera.orientation.toMatrix3x3();

        Normal3D geometricNormal = normalize(rotMat * Normal3D(0, 0, 1));
        VLRAssert(geometricNormal.length() < 1.01f, "Transform applied to camera can not include scaling.");

        ReferenceFrame shadingFrame;
        shadingFrame.z = (Vector3D)geometricNormal;
        shadingFrame.x = normalize(rotMat * Vector3D(1, 0, 0));
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint &surfPt = result->surfPt;
        surfPt.position = rotMat * Point3D::Zero() + pv_equirectangularCamera.position;
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

        return SampledSpectrum(pv_equirectangularCamera.sensitivity);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum EquirectangularCamera_sampleIDF(const SurfacePoint &surfPt, const IDFSample &sample, IDFQueryResult* result) {
        float phi = pv_equirectangularCamera.phiAngle * (sample.uDir[0] - 0.5f);
        float theta = 0.5f * M_PIf + pv_equirectangularCamera.thetaAngle * (sample.uDir[1] - 0.5f);
        result->dirLocal = Vector3D::fromPolarYUp(phi, theta);
        float sinTheta = std::sqrt(1.0f - result->dirLocal.y * result->dirLocal.y);
        result->dirPDF = 1.0f / (pv_equirectangularCamera.phiAngle * pv_equirectangularCamera.thetaAngle * sinTheta);
        result->sampledType = DirectionType::Acquisition() | DirectionType::LowFreq();

        return SampledSpectrum::One();
    }

    // END: EquirectangularCamera
    // ----------------------------------------------------------------
}
