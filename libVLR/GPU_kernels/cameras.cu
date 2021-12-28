#include "../shared/kernel_common.h"

namespace vlr {
    using namespace shared;

#define DEFINE_IDF_INTERFACES(IDF)\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(IDF ## _sampleInternal)(\
        const uint32_t* params,\
        const IDFQuery &query, const float uDir[2], IDFQueryResult* result) {\
        auto &p = *reinterpret_cast<const IDF*>(params);\
        return p.sampleInternal(query, uDir, result);\
    }\
\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(IDF ## _evaluateSpatialImportanceInternal)(\
        const uint32_t* params) {\
        auto &p = *reinterpret_cast<const IDF*>(params);\
        return p.evaluateSpatialImportanceInternal();\
    }\
\
    RT_CALLABLE_PROGRAM SampledSpectrum RT_DC_NAME(IDF ## _evaluateDirectionalImportanceInternal)(\
        const uint32_t* params,\
        const IDFQuery &query, const Vector3D &dirLocal) {\
        auto &p = *reinterpret_cast<const IDF*>(params);\
        return p.evaluateDirectionalImportanceInternal(query, dirLocal);\
    }\
\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(IDF ## _evaluatePDFInternal)(\
        const uint32_t* params,\
        const IDFQuery &query, const Vector3D &dirLocal) {\
        auto &p = *reinterpret_cast<const IDF*>(params);\
        return p.evaluatePDFInternal(query, dirLocal);\
    }



    RT_CALLABLE_PROGRAM void RT_DC_NAME(PerspectiveCamera_sample)(
        const LensPosSample &sample, LensPosQueryResult* result) {
        auto &cam = reinterpret_cast<const PerspectiveCamera &>(plp.cameraDescriptor.data);

        Matrix3x3 rotMat = cam.orientation.toMatrix3x3();

        float lensRadius = cam.lensRadius;

        float lx, ly;
        concentricSampleDisk(sample.uPos[0], sample.uPos[1], &lx, &ly);
        Point3D orgLocal = Point3D(lensRadius * lx, lensRadius * ly, 0.0f);

        Normal3D geometricNormal = normalize(rotMat * Normal3D(0, 0, 1));

        ReferenceFrame shadingFrame;
        shadingFrame.z = static_cast<Vector3D>(geometricNormal);
        shadingFrame.x = normalize(rotMat * Vector3D(1, 0, 0));
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint &surfPt = result->surfPt;
        surfPt.position = rotMat * orgLocal + cam.position;
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
    }

    class PerspectiveCameraIDF {
        float m_sensitivity;
        float m_xOnLens;
        float m_yOnLens;
        float m_opWidth;
        float m_opHeight;
        float m_imgPlaneDistance;
        float m_objPlaneDistance;
        float m_imgPlaneArea;

    public:
        CUDA_DEVICE_FUNCTION PerspectiveCameraIDF(
            float sensitivity,
            float xOnLens, float yOnLens, float opWidth, float opHeight,
            float imgPlaneDistance, float objPlaneDistance, float imgPlaneArea) :
            m_sensitivity(sensitivity),
            m_xOnLens(xOnLens), m_yOnLens(yOnLens), m_opWidth(opWidth), m_opHeight(opHeight),
            m_imgPlaneDistance(imgPlaneDistance),
            m_objPlaneDistance(objPlaneDistance),
            m_imgPlaneArea(imgPlaneArea) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const IDFQuery &query, const float uDir[2], IDFQueryResult* result) const {
            Point3D pFocus = Point3D(m_opWidth * (0.5f - uDir[0]),
                                     m_opHeight * (0.5f - uDir[1]),
                                     m_objPlaneDistance);
            Point3D orgLocal = Point3D(m_xOnLens, m_yOnLens, 0.0f);
            Vector3D dirLocal = normalize(pFocus - orgLocal);
            result->dirLocal = dirLocal;
            result->dirPDF = pow2(m_imgPlaneDistance) / (pow3(dirLocal.z) * m_imgPlaneArea);
            result->sampledType = DirectionType::Acquisition() | DirectionType::LowFreq();

            return SampledSpectrum::One();
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateSpatialImportanceInternal() const {
            return SampledSpectrum(m_sensitivity);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateDirectionalImportanceInternal(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            if (dirLocal.z <= 0.0f)
                return SampledSpectrum::Zero();

            Point3D orgLocal = Point3D(m_xOnLens, m_yOnLens, 0.0f);
            float objPlaneZ = m_objPlaneDistance / dirLocal.z;
            Point3D pFocus = dirLocal * objPlaneZ + orgLocal;
            float uDir0 = -pFocus.x / m_opWidth + 0.5f;
            float uDir1 = -pFocus.x / m_opHeight + 0.5f;
            if (uDir0 < 0.0f || uDir0 >= 1.0f || uDir1 < 0.0f || uDir1 >= 1.0f)
                return SampledSpectrum::Zero();

            return SampledSpectrum::One();
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            if (dirLocal.z <= 0.0f)
                return 0.0f;

            Point3D orgLocal = Point3D(m_xOnLens, m_yOnLens, 0.0f);
            float objPlaneZ = m_objPlaneDistance / dirLocal.z;
            Point3D pFocus = dirLocal * objPlaneZ + orgLocal;
            float uDir0 = -pFocus.x / m_opWidth + 0.5f;
            float uDir1 = -pFocus.x / m_opHeight + 0.5f;
            if (uDir0 < 0.0f || uDir0 >= 1.0f || uDir1 < 0.0f || uDir1 >= 1.0f)
                return 0.0f;

            float density = pow2(m_imgPlaneDistance) / (pow3(dirLocal.z) * m_imgPlaneArea);
            return density;
        }
    };

    RT_CALLABLE_PROGRAM void RT_DC_NAME(PerspectiveCamera_setupIDF)(
        const uint32_t* camDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<PerspectiveCameraIDF*>(params);
        auto &cam = *reinterpret_cast<const PerspectiveCamera*>(camDesc);

        p = PerspectiveCameraIDF(
            cam.sensitivity,
            cam.lensRadius * surfPt.u, cam.lensRadius * surfPt.v, cam.opWidth, cam.opHeight,
            cam.imgPlaneDistance, cam.objPlaneDistance, cam.imgPlaneArea);
    }

    DEFINE_IDF_INTERFACES(PerspectiveCameraIDF)



    RT_CALLABLE_PROGRAM void RT_DC_NAME(EquirectangularCamera_sample)(
        const LensPosSample &sample, LensPosQueryResult* result) {
        auto &cam = reinterpret_cast<const EquirectangularCamera &>(plp.cameraDescriptor.data);

        Matrix3x3 rotMat = cam.orientation.toMatrix3x3();

        Normal3D geometricNormal = normalize(rotMat * Normal3D(0, 0, 1));
        VLRAssert(geometricNormal.length() < 1.01f, "Transform applied to camera can not include scaling.");

        ReferenceFrame shadingFrame;
        shadingFrame.z = static_cast<Vector3D>(geometricNormal);
        shadingFrame.x = normalize(rotMat * Vector3D(1, 0, 0));
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint &surfPt = result->surfPt;
        surfPt.position = rotMat * Point3D::Zero() + cam.position;
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
    }

    class EquirectangularCameraIDF {
        float m_sensitivity;
        float m_phiAngle;
        float m_thetaAngle;

    public:
        CUDA_DEVICE_FUNCTION EquirectangularCameraIDF(
            float sensitivity,
            float phiAngle, float thetaAngle) :
            m_sensitivity(sensitivity),
            m_phiAngle(phiAngle), m_thetaAngle(thetaAngle) {}

        CUDA_DEVICE_FUNCTION SampledSpectrum sampleInternal(
            const IDFQuery &query, const float uDir[2], IDFQueryResult* result) const {
            float phi = m_phiAngle * (uDir[0] - 0.5f);
            float theta = 0.5f * VLR_M_PI + m_thetaAngle * (uDir[1] - 0.5f);
            result->dirLocal = Vector3D::fromPolarYUp(phi, theta);
            float sinTheta = std::sqrt(1.0f - pow2(result->dirLocal.y));
            result->dirPDF = 1.0f / (m_phiAngle * m_thetaAngle * sinTheta);
            result->sampledType = DirectionType::Acquisition() | DirectionType::LowFreq();

            return SampledSpectrum::One();
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateSpatialImportanceInternal() const {
            return SampledSpectrum(m_sensitivity);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrum evaluateDirectionalImportanceInternal(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            float phi, theta;
            dirLocal.toPolarYUp(&theta, &phi);
            float uDir0 = phi / m_phiAngle + 0.5f;
            float uDir1 = (theta - 0.5f * VLR_M_PI) / m_thetaAngle + 0.5f;
            if (uDir0 < 0.0f || uDir0 >= 1.0f || uDir1 < 0.0f || uDir1 >= 1.0f)
                return SampledSpectrum::Zero();

            return SampledSpectrum::One();
        }

        CUDA_DEVICE_FUNCTION float evaluatePDFInternal(
            const IDFQuery &query, const Vector3D &dirLocal) const {
            float phi, theta;
            dirLocal.toPolarYUp(&theta, &phi);
            float uDir0 = phi / m_phiAngle + 0.5f;
            float uDir1 = (theta - 0.5f * VLR_M_PI) / m_thetaAngle + 0.5f;
            if (uDir0 < 0.0f || uDir0 >= 1.0f || uDir1 < 0.0f || uDir1 >= 1.0f)
                return 0.0f;

            float sinTheta = std::sqrt(1.0f - pow2(dirLocal.y));
            float density = 1.0f / (m_phiAngle * m_thetaAngle * sinTheta);
            return density;
        }
    };

    RT_CALLABLE_PROGRAM void RT_DC_NAME(EquirectangularCamera_setupIDF)(
        const uint32_t* camDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls, uint32_t* params) {
        auto &p = *reinterpret_cast<EquirectangularCameraIDF*>(params);
        auto &cam = *reinterpret_cast<const EquirectangularCamera*>(camDesc);

        p = EquirectangularCameraIDF(cam.sensitivity, cam.phiAngle, cam.thetaAngle);
    }

    DEFINE_IDF_INTERFACES(EquirectangularCameraIDF)
}
