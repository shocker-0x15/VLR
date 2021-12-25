#include "../shared/kernel_common.h"

namespace vlr {
    using namespace shared;

    // Intersection Program for Infinite Sphere
    CUDA_DEVICE_KERNEL void RT_IS_NAME(intersectInfiniteSphere)(int32_t primIdx) {
        Vector3D direction = asVector3D(optixGetObjectRayDirection());
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        optixu::reportIntersection<InfiniteSphereAttributeSignature>(INFINITY, 0, phi, theta);
    }



    RT_CALLABLE_PROGRAM void RT_DC_NAME(decodeHitPointForInfiniteSphere)(
        const HitPointParameter &param, SurfacePoint* surfPt, float* hypAreaPDF) {
        Vector3D direction = Vector3D::fromPolarYUp(param.phi, param.theta);
        float sinPhi, cosPhi;
        ::vlr::sincos(param.phi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = Vector3D(-cosPhi, 0.0f, -sinPhi);

        surfPt->position = Point3D(direction.x, direction.y, direction.z);
        surfPt->shadingFrame.x = texCoord0Dir;
        surfPt->shadingFrame.z = -direction;
        surfPt->isPoint = false;
        surfPt->atInfinity = true;

        surfPt->geometricNormal = -direction;
        surfPt->u = param.phi;
        surfPt->v = param.theta;
        surfPt->texCoord = TexCoord2D(param.phi / (2 * VLR_M_PI), param.theta / VLR_M_PI);

        // calculate a hypothetical area PDF value in the case where the program sample this point as light.
        *hypAreaPDF = 0;
    }



    RT_CALLABLE_PROGRAM void RT_DC_NAME(sampleInfiniteSphere)(
        const Instance &inst, const GeometryInstance &geomInst,
        const SurfaceLightPosSample &sample, const Point3D &shadingPoint,
        SurfaceLightPosQueryResult* result) {
        (void)shadingPoint;

        float u, v;
        float uvPDF;
        geomInst.asInfSphere.importanceMap.sample(sample.uPos[0], sample.uPos[1], &u, &v, &uvPDF);
        float phi = 2 * VLR_M_PI * u;
        float theta = VLR_M_PI * v;

        float posPhi = phi - inst.rotationPhi;
        posPhi = posPhi - ::vlr::floor(posPhi / (2 * VLR_M_PI)) * 2 * VLR_M_PI;

        Vector3D direction = Vector3D::fromPolarYUp(posPhi, theta);
        Point3D position = Point3D(direction.x, direction.y, direction.z);

        float sinPhi, cosPhi;
        ::vlr::sincos(posPhi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));

        Normal3D geometricNormal = -static_cast<Vector3D>(position);

        ReferenceFrame shadingFrame;
        shadingFrame.x = texCoord0Dir;
        shadingFrame.z = geometricNormal;
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);
        VLRAssert(absDot(shadingFrame.z, shadingFrame.x) < 0.01f, "shading normal and tangent must be orthogonal.");

        SurfacePoint &surfPt = result->surfPt;

        surfPt.position = position;
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = false;
        surfPt.atInfinity = true;

        surfPt.geometricNormal = geometricNormal;
        surfPt.u = posPhi;
        surfPt.v = theta;
        surfPt.texCoord = TexCoord2D(phi / (2 * VLR_M_PI), theta / VLR_M_PI);

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * M_PI * M_PI * std::sin(theta)) / l^2
        result->areaPDF = uvPDF / (2 * VLR_M_PI * VLR_M_PI * std::sin(theta));
        result->posType = DirectionType::Emission() | DirectionType::LowFreq();
        result->materialIndex = geomInst.materialIndex;
    }
}
