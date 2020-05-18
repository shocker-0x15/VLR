#include "kernel_common.cuh"

namespace VLR {
    // Intersection Program for Infinite Sphere
    RT_PROGRAM void RT_IS_NAME(intersectInfiniteSphere)() {
        Vector3D direction = asVector3D(optixGetObjectRayDirection());
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        optixu::reportIntersection(INFINITY, 0, phi, theta);
    }

    //// Bounding Box Program for Infinite Sphere
    //RT_PROGRAM void calcBBoxForInfiniteSphere(int32_t primIdx, float result[6]) {
    //    auto bbox = reinterpret_cast<BoundingBox3D*>(result);
    //    *bbox = BoundingBox3D(Point3D(-INFINITY), Point3D(INFINITY));
    //}



    RT_CALLABLE_PROGRAM void RT_DC_NAME(decodeHitPointForInfiniteSphere)(const HitPointParameter &param, SurfacePoint* surfPt, float* hypAreaPDF) {
        float phi = param.b0;
        float theta = param.b1;
        Vector3D direction = transform<TransformKind::ObjectToWorld>(Vector3D::fromPolarYUp(phi, theta));
        float sinPhi, cosPhi;
        VLR::sincos(phi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = transform<TransformKind::ObjectToWorld>(Vector3D(-cosPhi, 0.0f, -sinPhi));

        Normal3D geometricNormal = -static_cast<Normal3D>(direction);

        surfPt->position = Point3D(direction);
        surfPt->shadingFrame = ReferenceFrame(texCoord0Dir, geometricNormal);
        surfPt->isPoint = false;
        surfPt->atInfinity = true;

        surfPt->geometricNormal = geometricNormal;
        surfPt->u = param.b0;
        surfPt->v = param.b1;
        surfPt->texCoord = TexCoord2D(phi / (2 * VLR_M_PI), theta / VLR_M_PI);

        // calculate a hypothetical area PDF value in the case where the program sample this point as light.
        *hypAreaPDF = 0;
    }



    RT_CALLABLE_PROGRAM void RT_DC_NAME(sampleInfiniteSphere)(const GeometryInstanceDescriptor::Body &desc, const SurfaceLightPosSample &sample, SurfaceLightPosQueryResult* result) {
        float u, v;
        float uvPDF;
        desc.asInfSphere.importanceMap.sample(sample.uPos[0], sample.uPos[1], &u, &v, &uvPDF);
        float phi = 2 * VLR_M_PI * u;
        float theta = VLR_M_PI * v;

        float posPhi = phi - desc.asInfSphere.rotationPhi;
        posPhi = posPhi - std::floor(posPhi / (2 * VLR_M_PI)) * 2 * VLR_M_PI;

        Vector3D direction = Vector3D::fromPolarYUp(posPhi, theta);
        Point3D position(direction);

        float sinPhi, cosPhi;
        VLR::sincos(posPhi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));

        Normal3D geometricNormal = -static_cast<Normal3D>(direction);

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
    }
}
