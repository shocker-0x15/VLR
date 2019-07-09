#include "kernel_common.cuh"

namespace VLR {
    // Intersection Program for Infinite Sphere
    RT_PROGRAM void intersectInfiniteSphere(int32_t primIdx) {
        float t = FLT_MAX;
        if (!rtPotentialIntersection(t))
            return;

        Vector3D direction = asVector3D(sm_ray.direction);
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        a_hitPointParam.b0 = phi;
        a_hitPointParam.b1 = theta;
        a_hitPointParam.primIndex = primIdx;

        const uint32_t materialIndex = 0;
        rtReportIntersection(materialIndex);
    }

    // Bounding Box Program for Infinite Sphere
    RT_PROGRAM void calcBBoxForInfiniteSphere(int32_t primIdx, float result[6]) {
        BoundingBox3D* bbox = (BoundingBox3D*)result;
        *bbox = BoundingBox3D(Point3D(-INFINITY), Point3D(INFINITY));
    }



    // bound
    RT_CALLABLE_PROGRAM void decodeHitPointForInfiniteSphere(const ObjectInfo &objInfo, const HitPointParameter &param, SurfacePoint* surfPt, float* hypAreaPDF) {
        float phi = param.b0;
        float theta = param.b1;
        Vector3D direction = objInfo.transform * Vector3D::fromPolarYUp(phi, theta);
        float sinPhi, cosPhi;
        VLR::sincos(phi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = objInfo.transform * Vector3D(-cosPhi, 0.0f, -sinPhi);

        surfPt->position = Point3D(direction.x, direction.y, direction.z);
        surfPt->shadingFrame = ReferenceFrame(texCoord0Dir, -direction);
        surfPt->isPoint = false;
        surfPt->atInfinity = true;

        surfPt->geometricNormal = -direction;
        surfPt->u = param.b0;
        surfPt->v = param.b1;
        surfPt->texCoord = TexCoord2D(phi / (2 * M_PIf), theta / M_PIf);

        // calculate a hypothetical area PDF value in the case where the program sample this point as light.
        *hypAreaPDF = 0;
    }



    RT_CALLABLE_PROGRAM void sampleInfiniteSphere(const SurfaceLightDescriptor::Body &desc, const SurfaceLightPosSample &sample, SurfaceLightPosQueryResult* result) {
        float u, v;
        float uvPDF;
        desc.asEnvironmentLight.importanceMap.sample(sample.uPos[0], sample.uPos[1], &u, &v, &uvPDF);
        float phi = 2 * M_PIf * u;
        float theta = M_PIf * v;

        float posPhi = phi - desc.asEnvironmentLight.rotationPhi;
        posPhi = posPhi - std::floor(posPhi / (2 * M_PIf)) * 2 * M_PIf;

        result->materialIndex = desc.asEnvironmentLight.materialIndex;

        Vector3D direction = Vector3D::fromPolarYUp(posPhi, theta);
        Point3D position = Point3D(direction.x, direction.y, direction.z);

        Matrix4x4 identity = Matrix4x4::Identity();
        result->objInfo.transform = StaticTransform(identity, identity);

        float sinPhi, cosPhi;
        VLR::sincos(posPhi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));

        Normal3D geometricNormal = -(Vector3D)position;

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
        surfPt.texCoord = TexCoord2D(phi / (2 * M_PIf), theta / M_PIf);

        // JP: テクスチャー空間中のPDFを面積に関するものに変換する。
        // EN: convert the PDF in texture space to one with respect to area.
        // The true value is: lim_{l to inf} uvPDF / (2 * M_PI * M_PI * std::sin(theta)) / l^2
        result->areaPDF = uvPDF / (2 * M_PIf * M_PIf * std::sin(theta));
        result->posType = DirectionType::Emission() | DirectionType::LowFreq();
    }
}
