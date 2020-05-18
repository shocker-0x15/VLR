#include "kernel_common.cuh"

namespace VLR {
    RT_FUNCTION bool intersectTriangle(const Point3D &rayOrg, const Vector3D &rayDir, float tMin, float tMax,
                                       const Point3D &p0, const Point3D &p1, const Point3D &p2,
                                       Normal3D* n, float* t, float* b1, float* b2) {
        Vector3D e0 = p1 - p0;
        Vector3D e1 = p0 - p2;
        *n = static_cast<Normal3D>(cross(e1, e0));

        const Vector3D e2 = (1.0f / dot(*n, rayDir)) * (p0 - rayOrg);
        const Vector3D i = cross(rayDir, e2);

        *b1 = dot(i, e1);
        *b2 = dot(i, e0);
        *t = dot(*n, e2);

        return ((*t < tMax) & (*t > tMin) & (*b1 >= 0.0f) & (*b2 >= 0.0f) & (*b1 + *b2 <= 1));
    }

    // Intersection Program
    RT_PROGRAM void RT_IS_NAME(intersectTriangle)() {
        auto sbtr = optixu::getHitGroupSBTRecordData();
        const GeometryInstanceData &geomInstData = plp.geomInstData[sbtr.geomInstData];
        uint32_t primIndex = optixGetPrimitiveIndex();

        const Triangle &triangle = geomInstData.triangleBuffer[primIndex];
        const Vertex &v0 = geomInstData.vertexBuffer[triangle.index0];
        const Vertex &v1 = geomInstData.vertexBuffer[triangle.index1];
        const Vertex &v2 = geomInstData.vertexBuffer[triangle.index2];

        Point3D rayOrg = asPoint3D(optixGetObjectRayOrigin());
        Vector3D rayDir = asVector3D(optixGetObjectRayDirection());
        float tMin = optixGetRayTmin();
        float tMax = optixGetRayTmax();

        // use a triangle intersection function defined in optix_math_namespace.h
        Normal3D gn;
        float t;
        float b0, b1, b2;
        if (!intersectTriangle(rayOrg, rayDir, tMin, tMax,
                               v0.position, v1.position, v2.position,
                               &gn, &t, &b1, &b2))
            return;

        b0 = 1.0f - b1 - b2;

        optixu::reportIntersection(t, 0, b0, b1);
    }

    //// Bounding Box Program
    //RT_PROGRAM void calcBBoxForTriangle(int32_t primIdx, float result[6]) {
    //    const Triangle &triangle = pv_triangleBuffer[primIdx];
    //    const Point3D &p0 = pv_vertexBuffer[triangle.index0].position;
    //    const Point3D &p1 = pv_vertexBuffer[triangle.index1].position;
    //    const Point3D &p2 = pv_vertexBuffer[triangle.index2].position;

    //    //optix::Aabb* bbox = (optix::Aabb*)result;
    //    //*bbox = optix::Aabb(asOptiXType(p0), asOptiXType(p1), asOptiXType(p2));

    //    BoundingBox3D* bbox = (BoundingBox3D*)result;
    //    *bbox = BoundingBox3D(Point3D(INFINITY), Point3D(-INFINITY));
    //    bbox->unify(p0);
    //    bbox->unify(p1);
    //    bbox->unify(p2);
    //}

    //// Attribute Program (for GeometryTriangles)
    //RT_PROGRAM void calcAttributeForTriangle() {
    //    optix::float2 bc = rtGetTriangleBarycentrics();
    //    a_hitPointParam.b0 = 1 - bc.x - bc.y;
    //    a_hitPointParam.b1 = bc.x;
    //    a_hitPointParam.primIndex = rtGetPrimitiveIndex();
    //}



    RT_CALLABLE_PROGRAM void RT_DC_NAME(decodeHitPointForTriangle)(const HitPointParameter &param, SurfacePoint* surfPt, float* hypAreaPDF) {
        auto sbtr = optixu::getHitGroupSBTRecordData();
        const GeometryInstanceData &geomInstData = plp.geomInstData[sbtr.geomInstData];

        const Triangle &triangle = geomInstData.triangleBuffer[param.primIndex];
        const Vertex &v0 = geomInstData.vertexBuffer[triangle.index0];
        const Vertex &v1 = geomInstData.vertexBuffer[triangle.index1];
        const Vertex &v2 = geomInstData.vertexBuffer[triangle.index2];

        Vector3D e1 = transform<TransformKind::ObjectToWorld>(v1.position - v0.position);
        Vector3D e2 = transform<TransformKind::ObjectToWorld>(v2.position - v0.position);
        Normal3D geometricNormal = static_cast<Normal3D>(cross(e1, e2));
        float area = geometricNormal.length() / 2; // TODO: スケーリングの考慮。
        geometricNormal /= 2 * area;

        // JP: プログラムがこの点を光源としてサンプルする場合の面積に関する(仮想的な)PDFを求める。
        // EN: calculate a hypothetical area PDF value in the case where the program sample this point as light.
        float probLightPrim = area / geomInstData.sumImportances;
        *hypAreaPDF = probLightPrim / area;

        float b0 = param.b0, b1 = param.b1, b2 = 1.0f - param.b0 - param.b1;
        Point3D position = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        Normal3D shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        Vector3D tc0Direction = b0 * v0.tc0Direction + b1 * v1.tc0Direction + b2 * v2.tc0Direction;
        TexCoord2D texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        position = transform<TransformKind::ObjectToWorld>(position);
        shadingNormal = normalize(transform<TransformKind::ObjectToWorld>(shadingNormal));
        tc0Direction = transform<TransformKind::ObjectToWorld>(tc0Direction);

        // JP: 法線と接線が直交することを保証する。
        //     直交性の消失は重心座標補間によっておこる？
        // EN: guarantee the orthogonality between the normal and tangent.
        //     Orthogonality break might be caused by barycentric interpolation?
        float dotNT = dot(shadingNormal, tc0Direction);
        tc0Direction = normalize(tc0Direction - dotNT * static_cast<Vector3D>(shadingNormal));

        surfPt->position = position;
        surfPt->shadingFrame = ReferenceFrame(tc0Direction, shadingNormal);
        surfPt->isPoint = false;
        surfPt->atInfinity = false;
        surfPt->geometricNormal = geometricNormal;
        surfPt->u = b0;
        surfPt->v = b1;
        surfPt->texCoord = texCoord;
    }



    RT_CALLABLE_PROGRAM void RT_DC_NAME(sampleTriangleMesh)(const GeometryInstanceDescriptor::Body &desc, const SurfaceLightPosSample &sample, SurfaceLightPosQueryResult* result) {
        float primProb;
        uint32_t primIdx = desc.asTriMesh.primDistribution.sample(sample.uElem, &primProb);

        const Triangle &triangle = desc.asTriMesh.triangleBuffer[primIdx];
        const Vertex &v0 = desc.asTriMesh.vertexBuffer[triangle.index0];
        const Vertex &v1 = desc.asTriMesh.vertexBuffer[triangle.index1];
        const Vertex &v2 = desc.asTriMesh.vertexBuffer[triangle.index2];

        StaticTransform transform = desc.asTriMesh.transform;

        Vector3D e1 = transform * (v1.position - v0.position);
        Vector3D e2 = transform * (v2.position - v0.position);
        Normal3D geometricNormal = static_cast<Normal3D>(cross(e1, e2));
        float area = geometricNormal.length() / 2;
        geometricNormal /= 2 * area;

        result->areaPDF = primProb / area;
        result->posType = DirectionType::Emission() | DirectionType::LowFreq();

        float b0, b1, b2;
        uniformSampleTriangle(sample.uPos[0], sample.uPos[1], &b0, &b1);
        b2 = 1.0f - b0 - b1;

        Point3D position = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        Normal3D shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        Vector3D tc0Direction = b0 * v0.tc0Direction + b1 * v1.tc0Direction + b2 * v2.tc0Direction;
        TexCoord2D texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        position = transform * position;
        shadingNormal = normalize(transform * shadingNormal);
        tc0Direction = transform * tc0Direction;

        // JP: 法線と接線が直交することを保証する。
        //     直交性の消失は重心座標補間によっておこる？
        // EN: guarantee the orthogonality between the normal and tangent.
        //     Orthogonality break might be caused by barycentric interpolation?
        float dotNT = dot(shadingNormal, tc0Direction);
        tc0Direction = normalize(tc0Direction - dotNT * static_cast<Vector3D>(shadingNormal));

        SurfacePoint &surfPt = result->surfPt;

        surfPt.position = position;
        surfPt.shadingFrame = ReferenceFrame(tc0Direction, shadingNormal);
        surfPt.isPoint = false;
        surfPt.atInfinity = false;
        surfPt.geometricNormal = geometricNormal;
        surfPt.u = b0;
        surfPt.v = b1;
        surfPt.texCoord = texCoord;
    }
}
