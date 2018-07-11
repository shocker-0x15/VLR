#include "kernel_common.cuh"
#include "random_distributions.cuh"

namespace VLR {
    // per GeometryInstance
    // closestHitProgramなどから呼ばれるdecodeHitPoint等で読み出すためにはGeometryInstanceレベルにバインドする必要がある。
    rtBuffer<Vertex> pv_vertexBuffer;
    rtBuffer<Triangle> pv_triangleBuffer;

    rtDeclareVariable(optix::Ray, sm_ray, rtCurrentRay, );
    rtDeclareVariable(HitPointParameter, a_hitPointParam, attribute hitPointParam, );

    // Intersection Program
    RT_PROGRAM void intersectTriangle(int32_t primIdx) {
        const Triangle &triangle = pv_triangleBuffer[primIdx];
        const Vertex &v0 = pv_vertexBuffer[triangle.index0];
        const Vertex &v1 = pv_vertexBuffer[triangle.index1];
        const Vertex &v2 = pv_vertexBuffer[triangle.index2];

        // use a triangle intersection function defined in optix_math_namespace.h
        optix::float3 gn;
        float t;
        float b0, b1, b2;
        if (!intersect_triangle(sm_ray, asOptiXType(v0.position), asOptiXType(v1.position), asOptiXType(v2.position),
                                gn, t, b1, b2))
            return;

        if (!rtPotentialIntersection(t))
            return;

        b0 = 1.0f - b1 - b2;
        a_hitPointParam.b0 = b0;
        a_hitPointParam.b1 = b1;
        a_hitPointParam.primIndex = primIdx;

        rtReportIntersection(triangle.matIndex);
    }

    // Bounding Box Program
    RT_PROGRAM void calcBBoxForTriangle(int32_t primIdx, float result[6]) {
        const Triangle &triangle = pv_triangleBuffer[primIdx];
        const Point3D &p0 = pv_vertexBuffer[triangle.index0].position;
        const Point3D &p1 = pv_vertexBuffer[triangle.index1].position;
        const Point3D &p2 = pv_vertexBuffer[triangle.index2].position;

        BoundingBox3D bbox;
        bbox.unify(p0);
        bbox.unify(p1);
        bbox.unify(p2);

        result[0] = bbox.minP.x;
        result[1] = bbox.minP.y;
        result[2] = bbox.minP.z;
        result[3] = bbox.maxP.x;
        result[4] = bbox.maxP.y;
        result[5] = bbox.maxP.z;
    }


    
    RT_CALLABLE_PROGRAM void decodeHitPoint(const HitPointParameter &param, SurfacePoint* surfPt) {
        const Triangle &triangle = pv_triangleBuffer[param.primIndex];
        const Vertex &v0 = pv_vertexBuffer[triangle.index0];
        const Vertex &v1 = pv_vertexBuffer[triangle.index1];
        const Vertex &v2 = pv_vertexBuffer[triangle.index2];

        Normal3D geometricNormal = normalize(cross(v1.position - v0.position, v2.position - v0.position));

        float b0 = param.b0, b1 = param.b1, b2 = 1.0f - param.b0 - param.b1;
        Point3D position = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        Normal3D shadingNormal = normalize(b0 * v0.normal + b1 * v1.normal + b2 * v2.normal);
        Vector3D shadingTangent = normalize(b0 * v0.tangent + b1 * v1.tangent + b2 * v2.tangent);
        TexCoord2D texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        //TexCoord2D dUV0m2 = v0.texCoord - v2.texCoord;
        //TexCoord2D dUV1m2 = v1.texCoord - v2.texCoord;
        //Vector3D dP0m2 = v0.position - v2.position;
        //Vector3D dP1m2 = v1.position - v2.position;

        //float invDetUV = 1.0f / (dUV0m2.u * dUV1m2.v - dUV0m2.v * dUV1m2.u);
        //Vector3D uDirection = invDetUV * Vector3D(dUV1m2.v * dP0m2.x - dUV0m2.v * dP1m2.x,
        //                                          dUV1m2.v * dP0m2.y - dUV0m2.v * dP1m2.y,
        //                                          dUV1m2.v * dP0m2.z - dUV0m2.v * dP1m2.z);
        //uDirection = normalize(cross(cross(shadingNormal, uDirection), shadingNormal));

        position = transform(RT_OBJECT_TO_WORLD, position);
        shadingNormal = normalize(transform(RT_OBJECT_TO_WORLD, shadingNormal));
        shadingTangent = normalize(transform(RT_OBJECT_TO_WORLD, shadingTangent));

        surfPt->position = position;
        surfPt->shadingFrame = ReferenceFrame(shadingTangent, shadingNormal);
        surfPt->atInfinity = false;

        surfPt->geometricNormal = normalize(transform(RT_OBJECT_TO_WORLD, geometricNormal));
        surfPt->u = b0;
        surfPt->v = b1;
        surfPt->texCoord = texCoord;
        //surfPt->tc0Direction = normalize(transform(RT_OBJECT_TO_WORLD, uDirection));
    }

    RT_CALLABLE_PROGRAM TexCoord2D decodeTexCoord(const HitPointParameter &param) {
        const Triangle &triangle = pv_triangleBuffer[param.primIndex];
        const Vertex &v0 = pv_vertexBuffer[triangle.index0];
        const Vertex &v1 = pv_vertexBuffer[triangle.index1];
        const Vertex &v2 = pv_vertexBuffer[triangle.index2];

        float b0 = param.b0, b1 = param.b1, b2 = 1.0f - param.b0 - param.b1;
        TexCoord2D texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        return texCoord;
    }



    // Bindless
    RT_CALLABLE_PROGRAM void sampleTriangleMesh(const SurfaceLightDescriptor::Body &desc, const SurfaceLightPosSample &sample, SurfaceLightPosQueryResult* result) {
        float primProb;
        uint32_t primIdx = desc.asMeshLight.primDistribution.sample(sample.uElem, &primProb);

        const Triangle &triangle = desc.asMeshLight.triangleBuffer[primIdx];
        const Vertex &v0 = desc.asMeshLight.vertexBuffer[triangle.index0];
        const Vertex &v1 = desc.asMeshLight.vertexBuffer[triangle.index1];
        const Vertex &v2 = desc.asMeshLight.vertexBuffer[triangle.index2];

        Normal3D geometricNormal = cross(v1.position - v0.position, v2.position - v0.position);
        float area = geometricNormal.length() / 2;
        geometricNormal /= 2 * area;

        float b0, b1, b2;
        uniformSampleTriangle(sample.uPos[0], sample.uPos[1], &b0, &b1);
        b2 = 1.0f - b0 - b1;

        Point3D position = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        Normal3D shadingNormal = normalize(b0 * v0.normal + b1 * v1.normal + b2 * v2.normal);
        Vector3D shadingTangent = normalize(b0 * v0.tangent + b1 * v1.tangent + b2 * v2.tangent);
        TexCoord2D texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        ReferenceFrame shadingFrame;
        shadingFrame.z = shadingNormal;
        shadingFrame.x = shadingTangent;
        // JP: 法線と接線が直交することを保証する。
        //     直交性の消失は重心座標補間によっておこる？
        // EN: guarantee the orthogonality between the normal and tangent.
        //     Orthogonality break might be caused by barycentric interpolation?
        float dotNT = dot(shadingFrame.z, shadingFrame.x);
        if (std::fabs(dotNT) >= 0.01f)
            shadingFrame.x = normalize(shadingFrame.x - dotNT * shadingFrame.z);
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        result->surfPt.position = position;
        result->surfPt.shadingFrame = shadingFrame;
        result->surfPt.atInfinity = false;

        result->surfPt.geometricNormal = normalize(desc.asMeshLight.orientation.toMatrix3x3() * geometricNormal);
        result->surfPt.u = b0;
        result->surfPt.v = b1;
        result->surfPt.texCoord = texCoord;

        result->areaPDF = primProb / area;
        result->posType = DirectionType::LowFreq();
    }
}
