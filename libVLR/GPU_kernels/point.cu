#include "../shared/kernel_common.h"

namespace vlr {
    using namespace shared;

    RT_CALLABLE_PROGRAM void RT_DC_NAME(samplePoint)(
        uint32_t instIndex, uint32_t geomInstIndex,
        const SurfaceLightPosSample &sample, const Point3D &shadingPoint,
        SurfaceLightPosQueryResult* result) {
        (void)shadingPoint;

        const Instance &inst = plp.instBuffer[instIndex];
        const GeometryInstance &geomInst = plp.geomInstBuffer[geomInstIndex];

        float primProb;
        uint32_t primIdx = geomInst.asPoints.primDistribution.sample(sample.uElem, &primProb);
        //printf("%g, %u, %g\n", sample.uElem, primIdx, primProb);

        uint32_t pointIndex = geomInst.asPoints.indexBuffer[primIdx];
        const Vertex &v = geomInst.asPoints.vertexBuffer[pointIndex];

        const StaticTransform &transform = inst.transform;

        // area = 4 * pi * r^2
        constexpr float area = 4 * VLR_M_PI/* * pow2(0.0f)*/;
        result->areaPDF = primProb / area;
        result->posType = DirectionType::Emission() | DirectionType::Delta0D();
        result->materialIndex = geomInst.materialIndex;

        Point3D position = transform * v.position;
        Normal3D shadingNormal = normalize(transform * v.normal);
        Vector3D tc0Direction = normalize(transform * v.tc0Direction);

        SurfacePoint &surfPt = result->surfPt;

        surfPt.position = position;
        surfPt.shadingFrame = ReferenceFrame(tc0Direction, shadingNormal);
        surfPt.isPoint = true;
        surfPt.atInfinity = false;
        surfPt.geometricNormal = shadingNormal;
        surfPt.u = 0.0f;
        surfPt.v = 0.0f;
        surfPt.texCoord = TexCoord2D(0.0f, 0.0f);
    }
}
