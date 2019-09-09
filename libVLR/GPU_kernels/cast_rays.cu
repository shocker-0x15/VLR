#include "light_transport_common.cuh"

namespace VLR {
    struct RayCastPayload {
        Point3D position;
        Normal3D geometricNormal;
    };

    rtDeclareVariable(RayCastPayload, sm_rayCastPayload, rtPayload, );

    // Context-scope Variables
    rtDeclareVariable(uint32_t, pv_numCastRays, , );
    rtBuffer<RayQuery, 1> pv_rayBuffer;
    rtBuffer<RayQueryResult, 1> pv_resultBuffer;




    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void castRaysClosestHit() {
        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(&surfPt, &hypAreaPDF);

        sm_rayCastPayload.position = surfPt.position;
        sm_rayCastPayload.geometricNormal = surfPt.geometricNormal;
    }



    // JP: 本当は無限大の球のIntersection/Bounding Box Programを使用して環境光に関する処理もClosest Hit Programで統一的に行いたい。
    //     が、OptiXのBVHビルダーがLBVHベースなので無限大のAABBを生成するのは危険。
    //     仕方なくMiss Programで環境光を処理する。
    RT_PROGRAM void castRaysMiss() {
        sm_rayCastPayload.position = Point3D(NAN, NAN, NAN);
        sm_rayCastPayload.geometricNormal = Normal3D(NAN, NAN, NAN);
    }



    // Common Ray Generation Program for All Camera Types
    RT_PROGRAM void castRaysRayGeneration() {
        const RayQuery &query = pv_rayBuffer[sm_launchIndex.x];

        optix::Ray ray = optix::make_Ray(asOptiXType(query.origin), asOptiXType(query.direction), RayType::RayCast, 0.0f, FLT_MAX);

        RayCastPayload payload;
        rtTrace(pv_topGroup, ray, payload);

        pv_resultBuffer[sm_launchIndex.x] = RayQueryResult{ payload.position, payload.geometricNormal };
    }



    // Exception Program
    RT_PROGRAM void castRaysException() {
        //uint32_t code = rtGetExceptionCode();
        rtPrintExceptionDetails();
    }
}
