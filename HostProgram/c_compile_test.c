#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include <VLR/VLR.h>

// Just for compile test for C language.

static void C_CompileTest() {
#define CHECK(call) \
    do { \
        VLRResult res = call; \
        assert(res == VLRResult_NoError); \
    } while (0)

    CUcontext cuContext;
    int32_t cuDeviceCount;
    cuInit(0);
    cuDeviceGetCount(&cuDeviceCount);
    cuCtxCreate(&cuContext, 0, 0);
    cuCtxSetCurrent(cuContext);

    VLRContext context;
    CHECK(vlrCreateContext(
        cuContext, true, 8, &context));

    VLRLinearImage2D imageA;
    CHECK(vlrLinearImage2DCreate(
        context,
        NULL, 128, 128, "RGBA8x4", "Reflectance", "Rec709(D65) sRGB Gamma",
        &imageA));

    uint32_t width;
    CHECK(vlrImage2DGetWidth(imageA, &width));

    VLRShaderNode imageNode;
    CHECK(vlrShaderNodeCreate(context, "Image2DTexture", &imageNode));
    CHECK(vlrShaderNodeSetImage(imageNode, imageA));

    VLRSurfaceMaterial ue4Mat;
    CHECK(vlrSurfaceMaterialCreate(context, "UE4", &ue4Mat));

    VLRShaderNodePlug plugImageNodeSpectrum;
    CHECK(vlrShaderNodeGetPlug(imageNode, VLRShaderNodePlugType_Spectrum, 0, &plugImageNodeSpectrum));
    CHECK(vlrQueryableSetShaderNodePlug(ue4Mat, "base color", plugImageNodeSpectrum));
}
