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

    VLRContext context;
    CHECK(vlrCreateContext(&context, true, true, 8, 0, NULL, 0));

    VLRLinearImage2D imageA;
    CHECK(vlrLinearImage2DCreate(context, &imageA, NULL, 128, 128, "RGBA8x4", "Reflectance", "Rec709(D65) sRGB Gamma"));

    uint32_t width;
    CHECK(vlrImage2DGetWidth(imageA, &width));

    VLRShaderNode imageNode;
    CHECK(vlrShaderNodeCreate(context, "Image2DTexture", &imageNode));
    CHECK(vlrShaderNodeSetImage(imageNode, imageA));

    VLRSurfaceMaterial ue4Mat;
    CHECK(vlrSurfaceMaterialCreate(context, "UE4", &ue4Mat));

    VLRShaderNodePlug plugImageNodeSpectrum;
    CHECK(vlrShaderNodeGetPlug(imageNode, VLRShaderNodePlugType_Spectrum, 0, &plugImageNodeSpectrum));
    CHECK(vlrConnectableSetShaderNodePlug(ue4Mat, "base color", plugImageNodeSpectrum));
}
