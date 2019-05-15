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
    CHECK(vlrLinearImage2DCreate(context, &imageA, NULL, 128, 128, VLRDataFormat_RGBA8x4, VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma));

    uint32_t width;
    CHECK(vlrImage2DGetWidth(imageA, &width));

    VLRImage2DTextureShaderNode imageNode;
    CHECK(vlrImage2DTextureShaderNodeCreate(context, &imageNode));
    CHECK(vlrImage2DTextureShaderNodeSetImage(imageNode, imageA));

    VLRUE4SurfaceMaterial ue4Mat;
    CHECK(vlrUE4SurfaceMaterialCreate(context, &ue4Mat));

    VLRShaderNodeSocket socketImageNodeSpectrum;
    CHECK(vlrShaderNodeGetSocket(imageNode, VLRShaderNodeSocketType_Spectrum, 0, &socketImageNodeSpectrum));
    CHECK(vlrUE4SufaceMaterialSetNodeBaseColor(ue4Mat, socketImageNodeSpectrum));
}
