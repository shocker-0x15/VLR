#define RT_PIPELINE_LAUNCH_PARAMETERS extern "C" __constant__
#include "../shared/shared.h"

namespace vlr {
    using namespace shared;

    CUDA_DEVICE_KERNEL void convertToRGB(const optixu::BlockBuffer2D<SpectrumStorage, 0> spectrumBuffer,
                                         optixu::NativeBlockBuffer2D<float4> rgbBuffer,
                                         uint32_t numAccumFrames) {
        uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                       blockDim.y * blockIdx.y + threadIdx.y);
        uint2 imageSize = spectrumBuffer.getSize();
        if (launchIndex.x >= imageSize.x || launchIndex.y >= imageSize.y)
            return;
        const DiscretizedSpectrum &spectrum = spectrumBuffer[launchIndex].getValue().result;
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        //if (launchIndex.x == 640 && launchIndex.y == 128) {
        //    spectrum.print();
        //    vlrprintf("%g, %g, %g\n", XYZ[0], XYZ[1], XYZ[2]);
        //}
        VLRAssert(XYZ[0] >= 0.0f && XYZ[1] >= 0.0f && XYZ[2] >= 0.0f, "each value of XYZ must not be negative.");
        float recNumAccums = 1.0f / numAccumFrames;
        XYZ[0] *= recNumAccums;
        XYZ[1] *= recNumAccums;
        XYZ[2] *= recNumAccums;
        //pv_RGBBuffer[sm_launchIndex] = RGBSpectrum(XYZ[0], XYZ[1], XYZ[2]);
        float RGB[3];
        transformTristimulus(mat_XYZ_to_Rec709_D65, XYZ, RGB);
        rgbBuffer.write(launchIndex, make_float4(RGB[0], RGB[1], RGB[2], 1.0f)); // not clamp out of gamut color.
    }
}
