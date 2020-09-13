#define RT_PIPELINE_LAUNCH_PARAMETERS __constant__
#include "../shared/shared.h"

namespace VLR {
    using namespace Shared;

    CUDA_DEVICE_KERNEL void convertToRGB(const optixu::BlockBuffer2D<SpectrumStorage, 1> spectrumBuffer,
                                         optixu::BlockBuffer2D<RGBSpectrum, 1> rgbBuffer,
                                         uint32_t numAccumFrames) {
        uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                       blockDim.y * blockIdx.y + threadIdx.y);
        const DiscretizedSpectrum &spectrum = spectrumBuffer[launchIndex].getValue().result;
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        VLRAssert(XYZ[0] >= 0.0f && XYZ[1] >= 0.0f && XYZ[2] >= 0.0f, "each value of XYZ must not be negative.");
        float recNumAccums = 1.0f / numAccumFrames;
        XYZ[0] *= recNumAccums;
        XYZ[1] *= recNumAccums;
        XYZ[2] *= recNumAccums;
        //pv_RGBBuffer[sm_launchIndex] = RGBSpectrum(XYZ[0], XYZ[1], XYZ[2]);
        float RGB[3];
        transformTristimulus(mat_XYZ_to_Rec709_D65, XYZ, RGB);
        rgbBuffer[launchIndex] = RGBSpectrum(RGB[0], RGB[1], RGB[2]); // not clamp out of gamut color.
    }
}
