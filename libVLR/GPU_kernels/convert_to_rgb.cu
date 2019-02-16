#include "../shared/shared.h"

namespace VLR {
    rtDeclareVariable(optix::uint2, sm_launchIndex, rtLaunchIndex, );

    rtDeclareVariable(uint32_t, pv_numAccumFrames, , );

    rtBuffer<SpectrumStorage, 2> pv_spectrumBuffer;
    rtBuffer<RGBSpectrum, 2> pv_RGBBuffer;

    // Ray Generation Program
    // TODO: port this kernel to ordinary CUDA kernel.
    RT_PROGRAM void convertToRGB() {
        const DiscretizedSpectrum &spectrum = pv_spectrumBuffer[sm_launchIndex].getValue().result;
        float XYZ[3];
        spectrum.toXYZ(XYZ);
        VLRAssert(XYZ[0] >= 0.0f && XYZ[1] >= 0.0f && XYZ[2] >= 0.0f, "each value of XYZ must not be negative.");
        float recNumAccums = 1.0f / pv_numAccumFrames;
        XYZ[0] *= recNumAccums;
        XYZ[1] *= recNumAccums;
        XYZ[2] *= recNumAccums;
        //pv_RGBBuffer[sm_launchIndex] = RGBSpectrum(XYZ[0], XYZ[1], XYZ[2]);
        float RGB[3];
        transformTristimulus(mat_XYZ_to_Rec709_D65, XYZ, RGB);
        pv_RGBBuffer[sm_launchIndex] = RGBSpectrum(RGB[0], RGB[1], RGB[2]); // not clamp out of gamut color.
    }
}
