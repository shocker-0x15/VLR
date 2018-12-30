#pragma once

#include "../include/VLR/public_types.h"
#include "common_internal.h"

namespace VLR {
    static constexpr float WavelengthLowBound = 360.0f;
    static constexpr float WavelengthHighBound = 830.0f;
    static constexpr uint32_t NumCMFSamples = 471;

#if defined(VLR_Host)
    extern const float xbar_CIE1931_2deg[NumCMFSamples];
    extern const float ybar_CIE1931_2deg[NumCMFSamples];
    extern const float zbar_CIE1931_2deg[NumCMFSamples];
    extern float integralCMF;

    void initializeColorSystem();
#endif

    template <typename RealType>
    RT_FUNCTION constexpr RealType sRGB_gamma(RealType value) {
        VLRAssert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
        if (value <= (RealType)0.0031308)
            return (RealType)12.92 * value;
        return (RealType)1.055 * std::pow(value, (RealType)(1.0 / 2.4)) - (RealType)0.055;
    }

    template <typename RealType>
    RT_FUNCTION constexpr RealType sRGB_degamma(RealType value) {
        VLRAssert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
        if (value <= (RealType)0.04045)
            return value / (RealType)12.92;
        return std::pow((RealType)(value + 0.055) / (RealType)1.055, (RealType)2.4);
    }

    // TODO: implement a method to generate arbitrary XYZ<->RGB matrices.
    // http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    //template <typename RealType>
    //static void RGB_to_XYZ(RealType xR, RealType yR, RealType xG, RealType yG, RealType xB, RealType yB, RealType xW, RealType yW) {
    //    RealType XR = xR / yR, YR = 1, ZR = (1 - xR - yR) / yR;
    //    RealType XG = xG / yG, YG = 1, ZG = (1 - xG - yG) / yG;
    //    RealType XB = xB / yB, YB = 1, ZB = (1 - xB - yB) / yB;
    //}

    RT_VARIABLE static HOST_CONSTEXPR float mat_sRGB_D65_to_XYZ[] = {
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041,
    };
    RT_VARIABLE static HOST_CONSTEXPR float mat_XYZ_to_sRGB_D65[] = {
        3.2404542, -1.5371385, -0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
        0.0556434, -0.2040259, 1.0572252,
    };
    RT_VARIABLE static HOST_CONSTEXPR float mat_sRGB_E_to_XYZ[] = {
        0.4969, 0.3391, 0.1640,
        0.2562, 0.6782, 0.0656,
        0.0233, 0.1130, 0.8637,
    };
    RT_VARIABLE static HOST_CONSTEXPR float mat_XYZ_to_sRGB_E[] = {
        2.6897, -1.2759, -0.4138,
        -1.0221, 1.9783, 0.0438,
        0.0612, -0.2245, 1.1633,
    };

    template <typename RealType>
    RT_FUNCTION constexpr void transformTristimulus(const float mat[9], const RealType src[3], RealType dst[3]) {
        dst[0] = mat[0] * src[0] + mat[1] * src[1] + mat[2] * src[2];
        dst[1] = mat[3] * src[0] + mat[4] * src[1] + mat[5] * src[2];
        dst[2] = mat[6] * src[0] + mat[7] * src[1] + mat[8] * src[2];
    }

    template <typename RealType>
    RT_FUNCTION constexpr void XYZ_to_xyY(const RealType xyz[3], RealType xyY[3]) {
        RealType b = xyz[0] + xyz[1] + xyz[2];
        if (b == 0) {
            xyY[0] = xyY[1] = (RealType)(1.0 / 3.0);
            xyY[2] = 0;
            return;
        }
        xyY[0] = xyz[0] / b;
        xyY[1] = xyz[1] / b;
        xyY[2] = xyz[1];
    }

    template <typename RealType>
    RT_FUNCTION constexpr void xyY_to_XYZ(const RealType xyY[3], RealType xyz[3]) {
        RealType b = xyY[2] / xyY[1];
        xyz[0] = xyY[0] * b;
        xyz[1] = xyY[2];
        xyz[2] = (1 - xyY[0] - xyY[1]) * b;
    }

    template <typename RealType>
    constexpr RealType calcLuminance(VLRColorSpace colorSpace, RealType e0, RealType e1, RealType e2) {
        switch (colorSpace) {
        case VLRColorSpace_Rec709_sRGBGamma:
            VLRAssert_NotImplemented();
            break;
        case VLRColorSpace_Rec709:
            break;
        case VLRColorSpace_XYZ:
            return e1;
        case VLRColorSpace_xyY:
            return e2;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
        return 0.0f;
    }
}
