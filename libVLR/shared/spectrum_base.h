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

    template <typename RealType>
    static constexpr void calcInverse3x3Matrix(const RealType mat[9], RealType invMat[9]) {
        RealType det = (mat[3 * 0 + 0] * mat[3 * 1 + 1] * mat[3 * 2 + 2] +
                        mat[3 * 1 + 0] * mat[3 * 2 + 1] * mat[3 * 0 + 2] +
                        mat[3 * 2 + 0] * mat[3 * 0 + 1] * mat[3 * 1 + 2]) -
                       (mat[3 * 2 + 0] * mat[3 * 1 + 1] * mat[3 * 0 + 2] +
                        mat[3 * 1 + 0] * mat[3 * 0 + 1] * mat[3 * 2 + 2] +
                        mat[3 * 0 + 0] * mat[3 * 2 + 1] * mat[3 * 1 + 2]);
        RealType recDet = 1 / det;
        invMat[3 * 0 + 0] = recDet * (mat[3 * 1 + 1] * mat[3 * 2 + 2] - mat[3 * 2 + 1] * mat[3 * 1 + 2]);
        invMat[3 * 0 + 1] = -recDet * (mat[3 * 0 + 1] * mat[3 * 2 + 2] - mat[3 * 2 + 1] * mat[3 * 0 + 2]);
        invMat[3 * 0 + 2] = recDet * (mat[3 * 0 + 1] * mat[3 * 1 + 2] - mat[3 * 1 + 1] * mat[3 * 0 + 2]);
        invMat[3 * 1 + 0] = -recDet * (mat[3 * 1 + 0] * mat[3 * 2 + 2] - mat[3 * 2 + 0] * mat[3 * 1 + 2]);
        invMat[3 * 1 + 1] = recDet * (mat[3 * 0 + 0] * mat[3 * 2 + 2] - mat[3 * 2 + 0] * mat[3 * 0 + 2]);
        invMat[3 * 1 + 2] = -recDet * (mat[3 * 0 + 0] * mat[3 * 1 + 2] - mat[3 * 1 + 0] * mat[3 * 0 + 2]);
        invMat[3 * 2 + 0] = recDet * (mat[3 * 1 + 0] * mat[3 * 2 + 1] - mat[3 * 2 + 0] * mat[3 * 1 + 1]);
        invMat[3 * 2 + 1] = -recDet * (mat[3 * 0 + 0] * mat[3 * 2 + 1] - mat[3 * 2 + 0] * mat[3 * 0 + 1]);
        invMat[3 * 2 + 2] = recDet * (mat[3 * 0 + 0] * mat[3 * 1 + 1] - mat[3 * 1 + 0] * mat[3 * 0 + 1]);
    }

    template <typename RealType>
    static constexpr void multiply3x3Matrix(const RealType matA[9], const RealType matB[9], RealType matC[9]) {
        matC[3 * 0 + 0] = matA[3 * 0 + 0] * matB[3 * 0 + 0] + matA[3 * 1 + 0] * matB[3 * 0 + 1] + matA[3 * 2 + 0] * matB[3 * 0 + 2];
        matC[3 * 0 + 1] = matA[3 * 0 + 1] * matB[3 * 0 + 0] + matA[3 * 1 + 1] * matB[3 * 0 + 1] + matA[3 * 2 + 1] * matB[3 * 0 + 2];
        matC[3 * 0 + 2] = matA[3 * 0 + 2] * matB[3 * 0 + 0] + matA[3 * 1 + 2] * matB[3 * 0 + 1] + matA[3 * 2 + 2] * matB[3 * 0 + 2];
        matC[3 * 1 + 0] = matA[3 * 0 + 0] * matB[3 * 1 + 0] + matA[3 * 1 + 0] * matB[3 * 1 + 1] + matA[3 * 2 + 0] * matB[3 * 1 + 2];
        matC[3 * 1 + 1] = matA[3 * 0 + 1] * matB[3 * 1 + 0] + matA[3 * 1 + 1] * matB[3 * 1 + 1] + matA[3 * 2 + 1] * matB[3 * 1 + 2];
        matC[3 * 1 + 2] = matA[3 * 0 + 2] * matB[3 * 1 + 0] + matA[3 * 1 + 2] * matB[3 * 1 + 1] + matA[3 * 2 + 2] * matB[3 * 1 + 2];
        matC[3 * 2 + 0] = matA[3 * 0 + 0] * matB[3 * 2 + 0] + matA[3 * 1 + 0] * matB[3 * 2 + 1] + matA[3 * 2 + 0] * matB[3 * 2 + 2];
        matC[3 * 2 + 1] = matA[3 * 0 + 1] * matB[3 * 2 + 0] + matA[3 * 1 + 1] * matB[3 * 2 + 1] + matA[3 * 2 + 1] * matB[3 * 2 + 2];
        matC[3 * 2 + 2] = matA[3 * 0 + 2] * matB[3 * 2 + 0] + matA[3 * 1 + 2] * matB[3 * 2 + 1] + matA[3 * 2 + 2] * matB[3 * 2 + 2];
    }

    // http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    template <typename RealType>
    static constexpr void calc_mat_RGB_to_XYZ(RealType xr, RealType yr, RealType xg, RealType yg, RealType xb, RealType yb, RealType xw, RealType yw,
                                              RealType mat[9]) {
        RealType Xr = xr / yr, Yr = 1, Zr = (1 - xr - yr) / yr;
        RealType Xg = xg / yg, Yg = 1, Zg = (1 - xg - yg) / yg;
        RealType Xb = xb / yb, Yb = 1, Zb = (1 - xb - yb) / yb;

        RealType Xw = xw / yw, Yw = 1, Zw = (1 - xw - yw) / yw;

        RealType m[] = { Xr, Yr, Zr, Xg, Yg, Zg, Xb, Yb, Zb };
        RealType invM[9];
        calcInverse3x3Matrix(m, invM);
        RealType Sr = invM[3 * 0 + 0] * Xw + invM[3 * 1 + 0] * Yw + invM[3 * 2 + 0] * Zw;
        RealType Sg = invM[3 * 0 + 1] * Xw + invM[3 * 1 + 1] * Yw + invM[3 * 2 + 1] * Zw;
        RealType Sb = invM[3 * 0 + 2] * Xw + invM[3 * 1 + 2] * Yw + invM[3 * 2 + 2] * Zw;

        mat[0] = Sr * Xr; mat[3] = Sg * Xg; mat[6] = Sb * Xb;
        mat[1] = Sr * Yr; mat[4] = Sg * Yg; mat[7] = Sb * Yb;
        mat[2] = Sr * Zr; mat[5] = Sg * Zg; mat[8] = Sb * Zb;
    }

    // These matrices are column-major.
    RT_VARIABLE HOST_STATIC_CONSTEXPR float mat_Rec709_D65_to_XYZ[] = {
        0.4124564, 0.2126729, 0.0193339,
        0.3575761, 0.7151522, 0.1191920,
        0.1804375, 0.0721750, 0.9503041,
    };
    RT_VARIABLE HOST_STATIC_CONSTEXPR float mat_XYZ_to_Rec709_D65[] = {
        3.2404542, -0.9692660, 0.0556434,
        -1.5371385 , 1.8760108, -0.2040259,
        -0.4985314, 0.0415560, 1.0572252,
    };
    RT_VARIABLE HOST_STATIC_CONSTEXPR float mat_Rec709_E_to_XYZ[] = {
        0.4969, 0.2562, 0.0233,
        0.3391, 0.6782, 0.1130,
        0.1640, 0.0656, 0.8637,
    };
    RT_VARIABLE HOST_STATIC_CONSTEXPR float mat_XYZ_to_Rec709_E[] = {
        2.6897, -1.0221, 0.0612,
        -1.2759, 1.9783, -0.2245,
        -0.4138, 0.0438, 1.1633,
    };

    template <typename RealType>
    RT_FUNCTION constexpr void transformTristimulus(const float matColMajor[9], const RealType src[3], RealType dst[3]) {
        dst[0] = matColMajor[0] * src[0] + matColMajor[3] * src[1] + matColMajor[6] * src[2];
        dst[1] = matColMajor[1] * src[0] + matColMajor[4] * src[1] + matColMajor[7] * src[2];
        dst[2] = matColMajor[2] * src[0] + matColMajor[5] * src[1] + matColMajor[8] * src[2];
    }

    template <typename RealType>
    RT_FUNCTION constexpr void transformToRenderingRGB(VLRSpectrumType spectrumType, const RealType XYZ[3], RealType RGB[3]) {
        switch (spectrumType) {
        case VLRSpectrumType_Reflectance:
        case VLRSpectrumType_IndexOfRefraction:
        case VLRSpectrumType_NA:
            transformTristimulus(mat_XYZ_to_Rec709_E, XYZ, RGB);
            break;
        case VLRSpectrumType_LightSource:
            transformTristimulus(mat_XYZ_to_Rec709_D65, XYZ, RGB);
            break;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    template <typename RealType>
    RT_FUNCTION constexpr void transformFromRenderingRGB(VLRSpectrumType spectrumType, const RealType RGB[3], RealType XYZ[3]) {
        switch (spectrumType) {
        case VLRSpectrumType_Reflectance:
        case VLRSpectrumType_IndexOfRefraction:
        case VLRSpectrumType_NA:
            transformTristimulus(mat_Rec709_E_to_XYZ, RGB, XYZ);
            break;
        case VLRSpectrumType_LightSource:
            transformTristimulus(mat_Rec709_D65_to_XYZ, RGB, XYZ);
            break;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    template <typename RealType>
    inline constexpr void transformToRenderingRGB(VLRSpectrumType spectrumType, VLRColorSpace srcSpace, const RealType src[3], RealType dstRGB[3]) {
        RealType srcTriplet[3] = { src[0], src[1], src[2] };
        switch (srcSpace) {
        case VLRColorSpace_Rec709_D65_sRGBGamma:
            dstRGB[0] = sRGB_degamma(srcTriplet[0]);
            dstRGB[1] = sRGB_degamma(srcTriplet[1]);
            dstRGB[2] = sRGB_degamma(srcTriplet[2]);
            break;
        case VLRColorSpace_Rec709_D65:
            dstRGB[0] = srcTriplet[0];
            dstRGB[1] = srcTriplet[1];
            dstRGB[2] = srcTriplet[2];
            break;
        case VLRColorSpace_xyY: {
            if (srcTriplet[1] == 0) {
                dstRGB[0] = dstRGB[1] = dstRGB[2] = 0.0f;
                break;
            }
            RealType z = 1 - (srcTriplet[0] + srcTriplet[1]);
            RealType b = srcTriplet[2] / srcTriplet[1];
            srcTriplet[0] = srcTriplet[0] * b;
            srcTriplet[1] = srcTriplet[2];
            srcTriplet[2] = z * b;
            // pass to XYZ
        }
        case VLRColorSpace_XYZ:
            switch (spectrumType) {
            case VLRSpectrumType_Reflectance:
            case VLRSpectrumType_IndexOfRefraction:
            case VLRSpectrumType_NA:
                transformTristimulus(mat_XYZ_to_Rec709_E, srcTriplet, dstRGB);
                break;
            case VLRSpectrumType_LightSource:
                transformTristimulus(mat_XYZ_to_Rec709_D65, srcTriplet, dstRGB);
                break;
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
            break;
        default:
            break;
        }
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
        case VLRColorSpace_Rec709_D65_sRGBGamma:
            VLRAssert_NotImplemented();
            break;
        case VLRColorSpace_Rec709_D65:
            VLRAssert_NotImplemented();
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
