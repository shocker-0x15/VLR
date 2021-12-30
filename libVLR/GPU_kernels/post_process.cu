#define RT_PIPELINE_LAUNCH_PARAMETERS extern "C" __constant__
#include "../shared/shared.h"

namespace vlr {
    using namespace shared;

    CUDA_DEVICE_KERNEL void resetAtomicAccumBuffer(
        DiscretizedSpectrum* atomicAccumBuffer,
        uint2 imageSize, uint32_t imageStrideInPixels) {
        uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                       blockDim.y * blockIdx.y + threadIdx.y);
        if (launchIndex.x >= imageSize.x || launchIndex.y >= imageSize.y)
            return;
        uint32_t linearIndex = launchIndex.y * imageStrideInPixels + launchIndex.x;
        atomicAccumBuffer[linearIndex] = DiscretizedSpectrum::Zero();
    }

    CUDA_DEVICE_KERNEL void accumulateFromAtomicAccumBuffer(
        const DiscretizedSpectrum* atomicAccumBuffer,
        optixu::BlockBuffer2D<SpectrumStorage, 0> accumBuffer,
        uint2 imageSize, uint32_t imageStrideInPixels, uint32_t numAccumFrames) {
        uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                       blockDim.y * blockIdx.y + threadIdx.y);
        if (launchIndex.x >= imageSize.x || launchIndex.y >= imageSize.y)
            return;
        uint32_t linearIndex = launchIndex.y * imageStrideInPixels + launchIndex.x;
        const DiscretizedSpectrum &srcValue = atomicAccumBuffer[linearIndex];
        if (numAccumFrames == 1)
            accumBuffer[launchIndex].reset();
        accumBuffer[launchIndex].add(srcValue);
    }

    CUDA_DEVICE_KERNEL void copyBuffers(const optixu::BlockBuffer2D<SpectrumStorage, 0> accumBuffer,
                                        const DiscretizedSpectrum* accumAlbedoBuffer,
                                        const Normal3D* accumNormalBuffer,
                                        Quaternion invOrientation,
                                        uint2 imageSize, uint32_t imageStrideInPixels,
                                        uint32_t numAccumFrames,
                                        float4* linearColorBuffer,
                                        float4* linearAlbedoBuffer,
                                        float4* linearNormalBuffer) {
        uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                       blockDim.y * blockIdx.y + threadIdx.y);
        uint32_t linearIndex = launchIndex.y * imageStrideInPixels + launchIndex.x;

        if (launchIndex.x >= imageSize.x || launchIndex.y >= imageSize.y)
            return;

        float recNumAccums = 1.0f / numAccumFrames;
        const DiscretizedSpectrum &color = accumBuffer[launchIndex].getValue().result;
        float colorXYZ[3];
        color.toXYZ(colorXYZ);
        colorXYZ[0] *= recNumAccums;
        colorXYZ[1] *= recNumAccums;
        colorXYZ[2] *= recNumAccums;
        VLRAssert(colorXYZ[0] >= 0.0f && colorXYZ[1] >= 0.0f && colorXYZ[2] >= 0.0f,
                  "each value of color XYZ must not be negative.");
        float colorRGB[3];
        transformTristimulus(mat_XYZ_to_Rec709_D65, colorXYZ, colorRGB);

        const DiscretizedSpectrum &albedo = accumAlbedoBuffer[linearIndex];
        float albedoXYZ[3];
        albedo.toXYZ(albedoXYZ);
        albedoXYZ[0] *= recNumAccums;
        albedoXYZ[1] *= recNumAccums;
        albedoXYZ[2] *= recNumAccums;
        VLRAssert(albedoXYZ[0] >= 0.0f && albedoXYZ[1] >= 0.0f && albedoXYZ[2] >= 0.0f,
                  "each value of albedo XYZ must not be negative.");
        float albedoRGB[3];
        transformTristimulus(mat_XYZ_to_Rec709_D65, albedoXYZ, albedoRGB);

        Normal3D normal = accumNormalBuffer[linearIndex];
        normal = invOrientation.toMatrix3x3() * normal;
        normal.x *= -1;
        if (normal.x != 0 || normal.y != 0 || normal.z != 0)
            normal.normalize();

        linearColorBuffer[linearIndex] = make_float4(colorRGB[0], colorRGB[1], colorRGB[2], 1.0f);
        linearAlbedoBuffer[linearIndex] = make_float4(albedoRGB[0], albedoRGB[1], albedoRGB[2], 1.0f);
        linearNormalBuffer[linearIndex] = make_float4(normal.x, normal.y, normal.z, 1.0f);
    }

    CUDA_DEVICE_KERNEL void convertToRGB(const optixu::BlockBuffer2D<SpectrumStorage, 0> accumBuffer,
                                         const float4* linearDenoisedColorBuffer,
                                         const float4* linearAlbedoBuffer,
                                         const float4* linearNormalBuffer,
                                         bool useDenoiser, bool debugRender, DebugRenderingAttribute debugAttr,
                                         uint2 imageSize, uint32_t imageStrideInPixels, uint32_t numAccumFrames,
                                         optixu::NativeBlockBuffer2D<float4> outputBuffer) {
        uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                       blockDim.y * blockIdx.y + threadIdx.y);
        if (launchIndex.x >= imageSize.x || launchIndex.y >= imageSize.y)
            return;

        float RGB[3];
        if (useDenoiser) {
            uint32_t linearIndex = launchIndex.y * imageStrideInPixels + launchIndex.x;
            float4 value;
            if (debugRender) {
                switch (debugAttr) {
                case DebugRenderingAttribute::DenoiserAlbedo:
                    value = linearAlbedoBuffer[linearIndex];
                    break;
                case DebugRenderingAttribute::DenoiserNormal:
                    value = linearNormalBuffer[linearIndex];
                    value = make_float4(0.5f * value.x + 0.5f,
                                        0.5f * value.y + 0.5f,
                                        0.5f * value.z + 0.5f,
                                        value.w);
                    break;
                }
            }
            else {
                value = linearDenoisedColorBuffer[linearIndex];
            }
            RGB[0] = value.x;
            RGB[1] = value.y;
            RGB[2] = value.z;
        }
        else {
            const DiscretizedSpectrum &spectrum = accumBuffer[launchIndex].getValue().result;
            float XYZ[3];
            spectrum.toXYZ(XYZ);
            float recNumAccums = 1.0f / numAccumFrames;
            XYZ[0] *= recNumAccums;
            XYZ[1] *= recNumAccums;
            XYZ[2] *= recNumAccums;
            VLRAssert(XYZ[0] >= 0.0f && XYZ[1] >= 0.0f && XYZ[2] >= 0.0f,
                      "each value of XYZ must not be negative (%g, %g, %g).",
                      XYZ[0], XYZ[1], XYZ[2]);
            transformTristimulus(mat_XYZ_to_Rec709_D65, XYZ, RGB);
        }

        outputBuffer.write(launchIndex, make_float4(RGB[0], RGB[1], RGB[2], 1.0f)); // not clamp out of gamut color.
    }
}
