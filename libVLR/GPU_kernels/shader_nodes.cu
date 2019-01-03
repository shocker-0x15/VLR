#include "kernel_common.cuh"

namespace VLR {
    RT_CALLABLE_PROGRAM Point3D GeometryShaderNode_Point3D(const uint32_t* rawNodeData, uint32_t option,
                                                           const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const GeometryShaderNode*)rawNodeData;
        return surfPt.position;
    }

    RT_CALLABLE_PROGRAM Normal3D GeometryShaderNode_Normal3D(const uint32_t* rawNodeData, uint32_t option,
                                                             const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const GeometryShaderNode*)rawNodeData;
        if (option == 0)
            return surfPt.geometricNormal;
        else if (option == 1)
            return surfPt.shadingFrame.z;
        return Normal3D(0, 0, 0);
    }

    RT_CALLABLE_PROGRAM Vector3D GeometryShaderNode_Vector3D(const uint32_t* rawNodeData, uint32_t option,
                                                             const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const GeometryShaderNode*)rawNodeData;
        if (option == 0)
            return surfPt.shadingFrame.x;
        else if (option == 1)
            return surfPt.shadingFrame.y;
        return Vector3D::Zero();
    }

    RT_CALLABLE_PROGRAM Point3D GeometryShaderNode_textureCoordinates(const uint32_t* rawNodeData, uint32_t option,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const GeometryShaderNode*)rawNodeData;
        return Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0);
    }



    RT_CALLABLE_PROGRAM float FloatShaderNode_float(const uint32_t* rawNodeData, uint32_t option,
                                                    const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const FloatShaderNode*)rawNodeData;
        return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
    }



    RT_CALLABLE_PROGRAM float Float2ShaderNode_float(const uint32_t* rawNodeData, uint32_t option,
                                                     const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float2ShaderNode*)rawNodeData;
        if (option == 0)
            return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
        else if (option == 1)
            return calcNode(nodeData.node1, nodeData.imm1, surfPt, wls);
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Float2ShaderNode_float2(const uint32_t* rawNodeData, uint32_t option,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float2ShaderNode*)rawNodeData;
        return optix::make_float2(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                  calcNode(nodeData.node1, nodeData.imm1, surfPt, wls));
    }



    RT_CALLABLE_PROGRAM float Float3ShaderNode_float(const uint32_t* rawNodeData, uint32_t option,
                                                     const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float3ShaderNode*)rawNodeData;
        if (option == 0)
            return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
        else if (option == 1)
            return calcNode(nodeData.node1, nodeData.imm1, surfPt, wls);
        else if (option == 2)
            return calcNode(nodeData.node2, nodeData.imm2, surfPt, wls);
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Float3ShaderNode_float2(const uint32_t* rawNodeData, uint32_t option,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float3ShaderNode*)rawNodeData;
        if (option == 0)
            return optix::make_float2(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                      calcNode(nodeData.node1, nodeData.imm1, surfPt, wls));
        else if (option == 1)
            return optix::make_float2(calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
        return optix::make_float2(0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float3 Float3ShaderNode_float3(const uint32_t* rawNodeData, uint32_t option, 
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float3ShaderNode*)rawNodeData;
        return optix::make_float3(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                  calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                  calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
    }



    RT_CALLABLE_PROGRAM float Float4ShaderNode_float(const uint32_t* rawNodeData, uint32_t option,
                                                     const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float4ShaderNode*)rawNodeData;
        if (option == 0)
            return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
        else if (option == 1)
            return calcNode(nodeData.node1, nodeData.imm1, surfPt, wls);
        else if (option == 2)
            return calcNode(nodeData.node2, nodeData.imm2, surfPt, wls);
        else if (option == 3)
            return calcNode(nodeData.node3, nodeData.imm3, surfPt, wls);
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Float4ShaderNode_float2(const uint32_t* rawNodeData, uint32_t option,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float4ShaderNode*)rawNodeData;
        if (option == 0)
            return optix::make_float2(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                      calcNode(nodeData.node1, nodeData.imm1, surfPt, wls));
        else if (option == 1)
            return optix::make_float2(calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
        else if (option == 2)
            return optix::make_float2(calcNode(nodeData.node2, nodeData.imm2, surfPt, wls),
                                      calcNode(nodeData.node3, nodeData.imm3, surfPt, wls));
        return optix::make_float2(0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float3 Float4ShaderNode_float3(const uint32_t* rawNodeData, uint32_t option,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float4ShaderNode*)rawNodeData;
        if (option == 0)
            return optix::make_float3(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                      calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
        else if (option == 1)
            return optix::make_float3(calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls),
                                      calcNode(nodeData.node3, nodeData.imm3, surfPt, wls));
        return optix::make_float3(0.0f, 0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float4 Float4ShaderNode_float4(const uint32_t* rawNodeData, uint32_t option,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Float4ShaderNode*)rawNodeData;
        return optix::make_float4(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                  calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                  calcNode(nodeData.node2, nodeData.imm2, surfPt, wls),
                                  calcNode(nodeData.node3, nodeData.imm3, surfPt, wls));
    }



    RT_CALLABLE_PROGRAM SampledSpectrum TripletSpectrumShaderNode_spectrum(const uint32_t* rawNodeData, uint32_t option,
                                                                           const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const TripletSpectrumShaderNode*)rawNodeData;
        return nodeData.value.evaluate(wls);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum RegularSampledSpectrumShaderNode_spectrum(const uint32_t* rawNodeData, uint32_t option,
                                                                                  const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const RegularSampledSpectrumShaderNode*)rawNodeData;
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return RegularSampledSpectrum(nodeData.minLambda, nodeData.maxLambda, nodeData.values, nodeData.numSamples).evaluate(wls);
#else
        return nodeData.value.evaluate(wls);
#endif
    }

    RT_CALLABLE_PROGRAM SampledSpectrum IrregularSampledSpectrumShaderNode_spectrum(const uint32_t* rawNodeData, uint32_t option,
                                                                                    const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const IrregularSampledSpectrumShaderNode*)rawNodeData;
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return IrregularSampledSpectrum(nodeData.lambdas, nodeData.values, nodeData.numSamples).evaluate(wls);
#else
        return nodeData.value.evaluate(wls);
#endif
    }



    RT_CALLABLE_PROGRAM SampledSpectrum Vector3DToSpectrumShaderNode_spectrum(const uint32_t* rawNodeData, uint32_t option,
                                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Vector3DToSpectrumShaderNode*)rawNodeData;
        Vector3D vector = calcNode(nodeData.nodeVector3D, nodeData.immVector3D, surfPt, wls);
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(nodeData.spectrumType, nodeData.colorSpace,
                                 clamp(0.5f * vector.x + 0.5f, 0.0f, 1.0f),
                                 clamp(0.5f * vector.y + 0.5f, 0.0f, 1.0f),
                                 clamp(0.5f * vector.z + 0.5f, 0.0f, 1.0f)).evaluate(wls);
#else
        return SampledSpectrum(clamp(0.5f * vector.x + 0.5f, 0.0f, 1.0f),
                               clamp(0.5f * vector.y + 0.5f, 0.0f, 1.0f),
                               clamp(0.5f * vector.z + 0.5f, 0.0f, 1.0f));
#endif
    }



    RT_CALLABLE_PROGRAM Point3D OffsetAndScaleUVTextureMap2DShaderNode_textureCoordinates(const uint32_t* rawNodeData, uint32_t option,
                                                                                          const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const OffsetAndScaleUVTextureMap2DShaderNode*)rawNodeData;
        return Point3D(nodeData.scale[0] * surfPt.texCoord.u + nodeData.offset[0],
                       nodeData.scale[1] * surfPt.texCoord.v + nodeData.offset[1],
                       0.0f);
    }



    RT_CALLABLE_PROGRAM SampledSpectrum Image2DTextureShaderNode_spectrum(const uint32_t* rawNodeData, uint32_t option,
                                                                          const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(nodeData.spectrumType, nodeData.colorSpace, texValue.x, texValue.y, texValue.z).evaluate(wls);
#else
        return SampledSpectrum(texValue.x, texValue.y, texValue.z); // assume given data is in rendering RGB.
#endif
    }

    RT_CALLABLE_PROGRAM float Image2DTextureShaderNode_float(const uint32_t* rawNodeData, uint32_t option,
                                                             const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        if (option == 0)
            return texValue.x;
        else if (option == 1)
            return texValue.y;
        else if (option == 2)
            return texValue.z;
        else if (option == 3)
            return texValue.w;

        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Image2DTextureShaderNode_float2(const uint32_t* rawNodeData, uint32_t option,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        if (option == 0)
            return optix::make_float2(texValue.x, texValue.y);
        else if (option == 1)
            return optix::make_float2(texValue.y, texValue.z);
        else if (option == 2)
            return optix::make_float2(texValue.z, texValue.w);

        return optix::make_float2(0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float3 Image2DTextureShaderNode_float3(const uint32_t* rawNodeData, uint32_t option,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        if (option == 0)
            return optix::make_float3(texValue.x, texValue.y, texValue.z);
        else if (option == 1)
            return optix::make_float3(texValue.y, texValue.z, texValue.w);

        return optix::make_float3(0.0f, 0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float4 Image2DTextureShaderNode_float4(const uint32_t* rawNodeData, uint32_t option,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        return texValue;
    }



    RT_CALLABLE_PROGRAM SampledSpectrum EnvironmentTextureShaderNode_spectrum(const uint32_t* rawNodeData, uint32_t option,
                                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *(const EnvironmentTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(VLRSpectrumType_LightSource, nodeData.colorSpace, texValue.x, texValue.y, texValue.z).evaluate(wls);
#else
        return SampledSpectrum(texValue.x, texValue.y, texValue.z); // assume given data is in rendering RGB.
#endif
    }
}
