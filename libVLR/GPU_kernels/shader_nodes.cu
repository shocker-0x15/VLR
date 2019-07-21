#include "kernel_common.cuh"

namespace VLR {
    template <typename T>
    RT_FUNCTION T* getData(uint32_t nodeDescIndex) {
        constexpr uint32_t sizeOfNodeInDW = sizeof(T) / 4;
        if /*constexpr*/ (sizeOfNodeInDW <= SmallNodeDescriptor::NumDWSlots())
            return pv_smallNodeDescriptorBuffer[nodeDescIndex].getData<T>();
        else if /*constexpr*/ (sizeOfNodeInDW <= MediumNodeDescriptor::NumDWSlots())
            return pv_mediumNodeDescriptorBuffer[nodeDescIndex].getData<T>();
        else if /*constexpr*/ (sizeOfNodeInDW <= LargeNodeDescriptor::NumDWSlots())
            return pv_largeNodeDescriptorBuffer[nodeDescIndex].getData<T>();
        return nullptr;
    }



    RT_CALLABLE_PROGRAM Point3D GeometryShaderNode_Point3D(const ShaderNodePlug &plug,
                                                           const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        return surfPt.position;
    }

    RT_CALLABLE_PROGRAM Normal3D GeometryShaderNode_Normal3D(const ShaderNodePlug &plug,
                                                             const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.option == 0)
            return surfPt.geometricNormal;
        else if (plug.option == 1)
            return surfPt.shadingFrame.z;
        return Normal3D(0, 0, 0);
    }

    RT_CALLABLE_PROGRAM Vector3D GeometryShaderNode_Vector3D(const ShaderNodePlug &plug,
                                                             const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        if (plug.option == 0)
            return surfPt.shadingFrame.x;
        else if (plug.option == 1)
            return surfPt.shadingFrame.y;
        return Vector3D::Zero();
    }

    RT_CALLABLE_PROGRAM Point3D GeometryShaderNode_TextureCoordinates(const ShaderNodePlug &plug,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        return Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0);
    }



    RT_CALLABLE_PROGRAM Vector3D TangentShaderNode_Vector3D(const ShaderNodePlug &plug,
                                                            const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<TangentShaderNode>(plug.nodeDescIndex);

        StaticTransform transform = pv_geometryInstanceDescriptorBuffer[surfPt.geometryInstanceIndex].body.asTriMesh.transform;
        
        // TODO: 同じGeometryGroup内でのstaticなtransformに関しても考慮する。
        Point3D localPosition = transform.mulInv(surfPt.position);

        Vector3D localTangent;
        switch (nodeData.tangentType) {
        case TangentType::TC0Direction: {
            return surfPt.shadingFrame.x;
        }
        case TangentType::RadialX: {
            localTangent = Vector3D(0, -localPosition.z, localPosition.y);
            break;
        }
        case TangentType::RadialY: {
            localTangent = Vector3D(localPosition.z, 0, -localPosition.x);
            break;
        }
        case TangentType::RadialZ: {
            localTangent = Vector3D(-localPosition.y, localPosition.x, 0);
            break;
        }
        default:
            break;
        }

        return transform * localTangent;
    }



    RT_CALLABLE_PROGRAM float Float2ShaderNode_float1(const ShaderNodePlug &plug,
                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float2ShaderNode>(plug.nodeDescIndex);
        if (plug.option == 0)
            return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
        else if (plug.option == 1)
            return calcNode(nodeData.node1, nodeData.imm1, surfPt, wls);
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Float2ShaderNode_float2(const ShaderNodePlug &plug,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float2ShaderNode>(plug.nodeDescIndex);
        return optix::make_float2(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                  calcNode(nodeData.node1, nodeData.imm1, surfPt, wls));
    }



    RT_CALLABLE_PROGRAM float Float3ShaderNode_float1(const ShaderNodePlug &plug,
                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float3ShaderNode>(plug.nodeDescIndex);
        if (plug.option == 0)
            return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
        else if (plug.option == 1)
            return calcNode(nodeData.node1, nodeData.imm1, surfPt, wls);
        else if (plug.option == 2)
            return calcNode(nodeData.node2, nodeData.imm2, surfPt, wls);
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Float3ShaderNode_float2(const ShaderNodePlug &plug,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float3ShaderNode>(plug.nodeDescIndex);
        if (plug.option == 0)
            return optix::make_float2(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                      calcNode(nodeData.node1, nodeData.imm1, surfPt, wls));
        else if (plug.option == 1)
            return optix::make_float2(calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
        return optix::make_float2(0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float3 Float3ShaderNode_float3(const ShaderNodePlug &plug, 
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float3ShaderNode>(plug.nodeDescIndex);
        return optix::make_float3(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                  calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                  calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
    }



    RT_CALLABLE_PROGRAM float Float4ShaderNode_float1(const ShaderNodePlug &plug,
                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float4ShaderNode>(plug.nodeDescIndex);
        if (plug.option == 0)
            return calcNode(nodeData.node0, nodeData.imm0, surfPt, wls);
        else if (plug.option == 1)
            return calcNode(nodeData.node1, nodeData.imm1, surfPt, wls);
        else if (plug.option == 2)
            return calcNode(nodeData.node2, nodeData.imm2, surfPt, wls);
        else if (plug.option == 3)
            return calcNode(nodeData.node3, nodeData.imm3, surfPt, wls);
        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Float4ShaderNode_float2(const ShaderNodePlug &plug,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float4ShaderNode>(plug.nodeDescIndex);
        if (plug.option == 0)
            return optix::make_float2(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                      calcNode(nodeData.node1, nodeData.imm1, surfPt, wls));
        else if (plug.option == 1)
            return optix::make_float2(calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
        else if (plug.option == 2)
            return optix::make_float2(calcNode(nodeData.node2, nodeData.imm2, surfPt, wls),
                                      calcNode(nodeData.node3, nodeData.imm3, surfPt, wls));
        return optix::make_float2(0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float3 Float4ShaderNode_float3(const ShaderNodePlug &plug,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float4ShaderNode>(plug.nodeDescIndex);
        if (plug.option == 0)
            return optix::make_float3(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                      calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls));
        else if (plug.option == 1)
            return optix::make_float3(calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                      calcNode(nodeData.node2, nodeData.imm2, surfPt, wls),
                                      calcNode(nodeData.node3, nodeData.imm3, surfPt, wls));
        return optix::make_float3(0.0f, 0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float4 Float4ShaderNode_float4(const ShaderNodePlug &plug,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float4ShaderNode>(plug.nodeDescIndex);
        return optix::make_float4(calcNode(nodeData.node0, nodeData.imm0, surfPt, wls),
                                  calcNode(nodeData.node1, nodeData.imm1, surfPt, wls),
                                  calcNode(nodeData.node2, nodeData.imm2, surfPt, wls),
                                  calcNode(nodeData.node3, nodeData.imm3, surfPt, wls));
    }



    RT_CALLABLE_PROGRAM float ScaleAndOffsetFloatShaderNode_float1(const ShaderNodePlug &plug,
                                                                   const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<ScaleAndOffsetFloatShaderNode>(plug.nodeDescIndex);
        float value = calcNode(nodeData.nodeValue, 0.0f, surfPt, wls);
        float scale = calcNode(nodeData.nodeScale, nodeData.immScale, surfPt, wls);
        float offset = calcNode(nodeData.nodeOffset, nodeData.immOffset, surfPt, wls);
        return scale * value + offset;
    }



    RT_CALLABLE_PROGRAM SampledSpectrum TripletSpectrumShaderNode_Spectrum(const ShaderNodePlug &plug,
                                                                           const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<TripletSpectrumShaderNode>(plug.nodeDescIndex);
        return nodeData.value.evaluate(wls);
    }



    RT_CALLABLE_PROGRAM SampledSpectrum RegularSampledSpectrumShaderNode_Spectrum(const ShaderNodePlug &plug,
                                                                                  const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<RegularSampledSpectrumShaderNode>(plug.nodeDescIndex);
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return RegularSampledSpectrum(nodeData.minLambda, nodeData.maxLambda, nodeData.values, nodeData.numSamples).evaluate(wls);
#else
        return nodeData.value.evaluate(wls);
#endif
    }



    RT_CALLABLE_PROGRAM SampledSpectrum IrregularSampledSpectrumShaderNode_Spectrum(const ShaderNodePlug &plug,
                                                                                    const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<IrregularSampledSpectrumShaderNode>(plug.nodeDescIndex);
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return IrregularSampledSpectrum(nodeData.lambdas, nodeData.values, nodeData.numSamples).evaluate(wls);
#else
        return nodeData.value.evaluate(wls);
#endif
    }



    RT_CALLABLE_PROGRAM SampledSpectrum Float3ToSpectrumShaderNode_Spectrum(const ShaderNodePlug &plug,
                                                                            const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Float3ToSpectrumShaderNode>(plug.nodeDescIndex);
        auto defaultValue = optix::make_float3(nodeData.immFloat3[0], nodeData.immFloat3[1], nodeData.immFloat3[2]);
        optix::float3 f3Value = calcNode(nodeData.nodeFloat3, defaultValue, surfPt, wls);
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(nodeData.spectrumType, nodeData.colorSpace,
                                 clamp(0.5f * f3Value.x + 0.5f, 0.0f, 1.0f),
                                 clamp(0.5f * f3Value.y + 0.5f, 0.0f, 1.0f),
                                 clamp(0.5f * f3Value.z + 0.5f, 0.0f, 1.0f)).evaluate(wls);
#else
        return SampledSpectrum(clamp(0.5f * f3Value.x + 0.5f, 0.0f, 1.0f),
                               clamp(0.5f * f3Value.y + 0.5f, 0.0f, 1.0f),
                               clamp(0.5f * f3Value.z + 0.5f, 0.0f, 1.0f));
#endif
    }



    RT_CALLABLE_PROGRAM Point3D ScaleAndOffsetUVTextureMap2DShaderNode_TextureCoordinates(const ShaderNodePlug &plug,
                                                                                          const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<ScaleAndOffsetUVTextureMap2DShaderNode>(plug.nodeDescIndex);
        return Point3D(nodeData.scale[0] * surfPt.texCoord.u + nodeData.offset[0],
                       nodeData.scale[1] * surfPt.texCoord.v + nodeData.offset[1],
                       0.0f);
    }



    RT_CALLABLE_PROGRAM float Image2DTextureShaderNode_float1(const ShaderNodePlug &plug,
                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);

        if (plug.option == 0)
            return texValue.x;
        else if (plug.option == 1)
            return texValue.y;
        else if (plug.option == 2)
            return texValue.z;
        else if (plug.option == 3)
            return texValue.w;

        return 0.0f;
    }

    RT_CALLABLE_PROGRAM optix::float2 Image2DTextureShaderNode_float2(const ShaderNodePlug &plug,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);

        if (plug.option == 0)
            return optix::make_float2(texValue.x, texValue.y);
        else if (plug.option == 1)
            return optix::make_float2(texValue.y, texValue.z);
        else if (plug.option == 2)
            return optix::make_float2(texValue.z, texValue.w);

        return optix::make_float2(0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float3 Image2DTextureShaderNode_float3(const ShaderNodePlug &plug,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);

        if (plug.option == 0)
            return optix::make_float3(texValue.x, texValue.y, texValue.z);
        else if (plug.option == 1)
            return optix::make_float3(texValue.y, texValue.z, texValue.w);

        return optix::make_float3(0.0f, 0.0f, 0.0f);
    }

    RT_CALLABLE_PROGRAM optix::float4 Image2DTextureShaderNode_float4(const ShaderNodePlug &plug,
                                                                      const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);

        return texValue;
    }

    RT_CALLABLE_PROGRAM Normal3D Image2DTextureShaderNode_Normal3D(const ShaderNodePlug &plug,
                                                                   const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);
        BumpType bumpType = nodeData.getBumpType();

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue;
        if (bumpType != BumpType::HeightMap) {
            texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);
        }
        else {
            // w z
            // x y
            texValue = optix::rtTex2DGather<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, plug.option);
        }

        float bumpCoeff = nodeData.getBumpCoeff();

        Normal3D ret(0.0f, 0.0f, 1.0f);
        if (bumpType != BumpType::HeightMap && plug.option < 2) {
            if (plug.option == 0)
                ret = Normal3D(texValue.x, texValue.y, texValue.z);
            else if (plug.option == 1)
                ret = Normal3D(texValue.y, texValue.z, texValue.w);

            ret = 2 * ret - 1.0f;

            if (bumpType == BumpType::NormalMap_DirectX)
                ret.y *= -1;
        }
        else if (bumpType == BumpType::HeightMap) {
            optix::uint3 texSize = optix::rtTexSize(nodeData.textureID);

            const float coeff = (5.0f / 1024);
            float dhdu = (coeff * texSize.x) * (texValue.y - texValue.x);
            float dhdv = (coeff * texSize.y) * (texValue.x - texValue.w);
            // cross(Vector3D(0, -1, dhdv), 
            //       Vector3D(1,  0, dhdu))
            ret = Normal3D(-dhdu, dhdv, 1);
        }

        ret.x *= bumpCoeff;
        ret.y *= bumpCoeff;

        return normalize(ret);
    }

    RT_CALLABLE_PROGRAM SampledSpectrum Image2DTextureShaderNode_Spectrum(const ShaderNodePlug &plug,
                                                                          const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);
        DataFormat dataFormat = nodeData.getDataFormat();
        if (dataFormat == DataFormat::Gray32F ||
            dataFormat == DataFormat::Gray8 ||
            dataFormat == DataFormat::GrayA8x2)
            texValue.z = texValue.y = texValue.x;

#if defined(VLR_USE_SPECTRAL_RENDERING)
        UpsampledSpectrum spectrum;
        if (dataFormat == DataFormat::uvsA8x4 ||
            dataFormat == DataFormat::uvsA16Fx4) {
            float u = texValue.x;
            float v = texValue.y;
            float s = texValue.z;
            if (dataFormat == DataFormat::uvsA8x4) {
                u *= UpsampledSpectrum::GridWidth();
                v *= UpsampledSpectrum::GridHeight();
                s *= 3;
            }
            // JP: uvsA16Fの場合もInf回避のために EqualEnergyReflectance で割っていないので
            //     どちらのフォーマットだとしても割る。
            // EN: 
            s /= UpsampledSpectrum::EqualEnergyReflectance();
            spectrum = UpsampledSpectrum(u, v, s);
        }
        else {
            spectrum = UpsampledSpectrum(nodeData.getSpectrumType(), nodeData.getColorSpace(), texValue.x, texValue.y, texValue.z);
        }
        return spectrum.evaluate(wls);
#else
        return SampledSpectrum(texValue.x, texValue.y, texValue.z); // assume given data is in rendering RGB.
#endif
    }

    RT_CALLABLE_PROGRAM float Image2DTextureShaderNode_Alpha(const ShaderNodePlug &plug,
                                                             const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<Image2DTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = calcNode(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt, wls);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);

        if (plug.option == 0)
            return texValue.x;
        else if (plug.option == 1)
            return texValue.y;
        else if (plug.option == 2)
            return texValue.z;
        else if (plug.option == 3)
            return texValue.w;

        return 0.0f;
    }



    RT_CALLABLE_PROGRAM SampledSpectrum EnvironmentTextureShaderNode_Spectrum(const ShaderNodePlug &plug,
                                                                              const SurfacePoint &surfPt, const WavelengthSamples &wls) {
        auto &nodeData = *getData<EnvironmentTextureShaderNode>(plug.nodeDescIndex);

        Point3D texCoord = Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f);
        optix::float4 texValue = optix::rtTex2DLod<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y, 0.0f);

#if defined(VLR_USE_SPECTRAL_RENDERING)
        DataFormat dataFormat = nodeData.getDataFormat();

        UpsampledSpectrum spectrum;
        if (dataFormat == DataFormat::uvsA16Fx4) {
            float u = texValue.x;
            float v = texValue.y;
            float s = texValue.z;
            s /= UpsampledSpectrum::EqualEnergyReflectance();
            spectrum = UpsampledSpectrum(u, v, s);
        }
        else {
            spectrum = UpsampledSpectrum(SpectrumType::LightSource, nodeData.getColorSpace(), texValue.x, texValue.y, texValue.z);
        }
        return spectrum.evaluate(wls);
#else
        return SampledSpectrum(texValue.x, texValue.y, texValue.z); // assume given data is in rendering RGB.
#endif
    }
}
