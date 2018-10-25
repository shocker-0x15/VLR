#include "kernel_common.cuh"

namespace VLR {
    RT_CALLABLE_PROGRAM Point3D OffsetAndScaleUVTextureMap2DShaderNode_TexCoord(const uint32_t* nodeData, const SurfacePoint &surfPt) {
        const OffsetAndScaleUVTextureMap2DShaderNode &texMap = *(const OffsetAndScaleUVTextureMap2DShaderNode*)nodeData;
        return Point3D(texMap.scale[0] * surfPt.texCoord.u + texMap.offset[0],
                       texMap.scale[1] * surfPt.texCoord.v + texMap.offset[1],
                       0.0f);
    }



    RT_CALLABLE_PROGRAM RGBSpectrum ConstantTextureShaderNode_RGBSpectrum(const uint32_t* nodeData, const SurfacePoint &surfPt) {
        const ConstantTextureShaderNode &value = *(const ConstantTextureShaderNode*)nodeData;

        return value.spectrum;
    }



    RT_CALLABLE_PROGRAM RGBSpectrum Image2DTextureShaderNode_RGBSpectrum(const uint32_t* nodeData, const SurfacePoint &surfPt) {
        const Image2DTextureShaderNode &image = *(const Image2DTextureShaderNode*)nodeData;

        Point3D texCoord = calcTextureCoodinate(image.nodeTexCoord, surfPt);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(image.textureID, texCoord.x, texCoord.y);

        return RGBSpectrum(texValue.x, texValue.y, texValue.z);
    }
}
