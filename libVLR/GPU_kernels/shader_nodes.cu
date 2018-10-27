#include "kernel_common.cuh"

namespace VLR {
    RT_CALLABLE_PROGRAM float FloatShaderNode_float(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const FloatShaderNode &nodeData = *(const FloatShaderNode*)rawNodeData;
        return calcFloat(nodeData.node0, nodeData.imm0, surfPt);
    }



    RT_CALLABLE_PROGRAM optix::float2 Float2ShaderNode_float2(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const Float2ShaderNode &nodeData = *(const Float2ShaderNode*)rawNodeData;
        return optix::make_float2(calcFloat(nodeData.node0, nodeData.imm0, surfPt),
                                  calcFloat(nodeData.node1, nodeData.imm1, surfPt));
    }



    RT_CALLABLE_PROGRAM optix::float3 Float3ShaderNode_float3(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const Float3ShaderNode &nodeData = *(const Float3ShaderNode*)rawNodeData;
        return optix::make_float3(calcFloat(nodeData.node0, nodeData.imm0, surfPt),
                                  calcFloat(nodeData.node1, nodeData.imm1, surfPt),
                                  calcFloat(nodeData.node2, nodeData.imm2, surfPt));
    }



    RT_CALLABLE_PROGRAM optix::float4 Float4ShaderNode_float4(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const Float4ShaderNode &nodeData = *(const Float4ShaderNode*)rawNodeData;
        return optix::make_float4(calcFloat(nodeData.node0, nodeData.imm0, surfPt),
                                  calcFloat(nodeData.node1, nodeData.imm1, surfPt),
                                  calcFloat(nodeData.node2, nodeData.imm2, surfPt),
                                  calcFloat(nodeData.node3, nodeData.imm3, surfPt));
    }



    RT_CALLABLE_PROGRAM Point3D OffsetAndScaleUVTextureMap2DShaderNode_TexCoord(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const OffsetAndScaleUVTextureMap2DShaderNode &nodeData = *(const OffsetAndScaleUVTextureMap2DShaderNode*)rawNodeData;
        return Point3D(nodeData.scale[0] * surfPt.texCoord.u + nodeData.offset[0],
                       nodeData.scale[1] * surfPt.texCoord.v + nodeData.offset[1],
                       0.0f);
    }



    RT_CALLABLE_PROGRAM RGBSpectrum ConstantTextureShaderNode_RGBSpectrum(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const ConstantTextureShaderNode &nodeData = *(const ConstantTextureShaderNode*)rawNodeData;

        return nodeData.spectrum;
    }

    RT_CALLABLE_PROGRAM float ConstantTextureShaderNode_Alpha(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const ConstantTextureShaderNode &nodeData = *(const ConstantTextureShaderNode*)rawNodeData;

        return nodeData.alpha;
    }



    RT_CALLABLE_PROGRAM RGBSpectrum Image2DTextureShaderNode_RGBSpectrum(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const Image2DTextureShaderNode &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcTextureCoodinate(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        return RGBSpectrum(texValue.x, texValue.y, texValue.z);
    }

    RT_CALLABLE_PROGRAM optix::float3 Image2DTextureShaderNode_float3(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const Image2DTextureShaderNode &nodeData = *(const Image2DTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcTextureCoodinate(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        return optix::make_float3(texValue.x, texValue.y, texValue.z);
    }



    RT_CALLABLE_PROGRAM RGBSpectrum EnvironmentTextureShaderNode_RGBSpectrum(const uint32_t* rawNodeData, const SurfacePoint &surfPt) {
        const EnvironmentTextureShaderNode &nodeData = *(const EnvironmentTextureShaderNode*)rawNodeData;

        Point3D texCoord = calcTextureCoodinate(nodeData.nodeTexCoord, Point3D(surfPt.texCoord.u, surfPt.texCoord.v, 0.0f), surfPt);
        optix::float4 texValue = optix::rtTex2D<optix::float4>(nodeData.textureID, texCoord.x, texCoord.y);

        return RGBSpectrum(texValue.x, texValue.y, texValue.z);
    }
}
