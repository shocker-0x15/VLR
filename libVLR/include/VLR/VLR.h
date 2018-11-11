#pragma once

#include "public_types.h"

#define VLR_ERROR_NO_ERROR               0x80000000
#define VLR_ERROR_INVALID_CONTEXT        0x80000001
#define VLR_ERROR_INVALID_TYPE           0x80000002
#define VLR_ERROR_INCOMPATIBLE_NODE_TYPE 0x80000003

extern "C" {
    typedef uint32_t VLRResult;

#if !defined(VLR_API_EXPORTS)
    typedef struct VLRObject_API* VLRObject;

    typedef struct VLRContext_API* VLRContext;

    typedef struct VLRImage2D_API* VLRImage2D;
    typedef struct VLRLinearImage2D_API* VLRLinearImage2D;

    typedef struct VLRGeometryShaderNode_API* VLRGeometryShaderNode;
    typedef struct VLRShaderNode_API* VLRShaderNode;
    typedef struct VLRFloatShaderNode_API* VLRFloatShaderNode;
    typedef struct VLRFloat2ShaderNode_API* VLRFloat2ShaderNode;
    typedef struct VLRFloat3ShaderNode_API* VLRFloat3ShaderNode;
    typedef struct VLRFloat4ShaderNode_API* VLRFloat4ShaderNode;
    typedef struct VLRVector3DToSpectrumShaderNode_API* VLRVector3DToSpectrumShaderNode;
    typedef struct VLROffsetAndScaleUVTextureMap2DShaderNode_API* VLROffsetAndScaleUVTextureMap2DShaderNode;
    typedef struct VLRConstantTextureShaderNode_API* VLRConstantTextureShaderNode;
    typedef struct VLRImage2DTextureShaderNode_API* VLRImage2DTextureShaderNode;
    typedef struct VLREnvironmentTextureShaderNode_API* VLREnvironmentTextureShaderNode;

    typedef struct VLRSurfaceMaterial_API* VLRSurfaceMaterial;
    typedef struct VLRMatteSurfaceMaterial_API* VLRMatteSurfaceMaterial;
    typedef struct VLRSpecularReflectionSurfaceMaterial_API* VLRSpecularReflectionSurfaceMaterial;
    typedef struct VLRSpecularScatteringSurfaceMaterial_API* VLRSpecularScatteringSurfaceMaterial;
    typedef struct VLRMicrofacetReflectionSurfaceMaterial_API* VLRMicrofacetReflectionSurfaceMaterial;
    typedef struct VLRMicrofacetScatteringSurfaceMaterial_API* VLRMicrofacetScatteringSurfaceMaterial;
    typedef struct VLRLambertianScatteringSurfaceMaterial_API* VLRLambertianScatteringSurfaceMaterial;
    typedef struct VLRUE4SurfaceMaterial_API* VLRUE4SurfaceMaterial;
    typedef struct VLRDiffuseEmitterSurfaceMaterial_API* VLRDiffuseEmitterSurfaceMaterial;
    typedef struct VLRMultiSurfaceMaterial_API* VLRMultiSurfaceMaterial;
    typedef struct VLREnvironmentEmitterSurfaceMaterial_API* VLREnvironmentEmitterSurfaceMaterial;

    typedef struct VLRTransform_API* VLRTransform;
    typedef const struct VLRTransform_API* VLRTransformConst;
    typedef struct VLRStaticTransform_API* VLRStaticTransform;

    typedef struct VLRSurfaceNode_API* VLRSurfaceNode;
    typedef struct VLRTriangleMeshSurfaceNode_API* VLRTriangleMeshSurfaceNode;
    typedef struct VLRInternalNode_API* VLRInternalNode;
    typedef struct VLRScene_API* VLRScene;

    typedef struct VLRCamera_API* VLRCamera;
    typedef struct VLRPerspectiveCamera_API* VLRPerspectiveCamera;
    typedef struct VLREquirectangularCamera_API* VLREquirectangularCamera;
#endif



    VLR_API VLRResult vlrPrintDevices();
    VLR_API VLRResult vlrGetDeviceName(uint32_t index, char* name, uint32_t bufferLength);

    VLR_API const char* vlrGetErrorMessage(VLRResult code);

    VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, uint32_t stackSize);
    VLR_API VLRResult vlrContextSetDevices(VLRContext context, const int32_t* devices, uint32_t numDevices);
    VLR_API VLRResult vlrDestroyContext(VLRContext context);

    VLR_API VLRResult vlrContextBindOutputBuffer(VLRContext context, uint32_t width, uint32_t height, uint32_t bufferID);
    VLR_API VLRResult vlrContextMapOutputBuffer(VLRContext context, void** ptr);
    VLR_API VLRResult vlrContextUnmapOutputBuffer(VLRContext context);
    VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCamera camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);



    VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                             uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData);
    VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image);
    VLR_API VLRResult vlrLinearImage2DGetWidth(VLRLinearImage2D image, uint32_t* width);
    VLR_API VLRResult vlrLinearImage2DGetHeight(VLRLinearImage2D image, uint32_t* height);
    VLR_API VLRResult vlrLinearImage2DGetStride(VLRLinearImage2D image, uint32_t* stride);



    VLR_API VLRResult vlrShaderNodeGetSocket(VLRShaderNode node, VLRShaderNodeSocketType socketType, uint32_t index, 
                                             VLRShaderNodeSocketInfo* socketInfo);

    VLR_API VLRResult vlrGeometryShaderNodeCreate(VLRContext context, VLRGeometryShaderNode* node);
    VLR_API VLRResult vlrGeometryShaderNodeDestroy(VLRContext context, VLRGeometryShaderNode node);
    
    VLR_API VLRResult vlrFloatShaderNodeCreate(VLRContext context, VLRFloatShaderNode* node);
    VLR_API VLRResult vlrFloatShaderNodeDestroy(VLRContext context, VLRFloatShaderNode node);
    VLR_API VLRResult vlrFloatShaderNodeSetNode0(VLRFloatShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloatShaderNodeSetImmediateValue0(VLRFloatShaderNode node, float value);

    VLR_API VLRResult vlrFloat2ShaderNodeCreate(VLRContext context, VLRFloat2ShaderNode* node);
    VLR_API VLRResult vlrFloat2ShaderNodeDestroy(VLRContext context, VLRFloat2ShaderNode node);
    VLR_API VLRResult vlrFloat2ShaderNodeSetNode0(VLRFloat2ShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue0(VLRFloat2ShaderNode node, float value);
    VLR_API VLRResult vlrFloat2ShaderNodeSetNode1(VLRFloat2ShaderNode node, VLRShaderNode node1, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue1(VLRFloat2ShaderNode node, float value);

    VLR_API VLRResult vlrFloat3ShaderNodeCreate(VLRContext context, VLRFloat3ShaderNode* node);
    VLR_API VLRResult vlrFloat3ShaderNodeDestroy(VLRContext context, VLRFloat3ShaderNode node);
    VLR_API VLRResult vlrFloat3ShaderNodeSetNode0(VLRFloat3ShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue0(VLRFloat3ShaderNode node, float value);
    VLR_API VLRResult vlrFloat3ShaderNodeSetNode1(VLRFloat3ShaderNode node, VLRShaderNode node1, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue1(VLRFloat3ShaderNode node, float value);
    VLR_API VLRResult vlrFloat3ShaderNodeSetNode2(VLRFloat3ShaderNode node, VLRShaderNode node2, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue2(VLRFloat3ShaderNode node, float value);

    VLR_API VLRResult vlrFloat4ShaderNodeCreate(VLRContext context, VLRFloat4ShaderNode* node);
    VLR_API VLRResult vlrFloat4ShaderNodeDestroy(VLRContext context, VLRFloat4ShaderNode node);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode0(VLRFloat4ShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue0(VLRFloat4ShaderNode node, float value);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode1(VLRFloat4ShaderNode node, VLRShaderNode node1, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue1(VLRFloat4ShaderNode node, float value);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode2(VLRFloat4ShaderNode node, VLRShaderNode node2, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue2(VLRFloat4ShaderNode node, float value);
    VLR_API VLRResult vlrFloat4ShaderNodeSetNode3(VLRFloat4ShaderNode node, VLRShaderNode node3, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue3(VLRFloat4ShaderNode node, float value);

    VLR_API VLRResult vlrVector3DToSpectrumShaderNodeCreate(VLRContext context, VLRVector3DToSpectrumShaderNode* node);
    VLR_API VLRResult vlrVector3DToSpectrumShaderNodeDestroy(VLRContext context, VLRVector3DToSpectrumShaderNode node);
    VLR_API VLRResult vlrVector3DToSpectrumShaderNodeSetNodeVector3D(VLRVector3DToSpectrumShaderNode node, VLRShaderNode nodeVector3D, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrVector3DToSpectrumShaderNodeSetImmediateValueVector3D(VLRVector3DToSpectrumShaderNode node, const VLRVector3D* value);
    
    VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeCreate(VLRContext context, VLROffsetAndScaleUVTextureMap2DShaderNode* node);
    VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeDestroy(VLRContext context, VLROffsetAndScaleUVTextureMap2DShaderNode node);
    VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeSetValues(VLROffsetAndScaleUVTextureMap2DShaderNode node, const float offset[2], const float scale[2]);

    VLR_API VLRResult vlrConstantTextureShaderNodeCreate(VLRContext context, VLRConstantTextureShaderNode* node);
    VLR_API VLRResult vlrConstantTextureShaderNodeDestroy(VLRContext context, VLRConstantTextureShaderNode node);
    VLR_API VLRResult vlrConstantTextureShaderNodeSetValues(VLRConstantTextureShaderNode node, const float spectrum[3], float alpha);

    VLR_API VLRResult vlrImage2DTextureShaderNodeCreate(VLRContext context, VLRImage2DTextureShaderNode* node);
    VLR_API VLRResult vlrImage2DTextureShaderNodeDestroy(VLRContext context, VLRImage2DTextureShaderNode node);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetImage(VLRImage2DTextureShaderNode node, VLRImage2D image);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetFilterMode(VLRImage2DTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetNodeTexCoord(VLRImage2DTextureShaderNode node, VLRShaderNode nodeTexCoord, VLRShaderNodeSocketInfo socketInfo);

    VLR_API VLRResult vlrEnvironmentTextureShaderNodeCreate(VLRContext context, VLREnvironmentTextureShaderNode* node);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeDestroy(VLRContext context, VLREnvironmentTextureShaderNode node);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetImage(VLREnvironmentTextureShaderNode node, VLRImage2D image);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetFilterMode(VLREnvironmentTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
    VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetNodeTexCoord(VLREnvironmentTextureShaderNode node, VLRShaderNode nodeTexCoord, VLRShaderNodeSocketInfo socketInfo);



    VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material);
    VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material);
    VLR_API VLRResult vlrMatteSurfaceMaterialSetNodeAlbedo(VLRMatteSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMatteSurfaceMaterialSetImmediateValueAlbedo(VLRMatteSurfaceMaterial material, const float value[3]);

    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialCreate(VLRContext context, VLRSpecularReflectionSurfaceMaterial* material);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialDestroy(VLRContext context, VLRSpecularReflectionSurfaceMaterial material);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR(VLRSpecularReflectionSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeEta(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta(VLRSpecularReflectionSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNode_k(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k(VLRSpecularReflectionSurfaceMaterial material, const float value[3]);

    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialCreate(VLRContext context, VLRSpecularScatteringSurfaceMaterial* material);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialDestroy(VLRContext context, VLRSpecularScatteringSurfaceMaterial material);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff(VLRSpecularScatteringSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRSpecularScatteringSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRSpecularScatteringSurfaceMaterial material, const float value[3]);

    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialCreate(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial* material);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial material);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta(VLRMicrofacetReflectionSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNode_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k(VLRMicrofacetReflectionSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetReflectionSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetReflectionSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetReflectionSurfaceMaterial material, float value);

    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialCreate(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial* material);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial material);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff(VLRMicrofacetScatteringSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetScatteringSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetScatteringSurfaceMaterial material, float value);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetScatteringSurfaceMaterial material, float value);

    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialCreate(VLRContext context, VLRLambertianScatteringSurfaceMaterial* material);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialDestroy(VLRContext context, VLRLambertianScatteringSurfaceMaterial material);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff(VLRLambertianScatteringSurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeF0(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueF0(VLRLambertianScatteringSurfaceMaterial material, float value);

    VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material);
    VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material);
    VLR_API VLRResult vlrUE4SufaceMaterialSetNodeBaseColor(VLRUE4SurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueBaseColor(VLRUE4SurfaceMaterial material, const float value[3]);
    VLR_API VLRResult vlrUE4SufaceMaterialSetNodeOcclusionRoughnessMetallic(VLRUE4SurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueOcclusion(VLRUE4SurfaceMaterial material, float value);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueRoughness(VLRUE4SurfaceMaterial material, float value);
    VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueMetallic(VLRUE4SurfaceMaterial material, float value);

    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialCreate(VLRContext context, VLRDiffuseEmitterSurfaceMaterial* material);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialDestroy(VLRContext context, VLRDiffuseEmitterSurfaceMaterial material);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance(VLRDiffuseEmitterSurfaceMaterial material, const float value[3]);

    VLR_API VLRResult vlrMultiSurfaceMaterialCreate(VLRContext context, VLRMultiSurfaceMaterial* material);
    VLR_API VLRResult vlrMultiSurfaceMaterialDestroy(VLRContext context, VLRMultiSurfaceMaterial material);
    VLR_API VLRResult vlrMultiSurfaceMaterialSetSubMaterial(VLRMultiSurfaceMaterial material, uint32_t index, VLRSurfaceMaterial mat);

    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialCreate(VLRContext context, VLREnvironmentEmitterSurfaceMaterial* material);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialDestroy(VLRContext context, VLREnvironmentEmitterSurfaceMaterial material);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittance(VLREnvironmentEmitterSurfaceMaterial material, VLREnvironmentTextureShaderNode node);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance(VLREnvironmentEmitterSurfaceMaterial material, const float value[3]);



    VLR_API VLRResult vlrStaticTransformCreate(VLRContext context, VLRStaticTransform* transform, 
                                               const float mat[16]);
    VLR_API VLRResult vlrStaticTransformDestroy(VLRContext context, VLRStaticTransform transform);



    VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode,
                                                       const char* name);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetName(VLRTriangleMeshSurfaceNode node, const char* name);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeGetName(VLRTriangleMeshSurfaceNode node, const char** name);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, VLRVertex* vertices, uint32_t numVertices);
    VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, uint32_t* indices, uint32_t numIndices, 
                                                                 VLRSurfaceMaterial material, 
                                                                 VLRShaderNode nodeNormal, VLRShaderNodeSocketInfo nodeNormalSocketInfo,
                                                                 VLRShaderNode nodeAlpha, VLRShaderNodeSocketInfo nodeAlphaSocketInfo,
                                                                 VLRTangentType tangentType);



    VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                            const char* name, VLRTransform transform);
    VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node);
    VLR_API VLRResult vlrInternalNodeSetName(VLRInternalNode node, const char* name);
    VLR_API VLRResult vlrInternalNodeGetName(VLRInternalNode node, const char** name);
    VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, VLRTransform localToWorld);
    VLR_API VLRResult vlrInternalNodeGetTransform(VLRInternalNode node, VLRTransformConst* localToWorld);
    VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRObject child);
    VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRObject child);



    VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                     VLRTransform transform);
    VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene);
    VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, VLRTransform localToWorld);
    VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRObject child);
    VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRObject child);
    VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLREnvironmentEmitterSurfaceMaterial material);



    VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera, 
                                                 const VLRPoint3D* position, const VLRQuaternion* orientation,
                                                 float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist);
    VLR_API VLRResult vlrPerspectiveCameraDestroy(VLRContext context, VLRPerspectiveCamera camera);
    VLR_API VLRResult vlrPerspectiveCameraSetPosition(VLRPerspectiveCamera camera, const VLRPoint3D* position);
    VLR_API VLRResult vlrPerspectiveCameraSetOrientation(VLRPerspectiveCamera camera, const VLRQuaternion* orientation);
    VLR_API VLRResult vlrPerspectiveCameraSetSensitivity(VLRPerspectiveCamera camera, float sensitivity);
    VLR_API VLRResult vlrPerspectiveCameraSetFovY(VLRPerspectiveCamera camera, float fovY);
    VLR_API VLRResult vlrPerspectiveCameraSetLensRadius(VLRPerspectiveCamera camera, float lensRadius);
    VLR_API VLRResult vlrPerspectiveCameraSetObjectPlaneDistance(VLRPerspectiveCamera camera, float distance);



    VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera,
                                                     const VLRPoint3D* position, const VLRQuaternion* orientation,
                                                     float sensitivity, float phiAngle, float thetaAngle);
    VLR_API VLRResult vlrEquirectangularCameraDestroy(VLRContext context, VLREquirectangularCamera camera);
    VLR_API VLRResult vlrEquirectangularCameraSetPosition(VLREquirectangularCamera camera, const VLRPoint3D* position);
    VLR_API VLRResult vlrEquirectangularCameraSetOrientation(VLREquirectangularCamera camera, const VLRQuaternion* orientation);
    VLR_API VLRResult vlrEquirectangularCameraSetSensitivity(VLREquirectangularCamera camera, float sensitivity);
    VLR_API VLRResult vlrEquirectangularCameraSetAngles(VLREquirectangularCamera camera, float phiAngle, float thetaAngle);
}
