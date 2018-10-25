#pragma once

#include "public_types.h"

#define VLR_ERROR_NO_ERROR        0x80000000
#define VLR_ERROR_INVALID_CONTEXT 0x80000001
#define VLR_ERROR_INVALID_TYPE    0x80000002

extern "C" {
    typedef uint32_t VLRResult;

#if !defined(VLR_API_EXPORTS)
    typedef struct VLRObject_API* VLRObject;

    typedef struct VLRContext_API* VLRContext;

    typedef struct VLRImage2D_API* VLRImage2D;
    typedef struct VLRLinearImage2D_API* VLRLinearImage2D;

    typedef struct VLRShaderNode_API* VLRShaderNode;
    typedef struct VLROffsetAndScaleUVTextureMap2DShaderNode_API* VLROffsetAndScaleUVTextureMap2DShaderNode;
    typedef struct VLRConstantTextureShaderNode_API* VLRConstantTextureShaderNode;
    typedef struct VLRImage2DTextureShaderNode_API* VLRImage2DTextureShaderNode;

    typedef struct VLRFloatTexture_API* VLRFloatTexture;
    typedef struct VLRConstantFloatTexture_API* VLRConstantFloatTexture;
    typedef struct VLRFloat2Texture_API* VLRFloat2Texture;
    typedef struct VLRConstantFloat2Texture_API* VLRConstantFloat2Texture;
    typedef struct VLRFloat3Texture_API* VLRFloat3Texture;
    typedef struct VLRConstantFloat3Texture_API* VLRConstantFloat3Texture;
    typedef struct VLRImageFloat3Texture_API* VLRImageFloat3Texture;
    typedef struct VLRFloat4Texture_API* VLRFloat4Texture;
    typedef struct VLRConstantFloat4Texture_API* VLRConstantFloat4Texture;
    typedef struct VLRImageFloat4Texture_API* VLRImageFloat4Texture;

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



    VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeCreate(VLRContext context, VLROffsetAndScaleUVTextureMap2DShaderNode* node,
                                                                      const float offset[2], const float scale[2]);
    VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeDestroy(VLRContext context, VLROffsetAndScaleUVTextureMap2DShaderNode node);

    VLR_API VLRResult vlrConstantTextureShaderNodeCreate(VLRContext context, VLRConstantTextureShaderNode* node,
                                                         const float spectrum[3], float alpha);
    VLR_API VLRResult vlrConstantTextureShaderNodeDestroy(VLRContext context, VLRConstantTextureShaderNode node);

    VLR_API VLRResult vlrImage2DTextureShaderNodeCreate(VLRContext context, VLRImage2DTextureShaderNode* node,
                                                        VLRImage2D image, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrImage2DTextureShaderNodeDestroy(VLRContext context, VLRImage2DTextureShaderNode node);
    VLR_API VLRResult vlrImage2DTextureShaderNodeSetFilterMode(VLRContext context, VLRImage2DTextureShaderNode node,
                                                               VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);



    VLR_API VLRResult vlrFloatTextureSetFilterMode(VLRContext context, VLRFloatTexture texture,
                                                   VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);

    VLR_API VLRResult vlrConstantFloatTextureCreate(VLRContext context, VLRConstantFloatTexture* texture,
                                                    const float value);
    VLR_API VLRResult vlrConstantFloatTextureDestroy(VLRContext context, VLRConstantFloatTexture texture);

    VLR_API VLRResult vlrFloat2TextureSetFilterMode(VLRContext context, VLRFloat2Texture texture,
                                                    VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);

    VLR_API VLRResult vlrConstantFloat2TextureCreate(VLRContext context, VLRConstantFloat2Texture* texture,
                                                     const float value[2]);
    VLR_API VLRResult vlrConstantFloat2TextureDestroy(VLRContext context, VLRConstantFloat2Texture texture);

    VLR_API VLRResult vlrFloat3TextureSetFilterMode(VLRContext context, VLRFloat3Texture texture,
                                                    VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);

    VLR_API VLRResult vlrConstantFloat3TextureCreate(VLRContext context, VLRConstantFloat3Texture* texture,
                                                     const float value[3]);
    VLR_API VLRResult vlrConstantFloat3TextureDestroy(VLRContext context, VLRConstantFloat3Texture texture);

    VLR_API VLRResult vlrImageFloat3TextureCreate(VLRContext context, VLRImageFloat3Texture* texture,
                                                  VLRImage2D image);
    VLR_API VLRResult vlrImageFloat3TextureDestroy(VLRContext context, VLRImageFloat3Texture texture);

    VLR_API VLRResult vlrFloat4TextureSetFilterMode(VLRContext context, VLRFloat4Texture texture,
                                                    VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);

    VLR_API VLRResult vlrConstantFloat4TextureCreate(VLRContext context, VLRConstantFloat4Texture* texture,
                                                     const float value[4]);
    VLR_API VLRResult vlrConstantFloat4TextureDestroy(VLRContext context, VLRConstantFloat4Texture texture);

    VLR_API VLRResult vlrImageFloat4TextureCreate(VLRContext context, VLRImageFloat4Texture* texture,
                                                  VLRImage2D image);
    VLR_API VLRResult vlrImageFloat4TextureDestroy(VLRContext context, VLRImageFloat4Texture texture);



    VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material,
                                                    VLRShaderNode nodeAlbedo);
    VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material);

    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialCreate(VLRContext context, VLRSpecularReflectionSurfaceMaterial* material,
                                                                 VLRFloat3Texture texCoeffR, VLRFloat3Texture texEta, VLRFloat3Texture tex_k, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialDestroy(VLRContext context, VLRSpecularReflectionSurfaceMaterial material);

    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialCreate(VLRContext context, VLRSpecularScatteringSurfaceMaterial* material,
                                                                 VLRFloat3Texture texCoeff, VLRFloat3Texture texEtaExt, VLRFloat3Texture texEtaInt, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialDestroy(VLRContext context, VLRSpecularScatteringSurfaceMaterial material);

    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialCreate(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial* material,
                                                                   VLRFloat3Texture texEta, VLRFloat3Texture tex_k, VLRFloat2Texture texRoughness, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial material);

    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialCreate(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial* material,
                                                                   VLRFloat3Texture texCoeff, VLRFloat3Texture texEtaExt, VLRFloat3Texture texEtaInt, VLRFloat2Texture texRoughness, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial material);

    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialCreate(VLRContext context, VLRLambertianScatteringSurfaceMaterial* material,
                                                                   VLRFloat3Texture texCoeff, VLRFloatTexture texF0, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialDestroy(VLRContext context, VLRLambertianScatteringSurfaceMaterial material);

    VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material,
                                                  VLRFloat3Texture texBaseColor, VLRFloat3Texture texOcclusionRoughnessMetallic, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material);

    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialCreate(VLRContext context, VLRDiffuseEmitterSurfaceMaterial* material,
                                                             VLRFloat3Texture texEmittance, VLRShaderNode nodeTexCoord);
    VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialDestroy(VLRContext context, VLRDiffuseEmitterSurfaceMaterial material);

    VLR_API VLRResult vlrMultiSurfaceMaterialCreate(VLRContext context, VLRMultiSurfaceMaterial* material,
                                                    const VLRSurfaceMaterial* materials, uint32_t numMaterials);
    VLR_API VLRResult vlrMultiSurfaceMaterialDestroy(VLRContext context, VLRMultiSurfaceMaterial material);

    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialCreate(VLRContext context, VLREnvironmentEmitterSurfaceMaterial* material,
                                                                 VLRFloat3Texture texEmittance);
    VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialDestroy(VLRContext context, VLREnvironmentEmitterSurfaceMaterial material);



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
                                                                 VLRSurfaceMaterial material, VLRFloat4Texture texNormalAlpha);



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
