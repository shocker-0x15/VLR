#pragma once

#include "scene.h"

typedef VLR::Object* VLRObject;

typedef VLR::Context* VLRContext;

typedef VLR::Image2D* VLRImage2D;
typedef VLR::LinearImage2D* VLRLinearImage2D;

typedef VLR::ShaderNode* VLRShaderNode;
typedef VLR::OffsetAndScaleUVTextureMap2DShaderNode* VLROffsetAndScaleUVTextureMap2DShaderNode;
typedef VLR::ConstantTextureShaderNode* VLRConstantTextureShaderNode;
typedef VLR::Image2DTextureShaderNode* VLRImage2DTextureShaderNode;

typedef VLR::FloatTexture* VLRFloatTexture;
typedef VLR::ConstantFloatTexture* VLRConstantFloatTexture;
typedef VLR::Float2Texture* VLRFloat2Texture;
typedef VLR::ConstantFloat2Texture* VLRConstantFloat2Texture;
typedef VLR::Float3Texture* VLRFloat3Texture;
typedef VLR::ConstantFloat3Texture* VLRConstantFloat3Texture;
typedef VLR::ImageFloat3Texture* VLRImageFloat3Texture;
typedef VLR::Float4Texture* VLRFloat4Texture;
typedef VLR::ConstantFloat4Texture* VLRConstantFloat4Texture;
typedef VLR::ImageFloat4Texture* VLRImageFloat4Texture;

typedef VLR::SurfaceMaterial* VLRSurfaceMaterial;
typedef VLR::MatteSurfaceMaterial* VLRMatteSurfaceMaterial;
typedef VLR::SpecularReflectionSurfaceMaterial* VLRSpecularReflectionSurfaceMaterial;
typedef VLR::SpecularScatteringSurfaceMaterial* VLRSpecularScatteringSurfaceMaterial;
typedef VLR::MicrofacetReflectionSurfaceMaterial* VLRMicrofacetReflectionSurfaceMaterial;
typedef VLR::MicrofacetScatteringSurfaceMaterial* VLRMicrofacetScatteringSurfaceMaterial;
typedef VLR::LambertianScatteringSurfaceMaterial* VLRLambertianScatteringSurfaceMaterial;
typedef VLR::UE4SurfaceMaterial* VLRUE4SurfaceMaterial;
typedef VLR::DiffuseEmitterSurfaceMaterial* VLRDiffuseEmitterSurfaceMaterial;
typedef VLR::MultiSurfaceMaterial* VLRMultiSurfaceMaterial;
typedef VLR::EnvironmentEmitterSurfaceMaterial* VLREnvironmentEmitterSurfaceMaterial;

typedef VLR::Transform* VLRTransform;
typedef VLR::Transform const* VLRTransformConst;
typedef VLR::StaticTransform* VLRStaticTransform;

typedef VLR::SurfaceNode* VLRSurfaceNode;
typedef VLR::TriangleMeshSurfaceNode* VLRTriangleMeshSurfaceNode;
typedef VLR::InternalNode* VLRInternalNode;
typedef VLR::Scene* VLRScene;

typedef VLR::Camera* VLRCamera;
typedef VLR::PerspectiveCamera* VLRPerspectiveCamera;
typedef VLR::EquirectangularCamera* VLREquirectangularCamera;

#include <VLR.h>



VLR_API VLRResult vlrPrintDevices() {
    const auto checkError = [](RTresult code) {
        if (code != RT_SUCCESS && code != RT_TIMEOUT_CALLBACK)
            throw optix::Exception::makeException(code, 0);
    };

    uint32_t numDevices;
    checkError(rtDeviceGetDeviceCount(&numDevices));

    for (int dev = 0; dev < numDevices; ++dev) {
        VLRDebugPrintf("----------------------------------------------------------------\n");

        char strBuffer[256];
        int32_t intBuffer[2];
        RTsize sizeValue;

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_NAME, sizeof(strBuffer), strBuffer);
        VLRDebugPrintf("%d: %s\n", dev, strBuffer);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    CUDA Device Ordinal: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(strBuffer), strBuffer);
        VLRDebugPrintf("    PCI Bus ID: %s\n", strBuffer);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(intBuffer), intBuffer);
        VLRDebugPrintf("    Compute Capability: %d, %d\n", intBuffer[0], intBuffer[1]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    TCC (Tesla Compute Cluster) Driver: %s\n", intBuffer[0] ? "Yes" : "No");

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(sizeValue), &sizeValue);
        VLRDebugPrintf("    Total Memory: %llu [Byte]\n", sizeValue);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    Clock Rate: %d [kHz]\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    Max Threads per Block: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    Multi Processor Count: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    Max Hardware Texture Count: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(intBuffer[0]), &intBuffer[0]);
        VLRDebugPrintf("    Execution Timeout Enabled: %s\n", intBuffer[0] ? "Yes" : "No");
    }
    VLRDebugPrintf("----------------------------------------------------------------\n");

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, uint32_t stackSize) {
    *context = new VLR::Context(logging, stackSize);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrContextSetDevices(VLRContext context, const int32_t* devices, uint32_t numDevices) {
    context->setDevices(devices, numDevices);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDestroyContext(VLRContext context) {
    delete context;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrContextBindOutputBuffer(VLRContext context, uint32_t width, uint32_t height, uint32_t bufferID) {
    context->bindOutputBuffer(width, height, bufferID);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrContextMapOutputBuffer(VLRContext context, void** ptr) {
    *ptr = context->mapOutputBuffer();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrContextUnmapOutputBuffer(VLRContext context) {
    context->unmapOutputBuffer();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCamera camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>())
        return VLR_ERROR_INVALID_TYPE;
    context->render(*scene, camera, shrinkCoeff, firstFrame, numAccumFrames);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                         uint32_t width, uint32_t height, VLRDataFormat format, bool applyDegamma, uint8_t* linearData) {
    *image = new VLR::LinearImage2D(*context, linearData, width, height, format, applyDegamma);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image) {
    if (!image->is<VLR::LinearImage2D>())
        return VLR_ERROR_INVALID_TYPE;
    delete image;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DGetWidth(VLRLinearImage2D image, uint32_t* width) {
    if (!image->is<VLR::LinearImage2D>())
        return VLR_ERROR_INVALID_TYPE;
    *width = image->getWidth();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DGetHeight(VLRLinearImage2D image, uint32_t* height) {
    if (!image->is<VLR::LinearImage2D>())
        return VLR_ERROR_INVALID_TYPE;
    *height = image->getHeight();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DGetStride(VLRLinearImage2D image, uint32_t* stride) {
    if (!image->is<VLR::LinearImage2D>())
        return VLR_ERROR_INVALID_TYPE;
    *stride = image->getStride();

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeCreate(VLRContext context, VLROffsetAndScaleUVTextureMap2DShaderNode* node,
                                                                  const float offset[2], const float scale[2]) {
    *node = new VLR::OffsetAndScaleUVTextureMap2DShaderNode(*context, offset, scale);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOffsetAndScaleUVTextureMap2DShaderNodeDestroy(VLRContext context, VLROffsetAndScaleUVTextureMap2DShaderNode node) {
    if (!node->is<VLR::OffsetAndScaleUVTextureMap2DShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrConstantTextureShaderNodeCreate(VLRContext context, VLRConstantTextureShaderNode* node,
                                                     const float spectrum[3], float alpha) {
    *node = new VLR::ConstantTextureShaderNode(*context, VLR::RGBSpectrum(spectrum[0], spectrum[1], spectrum[2]), alpha);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrConstantTextureShaderNodeDestroy(VLRContext context, VLRConstantTextureShaderNode node) {
    if (!node->is<VLR::ConstantTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrImage2DTextureShaderNodeCreate(VLRContext context, VLRImage2DTextureShaderNode* node,
                                                    VLRImage2D image, VLRShaderNode nodeTexCoord) {
    *node = new VLR::Image2DTextureShaderNode(*context, image, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeDestroy(VLRContext context, VLRImage2DTextureShaderNode node) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetFilterMode(VLRContext context, VLRImage2DTextureShaderNode node,
                                                           VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloatTextureSetFilterMode(VLRContext context, VLRFloatTexture texture,
                                               VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!texture->isMemberOf<VLR::FloatTexture>())
        return VLR_ERROR_INVALID_TYPE;
    texture->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrConstantFloatTextureCreate(VLRContext context, VLRConstantFloatTexture* texture,
                                                const float value) {
    *texture = new VLR::ConstantFloatTexture(*context, value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrConstantFloatTextureDestroy(VLRContext context, VLRConstantFloatTexture texture) {
    if (!texture->is<VLR::ConstantFloatTexture>())
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloat2TextureSetFilterMode(VLRContext context, VLRFloat2Texture texture,
                                                VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!texture->isMemberOf<VLR::Float2Texture>())
        return VLR_ERROR_INVALID_TYPE;
    texture->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrConstantFloat2TextureCreate(VLRContext context, VLRConstantFloat2Texture* texture,
                                                 const float value[2]) {
    *texture = new VLR::ConstantFloat2Texture(*context, value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrConstantFloat2TextureDestroy(VLRContext context, VLRConstantFloat2Texture texture) {
    if (!texture->is<VLR::ConstantFloat2Texture>())
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloat3TextureSetFilterMode(VLRContext context, VLRFloat3Texture texture,
                                                VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!texture->isMemberOf<VLR::Float3Texture>())
        return VLR_ERROR_INVALID_TYPE;
    texture->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrConstantFloat3TextureCreate(VLRContext context, VLRConstantFloat3Texture* texture,
                                                 const float value[3]) {
    *texture = new VLR::ConstantFloat3Texture(*context, value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrConstantFloat3TextureDestroy(VLRContext context, VLRConstantFloat3Texture texture) {
    if (!texture->is<VLR::ConstantFloat3Texture>())
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrImageFloat3TextureCreate(VLRContext context, VLRImageFloat3Texture* texture,
                                              VLRImage2D image) {
    *texture = new VLR::ImageFloat3Texture(*context, image);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImageFloat3TextureDestroy(VLRContext context, VLRImageFloat3Texture texture) {
    if (!texture->is<VLR::ImageFloat3Texture>())
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloat4TextureSetFilterMode(VLRContext context, VLRFloat4Texture texture,
                                                VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!texture->isMemberOf<VLR::Float4Texture>())
        return VLR_ERROR_INVALID_TYPE;
    texture->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrConstantFloat4TextureCreate(VLRContext context, VLRConstantFloat4Texture* texture,
                                                 const float value[4]) {
    *texture = new VLR::ConstantFloat4Texture(*context, value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrConstantFloat4TextureDestroy(VLRContext context, VLRConstantFloat4Texture texture) {
    if (!texture->is<VLR::ConstantFloat4Texture>())
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrImageFloat4TextureCreate(VLRContext context, VLRImageFloat4Texture* texture,
                                              VLRImage2D image) {
    *texture = new VLR::ImageFloat4Texture(*context, image);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImageFloat4TextureDestroy(VLRContext context, VLRImageFloat4Texture texture) {
    if (!texture->is<VLR::ImageFloat4Texture>())
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material,
                                                VLRShaderNode nodeAlbedo) {
    *material = new VLR::MatteSurfaceMaterial(*context, nodeAlbedo);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material) {
    if (!material->is<VLR::MatteSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialCreate(VLRContext context, VLRSpecularReflectionSurfaceMaterial* material,
                                                             VLRFloat3Texture texCoeffR, VLRFloat3Texture texEta, VLRFloat3Texture tex_k, VLRShaderNode nodeTexCoord) {
    *material = new VLR::SpecularReflectionSurfaceMaterial(*context, texCoeffR, texEta, tex_k, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialDestroy(VLRContext context, VLRSpecularReflectionSurfaceMaterial material) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialCreate(VLRContext context, VLRSpecularScatteringSurfaceMaterial* material,
                                                             VLRFloat3Texture texCoeff, VLRFloat3Texture texEtaExt, VLRFloat3Texture texEtaInt, VLRShaderNode nodeTexCoord) {
    *material = new VLR::SpecularScatteringSurfaceMaterial(*context, texCoeff, texEtaExt, texEtaInt, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialDestroy(VLRContext context, VLRSpecularScatteringSurfaceMaterial material) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialCreate(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial* material,
                                                               VLRFloat3Texture texEta, VLRFloat3Texture tex_k, VLRFloat2Texture texRoughness, VLRShaderNode nodeTexCoord) {
    *material = new VLR::MicrofacetReflectionSurfaceMaterial(*context, texEta, tex_k, texRoughness, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial material) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialCreate(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial* material,
                                                               VLRFloat3Texture texCoeff, VLRFloat3Texture texEtaExt, VLRFloat3Texture texEtaInt, VLRFloat2Texture texRoughness, VLRShaderNode nodeTexCoord) {
    *material = new VLR::MicrofacetScatteringSurfaceMaterial(*context, texCoeff, texEtaExt, texEtaInt, texRoughness, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial material) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialCreate(VLRContext context, VLRLambertianScatteringSurfaceMaterial* material,
                                                               VLRFloat3Texture texCoeff, VLRFloatTexture texF0, VLRShaderNode nodeTexCoord) {
    *material = new VLR::LambertianScatteringSurfaceMaterial(*context, texCoeff, texF0, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialDestroy(VLRContext context, VLRLambertianScatteringSurfaceMaterial material) {
    if (!material->is<VLR::LambertianScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material,
                                              VLRFloat3Texture texBaseColor, VLRFloat3Texture texOcclusionRoughnessMetallic, VLRShaderNode nodeTexCoord) {
    *material = new VLR::UE4SurfaceMaterial(*context, texBaseColor, texOcclusionRoughnessMetallic, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialCreate(VLRContext context, VLRDiffuseEmitterSurfaceMaterial* material,
                                                         VLRFloat3Texture texEmittance, VLRShaderNode nodeTexCoord) {
    *material = new VLR::DiffuseEmitterSurfaceMaterial(*context, texEmittance, nodeTexCoord);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialDestroy(VLRContext context, VLRDiffuseEmitterSurfaceMaterial material) {
    if (!material->is<VLR::DiffuseEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMultiSurfaceMaterialCreate(VLRContext context, VLRMultiSurfaceMaterial* material,
                                                const VLRSurfaceMaterial* materials, uint32_t numMaterials) {
    *material = new VLR::MultiSurfaceMaterial(*context, (const VLR::SurfaceMaterial**)materials, numMaterials);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMultiSurfaceMaterialDestroy(VLRContext context, VLRMultiSurfaceMaterial material) {
    if (!material->is<VLR::MultiSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialCreate(VLRContext context, VLREnvironmentEmitterSurfaceMaterial* material,
                                                             VLRFloat3Texture texEmittance) {
    *material = new VLR::EnvironmentEmitterSurfaceMaterial(*context, texEmittance);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialDestroy(VLRContext context, VLREnvironmentEmitterSurfaceMaterial material) {
    if (!material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrStaticTransformCreate(VLRContext context, VLRStaticTransform* transform,
                                           const float mat[16]) {
    VLR::Matrix4x4 mat4x4(mat);
    *transform = new VLR::StaticTransform(mat4x4);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrStaticTransformDestroy(VLRContext context, VLRStaticTransform transform) {
    if (!transform->is<VLR::StaticTransform>())
        return VLR_ERROR_INVALID_TYPE;
    delete transform;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode, 
                                                   const char* name) {
    *surfaceNode = new VLR::TriangleMeshSurfaceNode(*context, name);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode) {
    if (!surfaceNode->is<VLR::TriangleMeshSurfaceNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete surfaceNode;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetName(VLRTriangleMeshSurfaceNode node, const char* name) {
    if (!node->is<VLR::TriangleMeshSurfaceNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setName(name);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeGetName(VLRTriangleMeshSurfaceNode node, const char** name) {
    if (!node->is<VLR::TriangleMeshSurfaceNode>())
        return VLR_ERROR_INVALID_TYPE;
    *name = node->getName().c_str();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, VLRVertex* vertices, uint32_t numVertices) {
    if (!surfaceNode->is<VLR::TriangleMeshSurfaceNode>())
        return VLR_ERROR_INVALID_TYPE;

    std::vector<VLR::Vertex> vecVertices;
    vecVertices.resize(numVertices);
    std::copy_n((VLR::Vertex*)vertices, numVertices, vecVertices.data());

    surfaceNode->setVertices(std::move(vecVertices));

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, uint32_t* indices, uint32_t numIndices, 
                                                             VLRSurfaceMaterial material, VLRFloat4Texture texNormalAlpha) {
    if (!surfaceNode->is<VLR::TriangleMeshSurfaceNode>())
        return VLR_ERROR_INVALID_TYPE;

    std::vector<uint32_t> vecIndices;
    vecIndices.resize(numIndices);
    std::copy_n(indices, numIndices, vecIndices.data());

    if (!material->isMemberOf<VLR::SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;

    surfaceNode->addMaterialGroup(std::move(vecIndices), material, texNormalAlpha);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                        const char* name, VLRTransform transform) {
    *node = new VLR::InternalNode(*context, name, transform);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeSetName(VLRInternalNode node, const char* name) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setName(name);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeGetName(VLRInternalNode node, const char** name) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;
    *name = node->getName().c_str();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, VLRTransform localToWorld) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setTransform(localToWorld);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeGetTransform(VLRInternalNode node, VLRTransformConst* localToWorld) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;
    *localToWorld = node->getTransform();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRObject child) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;

    if (child->isMemberOf<VLR::InternalNode>())
        node->addChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        node->addChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRObject child) {
    if (!node->is<VLR::InternalNode>())
        return VLR_ERROR_INVALID_TYPE;

    if (child->isMemberOf<VLR::InternalNode>())
        node->removeChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        node->removeChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                 VLRTransform transform) {
    *scene = new VLR::Scene(*context, transform);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene) {
    if (!scene->is<VLR::Scene>())
        return VLR_ERROR_INVALID_TYPE;
    delete scene;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, VLRTransform localToWorld) {
    if (!scene->is<VLR::Scene>())
        return VLR_ERROR_INVALID_TYPE;
    scene->setTransform(localToWorld);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRObject child) {
    if (!scene->is<VLR::Scene>())
        return VLR_ERROR_INVALID_TYPE;

    if (child->isMemberOf<VLR::InternalNode>())
        scene->addChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        scene->addChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRObject child) {
    if (!scene->is<VLR::Scene>())
        return VLR_ERROR_INVALID_TYPE;

    if (child->isMemberOf<VLR::InternalNode>())
        scene->removeChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        scene->removeChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLREnvironmentEmitterSurfaceMaterial material) {
    if (!scene->is<VLR::Scene>() || !material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;

    scene->setEnvironment(material);

    return VLR_ERROR_NO_ERROR;
}




VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera,
                                             const VLRPoint3D* position, const VLRQuaternion* orientation,
                                             float sensitivity, float aspect, float fovY, float lensRadius, float imgPDist, float objPDist) {
    *camera = new VLR::PerspectiveCamera(*context, *(VLR::Point3D*)position, *(VLR::Quaternion*)orientation, sensitivity, aspect, fovY, lensRadius, imgPDist, objPDist);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraDestroy(VLRContext context, VLRPerspectiveCamera camera) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    delete camera;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraSetPosition(VLRPerspectiveCamera camera, const VLRPoint3D* position) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setPosition(*(VLR::Point3D*)position);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraSetOrientation(VLRPerspectiveCamera camera, const VLRQuaternion* orientation) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setOrientation(*(VLR::Quaternion*)orientation);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraSetSensitivity(VLRPerspectiveCamera camera, float sensitivity) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setSensitivity(sensitivity);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraSetFovY(VLRPerspectiveCamera camera, float fovY) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setFovY(fovY);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraSetLensRadius(VLRPerspectiveCamera camera, float lensRadius) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setLensRadius(lensRadius);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraSetObjectPlaneDistance(VLRPerspectiveCamera camera, float distance) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setObjectPlaneDistance(distance);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera,
                                                 const VLRPoint3D* position, const VLRQuaternion* orientation,
                                                 float sensitivity, float phiAngle, float thetaAngle) {
    *camera = new VLR::EquirectangularCamera(*context, *(VLR::Point3D*)position, *(VLR::Quaternion*)orientation, sensitivity, phiAngle, thetaAngle);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraDestroy(VLRContext context, VLREquirectangularCamera camera) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    delete camera;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraSetPosition(VLREquirectangularCamera camera, const VLRPoint3D* position) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setPosition(*(VLR::Point3D*)position);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraSetOrientation(VLREquirectangularCamera camera, const VLRQuaternion* orientation) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setOrientation(*(VLR::Quaternion*)orientation);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraSetSensitivity(VLREquirectangularCamera camera, float sensitivity) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setSensitivity(sensitivity);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraSetAngles(VLREquirectangularCamera camera, float phiAngle, float thetaAngle) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setAngles(phiAngle, thetaAngle);

    return VLR_ERROR_NO_ERROR;
}
