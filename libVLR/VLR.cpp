#pragma once

#include "scene.h"

typedef VLR::Object* VLRObject;

typedef VLR::Context* VLRContext;

typedef VLR::Image2D* VLRImage2D;
typedef VLR::LinearImage2D* VLRLinearImage2D;
typedef VLR::BlockCompressedImage2D* VLRBlockCompressedImage2D;

typedef VLR::ShaderNode* VLRShaderNode;
typedef VLR::GeometryShaderNode* VLRGeometryShaderNode;
typedef VLR::FloatShaderNode* VLRFloatShaderNode;
typedef VLR::Float2ShaderNode* VLRFloat2ShaderNode;
typedef VLR::Float3ShaderNode* VLRFloat3ShaderNode;
typedef VLR::Float4ShaderNode* VLRFloat4ShaderNode;
typedef VLR::ScaleAndOffsetFloatShaderNode* VLRScaleAndOffsetFloatShaderNode;
typedef VLR::TripletSpectrumShaderNode* VLRTripletSpectrumShaderNode;
typedef VLR::RegularSampledSpectrumShaderNode* VLRRegularSampledSpectrumShaderNode;
typedef VLR::IrregularSampledSpectrumShaderNode* VLRIrregularSampledSpectrumShaderNode;
typedef VLR::Vector3DToSpectrumShaderNode* VLRVector3DToSpectrumShaderNode;
typedef VLR::ScaleAndOffsetUVTextureMap2DShaderNode* VLRScaleAndOffsetUVTextureMap2DShaderNode;
typedef VLR::Image2DTextureShaderNode* VLRImage2DTextureShaderNode;
typedef VLR::EnvironmentTextureShaderNode* VLREnvironmentTextureShaderNode;

typedef VLR::SurfaceMaterial* VLRSurfaceMaterial;
typedef VLR::MatteSurfaceMaterial* VLRMatteSurfaceMaterial;
typedef VLR::SpecularReflectionSurfaceMaterial* VLRSpecularReflectionSurfaceMaterial;
typedef VLR::SpecularScatteringSurfaceMaterial* VLRSpecularScatteringSurfaceMaterial;
typedef VLR::MicrofacetReflectionSurfaceMaterial* VLRMicrofacetReflectionSurfaceMaterial;
typedef VLR::MicrofacetScatteringSurfaceMaterial* VLRMicrofacetScatteringSurfaceMaterial;
typedef VLR::LambertianScatteringSurfaceMaterial* VLRLambertianScatteringSurfaceMaterial;
typedef VLR::UE4SurfaceMaterial* VLRUE4SurfaceMaterial;
typedef VLR::OldStyleSurfaceMaterial* VLROldStyleSurfaceMaterial;
typedef VLR::DiffuseEmitterSurfaceMaterial* VLRDiffuseEmitterSurfaceMaterial;
typedef VLR::MultiSurfaceMaterial* VLRMultiSurfaceMaterial;
typedef VLR::EnvironmentEmitterSurfaceMaterial* VLREnvironmentEmitterSurfaceMaterial;

typedef VLR::Transform* VLRTransform;
typedef VLR::Transform const* VLRTransformConst;
typedef VLR::StaticTransform* VLRStaticTransform;

typedef VLR::Node* VLRNode;
typedef VLR::SurfaceNode* VLRSurfaceNode;
typedef VLR::TriangleMeshSurfaceNode* VLRTriangleMeshSurfaceNode;
typedef VLR::InternalNode* VLRInternalNode;
typedef VLR::Scene* VLRScene;

typedef VLR::Camera* VLRCamera;
typedef VLR::PerspectiveCamera* VLRPerspectiveCamera;
typedef VLR::EquirectangularCamera* VLREquirectangularCamera;

#include <VLR.h>



static void checkError(RTresult code) {
    if (code != RT_SUCCESS && code != RT_TIMEOUT_CALLBACK)
        throw optix::Exception::makeException(code, 0);
}



VLR_API VLRResult vlrPrintDevices() {
    uint32_t numDevices;
    checkError(rtDeviceGetDeviceCount(&numDevices));

    for (int dev = 0; dev < numDevices; ++dev) {
        vlrprintf("----------------------------------------------------------------\n");

        char strBuffer[256];
        int32_t intBuffer[2];
        RTsize sizeValue;

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_NAME, sizeof(strBuffer), strBuffer);
        vlrprintf("%d: %s\n", dev, strBuffer);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    CUDA Device Ordinal: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(strBuffer), strBuffer);
        vlrprintf("    PCI Bus ID: %s\n", strBuffer);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(intBuffer), intBuffer);
        vlrprintf("    Compute Capability: %d, %d\n", intBuffer[0], intBuffer[1]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    TCC (Tesla Compute Cluster) Driver: %s\n", intBuffer[0] ? "Yes" : "No");

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(sizeValue), &sizeValue);
        vlrprintf("    Total Memory: %llu [Byte]\n", sizeValue);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    Clock Rate: %d [kHz]\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    Max Threads per Block: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    Multi Processor Count: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    Max Hardware Texture Count: %d\n", intBuffer[0]);

        rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(intBuffer[0]), &intBuffer[0]);
        vlrprintf("    Execution Timeout Enabled: %s\n", intBuffer[0] ? "Yes" : "No");
    }
    vlrprintf("----------------------------------------------------------------\n");

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrGetDeviceName(uint32_t index, char* name, uint32_t bufferLength) {
    checkError(rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_NAME, bufferLength, name));

    return VLR_ERROR_NO_ERROR;
}



VLR_API const char* vlrGetErrorMessage(VLRResult code) {
    switch (code) {
    case VLR_ERROR_NO_ERROR:
        return "No Error";
    case VLR_ERROR_INVALID_CONTEXT:
        return "Invalid Context";
    case VLR_ERROR_INVALID_TYPE:
        return "Invalid Type";
    case VLR_ERROR_INCOMPATIBLE_NODE_TYPE:
        return "Incompatible Node Type";
    default:
        VLRAssert_ShouldNotBeCalled();
        break;
    }

    return "";
}



VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices) {
    *context = new VLR::Context(logging, enableRTX, maxCallableDepth, stackSize, devices, numDevices);

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

VLR_API VLRResult vlrContextGetOutputBufferSize(VLRContext context, uint32_t* width, uint32_t* height) {
    context->getOutputBufferSize(width, height);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCamera camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>())
        return VLR_ERROR_INVALID_TYPE;
    context->render(*scene, camera, shrinkCoeff, firstFrame, numAccumFrames);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrContextDebugRender(VLRContext context, VLRScene scene, VLRCamera camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>())
        return VLR_ERROR_INVALID_TYPE;
    context->debugRender(*scene, camera, renderMode, shrinkCoeff, firstFrame, numAccumFrames);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrImage2DGetWidth(VLRImage2D image, uint32_t* width) {
    if (!image->isMemberOf<VLR::Image2D>())
        return VLR_ERROR_INVALID_TYPE;
    *width = image->getWidth();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DGetHeight(VLRImage2D image, uint32_t* height) {
    if (!image->isMemberOf<VLR::Image2D>())
        return VLR_ERROR_INVALID_TYPE;
    *height = image->getHeight();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DGetStride(VLRImage2D image, uint32_t* stride) {
    if (!image->isMemberOf<VLR::Image2D>())
        return VLR_ERROR_INVALID_TYPE;
    *stride = image->getStride();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DGetDataFormat(VLRImage2D image, VLRDataFormat* format) {
    if (!image->isMemberOf<VLR::Image2D>())
        return VLR_ERROR_INVALID_TYPE;
    *format = (VLRDataFormat)image->getDataFormat();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DHasAlpha(VLRImage2D image, bool* hasAlpha) {
    if (!image->isMemberOf<VLR::Image2D>())
        return VLR_ERROR_INVALID_TYPE;
    *hasAlpha = image->hasAlpha();

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                         uint8_t* linearData, uint32_t width, uint32_t height,
                                         VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    *image = new VLR::LinearImage2D(*context, linearData, width, height, format, spectrumType, colorSpace);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image) {
    if (!image->is<VLR::LinearImage2D>())
        return VLR_ERROR_INVALID_TYPE;
    delete image;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrBlockCompressedImage2DCreate(VLRContext context, VLRBlockCompressedImage2D* image,
                                                  uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                                  VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    *image = new VLR::BlockCompressedImage2D(*context, data, sizes, mipCount, width, height, dataFormat, spectrumType, colorSpace);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrBlockCompressedImage2DDestroy(VLRContext context, VLRBlockCompressedImage2D image) {
    if (!image->is<VLR::BlockCompressedImage2D>())
        return VLR_ERROR_INVALID_TYPE;
    delete image;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrShaderNodeGetSocket(VLRShaderNode node, VLRShaderNodeSocketType socketType, uint32_t index,
                                         VLRShaderNodeSocketInfo* socketInfo) {
    if (!node->isMemberOf<VLR::ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    VLR::ShaderNodeSocketIdentifier socketID = node->getSocket(socketType, index);
    *socketInfo = socketID.getSocketInfo();

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrGeometryShaderNodeCreate(VLRContext context, VLRGeometryShaderNode* node) {
    *node = VLR::GeometryShaderNode::getInstance(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrGeometryShaderNodeDestroy(VLRContext context, VLRGeometryShaderNode node) {
    if (!node->is<VLR::GeometryShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    //delete node;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloatShaderNodeCreate(VLRContext context, VLRFloatShaderNode* node) {
    *node = new VLR::FloatShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloatShaderNodeDestroy(VLRContext context, VLRFloatShaderNode node) {
    if (!node->is<VLR::FloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloatShaderNodeSetNode0(VLRFloatShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::FloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode0(VLR::ShaderNodeSocketIdentifier(node0, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloatShaderNodeSetImmediateValue0(VLRFloatShaderNode node, float value) {
    if (!node->is<VLR::FloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue0(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloat2ShaderNodeCreate(VLRContext context, VLRFloat2ShaderNode* node) {
    *node = new VLR::Float2ShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat2ShaderNodeDestroy(VLRContext context, VLRFloat2ShaderNode node) {
    if (!node->is<VLR::Float2ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetNode0(VLRFloat2ShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float2ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode0(VLR::ShaderNodeSocketIdentifier(node0, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue0(VLRFloat2ShaderNode node, float value) {
    if (!node->is<VLR::Float2ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue0(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetNode1(VLRFloat2ShaderNode node, VLRShaderNode node1, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float2ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode1(VLR::ShaderNodeSocketIdentifier(node1, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue1(VLRFloat2ShaderNode node, float value) {
    if (!node->is<VLR::Float2ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue1(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloat3ShaderNodeCreate(VLRContext context, VLRFloat3ShaderNode* node) {
    *node = new VLR::Float3ShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeDestroy(VLRContext context, VLRFloat3ShaderNode node) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetNode0(VLRFloat3ShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode0(VLR::ShaderNodeSocketIdentifier(node0, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue0(VLRFloat3ShaderNode node, float value) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue0(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetNode1(VLRFloat3ShaderNode node, VLRShaderNode node1, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode1(VLR::ShaderNodeSocketIdentifier(node1, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue1(VLRFloat3ShaderNode node, float value) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue1(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetNode2(VLRFloat3ShaderNode node, VLRShaderNode node2, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode2(VLR::ShaderNodeSocketIdentifier(node2, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue2(VLRFloat3ShaderNode node, float value) {
    if (!node->is<VLR::Float3ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue2(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrFloat4ShaderNodeCreate(VLRContext context, VLRFloat4ShaderNode* node) {
    *node = new VLR::Float4ShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeDestroy(VLRContext context, VLRFloat4ShaderNode node) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode0(VLRFloat4ShaderNode node, VLRShaderNode node0, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode0(VLR::ShaderNodeSocketIdentifier(node0, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue0(VLRFloat4ShaderNode node, float value) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue0(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode1(VLRFloat4ShaderNode node, VLRShaderNode node1, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode1(VLR::ShaderNodeSocketIdentifier(node1, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue1(VLRFloat4ShaderNode node, float value) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue1(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode2(VLRFloat4ShaderNode node, VLRShaderNode node2, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode2(VLR::ShaderNodeSocketIdentifier(node2, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue2(VLRFloat4ShaderNode node, float value) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue2(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode3(VLRFloat4ShaderNode node, VLRShaderNode node3, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNode3(VLR::ShaderNodeSocketIdentifier(node3, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue3(VLRFloat4ShaderNode node, float value) {
    if (!node->is<VLR::Float4ShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValue3(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeCreate(VLRContext context, VLRScaleAndOffsetFloatShaderNode* node) {
    *node = new VLR::ScaleAndOffsetFloatShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeDestroy(VLRContext context, VLRScaleAndOffsetFloatShaderNode node) {
    if (!node->is<VLR::ScaleAndOffsetFloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeValue(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNode nodeValue, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::ScaleAndOffsetFloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNodeValue(VLR::ShaderNodeSocketIdentifier(nodeValue, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeScale(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNode nodeScale, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::ScaleAndOffsetFloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNodeScale(VLR::ShaderNodeSocketIdentifier(nodeScale, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeOffset(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNode nodeOffset, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::ScaleAndOffsetFloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNodeOffset(VLR::ShaderNodeSocketIdentifier(nodeOffset, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetImmediateValueScale(VLRScaleAndOffsetFloatShaderNode node, float value) {
    if (!node->is<VLR::ScaleAndOffsetFloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueScale(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetImmediateValueOffset(VLRScaleAndOffsetFloatShaderNode node, float value) {
    if (!node->is<VLR::ScaleAndOffsetFloatShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueOffset(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrTripletSpectrumShaderNodeCreate(VLRContext context, VLRTripletSpectrumShaderNode* node) {
    *node = new VLR::TripletSpectrumShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeDestroy(VLRContext context, VLRTripletSpectrumShaderNode node) {
    if (!node->is<VLR::TripletSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueSpectrumType(VLRTripletSpectrumShaderNode node, VLRSpectrumType spectrumType) {
    if (!node->is<VLR::TripletSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueSpectrumType(spectrumType);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueColorSpace(VLRTripletSpectrumShaderNode node, VLRColorSpace colorSpace) {
    if (!node->is<VLR::TripletSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueColorSpace(colorSpace);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueTriplet(VLRTripletSpectrumShaderNode node, float e0, float e1, float e2) {
    if (!node->is<VLR::TripletSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueTriplet(e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeCreate(VLRContext context, VLRRegularSampledSpectrumShaderNode* node) {
    *node = new VLR::RegularSampledSpectrumShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeDestroy(VLRContext context, VLRRegularSampledSpectrumShaderNode node) {
    if (!node->is<VLR::RegularSampledSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeSetImmediateValueSpectrum(VLRRegularSampledSpectrumShaderNode node, VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples) {
    if (!node->is<VLR::RegularSampledSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueSpectrum(spectrumType, minLambda, maxLambda, values, numSamples);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeCreate(VLRContext context, VLRIrregularSampledSpectrumShaderNode* node) {
    *node = new VLR::IrregularSampledSpectrumShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeDestroy(VLRContext context, VLRIrregularSampledSpectrumShaderNode node) {
    if (!node->is<VLR::IrregularSampledSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeSetImmediateValueSpectrum(VLRIrregularSampledSpectrumShaderNode node, VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples) {
    if (!node->is<VLR::IrregularSampledSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueSpectrum(spectrumType, lambdas, values, numSamples);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrVector3DToSpectrumShaderNodeCreate(VLRContext context, VLRVector3DToSpectrumShaderNode* node) {
    *node = new VLR::Vector3DToSpectrumShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrVector3DToSpectrumShaderNodeDestroy(VLRContext context, VLRVector3DToSpectrumShaderNode node) {
    if (!node->is<VLR::Vector3DToSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrVector3DToSpectrumShaderNodeSetNodeVector3D(VLRVector3DToSpectrumShaderNode node, VLRShaderNode nodeVector3D, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Vector3DToSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNodeVector3D(VLR::ShaderNodeSocketIdentifier(nodeVector3D, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrVector3DToSpectrumShaderNodeSetImmediateValueVector3D(VLRVector3DToSpectrumShaderNode node, const VLRVector3D* value) {
    if (!node->is<VLR::Vector3DToSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueVector3D(VLR::Vector3D(value->x, value->y, value->z));

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrVector3DToSpectrumShaderNodeSetImmediateValueSpectrumTypeAndColorSpace(VLRVector3DToSpectrumShaderNode node, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    if (!node->is<VLR::Vector3DToSpectrumShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImmediateValueSpectrumTypeAndColorSpace(spectrumType, colorSpace);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeCreate(VLRContext context, VLRScaleAndOffsetUVTextureMap2DShaderNode* node) {
    *node = new VLR::ScaleAndOffsetUVTextureMap2DShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeDestroy(VLRContext context, VLRScaleAndOffsetUVTextureMap2DShaderNode node) {
    if (!node->is<VLR::ScaleAndOffsetUVTextureMap2DShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeSetValues(VLRScaleAndOffsetUVTextureMap2DShaderNode node, const float offset[2], const float scale[2]) {
    if (!node->is<VLR::ScaleAndOffsetUVTextureMap2DShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setValues(offset, scale);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrImage2DTextureShaderNodeCreate(VLRContext context, VLRImage2DTextureShaderNode* node) {
    *node = new VLR::Image2DTextureShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeDestroy(VLRContext context, VLRImage2DTextureShaderNode node) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetImage(VLRImage2DTextureShaderNode node, VLRImage2D image) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImage(image);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetFilterMode(VLRImage2DTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetWrapMode(VLRImage2DTextureShaderNode node, VLRTextureWrapMode x, VLRTextureWrapMode y) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setTextureWrapMode(x, y);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetNodeTexCoord(VLRImage2DTextureShaderNode node, VLRShaderNode nodeTexCoord, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::Image2DTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNodeTexCoord(VLR::ShaderNodeSocketIdentifier(nodeTexCoord, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrEnvironmentTextureShaderNodeCreate(VLRContext context, VLREnvironmentTextureShaderNode* node) {
    *node = new VLR::EnvironmentTextureShaderNode(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeDestroy(VLRContext context, VLREnvironmentTextureShaderNode node) {
    if (!node->is<VLR::EnvironmentTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetImage(VLREnvironmentTextureShaderNode node, VLRImage2D image) {
    if (!node->is<VLR::EnvironmentTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setImage(image);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetFilterMode(VLREnvironmentTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    if (!node->is<VLR::EnvironmentTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setTextureFilterMode(minification, magnification, mipmapping);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetWrapMode(VLREnvironmentTextureShaderNode node, VLRTextureWrapMode x, VLRTextureWrapMode y) {
    if (!node->is<VLR::EnvironmentTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    node->setTextureWrapMode(x, y);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetNodeTexCoord(VLREnvironmentTextureShaderNode node, VLRShaderNode nodeTexCoord, VLRShaderNodeSocketInfo socketInfo) {
    if (!node->is<VLR::EnvironmentTextureShaderNode>())
        return VLR_ERROR_INVALID_TYPE;
    if (!node->setNodeTexCoord(VLR::ShaderNodeSocketIdentifier(nodeTexCoord, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material) {
    *material = new VLR::MatteSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material) {
    if (!material->is<VLR::MatteSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMatteSurfaceMaterialSetNodeAlbedo(VLRMatteSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MatteSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeAlbedo(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMatteSurfaceMaterialSetImmediateValueAlbedo(VLRMatteSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::MatteSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueAlbedo(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialCreate(VLRContext context, VLRSpecularReflectionSurfaceMaterial* material) {
    *material = new VLR::SpecularReflectionSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialDestroy(VLRContext context, VLRSpecularReflectionSurfaceMaterial material) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeCoeffR(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueCoeffR(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeEta(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEta(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEta(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNode_k(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo)
{
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNode_k(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::SpecularReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValue_k(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}




VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialCreate(VLRContext context, VLRSpecularScatteringSurfaceMaterial* material) {
    *material = new VLR::SpecularScatteringSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialDestroy(VLRContext context, VLRSpecularScatteringSurfaceMaterial material) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeCoeff(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueCoeff(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEtaExt(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEtaExt(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEtaInt(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::SpecularScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEtaInt(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialCreate(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial* material) {
    *material = new VLR::MicrofacetReflectionSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial material) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEta(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEta(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNode_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNode_k(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValue_k(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeRoughnessAnisotropyRotation(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetReflectionSurfaceMaterial material, float value) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueRoughness(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetReflectionSurfaceMaterial material, float value) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueAnisotropy(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetReflectionSurfaceMaterial material, float value) {
    if (!material->is<VLR::MicrofacetReflectionSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueRotation(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialCreate(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial* material) {
    *material = new VLR::MicrofacetScatteringSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial material) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeCoeff(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueCoeff(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEtaExt(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEtaExt(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEtaInt(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEtaInt(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeRoughnessAnisotropyRotation(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetScatteringSurfaceMaterial material, float value) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueRoughness(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetScatteringSurfaceMaterial material, float value) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueAnisotropy(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetScatteringSurfaceMaterial material, float value) {
    if (!material->is<VLR::MicrofacetScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueRotation(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialCreate(VLRContext context, VLRLambertianScatteringSurfaceMaterial* material) {
    *material = new VLR::LambertianScatteringSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialDestroy(VLRContext context, VLRLambertianScatteringSurfaceMaterial material) {
    if (!material->is<VLR::LambertianScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::LambertianScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeCoeff(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::LambertianScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueCoeff(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeF0(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::LambertianScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeF0(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueF0(VLRLambertianScatteringSurfaceMaterial material, float value) {
    if (!material->is<VLR::LambertianScatteringSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueF0(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material) {
    *material = new VLR::UE4SurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetNodeBaseColor(VLRUE4SurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeBaseColor(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueBaseColor(VLRUE4SurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueBaseColor(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetNodeOcclusionRoughnessMetallic(VLRUE4SurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeOcclusionRoughnessMetallic(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueOcclusion(VLRUE4SurfaceMaterial material, float value) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueOcclusion(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueRoughness(VLRUE4SurfaceMaterial material, float value) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueRoughness(value);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueMetallic(VLRUE4SurfaceMaterial material, float value) {
    if (!material->is<VLR::UE4SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueMetallic(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrOldStyleSurfaceMaterialCreate(VLRContext context, VLROldStyleSurfaceMaterial* material) {
    *material = new VLR::OldStyleSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSurfaceMaterialDestroy(VLRContext context, VLROldStyleSurfaceMaterial material) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeDiffuseColor(VLROldStyleSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeDiffuseColor(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueDiffuseColor(VLROldStyleSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueDiffuseColor(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeSpecularColor(VLROldStyleSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeSpecularColor(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueSpecularColor(VLROldStyleSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueSpecularColor(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeGlossiness(VLROldStyleSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeGlossiness(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueGlossiness(VLROldStyleSurfaceMaterial material, float value) {
    if (!material->is<VLR::OldStyleSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueGlossiness(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialCreate(VLRContext context, VLRDiffuseEmitterSurfaceMaterial* material) {
    *material = new VLR::DiffuseEmitterSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialDestroy(VLRContext context, VLRDiffuseEmitterSurfaceMaterial material) {
    if (!material->is<VLR::DiffuseEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRShaderNode node, VLRShaderNodeSocketInfo socketInfo) {
    if (!material->is<VLR::DiffuseEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEmittance(VLR::ShaderNodeSocketIdentifier(node, socketInfo)))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::DiffuseEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEmittance(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueScale(VLRDiffuseEmitterSurfaceMaterial material, float value) {
    if (!material->is<VLR::DiffuseEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueScale(value);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMultiSurfaceMaterialCreate(VLRContext context, VLRMultiSurfaceMaterial* material) {
    *material = new VLR::MultiSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMultiSurfaceMaterialDestroy(VLRContext context, VLRMultiSurfaceMaterial material) {
    if (!material->is<VLR::MultiSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMultiSurfaceMaterialSetSubMaterial(VLRMultiSurfaceMaterial material, uint32_t index, VLRSurfaceMaterial mat) {
    if (!material->is<VLR::MultiSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setSubMaterial(index, mat);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialCreate(VLRContext context, VLREnvironmentEmitterSurfaceMaterial* material) {
    *material = new VLR::EnvironmentEmitterSurfaceMaterial(*context);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialDestroy(VLRContext context, VLREnvironmentEmitterSurfaceMaterial material) {
    if (!material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceTextured(VLREnvironmentEmitterSurfaceMaterial material, VLREnvironmentTextureShaderNode node) {
    if (!material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEmittanceTextured(node))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceConstant(VLREnvironmentEmitterSurfaceMaterial material, VLRShaderNode node) {
    if (!material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    if (!material->setNodeEmittanceConstant(node))
        return VLR_ERROR_INCOMPATIBLE_NODE_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance(VLREnvironmentEmitterSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    if (!material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueEmittance(colorSpace, e0, e1, e2);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueScale(VLREnvironmentEmitterSurfaceMaterial material, float value) {
    if (!material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;
    material->setImmediateValueScale(value);

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



VLR_API VLRResult vlrNodeSetName(VLRNode node, const char* name) {
    if (!node->isMemberOf<VLR::Node>())
        return VLR_ERROR_INVALID_TYPE;
    node->setName(name);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrNodeGetName(VLRNode node, const char** name) {
    if (!node->isMemberOf<VLR::Node>())
        return VLR_ERROR_INVALID_TYPE;
    *name = node->getName().c_str();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrNodeGetType(VLRNode node, VLRNodeType* type) {
    if (!node->isMemberOf<VLR::Node>())
        return VLR_ERROR_INVALID_TYPE;
    *type = node->getType();

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
                                                             VLRSurfaceMaterial material,
                                                             VLRShaderNode nodeNormal, VLRShaderNodeSocketInfo nodeNormalSocketInfo,
                                                             VLRShaderNode nodeAlpha, VLRShaderNodeSocketInfo nodeAlphaSocketInfo,
                                                             VLRTangentType tangentType) {
    if (!surfaceNode->is<VLR::TriangleMeshSurfaceNode>())
        return VLR_ERROR_INVALID_TYPE;

    std::vector<uint32_t> vecIndices;
    vecIndices.resize(numIndices);
    std::copy_n(indices, numIndices, vecIndices.data());

    if (!material->isMemberOf<VLR::SurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;

    surfaceNode->addMaterialGroup(std::move(vecIndices), material, 
                                  VLR::ShaderNodeSocketIdentifier(nodeNormal, nodeNormalSocketInfo),
                                  VLR::ShaderNodeSocketIdentifier(nodeAlpha, nodeAlphaSocketInfo),
                                  tangentType);

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

VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLREnvironmentEmitterSurfaceMaterial material, float rotationPhi) {
    if (!scene->is<VLR::Scene>() || !material->is<VLR::EnvironmentEmitterSurfaceMaterial>())
        return VLR_ERROR_INVALID_TYPE;

    scene->setEnvironment(material, rotationPhi);

    return VLR_ERROR_NO_ERROR;
}




VLR_API VLRResult vlrCameraGetType(VLRCamera camera, VLRCameraType* type) {
    if (!camera->isMemberOf<VLR::Camera>())
        return VLR_ERROR_INVALID_TYPE;
    *type = camera->getType();

    return VLR_ERROR_NO_ERROR;
}




VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera) {
    *camera = new VLR::PerspectiveCamera(*context);

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

VLR_API VLRResult vlrPerspectiveCameraSetAspectRatio(VLRPerspectiveCamera camera, float aspect) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->setAspectRatio(aspect);

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

VLR_API VLRResult vlrPerspectiveCameraGetPosition(VLRPerspectiveCamera camera, VLRPoint3D* position) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getPosition((VLR::Point3D*)position);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraGetOrientation(VLRPerspectiveCamera camera, VLRQuaternion* orientation) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getOrientation((VLR::Quaternion*)orientation);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraGetAspectRatio(VLRPerspectiveCamera camera, float* aspect) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getAspectRatio(aspect);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraGetSensitivity(VLRPerspectiveCamera camera, float* sensitivity) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getSensitivity(sensitivity);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraGetFovY(VLRPerspectiveCamera camera, float* fovY) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getFovY(fovY);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraGetLensRadius(VLRPerspectiveCamera camera, float* lensRadius) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getLensRadius(lensRadius);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrPerspectiveCameraGetObjectPlaneDistance(VLRPerspectiveCamera camera, float* distance) {
    if (!camera->is<VLR::PerspectiveCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getObjectPlaneDistance(distance);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera) {
    *camera = new VLR::EquirectangularCamera(*context);

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

VLR_API VLRResult vlrEquirectangularCameraGetPosition(VLREquirectangularCamera camera, VLRPoint3D* position) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getPosition((VLR::Point3D*)position);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraGetOrientation(VLREquirectangularCamera camera, VLRQuaternion* orientation) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getOrientation((VLR::Quaternion*)orientation);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraGetSensitivity(VLREquirectangularCamera camera, float* sensitivity) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getSensitivity(sensitivity);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrEquirectangularCameraGetAngles(VLREquirectangularCamera camera, float* phiAngle, float* thetaAngle) {
    if (!camera->is<VLR::EquirectangularCamera>())
        return VLR_ERROR_INVALID_TYPE;
    camera->getAngles(phiAngle, thetaAngle);

    return VLR_ERROR_NO_ERROR;
}

