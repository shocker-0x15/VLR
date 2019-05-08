#pragma once

#include "scene.h"

#define VLR_DEFINE_C_ALIAS(name) typedef VLR::name* VLR ## name

VLR_DEFINE_C_ALIAS(Object);

VLR_DEFINE_C_ALIAS(Context);

VLR_DEFINE_C_ALIAS(Image2D);
VLR_DEFINE_C_ALIAS(LinearImage2D);
VLR_DEFINE_C_ALIAS(BlockCompressedImage2D);

VLR_DEFINE_C_ALIAS(ShaderNode);
VLR_DEFINE_C_ALIAS(GeometryShaderNode);
VLR_DEFINE_C_ALIAS(Float2ShaderNode);
VLR_DEFINE_C_ALIAS(Float3ShaderNode);
VLR_DEFINE_C_ALIAS(Float4ShaderNode);
VLR_DEFINE_C_ALIAS(ScaleAndOffsetFloatShaderNode);
VLR_DEFINE_C_ALIAS(TripletSpectrumShaderNode);
VLR_DEFINE_C_ALIAS(RegularSampledSpectrumShaderNode);
VLR_DEFINE_C_ALIAS(IrregularSampledSpectrumShaderNode);
VLR_DEFINE_C_ALIAS(Float3ToSpectrumShaderNode);
VLR_DEFINE_C_ALIAS(ScaleAndOffsetUVTextureMap2DShaderNode);
VLR_DEFINE_C_ALIAS(Image2DTextureShaderNode);
VLR_DEFINE_C_ALIAS(EnvironmentTextureShaderNode);

VLR_DEFINE_C_ALIAS(SurfaceMaterial);
VLR_DEFINE_C_ALIAS(MatteSurfaceMaterial);
VLR_DEFINE_C_ALIAS(SpecularReflectionSurfaceMaterial);
VLR_DEFINE_C_ALIAS(SpecularScatteringSurfaceMaterial);
VLR_DEFINE_C_ALIAS(MicrofacetReflectionSurfaceMaterial);
VLR_DEFINE_C_ALIAS(MicrofacetScatteringSurfaceMaterial);
VLR_DEFINE_C_ALIAS(LambertianScatteringSurfaceMaterial);
VLR_DEFINE_C_ALIAS(UE4SurfaceMaterial);
VLR_DEFINE_C_ALIAS(OldStyleSurfaceMaterial);
VLR_DEFINE_C_ALIAS(DiffuseEmitterSurfaceMaterial);
VLR_DEFINE_C_ALIAS(MultiSurfaceMaterial);
VLR_DEFINE_C_ALIAS(EnvironmentEmitterSurfaceMaterial);

VLR_DEFINE_C_ALIAS(Transform);
typedef VLR::Transform const* VLRTransformConst;
VLR_DEFINE_C_ALIAS(StaticTransform);

VLR_DEFINE_C_ALIAS(Node);
VLR_DEFINE_C_ALIAS(SurfaceNode);
VLR_DEFINE_C_ALIAS(TriangleMeshSurfaceNode);
VLR_DEFINE_C_ALIAS(InternalNode);
VLR_DEFINE_C_ALIAS(Scene);

VLR_DEFINE_C_ALIAS(Camera);
VLR_DEFINE_C_ALIAS(PerspectiveCamera);
VLR_DEFINE_C_ALIAS(EquirectangularCamera);

#include <VLR.h>



#define VLR_RETURN_INVALID_INSTANCE(var, type) \
    if (var == nullptr) \
        return VLRResult_InvalidArgument; \
    if (!var->isMemberOf<type>()) \
        return VLRResult_InvalidInstance

template <typename T>
inline bool validateArgument(const T* obj) {
    if (obj == nullptr)
        return false;
    if (!obj->isMemberOf<T>())
        return false;
    return true;
}



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

    return VLRResult_NoError;
}

VLR_API VLRResult vlrGetDeviceName(uint32_t index, char* name, uint32_t bufferLength) {
    checkError(rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_NAME, bufferLength, name));

    return VLRResult_NoError;
}



VLR_API const char* vlrGetErrorMessage(VLRResult code) {
    switch (code) {
    case VLRResult_NoError:
        return "No Error";
    case VLRResult_InvalidContext:
        return "Invalid Context";
    case VLRResult_InvalidInstance:
        return "Invalid Instance";
    case VLRResult_InvalidArgument:
        return "Invalid Argument";
    case VLRResult_IncompatibleNodeType:
        return "Incompatible Node Type";
    default:
        VLRAssert_ShouldNotBeCalled();
        break;
    }

    return "";
}



VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices) {
    *context = new VLR::Context(logging, enableRTX, maxCallableDepth, stackSize, devices, numDevices);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrDestroyContext(VLRContext context) {
    delete context;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrContextBindOutputBuffer(VLRContext context, uint32_t width, uint32_t height, uint32_t bufferID) {
    context->bindOutputBuffer(width, height, bufferID);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrContextMapOutputBuffer(VLRContext context, void** ptr) {
    if (ptr == nullptr)
        return VLRResult_InvalidArgument;
    *ptr = context->mapOutputBuffer();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrContextUnmapOutputBuffer(VLRContext context) {
    context->unmapOutputBuffer();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrContextGetOutputBufferSize(VLRContext context, uint32_t* width, uint32_t* height) {
    if (width == nullptr || height == nullptr)
        return VLRResult_InvalidArgument;

    context->getOutputBufferSize(width, height);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCamera camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>() || numAccumFrames == nullptr)
        return VLRResult_InvalidArgument;

    context->render(*scene, camera, shrinkCoeff, firstFrame, numAccumFrames);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrContextDebugRender(VLRContext context, VLRScene scene, VLRCamera camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>() || numAccumFrames == nullptr)
        return VLRResult_InvalidArgument;

    context->debugRender(*scene, camera, renderMode, shrinkCoeff, firstFrame, numAccumFrames);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrImage2DGetWidth(VLRImage2D image, uint32_t* width) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);
    if (width == nullptr)
        return VLRResult_InvalidArgument;

    *width = image->getWidth();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DGetHeight(VLRImage2D image, uint32_t* height) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);
    if (height == nullptr)
        return VLRResult_InvalidArgument;

    *height = image->getHeight();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DGetStride(VLRImage2D image, uint32_t* stride) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

    *stride = image->getStride();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DGetOriginalDataFormat(VLRImage2D image, VLRDataFormat* format) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

    *format = (VLRDataFormat)image->getOriginalDataFormat();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DOriginalHasAlpha(VLRImage2D image, bool* hasAlpha) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

    *hasAlpha = image->originalHasAlpha();

    return VLRResult_NoError;
}



VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                         uint8_t* linearData, uint32_t width, uint32_t height,
                                         VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    if (image == nullptr || linearData == nullptr)
        return VLRResult_InvalidArgument;

    *image = new VLR::LinearImage2D(*context, linearData, width, height, format, spectrumType, colorSpace);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::LinearImage2D);

    delete image;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrBlockCompressedImage2DCreate(VLRContext context, VLRBlockCompressedImage2D* image,
                                                  uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                                  VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    if (image == nullptr || data == nullptr || sizes == nullptr)
        return VLRResult_InvalidArgument;
    for (int m = 0; m < mipCount; ++m)
        if (data[m] == nullptr)
            return VLRResult_InvalidArgument;

    *image = new VLR::BlockCompressedImage2D(*context, data, sizes, mipCount, width, height, dataFormat, spectrumType, colorSpace);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrBlockCompressedImage2DDestroy(VLRContext context, VLRBlockCompressedImage2D image) {
    VLR_RETURN_INVALID_INSTANCE(image, VLR::BlockCompressedImage2D);

    delete image;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrShaderNodeGetSocket(VLRShaderNode node, VLRShaderNodeSocketType socketType, uint32_t option,
                                         VLRShaderNodeSocket* socket) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ShaderNode);
    if (socket == nullptr)
        return VLRResult_InvalidArgument;

    *socket = node->getSocket(socketType, option).getOpaqueType();

    return VLRResult_NoError;
}



VLR_API VLRResult vlrGeometryShaderNodeCreate(VLRContext context, VLRGeometryShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = VLR::GeometryShaderNode::getInstance(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrGeometryShaderNodeDestroy(VLRContext context, VLRGeometryShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::GeometryShaderNode);

    //delete node;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrFloat2ShaderNodeCreate(VLRContext context, VLRFloat2ShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::Float2ShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat2ShaderNodeDestroy(VLRContext context, VLRFloat2ShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float2ShaderNode);

    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetNode0(VLRFloat2ShaderNode node, VLRShaderNodeSocket node0) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float2ShaderNode);

    if (!node->setNode0(VLR::ShaderNodeSocket(node0)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue0(VLRFloat2ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float2ShaderNode);

    node->setImmediateValue0(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetNode1(VLRFloat2ShaderNode node, VLRShaderNodeSocket node1) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float2ShaderNode);

    if (!node->setNode1(VLR::ShaderNodeSocket(node1)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat2ShaderNodeSetImmediateValue1(VLRFloat2ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float2ShaderNode);

    node->setImmediateValue1(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrFloat3ShaderNodeCreate(VLRContext context, VLRFloat3ShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::Float3ShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeDestroy(VLRContext context, VLRFloat3ShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetNode0(VLRFloat3ShaderNode node, VLRShaderNodeSocket node0) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    if (!node->setNode0(VLR::ShaderNodeSocket(node0)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue0(VLRFloat3ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    node->setImmediateValue0(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetNode1(VLRFloat3ShaderNode node, VLRShaderNodeSocket node1) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    if (!node->setNode1(VLR::ShaderNodeSocket(node1)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue1(VLRFloat3ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    node->setImmediateValue1(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetNode2(VLRFloat3ShaderNode node, VLRShaderNodeSocket node2) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    if (!node->setNode2(VLR::ShaderNodeSocket(node2)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ShaderNodeSetImmediateValue2(VLRFloat3ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ShaderNode);

    node->setImmediateValue2(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrFloat4ShaderNodeCreate(VLRContext context, VLRFloat4ShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::Float4ShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeDestroy(VLRContext context, VLRFloat4ShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode0(VLRFloat4ShaderNode node, VLRShaderNodeSocket node0) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    if (!node->setNode0(VLR::ShaderNodeSocket(node0)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue0(VLRFloat4ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    node->setImmediateValue0(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode1(VLRFloat4ShaderNode node, VLRShaderNodeSocket node1) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    if (!node->setNode1(VLR::ShaderNodeSocket(node1)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue1(VLRFloat4ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    node->setImmediateValue1(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode2(VLRFloat4ShaderNode node, VLRShaderNodeSocket node2) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    if (!node->setNode2(VLR::ShaderNodeSocket(node2)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue2(VLRFloat4ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    node->setImmediateValue2(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetNode3(VLRFloat4ShaderNode node, VLRShaderNodeSocket node3) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    if (!node->setNode3(VLR::ShaderNodeSocket(node3)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat4ShaderNodeSetImmediateValue3(VLRFloat4ShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float4ShaderNode);

    node->setImmediateValue3(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeCreate(VLRContext context, VLRScaleAndOffsetFloatShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::ScaleAndOffsetFloatShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeDestroy(VLRContext context, VLRScaleAndOffsetFloatShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetFloatShaderNode);

    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeValue(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNodeSocket nodeValue) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetFloatShaderNode);

    if (!node->setNodeValue(VLR::ShaderNodeSocket(nodeValue)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeScale(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNodeSocket nodeScale) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetFloatShaderNode);

    if (!node->setNodeScale(VLR::ShaderNodeSocket(nodeScale)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetNodeOffset(VLRScaleAndOffsetFloatShaderNode node, VLRShaderNodeSocket nodeOffset) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetFloatShaderNode);

    if (!node->setNodeOffset(VLR::ShaderNodeSocket(nodeOffset)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetImmediateValueScale(VLRScaleAndOffsetFloatShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetFloatShaderNode);

    node->setImmediateValueScale(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetFloatShaderNodeSetImmediateValueOffset(VLRScaleAndOffsetFloatShaderNode node, float value) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetFloatShaderNode);

    node->setImmediateValueOffset(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrTripletSpectrumShaderNodeCreate(VLRContext context, VLRTripletSpectrumShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::TripletSpectrumShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeDestroy(VLRContext context, VLRTripletSpectrumShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::TripletSpectrumShaderNode);

    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueSpectrumType(VLRTripletSpectrumShaderNode node, VLRSpectrumType spectrumType) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::TripletSpectrumShaderNode);
;
    node->setImmediateValueSpectrumType(spectrumType);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueColorSpace(VLRTripletSpectrumShaderNode node, VLRColorSpace colorSpace) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::TripletSpectrumShaderNode);
;
    node->setImmediateValueColorSpace(colorSpace);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTripletSpectrumShaderNodeSetImmediateValueTriplet(VLRTripletSpectrumShaderNode node, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::TripletSpectrumShaderNode);
;
    node->setImmediateValueTriplet(e0, e1, e2);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeCreate(VLRContext context, VLRRegularSampledSpectrumShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::RegularSampledSpectrumShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeDestroy(VLRContext context, VLRRegularSampledSpectrumShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::RegularSampledSpectrumShaderNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrRegularSampledSpectrumShaderNodeSetImmediateValueSpectrum(VLRRegularSampledSpectrumShaderNode node, VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::RegularSampledSpectrumShaderNode);
    if (minLambda >= maxLambda || values == nullptr)
        return VLRResult_InvalidArgument;
;
    node->setImmediateValueSpectrum(spectrumType, minLambda, maxLambda, values, numSamples);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeCreate(VLRContext context, VLRIrregularSampledSpectrumShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::IrregularSampledSpectrumShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeDestroy(VLRContext context, VLRIrregularSampledSpectrumShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::IrregularSampledSpectrumShaderNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrIrregularSampledSpectrumShaderNodeSetImmediateValueSpectrum(VLRIrregularSampledSpectrumShaderNode node, VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::IrregularSampledSpectrumShaderNode);
    if (lambdas == nullptr || values == nullptr)
        return VLRResult_InvalidArgument;
;
    node->setImmediateValueSpectrum(spectrumType, lambdas, values, numSamples);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeCreate(VLRContext context, VLRFloat3ToSpectrumShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::Float3ToSpectrumShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeDestroy(VLRContext context, VLRFloat3ToSpectrumShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ToSpectrumShaderNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeSetNodeVector3D(VLRFloat3ToSpectrumShaderNode node, VLRShaderNodeSocket nodeFloat3) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ToSpectrumShaderNode);
;
    if (!node->setNodeFloat3(VLR::ShaderNodeSocket(nodeFloat3)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeSetImmediateValueVector3D(VLRFloat3ToSpectrumShaderNode node, const float value[3]) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ToSpectrumShaderNode);
;
    node->setImmediateValueFloat3(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrFloat3ToSpectrumShaderNodeSetImmediateValueSpectrumTypeAndColorSpace(VLRFloat3ToSpectrumShaderNode node, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Float3ToSpectrumShaderNode);
;
    node->setImmediateValueSpectrumTypeAndColorSpace(spectrumType, colorSpace);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeCreate(VLRContext context, VLRScaleAndOffsetUVTextureMap2DShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::ScaleAndOffsetUVTextureMap2DShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeDestroy(VLRContext context, VLRScaleAndOffsetUVTextureMap2DShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetUVTextureMap2DShaderNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrScaleAndOffsetUVTextureMap2DShaderNodeSetValues(VLRScaleAndOffsetUVTextureMap2DShaderNode node, const float offset[2], const float scale[2]) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::ScaleAndOffsetUVTextureMap2DShaderNode);
    if (offset == nullptr || scale == nullptr)
        return VLRResult_InvalidArgument;
;
    node->setValues(offset, scale);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrImage2DTextureShaderNodeCreate(VLRContext context, VLRImage2DTextureShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::Image2DTextureShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeDestroy(VLRContext context, VLRImage2DTextureShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Image2DTextureShaderNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetImage(VLRImage2DTextureShaderNode node, VLRImage2D image) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Image2DTextureShaderNode);
    if (!validateArgument<VLR::Image2D>(image))
        return VLRResult_InvalidArgument;
;
    node->setImage(image);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetFilterMode(VLRImage2DTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Image2DTextureShaderNode);
;
    node->setTextureFilterMode(minification, magnification, mipmapping);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetWrapMode(VLRImage2DTextureShaderNode node, VLRTextureWrapMode x, VLRTextureWrapMode y) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Image2DTextureShaderNode);
;
    node->setTextureWrapMode(x, y);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrImage2DTextureShaderNodeSetNodeTexCoord(VLRImage2DTextureShaderNode node, VLRShaderNodeSocket nodeTexCoord) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Image2DTextureShaderNode);
;
    if (!node->setNodeTexCoord(VLR::ShaderNodeSocket(nodeTexCoord)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrEnvironmentTextureShaderNodeCreate(VLRContext context, VLREnvironmentTextureShaderNode* node) {
    if (node == nullptr)
        return VLRResult_InvalidArgument;

    *node = new VLR::EnvironmentTextureShaderNode(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeDestroy(VLRContext context, VLREnvironmentTextureShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::EnvironmentTextureShaderNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetImage(VLREnvironmentTextureShaderNode node, VLRImage2D image) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::EnvironmentTextureShaderNode);
    if (!validateArgument<VLR::Image2D>(image))
        return VLRResult_InvalidArgument;
;
    node->setImage(image);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetFilterMode(VLREnvironmentTextureShaderNode node, VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::EnvironmentTextureShaderNode);
;
    node->setTextureFilterMode(minification, magnification, mipmapping);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetWrapMode(VLREnvironmentTextureShaderNode node, VLRTextureWrapMode x, VLRTextureWrapMode y) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::EnvironmentTextureShaderNode);
;
    node->setTextureWrapMode(x, y);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentTextureShaderNodeSetNodeTexCoord(VLREnvironmentTextureShaderNode node, VLRShaderNodeSocket nodeTexCoord) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::EnvironmentTextureShaderNode);
;
    if (!node->setNodeTexCoord(VLR::ShaderNodeSocket(nodeTexCoord)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::MatteSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MatteSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMatteSurfaceMaterialSetNodeAlbedo(VLRMatteSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MatteSurfaceMaterial);
;
    if (!material->setNodeAlbedo(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMatteSurfaceMaterialSetImmediateValueAlbedo(VLRMatteSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MatteSurfaceMaterial);
;
    material->setImmediateValueAlbedo(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialCreate(VLRContext context, VLRSpecularReflectionSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::SpecularReflectionSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialDestroy(VLRContext context, VLRSpecularReflectionSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    if (!material->setNodeCoeffR(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueCoeffR(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    material->setImmediateValueCoeffR(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNodeEta(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    if (!material->setNodeEta(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValueEta(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    material->setImmediateValueEta(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetNode_k(VLRSpecularReflectionSurfaceMaterial material, VLRShaderNodeSocket node)
{
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    if (!material->setNode_k(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularReflectionSurfaceMaterialSetImmediateValue_k(VLRSpecularReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularReflectionSurfaceMaterial);
;
    material->setImmediateValue_k(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}




VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialCreate(VLRContext context, VLRSpecularScatteringSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::SpecularScatteringSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialDestroy(VLRContext context, VLRSpecularScatteringSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    if (!material->setNodeCoeff(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueCoeff(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    material->setImmediateValueCoeff(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    if (!material->setNodeEtaExt(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    material->setImmediateValueEtaExt(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetNodeEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    if (!material->setNodeEtaInt(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSpecularScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRSpecularScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::SpecularScatteringSurfaceMaterial);
;
    material->setImmediateValueEtaInt(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialCreate(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::MicrofacetReflectionSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetReflectionSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    if (!material->setNodeEta(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueEta(VLRMicrofacetReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    material->setImmediateValueEta(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNode_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    if (!material->setNode_k(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValue_k(VLRMicrofacetReflectionSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    material->setImmediateValue_k(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetReflectionSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    if (!material->setNodeRoughnessAnisotropyRotation(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetReflectionSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    material->setImmediateValueRoughness(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetReflectionSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    material->setImmediateValueAnisotropy(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetReflectionSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetReflectionSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetReflectionSurfaceMaterial);
;
    material->setImmediateValueRotation(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialCreate(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::MicrofacetScatteringSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialDestroy(VLRContext context, VLRMicrofacetScatteringSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    if (!material->setNodeCoeff(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueCoeff(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    material->setImmediateValueCoeff(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    if (!material->setNodeEtaExt(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaExt(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    material->setImmediateValueEtaExt(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    if (!material->setNodeEtaInt(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueEtaInt(VLRMicrofacetScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    material->setImmediateValueEtaInt(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetNodeRoughnessAnisotropyRotation(VLRMicrofacetScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    if (!material->setNodeRoughnessAnisotropyRotation(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRoughness(VLRMicrofacetScatteringSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    material->setImmediateValueRoughness(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueAnisotropy(VLRMicrofacetScatteringSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    material->setImmediateValueAnisotropy(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMicrofacetScatteringSurfaceMaterialSetImmediateValueRotation(VLRMicrofacetScatteringSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MicrofacetScatteringSurfaceMaterial);
;
    material->setImmediateValueRotation(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialCreate(VLRContext context, VLRLambertianScatteringSurfaceMaterial* material) {
    if (material == nullptr)

        return VLRResult_InvalidArgument;
    *material = new VLR::LambertianScatteringSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialDestroy(VLRContext context, VLRLambertianScatteringSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::LambertianScatteringSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::LambertianScatteringSurfaceMaterial);
;
    if (!material->setNodeCoeff(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueCoeff(VLRLambertianScatteringSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::LambertianScatteringSurfaceMaterial);
;
    material->setImmediateValueCoeff(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetNodeF0(VLRLambertianScatteringSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::LambertianScatteringSurfaceMaterial);
;
    if (!material->setNodeF0(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrLambertianScatteringSurfaceMaterialSetImmediateValueF0(VLRLambertianScatteringSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::LambertianScatteringSurfaceMaterial);
;
    material->setImmediateValueF0(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::UE4SurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetNodeBaseColor(VLRUE4SurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    if (!material->setNodeBaseColor(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueBaseColor(VLRUE4SurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    material->setImmediateValueBaseColor(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetNodeOcclusionRoughnessMetallic(VLRUE4SurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    if (!material->setNodeOcclusionRoughnessMetallic(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueOcclusion(VLRUE4SurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    material->setImmediateValueOcclusion(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueRoughness(VLRUE4SurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    material->setImmediateValueRoughness(value);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrUE4SufaceMaterialSetImmediateValueMetallic(VLRUE4SurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::UE4SurfaceMaterial);
;
    material->setImmediateValueMetallic(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrOldStyleSurfaceMaterialCreate(VLRContext context, VLROldStyleSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::OldStyleSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSurfaceMaterialDestroy(VLRContext context, VLROldStyleSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeDiffuseColor(VLROldStyleSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    if (!material->setNodeDiffuseColor(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueDiffuseColor(VLROldStyleSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    material->setImmediateValueDiffuseColor(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeSpecularColor(VLROldStyleSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    if (!material->setNodeSpecularColor(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueSpecularColor(VLROldStyleSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    material->setImmediateValueSpecularColor(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetNodeGlossiness(VLROldStyleSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    if (!material->setNodeGlossiness(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrOldStyleSufaceMaterialSetImmediateValueGlossiness(VLROldStyleSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::OldStyleSurfaceMaterial);
;
    material->setImmediateValueGlossiness(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialCreate(VLRContext context, VLRDiffuseEmitterSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::DiffuseEmitterSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialDestroy(VLRContext context, VLRDiffuseEmitterSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::DiffuseEmitterSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetNodeEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRShaderNodeSocket node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::DiffuseEmitterSurfaceMaterial);
;
    if (!material->setNodeEmittance(VLR::ShaderNodeSocket(node)))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueEmittance(VLRDiffuseEmitterSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::DiffuseEmitterSurfaceMaterial);
;
    material->setImmediateValueEmittance(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrDiffuseEmitterSurfaceMaterialSetImmediateValueScale(VLRDiffuseEmitterSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::DiffuseEmitterSurfaceMaterial);
;
    material->setImmediateValueScale(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrMultiSurfaceMaterialCreate(VLRContext context, VLRMultiSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::MultiSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMultiSurfaceMaterialDestroy(VLRContext context, VLRMultiSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MultiSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrMultiSurfaceMaterialSetSubMaterial(VLRMultiSurfaceMaterial material, uint32_t index, VLRSurfaceMaterial mat) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::MultiSurfaceMaterial);
;
    if (index >= 4 || !validateArgument<VLR::SurfaceMaterial>(mat))
        return VLRResult_InvalidArgument;
    material->setSubMaterial(index, mat);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialCreate(VLRContext context, VLREnvironmentEmitterSurfaceMaterial* material) {
    if (material == nullptr)
        return VLRResult_InvalidArgument;

    *material = new VLR::EnvironmentEmitterSurfaceMaterial(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialDestroy(VLRContext context, VLREnvironmentEmitterSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::EnvironmentEmitterSurfaceMaterial);
;
    delete material;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceTextured(VLREnvironmentEmitterSurfaceMaterial material, VLREnvironmentTextureShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::EnvironmentEmitterSurfaceMaterial);
    if (!validateArgument<VLR::EnvironmentTextureShaderNode>(node))
        return VLRResult_InvalidArgument;
;
    if (!material->setNodeEmittanceTextured(node))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetNodeEmittanceConstant(VLREnvironmentEmitterSurfaceMaterial material, VLRShaderNode node) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::EnvironmentEmitterSurfaceMaterial);
    if (!validateArgument<VLR::ShaderNode>(node))
        return VLRResult_InvalidArgument;
;
    if (!material->setNodeEmittanceConstant(node))
        return VLRResult_IncompatibleNodeType;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueEmittance(VLREnvironmentEmitterSurfaceMaterial material, VLRColorSpace colorSpace, float e0, float e1, float e2) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::EnvironmentEmitterSurfaceMaterial);
;
    material->setImmediateValueEmittance(colorSpace, e0, e1, e2);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEnvironmentEmitterSurfaceMaterialSetImmediateValueScale(VLREnvironmentEmitterSurfaceMaterial material, float value) {
    VLR_RETURN_INVALID_INSTANCE(material, VLR::EnvironmentEmitterSurfaceMaterial);
;
    material->setImmediateValueScale(value);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrStaticTransformCreate(VLRContext context, VLRStaticTransform* transform,
                                           const float mat[16]) {
    if (transform == nullptr || mat == nullptr)
        return VLRResult_InvalidArgument;

    VLR::Matrix4x4 mat4x4(mat);
    *transform = new VLR::StaticTransform(mat4x4);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrStaticTransformDestroy(VLRContext context, VLRStaticTransform transform) {
    VLR_RETURN_INVALID_INSTANCE(transform, VLR::StaticTransform);
;
    delete transform;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrNodeSetName(VLRNode node, const char* name) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Node);
    if (name == nullptr)
        return VLRResult_InvalidArgument;

    node->setName(name);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrNodeGetName(VLRNode node, const char** name) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Node);
    if (name == nullptr)
        return VLRResult_InvalidArgument;

    *name = node->getName().c_str();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrNodeGetType(VLRNode node, VLRNodeType* type) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::Node);
    if (type == nullptr)
        return VLRResult_InvalidArgument;

    *type = node->getType();

    return VLRResult_NoError;
}



VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode, 
                                                   const char* name) {
    if (surfaceNode == nullptr)
        return VLRResult_InvalidArgument;

    *surfaceNode = new VLR::TriangleMeshSurfaceNode(*context, name);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode) {
    VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);
;
    delete surfaceNode;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, VLRVertex* vertices, uint32_t numVertices) {
    VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);
    if (vertices == nullptr)
        return VLRResult_InvalidArgument;

    std::vector<VLR::Vertex> vecVertices;
    vecVertices.resize(numVertices);
    std::copy_n((VLR::Vertex*)vertices, numVertices, vecVertices.data());

    surfaceNode->setVertices(std::move(vecVertices));

    return VLRResult_NoError;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, uint32_t* indices, uint32_t numIndices, 
                                                             VLRSurfaceMaterial material, VLRShaderNodeSocket nodeNormal, VLRShaderNodeSocket nodeAlpha,
                                                             VLRTangentType tangentType) {
    VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);
    if (indices == nullptr || !validateArgument<VLR::SurfaceMaterial>(material))
        return VLRResult_InvalidArgument;

    std::vector<uint32_t> vecIndices;
    vecIndices.resize(numIndices);
    std::copy_n(indices, numIndices, vecIndices.data());

    surfaceNode->addMaterialGroup(std::move(vecIndices), material, 
                                  VLR::ShaderNodeSocket(nodeNormal),
                                  VLR::ShaderNodeSocket(nodeAlpha),
                                  tangentType);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                        const char* name, VLRTransform transform) {
    if (node == nullptr || !validateArgument<VLR::Transform>(transform))
        return VLRResult_InvalidArgument;

    *node = new VLR::InternalNode(*context, name, transform);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
;
    delete node;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, VLRTransform localToWorld) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
    if (!validateArgument<VLR::Transform>(localToWorld))
        return VLRResult_InvalidArgument;
;
    node->setTransform(localToWorld);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrInternalNodeGetTransform(VLRInternalNode node, VLRTransformConst* localToWorld) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
    if (localToWorld == nullptr)
        return VLRResult_InvalidArgument;
;
    *localToWorld = node->getTransform();

    return VLRResult_NoError;
}

VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRObject child) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
    if (child == nullptr)
        return VLRResult_InvalidArgument;
;
    if (child->isMemberOf<VLR::InternalNode>())
        node->addChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        node->addChild((VLR::SurfaceNode*)child);
    else
        return VLRResult_InvalidArgument;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRObject child) {
    VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
    if (child == nullptr)
        return VLRResult_InvalidArgument;
;
    if (child->isMemberOf<VLR::InternalNode>())
        node->removeChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        node->removeChild((VLR::SurfaceNode*)child);
    else
        return VLRResult_InvalidArgument;

    return VLRResult_NoError;
}



VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                 VLRTransform transform) {
    if (scene == nullptr || !validateArgument<VLR::Transform>(transform))
        return VLRResult_InvalidArgument;

    *scene = new VLR::Scene(*context, transform);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene) {
    VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
;
    delete scene;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, VLRTransform localToWorld) {
    VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
    if (!validateArgument<VLR::Transform>(localToWorld))
        return VLRResult_InvalidArgument;
;
    scene->setTransform(localToWorld);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRObject child) {
    VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
    if (child == nullptr)
        return VLRResult_InvalidArgument;

    if (child->isMemberOf<VLR::InternalNode>())
        scene->addChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        scene->addChild((VLR::SurfaceNode*)child);
    else
        return VLRResult_InvalidArgument;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRObject child) {
    VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
    if (child == nullptr)
        return VLRResult_InvalidArgument;
;
    if (child->isMemberOf<VLR::InternalNode>())
        scene->removeChild((VLR::InternalNode*)child);
    else if (child->isMemberOf<VLR::SurfaceNode>())
        scene->removeChild((VLR::SurfaceNode*)child);
    else
        return VLRResult_InvalidArgument;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLREnvironmentEmitterSurfaceMaterial material) {
    VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
    if (!validateArgument<VLR::EnvironmentEmitterSurfaceMaterial>(material))
        return VLRResult_InvalidArgument;
;
    scene->setEnvironment(material);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrSceneSetEnvironmentRotation(VLRScene scene, float rotationPhi) {
    VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
;
    scene->setEnvironmentRotation(rotationPhi);

    return VLRResult_NoError;
}




VLR_API VLRResult vlrCameraGetType(VLRCamera camera, VLRCameraType* type) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::Camera);
    if (type == nullptr)
        return VLRResult_InvalidArgument;
    
    *type = camera->getType();

    return VLRResult_NoError;
}




VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera) {
    if (camera == nullptr)
        return VLRResult_InvalidArgument;

    *camera = new VLR::PerspectiveCamera(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraDestroy(VLRContext context, VLRPerspectiveCamera camera) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
;
    delete camera;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetPosition(VLRPerspectiveCamera camera, const VLRPoint3D* position) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (position == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->setPosition(*(VLR::Point3D*)position);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetOrientation(VLRPerspectiveCamera camera, const VLRQuaternion* orientation) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (orientation == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->setOrientation(*(VLR::Quaternion*)orientation);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetAspectRatio(VLRPerspectiveCamera camera, float aspect) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
;
    camera->setAspectRatio(aspect);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetSensitivity(VLRPerspectiveCamera camera, float sensitivity) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
;
    camera->setSensitivity(sensitivity);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetFovY(VLRPerspectiveCamera camera, float fovY) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
;
    camera->setFovY(fovY);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetLensRadius(VLRPerspectiveCamera camera, float lensRadius) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
;
    camera->setLensRadius(lensRadius);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraSetObjectPlaneDistance(VLRPerspectiveCamera camera, float distance) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
;
    camera->setObjectPlaneDistance(distance);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetPosition(VLRPerspectiveCamera camera, VLRPoint3D* position) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (position == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getPosition((VLR::Point3D*)position);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetOrientation(VLRPerspectiveCamera camera, VLRQuaternion* orientation) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (orientation == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getOrientation((VLR::Quaternion*)orientation);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetAspectRatio(VLRPerspectiveCamera camera, float* aspect) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (aspect == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getAspectRatio(aspect);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetSensitivity(VLRPerspectiveCamera camera, float* sensitivity) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (sensitivity == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getSensitivity(sensitivity);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetFovY(VLRPerspectiveCamera camera, float* fovY) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (fovY == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getFovY(fovY);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetLensRadius(VLRPerspectiveCamera camera, float* lensRadius) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (lensRadius == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getLensRadius(lensRadius);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrPerspectiveCameraGetObjectPlaneDistance(VLRPerspectiveCamera camera, float* distance) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
    if (distance == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getObjectPlaneDistance(distance);

    return VLRResult_NoError;
}



VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera) {
    if (camera == nullptr)
        return VLRResult_InvalidArgument;

    *camera = new VLR::EquirectangularCamera(*context);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraDestroy(VLRContext context, VLREquirectangularCamera camera) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
;
    delete camera;

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraSetPosition(VLREquirectangularCamera camera, const VLRPoint3D* position) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
    if (position == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->setPosition(*(VLR::Point3D*)position);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraSetOrientation(VLREquirectangularCamera camera, const VLRQuaternion* orientation) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
    if (orientation == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->setOrientation(*(VLR::Quaternion*)orientation);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraSetSensitivity(VLREquirectangularCamera camera, float sensitivity) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
;
    camera->setSensitivity(sensitivity);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraSetAngles(VLREquirectangularCamera camera, float phiAngle, float thetaAngle) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
;
    camera->setAngles(phiAngle, thetaAngle);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraGetPosition(VLREquirectangularCamera camera, VLRPoint3D* position) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
    if (position == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getPosition((VLR::Point3D*)position);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraGetOrientation(VLREquirectangularCamera camera, VLRQuaternion* orientation) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
    if (orientation == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getOrientation((VLR::Quaternion*)orientation);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraGetSensitivity(VLREquirectangularCamera camera, float* sensitivity) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
    if (sensitivity == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getSensitivity(sensitivity);

    return VLRResult_NoError;
}

VLR_API VLRResult vlrEquirectangularCameraGetAngles(VLREquirectangularCamera camera, float* phiAngle, float* thetaAngle) {
    VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
    if (phiAngle == nullptr || thetaAngle == nullptr)
        return VLRResult_InvalidArgument;
;
    camera->getAngles(phiAngle, thetaAngle);

    return VLRResult_NoError;
}

