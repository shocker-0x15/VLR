#pragma once

#include "scene.h"

// e.g. Object
// typedef VLR::Object* VLRObject;
// typedef const VLR::Object* VLRObjectConst;
#define VLR_PROCESS_CLASS(name) \
    typedef VLR::name* VLR ## name; \
    typedef const VLR::name* VLR ## name ## Const

VLR_PROCESS_CLASS(Context);

VLR_PROCESS_CLASS_LIST();
#undef VLR_PROCESS_CLASS

#include <VLR.h>



#define VLR_RETURN_INVALID_INSTANCE(var, type) \
    if (var == nullptr) \
        return VLRResult_InvalidArgument; \
    if (!var->isMemberOf<type>()) \
        return VLRResult_InvalidInstance

#define VLR_RETURN_INTERNAL_ERROR() \
    catch (const std::exception &ex) { \
        VLRUnused(ex); \
        VLRAssert(false, "%s", ex.what()); \
        return VLRResult_InternalError; \
    }

template <typename T>
inline bool nonNullAndCheckType(const VLR::TypeAwareClass* obj) {
    if (obj == nullptr)
        return false;
    if (!obj->isMemberOf<T>())
        return false;
    return true;
}



VLR_API VLRResult vlrPrintDevices() {
    uint32_t numDevices;
    RTresult res;
    res = rtDeviceGetDeviceCount(&numDevices);
    if (res != RT_SUCCESS)
        return VLRResult_InternalError;

    for (int dev = 0; dev < numDevices; ++dev) {
        vlrprintf("----------------------------------------------------------------\n");

        char strBuffer[256];
        int32_t intBuffer[2];
        RTsize sizeValue;

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_NAME, sizeof(strBuffer), strBuffer);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("%d: %s\n", dev, strBuffer);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    CUDA Device Ordinal: %d\n", intBuffer[0]);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(strBuffer), strBuffer);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    PCI Bus ID: %s\n", strBuffer);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(intBuffer), intBuffer);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Compute Capability: %d, %d\n", intBuffer[0], intBuffer[1]);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    TCC (Tesla Compute Cluster) Driver: %s\n", intBuffer[0] ? "Yes" : "No");

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(sizeValue), &sizeValue);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Total Memory: %llu [Byte]\n", sizeValue);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Clock Rate: %d [kHz]\n", intBuffer[0]);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Max Threads per Block: %d\n", intBuffer[0]);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Multi Processor Count: %d\n", intBuffer[0]);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Max Hardware Texture Count: %d\n", intBuffer[0]);

        res = rtDeviceGetAttribute(dev, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(intBuffer[0]), &intBuffer[0]);
        if (res != RT_SUCCESS)
            return VLRResult_InternalError;
        vlrprintf("    Execution Timeout Enabled: %s\n", intBuffer[0] ? "Yes" : "No");
    }
    vlrprintf("----------------------------------------------------------------\n");

    return VLRResult_NoError;
}

VLR_API VLRResult vlrGetDeviceName(uint32_t index, char* name, uint32_t bufferLength) {
    RTresult res = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_NAME, bufferLength, name);
    if (res == RT_SUCCESS)
        return VLRResult_NoError;
    else
        return VLRResult_InternalError;
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
    case VLRResult_InternalError:
        return "Internal Error";
    default:
        VLRAssert_ShouldNotBeCalled();
        break;
    }

    return "";
}



VLR_API VLRResult vlrCreateContext(VLRContext* context, bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices) {
    try {
        *context = new VLR::Context(logging, enableRTX, maxCallableDepth, stackSize, devices, numDevices);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrDestroyContext(VLRContext context) {
    try {
        delete context;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrContextGetNumDevices(VLRContext context, uint32_t* numDevices) {
    try {
        if (numDevices == nullptr)
            return VLRResult_InvalidArgument;

        *numDevices = context->getNumDevices();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextGetDeviceIndexAt(VLRContext context, uint32_t index, int32_t* deviceIndex) {
    try {
        if (deviceIndex == nullptr)
            return VLRResult_InvalidArgument;

        *deviceIndex = context->getDeviceIndexAt(index);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrContextBindOutputBuffer(VLRContext context, uint32_t width, uint32_t height, uint32_t bufferID) {
    try {
        context->bindOutputBuffer(width, height, bufferID);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextMapOutputBuffer(VLRContext context, const void** ptr) {
    try {
        if (ptr == nullptr)
            return VLRResult_InvalidArgument;
        *ptr = context->mapOutputBuffer();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextUnmapOutputBuffer(VLRContext context) {
    try {
        context->unmapOutputBuffer();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextGetOutputBufferSize(VLRContext context, uint32_t* width, uint32_t* height) {
    try {
        if (width == nullptr || height == nullptr)
            return VLRResult_InvalidArgument;

        context->getOutputBufferSize(width, height);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextRender(VLRContext context, VLRScene scene, VLRCameraConst camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    try {
        if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>() || numAccumFrames == nullptr)
            return VLRResult_InvalidArgument;

        context->render(*scene, camera, shrinkCoeff, firstFrame, numAccumFrames);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextDebugRender(VLRContext context, VLRScene scene, VLRCameraConst camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    try {
        if (!scene->is<VLR::Scene>() || !camera->isMemberOf<VLR::Camera>() || numAccumFrames == nullptr)
            return VLRResult_InvalidArgument;

        context->debugRender(*scene, camera, renderMode, shrinkCoeff, firstFrame, numAccumFrames);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrImage2DGetWidth(VLRImage2DConst image, uint32_t* width) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);
        if (width == nullptr)
            return VLRResult_InvalidArgument;

        *width = image->getWidth();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DGetHeight(VLRImage2DConst image, uint32_t* height) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);
        if (height == nullptr)
            return VLRResult_InvalidArgument;

        *height = image->getHeight();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DGetStride(VLRImage2DConst image, uint32_t* stride) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

        *stride = image->getStride();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DGetOriginalDataFormat(VLRImage2DConst image, VLRDataFormat* format) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

        *format = (VLRDataFormat)image->getOriginalDataFormat();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DOriginalHasAlpha(VLRImage2DConst image, bool* hasAlpha) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

        *hasAlpha = image->originalHasAlpha();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                         uint8_t* linearData, uint32_t width, uint32_t height,
                                         VLRDataFormat format, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    try {
        if (image == nullptr || linearData == nullptr)
            return VLRResult_InvalidArgument;

        *image = new VLR::LinearImage2D(*context, linearData, width, height,
                                        (VLR::DataFormat)format, (VLR::SpectrumType)spectrumType, (VLR::ColorSpace)colorSpace);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::LinearImage2D);

        delete image;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrBlockCompressedImage2DCreate(VLRContext context, VLRBlockCompressedImage2D* image,
                                                  uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                                  VLRDataFormat dataFormat, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    try {
        if (image == nullptr || data == nullptr || sizes == nullptr)
            return VLRResult_InvalidArgument;
        for (int m = 0; m < mipCount; ++m) {
            if (data[m] == nullptr)
                return VLRResult_InvalidArgument;
        }

        *image = new VLR::BlockCompressedImage2D(*context, data, sizes, mipCount, width, height,
                                                 (VLR::DataFormat)dataFormat, (VLR::SpectrumType)spectrumType, (VLR::ColorSpace)colorSpace);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrBlockCompressedImage2DDestroy(VLRContext context, VLRBlockCompressedImage2D image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::BlockCompressedImage2D);

        delete image;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrParameterInfoGetName(VLRParameterInfoConst paramInfo, const char** name) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, VLR::ParameterInfo);

        *name = paramInfo->name;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrParameterInfoGetSocketForm(VLRParameterInfoConst paramInfo, VLRParameterFormFlag* form) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, VLR::ParameterInfo);

        *form = paramInfo->formFlags;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrParameterInfoGetType(VLRParameterInfoConst paramInfo, const char** type) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, VLR::ParameterInfo);

        *type = paramInfo->typeName;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrParameterInfoGetTupleSize(VLRParameterInfoConst paramInfo, uint32_t* size) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, VLR::ParameterInfo);

        *size = paramInfo->tupleSize;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrGetNumEnumMembers(const char* typeName, uint32_t* numMembers) {
    try {
        *numMembers = VLR::getNumEnumMembers(typeName);
        if (*numMembers == 0)
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrGetEnumMember(const char* typeName, uint32_t index, const char** value) {
    try {
        *value = VLR::getEnumMemberAt(typeName, index);
        if (*value == nullptr)
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrConnectableGetNumParameters(VLRConnectableConst node, uint32_t* numParams) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        *numParams = node->getNumParameters();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetParameterInfo(VLRConnectableConst node, uint32_t index, VLRParameterInfoConst* paramInfo) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        const VLR::ParameterInfo* iParamInfo = node->getParameterInfo(index);
        if (iParamInfo == nullptr)
            return VLRResult_InvalidArgument;
        *paramInfo = iParamInfo;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrConnectableGetEnumValue(VLRConnectableConst node, const char* paramName,
                                             const char** value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->get(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetPoint3D(VLRConnectableConst node, const char* paramName,
                                           VLRPoint3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::Point3D iValue;
        if (!node->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetVector3D(VLRConnectableConst node, const char* paramName,
                                            VLRVector3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::Vector3D iValue;
        if (!node->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetNormal3D(VLRConnectableConst node, const char* paramName,
                                            VLRNormal3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::Normal3D iValue;
        if (!node->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetFloat(VLRConnectableConst node, const char* paramName,
                                         float* value) {
    return vlrConnectableGetFloatTuple(node, paramName, value, 1);
}

VLR_API VLRResult vlrConnectableGetFloatTuple(VLRConnectableConst node, const char* paramName,
                                              float* values, uint32_t length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->get(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetFloatArray(VLRConnectableConst node, const char* paramName,
                                              const float** values, uint32_t* length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->get(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetImage2D(VLRConnectableConst node, const char* paramName,
                                           VLRImage2DConst* image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->get(paramName, image))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetImmediateSpectrum(VLRConnectableConst material, const char* paramName,
                                                     VLRImmediateSpectrum* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Connectable);

        VLR::ImmediateSpectrum iValue;
        if (!material->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        *value = iValue.getPublicType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetSurfaceMaterial(VLRConnectableConst material, const char* paramName,
                                                   VLRSurfaceMaterialConst* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Connectable);

        if (!material->get(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableGetShaderNodePlug(VLRConnectableConst node, const char* paramName,
                                                  VLRShaderNodePlug* plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::ShaderNodePlug iValue;
        if (!node->get(paramName, &iValue))
            return VLRResult_InvalidArgument;

        *plug = iValue.getOpaqueType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrConnectableSetEnumValue(VLRConnectable node, const char* paramName,
                                             const char* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->set(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetPoint3D(VLRConnectable node, const char* paramName,
                                           const VLRPoint3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::Point3D iValue(value->x, value->y, value->z);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetVector3D(VLRConnectable node, const char* paramName,
                                            const VLRVector3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::Vector3D iValue(value->x, value->y, value->z);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetNormal3D(VLRConnectable node, const char* paramName,
                                            const VLRNormal3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        VLR::Normal3D iValue(value->x, value->y, value->z);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetFloat(VLRConnectable node, const char* paramName,
                                         float value) {
    return vlrConnectableSetFloatTuple(node, paramName, &value, 1);
}

VLR_API VLRResult vlrConnectableSetFloatTuple(VLRConnectable node, const char* paramName,
                                              const float* values, uint32_t length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->set(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetImage2D(VLRConnectable node, const char* paramName,
                                           VLRImage2DConst image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);
        if (image != nullptr)
            if (!image->isMemberOf<VLR::Image2D>())
                return VLRResult_InvalidArgument;

        if (!node->set(paramName, image))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetImmediateSpectrum(VLRConnectable material, const char* paramName,
                                                     const VLRImmediateSpectrum* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Connectable);

        VLR::ImmediateSpectrum iValue = *value;
        if (!material->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetSurfaceMaterial(VLRConnectable material, const char* paramName,
                                                   VLRSurfaceMaterialConst value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Connectable);
        if (value != nullptr)
            if (!value->isMemberOf<VLR::Connectable>())
                return VLRResult_InvalidArgument;

        if (!material->set(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrConnectableSetShaderNodePlug(VLRConnectable node, const char* paramName,
                                                  VLRShaderNodePlug plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Connectable);

        if (!node->set(paramName, plug))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrShaderNodeCreate(VLRContext context, const char* typeName, VLRShaderNode* node) {
    try {
        std::string sTypeName = typeName;
        if (sTypeName == "Geometry") {
            *node = new VLR::GeometryShaderNode(*context);
        }
        else if (sTypeName == "Float2") {
            *node = new VLR::Float2ShaderNode(*context);
        }
        else if (sTypeName == "Float3") {
            *node = new VLR::Float3ShaderNode(*context);
        }
        else if (sTypeName == "Float4") {
            *node = new VLR::Float4ShaderNode(*context);
        }
        else if (sTypeName == "ScaleAndOffsetFloat") {
            *node = new VLR::ScaleAndOffsetFloatShaderNode(*context);
        }
        else if (sTypeName == "TripletSpectrum") {
            *node = new VLR::TripletSpectrumShaderNode(*context);
        }
        else if (sTypeName == "RegularSampledSpectrum") {
            *node = new VLR::RegularSampledSpectrumShaderNode(*context);
        }
        else if (sTypeName == "IrregularSampledSpectrum") {
            *node = new VLR::IrregularSampledSpectrumShaderNode(*context);
        }
        else if (sTypeName == "Float3ToSpectrum") {
            *node = new VLR::Float3ToSpectrumShaderNode(*context);
        }
        else if (sTypeName == "ScaleAndOffsetUVTextureMap2D") {
            *node = new VLR::ScaleAndOffsetUVTextureMap2DShaderNode(*context);
        }
        else if (sTypeName == "Image2DTexture") {
            *node = new VLR::Image2DTextureShaderNode(*context);
        }
        else if (sTypeName == "EnvironmentTexture") {
            *node = new VLR::EnvironmentTextureShaderNode(*context);
        }
        else {
            return VLRResult_InvalidArgument;
        }

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrShaderNodeDestroy(VLRContext context, VLRShaderNode node) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::ShaderNode);

        delete node;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrShaderNodeGetPlug(VLRShaderNodeConst node, VLRShaderNodePlugType plugType, uint32_t option,
                                       VLRShaderNodePlug* plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::ShaderNode);
        if (plug == nullptr)
            return VLRResult_InvalidArgument;

        *plug = node->getPlug((VLR::ShaderNodePlugType)plugType, option).getOpaqueType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrSurfaceMaterialCreate(VLRContext context, const char* typeName, VLRSurfaceMaterial* material) {
    try {
        std::string sTypeName = typeName;
        if (sTypeName == "Matte") {
            *material = new VLR::MatteSurfaceMaterial(*context);
        }
        else if (sTypeName == "SpecularReflection") {
            *material = new VLR::SpecularReflectionSurfaceMaterial(*context);
        }
        else if (sTypeName == "SpecularScattering") {
            *material = new VLR::SpecularScatteringSurfaceMaterial(*context);
        }
        else if (sTypeName == "MicrofacetReflection") {
            *material = new VLR::MicrofacetReflectionSurfaceMaterial(*context);
        }
        else if (sTypeName == "MicrofacetScattering") {
            *material = new VLR::MicrofacetScatteringSurfaceMaterial(*context);
        }
        else if (sTypeName == "LambertianScattering") {
            *material = new VLR::LambertianScatteringSurfaceMaterial(*context);
        }
        else if (sTypeName == "UE4") {
            *material = new VLR::UE4SurfaceMaterial(*context);
        }
        else if (sTypeName == "OldStyle") {
            *material = new VLR::OldStyleSurfaceMaterial(*context);
        }
        else if (sTypeName == "DiffuseEmitter") {
            *material = new VLR::DiffuseEmitterSurfaceMaterial(*context);
        }
        else if (sTypeName == "Multi") {
            *material = new VLR::MultiSurfaceMaterial(*context);
        }
        else if (sTypeName == "EnvironmentEmitter") {
            *material = new VLR::EnvironmentEmitterSurfaceMaterial(*context);
        }
        else {
            return VLRResult_InvalidArgument;
        }

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSurfaceMaterialDestroy(VLRContext context, VLRSurfaceMaterial material) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::SurfaceMaterial);

        delete material;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrTransformGetType(VLRTransformConst transform, VLRTransformType* type) {
    try {
        VLR_RETURN_INVALID_INSTANCE(transform, VLR::Transform);
        if (type == nullptr)
            return VLRResult_InvalidArgument;

        *type = transform->getType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrStaticTransformCreate(VLRContext context, VLRStaticTransform* transform,
                                           const float mat[16]) {
    try {
        if (transform == nullptr || mat == nullptr)
            return VLRResult_InvalidArgument;

        VLR::Matrix4x4 mat4x4(mat);
        *transform = new VLR::StaticTransform(mat4x4);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrStaticTransformDestroy(VLRContext context, VLRStaticTransform transform) {
    try {
        VLR_RETURN_INVALID_INSTANCE(transform, VLR::StaticTransform);

        delete transform;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrStaticTransformGetArrays(VLRStaticTransformConst transform, float mat[16], float invMat[16]) {
    try {
        VLR_RETURN_INVALID_INSTANCE(transform, VLR::StaticTransform);
        if (mat == nullptr || invMat == nullptr)
            return VLRResult_InvalidArgument;

        transform->getArrays(mat, invMat);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrNodeSetName(VLRNode node, const char* name) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Node);
        if (name == nullptr)
            return VLRResult_InvalidArgument;

        node->setName(name);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrNodeGetName(VLRNodeConst node, const char** name) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Node);
        if (name == nullptr)
            return VLRResult_InvalidArgument;

        *name = node->getName().c_str();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrNodeGetType(VLRNodeConst node, VLRNodeType* type) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Node);
        if (type == nullptr)
            return VLRResult_InvalidArgument;

        *type = node->getType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode, 
                                                   const char* name) {
    try {
        if (surfaceNode == nullptr)
            return VLRResult_InvalidArgument;

        *surfaceNode = new VLR::TriangleMeshSurfaceNode(*context, name);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);

        delete surfaceNode;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, const VLRVertex* vertices, uint32_t numVertices) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);
        if (vertices == nullptr)
            return VLRResult_InvalidArgument;

        std::vector<VLR::Vertex> vecVertices;
        vecVertices.resize(numVertices);
        std::copy_n((VLR::Vertex*)vertices, numVertices, vecVertices.data());

        surfaceNode->setVertices(std::move(vecVertices));

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, const uint32_t* indices, uint32_t numIndices, 
                                                             VLRSurfaceMaterialConst material, VLRShaderNodePlug nodeNormal, VLRShaderNodePlug nodeAlpha,
                                                             VLRTangentType tangentType) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);
        if (indices == nullptr || !nonNullAndCheckType<VLR::SurfaceMaterial>(material))
            return VLRResult_InvalidArgument;

        std::vector<uint32_t> vecIndices;
        vecIndices.resize(numIndices);
        std::copy_n(indices, numIndices, vecIndices.data());

        surfaceNode->addMaterialGroup(std::move(vecIndices), material,
            VLR::ShaderNodePlug(nodeNormal),
            VLR::ShaderNodePlug(nodeAlpha),
            tangentType);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                        const char* name, VLRTransformConst transform) {
    try {
        if (node == nullptr || !nonNullAndCheckType<VLR::Transform>(transform))
            return VLRResult_InvalidArgument;

        *node = new VLR::InternalNode(*context, name, transform);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);

        delete node;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, VLRTransformConst localToWorld) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (!nonNullAndCheckType<VLR::Transform>(localToWorld))
            return VLRResult_InvalidArgument;

        node->setTransform(localToWorld);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetTransform(VLRInternalNodeConst node, VLRTransformConst* localToWorld) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (localToWorld == nullptr)
            return VLRResult_InvalidArgument;

        *localToWorld = node->getTransform();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->isMemberOf<VLR::InternalNode>())
            node->addChild((VLR::InternalNode*)child);
        else if (child->isMemberOf<VLR::SurfaceNode>())
            node->addChild((VLR::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->isMemberOf<VLR::InternalNode>())
            node->removeChild((VLR::InternalNode*)child);
        else if (child->isMemberOf<VLR::SurfaceNode>())
            node->removeChild((VLR::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetNumChildren(VLRInternalNodeConst node, uint32_t* numChildren) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (numChildren == nullptr)
            return VLRResult_InvalidArgument;

        *numChildren = node->getNumChildren();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetChildren(VLRInternalNodeConst node, VLRNode* children) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (children == nullptr)
            return VLRResult_InvalidArgument;

        node->getChildren(children);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetChildAt(VLRInternalNodeConst node, uint32_t index, VLRNode* child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::InternalNode);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        *child = node->getChildAt(index);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                 VLRTransformConst transform) {
    try {
        if (scene == nullptr || !nonNullAndCheckType<VLR::Transform>(transform))
            return VLRResult_InvalidArgument;

        *scene = new VLR::Scene(*context, transform);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);

        delete scene;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, VLRTransformConst localToWorld) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
        if (!nonNullAndCheckType<VLR::Transform>(localToWorld))
            return VLRResult_InvalidArgument;

        scene->setTransform(localToWorld);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRNode child) {
    try {
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
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->isMemberOf<VLR::InternalNode>())
            scene->removeChild((VLR::InternalNode*)child);
        else if (child->isMemberOf<VLR::SurfaceNode>())
            scene->removeChild((VLR::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneGetNumChildren(VLRSceneConst scene, uint32_t* numChildren) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
        if (numChildren == nullptr)
            return VLRResult_InvalidArgument;

        *numChildren = scene->getNumChildren();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneGetChildren(VLRSceneConst scene, VLRNode* children) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
        if (children == nullptr)
            return VLRResult_InvalidArgument;

        scene->getChildren(children);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneGetChildAt(VLRSceneConst scene, uint32_t index, VLRNode* child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        *child = scene->getChildAt(index);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneSetEnvironment(VLRScene scene, VLRSurfaceMaterial material) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);
        if (!nonNullAndCheckType<VLR::EnvironmentEmitterSurfaceMaterial>(material))
            return VLRResult_InvalidArgument;

        scene->setEnvironment((VLR::EnvironmentEmitterSurfaceMaterial*)material);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneSetEnvironmentRotation(VLRScene scene, float rotationPhi) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, VLR::Scene);

        scene->setEnvironmentRotation(rotationPhi);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}




VLR_API VLRResult vlrCameraGetType(VLRCameraConst camera, VLRCameraType* type) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::Camera);
        if (type == nullptr)
            return VLRResult_InvalidArgument;

        *type = camera->getType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}




VLR_API VLRResult vlrPerspectiveCameraCreate(VLRContext context, VLRPerspectiveCamera* camera) {
    try {
        if (camera == nullptr)
            return VLRResult_InvalidArgument;

        *camera = new VLR::PerspectiveCamera(*context);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraDestroy(VLRContext context, VLRPerspectiveCamera camera) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);

        delete camera;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetPosition(VLRPerspectiveCamera camera, const VLRPoint3D* position) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (position == nullptr)
            return VLRResult_InvalidArgument;

        camera->setPosition(*(VLR::Point3D*)position);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetOrientation(VLRPerspectiveCamera camera, const VLRQuaternion* orientation) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (orientation == nullptr)
            return VLRResult_InvalidArgument;

        camera->setOrientation(*(VLR::Quaternion*)orientation);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetAspectRatio(VLRPerspectiveCamera camera, float aspect) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);

        camera->setAspectRatio(aspect);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetSensitivity(VLRPerspectiveCamera camera, float sensitivity) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);

        camera->setSensitivity(sensitivity);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetFovY(VLRPerspectiveCamera camera, float fovY) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);

        camera->setFovY(fovY);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetLensRadius(VLRPerspectiveCamera camera, float lensRadius) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);

        camera->setLensRadius(lensRadius);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraSetObjectPlaneDistance(VLRPerspectiveCamera camera, float distance) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);

        camera->setObjectPlaneDistance(distance);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetPosition(VLRPerspectiveCameraConst camera, VLRPoint3D* position) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (position == nullptr)
            return VLRResult_InvalidArgument;

        camera->getPosition((VLR::Point3D*)position);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetOrientation(VLRPerspectiveCameraConst camera, VLRQuaternion* orientation) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (orientation == nullptr)
            return VLRResult_InvalidArgument;

        camera->getOrientation((VLR::Quaternion*)orientation);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetAspectRatio(VLRPerspectiveCameraConst camera, float* aspect) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (aspect == nullptr)
            return VLRResult_InvalidArgument;

        camera->getAspectRatio(aspect);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetSensitivity(VLRPerspectiveCameraConst camera, float* sensitivity) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (sensitivity == nullptr)
            return VLRResult_InvalidArgument;

        camera->getSensitivity(sensitivity);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetFovY(VLRPerspectiveCameraConst camera, float* fovY) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (fovY == nullptr)
            return VLRResult_InvalidArgument;

        camera->getFovY(fovY);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetLensRadius(VLRPerspectiveCameraConst camera, float* lensRadius) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (lensRadius == nullptr)
            return VLRResult_InvalidArgument;

        camera->getLensRadius(lensRadius);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrPerspectiveCameraGetObjectPlaneDistance(VLRPerspectiveCameraConst camera, float* distance) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::PerspectiveCamera);
        if (distance == nullptr)
            return VLRResult_InvalidArgument;

        camera->getObjectPlaneDistance(distance);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrEquirectangularCameraCreate(VLRContext context, VLREquirectangularCamera* camera) {
    try {
        if (camera == nullptr)
            return VLRResult_InvalidArgument;

        *camera = new VLR::EquirectangularCamera(*context);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraDestroy(VLRContext context, VLREquirectangularCamera camera) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);

        delete camera;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraSetPosition(VLREquirectangularCamera camera, const VLRPoint3D* position) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
        if (position == nullptr)
            return VLRResult_InvalidArgument;

        camera->setPosition(*(VLR::Point3D*)position);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraSetOrientation(VLREquirectangularCamera camera, const VLRQuaternion* orientation) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
        if (orientation == nullptr)
            return VLRResult_InvalidArgument;

        camera->setOrientation(*(VLR::Quaternion*)orientation);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraSetSensitivity(VLREquirectangularCamera camera, float sensitivity) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);

        camera->setSensitivity(sensitivity);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraSetAngles(VLREquirectangularCamera camera, float phiAngle, float thetaAngle) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);

        camera->setAngles(phiAngle, thetaAngle);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraGetPosition(VLREquirectangularCameraConst camera, VLRPoint3D* position) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
        if (position == nullptr)
            return VLRResult_InvalidArgument;

        camera->getPosition((VLR::Point3D*)position);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraGetOrientation(VLREquirectangularCameraConst camera, VLRQuaternion* orientation) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
        if (orientation == nullptr)
            return VLRResult_InvalidArgument;

        camera->getOrientation((VLR::Quaternion*)orientation);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraGetSensitivity(VLREquirectangularCameraConst camera, float* sensitivity) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
        if (sensitivity == nullptr)
            return VLRResult_InvalidArgument;

        camera->getSensitivity(sensitivity);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrEquirectangularCameraGetAngles(VLREquirectangularCameraConst camera, float* phiAngle, float* thetaAngle) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::EquirectangularCamera);
        if (phiAngle == nullptr || thetaAngle == nullptr)
            return VLRResult_InvalidArgument;

        camera->getAngles(phiAngle, thetaAngle);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

