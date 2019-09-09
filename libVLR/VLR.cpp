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

VLR_API VLRResult vlrContextCastRays(VLRContext context, VLRScene scene,
                                     const VLRPoint3D* origins, const VLRVector3D* directions, uint32_t numRays,
                                     VLRPoint3D* hitPoints, VLRNormal3D* geometricNormals) {
    try {
        if (!scene->is<VLR::Scene>())
            return VLRResult_InvalidArgument;

        context->castRays(*scene, (const VLR::Point3D*)origins, (const VLR::Vector3D*)directions, numRays,
                          (VLR::Point3D*)hitPoints, (VLR::Normal3D*)geometricNormals);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrObjectGetType(VLRObjectConst object, const char** typeName) {
    try {
        VLR_RETURN_INVALID_INSTANCE(object, VLR::Object);

        *typeName = object->getType();

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



VLR_API VLRResult vlrQueryableGetNumParameters(VLRQueryableConst node, uint32_t* numParams) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        *numParams = node->getNumParameters();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetParameterInfo(VLRQueryableConst node, uint32_t index, VLRParameterInfoConst* paramInfo) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        const VLR::ParameterInfo* iParamInfo = node->getParameterInfo(index);
        if (iParamInfo == nullptr)
            return VLRResult_InvalidArgument;
        *paramInfo = iParamInfo;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrQueryableGetEnumValue(VLRQueryableConst node, const char* paramName,
                                           const char** value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->get(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetPoint3D(VLRQueryableConst node, const char* paramName,
                                         VLRPoint3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

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

VLR_API VLRResult vlrQueryableGetVector3D(VLRQueryableConst node, const char* paramName,
                                          VLRVector3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

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

VLR_API VLRResult vlrQueryableGetNormal3D(VLRQueryableConst node, const char* paramName,
                                          VLRNormal3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

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

VLR_API VLRResult vlrQueryableGetQuaternion(VLRQueryableConst node, const char* paramName,
                                            VLRQuaternion* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        VLR::Quaternion iValue;
        if (!node->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;
        value->w = iValue.w;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetFloat(VLRQueryableConst node, const char* paramName,
                                       float* value) {
    return vlrQueryableGetFloatTuple(node, paramName, value, 1);
}

VLR_API VLRResult vlrQueryableGetFloatTuple(VLRQueryableConst node, const char* paramName,
                                            float* values, uint32_t length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->get(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetFloatArray(VLRQueryableConst node, const char* paramName,
                                            const float** values, uint32_t* length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->get(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetImage2D(VLRQueryableConst node, const char* paramName,
                                         VLRImage2DConst* image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->get(paramName, image))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetImmediateSpectrum(VLRQueryableConst material, const char* paramName,
                                                   VLRImmediateSpectrum* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Queryable);

        VLR::ImmediateSpectrum iValue;
        if (!material->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        *value = iValue.getPublicType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetSurfaceMaterial(VLRQueryableConst material, const char* paramName,
                                                 VLRSurfaceMaterialConst* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Queryable);

        if (!material->get(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetShaderNodePlug(VLRQueryableConst node, const char* paramName,
                                                VLRShaderNodePlug* plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        VLR::ShaderNodePlug iValue;
        if (!node->get(paramName, &iValue))
            return VLRResult_InvalidArgument;

        *plug = iValue.getOpaqueType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrQueryableSetEnumValue(VLRQueryable node, const char* paramName,
                                           const char* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->set(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetPoint3D(VLRQueryable node, const char* paramName,
                                         const VLRPoint3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        VLR::Point3D iValue(value->x, value->y, value->z);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetVector3D(VLRQueryable node, const char* paramName,
                                          const VLRVector3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        VLR::Vector3D iValue(value->x, value->y, value->z);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetNormal3D(VLRQueryable node, const char* paramName,
                                          const VLRNormal3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        VLR::Normal3D iValue(value->x, value->y, value->z);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetQuaternion(VLRQueryable node, const char* paramName,
                                            const VLRQuaternion* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        VLR::Quaternion iValue(value->x, value->y, value->z, value->w);
        if (!node->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetFloat(VLRQueryable node, const char* paramName,
                                       float value) {
    return vlrQueryableSetFloatTuple(node, paramName, &value, 1);
}

VLR_API VLRResult vlrQueryableSetFloatTuple(VLRQueryable node, const char* paramName,
                                            const float* values, uint32_t length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->set(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetImage2D(VLRQueryable node, const char* paramName,
                                         VLRImage2DConst image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);
        if (image != nullptr)
            if (!image->isMemberOf<VLR::Image2D>())
                return VLRResult_InvalidArgument;

        if (!node->set(paramName, image))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetImmediateSpectrum(VLRQueryable material, const char* paramName,
                                                   const VLRImmediateSpectrum* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Queryable);

        VLR::ImmediateSpectrum iValue = *value;
        if (!material->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetSurfaceMaterial(VLRQueryable material, const char* paramName,
                                                 VLRSurfaceMaterialConst value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, VLR::Queryable);
        if (value != nullptr)
            if (!value->isMemberOf<VLR::Queryable>())
                return VLRResult_InvalidArgument;

        if (!material->set(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetShaderNodePlug(VLRQueryable node, const char* paramName,
                                                VLRShaderNodePlug plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, VLR::Queryable);

        if (!node->set(paramName, plug))
            return VLRResult_InvalidArgument;

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

VLR_API VLRResult vlrImage2DGetOriginalDataFormat(VLRImage2DConst image, const char** format) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, VLR::Image2D);

        *format = VLR::getEnumMemberFromValue(image->getOriginalDataFormat());

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
                                         const char* format, const char* spectrumType, const char* colorSpace) {
    try {
        if (image == nullptr || linearData == nullptr)
            return VLRResult_InvalidArgument;

        *image = new VLR::LinearImage2D(*context, linearData, width, height,
                                        VLR::getEnumValueFromMember<VLR::DataFormat>(format),
                                        VLR::getEnumValueFromMember<VLR::SpectrumType>(spectrumType),
                                        VLR::getEnumValueFromMember<VLR::ColorSpace>(colorSpace));

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
                                                  const char* dataFormat, const char* spectrumType, const char* colorSpace) {
    try {
        if (image == nullptr || data == nullptr || sizes == nullptr)
            return VLRResult_InvalidArgument;
        for (int m = 0; m < mipCount; ++m) {
            if (data[m] == nullptr)
                return VLRResult_InvalidArgument;
        }

        *image = new VLR::BlockCompressedImage2D(*context, data, sizes, mipCount, width, height,
                                                 VLR::getEnumValueFromMember<VLR::DataFormat>(dataFormat),
                                                 VLR::getEnumValueFromMember<VLR::SpectrumType>(spectrumType),
                                                 VLR::getEnumValueFromMember<VLR::ColorSpace>(colorSpace));

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



VLR_API VLRResult vlrShaderNodeCreate(VLRContext context, const char* typeName, VLRShaderNode* node) {
    try {
        std::string sTypeName = typeName;
        if (VLR::testParamName(sTypeName, "Geometry")) {
            *node = new VLR::GeometryShaderNode(*context);
        }
        if (VLR::testParamName(sTypeName, "Tangent")) {
            *node = new VLR::TangentShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "Float2")) {
            *node = new VLR::Float2ShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "Float3")) {
            *node = new VLR::Float3ShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "Float4")) {
            *node = new VLR::Float4ShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "ScaleAndOffsetFloat")) {
            *node = new VLR::ScaleAndOffsetFloatShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "TripletSpectrum")) {
            *node = new VLR::TripletSpectrumShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "RegularSampledSpectrum")) {
            *node = new VLR::RegularSampledSpectrumShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "IrregularSampledSpectrum")) {
            *node = new VLR::IrregularSampledSpectrumShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "Float3ToSpectrum")) {
            *node = new VLR::Float3ToSpectrumShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "ScaleAndOffsetUVTextureMap2D")) {
            *node = new VLR::ScaleAndOffsetUVTextureMap2DShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "Image2DTexture")) {
            *node = new VLR::Image2DTextureShaderNode(*context);
        }
        else if (VLR::testParamName(sTypeName, "EnvironmentTexture")) {
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
        if (VLR::testParamName(sTypeName, "Matte")) {
            *material = new VLR::MatteSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "SpecularReflection")) {
            *material = new VLR::SpecularReflectionSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "SpecularScattering")) {
            *material = new VLR::SpecularScatteringSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "MicrofacetReflection")) {
            *material = new VLR::MicrofacetReflectionSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "MicrofacetScattering")) {
            *material = new VLR::MicrofacetScatteringSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "LambertianScattering")) {
            *material = new VLR::LambertianScatteringSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "UE4")) {
            *material = new VLR::UE4SurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "OldStyle")) {
            *material = new VLR::OldStyleSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "DiffuseEmitter")) {
            *material = new VLR::DiffuseEmitterSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "Multi")) {
            *material = new VLR::MultiSurfaceMaterial(*context);
        }
        else if (VLR::testParamName(sTypeName, "EnvironmentEmitter")) {
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
                                                             VLRSurfaceMaterialConst material,
                                                             VLRShaderNodePlug nodeNormal, VLRShaderNodePlug nodeTangent, VLRShaderNodePlug nodeAlpha) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, VLR::TriangleMeshSurfaceNode);
        if (indices == nullptr || !nonNullAndCheckType<VLR::SurfaceMaterial>(material))
            return VLRResult_InvalidArgument;

        std::vector<uint32_t> vecIndices;
        vecIndices.resize(numIndices);
        std::copy_n(indices, numIndices, vecIndices.data());

        surfaceNode->addMaterialGroup(std::move(vecIndices), material,
            VLR::ShaderNodePlug(nodeNormal),
            VLR::ShaderNodePlug(nodeTangent),
            VLR::ShaderNodePlug(nodeAlpha));

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




VLR_API VLRResult vlrCameraCreate(VLRContext context, const char* typeName, VLRCamera* camera) {
    try {
        std::string sTypeName = typeName;
        if (VLR::testParamName(sTypeName, "Perspective")) {
            *camera = new VLR::PerspectiveCamera(*context);
        }
        else if (VLR::testParamName(sTypeName, "Equirectangular")) {
            *camera = new VLR::EquirectangularCamera(*context);
        }
        else {
            return VLRResult_InvalidArgument;
        }

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrCameraDestroy(VLRContext context, VLRCamera camera) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, VLR::Camera);

        delete camera;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}
