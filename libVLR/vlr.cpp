#pragma once

#include "scene.h"

// e.g. Object
// typedef vlr::Object* VLRObject;
// typedef const vlr::Object* VLRObjectConst;
#define VLR_PROCESS_CLASS(name) \
    typedef vlr::name* VLR ## name; \
    typedef const vlr::name* VLR ## name ## Const

VLR_PROCESS_CLASS(Context);

VLR_PROCESS_CLASS_LIST();
#undef VLR_PROCESS_CLASS

#include <vlr.h>



#define VLR_RETURN_INVALID_INSTANCE(var, type) \
    if (var == nullptr) \
        return VLRResult_InvalidArgument; \
    if (!var->belongsTo<type>()) \
        return VLRResult_InvalidInstance

#define VLR_RETURN_INTERNAL_ERROR() \
    catch (const std::exception &ex) { \
        VLRUnused(ex); \
        VLRAssert(false, "%s", ex.what()); \
        return VLRResult_InternalError; \
    }

template <typename T>
inline bool nonNullAndCheckType(const vlr::TypeAwareClass* obj) {
    if (obj == nullptr)
        return false;
    if (!obj->belongsTo<T>())
        return false;
    return true;
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



VLR_API VLRResult vlrCreateContext(
    CUcontext cuContext, bool logging, uint32_t maxCallableDepth,
    VLRContext* context) {
    try {
        *context = new vlr::Context(cuContext, logging, maxCallableDepth);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrDestroyContext(
    VLRContext context) {
    try {
        delete context;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextGetCUcontext(
    VLRContext context,
    CUcontext* cuContext) {
    try {
        *cuContext = context->getCUcontext();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrContextBindOutputBuffer(
    VLRContext context,
    uint32_t width, uint32_t height, uint32_t bufferID) {
    try {
        context->bindOutputBuffer(width, height, bufferID);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextGetOutputBufferSize(
    VLRContext context,
    uint32_t* width, uint32_t* height) {
    try {
        if (width == nullptr || height == nullptr)
            return VLRResult_InvalidArgument;

        context->getOutputBufferSize(width, height);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextGetOutputBuffer(
    VLRContext context,
    CUarray* array) {
    try {
        if (array == nullptr)
            return VLRResult_InvalidArgument;

        const cudau::Array &cudauArray = context->getOutputBuffer();
        *array = cudauArray.getCUarray(0);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextReadOutputBuffer(
    VLRContext context,
    float* data) {
    try {
        if (data == nullptr)
            return VLRResult_InvalidArgument;

        context->readOutputBuffer(data);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextSetScene(
    VLRContext context,
    VLRScene scene) {
    try {
        if (!scene->is<vlr::Scene>())
            return VLRResult_InvalidArgument;

        context->setScene(scene);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextRender(
    VLRContext context,
    CUstream stream, VLRCameraConst camera, bool denoise,
    uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    try {
        if (!camera->belongsTo<vlr::Camera>() || numAccumFrames == nullptr)
            return VLRResult_InvalidArgument;

        context->render(stream, camera, denoise, shrinkCoeff, firstFrame, numAccumFrames);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrContextDebugRender(
    VLRContext context,
    CUstream stream, VLRCameraConst camera, VLRDebugRenderingMode renderMode,
    uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) {
    try {
        if (!camera->belongsTo<vlr::Camera>() || numAccumFrames == nullptr)
            return VLRResult_InvalidArgument;

        context->debugRender(stream, camera, renderMode, shrinkCoeff, firstFrame, numAccumFrames);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrObjectGetType(
    VLRObjectConst object,
    const char** typeName) {
    try {
        VLR_RETURN_INVALID_INSTANCE(object, vlr::Object);

        *typeName = object->getType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrParameterInfoGetName(
    VLRParameterInfoConst paramInfo,
    const char** name) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, vlr::ParameterInfo);

        *name = paramInfo->name;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrParameterInfoGetSocketForm(
    VLRParameterInfoConst paramInfo,
    VLRParameterFormFlag* form) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, vlr::ParameterInfo);

        *form = paramInfo->formFlags;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrParameterInfoGetType(
    VLRParameterInfoConst paramInfo,
    const char** type) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, vlr::ParameterInfo);

        *type = paramInfo->typeName;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrParameterInfoGetTupleSize(
    VLRParameterInfoConst paramInfo,
    uint32_t* size) {
    try {
        //VLR_RETURN_INVALID_INSTANCE(paramInfo, vlr::ParameterInfo);

        *size = paramInfo->tupleSize;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrGetNumEnumMembers(
    const char* typeName,
    uint32_t* numMembers) {
    try {
        *numMembers = vlr::getNumEnumMembers(typeName);
        if (*numMembers == 0)
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrGetEnumMember(
    const char* typeName, uint32_t index,
    const char** value) {
    try {
        *value = vlr::getEnumMemberAt(typeName, index);
        if (*value == nullptr)
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrQueryableGetNumParameters(
    VLRQueryableConst queryable,
    uint32_t* numParams) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        *numParams = queryable->getNumParameters();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetParameterInfo(
    VLRQueryableConst queryable,
    uint32_t index, VLRParameterInfoConst* paramInfo) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        const vlr::ParameterInfo* iParamInfo = queryable->getParameterInfo(index);
        if (iParamInfo == nullptr)
            return VLRResult_InvalidArgument;
        *paramInfo = iParamInfo;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrQueryableGetEnumValue(
    VLRQueryableConst queryable,
    const char* paramName,
    const char** value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->get(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetPoint3D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRPoint3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Point3D iValue;
        if (!queryable->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetVector3D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRVector3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Vector3D iValue;
        if (!queryable->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetNormal3D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRNormal3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Normal3D iValue;
        if (!queryable->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetQuaternion(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRQuaternion* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Quaternion iValue;
        if (!queryable->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        value->x = iValue.x;
        value->y = iValue.y;
        value->z = iValue.z;
        value->w = iValue.w;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetFloat(
    VLRQueryableConst queryable,
    const char* paramName,
    float* value) {
    return vlrQueryableGetFloatTuple(queryable, paramName, value, 1);
}

VLR_API VLRResult vlrQueryableGetFloatTuple(
    VLRQueryableConst queryable,
    const char* paramName,
    float* values, uint32_t length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->get(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetFloatArray(
    VLRQueryableConst queryable,
    const char* paramName,
    const float** values, uint32_t* length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->get(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetImage2D(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRImage2DConst* image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->get(paramName, image))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetImmediateSpectrum(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRImmediateSpectrum* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::ImmediateSpectrum iValue;
        if (!queryable->get(paramName, &iValue))
            return VLRResult_InvalidArgument;
        *value = iValue.getPublicType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetSurfaceMaterial(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRSurfaceMaterialConst* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->get(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableGetShaderNodePlug(
    VLRQueryableConst queryable,
    const char* paramName,
    VLRShaderNodePlug* plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::ShaderNodePlug iValue;
        if (!queryable->get(paramName, &iValue))
            return VLRResult_InvalidArgument;

        *plug = iValue.getOpaqueType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrQueryableSetEnumValue(
    VLRQueryable queryable,
    const char* paramName, const char* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->set(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetPoint3D(
    VLRQueryable queryable,
    const char* paramName, const VLRPoint3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Point3D iValue(value->x, value->y, value->z);
        if (!queryable->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetVector3D(
    VLRQueryable queryable,
    const char* paramName, const VLRVector3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Vector3D iValue(value->x, value->y, value->z);
        if (!queryable->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetNormal3D(
    VLRQueryable queryable,
    const char* paramName, const VLRNormal3D* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Normal3D iValue(value->x, value->y, value->z);
        if (!queryable->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetQuaternion(
    VLRQueryable queryable,
    const char* paramName, const VLRQuaternion* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::Quaternion iValue(value->x, value->y, value->z, value->w);
        if (!queryable->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetFloat(
    VLRQueryable queryable,
    const char* paramName, float value) {
    return vlrQueryableSetFloatTuple(queryable, paramName, &value, 1);
}

VLR_API VLRResult vlrQueryableSetFloatTuple(
    VLRQueryable queryable,
    const char* paramName, const float* values, uint32_t length) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->set(paramName, values, length))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetImage2D(
    VLRQueryable queryable,
    const char* paramName,
    VLRImage2DConst image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);
        if (image != nullptr)
            if (!image->belongsTo<vlr::Image2D>())
                return VLRResult_InvalidArgument;

        if (!queryable->set(paramName, image))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetImmediateSpectrum(
    VLRQueryable queryable,
    const char* paramName, const VLRImmediateSpectrum* value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        vlr::ImmediateSpectrum iValue = *value;
        if (!queryable->set(paramName, iValue))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetSurfaceMaterial(
    VLRQueryable queryable,
    const char* paramName, VLRSurfaceMaterialConst value) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);
        if (value != nullptr)
            if (!value->belongsTo<vlr::Queryable>())
                return VLRResult_InvalidArgument;

        if (!queryable->set(paramName, value))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrQueryableSetShaderNodePlug(
    VLRQueryable queryable,
    const char* paramName, VLRShaderNodePlug plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(queryable, vlr::Queryable);

        if (!queryable->set(paramName, plug))
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrImage2DGetWidth(
    VLRImage2DConst image,
    uint32_t* width) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::Image2D);
        if (width == nullptr)
            return VLRResult_InvalidArgument;

        *width = image->getWidth();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DGetHeight(
    VLRImage2DConst image,
    uint32_t* height) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::Image2D);
        if (height == nullptr)
            return VLRResult_InvalidArgument;

        *height = image->getHeight();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DGetStride(
    VLRImage2DConst image,
    uint32_t* stride) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::Image2D);

        *stride = image->getStride();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DGetOriginalDataFormat(
    VLRImage2DConst image,
    const char** format) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::Image2D);

        *format = vlr::getEnumMemberFromValue(image->getOriginalDataFormat());

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrImage2DOriginalHasAlpha(
    VLRImage2DConst image,
    bool* hasAlpha) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::Image2D);

        *hasAlpha = image->originalHasAlpha();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrLinearImage2DCreate(
    VLRContext context,
    uint8_t* linearData, uint32_t width, uint32_t height,
    const char* format, const char* spectrumType, const char* colorSpace,
    VLRLinearImage2D* image) {
    try {
        if (image == nullptr || linearData == nullptr)
            return VLRResult_InvalidArgument;

        *image = new vlr::LinearImage2D(*context, linearData, width, height,
                                        vlr::getEnumValueFromMember<vlr::DataFormat>(format),
                                        vlr::getEnumValueFromMember<vlr::SpectrumType>(spectrumType),
                                        vlr::getEnumValueFromMember<vlr::ColorSpace>(colorSpace));

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrLinearImage2DDestroy(
    VLRContext context,
    VLRLinearImage2D image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::LinearImage2D);

        delete image;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrBlockCompressedImage2DCreate(
    VLRContext context,
    uint8_t** data, size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
    const char* dataFormat, const char* spectrumType, const char* colorSpace,
    VLRBlockCompressedImage2D* image) {
    try {
        if (image == nullptr || data == nullptr || sizes == nullptr)
            return VLRResult_InvalidArgument;
        for (int m = 0; m < mipCount; ++m) {
            if (data[m] == nullptr)
                return VLRResult_InvalidArgument;
        }

        *image = new vlr::BlockCompressedImage2D(*context, data, sizes, mipCount, width, height,
                                                 vlr::getEnumValueFromMember<vlr::DataFormat>(dataFormat),
                                                 vlr::getEnumValueFromMember<vlr::SpectrumType>(spectrumType),
                                                 vlr::getEnumValueFromMember<vlr::ColorSpace>(colorSpace));

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrBlockCompressedImage2DDestroy(
    VLRContext context,
    VLRBlockCompressedImage2D image) {
    try {
        VLR_RETURN_INVALID_INSTANCE(image, vlr::BlockCompressedImage2D);

        delete image;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrShaderNodeCreate(
    VLRContext context,
    const char* typeName,
    VLRShaderNode* node) {
    try {
        std::string sTypeName = typeName;
        if (vlr::testParamName(sTypeName, "Geometry")) {
            *node = new vlr::GeometryShaderNode(*context);
        }
        if (vlr::testParamName(sTypeName, "Tangent")) {
            *node = new vlr::TangentShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "Float2")) {
            *node = new vlr::Float2ShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "Float3")) {
            *node = new vlr::Float3ShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "Float4")) {
            *node = new vlr::Float4ShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "ScaleAndOffsetFloat")) {
            *node = new vlr::ScaleAndOffsetFloatShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "TripletSpectrum")) {
            *node = new vlr::TripletSpectrumShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "RegularSampledSpectrum")) {
            *node = new vlr::RegularSampledSpectrumShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "IrregularSampledSpectrum")) {
            *node = new vlr::IrregularSampledSpectrumShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "Float3ToSpectrum")) {
            *node = new vlr::Float3ToSpectrumShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "ScaleAndOffsetUVTextureMap2D")) {
            *node = new vlr::ScaleAndOffsetUVTextureMap2DShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "Image2DTexture")) {
            *node = new vlr::Image2DTextureShaderNode(*context);
        }
        else if (vlr::testParamName(sTypeName, "EnvironmentTexture")) {
            *node = new vlr::EnvironmentTextureShaderNode(*context);
        }
        else {
            return VLRResult_InvalidArgument;
        }

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrShaderNodeDestroy(
    VLRContext context,
    VLRShaderNode node) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::ShaderNode);

        delete node;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrShaderNodeGetPlug(
    VLRShaderNodeConst node,
    VLRShaderNodePlugType plugType, uint32_t option,
    VLRShaderNodePlug* plug) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::ShaderNode);
        if (plug == nullptr)
            return VLRResult_InvalidArgument;

        *plug = node->getPlug((vlr::ShaderNodePlugType)plugType, option).getOpaqueType();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrSurfaceMaterialCreate(
    VLRContext context,
    const char* typeName,
    VLRSurfaceMaterial* material) {
    try {
        std::string sTypeName = typeName;
        if (vlr::testParamName(sTypeName, "Matte")) {
            *material = new vlr::MatteSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "SpecularReflection")) {
            *material = new vlr::SpecularReflectionSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "SpecularScattering")) {
            *material = new vlr::SpecularScatteringSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "MicrofacetReflection")) {
            *material = new vlr::MicrofacetReflectionSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "MicrofacetScattering")) {
            *material = new vlr::MicrofacetScatteringSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "LambertianScattering")) {
            *material = new vlr::LambertianScatteringSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "UE4")) {
            *material = new vlr::UE4SurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "OldStyle")) {
            *material = new vlr::OldStyleSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "DiffuseEmitter")) {
            *material = new vlr::DiffuseEmitterSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "Multi")) {
            *material = new vlr::MultiSurfaceMaterial(*context);
        }
        else if (vlr::testParamName(sTypeName, "EnvironmentEmitter")) {
            *material = new vlr::EnvironmentEmitterSurfaceMaterial(*context);
        }
        else {
            return VLRResult_InvalidArgument;
        }

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSurfaceMaterialDestroy(
    VLRContext context,
    VLRSurfaceMaterial material) {
    try {
        VLR_RETURN_INVALID_INSTANCE(material, vlr::SurfaceMaterial);

        delete material;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrStaticTransformCreate(
    VLRContext context,
    const float mat[16],
    VLRStaticTransform* transform) {
    try {
        if (transform == nullptr || mat == nullptr)
            return VLRResult_InvalidArgument;

        vlr::Matrix4x4 mat4x4(mat);
        *transform = new vlr::StaticTransform(mat4x4);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrStaticTransformDestroy(
    VLRContext context,
    VLRStaticTransform transform) {
    try {
        VLR_RETURN_INVALID_INSTANCE(transform, vlr::StaticTransform);

        delete transform;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrStaticTransformGetArrays(
    VLRStaticTransformConst transform,
    float mat[16], float invMat[16]) {
    try {
        VLR_RETURN_INVALID_INSTANCE(transform, vlr::StaticTransform);
        if (mat == nullptr || invMat == nullptr)
            return VLRResult_InvalidArgument;

        transform->getArrays(mat, invMat);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrNodeSetName(
    VLRNode node,
    const char* name) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::Node);
        if (name == nullptr)
            return VLRResult_InvalidArgument;

        node->setName(name);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrNodeGetName(
    VLRNodeConst node,
    const char** name) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::Node);
        if (name == nullptr)
            return VLRResult_InvalidArgument;

        *name = node->getName().c_str();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(
    VLRContext context,
    const char* name,
    VLRTriangleMeshSurfaceNode* surfaceNode) {
    try {
        if (surfaceNode == nullptr)
            return VLRResult_InvalidArgument;

        *surfaceNode = new vlr::TriangleMeshSurfaceNode(*context, name);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(
    VLRContext context,
    VLRTriangleMeshSurfaceNode surfaceNode) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, vlr::TriangleMeshSurfaceNode);

        delete surfaceNode;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(
    VLRTriangleMeshSurfaceNode surfaceNode,
    const VLRVertex* vertices, uint32_t numVertices) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, vlr::TriangleMeshSurfaceNode);
        if (vertices == nullptr)
            return VLRResult_InvalidArgument;

        std::vector<vlr::Vertex> vecVertices(numVertices);
        std::copy_n((vlr::Vertex*)vertices, numVertices, vecVertices.data());

        surfaceNode->setVertices(std::move(vecVertices));

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(
    VLRTriangleMeshSurfaceNode surfaceNode,
    const uint32_t* indices, uint32_t numIndices, 
    VLRSurfaceMaterialConst material,
    VLRShaderNodePlug nodeNormal, VLRShaderNodePlug nodeTangent, VLRShaderNodePlug nodeAlpha) {
    try {
        VLR_RETURN_INVALID_INSTANCE(surfaceNode, vlr::TriangleMeshSurfaceNode);
        if (indices == nullptr || !nonNullAndCheckType<vlr::SurfaceMaterial>(material))
            return VLRResult_InvalidArgument;

        std::vector<uint32_t> vecIndices(numIndices);
        std::copy_n(indices, numIndices, vecIndices.data());

        surfaceNode->addMaterialGroup(std::move(vecIndices), material,
            vlr::ShaderNodePlug(nodeNormal),
            vlr::ShaderNodePlug(nodeTangent),
            vlr::ShaderNodePlug(nodeAlpha));

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrInternalNodeCreate(
    VLRContext context,
    const char* name, VLRTransformConst transform,
    VLRInternalNode* node) {
    try {
        if (node == nullptr || !nonNullAndCheckType<vlr::Transform>(transform))
            return VLRResult_InvalidArgument;

        *node = new vlr::InternalNode(*context, name, transform);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeDestroy(
    VLRContext context,
    VLRInternalNode node) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);

        delete node;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeSetTransform(
    VLRInternalNode node,
    VLRTransformConst localToWorld) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (!nonNullAndCheckType<vlr::Transform>(localToWorld))
            return VLRResult_InvalidArgument;

        node->setTransform(localToWorld);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetTransform(
    VLRInternalNodeConst node,
    VLRTransformConst* localToWorld) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (localToWorld == nullptr)
            return VLRResult_InvalidArgument;

        *localToWorld = node->getTransform();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeAddChild(
    VLRInternalNode node,
    VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->belongsTo<vlr::InternalNode>())
            node->addChild((vlr::InternalNode*)child);
        else if (child->belongsTo<vlr::SurfaceNode>())
            node->addChild((vlr::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeRemoveChild(
    VLRInternalNode node,
    VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->belongsTo<vlr::InternalNode>())
            node->removeChild((vlr::InternalNode*)child);
        else if (child->belongsTo<vlr::SurfaceNode>())
            node->removeChild((vlr::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetNumChildren(
    VLRInternalNodeConst node,
    uint32_t* numChildren) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (numChildren == nullptr)
            return VLRResult_InvalidArgument;

        *numChildren = node->getNumChildren();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetChildren(
    VLRInternalNodeConst node,
    VLRNode* children) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (children == nullptr)
            return VLRResult_InvalidArgument;

        node->getChildren(children);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrInternalNodeGetChildAt(
    VLRInternalNodeConst node,
    uint32_t index,
    VLRNode* child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(node, vlr::InternalNode);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        *child = node->getChildAt(index);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}



VLR_API VLRResult vlrSceneCreate(
    VLRContext context,
    VLRTransformConst transform,
    VLRScene* scene) {
    try {
        if (scene == nullptr || !nonNullAndCheckType<vlr::Transform>(transform))
            return VLRResult_InvalidArgument;

        *scene = new vlr::Scene(*context, transform);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneDestroy(
    VLRContext context,
    VLRScene scene) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);

        delete scene;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneSetTransform(
    VLRScene scene,
    VLRTransformConst localToWorld) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (!nonNullAndCheckType<vlr::Transform>(localToWorld))
            return VLRResult_InvalidArgument;

        scene->setTransform(localToWorld);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneAddChild(
    VLRScene scene,
    VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->belongsTo<vlr::InternalNode>())
            scene->addChild((vlr::InternalNode*)child);
        else if (child->belongsTo<vlr::SurfaceNode>())
            scene->addChild((vlr::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneRemoveChild(
    VLRScene scene,
    VLRNode child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        if (child->belongsTo<vlr::InternalNode>())
            scene->removeChild((vlr::InternalNode*)child);
        else if (child->belongsTo<vlr::SurfaceNode>())
            scene->removeChild((vlr::SurfaceNode*)child);
        else
            return VLRResult_InvalidArgument;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneGetNumChildren(
    VLRSceneConst scene,
    uint32_t* numChildren) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (numChildren == nullptr)
            return VLRResult_InvalidArgument;

        *numChildren = scene->getNumChildren();

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneGetChildren(
    VLRSceneConst scene,
    VLRNode* children) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (children == nullptr)
            return VLRResult_InvalidArgument;

        scene->getChildren(children);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneGetChildAt(
    VLRSceneConst scene,
    uint32_t index,
    VLRNode* child) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (child == nullptr)
            return VLRResult_InvalidArgument;

        *child = scene->getChildAt(index);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneSetEnvironment(
    VLRScene scene,
    VLRSurfaceMaterial material) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);
        if (!nonNullAndCheckType<vlr::EnvironmentEmitterSurfaceMaterial>(material))
            return VLRResult_InvalidArgument;

        scene->setEnvironment((vlr::EnvironmentEmitterSurfaceMaterial*)material);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrSceneSetEnvironmentRotation(
    VLRScene scene,
    float rotationPhi) {
    try {
        VLR_RETURN_INVALID_INSTANCE(scene, vlr::Scene);

        scene->setEnvironmentRotation(rotationPhi);

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}




VLR_API VLRResult vlrCameraCreate(
    VLRContext context,
    const char* typeName,
    VLRCamera* camera) {
    try {
        std::string sTypeName = typeName;
        if (vlr::testParamName(sTypeName, "Perspective")) {
            *camera = new vlr::PerspectiveCamera(*context);
        }
        else if (vlr::testParamName(sTypeName, "Equirectangular")) {
            *camera = new vlr::EquirectangularCamera(*context);
        }
        else {
            return VLRResult_InvalidArgument;
        }

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}

VLR_API VLRResult vlrCameraDestroy(
    VLRContext context,
    VLRCamera camera) {
    try {
        VLR_RETURN_INVALID_INSTANCE(camera, vlr::Camera);

        delete camera;

        return VLRResult_NoError;
    }
    VLR_RETURN_INTERNAL_ERROR();
}
