#pragma once

#include "context.h"

namespace VLR {
    struct ParameterInfo {
        const char* name;
        VLRParameterFormFlag formFlags;
        const char* typeName;
        uint32_t tupleSize; // 0 means variable sized array

        ParameterInfo() :
            name(nullptr),
            formFlags((VLRParameterFormFlag)0),
            typeName(nullptr), tupleSize(0) {}
        ParameterInfo(const char* _name,
                      VLRParameterFormFlag _formFlags,
                      const char* _typeName, uint32_t _tupleSize = 1) :
            name(_name),
            formFlags(_formFlags),
            typeName(_typeName), tupleSize(_tupleSize) {}
    };

    extern const char* ParameterFloat;
    extern const char* ParameterPoint3D;
    extern const char* ParameterVector3D;
    extern const char* ParameterNormal3D;
    extern const char* ParameterSpectrum;
    extern const char* ParameterAlpha;
    extern const char* ParameterTextureCoordinates;

    extern const char* ParameterImage;
    extern const char* ParameterSurfaceMaterial;

    extern const char* ParameterSpectrumType;
    extern const char* ParameterColorSpace;
    extern const char* ParameterBumpType;
    extern const char* ParameterTextureFilter;
    extern const char* ParameterTextureWrapMode;
    extern const char* ParameterCameraType;

    uint32_t getNumEnumMembers(const char* typeName);
    const char* getEnumMemberAt(const char* typeName, uint32_t index);
    uint32_t getEnumValueFromMember(const char* typeName, const char* member);
    const char* getEnumMemberFromValue(const char* typeName, uint32_t value);



    class Image2D;
    struct ImmediateSpectrum;
    class SurfaceMaterial;
    struct ShaderNodePlug;
    
    class Connectable : public Object {
        virtual const std::vector<ParameterInfo>& getParamInfos() const = 0;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        Connectable(Context& context) : Object(context) {}

        virtual bool get(const char* paramName, const char** enumValue) const {
            return false;
        }
        virtual bool get(const char* paramName, Point3D* value) const {
            return false;
        }
        virtual bool get(const char* paramName, Vector3D* value) const {
            return false;
        }
        virtual bool get(const char* paramName, Normal3D* value) const {
            return false;
        }
        virtual bool get(const char* paramName, float* values, uint32_t length) const {
            return false;
        }
        virtual bool get(const char* paramName, const float** values, uint32_t* length) const {
            return false;
        }
        virtual bool get(const char* paramName, const Image2D** image) const {
            return false;
        }
        virtual bool get(const char* paramName, ImmediateSpectrum* spectrum) const {
            return false;
        }
        virtual bool get(const char* paramName, const SurfaceMaterial** material) const {
            return false;
        }
        virtual bool get(const char* paramName, ShaderNodePlug* plug) const {
            return false;
        }

        virtual bool set(const char* paramName, const char* enumValue) {
            return false;
        }
        virtual bool set(const char* paramName, const Point3D &value) const {
            return false;
        }
        virtual bool set(const char* paramName, const Vector3D &value) const {
            return false;
        }
        virtual bool set(const char* paramName, const Normal3D &value) const {
            return false;
        }
        virtual bool set(const char* paramName, const float* values, uint32_t length) {
            return false;
        }
        virtual bool set(const char* paramName, const Image2D* image) {
            return false;
        }
        virtual bool set(const char* paramName, const ImmediateSpectrum& spectrum) {
            return false;
        }
        virtual bool set(const char* paramName, const SurfaceMaterial* material) {
            return false;
        }
        virtual bool set(const char* paramName, const ShaderNodePlug& plug) {
            return false;
        }

        uint32_t getNumParameters() const {
            const auto &paramInfos = getParamInfos();
            return (uint32_t)paramInfos.size();
        }
        const ParameterInfo* getParameterInfo(uint32_t index) const {
            const auto &paramInfos = getParamInfos();
            if (index < paramInfos.size())
                return &paramInfos[index];
            return nullptr;
        }
    };

#define VLR_DECLARE_CONNECTABLE_INTERFACE() \
    static std::vector<ParameterInfo> ParameterInfos; \
    const std::vector<ParameterInfo> &getParamInfos() const override { \
        return ParameterInfos; \
    }
}
