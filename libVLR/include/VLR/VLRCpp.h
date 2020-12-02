#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <stdexcept>

#include "VLR.h"
#include "basic_types.h"

namespace VLRCpp {
    class Context;
    using ContextRef = std::shared_ptr<Context>;
    using ContextConstRef = std::shared_ptr<const Context>;

    // e.g. Object
    // class ObjectHolder;
    // using ObjectRef = std::shared_ptr<ObjectHolder>;
#define VLR_PROCESS_CLASS(Name)\
    class Name ## Holder;\
    using Name ## Ref = std::shared_ptr<Name ## Holder>

    VLR_PROCESS_CLASS_LIST();
#undef VLR_PROCESS_CLASS



    static inline VLRContext getRawContext(const ContextConstRef &context);



    class ObjectHolder : public std::enable_shared_from_this<ObjectHolder> {
    protected:
        ContextConstRef m_context;
        VLRObject m_raw;

        inline VLRResult errorCheck(VLRResult errorCode) const;

    public:
        ObjectHolder(const ContextConstRef &context) : m_context(context), m_raw(nullptr) {}
        virtual ~ObjectHolder() {}

        std::string getType() const {
            const char* ret;
            errorCheck(vlrObjectGetType(m_raw, &ret));
            return ret;
        }

        template <typename VLRType>
        VLRType getRaw() const { return (VLRType)m_raw; }
    };



    struct ShaderNodePlug {
        ShaderNodeRef node; // To retain reference count.
        VLRShaderNodePlug plug;

        ShaderNodePlug() {
            std::memset(&plug, 0, sizeof(plug));
        }
        ShaderNodePlug(const ShaderNodeRef& _node, const VLRShaderNodePlug& _plug) :
            node(_node), plug(_plug) {}
    };



    struct ParameterInfo {
        ContextConstRef context;
        VLRParameterInfoConst raw;

        ParameterInfo(const ContextConstRef &_context, VLRParameterInfoConst _raw) : context(_context), raw(_raw) {}

        inline VLRResult errorCheck(VLRResult errorCode) const;

        const char* getName(VLRResult* error = nullptr) const {
            const char* ret = nullptr;
            VLRResult err = errorCheck(vlrParameterInfoGetName(raw, &ret));
            if (error)
                *error = err;
            return ret;
        }
        const char* getType(VLRResult* error = nullptr) const {
            const char* ret = nullptr;
            VLRResult err = errorCheck(vlrParameterInfoGetType(raw, &ret));
            if (error)
                *error = err;
            return ret;
        }
        uint32_t getTupleSize(VLRResult* error = nullptr) const {
            uint32_t ret;
            VLRResult err = errorCheck(vlrParameterInfoGetTupleSize(raw, &ret));
            if (error)
                *error = err;
            return ret;
        }
        VLRParameterFormFlag getFormFlags(VLRResult* error = nullptr) const {
            VLRParameterFormFlag ret;
            VLRResult err = errorCheck(vlrParameterInfoGetSocketForm(raw, &ret));
            if (error)
                  *error = err;
            return ret;
        }
    };



    class QueryableHolder : public ObjectHolder {
        std::map<const char*, ObjectRef> m_objects;

    public:
        QueryableHolder(const ContextConstRef& context) : ObjectHolder(context) {}

        inline bool get(const char* paramName, const char** enumValue) const {
            VLRResult err = errorCheck(vlrQueryableGetEnumValue(getRaw<VLRQueryable>(), paramName, enumValue));
            return err == VLRResult_NoError;
        }
        inline bool get(const char* paramName, VLR::Point3D* value) const {
            VLRPoint3D cValue;
            VLRResult err = errorCheck(vlrQueryableGetPoint3D(getRaw<VLRQueryable>(), paramName, &cValue));
            if (err != VLRResult_NoError)
                return false;
            *value = VLR::Point3D(cValue.x, cValue.y, cValue.z);
            return true;
        }
        inline bool get(const char* paramName, VLR::Vector3D* value) const {
            VLRVector3D cValue;
            VLRResult err = errorCheck(vlrQueryableGetVector3D(getRaw<VLRQueryable>(), paramName, &cValue));
            if (err != VLRResult_NoError)
                return false;
            *value = VLR::Vector3D(cValue.x, cValue.y, cValue.z);
            return true;
        }
        inline bool get(const char* paramName, VLR::Normal3D* value) const {
            VLRNormal3D cValue;
            VLRResult err = errorCheck(vlrQueryableGetNormal3D(getRaw<VLRQueryable>(), paramName, &cValue));
            if (err != VLRResult_NoError)
                return false;
            *value = VLR::Normal3D(cValue.x, cValue.y, cValue.z);
            return true;
        }
        inline bool get(const char* paramName, VLR::Quaternion* value) const {
            VLRQuaternion cValue;
            VLRResult err = errorCheck(vlrQueryableGetQuaternion(getRaw<VLRQueryable>(), paramName, &cValue));
            if (err != VLRResult_NoError)
                return false;
            *value = VLR::Quaternion(cValue.x, cValue.y, cValue.z, cValue.w);
            return true;
        }
        inline bool get(const char* paramName, float* value) const {
            VLRResult err = errorCheck(vlrQueryableGetFloat(getRaw<VLRQueryable>(), paramName, value));
            return err == VLRResult_NoError;
        }
        inline bool get(const char* paramName, float* values, uint32_t length) const {
            VLRResult err = errorCheck(vlrQueryableGetFloatTuple(getRaw<VLRQueryable>(), paramName, values, length));
            return err == VLRResult_NoError;
        }
        inline bool get(const char* paramName, const float** values, uint32_t* length) const {
            VLRResult err = errorCheck(vlrQueryableGetFloatArray(getRaw<VLRQueryable>(), paramName, values, length));
            return err == VLRResult_NoError;
        }
        inline bool get(const char* paramName, Image2DRef* image) const;
        inline bool get(const char* paramName, VLRImmediateSpectrum* spectrum) const {
            VLRResult err = errorCheck(vlrQueryableGetImmediateSpectrum(getRaw<VLRQueryable>(), paramName, spectrum));
            return err == VLRResult_NoError;
        }
        inline bool get(const char* paramName, SurfaceMaterialRef* material) const;
        inline bool get(const char* paramName, ShaderNodePlug* plug) const;

        inline bool set(const char* paramName, const char* enumValue) const {
            VLRResult err = errorCheck(vlrQueryableSetEnumValue(getRaw<VLRQueryable>(), paramName, enumValue));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const VLR::Point3D& value) const {
            VLRResult err = errorCheck(vlrQueryableSetPoint3D(getRaw<VLRQueryable>(), paramName, (VLRPoint3D*)&value));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const VLR::Vector3D& value) const {
            VLRResult err = errorCheck(vlrQueryableSetVector3D(getRaw<VLRQueryable>(), paramName, (VLRVector3D*)&value));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const VLR::Normal3D& value) const {
            VLRResult err = errorCheck(vlrQueryableSetNormal3D(getRaw<VLRQueryable>(), paramName, (VLRNormal3D*)&value));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const VLR::Quaternion& value) const {
            VLRResult err = errorCheck(vlrQueryableSetQuaternion(getRaw<VLRQueryable>(), paramName, (VLRQuaternion*)&value));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, float value) const {
            VLRResult err = errorCheck(vlrQueryableSetFloat(getRaw<VLRQueryable>(), paramName, value));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const float* values, uint32_t length) const {
            VLRResult err = errorCheck(vlrQueryableSetFloatTuple(getRaw<VLRQueryable>(), paramName, values, length));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const Image2DRef& image);
        inline bool set(const char* paramName, const VLRImmediateSpectrum& spectrum) const {
            VLRResult err = errorCheck(vlrQueryableSetImmediateSpectrum(getRaw<VLRQueryable>(), paramName, &spectrum));
            return err == VLRResult_NoError;
        }
        inline bool set(const char* paramName, const SurfaceMaterialRef& material);
        inline bool set(const char* paramName, const ShaderNodePlug& plug);

        uint32_t getNumParameters() const {
            uint32_t numParams;
            errorCheck(vlrQueryableGetNumParameters(getRaw<VLRQueryable>(), &numParams));
            return numParams;
        }
        ParameterInfo getParameterInfo(uint32_t index) const {
            VLRParameterInfoConst paramInfo;
            errorCheck(vlrQueryableGetParameterInfo(getRaw<VLRQueryable>(), index, &paramInfo));
            return ParameterInfo(m_context, paramInfo);
        }
    };



    class Image2DHolder : public QueryableHolder {
    public:
        Image2DHolder(const ContextConstRef &context) : QueryableHolder(context) {}

        uint32_t getWidth() const {
            uint32_t width;
            errorCheck(vlrImage2DGetWidth(getRaw<VLRImage2D>(), &width));
            return width;
        }
        uint32_t getHeight() const {
            uint32_t height;
            errorCheck(vlrImage2DGetHeight(getRaw<VLRImage2D>(), &height));
            return height;
        }
        uint32_t getStride() const {
            uint32_t stride;
            errorCheck(vlrImage2DGetStride(getRaw<VLRImage2D>(), &stride));
            return stride;
        }
        const char* getOriginalDataFormat() const {
            const char* format;
            errorCheck(vlrImage2DGetOriginalDataFormat(getRaw<VLRImage2D>(), &format));
            return format;
        }
        bool originalHasAlpha() const {
            bool hasAlpha;
            errorCheck(vlrImage2DOriginalHasAlpha(getRaw<VLRImage2D>(), &hasAlpha));
            return hasAlpha;
        }
    };



    class LinearImage2DHolder : public Image2DHolder {
    public:
        LinearImage2DHolder(const ContextConstRef &context,
                            const uint8_t* linearData, uint32_t width, uint32_t height,
                            const char* format, const char* spectrumType, const char* colorSpace) :
            Image2DHolder(context) {
            errorCheck(vlrLinearImage2DCreate(getRawContext(m_context), (VLRLinearImage2D*)&m_raw, const_cast<uint8_t*>(linearData), width, height, format, spectrumType, colorSpace));
        }
        ~LinearImage2DHolder() {
            errorCheck(vlrLinearImage2DDestroy(getRawContext(m_context), getRaw<VLRLinearImage2D>()));
        }
    };



    class BlockCompressedImage2DHolder : public Image2DHolder {
    public:
        BlockCompressedImage2DHolder(const ContextConstRef &context,
                                     const uint8_t* const* data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                     const char* dataFormat, const char* spectrumType, const char* colorSpace) :
            Image2DHolder(context) {
            errorCheck(vlrBlockCompressedImage2DCreate(getRawContext(m_context), (VLRBlockCompressedImage2D*)&m_raw, const_cast<uint8_t**>(data), const_cast<size_t*>(sizes), mipCount, width, height, dataFormat, spectrumType, colorSpace));
        }
        ~BlockCompressedImage2DHolder() {
            errorCheck(vlrBlockCompressedImage2DDestroy(getRawContext(m_context), getRaw<VLRBlockCompressedImage2D>()));
        }
    };



    class ShaderNodeHolder : public QueryableHolder {
    public:
        ShaderNodeHolder(const ContextConstRef &context, const char* typeName) : QueryableHolder(context) {
            errorCheck(vlrShaderNodeCreate(getRawContext(m_context), typeName, (VLRShaderNode*)&m_raw));
        }

        ShaderNodePlug getPlug(VLRShaderNodePlugType plugType, uint32_t option) {
            VLRShaderNodePlug plug;
            errorCheck(vlrShaderNodeGetPlug(getRaw<VLRShaderNode>(), plugType, option, &plug));
            return ShaderNodePlug(std::dynamic_pointer_cast<ShaderNodeHolder>(shared_from_this()), plug);
        }
    };



    class SurfaceMaterialHolder : public QueryableHolder {
    public:
        SurfaceMaterialHolder(const ContextConstRef &context, const char* typeName) : QueryableHolder(context) {
            errorCheck(vlrSurfaceMaterialCreate(getRawContext(m_context), typeName, (VLRSurfaceMaterial*)&m_raw));
        }
    };



    bool QueryableHolder::get(const char* paramName, Image2DRef* image) const {
        VLRImage2DConst cImage;
        VLRResult err = errorCheck(vlrQueryableGetImage2D(getRaw<VLRQueryable>(), paramName, &cImage));
        if (err != VLRResult_NoError)
            return false;
        if (cImage == nullptr) {
            *image = Image2DRef();
            return true;
        }
        VLRAssert(m_objects.count(paramName) > 0, "Object not owend.");
        *image = std::dynamic_pointer_cast<Image2DHolder>(m_objects.at(paramName));
        return true;
    }
    bool QueryableHolder::get(const char* paramName, SurfaceMaterialRef* material) const {
        VLRSurfaceMaterialConst cMaterial;
        VLRResult err = errorCheck(vlrQueryableGetSurfaceMaterial(getRaw<VLRQueryable>(), paramName, &cMaterial));
        if (err != VLRResult_NoError)
            return false;
        if (cMaterial == nullptr) {
            *material = SurfaceMaterialRef();
            return true;
        }
        VLRAssert(m_objects.count(paramName) > 0, "Object not owend.");
        *material = std::dynamic_pointer_cast<SurfaceMaterialHolder>(m_objects.at(paramName));
        return true;
    }
    bool QueryableHolder::get(const char* paramName, ShaderNodePlug* plug) const {
        VLRShaderNodePlug cPlug;
        VLRResult err = errorCheck(vlrQueryableGetShaderNodePlug(getRaw<VLRQueryable>(), paramName, &cPlug));
        if (err != VLRResult_NoError)
            return false;
        if (cPlug.nodeRef == (uintptr_t)nullptr) {
            *plug = ShaderNodePlug();
            return true;
        }
        VLRAssert(m_objects.count(paramName) > 0, "Object not owend.");
        *plug = ShaderNodePlug(std::dynamic_pointer_cast<ShaderNodeHolder>(m_objects.at(paramName)), cPlug);
        return true;
    }

    bool QueryableHolder::set(const char* paramName, const Image2DRef& image) {
        VLRImage2D cImage = nullptr;
        if (image)
            cImage = image->getRaw<VLRImage2D>();
        VLRResult err = errorCheck(vlrQueryableSetImage2D(getRaw<VLRQueryable>(), paramName, cImage));
        if (err != VLRResult_NoError)
            return false;
        m_objects[paramName] = image;
        return true;
    }
    bool QueryableHolder::set(const char* paramName, const SurfaceMaterialRef& material) {
        VLRSurfaceMaterial cMaterial = nullptr;
        if (material)
            cMaterial = material->getRaw<VLRSurfaceMaterial>();
        VLRResult err = errorCheck(vlrQueryableSetSurfaceMaterial(getRaw<VLRQueryable>(), paramName, cMaterial));
        if (err != VLRResult_NoError)
            return false;
        m_objects[paramName] = material;
        return true;
    }
    bool QueryableHolder::set(const char* paramName, const ShaderNodePlug& plug) {
        VLRResult err = errorCheck(vlrQueryableSetShaderNodePlug(getRaw<VLRQueryable>(), paramName, plug.plug));
        if (err != VLRResult_NoError)
            return false;
        VLRShaderNode cShaderNode = plug.node->getRaw<VLRShaderNode>();
        m_objects[paramName] = plug.node;
        return true;
    }



    class TransformHolder : public ObjectHolder {
    public:
        TransformHolder(const ContextConstRef &context) : ObjectHolder(context) {}
    };



    class StaticTransformHolder : public TransformHolder {
    public:
        StaticTransformHolder(const ContextConstRef &context, const float mat[16]) : TransformHolder(context) {
            errorCheck(vlrStaticTransformCreate(getRawContext(m_context), (VLRStaticTransform*)&m_raw, mat));
        }
        StaticTransformHolder(const ContextConstRef &context, const VLR::Matrix4x4 &mat) : TransformHolder(context) {
            float matArray[16];
            mat.getArray(matArray);
            errorCheck(vlrStaticTransformCreate(getRawContext(m_context), (VLRStaticTransform*)&m_raw, matArray));
        }
        ~StaticTransformHolder() {
            errorCheck(vlrStaticTransformDestroy(getRawContext(m_context), getRaw<VLRStaticTransform>()));
        }

        void getArrays(float mat[16], float invMat[16]) const {
            errorCheck(vlrStaticTransformGetArrays(getRaw<VLRStaticTransform>(), mat, invMat));
        }
        void getMatrices(VLR::Matrix4x4* mat, VLR::Matrix4x4* invMat) const {
            float aMat[16], aInvMat[16];
            getArrays(aMat, aInvMat);
            *mat = VLR::Matrix4x4(aMat);
            *invMat = VLR::Matrix4x4(aInvMat);
        }
    };



    class NodeHolder : public ObjectHolder {
    public:
        NodeHolder(const ContextConstRef &context) : ObjectHolder(context) {}

        void setName(const std::string &name) const {
            errorCheck(vlrNodeSetName(getRaw<VLRNode>(), name.c_str()));
        }
        const char* getName() const {
            const char* name;
            errorCheck(vlrNodeGetName(getRaw<VLRNode>(), &name));
            return name;
        }
    };



    class SurfaceNodeHolder : public NodeHolder {
    public:
        SurfaceNodeHolder(const ContextConstRef &context) : NodeHolder(context) {}
    };



    class TriangleMeshSurfaceNodeHolder : public SurfaceNodeHolder {
        std::vector<SurfaceMaterialRef> m_materials;
        std::vector<ShaderNodePlug> m_nodeNormals;
        std::vector<ShaderNodePlug> m_nodeTangents;
        std::vector<ShaderNodePlug> m_nodeAlphas;

    public:
        TriangleMeshSurfaceNodeHolder(const ContextConstRef &context, const char* name) :
            SurfaceNodeHolder(context) {
            errorCheck(vlrTriangleMeshSurfaceNodeCreate(getRawContext(m_context), (VLRTriangleMeshSurfaceNode*)&m_raw, name));
        }
        ~TriangleMeshSurfaceNodeHolder() {
            errorCheck(vlrTriangleMeshSurfaceNodeDestroy(getRawContext(m_context), getRaw<VLRTriangleMeshSurfaceNode>()));
        }

        void setVertices(VLR::Vertex* vertices, uint32_t numVertices) {
            errorCheck(vlrTriangleMeshSurfaceNodeSetVertices(getRaw<VLRTriangleMeshSurfaceNode>(), (VLRVertex*)vertices, numVertices));
        }
        void addMaterialGroup(uint32_t* indices, uint32_t numIndices,
                              const SurfaceMaterialRef &material,
                              const ShaderNodePlug &nodeNormal, const ShaderNodePlug& nodeTangent, const ShaderNodePlug &nodeAlpha) {
            m_materials.push_back(material);
            m_nodeNormals.push_back(nodeNormal);
            m_nodeTangents.push_back(nodeTangent);
            m_nodeAlphas.push_back(nodeAlpha);
            errorCheck(vlrTriangleMeshSurfaceNodeAddMaterialGroup(getRaw<VLRTriangleMeshSurfaceNode>(), indices, numIndices,
                                                                  material->getRaw<VLRSurfaceMaterial>(),
                                                                  nodeNormal.plug, nodeTangent.plug, nodeAlpha.plug));
        }
    };



    class InternalNodeHolder : public NodeHolder {
        TransformRef m_transform;
        std::map<VLRNode, NodeRef> m_children;

    public:
        InternalNodeHolder(const ContextConstRef &context, const char* name, const TransformRef &transform) :
            NodeHolder(context), m_transform(transform) {
            errorCheck(vlrInternalNodeCreate(getRawContext(m_context), (VLRInternalNode*)&m_raw, name, m_transform->getRaw<VLRTransform>()));
        }
        ~InternalNodeHolder() {
            errorCheck(vlrInternalNodeDestroy(getRawContext(m_context), getRaw<VLRInternalNode>()));
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrInternalNodeSetTransform(getRaw<VLRInternalNode>(), transform->getRaw<VLRTransform>()));
        }
        TransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrInternalNodeAddChild(getRaw<VLRInternalNode>(), rawChild));
        }
        void removeChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrInternalNodeRemoveChild(getRaw<VLRInternalNode>(), rawChild));
        }
        void addChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrInternalNodeAddChild(getRaw<VLRInternalNode>(), rawChild));
        }
        void removeChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrInternalNodeRemoveChild(getRaw<VLRInternalNode>(), rawChild));
        }
        uint32_t getNumChildren() const {
            uint32_t numChildren;
            errorCheck(vlrInternalNodeGetNumChildren(getRaw<VLRInternalNode>(), &numChildren));
            return numChildren;
        }
        void getChildren(uint32_t numChildren, NodeRef* children) const {
            auto rawChildren = new VLRNode[numChildren];
            errorCheck(vlrInternalNodeGetChildren(getRaw<VLRInternalNode>(), rawChildren));

            for (int i = 0; i < numChildren; ++i) {
                VLRNode rawChild = rawChildren[i];
                if (m_children.count(rawChild))
                    children[i] = m_children.at(rawChild);
            }

            delete[] rawChildren;
        }
        NodeRef getChildAt(uint32_t index) const {
            VLRNode rawChild;
            errorCheck(vlrInternalNodeGetChildAt(getRaw<VLRInternalNode>(), index, &rawChild));

            if (m_children.count(rawChild))
                return m_children.at(rawChild);

            return nullptr;
        }
    };



    class SceneHolder : public ObjectHolder {
        TransformRef m_transform;
        std::map<VLRNode, NodeRef> m_children;
        SurfaceMaterialRef m_matEnv;

    public:
        SceneHolder(const ContextConstRef &context, const TransformRef &transform) :
            ObjectHolder(context), m_transform(transform) {
            errorCheck(vlrSceneCreate(getRawContext(m_context), (VLRScene*)&m_raw, m_transform->getRaw<VLRTransform>()));
        }
        ~SceneHolder() {
            errorCheck(vlrSceneDestroy(getRawContext(m_context), getRaw<VLRScene>()));
        }

        void setTransform(const StaticTransformRef &transform) {
            m_transform = transform;
            errorCheck(vlrSceneSetTransform(getRaw<VLRScene>(), transform->getRaw<VLRTransform>()));
        }
        TransformRef getTransform() const {
            return m_transform;
        }

        void addChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrSceneAddChild(getRaw<VLRScene>(), rawChild));
        }
        void removeChild(const InternalNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrSceneRemoveChild(getRaw<VLRScene>(), rawChild));
        }
        void addChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children[rawChild] = child;
            errorCheck(vlrSceneAddChild(getRaw<VLRScene>(), rawChild));
        }
        void removeChild(const SurfaceNodeRef &child) {
            VLRNode rawChild = child->getRaw<VLRNode>();
            m_children.erase(rawChild);
            errorCheck(vlrSceneRemoveChild(getRaw<VLRScene>(), rawChild));
        }
        uint32_t getNumChildren() const {
            uint32_t numChildren;
            errorCheck(vlrSceneGetNumChildren(getRaw<VLRScene>(), &numChildren));
            return numChildren;
        }
        void getChildren(uint32_t numChildren, NodeRef* children) const {
            auto rawChildren = new VLRNode[numChildren];
            errorCheck(vlrSceneGetChildren(getRaw<VLRScene>(), rawChildren));

            for (int i = 0; i < numChildren; ++i) {
                VLRNode rawChild = rawChildren[i];
                if (m_children.count(rawChild))
                    children[i] = m_children.at(rawChild);
            }

            delete[] rawChildren;
        }
        NodeRef getChildAt(uint32_t index) const {
            VLRNode rawChild;
            errorCheck(vlrSceneGetChildAt(getRaw<VLRScene>(), index, &rawChild));

            if (m_children.count(rawChild))
                return m_children.at(rawChild);

            return nullptr;
        }

        void setEnvironment(const SurfaceMaterialRef &matEnv, float rotationPhi) {
            m_matEnv = matEnv;
            errorCheck(vlrSceneSetEnvironment(getRaw<VLRScene>(), m_matEnv->getRaw<VLRSurfaceMaterial>()));
            errorCheck(vlrSceneSetEnvironmentRotation(getRaw<VLRScene>(), rotationPhi));
        }
        void setEnvironmentRotation(float rotationPhi) {
            errorCheck(vlrSceneSetEnvironmentRotation(getRaw<VLRScene>(), rotationPhi));
        }
    };



    class CameraHolder : public QueryableHolder {
    public:
        CameraHolder(const ContextConstRef &context, const char* typeName) : QueryableHolder(context) {
            errorCheck(vlrCameraCreate(getRawContext(m_context), typeName, (VLRCamera*)&m_raw));
        }
    };



    class Context : public std::enable_shared_from_this<Context> {
        std::set<VLRResult> m_enabledErrors;
        VLRContext m_rawContext;
        StaticTransformRef m_identityTransform;

        Context() : m_rawContext(nullptr) {}

        void initialize(CUcontext cuContext, bool logging, uint32_t maxCallableDepth) {
            errorCheck(vlrCreateContext(&m_rawContext, cuContext, logging, maxCallableDepth));
            m_identityTransform = std::make_shared<StaticTransformHolder>(shared_from_this(), VLR::Matrix4x4::Identity());
        }

    public:
        void enableException(VLRResult errorCode) {
            if (errorCode != VLRResult_NoError && errorCode < VLRResult_NumErrors)
                m_enabledErrors.insert(errorCode);
        }
        void disableException(VLRResult errorCode) {
            if (m_enabledErrors.count(errorCode))
                m_enabledErrors.erase(errorCode);
        }
        void enableAllExceptions() {
            for (int i = 0; i < VLRResult_NumErrors; ++i) {
                if (i != VLRResult_NoError)
                    m_enabledErrors.insert((VLRResult)i);
            }
        }
        void disableAllExceptions() {
            m_enabledErrors.clear();
        }

        static ContextRef create(CUcontext cuContext, bool logging, uint32_t maxCallableDepth = 8) {
            auto ret = std::shared_ptr<Context>(new Context());
            ret->initialize(cuContext, logging, maxCallableDepth);
            ret->enableException(VLRResult_InvalidContext);
            ret->enableException(VLRResult_InvalidInstance);
            ret->enableException(VLRResult_InternalError);
            return ret;
        }

        ~Context() {
            errorCheck(vlrDestroyContext(m_rawContext));
        }

        VLRResult errorCheck(VLRResult errorCode) const {
            if (m_enabledErrors.count(errorCode))
                throw std::runtime_error(vlrGetErrorMessage(errorCode));
            return errorCode;
        }

        VLRContext get() const {
            return m_rawContext;
        }

        CUcontext getCUcontext() const {
            CUcontext cuContext;
            errorCheck(vlrContextGetCUcontext(m_rawContext, &cuContext));
            return cuContext;
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID) const {
            errorCheck(vlrContextBindOutputBuffer(m_rawContext, width, height, glBufferID));
        }

        void getOutputBuffer(CUarray* array, uint32_t* width, uint32_t* height) const {
            errorCheck(vlrContextGetOutputBuffer(m_rawContext, array, width, height));
        }

        void render(const SceneRef &scene, const CameraRef &camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            errorCheck(vlrContextRender(m_rawContext, scene->getRaw<VLRScene>(), camera->getRaw<VLRCamera>(), shrinkCoeff, firstFrame, numAccumFrames));
        }

        void debugRender(const SceneRef &scene, const CameraRef &camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames) const {
            errorCheck(vlrContextDebugRender(m_rawContext, scene->getRaw<VLRScene>(), camera->getRaw<VLRCamera>(), renderMode, shrinkCoeff, firstFrame, numAccumFrames));
        }



        LinearImage2DRef createLinearImage2D(const uint8_t* linearData, uint32_t width, uint32_t height,
                                             const char* format, const char* spectrumType, const char* colorSpace) const {
            return std::make_shared<LinearImage2DHolder>(shared_from_this(),
                                                         linearData, width, height,
                                                         format, spectrumType, colorSpace);
        }

        BlockCompressedImage2DRef createBlockCompressedImage2D(uint8_t** data, const size_t* sizes, uint32_t mipCount, uint32_t width, uint32_t height,
                                                               const char* format, const char* spectrumType, const char* colorSpace) const {
            return std::make_shared<BlockCompressedImage2DHolder>(shared_from_this(),
                                                                  data, sizes, mipCount, width, height,
                                                                  format, spectrumType, colorSpace);
        }



        ShaderNodeRef createShaderNode(const char* typeName) const {
            return std::make_shared<ShaderNodeHolder>(shared_from_this(), typeName);
        }

        SurfaceMaterialRef createSurfaceMaterial(const char* typeName) const {
            return std::make_shared<SurfaceMaterialHolder>(shared_from_this(), typeName);
        }



        StaticTransformRef createStaticTransform(const float mat[16]) const {
            return std::make_shared<StaticTransformHolder>(shared_from_this(), mat);
        }

        StaticTransformRef createStaticTransform(const VLR::Matrix4x4 &mat) const {
            return std::make_shared<StaticTransformHolder>(shared_from_this(), mat);
        }

        const StaticTransformRef &getIdentityTransform() const {
            return m_identityTransform;
        }

        TriangleMeshSurfaceNodeRef createTriangleMeshSurfaceNode(const char* name) const {
            return std::make_shared<TriangleMeshSurfaceNodeHolder>(shared_from_this(), name);
        }

        InternalNodeRef createInternalNode(const char* name, const StaticTransformRef &transform = nullptr) const {
            return std::make_shared<InternalNodeHolder>(shared_from_this(), name, transform ? transform : getIdentityTransform());
        }

        SceneRef createScene(const StaticTransformRef &transform = nullptr) const {
            return std::make_shared<SceneHolder>(shared_from_this(), transform ? transform : getIdentityTransform());
        }

        CameraRef createCamera(const char* typeName) const {
            return std::make_shared<CameraHolder>(shared_from_this(), typeName);
        }
    };



    VLRResult ObjectHolder::errorCheck(VLRResult errorCode) const {
        return m_context->errorCheck(errorCode);
    }



    VLRResult ParameterInfo::errorCheck(VLRResult errorCode) const {
        return context->errorCheck(errorCode);
    }



    inline VLRContext getRawContext(const ContextConstRef &context) {
        return context->get();
    }
}
