#pragma once

#include "image.h"

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

    extern const std::map<std::string, std::map<std::string, uint32_t>> g_enumTables;
    extern const std::map<std::string, std::map<uint32_t, std::string>> g_invEnumTables;



    class ShaderNode;

    struct ShaderNodeSocket {
        const ShaderNode* node;
        union {
            struct Info {
                unsigned int outputType: 4;
                unsigned int option : 2;
            } info;
            static_assert(sizeof(Info) == sizeof(uint32_t), "sizeof(Info) is expected to be 4.");
            uint32_t socketInfoAsUInt;
        };

        ShaderNodeSocket() : node(nullptr), socketInfoAsUInt(0) {
            info.outputType = 0;
        }
        // used in this file
        ShaderNodeSocket(const ShaderNode* _node, ShaderNodeSocketType _socketType, uint32_t _option) :
            node(_node), socketInfoAsUInt(0) {
            info.outputType = (unsigned int)_socketType;
            info.option = _option;
        }
        // used in VLR.cpp
        ShaderNodeSocket(const VLRShaderNodeSocket &_socket) :
            node((const ShaderNode*)_socket.nodeRef), socketInfoAsUInt(_socket.info) {}

        ShaderNodeSocketType getType() const {
            return (ShaderNodeSocketType)info.outputType;
        }

        VLRShaderNodeSocket getOpaqueType() const {
            VLRShaderNodeSocket ret;
            ret.nodeRef = (uintptr_t)node;
            ret.info = socketInfoAsUInt;
            return ret;
        }

        Shared::ShaderNodeSocket getSharedType() const;
    };



    class ShaderNode : public Object {
    protected:
        virtual const std::vector<ParameterInfo>& getParamInfos() const = 0;

        struct OptiXProgramSet {
            optix::Program callablePrograms[nextPowerOf2((uint32_t)ShaderNodeSocketType::NumTypes)];
            uint32_t nodeProcedureSetIndex;
        };
    public:
        virtual uint32_t getProcedureSetIndex() const = 0;

    protected:
        uint32_t m_nodeIndex;
        union {
            Shared::SmallNodeDescriptor smallNodeDesc;
            Shared::MediumNodeDescriptor mediumNodeDesc;
            Shared::LargeNodeDescriptor largeNodeDesc;
        };
        int32_t m_nodeSizeClass;

        struct SocketTypeToProgramPair {
            ShaderNodeSocketType stype;
            const char* programName;
        };
        static void commonInitializeProcedure(Context &context, const SocketTypeToProgramPair* pairs, uint32_t numPairs, OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet);

        template <typename T>
        T* getData() const {
            if (m_nodeSizeClass == 0)
                return smallNodeDesc.getData<T>();
            else if (m_nodeSizeClass == 1)
                return mediumNodeDesc.getData<T>();
            else if (m_nodeSizeClass == 2)
                return largeNodeDesc.getData<T>();
            else
                VLRAssert_ShouldNotBeCalled();
            return nullptr;
        }
        void updateNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        ShaderNode(Context &context, size_t sizeOfNode);
        virtual ~ShaderNode();

        virtual bool get(const char* paramName, const char** enumValue) const {
            return false;
        };
        virtual bool get(const char* paramName, Point3D* point) const {
            return false;
        }
        virtual bool get(const char* paramName, Vector3D* vector) const {
            return false;
        }
        virtual bool get(const char* paramName, Normal3D* normal) const {
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
        virtual bool get(const char* paramName, ShaderNodeSocket* socket) const {
            return false;
        }

        virtual bool set(const char* paramName, const char* enumValue) {
            return false;
        };
        virtual bool set(const char* paramName, const Point3D &point) {
            return false;
        }
        virtual bool set(const char* paramName, const Vector3D &vector) {
            return false;
        }
        virtual bool set(const char* paramName, const Normal3D &normal) {
            return false;
        }
        virtual bool set(const char* paramName, const float* values, uint32_t length) {
            return false;
        }
        virtual bool set(const char* paramName, const Image2D* image) {
            return false;
        }
        virtual bool set(const char* paramName, const ShaderNodeSocket &socket) {
            return false;
        }

        virtual ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const = 0;
        
        uint32_t getShaderNodeIndex() const { return m_nodeIndex; }
    };

#define VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS() \
    static std::vector<ParameterInfo> ParameterInfos; \
    const std::vector<ParameterInfo>& getParamInfos() const override { \
        return ParameterInfos; \
    }

#define VLR_SHADER_NODE_DECLARE_PROGRAM_SET() \
    static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets; \
    uint32_t getProcedureSetIndex() const override { \
        return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex; \
    }



    class GeometryShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();
        static std::map<uint32_t, GeometryShaderNode*> Instances;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        GeometryShaderNode(Context &context);
        ~GeometryShaderNode();

        // Out Socket | option |
        // Point3D    |      0 | Position
        // Normal3D   |   0, 1 | Geometric Normal, Shading Normal
        // Vector3D   |   0, 1 | Shading Tangent, Shading Bitangent
        // TexCoord   |      0 | Texture Coordinates
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if ((stype == ShaderNodeSocketType::Point3D && option < 1) ||
                (stype == ShaderNodeSocketType::Normal3D && option < 2) ||
                (stype == ShaderNodeSocketType::Vector3D && option < 2) ||
                (stype == ShaderNodeSocketType::TextureCoordinates && option < 1))
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }

        static GeometryShaderNode* getInstance(Context &context);
    };



    class Float2ShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodeSocket m_node0;
        ShaderNodeSocket m_node1;
        float m_imm0;
        float m_imm1;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float2ShaderNode(Context &context);
        ~Float2ShaderNode();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket);

        // Out Socket | option |
        // float      |    0-1 | s0, s1
        // float2     |      0 | (s0, s1)
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if ((stype == ShaderNodeSocketType::float1 && option < 2) ||
                (stype == ShaderNodeSocketType::float2 && option < 1))
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class Float3ShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocket m_node0;
        ShaderNodeSocket m_node1;
        ShaderNodeSocket m_node2;
        float m_imm0;
        float m_imm1;
        float m_imm2;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float3ShaderNode(Context &context);
        ~Float3ShaderNode();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket);

        // Out Socket | option |
        // float      |    0-2 | s0, s1, s2
        // float2     |    0-1 | (s0, s1), (s1, s2)
        // float3     |      0 | (s0, s1, s2)
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if ((stype == ShaderNodeSocketType::float1 && option < 3) ||
                (stype == ShaderNodeSocketType::float2 && option < 2) ||
                (stype == ShaderNodeSocketType::float3 && option < 1))
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class Float4ShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodeSocket m_node0;
        ShaderNodeSocket m_node1;
        ShaderNodeSocket m_node2;
        ShaderNodeSocket m_node3;
        float m_imm0;
        float m_imm1;
        float m_imm2;
        float m_imm3;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float4ShaderNode(Context &context);
        ~Float4ShaderNode();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket);

        // Out Socket | option |
        // float      |    0-3 | s0, s1, s2, s3
        // float2     |    0-2 | (s0, s1), (s1, s2), (s2, s3)
        // float3     |    0-1 | (s0, s1, s2), (s1, s2, s3)
        // float4     |      0 | (s0, s1, s2, s3)
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if ((stype == ShaderNodeSocketType::float1 && option < 4) ||
                (stype == ShaderNodeSocketType::float2 && option < 3) ||
                (stype == ShaderNodeSocketType::float3 && option < 2) ||
                (stype == ShaderNodeSocketType::float4 && option < 1))
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class ScaleAndOffsetFloatShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodeSocket m_nodeValue;
        ShaderNodeSocket m_nodeScale;
        ShaderNodeSocket m_nodeOffset;
        float m_immScale;
        float m_immOffset;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        ScaleAndOffsetFloatShaderNode(Context &context);
        ~ScaleAndOffsetFloatShaderNode();

        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodeSocket& socket);

        // Out Socket | option |
        // float      |      0 | s0
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::float1 && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class TripletSpectrumShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        SpectrumType m_spectrumType;
        ColorSpace m_colorSpace;
        float m_immE0, m_immE1, m_immE2;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        TripletSpectrumShaderNode(Context &context);
        ~TripletSpectrumShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, float* values, uint32_t length) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class RegularSampledSpectrumShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        SpectrumType m_spectrumType;
        float m_minLambda;
        float m_maxLambda;
        float* m_values;
        uint32_t m_numSamples;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        RegularSampledSpectrumShaderNode(Context &context);
        ~RegularSampledSpectrumShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, const float** values, uint32_t* length) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class IrregularSampledSpectrumShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        SpectrumType m_spectrumType;
        float* m_lambdas;
        float* m_values;
        uint32_t m_numSamples;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        IrregularSampledSpectrumShaderNode(Context &context);
        ~IrregularSampledSpectrumShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, const float** values, uint32_t* length) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class Float3ToSpectrumShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodeSocket m_nodeFloat3;
        float m_immFloat3[3];
        SpectrumType m_spectrumType;
        ColorSpace m_colorSpace;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float3ToSpectrumShaderNode(Context &context);
        ~Float3ToSpectrumShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodeSocket &socket) override;

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class ScaleAndOffsetUVTextureMap2DShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        float m_offset[2];
        float m_scale[2];

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        ScaleAndOffsetUVTextureMap2DShaderNode(Context &context);
        ~ScaleAndOffsetUVTextureMap2DShaderNode();

        bool get(const char* paramName, float* values, uint32_t length) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;

        // Out Socket  | option |
        // TexCoord    |      0 | Texture Coordinates
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::TextureCoordinates && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
    };



    class Image2DTextureShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();
        static std::map<uint32_t, LinearImage2D*> NullImages;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        BumpType m_bumpType;
        VLRTextureFilter m_minFilter;
        VLRTextureFilter m_magFilter;
        VLRTextureWrapMode m_wrapU;
        VLRTextureWrapMode m_wrapV;
        ShaderNodeSocket m_nodeTexCoord;

        void setupNodeDescriptor();

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Image2DTextureShaderNode(Context &context);
        ~Image2DTextureShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, const Image2D** image) const override;
        bool get(const char* paramName, ShaderNodeSocket* socket) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const Image2D* image) override;
        bool set(const char* paramName, const ShaderNodeSocket &socket) override;

        // Out Socket | option |
        // float      |    0-3 | s0, s1, s2, s3
        // float2     |    0-2 | (s0, s1), (s1, s2), (s2, s3)
        // float3     |    0-1 | (s0, s1, s2), (s1, s2, s3)
        // float4     |      0 | (s0, s1, s2, s3)
        // Normal3D   |    0-3 | DX Normal Map, GL Normal Map, Height Map, option 2, 3 are supported only with height map.
        // Spectrum   |      0 | Spectrum
        // Alpha      |    0-3 | s0, s1, s2, s3
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            uint32_t cIndex = getComponentStartIndex(m_image->getDataFormat(), m_bumpType, stype, option);
            if (cIndex != 0xFFFFFFFF)
                return ShaderNodeSocket(this, stype, cIndex);
            return ShaderNodeSocket();
        }
    };



    class EnvironmentTextureShaderNode : public ShaderNode {
        VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();
        static std::map<uint32_t, LinearImage2D*> NullImages;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        VLRTextureFilter m_minFilter;
        VLRTextureFilter m_magFilter;

        void setupNodeDescriptor();

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentTextureShaderNode(Context &context);
        ~EnvironmentTextureShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, const Image2D** image) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const Image2D* image) override;

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }

        void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const;
    };
}

#undef VLR_SHADER_NODE_DECLARE_PROGRAM_SET
#undef VLR_SHADER_NODE_DECLARE_PARAMETER_INFOS
