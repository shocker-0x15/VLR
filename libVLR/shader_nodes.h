#pragma once

#include "image.h"

namespace VLR {
    class ShaderNode;

    struct ShaderNodePlug {
        const ShaderNode* node;
        union {
            struct Info {
                unsigned int outputType: 4;
                unsigned int option : 2;
            } info;
            static_assert(sizeof(Info) == sizeof(uint32_t), "sizeof(Info) is expected to be 4.");
            uint32_t plugInfoAsUInt;
        };

        ShaderNodePlug() : node(nullptr), plugInfoAsUInt(0) {
            info.outputType = 0;
        }
        // used in this file
        ShaderNodePlug(const ShaderNode* _node, ShaderNodePlugType _plugType, uint32_t _option) :
            node(_node), plugInfoAsUInt(0) {
            info.outputType = (unsigned int)_plugType;
            info.option = _option;
        }
        // used in VLR.cpp
        ShaderNodePlug(const VLRShaderNodePlug &_plug) :
            node((const ShaderNode*)_plug.nodeRef), plugInfoAsUInt(_plug.info) {}

        bool isValid() const {
            return node != nullptr;
        }

        ShaderNodePlugType getType() const {
            return (ShaderNodePlugType)info.outputType;
        }

        VLRShaderNodePlug getOpaqueType() const {
            VLRShaderNodePlug ret;
            ret.nodeRef = (uintptr_t)node;
            ret.info = plugInfoAsUInt;
            return ret;
        }

        Shared::ShaderNodePlug getSharedType() const;
    };



    class ShaderNode : public Queryable {
    protected:
        struct OptiXProgramSet {
            CallableProgram callablePrograms[nextPowerOf2((uint32_t)ShaderNodePlugType::NumTypes)];
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

        struct PlugTypeToProgramPair {
            ShaderNodePlugType ptype;
            const char* programName;
        };
        static optixu::Module s_optixModule;
        static void commonInitializeProcedure(Context &context, const PlugTypeToProgramPair* pairs, uint32_t numPairs, OptiXProgramSet* programSet);
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
        ~ShaderNode();

        virtual ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const = 0;

        uint32_t getShaderNodeIndex() const { return m_nodeIndex; }
    };

#define VLR_SHADER_NODE_DECLARE_PROGRAM_SET() \
    static std::map<uint32_t, OptiXProgramSet> s_optiXProgramSets; \
    uint32_t getProcedureSetIndex() const override { \
        return s_optiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex; \
    }



    class GeometryShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();
        static std::map<uint32_t, GeometryShaderNode*> s_instances;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        GeometryShaderNode(Context &context);
        ~GeometryShaderNode();

        // Out Plug | option |
        // Point3D  |      0 | Position
        // Normal3D |   0, 1 | Geometric Normal, Shading Normal
        // Vector3D |   0, 1 | Shading Tangent, Shading Bitangent
        // TexCoord |      0 | Texture Coordinates
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if ((ptype == ShaderNodePlugType::Point3D && option < 1) ||
                (ptype == ShaderNodePlugType::Normal3D && option < 2) ||
                (ptype == ShaderNodePlugType::Vector3D && option < 2) ||
                (ptype == ShaderNodePlugType::TextureCoordinates && option < 1))
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }

        static GeometryShaderNode* getInstance(Context &context);
    };



    class TangentShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        TangentType m_immTangentType;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context& context);
        static void finalize(Context& context);

        TangentShaderNode(Context& context);
        ~TangentShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;

        bool set(const char* paramName, const char* enumValue) override;

        // Out Plug | option |
        // Vector3D |      0 | tangent
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::Vector3D && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class Float2ShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodePlug m_node0;
        ShaderNodePlug m_node1;
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
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        // Out Plug | option |
        // float    |    0-1 | s0, s1
        // float2   |      0 | (s0, s1)
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if ((ptype == ShaderNodePlugType::float1 && option < 2) ||
                (ptype == ShaderNodePlugType::float2 && option < 1))
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class Float3ShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodePlug m_node0;
        ShaderNodePlug m_node1;
        ShaderNodePlug m_node2;
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
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        // Out Plug | option |
        // float    |    0-2 | s0, s1, s2
        // float2   |    0-1 | (s0, s1), (s1, s2)
        // float3   |      0 | (s0, s1, s2)
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if ((ptype == ShaderNodePlugType::float1 && option < 3) ||
                (ptype == ShaderNodePlugType::float2 && option < 2) ||
                (ptype == ShaderNodePlugType::float3 && option < 1))
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class Float4ShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodePlug m_node0;
        ShaderNodePlug m_node1;
        ShaderNodePlug m_node2;
        ShaderNodePlug m_node3;
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
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        // Out Plug | option |
        // float    |    0-3 | s0, s1, s2, s3
        // float2   |    0-2 | (s0, s1), (s1, s2), (s2, s3)
        // float3   |    0-1 | (s0, s1, s2), (s1, s2, s3)
        // float4   |      0 | (s0, s1, s2, s3)
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if ((ptype == ShaderNodePlugType::float1 && option < 4) ||
                (ptype == ShaderNodePlugType::float2 && option < 3) ||
                (ptype == ShaderNodePlugType::float3 && option < 2) ||
                (ptype == ShaderNodePlugType::float4 && option < 1))
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class ScaleAndOffsetFloatShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodePlug m_nodeValue;
        ShaderNodePlug m_nodeScale;
        ShaderNodePlug m_nodeOffset;
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
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodePlug& plug) override;

        // Out Plug | option |
        // float    |      0 | s0
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::float1 && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class TripletSpectrumShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

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

        // Out Plug | option |
        // Spectrum |      0 | Spectrum
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::Spectrum && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class RegularSampledSpectrumShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

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

        // Out Plug | option |
        // Spectrum |      0 | Spectrum
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::Spectrum && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class IrregularSampledSpectrumShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

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

        // Out Plug | option |
        // Spectrum |      0 | Spectrum
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::Spectrum && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class Float3ToSpectrumShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();

        ShaderNodePlug m_nodeFloat3;
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
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const ShaderNodePlug & plug) override;

        // Out Plug | option |
        // Spectrum |      0 | Spectrum
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::Spectrum && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class ScaleAndOffsetUVTextureMap2DShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

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

        // Out Plug | option |
        // TexCoord |      0 | Texture Coordinates
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::TextureCoordinates && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }
    };



    class Image2DTextureShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();
        static std::map<uint32_t, LinearImage2D*> NullImages;

        cudau::TextureSampler m_textureSampler;
        const Image2D* m_image;
        BumpType m_bumpType;
        float m_bumpCoeff;
        TextureFilter m_xyFilter;
        TextureWrapMode m_wrapU;
        TextureWrapMode m_wrapV;
        ShaderNodePlug m_nodeTexCoord;

        void setupNodeDescriptor();

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Image2DTextureShaderNode(Context &context);
        ~Image2DTextureShaderNode();

        bool get(const char* paramName, const char** enumValue) const override;
        bool get(const char* paramName, float* values, uint32_t length) const override;
        bool get(const char* paramName, const Image2D** image) const override;
        bool get(const char* paramName, ShaderNodePlug* plug) const override;

        bool set(const char* paramName, const char* enumValue) override;
        bool set(const char* paramName, const float* values, uint32_t length) override;
        bool set(const char* paramName, const Image2D* image) override;
        bool set(const char* paramName, const ShaderNodePlug &plug) override;

        // Out Plug | option |
        // float    |    0-3 | s0, s1, s2, s3
        // float2   |    0-2 | (s0, s1), (s1, s2), (s2, s3)
        // float3   |    0-1 | (s0, s1, s2), (s1, s2, s3)
        // float4   |      0 | (s0, s1, s2, s3)
        // Normal3D |    0-3 | DX Normal Map, GL Normal Map, Height Map, option 2, 3 are supported only with height map.
        // Spectrum |      0 | Spectrum
        // Alpha    |    0-3 | s0, s1, s2, s3
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            uint32_t cIndex = getComponentStartIndex(m_image->getDataFormat(), m_bumpType, ptype, option);
            if (cIndex != 0xFFFFFFFF)
                return ShaderNodePlug(this, ptype, cIndex);
            return ShaderNodePlug();
        }
    };



    class EnvironmentTextureShaderNode : public ShaderNode {
        VLR_DECLARE_QUERYABLE_INTERFACE();

        VLR_SHADER_NODE_DECLARE_PROGRAM_SET();
        static std::map<uint32_t, LinearImage2D*> NullImages;

        cudau::TextureSampler m_textureSampler;
        const Image2D* m_image;
        TextureFilter m_xyFilter;

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

        // Out Plug | option |
        // Spectrum |      0 | Spectrum
        ShaderNodePlug getPlug(ShaderNodePlugType ptype, uint32_t option) const override {
            if (ptype == ShaderNodePlugType::Spectrum && option < 1)
                return ShaderNodePlug(this, ptype, option);
            return ShaderNodePlug();
        }

        void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const;
    };



#undef VLR_SHADER_NODE_DECLARE_PROGRAM_SET
}
