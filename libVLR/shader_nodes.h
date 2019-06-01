#pragma once

#include "image.h"

namespace VLR {
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
        struct OptiXProgramSet {
            optix::Program callablePrograms[nextPowerOf2((uint32_t)ShaderNodeSocketType::NumTypes)];
            uint32_t nodeProcedureSetIndex;
        };

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

        virtual ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const = 0;
        virtual uint32_t getProcedureSetIndex() const = 0;

        uint32_t getShaderNodeIndex() const { return m_nodeIndex; }
    };



    class GeometryShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
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
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        static GeometryShaderNode* getInstance(Context &context);
    };



    class Float2ShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        // Out Socket | option |
        // float      |    0-1 | s0, s1
        // float2     |      0 | (s0, s1)
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if ((stype == ShaderNodeSocketType::float1 && option < 2) ||
                (stype == ShaderNodeSocketType::float2 && option < 1))
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        bool set0(const ShaderNodeSocket &outputSocket);
        void set0(float value);
        bool set1(const ShaderNodeSocket &outputSocket);
        void set1(float value);
    };



    class Float3ShaderNode : public ShaderNode {
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
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        bool set0(const ShaderNodeSocket &outputSocket);
        void set0(float value);
        bool set1(const ShaderNodeSocket &outputSocket);
        void set1(float value);
        bool set2(const ShaderNodeSocket &outputSocket);
        void set2(float value);
    };



    class Float4ShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        bool set0(const ShaderNodeSocket &outputSocket);
        void set0(float value);
        bool set1(const ShaderNodeSocket &outputSocket);
        void set1(float value);
        bool set2(const ShaderNodeSocket &outputSocket);
        void set2(float value);
        bool set3(const ShaderNodeSocket &outputSocket);
        void set3(float value);
    };



    class ScaleAndOffsetFloatShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        // Out Socket | option |
        // float      |      0 | s0
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::float1 && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        bool setValue(const ShaderNodeSocket &outputSocket);
        bool setScale(const ShaderNodeSocket &outputSocket);
        bool setOffset(const ShaderNodeSocket &outputSocket);
        void setScale(float value);
        void setOffset(float value);
    };



    class TripletSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        void setSpectrumType(SpectrumType spectrumType);
        void setColorSpace(ColorSpace colorSpace);
        void setTriplet(float e0, float e1, float e2);
    };



    class RegularSampledSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        void setSpectrum(SpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples);
    };



    class IrregularSampledSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        void setSpectrum(SpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples);
    };



    class Float3ToSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

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

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        bool setFloat3(const ShaderNodeSocket &outputSocket);
        void setFloat3(const float value[3]);
        void setSpectrumTypeAndColorSpace(SpectrumType spectrumType, ColorSpace colorSpace);
    };



    class ScaleAndOffsetUVTextureMap2DShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        float m_offset[2];
        float m_scale[2];

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        ScaleAndOffsetUVTextureMap2DShaderNode(Context &context);
        ~ScaleAndOffsetUVTextureMap2DShaderNode();

        // Out Socket  | option |
        // TexCoord    |      0 | Texture Coordinates
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::TextureCoordinates && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        void setValues(const float offset[2], const float scale[2]);
    };



    class Image2DTextureShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, LinearImage2D*> NullImages;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        BumpType m_bumpType;
        ShaderNodeSocket m_nodeTexCoord;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        Image2DTextureShaderNode(Context &context);
        ~Image2DTextureShaderNode();

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
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        void setImage(const Image2D* image);
        void setBumpType(BumpType bumpType);
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification);
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y);
        bool setTexCoord(const ShaderNodeSocket &outputSocket);
    };



    class EnvironmentTextureShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, LinearImage2D*> NullImages;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        ShaderNodeSocket m_nodeTexCoord;

        void setupNodeDescriptor() const;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentTextureShaderNode(Context &context);
        ~EnvironmentTextureShaderNode();

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocket getSocket(ShaderNodeSocketType stype, uint32_t option) const override {
            if (stype == ShaderNodeSocketType::Spectrum && option < 1)
                return ShaderNodeSocket(this, stype, option);
            return ShaderNodeSocket();
        }
        uint32_t getProcedureSetIndex() const override {
            return OptiXProgramSets.at(m_context.getID()).nodeProcedureSetIndex;
        }

        void setImage(const Image2D* image);
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y);
        bool setTexCoord(const ShaderNodeSocket &outputSocket);

        void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const;
    };
}
