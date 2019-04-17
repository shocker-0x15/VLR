#pragma once

#include "image.h"

namespace VLR {
    class ShaderNode;

    struct ShaderNodeSocketIdentifier {
        const ShaderNode* node;
        union {
            struct SocketInfo {
                unsigned int outputType: 4;
                unsigned int option : 2;
            } socketInfo;
            static_assert(sizeof(SocketInfo) == sizeof(uint32_t), "sizeof(SocketInfo) is expected to be 4.");
            uint32_t socketInfoAsUInt;
        };

        ShaderNodeSocketIdentifier() : node(nullptr), socketInfoAsUInt(0) {
            socketInfo.outputType = 0;
        }
        // used in this file
        ShaderNodeSocketIdentifier(const ShaderNode* _node, VLRShaderNodeSocketType _socketType, uint32_t _option) :
            node(_node) {
            socketInfo.outputType = _socketType;
            socketInfo.option = _option;
        }
        // used in VLR.cpp
        ShaderNodeSocketIdentifier(const ShaderNode* _node, const VLRShaderNodeSocketInfo &_socketInfo) :
            node(_node), socketInfoAsUInt(_socketInfo.dummy) {}

        VLRShaderNodeSocketInfo getSocketInfo() const {
            VLRShaderNodeSocketInfo ret;
            ret.dummy = socketInfoAsUInt;
            return ret;
        }

        VLRShaderNodeSocketType getType() const {
            return (VLRShaderNodeSocketType)socketInfo.outputType;
        }

        Shared::ShaderNodeSocketID getSharedType() const;
    };



    class ShaderNode : public Object {
    protected:
        struct OptiXProgramSet {
            optix::Program callablePrograms[nextPowerOf2(NumVLRShaderNodeSocketTypes)];
            uint32_t nodeProcedureSetIndex;
        };

        uint32_t m_nodeIndex;
        const bool m_isSpectrumNode;

        struct SocketTypeToProgramPair {
            VLRShaderNodeSocketType stype;
            const char* programName;
        };
        static void commonInitializeProcedure(Context &context, const SocketTypeToProgramPair* pairs, uint32_t numPairs, OptiXProgramSet* programSet);
        static void commonFinalizeProcedure(Context &context, OptiXProgramSet &programSet);

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ShaderNode(Context &context, bool isSpectrumNode = false);
        virtual ~ShaderNode();

        virtual ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const = 0;

        uint32_t getShaderNodeIndex() const { return m_nodeIndex; }
        bool isSpectrumNode() const { return m_isSpectrumNode; }
    };



    class GeometryShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, GeometryShaderNode*> Instances;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        GeometryShaderNode(Context &context);
        ~GeometryShaderNode();

        // Out Socket   | option |
        // 0 (Point3D)  |      0 | Position
        // 1 (Normal3D) |   0, 1 | Geometric Normal, Shading Normal
        // 2 (Vector3D) |   0, 1 | Shading Tangent, Shading Bitangent
        // 3 (Point3D)  |      0 | Texture Coordinates
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if ((stype == VLRShaderNodeSocketType_Point3D && index < 1) ||
                (stype == VLRShaderNodeSocketType_Normal3D && index < 2) ||
                (stype == VLRShaderNodeSocketType_Vector3D && index < 2) ||
                (stype == VLRShaderNodeSocketType_TextureCoordinates && index < 1))
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        static GeometryShaderNode* getInstance(Context &context);
    };



    class FloatShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_node0;
        float m_imm0;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        FloatShaderNode(Context &context);
        ~FloatShaderNode();

        // Out Socket | option |
        // float      |      0 | s0
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_float && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
    };



    class Float2ShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_node0;
        ShaderNodeSocketIdentifier m_node1;
        float m_imm0;
        float m_imm1;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float2ShaderNode(Context &context);
        ~Float2ShaderNode();

        // Out Socket | option |
        // float      |    0-1 | s0, s1
        // float2     |      0 | (s0, s1)
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if ((stype == VLRShaderNodeSocketType_float && index < 2) ||
                (stype == VLRShaderNodeSocketType_float2 && index < 1))
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
        bool setNode1(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue1(float value);
    };



    class Float3ShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_node0;
        ShaderNodeSocketIdentifier m_node1;
        ShaderNodeSocketIdentifier m_node2;
        float m_imm0;
        float m_imm1;
        float m_imm2;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float3ShaderNode(Context &context);
        ~Float3ShaderNode();

        // Out Socket | option |
        // float      |    0-2 | s0, s1, s2
        // float2     |    0-1 | (s0, s1), (s1, s2)
        // float3     |      0 | (s0, s1, s2)
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if ((stype == VLRShaderNodeSocketType_float && index < 3) ||
                (stype == VLRShaderNodeSocketType_float2 && index < 2) ||
                (stype == VLRShaderNodeSocketType_float3 && index < 1))
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
        bool setNode1(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue1(float value);
        bool setNode2(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue2(float value);
    };



    class Float4ShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_node0;
        ShaderNodeSocketIdentifier m_node1;
        ShaderNodeSocketIdentifier m_node2;
        ShaderNodeSocketIdentifier m_node3;
        float m_imm0;
        float m_imm1;
        float m_imm2;
        float m_imm3;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Float4ShaderNode(Context &context);
        ~Float4ShaderNode();

        // Out Socket | option |
        // float      |    0-3 | s0, s1, s2, s3
        // float2     |    0-2 | (s0, s1), (s1, s2), (s2, s3)
        // float3     |    0-1 | (s0, s1, s2), (s1, s2, s3)
        // float4     |      0 | (s0, s1, s2, s3)
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if ((stype == VLRShaderNodeSocketType_float && index < 4) ||
                (stype == VLRShaderNodeSocketType_float2 && index < 3) ||
                (stype == VLRShaderNodeSocketType_float3 && index < 2) ||
                (stype == VLRShaderNodeSocketType_float4 && index < 1))
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
        bool setNode1(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue1(float value);
        bool setNode2(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue2(float value);
        bool setNode3(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue3(float value);
    };



    class ScaleAndOffsetFloatShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeValue;
        ShaderNodeSocketIdentifier m_nodeScale;
        ShaderNodeSocketIdentifier m_nodeOffset;
        float m_immScale;
        float m_immOffset;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ScaleAndOffsetFloatShaderNode(Context &context);
        ~ScaleAndOffsetFloatShaderNode();

        // Out Socket | option |
        // float      |      0 | s0
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_float && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        bool setNodeValue(const ShaderNodeSocketIdentifier &outputSocket);
        bool setNodeScale(const ShaderNodeSocketIdentifier &outputSocket);
        bool setNodeOffset(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueScale(float value);
        void setImmediateValueOffset(float value);
    };



    class TripletSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        VLRSpectrumType m_spectrumType;
        ColorSpace m_colorSpace;
        float m_immE0, m_immE1, m_immE2;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        TripletSpectrumShaderNode(Context &context);
        ~TripletSpectrumShaderNode();

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_Spectrum && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        void setImmediateValueSpectrumType(VLRSpectrumType spectrumType);
        void setImmediateValueColorSpace(ColorSpace colorSpace);
        void setImmediateValueTriplet(float e0, float e1, float e2);
    };



    class RegularSampledSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        VLRSpectrumType m_spectrumType;
        float m_minLambda;
        float m_maxLambda;
        float* m_values;
        uint32_t m_numSamples;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        RegularSampledSpectrumShaderNode(Context &context);
        ~RegularSampledSpectrumShaderNode();

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_Spectrum && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        void setImmediateValueSpectrum(VLRSpectrumType spectrumType, float minLambda, float maxLambda, const float* values, uint32_t numSamples);
    };



    class IrregularSampledSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        VLRSpectrumType m_spectrumType;
        float* m_lambdas;
        float* m_values;
        uint32_t m_numSamples;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        IrregularSampledSpectrumShaderNode(Context &context);
        ~IrregularSampledSpectrumShaderNode();

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_Spectrum && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        void setImmediateValueSpectrum(VLRSpectrumType spectrumType, const float* lambdas, const float* values, uint32_t numSamples);
    };



    class Vector3DToSpectrumShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        ShaderNodeSocketIdentifier m_nodeVector3D;
        Vector3D m_immVector3D;
        VLRSpectrumType m_spectrumType;
        ColorSpace m_colorSpace;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Vector3DToSpectrumShaderNode(Context &context);
        ~Vector3DToSpectrumShaderNode();

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_Spectrum && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        bool setNodeVector3D(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValueVector3D(const Vector3D &value);
        void setImmediateValueSpectrumTypeAndColorSpace(VLRSpectrumType spectrumType, ColorSpace colorSpace);
    };



    class ScaleAndOffsetUVTextureMap2DShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        float m_offset[2];
        float m_scale[2];

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ScaleAndOffsetUVTextureMap2DShaderNode(Context &context);
        ~ScaleAndOffsetUVTextureMap2DShaderNode();

        // Out Socket  | option |
        // TexCoord    |      0 | TexCoord
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_TextureCoordinates && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        void setValues(const float offset[2], const float scale[2]);
    };



    class Image2DTextureShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;
        static std::map<uint32_t, LinearImage2D*> NullImages;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        ShaderNodeSocketIdentifier m_nodeTexCoord;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        Image2DTextureShaderNode(Context &context);
        ~Image2DTextureShaderNode();

        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            uint32_t cIndex = getComponentStartIndex(m_image->getDataFormat(), stype, index);
            if (cIndex != 0xFFFFFFFF)
                return ShaderNodeSocketIdentifier(this, stype, cIndex);
            return ShaderNodeSocketIdentifier();
        }

        void setImage(const Image2D* image);
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y);
        bool setNodeTexCoord(const ShaderNodeSocketIdentifier &outputSocket);
    };



    class EnvironmentTextureShaderNode : public ShaderNode {
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        optix::TextureSampler m_optixTextureSampler;
        const Image2D* m_image;
        ShaderNodeSocketIdentifier m_nodeTexCoord;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        EnvironmentTextureShaderNode(Context &context);
        ~EnvironmentTextureShaderNode();

        // Out Socket   | option |
        // Spectrum     |      0 | Spectrum
        ShaderNodeSocketIdentifier getSocket(VLRShaderNodeSocketType stype, uint32_t index) const {
            if (stype == VLRShaderNodeSocketType_Spectrum && index < 1)
                return ShaderNodeSocketIdentifier(this, stype, index);
            return ShaderNodeSocketIdentifier();
        }

        void setImage(const Image2D* image);
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
        void setTextureWrapMode(VLRTextureWrapMode x, VLRTextureWrapMode y);
        bool setNodeTexCoord(const ShaderNodeSocketIdentifier &outputSocket);

        void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const;
    };
}
