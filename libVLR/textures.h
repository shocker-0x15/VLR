#pragma once

#include "context.h"
#include "ext/include/half.hpp"

using half_float::half;

namespace VLR {
    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    struct RGBA16Fx4 { half r, g, b, a; };
    struct RGBA32Fx4 { float r, g, b, a; };
    struct RG32Fx2 { float r, g; };
    struct Gray32F { float v; };
    struct Gray8 { uint8_t v; };

    extern const size_t sizesOfDataFormats[(uint32_t)VLRDataFormat::NumVLRDataFormats];

    class Image2D : public Object {
        uint32_t m_width, m_height;
        VLRDataFormat m_dataFormat;
        mutable optix::Buffer m_optixDataBuffer;
        mutable bool m_initOptiXObject;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static VLRDataFormat getInternalFormat(VLRDataFormat inputFormat);

        Image2D(Context &context, uint32_t width, uint32_t height, VLRDataFormat dataFormat);
        virtual ~Image2D();

        virtual Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const = 0;
        virtual Image2D* createLuminanceImage2D() const = 0;
        virtual void* createLinearImageData() const = 0;

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        VLRDataFormat getDataFormat() const {
            return m_dataFormat;
        }
        uint32_t getStride() const {
            return (uint32_t)sizesOfDataFormats[(uint32_t)m_dataFormat];
        }

        virtual optix::Buffer getOptiXObject() const;
    };



    class LinearImage2D : public Image2D {
        std::vector<uint8_t> m_data;
        mutable bool m_copyDone;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        // EN: "linearData" means data layout is linear, it doesn't mean gamma curve.
        LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, VLRDataFormat dataFormat, bool applyDegamma);

        template <typename PixelType>
        PixelType get(uint32_t x, uint32_t y) const {
            return *(PixelType*)(m_data.data() + (y * getWidth() + x) * getStride());
        }

        Image2D* createShrinkedImage2D(uint32_t width, uint32_t height) const override;
        Image2D* createLuminanceImage2D() const override;
        void* createLinearImageData() const override;

        optix::Buffer getOptiXObject() const override;
    };



    enum ShaderNodeSocketType {
        ShaderNodeSocketType_float = 0,
        ShaderNodeSocketType_float2 = 0,
        ShaderNodeSocketType_float3 = 0,
        ShaderNodeSocketType_float4 = 0,
        ShaderNodeSocketType_RGBSpectrum = 0,
        ShaderNodeSocketType_TextureCoordinates = 0,
        NumShaderNodeSocketTypes,
        ShaderNodeSocketType_Invalid
    };



    class ShaderNode : public Object {
    protected:
        uint32_t m_nodeIndex;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ShaderNode(Context &context);
        virtual ~ShaderNode();

        virtual uint32_t getNumOutputSockets() const = 0;
        virtual ShaderNodeSocketType getSocketType(uint32_t index) const = 0;

        uint32_t getShaderNodeIndex() const { return m_nodeIndex; }
    };



    struct ShaderNodeSocketIdentifier {
        const ShaderNode* node;
        uint32_t index;

        ShaderNodeSocketIdentifier() : node(nullptr), index(0) {}
        ShaderNodeSocketIdentifier(const ShaderNode* _node, uint32_t _index) : node(_node), index(_index) {}

        ShaderNodeSocketType getType() const {
            if (node == nullptr)
                return ShaderNodeSocketType_Invalid;
            return node->getSocketType(index);
        }

        Shared::NodeIndex getNodeIndex() const {
            if (node && index < node->getNumOutputSockets()) {
                Shared::NodeIndex ret;
                ret.bufferIndex = node->getShaderNodeIndex();
                ret.outSocketIndex = index;
            }
            return Shared::NodeIndex::Invalid();
        }
    };



    class FloatShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramFloat;
        };
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

        uint32_t getNumOutputSockets() const {
            return 1;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_float
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
    };



    class Float2ShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramFloat2;
        };
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

        uint32_t getNumOutputSockets() const {
            return 1;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_float2
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
        bool setNode1(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue1(float value);
    };



    class Float3ShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramFloat3;
        };
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

        uint32_t getNumOutputSockets() const {
            return 1;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_float3
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        bool setNode0(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue0(float value);
        bool setNode1(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue1(float value);
        bool setNode2(const ShaderNodeSocketIdentifier &outputSocket);
        void setImmediateValue2(float value);
    };



    class Float4ShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramFloat4;
        };
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

        uint32_t getNumOutputSockets() const {
            return 1;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_float4
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
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



    class OffsetAndScaleUVTextureMap2DShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramTexCoord;
        };
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        float m_offset[2];
        float m_scale[2];

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        OffsetAndScaleUVTextureMap2DShaderNode(Context &context);
        ~OffsetAndScaleUVTextureMap2DShaderNode();

        uint32_t getNumOutputSockets() const {
            return 1;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_TextureCoordinates
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        void setValues(const float offset[2], const float scale[2]);
    };



    class ConstantTextureShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramRGBSpectrum;
            optix::Program callableProgramAlpha;
        };
        static std::map<uint32_t, OptiXProgramSet> OptiXProgramSets;

        RGBSpectrum m_spectrum;
        float m_alpha;

        void setupNodeDescriptor() const;

    public:
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        static void initialize(Context &context);
        static void finalize(Context &context);

        ConstantTextureShaderNode(Context &context);
        ~ConstantTextureShaderNode();

        uint32_t getNumOutputSockets() const {
            return 2;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_RGBSpectrum,
                ShaderNodeSocketType_float
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        void setValues(const RGBSpectrum &spectrum, float alpha);
    };



    class Image2DTextureShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramRGBSpectrum;
            optix::Program callableProgramAlpha;
        };
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

        Image2DTextureShaderNode(Context &context);
        ~Image2DTextureShaderNode();

        uint32_t getNumOutputSockets() const {
            return 5;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_RGBSpectrum,
                ShaderNodeSocketType_float,
                ShaderNodeSocketType_float2,
                ShaderNodeSocketType_float3,
                ShaderNodeSocketType_float4,
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        void setImage(const Image2D* image);
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
        bool setNodeTexCoord(const ShaderNodeSocketIdentifier &outputSocket);
    };



    class EnvironmentTextureShaderNode : public ShaderNode {
        struct OptiXProgramSet {
            optix::Program callableProgramRGBSpectrum;
        };
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

        uint32_t getNumOutputSockets() const {
            return 1;
        }
        ShaderNodeSocketType getSocketType(uint32_t index) const override {
            ShaderNodeSocketType types[] = {
                ShaderNodeSocketType_RGBSpectrum,
            };
            if (index >= getNumOutputSockets())
                return ShaderNodeSocketType_Invalid;
            return types[index];
        }

        void setImage(const Image2D* image);
        void setTextureFilterMode(VLRTextureFilter minification, VLRTextureFilter magnification, VLRTextureFilter mipmapping);
        bool setNodeTexCoord(const ShaderNodeSocketIdentifier &outputSocket);

        void createImportanceMap(RegularConstantContinuousDistribution2D* importanceMap) const;
    };
}
