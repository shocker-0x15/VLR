#pragma once

#include "basic_types_internal.h"
#include "rgb_spectrum_types.h"
#include "spectrum_types.h"
#if defined(VLR_Host)
#include "../ext/include/half.hpp"
#endif

namespace VLR {
#if defined(VLR_Host)
    using half_float::half;
#else
    struct half {
        uint16_t raw;

        operator float() const {
            uint32_t bits = (uint32_t)(raw & 0x8000) << 16;
            uint32_t abs = raw & 0x7FFF;
            if (abs) {
                // JP: halfの指数部が   無限大 or NaN       を表す(11111)       場合: floatビット: (* 11100000 00000000000000000000000)
                //                    正規化数 or 非正規化数を表す(00000-11110) 場合:              (* 01110000 00000000000000000000000)
                bits |= 0x38000000 << (uint32_t)(abs >= 0x7C00);
                // JP: halfの指数部が非正規化数を表す(00000) 場合: 0x0001-0x03FF (* 00000 **********)
                //     正規化数になるまでhalfをビットシフト、floatの指数部を1ずつ減算。
                for (; abs < 0x400; abs <<= 1, bits -= 0x800000);
                // JP: halfの指数部が 無限大 or NaN を表す場合 0x7C00-0x7FFF (0       11111 **********): (0          00011111 **********0000000000000) を加算 => floatの指数ビットは0xFFになる。
                //                    正規化数      を表す場合 0x0400-0x7BFF (0 00001-11110 **********): (0 00000001-00011110 **********0000000000000) を加算 => floatの指数ビットは0x71-0x8Eになる。
                bits += (uint32_t)(abs) << 13;
            }
            return *(float*)&bits;
        }
    };
#endif

#if defined(VLR_USE_SPECTRAL_RENDERING)
    using WavelengthSamples = WavelengthSamplesTemplate<float, NumSpectralSamples>;
    using SampledSpectrum = SampledSpectrumTemplate<float, NumSpectralSamples>;
    using DiscretizedSpectrum = DiscretizedSpectrumTemplate<float, NumStrataForStorage>;
    using SpectrumStorage = SpectrumStorageTemplate<float, NumStrataForStorage>;
    using TripletSpectrum = UpsampledSpectrum;
#else
    using WavelengthSamples = RGBWavelengthSamplesTemplate<float>;
    using SampledSpectrum = RGBSpectrumTemplate<float>;
    using DiscretizedSpectrum = RGBSpectrumTemplate<float>;
    using SpectrumStorage = RGBStorageTemplate<float>;
    using TripletSpectrum = RGBSpectrum;
#endif

    using DiscretizedSpectrumAlwaysSpectral = DiscretizedSpectrumTemplate<float, NumStrataForStorage>;

#if defined(VLR_Device)
#   if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
    rtDeclareVariable(int32_t, UpsampledSpectrum_spectrum_grid, , );
    rtDeclareVariable(int32_t, UpsampledSpectrum_spectrum_data_points, , );
#   elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
    rtDeclareVariable(int32_t, UpsampledSpectrum_maxBrightnesses, , );
    rtDeclareVariable(int32_t, UpsampledSpectrum_coefficients_sRGB_D65, , );
    rtDeclareVariable(int32_t, UpsampledSpectrum_coefficients_sRGB_E, , );
#   endif

    rtDeclareVariable(DiscretizedSpectrumAlwaysSpectral::CMF, DiscretizedSpectrum_xbar, , );
    rtDeclareVariable(DiscretizedSpectrumAlwaysSpectral::CMF, DiscretizedSpectrum_ybar, , );
    rtDeclareVariable(DiscretizedSpectrumAlwaysSpectral::CMF, DiscretizedSpectrum_zbar, , );
    rtDeclareVariable(float, DiscretizedSpectrum_integralCMF, , );
#endif

    RT_FUNCTION HOST_INLINE TripletSpectrum createTripletSpectrum(VLRSpectrumType spectrumType, ColorSpace colorSpace, float e0, float e1, float e2) {
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(spectrumType, colorSpace, e0, e1, e2);
#else
        float XYZ[3];

        switch (colorSpace.value) {
        case ColorSpace::Rec709_D65_sRGBGamma: {
            e0 = sRGB_degamma(e0);
            e1 = sRGB_degamma(e1);
            e2 = sRGB_degamma(e2);
            // pass to Rec709 (D65)
        }
        case ColorSpace::Rec709_D65: {
            float RGB[3] = { e0, e1, e2 };
            switch (spectrumType) {
            case VLRSpectrumType_Reflectance:
            case VLRSpectrumType_IndexOfRefraction:
            case VLRSpectrumType_NA:
                transformTristimulus(mat_Rec709_E_to_XYZ, RGB, XYZ);
                break;
            case VLRSpectrumType_LightSource:
                transformTristimulus(mat_Rec709_D65_to_XYZ, RGB, XYZ);
                break;
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
            break;
        }
        case ColorSpace::XYZ: {
            XYZ[0] = e0;
            XYZ[1] = e1;
            XYZ[2] = e2;
            break;
        }
        case ColorSpace::xyY: {
            VLRAssert(e0 >= 0.0f && e1 >= 0.0f && e0 <= 1.0f && e1 <= 1.0f && e2 >= 0.0f,
                      "xy should be in [0, 1], Y should not be negative.");
            if (e1 == 0) {
                XYZ[0] = XYZ[1] = XYZ[2] = 0;
                break;
            }
            float z = 1 - (e0 + e1);
            float b = e2 / e1;
            XYZ[0] = e0 * b;
            XYZ[1] = e2;
            XYZ[2] = z * b;
            break;
        }
        default:
            VLRAssert_NotImplemented();
            break;
        }

        float RGB[3];
        transformToRenderingRGB(spectrumType, XYZ, RGB);
        return RGBSpectrum(RGB[0], RGB[1], RGB[2]);
#endif
    }



    struct DataFormat {
        enum Value {
            RGB8x3 = 0,
            RGB_8x4,
            RGBA8x4,
            RGBA16Fx4,
            RGBA32Fx4,
            RG32Fx2,
            Gray32F,
            Gray8,
            GrayA8x2,
            BC1,
            BC2,
            BC3,
            BC4,
            BC4_Signed,
            BC5,
            BC5_Signed,
            BC6H,
            BC6H_Signed,
            BC7,
            // ---- Internal Formats ----
            uvsA8x4,
            uvsA16Fx4,
            NumDataFormats
        } value;

        RT_FUNCTION constexpr DataFormat(Value v = RGBA8x4) : value(v) {}
        RT_FUNCTION constexpr DataFormat(VLRDataFormat v) : value((Value)v) {}
        RT_FUNCTION explicit operator uint32_t() const {
            return value;
        }
        RT_FUNCTION explicit operator VLRDataFormat() const {
            return (VLRDataFormat)value;
        }
        RT_FUNCTION constexpr bool operator==(Value v) const {
            return value == v;
        }
        RT_FUNCTION constexpr bool operator<(DataFormat v) const {
            return value < v.value;
        }
        RT_FUNCTION constexpr bool operator>(DataFormat v) const {
            return value >
 v.value;
        }
        RT_FUNCTION constexpr bool operator<=(DataFormat v) const {
            return value < v.value && value == v.value;
        }
        RT_FUNCTION constexpr bool operator>=(DataFormat v) const {
            return value > v.value && value == v.value;
        }
    };



    namespace Shared {
        template <typename RealType>
        class DiscreteDistribution1DTemplate {
            rtBufferId<RealType, 1> m_PMF;
            rtBufferId<RealType, 1> m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            DiscreteDistribution1DTemplate(const rtBufferId<RealType, 1> &PMF, const rtBufferId<RealType, 1> &CDF, RealType integral, uint32_t numValues) : 
            m_PMF(PMF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {
            }

            RT_FUNCTION DiscreteDistribution1DTemplate() {}
            RT_FUNCTION ~DiscreteDistribution1DTemplate() {}

            RT_FUNCTION uint32_t sample(RealType u, RealType* prob) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1) {
                    int newIdx = idx - d;
                    if (newIdx > 0 && m_CDF[newIdx] > u)
                        idx = newIdx;
                }
                --idx;
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *prob = m_PMF[idx];
                return idx;
            }
            RT_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1) {
                    int newIdx = idx - d;
                    if (newIdx > 0 && m_CDF[newIdx] > u)
                        idx = newIdx;
                }
                --idx;
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *prob = m_PMF[idx];
                *remapped = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return idx;
            }
            RT_FUNCTION RealType evaluatePMF(uint32_t idx) const {
                VLRAssert(idx >= 0 && idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
                return m_PMF[idx];
            }

            RT_FUNCTION RealType integral() const { return m_integral; }
            RT_FUNCTION uint32_t numValues() const { return m_numValues; }
        };

        using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution1DTemplate {
            rtBufferId<RealType, 1> m_PDF;
            rtBufferId<RealType, 1> m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            RegularConstantContinuousDistribution1DTemplate(const rtBufferId<RealType, 1> &PDF, const rtBufferId<RealType, 1> &CDF, RealType integral, uint32_t numValues) :
                m_PDF(PDF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {
            }

            RT_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}
            RT_FUNCTION ~RegularConstantContinuousDistribution1DTemplate() {}

            RT_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
                VLRAssert(u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1) {
                    int newIdx = idx - d;
                    if (newIdx > 0 && m_CDF[newIdx] > u)
                        idx = newIdx;
                }
                --idx;
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *probDensity = m_PDF[idx];
                RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return (idx + t) / m_numValues;
            }
            RT_FUNCTION RealType evaluatePDF(RealType smp) const {
                VLRAssert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
                int32_t idx = std::min<int32_t>(m_numValues - 1, smp * m_numValues);
                return m_PDF[idx];
            }
            RT_FUNCTION RealType integral() const { return m_integral; }

            RT_FUNCTION uint32_t numValues() const { return m_numValues; }
        };

        using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution2DTemplate {
            rtBufferId<RegularConstantContinuousDistribution1DTemplate<RealType>, 1> m_1DDists;
            RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

        public:
            RegularConstantContinuousDistribution2DTemplate(const rtBufferId<RegularConstantContinuousDistribution1DTemplate<RealType>, 1> &_1DDists, 
                                                            const RegularConstantContinuousDistribution1DTemplate<RealType> &top1DDist) :
                m_1DDists(_1DDists), m_top1DDist(top1DDist) {
            }

            RT_FUNCTION RegularConstantContinuousDistribution2DTemplate() {}
            RT_FUNCTION ~RegularConstantContinuousDistribution2DTemplate() {}

            RT_FUNCTION void sample(RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
                RealType topPDF;
                *d1 = m_top1DDist.sample(u1, &topPDF);
                uint32_t idx1D = std::min(uint32_t(m_top1DDist.numValues() * *d1), m_top1DDist.numValues() - 1);
                *d0 = m_1DDists[idx1D].sample(u0, probDensity);
                *probDensity *= topPDF;
            }
            RT_FUNCTION RealType evaluatePDF(RealType d0, RealType d1) const {
                uint32_t idx1D = std::min(uint32_t(m_top1DDist.numValues() * d1), m_top1DDist.numValues() - 1);
                return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
            }
        };

        using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



        class StaticTransform {
            Matrix4x4 m_matrix;
            Matrix4x4 m_invMatrix;

        public:
            RT_FUNCTION StaticTransform(const Matrix4x4 &m = Matrix4x4::Identity()) : m_matrix(m), m_invMatrix(invert(m)) {}

            RT_FUNCTION Vector3D operator*(const Vector3D &v) const { return m_matrix * v; }
            RT_FUNCTION Vector4D operator*(const Vector4D &v) const { return m_matrix * v; }
            RT_FUNCTION Point3D operator*(const Point3D &p) const { return m_matrix * p; }
            RT_FUNCTION Normal3D operator*(const Normal3D &n) const {
                // The length of the normal is changed if the transform has scaling, so it requires normalization.
                return Normal3D(m_invMatrix.m00 * n.x + m_invMatrix.m10 * n.y + m_invMatrix.m20 * n.z,
                                m_invMatrix.m01 * n.x + m_invMatrix.m11 * n.y + m_invMatrix.m21 * n.z,
                                m_invMatrix.m02 * n.x + m_invMatrix.m12 * n.y + m_invMatrix.m22 * n.z);
            }

            RT_FUNCTION StaticTransform operator*(const Matrix4x4 &m) const { return StaticTransform(m_matrix * m); }
            RT_FUNCTION StaticTransform operator*(const StaticTransform &t) const { return StaticTransform(m_matrix * t.m_matrix); }
            RT_FUNCTION bool operator==(const StaticTransform &t) const { return m_matrix == t.m_matrix; }
            RT_FUNCTION bool operator!=(const StaticTransform &t) const { return m_matrix != t.m_matrix; }
        };



        struct NodeProcedureSet {
            int32_t progs[nextPowerOf2(NumVLRShaderNodeSocketTypes)];
        };



        union ShaderNodeSocketID {
            struct {
                unsigned int nodeDescIndex : 25;
                unsigned int socketType : 4;
                unsigned int option : 2;
                //bool isSpectrumNode : 1; // using bool leads the size of this struct to be 8 in MSVC (C++ spec).
                unsigned int isSpectrumNode : 1;
            };
            uint32_t asUInt;

            // const NodeDescriptor &nodeDesc = pv_nodeDescriptorBuffer[socket.nodeDescIndex];
            // ProgSigT program = (ProgSigT)pv_nodeProcedureSetBuffer[nodeDesc.procSetIndex].progs[socket.socketIndex];
            // return program(nodeDesc.data, socket.option, surfPt, wls);

            RT_FUNCTION ShaderNodeSocketID() {}
            explicit constexpr ShaderNodeSocketID(uint32_t ui) : asUInt(ui) {}
            RT_FUNCTION bool isValid() const { return asUInt != 0xFFFFFFFF; }

            static constexpr ShaderNodeSocketID Invalid() { return ShaderNodeSocketID(0xFFFFFFFF); }
        };
        static_assert(sizeof(ShaderNodeSocketID) == 4, "Unexpected Size");

        struct NodeDescriptor {
            uint32_t procSetIndex;
#define VLR_MAX_NUM_NODE_DESCRIPTOR_SLOTS (15)
            uint32_t data[VLR_MAX_NUM_NODE_DESCRIPTOR_SLOTS];

            template <typename T>
            T* getData() const {
                static_assert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };

        struct SpectrumNodeDescriptor {
            uint32_t procSetIndex;
#if defined(VLR_USE_SPECTRAL_RENDERING)
#   define VLR_MAX_NUM_SPECTRUM_NODE_DESCRIPTOR_SLOTS (63)
#else
#   define VLR_MAX_NUM_SPECTRUM_NODE_DESCRIPTOR_SLOTS (3)
#endif
            uint32_t data[VLR_MAX_NUM_SPECTRUM_NODE_DESCRIPTOR_SLOTS];

            template <typename T>
            T* getData() const {
                static_assert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };



        struct BSDFProcedureSet {
            int32_t progGetBaseColor;
            int32_t progMatches;
            int32_t progSampleInternal;
            int32_t progEvaluateInternal;
            int32_t progEvaluatePDFInternal;
            int32_t progWeightInternal;
        };

        struct EDFProcedureSet {
            int32_t progEvaluateEmittanceInternal;
            int32_t progEvaluateInternal;
        };



        struct SurfaceMaterialDescriptor {
            int32_t progSetupBSDF;
            uint32_t bsdfProcedureSetIndex;
            int32_t progSetupEDF;
            uint32_t edfProcedureSetIndex;
#define VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS (28)
            uint32_t data[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS];

            template <typename T>
            T* getData() const {
                static_assert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };



        struct Triangle {
            uint32_t index0, index1, index2;
        };

        struct SurfaceLightDescriptor {
            union Body {
                struct {
                    rtBufferId<Vertex> vertexBuffer;
                    rtBufferId<Triangle> triangleBuffer;
                    uint32_t materialIndex;
                    DiscreteDistribution1D primDistribution;
                    StaticTransform transform;
                } asMeshLight;
                struct {
                    uint32_t materialIndex;
                    float rotationPhi;
                    RegularConstantContinuousDistribution2D importanceMap;
                } asEnvironmentLight;

                RT_FUNCTION Body() {}
                RT_FUNCTION ~Body() {}
            } body;
            float importance;
            int32_t sampleFunc;
        };



        struct PerspectiveCamera {
            Point3D position;
            Quaternion orientation;

            float sensitivity;
            float aspect;
            float fovY;
            float lensRadius;
            float imgPlaneDistance;
            float objPlaneDistance;

            float opWidth;
            float opHeight;
            float imgPlaneArea;

            RT_FUNCTION PerspectiveCamera() {}

            void setImagePlaneArea() {
                opHeight = 2.0f * objPlaneDistance * std::tan(fovY * 0.5f);
                opWidth = opHeight * aspect;
                imgPlaneDistance = 1.0f;
                imgPlaneArea = 1;// opWidth * opHeight * std::pow(imgPlaneDistance / objPlaneDistance, 2);
            }
        };



        struct EquirectangularCamera {
            Point3D position;
            Quaternion orientation;

            float sensitivity;

            float phiAngle;
            float thetaAngle;

            RT_FUNCTION EquirectangularCamera() {}
        };



        struct RayType {
            enum Value {
                Primary = 0,
                Scattered,
                Shadow,
                DebugPrimary,
                NumTypes
            } value;

            RT_FUNCTION constexpr RayType(Value v = Primary) : value(v) {}
        };

        struct DebugRenderingAttribute {
            enum Value {
                GeometricNormal = 0,
                ShadingTangent,
                ShadingBitangent,
                ShadingNormal,
                TC0Direction,
                TextureCoordinates,
                GeometricVsShadingNormal,
                ShadingFrameLengths,
                ShadingFrameOrthogonality,
            } value;

            RT_FUNCTION constexpr DebugRenderingAttribute(Value v = GeometricNormal) : value(v) {}

            RT_FUNCTION operator int32_t() const {
                return value;
            }
        };



        struct TangentType {
            enum Value {
                TC0Direction = 0,
                RadialX,
                RadialY,
                RadialZ
            } value;

            RT_FUNCTION constexpr TangentType(Value v = TC0Direction) : value(v) {}
            RT_FUNCTION bool operator==(const TangentType &r) const {
                return value == r.value;
            }
        };



        // ----------------------------------------------------------------
        // Shader Nodes

        struct GeometryShaderNode {

        };

        struct FloatShaderNode {
            ShaderNodeSocketID node0;
            float imm0;
        };

        struct Float2ShaderNode {
            ShaderNodeSocketID node0;
            ShaderNodeSocketID node1;
            float imm0;
            float imm1;
        };

        struct Float3ShaderNode {
            ShaderNodeSocketID node0;
            ShaderNodeSocketID node1;
            ShaderNodeSocketID node2;
            float imm0;
            float imm1;
            float imm2;
        };

        struct Float4ShaderNode {
            ShaderNodeSocketID node0;
            ShaderNodeSocketID node1;
            ShaderNodeSocketID node2;
            ShaderNodeSocketID node3;
            float imm0;
            float imm1;
            float imm2;
            float imm3;
        };

        struct ScaleAndOffsetFloatShaderNode {
            ShaderNodeSocketID nodeValue;
            ShaderNodeSocketID nodeScale;
            ShaderNodeSocketID nodeOffset;
            float immScale;
            float immOffset;
        };

#if defined(VLR_USE_SPECTRAL_RENDERING)
        struct TripletSpectrumShaderNode {
            UpsampledSpectrum value;
        };

        struct RegularSampledSpectrumShaderNode {
            float minLambda;
            float maxLambda;
            float values[VLR_MAX_NUM_SPECTRUM_NODE_DESCRIPTOR_SLOTS - 3];
            uint32_t numSamples;
        };

        struct IrregularSampledSpectrumShaderNode {
            float lambdas[(VLR_MAX_NUM_SPECTRUM_NODE_DESCRIPTOR_SLOTS - 1) / 2];
            float values[(VLR_MAX_NUM_SPECTRUM_NODE_DESCRIPTOR_SLOTS - 1) / 2];
            uint32_t numSamples;
        };
#else
        struct TripletSpectrumShaderNode {
            RGBSpectrum value;
        };

        struct RegularSampledSpectrumShaderNode {
            RGBSpectrum value;
        };

        struct IrregularSampledSpectrumShaderNode {
            RGBSpectrum value;
        };
#endif

        struct Vector3DToSpectrumShaderNode {
            ShaderNodeSocketID nodeVector3D;
            Vector3D immVector3D;
            VLRSpectrumType spectrumType;
            ColorSpace colorSpace;
        };

        struct ScaleAndOffsetUVTextureMap2DShaderNode {
            float offset[2];
            float scale[2];
        };

        struct Image2DTextureShaderNode {
            int32_t textureID;
            struct {
                unsigned int dataFormat : 5;
                unsigned int spectrumType : 3;
                unsigned int colorSpace : 3;
            };
            ShaderNodeSocketID nodeTexCoord;

            RT_FUNCTION DataFormat getDataFormat() const { return DataFormat((DataFormat::Value)dataFormat); }
            RT_FUNCTION VLRSpectrumType getSpectrumType() const { return (VLRSpectrumType)spectrumType; }
            RT_FUNCTION ColorSpace getColorSpace() const { return ColorSpace((ColorSpace::Value)colorSpace); }
        };
        static_assert(sizeof(Image2DTextureShaderNode) == 12, "Unexpected sizeof(Image2DTextureShaderNode).");

        struct EnvironmentTextureShaderNode {
            int32_t textureID;
            struct {
                unsigned int dataFormat : 5;
                unsigned int colorSpace : 3;
            };
            ShaderNodeSocketID nodeTexCoord;

            RT_FUNCTION DataFormat getDataFormat() const { return DataFormat((DataFormat::Value)dataFormat); }
            RT_FUNCTION ColorSpace getColorSpace() const { return ColorSpace((ColorSpace::Value)colorSpace); }
        };

        // END: Shader Nodes
        // ----------------------------------------------------------------



        // ----------------------------------------------------------------
        // Surface Materials

        struct MatteSurfaceMaterial {
            ShaderNodeSocketID nodeAlbedo;
            TripletSpectrum immAlbedo;
        };

        struct SpecularReflectionSurfaceMaterial {
            ShaderNodeSocketID nodeCoeffR;
            ShaderNodeSocketID nodeEta;
            ShaderNodeSocketID node_k;
            TripletSpectrum immCoeffR;
            TripletSpectrum immEta;
            TripletSpectrum imm_k;
        };

        struct SpecularScatteringSurfaceMaterial {
            ShaderNodeSocketID nodeCoeff;
            ShaderNodeSocketID nodeEtaExt;
            ShaderNodeSocketID nodeEtaInt;
            TripletSpectrum immCoeff;
            TripletSpectrum immEtaExt;
            TripletSpectrum immEtaInt;
        };

        struct MicrofacetReflectionSurfaceMaterial {
            ShaderNodeSocketID nodeEta;
            ShaderNodeSocketID node_k;
            ShaderNodeSocketID nodeRoughnessAnisotropyRotation;
            TripletSpectrum immEta;
            TripletSpectrum imm_k;
            float immRoughness;
            float immAnisotropy;
            float immRotation;
        };

        struct MicrofacetScatteringSurfaceMaterial {
            ShaderNodeSocketID nodeCoeff;
            ShaderNodeSocketID nodeEtaExt;
            ShaderNodeSocketID nodeEtaInt;
            ShaderNodeSocketID nodeRoughnessAnisotropyRotation;
            TripletSpectrum immCoeff;
            TripletSpectrum immEtaExt;
            TripletSpectrum immEtaInt;
            float immRoughness;
            float immAnisotropy;
            float immRotation;
        };

        struct LambertianScatteringSurfaceMaterial {
            ShaderNodeSocketID nodeCoeff;
            ShaderNodeSocketID nodeF0;
            TripletSpectrum immCoeff;
            float immF0;
        };

        struct UE4SurfaceMaterial {
            ShaderNodeSocketID nodeBaseColor;
            ShaderNodeSocketID nodeOcclusionRoughnessMetallic;
            TripletSpectrum immBaseColor;
            float immOcclusion;
            float immRoughness;
            float immMetallic;
        };

        struct OldStyleSurfaceMaterial {
            ShaderNodeSocketID nodeDiffuseColor;
            ShaderNodeSocketID nodeSpecularColor;
            ShaderNodeSocketID nodeGlossiness;
            TripletSpectrum immDiffuseColor;
            TripletSpectrum immSpecularColor;
            float immGlossiness;
        };

        struct DiffuseEmitterSurfaceMaterial {
            ShaderNodeSocketID nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
        };

        struct MultiSurfaceMaterial {
            uint32_t subMatIndices[4];
            uint32_t numSubMaterials;
        };

        struct EnvironmentEmitterSurfaceMaterial {
            ShaderNodeSocketID nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
        };

        // END: Surface Materials
        // ----------------------------------------------------------------
    }
}

#if defined(VLR_Device)
#include "spectrum_types.cpp"
#endif
