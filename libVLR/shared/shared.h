﻿#pragma once

#include "basic_types_internal.h"
#include "rgb_spectrum_types.h"
#include "spectrum_types.h"
#if defined(VLR_Host)
#include "half/half.hpp"
#endif
#include "random_distributions.h"

namespace vlr {
#if defined(VLR_Host)
    using half_float::half;
#else
    struct half {
        uint16_t raw;

        operator float() const {
            uint32_t bits = static_cast<uint32_t>(raw & 0x8000) << 16;
            uint32_t abs = raw & 0x7FFF;
            if (abs) {
                // JP: halfの指数部が   無限大 or NaN       を表す(11111)       場合: floatビット: (* 11100000 00000000000000000000000)
                //                    正規化数 or 非正規化数を表す(00000-11110) 場合:              (* 01110000 00000000000000000000000)
                bits |= 0x38000000 << static_cast<uint32_t>(abs >= 0x7C00);
                // JP: halfの指数部が非正規化数を表す(00000) 場合: 0x0001-0x03FF (* 00000 **********)
                //     正規化数になるまでhalfをビットシフト、floatの指数部を1ずつ減算。
                for (; abs < 0x400; abs <<= 1, bits -= 0x800000);
                // JP: halfの指数部が 無限大 or NaN を表す場合 0x7C00-0x7FFF (0       11111 **********): (0          00011111 **********0000000000000) を加算 => floatの指数ビットは0xFFになる。
                //                    正規化数      を表す場合 0x0400-0x7BFF (0 00001-11110 **********): (0 00000001-00011110 **********0000000000000) を加算 => floatの指数ビットは0x71-0x8Eになる。
                bits += static_cast<uint32_t>(abs) << 13;
            }
            return *reinterpret_cast<float*>(&bits);
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

    CUDA_DEVICE_FUNCTION CUDA_INLINE TripletSpectrum createTripletSpectrum(SpectrumType spectrumType, ColorSpace colorSpace, float e0, float e1, float e2) {
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(spectrumType, colorSpace, e0, e1, e2);
#else
        float XYZ[3];

        switch (colorSpace) {
        case ColorSpace::Rec709_D65_sRGBGamma: {
            e0 = sRGB_degamma(e0);
            e1 = sRGB_degamma(e1);
            e2 = sRGB_degamma(e2);
            // pass to Rec709 (D65)
            [[fallthrough]];
        }
        case ColorSpace::Rec709_D65: {
            float RGB[3] = { e0, e1, e2 };
            switch (spectrumType) {
            case SpectrumType::Reflectance:
            case SpectrumType::IndexOfRefraction:
            case SpectrumType::NA:
                transformTristimulus(mat_Rec709_E_to_XYZ, RGB, XYZ);
                break;
            case SpectrumType::LightSource:
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

        // JP: ここの変換で内部的に使っている色空間がRec.709の場合は、
        //     Rec.709のRGB入力に対しては入力値に戻しているだけになるが、ここでRec.2020などを使うこともできる。
        float RGB[3];
        transformToRenderingRGB(spectrumType, XYZ, RGB);
        return RGBSpectrum(RGB[0], RGB[1], RGB[2]);
#endif
    }



    enum class DataFormat {
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
        NumFormats
    };



    enum class BumpType {
        NormalMap_DirectX = 0,
        NormalMap_OpenGL,
        HeightMap
    };



    enum class ShaderNodePlugType {
        float1 = 0,
        float2,
        float3,
        float4,
        Point3D,
        Vector3D,
        Normal3D,
        Spectrum,
        Alpha,
        TextureCoordinates,
        NumTypes
    };



    enum class TangentType {
        TC0Direction = 0,
        RadialX,
        RadialY,
        RadialZ,
    };



    namespace shared {
        template <typename RealType>
        class DiscreteDistribution1DTemplate {
            const RealType* m_PMF;
            const RealType* m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            DiscreteDistribution1DTemplate(const RealType* PMF, const RealType* CDF, RealType integral, uint32_t numValues) :
                m_PMF(PMF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE DiscreteDistribution1DTemplate() {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t sample(RealType u, RealType* prob) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = 0;
                for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                    if (idx + d >= m_numValues)
                        continue;
                    if (m_CDF[idx + d] <= u)
                        idx += d;
                }
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *prob = m_PMF[idx];
                return idx;
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = 0;
                for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                    if (idx + d >= m_numValues)
                        continue;
                    if (m_CDF[idx + d] <= u)
                        idx += d;
                }
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *prob = m_PMF[idx];
                *remapped = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return idx;
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE RealType evaluatePMF(uint32_t idx) const {
                VLRAssert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
                return m_PMF[idx];
            }

            CUDA_DEVICE_FUNCTION CUDA_INLINE RealType integral() const { return m_integral; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t numValues() const { return m_numValues; }
        };

        using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution1DTemplate {
            const RealType* m_PDF;
            const RealType* m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            RegularConstantContinuousDistribution1DTemplate(const RealType* PDF, const RealType* CDF, RealType integral, uint32_t numValues) :
                m_PDF(PDF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE RegularConstantContinuousDistribution1DTemplate() {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE RealType sample(RealType u, RealType* probDensity) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = 0;
                for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
                    if (idx + d >= m_numValues)
                        continue;
                    if (m_CDF[idx + d] <= u)
                        idx += d;
                }
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *probDensity = m_PDF[idx];
                RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return (idx + t) / m_numValues;
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE RealType evaluatePDF(RealType smp) const {
                VLRAssert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
                int32_t idx = ::vlr::min<int32_t>(m_numValues - 1, smp * m_numValues);
                return m_PDF[idx];
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE RealType integral() const { return m_integral; }

            CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t numValues() const { return m_numValues; }
        };

        using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution2DTemplate {
            const RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
            RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

        public:
            RegularConstantContinuousDistribution2DTemplate(const RegularConstantContinuousDistribution1DTemplate<RealType>* _1DDists,
                                                            const RegularConstantContinuousDistribution1DTemplate<RealType> &top1DDist) :
                m_1DDists(_1DDists), m_top1DDist(top1DDist) {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE RegularConstantContinuousDistribution2DTemplate() {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE void sample(RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
                RealType topPDF;
                *d1 = m_top1DDist.sample(u1, &topPDF);
                uint32_t idx1D = ::vlr::min(uint32_t(m_top1DDist.numValues() * *d1), m_top1DDist.numValues() - 1);
                *d0 = m_1DDists[idx1D].sample(u0, probDensity);
                *probDensity *= topPDF;
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE RealType evaluatePDF(RealType d0, RealType d1) const {
                uint32_t idx1D = ::vlr::min(uint32_t(m_top1DDist.numValues() * d1), m_top1DDist.numValues() - 1);
                return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
            }
        };

        using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



        class StaticTransform {
            Matrix4x4 m_matrix;
            Matrix4x4 m_invMatrix;

        public:
            CUDA_DEVICE_FUNCTION CUDA_INLINE StaticTransform() {}
            CUDA_DEVICE_FUNCTION CUDA_INLINE StaticTransform(const Matrix4x4 &m) : m_matrix(m), m_invMatrix(invert(m)) {}
            CUDA_DEVICE_FUNCTION CUDA_INLINE StaticTransform(const Matrix4x4 &m, const Matrix4x4 &mInv) : m_matrix(m), m_invMatrix(mInv) {}

            CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D operator*(const Vector3D &v) const { return m_matrix * v; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE Vector4D operator*(const Vector4D &v) const { return m_matrix * v; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D operator*(const Point3D &p) const { return m_matrix * p; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D operator*(const Normal3D &n) const {
                // The length of the normal is changed if the transform has scaling, so it requires normalization.
                return Normal3D(m_invMatrix.m00 * n.x + m_invMatrix.m10 * n.y + m_invMatrix.m20 * n.z,
                                m_invMatrix.m01 * n.x + m_invMatrix.m11 * n.y + m_invMatrix.m21 * n.z,
                                m_invMatrix.m02 * n.x + m_invMatrix.m12 * n.y + m_invMatrix.m22 * n.z);
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE BoundingBox3D operator*(const BoundingBox3D &b) const { return m_matrix * b; }


            CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D mulInv(const Vector3D& v) const { return m_invMatrix * v; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE Vector4D mulInv(const Vector4D& v) const { return m_invMatrix * v; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE Point3D mulInv(const Point3D& p) const { return m_invMatrix * p; }
            CUDA_DEVICE_FUNCTION CUDA_INLINE Normal3D mulInv(const Normal3D& n) const {
                // The length of the normal is changed if the transform has scaling, so it requires normalization.
                return Normal3D(m_matrix.m00 * n.x + m_matrix.m10 * n.y + m_matrix.m20 * n.z,
                                m_matrix.m01 * n.x + m_matrix.m11 * n.y + m_matrix.m21 * n.z,
                                m_matrix.m02 * n.x + m_matrix.m12 * n.y + m_matrix.m22 * n.z);
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE BoundingBox3D mulInv(const BoundingBox3D &b) const { return m_invMatrix * b; }
        };



        struct NodeProcedureSet {
            int32_t progs[nextPowerOf2Const(static_cast<uint32_t>(ShaderNodePlugType::NumTypes))];
        };



        union ShaderNodePlug {
            struct {
                unsigned int nodeType : 8;
                unsigned int plugType : 4;
                unsigned int nodeDescIndex : 18;
                unsigned int option : 2;
            };
            uint32_t asUInt;

            CUDA_DEVICE_FUNCTION CUDA_INLINE ShaderNodePlug() {}
            explicit constexpr ShaderNodePlug(uint32_t ui) : asUInt(ui) {}
            CUDA_DEVICE_FUNCTION CUDA_INLINE bool isValid() const { return asUInt != 0xFFFFFFFF; }

            static constexpr ShaderNodePlug Invalid() { return ShaderNodePlug(0xFFFFFFFF); }
        };
        static_assert(sizeof(ShaderNodePlug) == 4, "Unexpected Size");

        template <uint32_t Size>
        struct NodeDescriptor {
            uint32_t data[Size];

            template <typename T>
            CUDA_DEVICE_FUNCTION CUDA_INLINE T* getData() {
                VLRAssert(sizeof(T) <= sizeof(data), "Too big node data.");
                return reinterpret_cast<T*>(data);
            }
            template <typename T>
            CUDA_DEVICE_FUNCTION CUDA_INLINE const T* getData() const {
                VLRAssert(sizeof(T) <= sizeof(data), "Too big node data.");
                return reinterpret_cast<const T*>(data);
            }

            CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr uint32_t NumDWSlots() { return Size; }
        };

        using SmallNodeDescriptor = NodeDescriptor<4>;
        using MediumNodeDescriptor = NodeDescriptor<16>;
        using LargeNodeDescriptor = NodeDescriptor<64>;



        // ----------------------------------------------------------------
        // JP: シェーダーノードソケット間の暗黙的な型変換を定義する。
        // EN: Define implicit type conversion between shader node sockets.

        template <typename Type>
        struct NodeTypeInfo {
            template <typename SrcType>
            CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr bool ConversionIsDefinedFrom() {
                return false;
            }
            CUDA_DEVICE_FUNCTION CUDA_INLINE static constexpr bool ConversionIsDefinedFrom(ShaderNodePlugType plugType);
            template <typename SrcType>
            CUDA_DEVICE_FUNCTION CUDA_INLINE static Type convertFrom(const SrcType &) {
                return Type();
            }
        };

#define VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(DstType, SrcType) \
    template <> template <> \
    CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool NodeTypeInfo<DstType>::ConversionIsDefinedFrom<SrcType>() { return true; } \
    template <> template <> \
    CUDA_DEVICE_FUNCTION CUDA_INLINE DstType NodeTypeInfo<DstType>::convertFrom<SrcType>(const SrcType &srcValue)

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float, float) { return srcValue; }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float2, float2) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float2, float) { return make_float2(srcValue, srcValue); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float3, float3) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float3, float) { return make_float3(srcValue, srcValue, srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float3, Point3D) { return asOptiXType(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float3, Vector3D) { return asOptiXType(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float3, Normal3D) { return asOptiXType(srcValue); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float4, float4) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float4, float) { return make_float4(srcValue, srcValue, srcValue, srcValue); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, Point3D) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, float3) { return asPoint3D(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, Vector3D) { return Point3D(srcValue.x, srcValue.y, srcValue.z); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, Normal3D) { return Point3D(srcValue.x, srcValue.y, srcValue.z); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, Vector3D) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, float3) { return asVector3D(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, Point3D) { return Vector3D(srcValue.x, srcValue.y, srcValue.z); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, Normal3D) { return Vector3D(srcValue.x, srcValue.y, srcValue.z); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, Normal3D) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, float3) { return asNormal3D(srcValue).normalize(); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, Point3D) { return Normal3D(srcValue.x, srcValue.y, srcValue.z).normalize(); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, Vector3D) { return Normal3D(srcValue.x, srcValue.y, srcValue.z).normalize(); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(SampledSpectrum, SampledSpectrum) { return srcValue; }

#undef VLR_NODE_TYPE_INFO_DEFINE_CONVERSION

        template <typename Type>
        CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool NodeTypeInfo<Type>::ConversionIsDefinedFrom(ShaderNodePlugType plugType) {
            switch (plugType) {
            case ShaderNodePlugType::float1:
                return ConversionIsDefinedFrom<float>();
            case ShaderNodePlugType::float2:
                return ConversionIsDefinedFrom<float2>();
            case ShaderNodePlugType::float3:
                return ConversionIsDefinedFrom<float3>();
            case ShaderNodePlugType::float4:
                return ConversionIsDefinedFrom<float4>();
            case ShaderNodePlugType::Point3D:
                return ConversionIsDefinedFrom<Point3D>();
            case ShaderNodePlugType::Vector3D:
                return ConversionIsDefinedFrom<Vector3D>();
            case ShaderNodePlugType::Normal3D:
                return ConversionIsDefinedFrom<Normal3D>();
            case ShaderNodePlugType::Spectrum:
                return ConversionIsDefinedFrom<SampledSpectrum>();
            case ShaderNodePlugType::Alpha:
                return ConversionIsDefinedFrom<float>();
            case ShaderNodePlugType::TextureCoordinates:
                return ConversionIsDefinedFrom<Point3D>();
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
            return false;
        }

        // END: Define implicit type conversion between shader node sockets.
        // ----------------------------------------------------------------



        struct BSDFProcedureSet {
            int32_t progGetBaseColor;
            int32_t progMatches;
            int32_t progSampleInternal;
            int32_t progSampleWithRevInternal;
            int32_t progEvaluateInternal;
            int32_t progEvaluateWithRevInternal;
            int32_t progEvaluatePDFInternal;
            int32_t progEvaluatePDFWithRevInternal;
            int32_t progWeightInternal;
        };

        struct EDFProcedureSet {
            int32_t progMatches;
            int32_t progSampleInternal;
            int32_t progEvaluateEmittanceInternal;
            int32_t progEvaluateInternal;
            int32_t progEvaluatePDFInternal;
            int32_t progWeightInternal;

            int32_t progGetBaseColorAsBSDF;
            int32_t progMatchesAsBSDF;
            int32_t progSampleInternalAsBSDF;
            int32_t progSampleWithRevInternalAsBSDF;
            int32_t progEvaluateInternalAsBSDF;
            int32_t progEvaluateWithRevInternalAsBSDF;
            int32_t progEvaluatePDFInternalAsBSDF;
            int32_t progEvaluatePDFWithRevInternalAsBSDF;
            int32_t progWeightInternalAsBSDF;
        };

        struct IDFProcedureSet {
            int32_t progSampleInternal;
            int32_t progEvaluateSpatialImportanceInternal;
            int32_t progEvaluateDirectionalImportanceInternal;
            int32_t progEvaluatePDFInternal;
            int32_t progBackProjectDirectionInternal;
        };



        struct SurfaceMaterialDescriptor {
            int32_t progSetupBSDF;
            uint32_t bsdfProcedureSetIndex;
            int32_t progSetupEDF;
            uint32_t edfProcedureSetIndex;
            uint32_t data[28];

            template <typename T>
            T* getData() {
                static_assert(sizeof(T) <= sizeof(data), "Too big node data.");
                return reinterpret_cast<T*>(data);
            }
            template <typename T>
            const T* getData() const {
                static_assert(sizeof(T) <= sizeof(data), "Too big node data.");
                return reinterpret_cast<const T*>(data);
            }
        };



        struct Triangle {
            uint32_t index0, index1, index2;
        };



        struct CameraDescriptor {
            int32_t progSetupIDF;
            uint32_t idfProcedureSetIndex;
            uint32_t data[30];
        };



        struct PTRayType {
            enum Value {
                Closest = 0,
                Shadow,
                NumTypes
            } value;

            CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr PTRayType(Value v = Closest) : value(v) {}
        };

        struct LTRayType {
            enum Value {
                Closest = 0,
                Shadow,
                NumTypes
            } value;

            CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr LTRayType(Value v = Closest) : value(v) {}
        };

        struct LVCBPTRayType {
            enum Value {
                LightPath = 0,
                EyePath,
                Connection,
                NumTypes
            } value;

            CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr LVCBPTRayType(Value v = LightPath) : value(v) {}
        };

        struct AuxBufGenRayType {
            enum Value {
                Primary,
                NumTypes
            } value;

            CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr AuxBufGenRayType(Value v = Primary) : value(v) {}
        };

        struct DebugRayType {
            enum Value {
                Primary,
                NumTypes
            } value;

            CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DebugRayType(Value v = Primary) : value(v) {}
        };

        static constexpr uint32_t MaxNumRayTypes = 3;



        enum class DebugRenderingAttribute {
            BaseColor = 0,
            GeometricNormal,
            ShadingTangent,
            ShadingBitangent,
            ShadingNormal,
            TextureCoordinates,
            ShadingNormalViewCos,
            GeometricVsShadingNormal,
            ShadingFrameLengths,
            ShadingFrameOrthogonality,
            DenoiserAlbedo,
            DenoiserNormal,
            NumAttributes
        };



        // ----------------------------------------------------------------
        // Shader Nodes

        struct GeometryShaderNode {
        };

        struct TangentShaderNode {
            TangentType tangentType;
        };

        struct FloatShaderNode {
            ShaderNodePlug node0;
            float imm0;
        };

        struct Float2ShaderNode {
            ShaderNodePlug node0;
            ShaderNodePlug node1;
            float imm0;
            float imm1;
        };

        struct Float3ShaderNode {
            ShaderNodePlug node0;
            ShaderNodePlug node1;
            ShaderNodePlug node2;
            float imm0;
            float imm1;
            float imm2;
        };

        struct Float4ShaderNode {
            ShaderNodePlug node0;
            ShaderNodePlug node1;
            ShaderNodePlug node2;
            ShaderNodePlug node3;
            float imm0;
            float imm1;
            float imm2;
            float imm3;
        };

        struct ScaleAndOffsetFloatShaderNode {
            ShaderNodePlug nodeValue;
            ShaderNodePlug nodeScale;
            ShaderNodePlug nodeOffset;
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
            float values[LargeNodeDescriptor::NumDWSlots() - 3];
            uint32_t numSamples;
        };
        static_assert(sizeof(RegularSampledSpectrumShaderNode) == LargeNodeDescriptor::NumDWSlots() * 4,
                      "sizeof(RegularSampledSpectrumShaderNode) must match the size of LargeNodeDescriptor.");

        struct IrregularSampledSpectrumShaderNode {
            float lambdas[(LargeNodeDescriptor::NumDWSlots() - 1) / 2];
            float values[(LargeNodeDescriptor::NumDWSlots() - 1) / 2];
            uint32_t numSamples;
            uint32_t dummy;
        };
        static_assert(sizeof(IrregularSampledSpectrumShaderNode) == LargeNodeDescriptor::NumDWSlots() * 4,
                      "sizeof(IrregularSampledSpectrumShaderNode) must match the size of LargeNodeDescriptor.");
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

        struct Float3ToSpectrumShaderNode {
            ShaderNodePlug nodeFloat3;
            float immFloat3[3];
            SpectrumType spectrumType;
            ColorSpace colorSpace;
        };

        struct ScaleAndOffsetUVTextureMap2DShaderNode {
            float offset[2];
            float scale[2];
        };

        struct Image2DTextureShaderNode {
#define VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH (5)

            CUtexObject texture;
            struct {
                unsigned int dataFormat : 5;
                unsigned int spectrumType : 3;
                unsigned int colorSpace : 3;
                unsigned int bumpType : 2;
                unsigned int bumpCoeff : VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH;
            };
            ShaderNodePlug nodeTexCoord;
            struct {
                unsigned int width : 16;
                unsigned int height : 16;
            };

            CUDA_DEVICE_FUNCTION CUDA_INLINE DataFormat getDataFormat() const { return DataFormat(dataFormat); }
            CUDA_DEVICE_FUNCTION CUDA_INLINE SpectrumType getSpectrumType() const { return SpectrumType(spectrumType); }
            CUDA_DEVICE_FUNCTION CUDA_INLINE ColorSpace getColorSpace() const { return ColorSpace(colorSpace); }
            CUDA_DEVICE_FUNCTION CUDA_INLINE BumpType getBumpType() const { return BumpType(bumpType); }
            CUDA_DEVICE_FUNCTION CUDA_INLINE float getBumpCoeff() const {
                // map to (0, 2]
                return static_cast<float>(bumpCoeff + 1) / (1 << (VLR_IMAGE2D_TEXTURE_SHADER_NODE_BUMP_COEFF_BITWIDTH - 1));
            }
        };
        static_assert(sizeof(Image2DTextureShaderNode) == 24, "Unexpected sizeof(Image2DTextureShaderNode).");

        struct EnvironmentTextureShaderNode {
            CUtexObject texture;
            struct {
                unsigned int dataFormat : 5;
                unsigned int colorSpace : 3;
            };

            CUDA_DEVICE_FUNCTION CUDA_INLINE DataFormat getDataFormat() const { return DataFormat(dataFormat); }
            CUDA_DEVICE_FUNCTION CUDA_INLINE ColorSpace getColorSpace() const { return ColorSpace(colorSpace); }
        };

        // END: Shader Nodes
        // ----------------------------------------------------------------



        // ----------------------------------------------------------------
        // Surface Materials

        struct MatteSurfaceMaterial {
            ShaderNodePlug nodeAlbedo;
            TripletSpectrum immAlbedo;
        };

        struct SpecularReflectionSurfaceMaterial {
            ShaderNodePlug nodeCoeffR;
            ShaderNodePlug nodeEta;
            ShaderNodePlug node_k;
            TripletSpectrum immCoeffR;
            TripletSpectrum immEta;
            TripletSpectrum imm_k;
        };

        struct SpecularScatteringSurfaceMaterial {
            ShaderNodePlug nodeCoeff;
            ShaderNodePlug nodeEtaExt;
            ShaderNodePlug nodeEtaInt;
            TripletSpectrum immCoeff;
            TripletSpectrum immEtaExt;
            TripletSpectrum immEtaInt;
        };

        struct MicrofacetReflectionSurfaceMaterial {
            ShaderNodePlug nodeEta;
            ShaderNodePlug node_k;
            ShaderNodePlug nodeRoughnessAnisotropyRotation;
            TripletSpectrum immEta;
            TripletSpectrum imm_k;
            float immRoughness;
            float immAnisotropy;
            float immRotation;
        };

        struct MicrofacetScatteringSurfaceMaterial {
            ShaderNodePlug nodeCoeff;
            ShaderNodePlug nodeEtaExt;
            ShaderNodePlug nodeEtaInt;
            ShaderNodePlug nodeRoughnessAnisotropyRotation;
            TripletSpectrum immCoeff;
            TripletSpectrum immEtaExt;
            TripletSpectrum immEtaInt;
            float immRoughness;
            float immAnisotropy;
            float immRotation;
        };

        struct LambertianScatteringSurfaceMaterial {
            ShaderNodePlug nodeCoeff;
            ShaderNodePlug nodeF0;
            TripletSpectrum immCoeff;
            float immF0;
        };

        struct UE4SurfaceMaterial {
            ShaderNodePlug nodeBaseColor;
            ShaderNodePlug nodeOcclusionRoughnessMetallic;
            TripletSpectrum immBaseColor;
            float immOcclusion;
            float immRoughness;
            float immMetallic;
        };

        struct OldStyleSurfaceMaterial {
            ShaderNodePlug nodeDiffuseColor;
            ShaderNodePlug nodeSpecularColor;
            ShaderNodePlug nodeGlossiness;
            TripletSpectrum immDiffuseColor;
            TripletSpectrum immSpecularColor;
            float immGlossiness;
        };

        struct DiffuseEmitterSurfaceMaterial {
            ShaderNodePlug nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
        };

        struct DirectionalEmitterSurfaceMaterial {
            ShaderNodePlug nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
            ShaderNodePlug nodeDirection;
            Vector3D immDirection;
        };

        struct PointEmitterSurfaceMaterial {
            ShaderNodePlug nodeIntensity;
            TripletSpectrum immIntensity;
            float immScale;
        };

        struct MultiSurfaceMaterial {
            uint32_t subMatIndices[4];
            uint32_t numSubMaterials;
        };

        struct EnvironmentEmitterSurfaceMaterial {
            ShaderNodePlug nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
        };

        // END: Surface Materials
        // ----------------------------------------------------------------



        // ----------------------------------------------------------------
        // Cameras

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

            CUDA_DEVICE_FUNCTION CUDA_INLINE PerspectiveCamera() {}

            void setImagePlaneArea() {
                opHeight = 2.0f * objPlaneDistance * std::tan(fovY * 0.5f);
                opWidth = opHeight * aspect;
                imgPlaneDistance = 1.0f;
                imgPlaneArea = opWidth * opHeight * pow2(imgPlaneDistance / objPlaneDistance);
            }
        };

        struct EquirectangularCamera {
            Point3D position;
            Quaternion orientation;

            float sensitivity;

            float phiAngle;
            float thetaAngle;

            CUDA_DEVICE_FUNCTION CUDA_INLINE EquirectangularCamera() {}
        };

        // END: Camera
        // ----------------------------------------------------------------
    }
}
