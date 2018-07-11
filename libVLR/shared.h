#pragma once

#include "basic_types_internal.h"

namespace VLR {
    namespace Shared {
        template <typename RealType>
        class DiscreteDistribution1DTemplate {
            rtBufferId<RealType, 1> m_PMF;
            rtBufferId<RealType, 1> m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            ~DiscreteDistribution1DTemplate() {}

            RT_FUNCTION uint32_t sample(RealType u, RealType* prob) const {
                VLRAssert(u >= 0 && u < 1, "\"u\" must be in range [0, 1).");
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1)
                    if (idx - d > 0 && m_CDF[idx - d] >= u)
                        idx -= d;
                --idx;
                *prob = m_PMF[idx];
                return idx;
            }
            RT_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
                VLRAssert(u >= 0 && u < 1, "\"u\" must be in range [0, 1).");
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1)
                    if (idx - d > 0 && m_CDF[idx - d] >= u)
                        idx -= d;
                --idx;
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
            ~RegularConstantContinuousDistribution1DTemplate() {}

            RT_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
                VLRAssert(u < 1, "\"u\" must be in range [0, 1).");
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1)
                    if (idx - d > 0 && m_CDF[idx - d] >= u)
                        idx -= d;
                --idx;
                *probDensity = m_PDF[idx];
                RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return (idx + t) / m_numValues;
            }
            RT_FUNCTION RealType evaluatePDF(RealType smp) const {
                VLRAssert(smp >= 0 && smp < 1.0, "\"smp\" is out of range [0, 1)");
                return m_PDF[(int32_t)(smp * m_numValues)];
            }
            RT_FUNCTION RealType integral() const { return m_integral; }

            RT_FUNCTION uint32_t numValues() const { return m_numValues; }
        };

        using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution2DTemplate {
            rtBufferId<RegularConstantContinuousDistribution1DTemplate<RealType>, 1> m_1DDists;
            uint32_t m_num1DDists;
            RealType m_integral;
            RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

        public:
            ~RegularConstantContinuousDistribution2DTemplate() {}

            void sample(RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
                VLRAssert(u0 >= 0 && u0 < 1, "\"u0\" must be in range [0, 1).: %g", u0);
                VLRAssert(u1 >= 0 && u1 < 1, "\"u1\" must be in range [0, 1).: %g", u1);
                RealType topPDF;
                *d1 = m_top1DDist.sample(u1, &topPDF);
                uint32_t idx1D = std::min(uint32_t(m_num1DDists * *d1), m_num1DDists - 1);
                *d0 = m_1DDists[idx1D].sample(u0, probDensity);
                *probDensity *= topPDF;
            }
            RealType evaluatePDF(RealType d0, RealType d1) const {
                VLRAssert(d0 >= 0 && d0 < 1.0, "\"d0\" is out of range [0, 1)");
                VLRAssert(d1 >= 0 && d1 < 1.0, "\"d1\" is out of range [0, 1)");
                uint32_t idx1D = std::min(uint32_t(m_num1DDists * d1), m_num1DDists - 1);
                return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
            }
        };

        using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



        struct Vertex {
            Point3D position;
            Normal3D normal;
            Vector3D tangent;
            TexCoord2D texCoord;
        };

        struct Triangle {
            uint32_t index0, index1, index2;
            uint32_t matIndex;
        };
        
        struct SurfaceLightDescriptor {
            union Body {
                struct MeshLight {
                    rtBufferId<Vertex> vertexBuffer;
                    rtBufferId<Triangle> triangleBuffer;
                    DiscreteDistribution1D primDistribution;
                    Point3D position;
                    Quaternion orientation;
                } asMeshLight;
                struct InfiniteSphericalLight {
                    RegularConstantContinuousDistribution2D distribution;
                } asInfiniteSphericalLight;

                RT_FUNCTION Body() {}
                RT_FUNCTION ~Body() {}
            } body;
            int32_t sampleFunc;
            int32_t evaluateFunc;
        };



        struct ThinLensCamera {
            Point3D position;
            Quaternion orientation;

            float aspect;
            float fovY;
            float lensRadius;
            float imgPlaneDistance;
            float objPlaneDistance;

            float opWidth;
            float opHeight;
            float imgPlaneArea;
        };



        struct RayType {
            enum Value {
                Primary = 0,
                Scattered,
                Shadow,
                NumTypes
            } value;

            RT_FUNCTION constexpr RayType(Value v = Primary) : value(v) { }
        };
    }
}
