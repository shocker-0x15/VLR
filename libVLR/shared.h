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



        struct BSDFProcedureSet {
            int32_t progGetBaseColor;
            int32_t progBSDFmatches;
            int32_t progSampleBSDFInternal;
            int32_t progEvaluateBSDFInternal;
            int32_t progEvaluateBSDF_PDFInternal;
            int32_t progWeightInternal;
        };

        struct EDFProcedureSet {
            int32_t progEvaluateEmittanceInternal;
            int32_t progEvaluateEDFInternal;
        };



        struct TextureMapDescriptor {
            int32_t progTextureMap;
#define VLR_MAX_NUM_TEXTURE_MAPPING_DESCRIPTOR_SLOTS (32)
            uint32_t data[VLR_MAX_NUM_TEXTURE_MAPPING_DESCRIPTOR_SLOTS];
        };



        struct SurfaceMaterialDescriptor {
#define VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS (32)
            union {
                int32_t i1[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS];
                uint32_t ui1[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS];
                float f1[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS];

                optix::int2 i2[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS >> 1];
                optix::float2 f2[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS >> 1];

                optix::float4 f4[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS >> 2];
            };
        };



        struct Vertex {
            Point3D position;
            Normal3D normal;
            Vector3D tangent;
            TexCoord2D texCoord;
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
            PerspectiveCamera(float _sensitivity, float _aspect, float _fovY, float _lensRadius, float _imgPDist, float _objPDist) :
                sensitivity(_sensitivity), aspect(_aspect), fovY(_fovY), lensRadius(_lensRadius), imgPlaneDistance(_imgPDist), objPlaneDistance(_objPDist) {
                opHeight = 2.0f * objPlaneDistance * std::tan(fovY * 0.5f);
                opWidth = opHeight * aspect;
                imgPlaneArea = 1;// opWidth * opHeight * std::pow(imgPlaneDistance / objPlaneDistance, 2);
            }

            void setImagePlaneArea() {
                opHeight = 2.0f * objPlaneDistance * std::tan(fovY * 0.5f);
                opWidth = opHeight * aspect;
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
            EquirectangularCamera(float _sensitivity, float _phiAngle, float _thetaAngle) :
                sensitivity(_sensitivity), phiAngle(_phiAngle), thetaAngle(_thetaAngle) {
            }
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



        struct OffsetAndScaleUVTextureMap2D {
            float offset[2];
            float scale[2];
        };



        struct SurfaceMaterialHead {
            int32_t progSetupBSDF;
            uint32_t bsdfProcedureSetIndex;
            int32_t progSetupEDF;
            uint32_t edfProcedureSetIndex;
        };

        struct MatteSurfaceMaterial {
            int32_t texAlbedoRoughness;
            uint32_t texMap;
        };

        struct SpecularReflectionSurfaceMaterial {
            int32_t texCoeffR;
            int32_t texEta;
            int32_t tex_k;
            uint32_t texMap;
        };

        struct SpecularScatteringSurfaceMaterial {
            int32_t texCoeff;
            int32_t texEtaExt;
            int32_t texEtaInt;
            uint32_t texMap;
        };

        struct MicrofacetReflectionSurfaceMaterial {
            int32_t texEta;
            int32_t tex_k;
            int32_t texRoughness;
            uint32_t texMap;
        };

        struct MicrofacetScatteringSurfaceMaterial {
            int32_t texCoeff;
            int32_t texEtaExt;
            int32_t texEtaInt;
            int32_t texRoughness;
            uint32_t texMap;
        };

        struct UE4SurfaceMaterial {
            int32_t texBaseColor;
            int32_t texOcclusionRoughnessMetallic;
            uint32_t texMap;
        };

        struct DiffuseEmitterSurfaceMaterial {
            int32_t texEmittance;
            uint32_t texMap;
        };

        struct MultiSurfaceMaterial {
            struct { // offsets in DWs
                unsigned int matOffset0 : 6;
                unsigned int matOffset1 : 6;
                unsigned int matOffset2 : 6;
                unsigned int matOffset3 : 6;
                unsigned int numMaterials : 8;
            };
        };

        struct EnvironmentEmitterSurfaceMaterial {
            int32_t texEmittance;
        };
    }
}
