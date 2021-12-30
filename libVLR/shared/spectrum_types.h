#pragma once

#include "spectrum_base.h"

namespace vlr {
    template <typename RealType, uint32_t NumSpectralSamples>
    struct WavelengthSamplesTemplate {
        RealType lambdas[NumSpectralSamples];
        struct {
            unsigned int _selectedLambdaIndex : 16;
            //bool _singleIsSelected : 1;
            unsigned int _singleIsSelected : 1;
        };

        CUDA_DEVICE_FUNCTION WavelengthSamplesTemplate() {}
        CUDA_DEVICE_FUNCTION WavelengthSamplesTemplate(const RealType* values) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                lambdas[i] = values[i];
            _selectedLambdaIndex = 0;
            _singleIsSelected = false;
        }
        CUDA_DEVICE_FUNCTION WavelengthSamplesTemplate(const WavelengthSamplesTemplate &wls) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                lambdas[i] = wls.lambdas[i];
            _selectedLambdaIndex = wls._selectedLambdaIndex;
            _singleIsSelected = wls._singleIsSelected;
        }

        CUDA_DEVICE_FUNCTION RealType &operator[](uint32_t index) {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return lambdas[index];
        }
        CUDA_DEVICE_FUNCTION RealType operator[](uint32_t index) const {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return lambdas[index];
        }

        CUDA_DEVICE_FUNCTION uint32_t selectedLambdaIndex() const {
            return _selectedLambdaIndex;
        }
        CUDA_DEVICE_FUNCTION void setSelectedLambdaIndex(uint32_t index) const {
            _selectedLambdaIndex = index;
        }
        CUDA_DEVICE_FUNCTION bool singleIsSelected() const {
            return _singleIsSelected;
        }
        CUDA_DEVICE_FUNCTION void setSingleIsSelected() {
            _singleIsSelected = true;
        }

        CUDA_DEVICE_FUNCTION RealType selectedWavelength() const {
            return lambdas[_selectedLambdaIndex];
        }

        CUDA_DEVICE_FUNCTION static WavelengthSamplesTemplate createWithEqualOffsets(RealType offset, RealType uLambda, RealType* PDF) {
            VLRAssert(offset >= 0 && offset < 1, "\"offset\" must be in range [0, 1).");
            VLRAssert(uLambda >= 0 && uLambda < 1, "\"uLambda\" must be in range [0, 1).");
            WavelengthSamplesTemplate wls;
            for (int i = 0; i < NumSpectralSamples; ++i)
                wls.lambdas[i] = WavelengthLowBound + (WavelengthHighBound - WavelengthLowBound) * (i + offset) / NumSpectralSamples;
            wls._selectedLambdaIndex = ::vlr::min<uint16_t>(NumSpectralSamples * uLambda, NumSpectralSamples - 1);
            wls._singleIsSelected = false;
            *PDF = NumSpectralSamples / (WavelengthHighBound - WavelengthLowBound);
            return wls;
        }

        CUDA_DEVICE_FUNCTION static constexpr uint32_t NumComponents() { return NumSpectralSamples; }
    };



    template <typename RealType, uint32_t NumSpectralSamples>
    struct SampledSpectrumTemplate {
        RealType values[NumSpectralSamples];

        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate() {}
        CUDA_DEVICE_FUNCTION constexpr SampledSpectrumTemplate(RealType v) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] = v;
        }
        CUDA_DEVICE_FUNCTION constexpr SampledSpectrumTemplate(const RealType* vals) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] = vals[i];
        }



        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator+() const { return *this; };
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator-() const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = -values[i];
            return SampledSpectrumTemplate(vals);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator+(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] + c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator-(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] - c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator*(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] * c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator/(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] / c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate safeDivide(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = c.values[i] > 0 ? values[i] / c.values[i] : 0.0f;
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator*(RealType s) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] * s;
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate operator/(RealType s) const {
            RealType vals[NumSpectralSamples];
            RealType r = 1 / s;
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] * r;
            return SampledSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION friend SampledSpectrumTemplate operator*(RealType s, const SampledSpectrumTemplate &c) {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = c.values[i] * s;
            return SampledSpectrumTemplate(vals);
        }

        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate &operator+=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] += c.values[i];
            return *this;
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate &operator-=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] -= c.values[i];
            return *this;
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate &operator*=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] *= c.values[i];
            return *this;
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate &operator/=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] /= c.values[i];
            return *this;
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate &operator*=(RealType s) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] *= s;
            return *this;
        }
        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate &operator/=(RealType s) {
            RealType r = 1 / s;
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] *= r;
            return *this;
        }

        CUDA_DEVICE_FUNCTION bool operator==(const SampledSpectrumTemplate &c) const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] != c.values[i])
                    return false;
            return true;
        }
        CUDA_DEVICE_FUNCTION bool operator!=(const SampledSpectrumTemplate &c) const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] != c.values[i])
                    return true;
            return false;
        }

        CUDA_DEVICE_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return values[index];
        }
        CUDA_DEVICE_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return values[index];
        }

        CUDA_DEVICE_FUNCTION RealType avgValue() const {
            RealType sumVal = values[0];
            for (int i = 1; i < NumSpectralSamples; ++i)
                sumVal += values[i];
            return sumVal / NumSpectralSamples;
        }
        CUDA_DEVICE_FUNCTION RealType maxValue() const {
            RealType maxVal = values[0];
            for (int i = 1; i < NumSpectralSamples; ++i)
                maxVal = std::fmax(values[i], maxVal);
            return maxVal;
        }
        CUDA_DEVICE_FUNCTION RealType minValue() const {
            RealType minVal = values[0];
            for (int i = 1; i < NumSpectralSamples; ++i)
                minVal = std::fmin(values[i], minVal);
            return minVal;
        }
        CUDA_DEVICE_FUNCTION bool hasNonZero() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] != 0)
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool hasNaN() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (::vlr::isnan(values[i]))
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool hasInf() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (::vlr::isinf(values[i]))
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool allFinite() const {
            return !hasNaN() && !hasInf();
        }
        CUDA_DEVICE_FUNCTION bool hasNegative() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] < 0)
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool allPositiveFinite() const {
            return allFinite() && !hasNegative();
        }

        // setting "primary" to 1.0 might introduce bias.
        CUDA_DEVICE_FUNCTION RealType importance(uint32_t selectedLambda) const {
            // I hope a compiler to optimize away this if statement...
            // What I want to do is just only member function specialization of a template class while reusing other function definitions.
            if (NumSpectralSamples > 1) {
                RealType sum = 0;
                for (int i = 0; i < NumSpectralSamples; ++i)
                    sum += values[i];
                const RealType primary = 0.9f;
                const RealType marginal = (1 - primary) / (NumSpectralSamples - 1);
                return sum * marginal + values[selectedLambda] * (primary - marginal);
            }
            else {
                return values[0];
            }
        }

        CUDA_DEVICE_FUNCTION static constexpr uint32_t NumComponents() { return NumSpectralSamples; }
        CUDA_DEVICE_FUNCTION static constexpr SampledSpectrumTemplate Zero() { return SampledSpectrumTemplate(0.0); }
        CUDA_DEVICE_FUNCTION static constexpr SampledSpectrumTemplate One() { return SampledSpectrumTemplate(1.0); }
        CUDA_DEVICE_FUNCTION static constexpr SampledSpectrumTemplate Inf() { return SampledSpectrumTemplate(VLR_INFINITY); }
        CUDA_DEVICE_FUNCTION static constexpr SampledSpectrumTemplate NaN() { return SampledSpectrumTemplate(VLR_NAN); }
    };

    template <typename RealType, uint32_t NumSpectralSamples>
    CUDA_DEVICE_FUNCTION constexpr SampledSpectrumTemplate<RealType, NumSpectralSamples> min(const SampledSpectrumTemplate<RealType, NumSpectralSamples> &value, RealType minValue) {
        SampledSpectrumTemplate<RealType, NumSpectralSamples> ret;
        for (int i = 0; i < NumSpectralSamples; ++i)
            ret[i] = std::fmin(value[0], minValue);
        return ret;
    }

    template <typename RealType, uint32_t NumSpectralSamples>
    CUDA_DEVICE_FUNCTION constexpr SampledSpectrumTemplate<RealType, NumSpectralSamples> max(const SampledSpectrumTemplate<RealType, NumSpectralSamples> &value, RealType maxValue) {
        SampledSpectrumTemplate<RealType, NumSpectralSamples> ret;
        for (int i = 0; i < NumSpectralSamples; ++i)
            ret[i] = std::fmax(value[0], maxValue);
        return ret;
    }

    template <typename RealType, uint32_t NumSpectralSamples>
    CUDA_DEVICE_FUNCTION constexpr SampledSpectrumTemplate<RealType, NumSpectralSamples> lerp(const SampledSpectrumTemplate<RealType, NumSpectralSamples> &v0, const SampledSpectrumTemplate<RealType, NumSpectralSamples> &v1, RealType t) {
        return (1 - t) * v0 + t * v1;
    }



    // JP: これは理想的には不要である。
    //     環境テクスチャーをuvs16Fx3(uvsA16Fx4)フォーマットに変換して保持する際に、典型的な値の場合、変換後のsの値が容易にhalf floatの限界に達するため、
    //     適当なスケール値をかけて小さな値にする。
    // EN: This is not ideally needed.
    //     When converting an environment texture into uvs16Fx3 (uvsA16Fx4) format and holding it, 
    //     a resulting s value from a typical value easily reaches the limit of half float therefore make it small by multiplying an appropriate scaling value.  
#define UPSAMPLED_CONTINOUS_SPECTRUM_SCALE_FACTOR (0.009355121400914532) // corresponds to cancel dividing by EqualEnergyReflectance.

    template <typename RealType, uint32_t NumSpectralSamples>
    class UpsampledSpectrumTemplate {
#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        uint32_t m_adjIndices;
        uint16_t m_s, m_t;
        RealType m_scale;

        CUDA_DEVICE_FUNCTION void computeAdjacents(RealType u, RealType v);
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
    public:
        struct PolynomialCoefficients {
            float c0, c1, c2;

            CUDA_DEVICE_FUNCTION friend PolynomialCoefficients operator*(RealType s, const PolynomialCoefficients &v) {
                return PolynomialCoefficients{ s * v.c0, s * v.c1, s * v.c2 };
            }
            CUDA_DEVICE_FUNCTION PolynomialCoefficients operator+(const PolynomialCoefficients &v) const {
                return PolynomialCoefficients{ c0 + v.c0, c1 + v.c1, c2 + v.c2 };
            }
        };

    private:
        RealType m_c[3];

        CUDA_DEVICE_FUNCTION void interpolateCoefficients(RealType e0, RealType e1, RealType e2, const PolynomialCoefficients* table);
#endif

    public:
#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        CUDA_DEVICE_FUNCTION UpsampledSpectrumTemplate(RealType u, RealType v, RealType scale) {
            computeAdjacents(u, v);
            m_scale = scale;
        }
        CUDA_DEVICE_FUNCTION constexpr UpsampledSpectrumTemplate(
            SpectrumType spType, ColorSpace space, RealType e0, RealType e1, RealType e2);
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        CUDA_DEVICE_FUNCTION UpsampledSpectrumTemplate(SpectrumType spType, ColorSpace space, RealType e0, RealType e1, RealType e2);
#endif
        CUDA_DEVICE_FUNCTION UpsampledSpectrumTemplate() {}

        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate<RealType, NumSpectralSamples> evaluate(const WavelengthSamplesTemplate<RealType, NumSpectralSamples> &wls) const;

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        CUDA_DEVICE_FUNCTION static constexpr RealType MinWavelength() { return static_cast<RealType>(360.0); }
        CUDA_DEVICE_FUNCTION static constexpr RealType MaxWavelength() { return static_cast<RealType>(830.0); }
        CUDA_DEVICE_FUNCTION static constexpr uint32_t NumWavelengthSamples() { return 95; }
        CUDA_DEVICE_FUNCTION static constexpr uint32_t GridWidth() { return 12; }
        CUDA_DEVICE_FUNCTION static constexpr uint32_t GridHeight() { return 14; }

        // Grid cells. Laid out in row-major format.
        // num_points = 0 for cells without data points.
        struct spectrum_grid_cell_t {
            uint8_t inside;
            uint8_t num_points;
            uint8_t idx[6];
        };

        // Grid data points.
        struct spectrum_data_point_t {
            float xystar[2];
            float uv[2];
            float spectrum[95]; // X+Y+Z = 1
        };

#   if defined(VLR_Host)
        static const spectrum_grid_cell_t spectrum_grid[];
        static const spectrum_data_point_t spectrum_data_points[];

        static void initialize();
#   endif

        // This is 1 over the integral over either CMF.
        // Spectra can be mapped so that xyz=(1,1,1) is converted to constant 1 by
        // dividing by this value. This is important for valid reflectances.
        CUDA_DEVICE_FUNCTION static constexpr RealType EqualEnergyReflectance() {
            return static_cast<RealType>(0.009355121400914532);
        }

        CUDA_DEVICE_FUNCTION static constexpr void xy_to_uv(const RealType xy[2], RealType uv[2]) {
            uv[0] =
                static_cast<RealType>(16.730260708356887) * xy[0]
                + static_cast<RealType>(7.7801960340706) * xy[1]
                - static_cast<RealType>(2.170152247475828);
            uv[1] =
                static_cast<RealType>(-7.530081094743006) * xy[0]
                + static_cast<RealType>(16.192422314095225) * xy[1]
                + static_cast<RealType>(1.1125529268825947);
        }

        CUDA_DEVICE_FUNCTION static constexpr void uv_to_xy(const RealType uv[2], RealType xy[2]) {
            xy[0] =
                static_cast<RealType>(0.0491440520940413) * uv[0]
                - static_cast<RealType>(0.02361291916573777) * uv[1]
                + static_cast<RealType>(0.13292069743203658);
            xy[1] =
                static_cast<RealType>(0.022853819546830627) * uv[0]
                + static_cast<RealType>(0.05077639329371236) * uv[1]
                - static_cast<RealType>(0.006895157122499944);
        }
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        static constexpr uint32_t kTableResolution = 64;
#   if defined(VLR_Host)
        static float maxBrightnesses[kTableResolution];
        static PolynomialCoefficients coefficients_sRGB_D65[3 * pow3(kTableResolution)];
        static PolynomialCoefficients coefficients_sRGB_E[3 * pow3(kTableResolution)];

        static void initialize();
#   endif
#endif

        CUDA_DEVICE_FUNCTION void print() const {
#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
            vlrprintf("%08x, %.6f, %.6f, %g\n", m_adjIndices,
                      static_cast<float>(m_s) / (UINT16_MAX - 1),
                      static_cast<float>(m_t) / (UINT16_MAX - 1),
                      m_scale);
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
            vlrprintf("%g, %g, %g\n", m_c[0], m_c[1], m_c[2]);
#endif
        }
    };



    template <typename RealType, uint32_t NumSpectralSamples>
    class RegularSampledSpectrumTemplate {
        float m_minLambda;
        float m_maxLambda;
        const RealType* m_values;
        uint32_t m_numSamples;

    public:
        CUDA_DEVICE_FUNCTION RegularSampledSpectrumTemplate(RealType minLambda, RealType maxLambda, const RealType* values, uint32_t numSamples) :
            m_minLambda(minLambda), m_maxLambda(maxLambda), m_values(values), m_numSamples(numSamples) {}

        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate<RealType, NumSpectralSamples> evaluate(const WavelengthSamplesTemplate<RealType, NumSpectralSamples> &wls) const;

#if defined(VLR_Host)
        void toXYZ(RealType XYZ[3]) const;
#endif
    };



    template <typename RealType, uint32_t NumSpectralSamples>
    class IrregularSampledSpectrumTemplate {
        const RealType* m_lambdas;
        const RealType* m_values;
        uint32_t m_numSamples;

    public:
        CUDA_DEVICE_FUNCTION IrregularSampledSpectrumTemplate(const RealType* lambdas, const RealType* values, uint32_t numSamples) :
            m_lambdas(lambdas), m_values(values), m_numSamples(numSamples) {}

        CUDA_DEVICE_FUNCTION SampledSpectrumTemplate<RealType, NumSpectralSamples> evaluate(const WavelengthSamplesTemplate<RealType, NumSpectralSamples> &wls) const;

#if defined(VLR_Host)
        void toXYZ(RealType XYZ[3]) const;
#endif
    };



    template <typename RealType, uint32_t NumStrataForStorage>
    class DiscretizedSpectrumTemplate {
        RealType values[NumStrataForStorage];

    public:
        static_assert(NumStrataForStorage == 16, "Code assumes NumStrataForStorage == 16.");
        CUDA_DEVICE_FUNCTION constexpr DiscretizedSpectrumTemplate(RealType v = 0.0f) :
            values{ v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v } {}
        CUDA_DEVICE_FUNCTION constexpr DiscretizedSpectrumTemplate(const RealType* vals) : 
            values{ vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15] } {}
        template <uint32_t N>
        CUDA_DEVICE_FUNCTION constexpr DiscretizedSpectrumTemplate(const WavelengthSamplesTemplate<RealType, N> &wls,
                                                                   const SampledSpectrumTemplate<RealType, N> &val) {
            const RealType recBinWidth = NumStrataForStorage / (WavelengthHighBound - WavelengthLowBound);
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] = 0.0f;
            for (int i = 0; i < N; ++i) {
                uint32_t sBin = ::vlr::min<uint32_t>((wls[i] - WavelengthLowBound) / (WavelengthHighBound - WavelengthLowBound) * NumStrataForStorage, NumStrataForStorage - 1);
                values[sBin] += val[i] * recBinWidth;
            }
        }

        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate operator+() const { return *this; }
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate operator-() const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = -values[i];
            return DiscretizedSpectrumTemplate(vals);
        }

        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate operator+(const DiscretizedSpectrumTemplate &c) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] + c.values[i];
            return DiscretizedSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate operator-(const DiscretizedSpectrumTemplate &c) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] - c.values[i];
            return DiscretizedSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate operator*(const DiscretizedSpectrumTemplate &c) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] * c.values[i];
            return DiscretizedSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate operator*(RealType s) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] * s;
            return DiscretizedSpectrumTemplate(vals);
        }
        CUDA_DEVICE_FUNCTION friend DiscretizedSpectrumTemplate operator*(RealType s, const DiscretizedSpectrumTemplate &c) {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = c.values[i] * s;
            return DiscretizedSpectrumTemplate(vals);
        }

        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate &operator+=(const DiscretizedSpectrumTemplate &c) {
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] += c.values[i];
            return *this;
        }
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate &operator*=(const DiscretizedSpectrumTemplate &c) {
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] *= c.values[i];
            return *this;
        }
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate &operator*=(RealType s) {
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] *= s;
            return *this;
        }

        CUDA_DEVICE_FUNCTION bool operator==(const DiscretizedSpectrumTemplate &c) const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] != c.values[i])
                    return false;
            return true;
        }
        CUDA_DEVICE_FUNCTION bool operator!=(const DiscretizedSpectrumTemplate &c) const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] != c.values[i])
                    return true;
            return false;
        }

        CUDA_DEVICE_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < NumStrataForStorage, "\"index\" is out of range [0, %u].", NumStrataForStorage - 1);
            return values[index];
        }
        CUDA_DEVICE_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < NumStrataForStorage, "\"index\" is out of range [0, %u].", NumStrataForStorage - 1);
            return values[index];
        }

        CUDA_DEVICE_FUNCTION RealType maxValue() const {
            RealType maxVal = values[0];
            for (int i = 1; i < NumStrataForStorage; ++i)
                maxVal = std::fmax(values[i], maxVal);
            return maxVal;
        }
        CUDA_DEVICE_FUNCTION RealType minValue() const {
            RealType minVal = values[0];
            for (int i = 1; i < NumStrataForStorage; ++i)
                minVal = std::fmin(values[i], minVal);
            return minVal;
        }
        CUDA_DEVICE_FUNCTION bool hasNonZero() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] != 0)
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool hasNaN() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (vlr::isnan(values[i]))
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool hasInf() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (vlr::isinf(values[i]))
                    return true;
            return false;
        }
        CUDA_DEVICE_FUNCTION bool hasNegative() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] < 0)
                    return true;
            return false;
        }

        CUDA_DEVICE_FUNCTION void toXYZ(RealType XYZ[3]) const;

        template <uint32_t N>
        CUDA_DEVICE_FUNCTION DiscretizedSpectrumTemplate &add(const WavelengthSamplesTemplate<RealType, N> &wls, const SampledSpectrumTemplate<RealType, N> &val) {
            const RealType recBinWidth = NumStrataForStorage / (WavelengthHighBound - WavelengthLowBound);
            for (int i = 0; i < N; ++i) {
                uint32_t sBin = ::vlr::min<uint32_t>((wls[i] - WavelengthLowBound) / (WavelengthHighBound - WavelengthLowBound) * NumStrataForStorage, NumStrataForStorage - 1);
                values[sBin] += val[i] * recBinWidth;
            }
            return *this;
        }

#if defined(VLR_Device) || defined(OPTIXU_Platform_CodeCompletion)
        template <uint32_t N>
        CUDA_DEVICE_FUNCTION void atomicAdd(const WavelengthSamplesTemplate<RealType, N> &wls, const SampledSpectrumTemplate<RealType, N> &val) {
            const RealType recBinWidth = NumStrataForStorage / (WavelengthHighBound - WavelengthLowBound);
            for (int i = 0; i < N; ++i) {
                uint32_t sBin = ::vlr::min<uint32_t>((wls[i] - WavelengthLowBound) / (WavelengthHighBound - WavelengthLowBound) * NumStrataForStorage, NumStrataForStorage - 1);
                ::atomicAdd(&values[sBin], val[i] * recBinWidth);
            }
        }
#endif

        CUDA_DEVICE_FUNCTION static constexpr uint32_t NumStrata() { return NumStrataForStorage; }
        CUDA_DEVICE_FUNCTION static constexpr DiscretizedSpectrumTemplate Zero() { return DiscretizedSpectrumTemplate(0.0f); }
        CUDA_DEVICE_FUNCTION static constexpr DiscretizedSpectrumTemplate One() { return DiscretizedSpectrumTemplate(1.0f); }
        CUDA_DEVICE_FUNCTION static constexpr DiscretizedSpectrumTemplate Inf() { return DiscretizedSpectrumTemplate(VLR_INFINITY); }
        CUDA_DEVICE_FUNCTION static constexpr DiscretizedSpectrumTemplate NaN() { return DiscretizedSpectrumTemplate(VLR_NAN); }

        struct CMF {
            RealType values[NumStrataForStorage];

            CUDA_DEVICE_FUNCTION RealType &operator[](uint32_t index) {
                return values[index];
            }
            CUDA_DEVICE_FUNCTION const RealType &operator[](uint32_t index) const {
                return values[index];
            }
        };

#if defined(VLR_Host)
        static CMF xbar;
        static CMF ybar;
        static CMF zbar;
        static RealType integralCMF;

        static void initialize();
#endif

        CUDA_DEVICE_FUNCTION void print() const {
            vlrprintf("%g", values[0]);
            for (int i = 1; i < NumStrataForStorage; ++i)
                vlrprintf(", %g", values[i]);
            vlrprintf("\n");
        }
    };



    template <typename RealType, uint32_t NumStrataForStorage>
    class SpectrumStorageTemplate {
        typedef DiscretizedSpectrumTemplate<RealType, NumStrataForStorage> ValueType;
        CompensatedSum<ValueType> value;

    public:
        CUDA_DEVICE_FUNCTION SpectrumStorageTemplate(const ValueType &v = ValueType::Zero()) : value(v) {}

        CUDA_DEVICE_FUNCTION void reset() {
            value = CompensatedSum<ValueType>(ValueType::Zero());
        }

        template <uint32_t N>
        CUDA_DEVICE_FUNCTION SpectrumStorageTemplate &add(const WavelengthSamplesTemplate<RealType, N> &wls, const SampledSpectrumTemplate<RealType, N> &val) {
            const RealType recBinWidth = NumStrataForStorage / (WavelengthHighBound - WavelengthLowBound);
            ValueType addend(0.0);
            for (int i = 0; i < N; ++i) {
                uint32_t sBin = ::vlr::min<uint32_t>((wls[i] - WavelengthLowBound) / (WavelengthHighBound - WavelengthLowBound) * NumStrataForStorage, NumStrataForStorage - 1);
                addend[sBin] += val[i] * recBinWidth;
            }
            value += addend;
            return *this;
        }

        CUDA_DEVICE_FUNCTION SpectrumStorageTemplate &add(const ValueType &val) {
            value += val;
            return *this;
        }

        CUDA_DEVICE_FUNCTION const CompensatedSum<ValueType> &getValue() const {
            return value;
        }
        CUDA_DEVICE_FUNCTION CompensatedSum<ValueType> &getValue() {
            return value;
        }
    };



    using UpsampledSpectrum = UpsampledSpectrumTemplate<float, NumSpectralSamples>;
    using RegularSampledSpectrum = RegularSampledSpectrumTemplate<float, NumSpectralSamples>;
    using IrregularSampledSpectrum = IrregularSampledSpectrumTemplate<float, NumSpectralSamples>;
}
