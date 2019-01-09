#pragma once

#include "spectrum_base.h"

namespace VLR {
    template <typename RealType, uint32_t NumSpectralSamples>
    struct WavelengthSamplesTemplate {
        RealType lambdas[NumSpectralSamples];
        struct {
            unsigned int _selectedLambdaIndex : 16;
            bool _singleIsSelected : 1;
        };

        RT_FUNCTION WavelengthSamplesTemplate() {}
        RT_FUNCTION WavelengthSamplesTemplate(const RealType* values) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                lambdas[i] = values[i];
            _selectedLambdaIndex = 0;
            _singleIsSelected = false;
        }
        RT_FUNCTION WavelengthSamplesTemplate(const WavelengthSamplesTemplate &wls) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                lambdas[i] = wls.lambdas[i];
            _selectedLambdaIndex = wls._selectedLambdaIndex;
            _singleIsSelected = wls._singleIsSelected;
        }

        RT_FUNCTION RealType &operator[](uint32_t index) {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return lambdas[index];
        }
        RT_FUNCTION RealType operator[](uint32_t index) const {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return lambdas[index];
        }

        RT_FUNCTION uint32_t selectedLambdaIndex() const {
            return _selectedLambdaIndex;
        }
        RT_FUNCTION void setSelectedLambdaIndex(uint32_t index) const {
            _selectedLambdaIndex = index;
        }
        RT_FUNCTION bool singleIsSelected() const {
            return _singleIsSelected;
        }
        RT_FUNCTION void setSingleIsSelected() {
            _singleIsSelected = true;
        }

        RT_FUNCTION RealType selectedWavelength() const {
            return lambdas[_selectedLambdaIndex];
        }

        RT_FUNCTION static WavelengthSamplesTemplate createWithEqualOffsets(RealType offset, RealType uLambda, RealType* PDF) {
            VLRAssert(offset >= 0 && offset < 1, "\"offset\" must be in range [0, 1).");
            VLRAssert(uLambda >= 0 && uLambda < 1, "\"uLambda\" must be in range [0, 1).");
            WavelengthSamplesTemplate wls;
            for (int i = 0; i < NumSpectralSamples; ++i)
                wls.lambdas[i] = WavelengthLowBound + (WavelengthHighBound - WavelengthLowBound) * (i + offset) / NumSpectralSamples;
            wls._selectedLambdaIndex = std::min<uint16_t>(NumSpectralSamples * uLambda, NumSpectralSamples - 1);
            wls._singleIsSelected = false;
            *PDF = NumSpectralSamples / (WavelengthHighBound - WavelengthLowBound);
            return wls;
        }

        RT_FUNCTION static constexpr uint32_t NumComponents() { return NumSpectralSamples; }
    };



    template <typename RealType, uint32_t NumSpectralSamples>
    struct SampledSpectrumTemplate {
        RealType values[NumSpectralSamples];

        static_assert(NumSpectralSamples == 4, "Code assumes NumSpectralSamples == 4.");
        RT_FUNCTION SampledSpectrumTemplate() {}
        RT_FUNCTION constexpr SampledSpectrumTemplate(RealType v) : values{ v, v, v, v } {}
        RT_FUNCTION constexpr SampledSpectrumTemplate(const RealType* vals) : values{ vals[0], vals[1], vals[2], vals[3] } {}



        RT_FUNCTION SampledSpectrumTemplate operator+() const { return *this; };
        RT_FUNCTION SampledSpectrumTemplate operator-() const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = -values[i];
            return SampledSpectrumTemplate(vals);
        }

        RT_FUNCTION SampledSpectrumTemplate operator+(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] + c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION SampledSpectrumTemplate operator-(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] - c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION SampledSpectrumTemplate operator*(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] * c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION SampledSpectrumTemplate operator/(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] / c.values[i];
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION SampledSpectrumTemplate safeDivide(const SampledSpectrumTemplate &c) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = c.values[i] > 0 ? values[i] / c.values[i] : 0.0f;
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION SampledSpectrumTemplate operator*(RealType s) const {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] * s;
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION SampledSpectrumTemplate operator/(RealType s) const {
            RealType vals[NumSpectralSamples];
            RealType r = 1 / s;
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = values[i] * r;
            return SampledSpectrumTemplate(vals);
        }
        RT_FUNCTION friend inline SampledSpectrumTemplate operator*(RealType s, const SampledSpectrumTemplate &c) {
            RealType vals[NumSpectralSamples];
            for (int i = 0; i < NumSpectralSamples; ++i)
                vals[i] = c.values[i] * s;
            return SampledSpectrumTemplate(vals);
        }

        RT_FUNCTION SampledSpectrumTemplate &operator+=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] += c.values[i];
            return *this;
        }
        RT_FUNCTION SampledSpectrumTemplate &operator-=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] -= c.values[i];
            return *this;
        }
        RT_FUNCTION SampledSpectrumTemplate &operator*=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] *= c.values[i];
            return *this;
        }
        RT_FUNCTION SampledSpectrumTemplate &operator/=(const SampledSpectrumTemplate &c) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] /= c.values[i];
            return *this;
        }
        RT_FUNCTION SampledSpectrumTemplate &operator*=(RealType s) {
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] *= s;
            return *this;
        }
        RT_FUNCTION SampledSpectrumTemplate &operator/=(RealType s) {
            RealType r = 1 / s;
            for (int i = 0; i < NumSpectralSamples; ++i)
                values[i] *= r;
            return *this;
        }

        RT_FUNCTION bool operator==(const SampledSpectrumTemplate &c) const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] != c.values[i])
                    return false;
            return true;
        }
        RT_FUNCTION bool operator!=(const SampledSpectrumTemplate &c) const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] != c.values[i])
                    return true;
            return false;
        }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return values[index];
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < NumSpectralSamples, "\"index\" is out of range [0, %u].", NumSpectralSamples - 1);
            return values[index];
        }

        RT_FUNCTION RealType avgValue() const {
            RealType sumVal = values[0];
            for (int i = 1; i < NumSpectralSamples; ++i)
                sumVal += values[i];
            return sumVal / NumSpectralSamples;
        }
        RT_FUNCTION RealType maxValue() const {
            RealType maxVal = values[0];
            for (int i = 1; i < NumSpectralSamples; ++i)
                maxVal = std::fmax(values[i], maxVal);
            return maxVal;
        }
        RT_FUNCTION RealType minValue() const {
            RealType minVal = values[0];
            for (int i = 1; i < NumSpectralSamples; ++i)
                minVal = std::fmin(values[i], minVal);
            return minVal;
        }
        RT_FUNCTION bool hasNonZero() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] != 0)
                    return true;
            return false;
        }
        RT_FUNCTION bool hasNaN() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (std::isnan(values[i]))
                    return true;
            return false;
        }
        RT_FUNCTION bool hasInf() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (std::isinf(values[i]))
                    return true;
            return false;
        }
        RT_FUNCTION bool allFinite() const {
            return !hasNaN() && !hasInf();
        }
        RT_FUNCTION bool hasNegative() const {
            for (int i = 0; i < NumSpectralSamples; ++i)
                if (values[i] < 0)
                    return true;
            return false;
        }
        RT_FUNCTION bool allPositiveFinite() const {
            return allFinite() && !hasNegative();
        }

        // setting "primary" to 1.0 might introduce bias.
        RT_FUNCTION RealType importance(uint32_t selectedLambda) const {
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

        RT_FUNCTION static constexpr uint32_t NumComponents() { return NumSpectralSamples; }
        RT_FUNCTION static constexpr SampledSpectrumTemplate Zero() { return SampledSpectrumTemplate(0.0); }
        RT_FUNCTION static constexpr SampledSpectrumTemplate One() { return SampledSpectrumTemplate(1.0); }
        RT_FUNCTION static constexpr SampledSpectrumTemplate Inf() { return SampledSpectrumTemplate(VLR_INFINITY); }
        RT_FUNCTION static constexpr SampledSpectrumTemplate NaN() { return SampledSpectrumTemplate(VLR_NAN); }
    };

    template <typename RealType, uint32_t NumSpectralSamples>
    RT_FUNCTION constexpr SampledSpectrumTemplate<RealType, NumSpectralSamples> min(const SampledSpectrumTemplate<RealType, NumSpectralSamples> &value, RealType minValue) {
        SampledSpectrumTemplate<RealType, NumSpectralSamples> ret;
        for (int i = 0; i < NumSpectralSamples; ++i)
            ret[i] = std::fmin(value[0], minValue);
        return ret;
    }

    template <typename RealType, uint32_t NumSpectralSamples>
    RT_FUNCTION constexpr SampledSpectrumTemplate<RealType, NumSpectralSamples> max(const SampledSpectrumTemplate<RealType, NumSpectralSamples> &value, RealType maxValue) {
        SampledSpectrumTemplate<RealType, NumSpectralSamples> ret;
        for (int i = 0; i < NumSpectralSamples; ++i)
            ret[i] = std::fmax(value[0], maxValue);
        return ret;
    }

    template <typename RealType, uint32_t NumSpectralSamples>
    RT_FUNCTION constexpr SampledSpectrumTemplate<RealType, NumSpectralSamples> lerp(const SampledSpectrumTemplate<RealType, NumSpectralSamples> &v0, const SampledSpectrumTemplate<RealType, NumSpectralSamples> &v1, RealType t) {
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
        uint32_t m_adjIndices;
        uint16_t m_s, m_t;
        RealType m_scale;

        RT_FUNCTION void computeAdjacents(RealType u, RealType v);

    public:
        RT_FUNCTION constexpr UpsampledSpectrumTemplate(uint32_t adjIndices, uint16_t s, uint16_t t, RealType scale) :
        m_adjIndices(adjIndices), m_s((RealType)s / (UINT16_MAX - 1)), m_t((RealType)t / (UINT16_MAX - 1)), m_scale(scale) {}
        RT_FUNCTION constexpr UpsampledSpectrumTemplate(VLRSpectrumType spType, VLRColorSpace space, RealType e0, RealType e1, RealType e2);

        RT_FUNCTION SampledSpectrumTemplate<RealType, NumSpectralSamples> evaluate(const WavelengthSamplesTemplate<RealType, NumSpectralSamples> &wls) const;

        RT_FUNCTION static constexpr RealType MinWavelength() { return 360.0; }
        RT_FUNCTION static constexpr RealType MaxWavelength() { return 830.0; }
        RT_FUNCTION static constexpr uint32_t NumWavelengthSamples() { return 95; }
        RT_FUNCTION static constexpr uint32_t GridWidth() { return 12; }
        RT_FUNCTION static constexpr uint32_t GridHeight() { return 14; }

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

#if defined(VLR_Host)
        static const spectrum_grid_cell_t spectrum_grid[];
        static const spectrum_data_point_t spectrum_data_points[];
#endif

        // This is 1 over the integral over either CMF.
        // Spectra can be mapped so that xyz=(1,1,1) is converted to constant 1 by
        // dividing by this value. This is important for valid reflectances.
        RT_FUNCTION static constexpr RealType EqualEnergyReflectance() { return 0.009355121400914532; }

        RT_FUNCTION static constexpr void xy_to_uv(const RealType xy[2], RealType uv[2]) {
            uv[0] = 16.730260708356887 * xy[0] + 7.7801960340706 * xy[1] - 2.170152247475828;
            uv[1] = -7.530081094743006 * xy[0] + 16.192422314095225 * xy[1] + 1.1125529268825947;
        }

        RT_FUNCTION static constexpr void uv_to_xy(const RealType uv[2], RealType xy[2]) {
            xy[0] = 0.0491440520940413 * uv[0] - 0.02361291916573777 * uv[1] + 0.13292069743203658;
            xy[1] = 0.022853819546830627 * uv[0] + 0.05077639329371236 * uv[1] - 0.006895157122499944;
        }
    };



    template <typename RealType, uint32_t NumSpectralSamples>
    class RegularSampledSpectrumTemplate {
        float m_minLambda;
        float m_maxLambda;
        const RealType* m_values;
        uint32_t m_numSamples;

    public:
        RT_FUNCTION RegularSampledSpectrumTemplate(RealType minLambda, RealType maxLambda, const RealType* values, uint32_t numSamples) :
            m_minLambda(minLambda), m_maxLambda(maxLambda), m_values(values), m_numSamples(numSamples) {}

        RT_FUNCTION SampledSpectrumTemplate<RealType, NumSpectralSamples> evaluate(const WavelengthSamplesTemplate<RealType, NumSpectralSamples> &wls) const;

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
        RT_FUNCTION IrregularSampledSpectrumTemplate(const RealType* lambdas, const RealType* values, uint32_t numSamples) :
            m_lambdas(lambdas), m_values(values), m_numSamples(numSamples) {}

        RT_FUNCTION SampledSpectrumTemplate<RealType, NumSpectralSamples> evaluate(const WavelengthSamplesTemplate<RealType, NumSpectralSamples> &wls) const;

#if defined(VLR_Host)
        void toXYZ(RealType XYZ[3]) const;
#endif
    };



    template <typename RealType, uint32_t NumStrataForStorage>
    struct DiscretizedSpectrumTemplate {
        RealType values[NumStrataForStorage];

    public:
        static_assert(NumStrataForStorage == 16, "Code assumes NumStrataForStorage == 16.");
        RT_FUNCTION constexpr DiscretizedSpectrumTemplate(RealType v = 0.0f) :
            values{ v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v } {}
        RT_FUNCTION constexpr DiscretizedSpectrumTemplate(const RealType* vals) : 
            values{ vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15] } {}

        RT_FUNCTION DiscretizedSpectrumTemplate operator+() const { return *this; }
        RT_FUNCTION DiscretizedSpectrumTemplate operator-() const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = -values[i];
            return DiscretizedSpectrumTemplate(vals);
        }

        RT_FUNCTION DiscretizedSpectrumTemplate operator+(const DiscretizedSpectrumTemplate &c) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] + c.values[i];
            return DiscretizedSpectrumTemplate(vals);
        }
        RT_FUNCTION DiscretizedSpectrumTemplate operator-(const DiscretizedSpectrumTemplate &c) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] - c.values[i];
            return DiscretizedSpectrumTemplate(vals);
        }
        RT_FUNCTION DiscretizedSpectrumTemplate operator*(const DiscretizedSpectrumTemplate &c) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] * c.values[i];
            return DiscretizedSpectrumTemplate(vals);
        }
        RT_FUNCTION DiscretizedSpectrumTemplate operator*(RealType s) const {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = values[i] * s;
            return DiscretizedSpectrumTemplate(vals);
        }
        RT_FUNCTION friend inline DiscretizedSpectrumTemplate operator*(RealType s, const DiscretizedSpectrumTemplate &c) {
            RealType vals[NumStrataForStorage];
            for (int i = 0; i < NumStrataForStorage; ++i)
                vals[i] = c.values[i] * s;
            return DiscretizedSpectrumTemplate(vals);
        }

        RT_FUNCTION DiscretizedSpectrumTemplate &operator+=(const DiscretizedSpectrumTemplate &c) {
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] += c.values[i];
            return *this;
        }
        RT_FUNCTION DiscretizedSpectrumTemplate &operator*=(const DiscretizedSpectrumTemplate &c) {
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] *= c.values[i];
            return *this;
        }
        RT_FUNCTION DiscretizedSpectrumTemplate &operator*=(RealType s) {
            for (int i = 0; i < NumStrataForStorage; ++i)
                values[i] *= s;
            return *this;
        }

        RT_FUNCTION bool operator==(const DiscretizedSpectrumTemplate &c) const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] != c.values[i])
                    return false;
            return true;
        }
        RT_FUNCTION bool operator!=(const DiscretizedSpectrumTemplate &c) const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] != c.values[i])
                    return true;
            return false;
        }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < NumStrataForStorage, "\"index\" is out of range [0, %u].", NumStrataForStorage - 1);
            return values[index];
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < NumStrataForStorage, "\"index\" is out of range [0, %u].", NumStrataForStorage - 1);
            return values[index];
        }

        RT_FUNCTION RealType maxValue() const {
            RealType maxVal = values[0];
            for (int i = 1; i < NumStrataForStorage; ++i)
                maxVal = std::fmax(values[i], maxVal);
            return maxVal;
        }
        RT_FUNCTION RealType minValue() const {
            RealType minVal = values[0];
            for (int i = 1; i < NumStrataForStorage; ++i)
                minVal = std::fmin(values[i], minVal);
            return minVal;
        }
        RT_FUNCTION bool hasNonZero() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] != 0)
                    return true;
            return false;
        }
        RT_FUNCTION bool hasNaN() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (std::isnan(values[i]))
                    return true;
            return false;
        }
        RT_FUNCTION bool hasInf() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (std::isinf(values[i]))
                    return true;
            return false;
        }
        RT_FUNCTION bool hasNegative() const {
            for (int i = 0; i < NumStrataForStorage; ++i)
                if (values[i] < 0)
                    return true;
            return false;
        }

        RT_FUNCTION void toXYZ(RealType XYZ[3]) const;

        RT_FUNCTION static constexpr uint32_t NumStrata() { return NumStrataForStorage; }
        RT_FUNCTION static constexpr DiscretizedSpectrumTemplate Zero() { return DiscretizedSpectrumTemplate(0.0f); }
        RT_FUNCTION static constexpr DiscretizedSpectrumTemplate One() { return DiscretizedSpectrumTemplate(1.0f); }
        RT_FUNCTION static constexpr DiscretizedSpectrumTemplate Inf() { return DiscretizedSpectrumTemplate(VLR_INFINITY); }
        RT_FUNCTION static constexpr DiscretizedSpectrumTemplate NaN() { return DiscretizedSpectrumTemplate(VLR_NAN); }

        struct CMF {
            RealType values[NumStrataForStorage];

            RT_FUNCTION RealType &operator[](uint32_t index) {
                return values[index];
            }
            RT_FUNCTION const RealType &operator[](uint32_t index) const {
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
    };



    template <typename RealType, uint32_t NumStrataForStorage>
    class SpectrumStorageTemplate {
        typedef DiscretizedSpectrumTemplate<RealType, NumStrataForStorage> ValueType;
        CompensatedSum<ValueType> value;

    public:
        RT_FUNCTION SpectrumStorageTemplate(const ValueType &v = ValueType::Zero()) : value(v) {}

        RT_FUNCTION void reset() {
            value = CompensatedSum<ValueType>(ValueType::Zero());
        }

        template <uint32_t N>
        RT_FUNCTION SpectrumStorageTemplate &add(const WavelengthSamplesTemplate<RealType, N> &wls, const SampledSpectrumTemplate<RealType, N> &val) {
            const RealType recBinWidth = NumStrataForStorage / (WavelengthHighBound - WavelengthLowBound);
            ValueType addend(0.0);
            for (int i = 0; i < N; ++i) {
                uint32_t sBin = std::min<uint32_t>((wls[i] - WavelengthLowBound) / (WavelengthHighBound - WavelengthLowBound) * NumStrataForStorage, NumStrataForStorage - 1);
                addend[sBin] += val[i] * recBinWidth;
            }
            value += addend;
            return *this;
        }

        RT_FUNCTION CompensatedSum<ValueType> &getValue() {
            return value;
        }
    };



    using UpsampledSpectrum = UpsampledSpectrumTemplate<float, NumSpectralSamples>;
    using RegularSampledSpectrum = RegularSampledSpectrumTemplate<float, NumSpectralSamples>;
    using IrregularSampledSpectrum = IrregularSampledSpectrumTemplate<float, NumSpectralSamples>;
}

#if defined(VLR_Device)
#include "spectrum_types.cpp"
#endif