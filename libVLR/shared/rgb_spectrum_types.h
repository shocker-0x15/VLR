#pragma once

#include "spectrum_base.h"

namespace VLR {
    template <typename RealType>
    struct RGBWavelengthSamplesTemplate {
        struct {
            unsigned int _selectedLambdaIndex : 16;
            bool _singleIsSelected : 1;
        };

        RT_FUNCTION RGBWavelengthSamplesTemplate() {}
        RT_FUNCTION RGBWavelengthSamplesTemplate(const RGBWavelengthSamplesTemplate &wls) {
            _selectedLambdaIndex = wls._selectedLambdaIndex;
            _singleIsSelected = wls._singleIsSelected;
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

        RT_FUNCTION static RGBWavelengthSamplesTemplate createWithEqualOffsets(RealType offset, RealType uLambda, RealType* PDF) {
            VLRAssert(offset >= 0 && offset < 1, "\"offset\" must be in range [0, 1).");
            VLRAssert(uLambda >= 0 && uLambda < 1, "\"uLambda\" must be in range [0, 1).");
            RGBWavelengthSamplesTemplate wls;
            wls._selectedLambdaIndex = std::min<uint16_t>(3 * uLambda, 3 - 1);
            wls._singleIsSelected = false;
            *PDF = 1;
            return wls;
        }

        RT_FUNCTION static constexpr uint32_t NumComponents() { return 3; }
    };



    template <typename RealType>
    struct RGBSpectrumTemplate {
        RealType r, g, b;

    public:
        RT_FUNCTION RGBSpectrumTemplate() {}
        RT_FUNCTION constexpr RGBSpectrumTemplate(RealType v) : r(v), g(v), b(v) {}
        RT_FUNCTION constexpr RGBSpectrumTemplate(RealType rr, RealType gg, RealType bb) : r(rr), g(gg), b(bb) {}

        RT_FUNCTION RGBSpectrumTemplate operator+() const {
            return *this;
        }
        RT_FUNCTION RGBSpectrumTemplate operator-() const {
            return RGBSpectrumTemplate(-r, -g, -b);
        }

        RT_FUNCTION RGBSpectrumTemplate operator+(const RGBSpectrumTemplate &c) const {
            return RGBSpectrumTemplate(r + c.r, g + c.g, b + c.b);
        }
        RT_FUNCTION RGBSpectrumTemplate operator-(const RGBSpectrumTemplate &c) const {
            return RGBSpectrumTemplate(r - c.r, g - c.g, b - c.b);
        }
        RT_FUNCTION RGBSpectrumTemplate operator*(const RGBSpectrumTemplate &c) const {
            return RGBSpectrumTemplate(r * c.r, g * c.g, b * c.b);
        }
        RT_FUNCTION RGBSpectrumTemplate operator/(const RGBSpectrumTemplate &c) const {
            return RGBSpectrumTemplate(r / c.r, g / c.g, b / c.b);
        }
        RT_FUNCTION RGBSpectrumTemplate safeDivide(const RGBSpectrumTemplate &c) const {
            return RGBSpectrumTemplate(c.r > 0.0f ? r / c.r : 0.0f,
                                       c.g > 0.0f ? g / c.g : 0.0f,
                                       c.b > 0.0f ? b / c.b : 0.0f);
        }
        RT_FUNCTION RGBSpectrumTemplate operator*(RealType s) const {
            return RGBSpectrumTemplate(r * s, g * s, b * s);
        }
        RT_FUNCTION RGBSpectrumTemplate operator/(RealType s) const {
            RealType rc = 1.0f / s;
            return RGBSpectrumTemplate(r * rc, g * rc, b * rc);
        }
        RT_FUNCTION friend inline RGBSpectrumTemplate operator*(RealType s, const RGBSpectrumTemplate &c) {
            return RGBSpectrumTemplate(s * c.r, s * c.g, s * c.b);
        }

        RT_FUNCTION RGBSpectrumTemplate &operator+=(const RGBSpectrumTemplate &c) {
            r += c.r; g += c.g; b += c.b;
            return *this;
        }
        RT_FUNCTION RGBSpectrumTemplate &operator-=(const RGBSpectrumTemplate &c) {
            r -= c.r; g -= c.g; b -= c.b;
            return *this;
        }
        RT_FUNCTION RGBSpectrumTemplate &operator*=(const RGBSpectrumTemplate &c) {
            r *= c.r; g *= c.g; b *= c.b;
            return *this;
        }
        RT_FUNCTION RGBSpectrumTemplate &operator/=(const RGBSpectrumTemplate &c) {
            r /= c.r; g /= c.g; b /= c.b;
            return *this;
        }
        RT_FUNCTION RGBSpectrumTemplate &operator*=(RealType s) {
            r *= s; g *= s; b *= s; return *this;
        }
        RT_FUNCTION RGBSpectrumTemplate &operator/=(RealType s) {
            RealType rc = 1.0f / s; r *= rc; g *= rc; b *= rc;
            return *this;
        }

        RT_FUNCTION bool operator==(const RGBSpectrumTemplate &c) const {
            return r == c.r && g == c.g && b == c.b;
        }
        RT_FUNCTION bool operator!=(const RGBSpectrumTemplate &c) const {
            return r != c.r || g != c.g || b != c.b;
        }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&r + index);
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&r + index);
        }

        RT_FUNCTION RealType avgValue() const {
            return (r + g + b) / 3;
        }
        RT_FUNCTION RealType maxValue() const {
            return std::fmax(r, std::max(g, b));
        }
        RT_FUNCTION RealType minValue() const {
            return std::fmin(r, std::min(g, b));
        }
        RT_FUNCTION bool hasNonZero() const {
            return r != 0.0f || g != 0.0f || b != 0.0f;
        }
        RT_FUNCTION bool hasNaN() const {
            using std::isnan;
            return isnan(r) || isnan(g) || isnan(b);
        }
        RT_FUNCTION bool hasInf() const {
            using std::isinf;
            return isinf(r) || isinf(g) || isinf(b);
        }
        RT_FUNCTION bool allFinite() const {
            return !hasNaN() && !hasInf();
        }
        RT_FUNCTION bool hasNegative() const {
            return r < 0.0f || g < 0.0f || b < 0.0f;
        }
        RT_FUNCTION bool allPositiveFinite() const {
            return allFinite() && !hasNegative();
        }

        // setting "primary" to 1.0 might introduce bias.
        RT_FUNCTION RealType importance(uint16_t selectedLambda) const {
            RealType sum = r + g + b;
            const RealType primary = 0.9f;
            const RealType marginal = (1 - primary) / 2;
            return sum * marginal + (*this)[selectedLambda] * (primary - marginal);
        }

        RT_FUNCTION static constexpr uint32_t NumComponents() { return 3; }
        RT_FUNCTION static constexpr RGBSpectrumTemplate Zero() { return RGBSpectrumTemplate(0.0); }
        RT_FUNCTION static constexpr RGBSpectrumTemplate One() { return RGBSpectrumTemplate(1.0); }
        RT_FUNCTION static constexpr RGBSpectrumTemplate Inf() { return RGBSpectrumTemplate(VLR_INFINITY); }
        RT_FUNCTION static constexpr RGBSpectrumTemplate NaN() { return RGBSpectrumTemplate(VLR_NAN); }

        // ----------------------------------------------------------------
        // Methods for compatibility with ContinuousSpectrumTemplate

        RT_FUNCTION const RGBSpectrumTemplate &evaluate(const RGBWavelengthSamplesTemplate<RealType> &wls) const {
            return *this;
        }

        // ----------------------------------------------------------------

        // ----------------------------------------------------------------
        // Methods for compatibility with DiscretizedSpectrumTemplate

        RT_FUNCTION void toXYZ(RealType XYZ[3]) const {
            const RealType RGB[3] = { r, g, b };
            transformFromRenderingRGB(SpectrumType::LightSource, RGB, XYZ);
        }

#if defined(VLR_Host)
        static void initialize() {}
#endif

        // ----------------------------------------------------------------
    };

    template <typename RealType>
    RT_FUNCTION constexpr RGBSpectrumTemplate<RealType> min(const RGBSpectrumTemplate<RealType> &value, RealType minValue) {
        return RGBSpectrumTemplate<RealType>(std::fmin(minValue, value.r), std::fmin(minValue, value.g), std::fmin(minValue, value.b));
    }

    template <typename RealType>
    RT_FUNCTION constexpr RGBSpectrumTemplate<RealType> max(const RGBSpectrumTemplate<RealType> &value, RealType maxValue) {
        return RGBSpectrumTemplate<RealType>(std::fmax(maxValue, value.r), std::fmax(maxValue, value.g), std::fmax(maxValue, value.b));
    }

    template <typename RealType>
    RT_FUNCTION constexpr RGBSpectrumTemplate<RealType> lerp(const RGBSpectrumTemplate<RealType> &v0, const RGBSpectrumTemplate<RealType> &v1, RealType t) {
        return (1 - t) * v0 + t * v1;
    }



    template <typename RealType>
    class RGBStorageTemplate {
        typedef RGBSpectrumTemplate<RealType> ValueType;
        CompensatedSum<ValueType> value;

    public:
        RT_FUNCTION RGBStorageTemplate(const ValueType &v = ValueType::Zero()) : value(v) {}

        RT_FUNCTION void reset() {
            value = CompensatedSum<ValueType>(ValueType::Zero());
        }

        RT_FUNCTION RGBStorageTemplate &add(const RGBWavelengthSamplesTemplate<RealType> &wls, const RGBSpectrumTemplate<RealType> &val) {
            value += val;
            return *this;
        }

        RT_FUNCTION const CompensatedSum<ValueType> &getValue() const {
            return value;
        }
        RT_FUNCTION CompensatedSum<ValueType> &getValue() {
            return value;
        }
    };



    using RGBSpectrum = RGBSpectrumTemplate<float>;
}
