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
        RT_FUNCTION constexpr RGBSpectrumTemplate(RealType rr, RealType gg, RealType bb) : r(rr), g(gg), b(bb) {}

        RT_FUNCTION static constexpr uint32_t NumComponents() { return 3; }
        RT_FUNCTION static constexpr RGBSpectrumTemplate Zero() { return RGBSpectrumTemplate(0.0); }
        RT_FUNCTION static constexpr RGBSpectrumTemplate One() { return RGBSpectrumTemplate(1.0); }
        RT_FUNCTION static constexpr RGBSpectrumTemplate Inf() { return RGBSpectrumTemplate(VLR_INFINITY); }
        RT_FUNCTION static constexpr RGBSpectrumTemplate NaN() { return RGBSpectrumTemplate(VLR_NAN); }
    };



    template <typename RealType>
    class RGBStorageTemplate {
        typedef RGBSpectrumTemplate<RealType> ValueType;
        CompensatedSum<ValueType> value;

    public:
        RT_FUNCTION RGBStorageTemplate(const ValueType &v = ValueType::Zero()) : value(v) {}

        template <uint32_t N>
        RT_FUNCTION RGBStorageTemplate &add(const RGBWavelengthSamplesTemplate<RealType> &wls, const RGBSpectrumTemplate<RealType> &val) {
            value += val;
            return *this;
        }

        RT_FUNCTION CompensatedSum<ValueType> &getValue() {
            return value;
        }
    };



    using RGBSpectrum = RGBSpectrumTemplate<float>;
}
