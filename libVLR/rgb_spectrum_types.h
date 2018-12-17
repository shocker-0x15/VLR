#pragma once

#include "spectrum_base.h"

namespace VLR {
    class RGBSpectrum : public Spectrum {
        float r, g, b;

    public:
        RGBSpectrum(float rr, float gg, float bb) : r(rr), g(gg), b(bb) {}
    };
}
