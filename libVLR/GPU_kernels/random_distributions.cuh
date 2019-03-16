#pragma once

#include "../shared/basic_types_internal.h"

namespace VLR {
    class PCG32RNG {
        uint64_t state;

    public:
        RT_FUNCTION PCG32RNG() {}

        RT_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
        }

        RT_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    class XORShiftRNG {
        uint32_t m_state[4];

    public:
        RT_FUNCTION XORShiftRNG() {}
        RT_FUNCTION uint32_t operator()() {
            uint32_t* a = m_state;
            uint32_t t(a[0] ^ (a[0] << 11));
            a[0] = a[1];
            a[1] = a[2];
            a[2] = a[3];
            return a[3] = (a[3] ^ (a[3] >> 19)) ^ (t ^ (t >> 8));
        }

        RT_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    using KernelRNG = PCG32RNG;



    template <typename RealType>
    RT_FUNCTION uint32_t sampleDiscrete(const RealType* importances, uint32_t numImportances, RealType u,
                                        RealType* prob, RealType* sumImportances, RealType* remapped) {
        RealType sum = 0;
        for (int i = 0; i < numImportances; ++i)
            sum += importances[i];
        *sumImportances = sum;

        RealType base = 0;
        RealType su = u * sum;
        RealType cum = 0;
        for (int i = 0; i < numImportances; ++i) {
            base = cum;
            cum += importances[i];
            if (su < cum) {
                *prob = importances[i] / sum;
                *remapped = (su - base) / importances[i];
                return i;
            }
        }
        *prob = importances[0] / sum;
        return 0;
    }



    template <typename RealType>
    RT_FUNCTION void concentricSampleDisk(RealType u0, RealType u1, RealType* dx, RealType* dy) {
        RealType r, theta;
        RealType sx = 2 * u0 - 1;
        RealType sy = 2 * u1 - 1;

        if (sx == 0 && sy == 0) {
            *dx = 0;
            *dy = 0;
            return;
        }
        if (sx >= -sy) { // region 1 or 2
            if (sx > sy) { // region 1
                r = sx;
                theta = sy / sx;
            }
            else { // region 2
                r = sy;
                theta = 2 - sx / sy;
            }
        }
        else { // region 3 or 4
            if (sx > sy) {/// region 4
                r = -sy;
                theta = 6 + sx / sy;
            }
            else {// region 3
                r = -sx;
                theta = 4 + sy / sx;
            }
        }
        theta *= M_PI_4f;
        *dx = r * cos(theta);
        *dy = r * sin(theta);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> cosineSampleHemisphere(RealType u0, RealType u1) {
        //        RealType phi = 2 * M_PI * u1;
        //        RealType theta = std::asin(std::sqrt(u0));
        //        return Vector3DTemplate<RealType>(std::cos(phi) * std::sin(theta), std::sin(phi) * std::sin(theta), std::cos(theta));
        RealType x, y;
        concentricSampleDisk(u0, u1, &x, &y);
        return Vector3DTemplate<RealType>(x, y, std::sqrt(std::fmax(0.0f, 1.0f - x * x - y * y)));
    }

    template <typename RealType, int N>
    RT_FUNCTION inline Vector3DTemplate<RealType> cosNSampleHemisphere(RealType u0, RealType u1) {
        RealType phi = 2 * M_PIf * u1;
        RealType theta = std::acos(std::pow(u0, 1.0 / (1 + N)));
        return Vector3DTemplate<RealType>::fromPolarZUp(phi, theta);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> uniformSampleHemisphere(RealType u0, RealType u1) {
        RealType phi = 2 * M_PIf * u1;
        RealType theta = std::acos(1 - u0);
        return Vector3DTemplate<RealType>::fromPolarZUp(phi, theta);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> uniformSampleSphere(RealType u0, RealType u1) {
        RealType phi = 2 * M_PIf * u1;
        RealType theta = std::acos(1 - 2 * u0);
        return Vector3DTemplate<RealType>::fromPolarZUp(phi, theta);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> uniformSampleCone(RealType u0, RealType u1, RealType cosThetaMax) {
        RealType phi = 2 * M_PIf * u1;
        RealType theta = std::acos(1 - (1 - cosThetaMax) * u0);
        return Vector3DTemplate<RealType>::fromPolarZUp(phi, theta);
    }

    template <typename RealType>
    RT_FUNCTION inline void uniformSampleTriangle(RealType u0, RealType u1, RealType* b0, RealType* b1) {
        // Square-Root Parameterization
        //RealType su1 = std::sqrt(u0);
        //*b0 = 1.0f - su1;
        //*b1 = u1 * su1;
        
        // Reference
        // A Low-Distortion Map Between Triangle and Square
        *b0 = 0.5f * u0;
        *b1 = 0.5f * u1;
        RealType offset = *b1 - *b0;
        if (offset > 0)
            *b1 += offset;
        else
            *b0 -= offset;
    }
}
