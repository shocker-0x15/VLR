#pragma once

#include "common.h"

namespace VLR {
    template <typename RealType>
    struct Vector3DTemplate {
        RealType x, y, z;

#if defined(VLR_Device)
        RT_FUNCTION constexpr Vector3DTemplate() { }
#else
        constexpr Vector3DTemplate() : x(0), y(0), z(0) { }
#endif
        RT_FUNCTION constexpr Vector3DTemplate(RealType v) : x(v), y(v), z(v) { }
        RT_FUNCTION constexpr Vector3DTemplate(RealType xx, RealType yy, RealType zz) : x(xx), y(yy), z(zz) { }

        RT_FUNCTION Vector3DTemplate operator+() const { return *this; }
        RT_FUNCTION Vector3DTemplate operator-() const { return Vector3DTemplate(-x, -y, -z); }

        RT_FUNCTION Vector3DTemplate operator+(const Vector3DTemplate &v) const { return Vector3DTemplate(x + v.x, y + v.y, z + v.z); }
        RT_FUNCTION Vector3DTemplate operator-(const Vector3DTemplate &v) const { return Vector3DTemplate(x - v.x, y - v.y, z - v.z); }
        RT_FUNCTION Vector3DTemplate operator*(const Vector3DTemplate &v) const { return Vector3DTemplate(x * v.x, y * v.y, z * v.z); }
        RT_FUNCTION Vector3DTemplate operator/(const Vector3DTemplate &v) const { return Vector3DTemplate(x / v.x, y / v.y, z / v.z); }
        RT_FUNCTION Vector3DTemplate operator*(RealType s) const { return Vector3DTemplate(x * s, y * s, z * s); }
        RT_FUNCTION Vector3DTemplate operator/(RealType s) const { RealType r = 1 / s; return Vector3DTemplate(x * r, y * r, z * r); }
        RT_FUNCTION friend inline Vector3DTemplate operator*(RealType s, const Vector3DTemplate &v) { return Vector3DTemplate(s * v.x, s * v.y, s * v.z); }

        RT_FUNCTION Vector3DTemplate &operator+=(const Vector3DTemplate &v) { x += v.x; y += v.y; z += v.z; return *this; }
        RT_FUNCTION Vector3DTemplate &operator-=(const Vector3DTemplate &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
        RT_FUNCTION Vector3DTemplate &operator*=(RealType s) { x *= s; y *= s; z *= s; return *this; }
        RT_FUNCTION Vector3DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; return *this; }

        RT_FUNCTION bool operator==(const Vector3DTemplate &v) const { return x == v.x && y == v.y && z == v.z; }
        RT_FUNCTION bool operator!=(const Vector3DTemplate &v) const { return x != v.x || y != v.y || z != v.z; }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }

        RT_FUNCTION RealType length() const {
            return std::sqrt(x * x + y * y + z * z);
        }
        RT_FUNCTION RealType sqLength() const { return x * x + y * y + z * z; }
        RT_FUNCTION Vector3DTemplate& normalize() {
            RealType length = std::sqrt(x * x + y * y + z * z);
            return *this /= length;
        }
        RT_FUNCTION Vector3DTemplate reciprocal() const { return Vector3DTemplate(1 / x, 1 / y, 1 / z); }

        // References
        // Building an Orthonormal Basis, Revisited
        RT_FUNCTION void makeCoordinateSystem(Vector3DTemplate<RealType>* vx, Vector3DTemplate<RealType>* vy) const {
            RealType sign = z >= 0 ? 1 : -1;
            const RealType a = -1 / (sign + z);
            const RealType b = x * y * a;
            *vx = Vector3DTemplate<RealType>(1 + sign * x * x * a, sign * b, -sign * x);
            *vy = Vector3DTemplate<RealType>(b, sign + y * y * a, -y);
        }
        RT_FUNCTION static Vector3DTemplate fromPolarZUp(RealType phi, RealType theta) {
            RealType sinPhi, cosPhi;
            RealType sinTheta, cosTheta;
            VLR::sincos(phi, &sinPhi, &cosPhi);
            VLR::sincos(theta, &sinTheta, &cosTheta);
            return Vector3DTemplate(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
        }
        RT_FUNCTION static Vector3DTemplate fromPolarYUp(RealType phi, RealType theta) {
            RealType sinPhi, cosPhi;
            RealType sinTheta, cosTheta;
            VLR::sincos(phi, &sinPhi, &cosPhi);
            VLR::sincos(theta, &sinTheta, &cosTheta);
            return Vector3DTemplate(-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
        }
        RT_FUNCTION void toPolarZUp(RealType* theta, RealType* phi) const {
            *theta = std::acos(clamp(z, (RealType)-1, (RealType)1));
            *phi = std::fmod((RealType)(std::atan2(y, x) + 2 * VLR_M_PI), (RealType)(2 * VLR_M_PI));
        }
        RT_FUNCTION void toPolarYUp(RealType* theta, RealType* phi) const {
            *theta = std::acos(clamp(y, (RealType)-1, (RealType)1));
            *phi = std::fmod((RealType)(std::atan2(-x, z) + 2 * VLR_M_PI), (RealType)(2 * VLR_M_PI));
        }

        RT_FUNCTION RealType maxValue() const { return fmaxf(x, fmaxf(y, z)); }
        RT_FUNCTION RealType minValue() const { return fminf(x, fminf(y, z)); }
        RT_FUNCTION bool hasNaN() const { using std::isnan; return isnan(x) || isnan(y) || isnan(z); }
        RT_FUNCTION bool hasInf() const { using std::isinf; return isinf(x) || isinf(y) || isinf(z); }

        RT_FUNCTION static constexpr Vector3DTemplate Zero() { return Vector3DTemplate(0); }
        RT_FUNCTION static constexpr Vector3DTemplate Ex() { return Vector3DTemplate(1, 0, 0); }
        RT_FUNCTION static constexpr Vector3DTemplate Ey() { return Vector3DTemplate(0, 1, 0); }
        RT_FUNCTION static constexpr Vector3DTemplate Ez() { return Vector3DTemplate(0, 0, 1); }
    };



    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> normalize(const Vector3DTemplate<RealType> &v) {
        RealType l = v.length();
        return v / l;
    }

    template <typename RealType>
    RT_FUNCTION inline RealType dot(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> cross(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return Vector3DTemplate<RealType>(vec1.y * vec2.z - vec1.z * vec2.y,
                                          vec1.z * vec2.x - vec1.x * vec2.z,
                                          vec1.x * vec2.y - vec1.y * vec2.x);
    }

    template <typename RealType>
    RT_FUNCTION inline RealType absDot(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return std::fabs(vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> min(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        using std::fmin;
        return Vector3DTemplate<RealType>(fmin(vec1.x, vec2.x), fmin(vec1.y, vec2.y), fmin(vec1.z, vec2.z));
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> max(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        using std::fmax;
        return Vector3DTemplate<RealType>(fmax(vec1.x, vec2.x), fmax(vec1.y, vec2.y), fmax(vec1.z, vec2.z));
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> halfVector(const Vector3DTemplate<RealType> &vec1, const Vector3DTemplate<RealType> &vec2) {
        return normalize(vec1 + vec2);
    }



    template <typename RealType>
    struct Vector4DTemplate {
    public:
        RealType x, y, z, w;

#if defined(VLR_Device)
        RT_FUNCTION constexpr Vector4DTemplate() { }
#else
        constexpr Vector4DTemplate() : x(0), y(0), z(0), w(0) { }
#endif
        RT_FUNCTION constexpr Vector4DTemplate(RealType v) : x(v), y(v), z(v), w(v) { }
        RT_FUNCTION constexpr Vector4DTemplate(RealType xx, RealType yy, RealType zz, RealType ww) : x(xx), y(yy), z(zz), w(ww) { }
        RT_FUNCTION constexpr Vector4DTemplate(const Vector3DTemplate<RealType> &vec3, RealType ww) : x(vec3.x), y(vec3.y), z(vec3.z), w(ww) { }

        RT_FUNCTION Vector4DTemplate operator+() const { return *this; }
        RT_FUNCTION Vector4DTemplate operator-() const { return Vector4DTemplate(-x, -y, -z, -w); }

        RT_FUNCTION Vector4DTemplate operator+(const Vector4DTemplate &v) const { return Vector4DTemplate(x + v.x, y + v.y, z + v.z, w + v.w); }
        RT_FUNCTION Vector4DTemplate operator-(const Vector4DTemplate &v) const { return Vector4DTemplate(x - v.x, y - v.y, z - v.z, w - v.w); }
        RT_FUNCTION Vector4DTemplate operator*(RealType s) const { return Vector4DTemplate(x * s, y * s, z * s, w * s); }
        RT_FUNCTION Vector4DTemplate operator/(RealType s) const { RealType r = 1 / s; return Vector4DTemplate(x * r, y * r, z * r, w * r); }
        RT_FUNCTION friend inline Vector4DTemplate operator*(RealType s, const Vector4DTemplate &v) { return Vector4DTemplate(s * v.x, s * v.y, s * v.z, s * v.w); }

        RT_FUNCTION Vector4DTemplate &operator+=(const Vector4DTemplate &v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
        RT_FUNCTION Vector4DTemplate &operator-=(const Vector4DTemplate &v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
        RT_FUNCTION Vector4DTemplate &operator*=(RealType s) { x *= s; y *= s; z *= s; w *= s; return *this; }
        RT_FUNCTION Vector4DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; w *= r; return *this; }

        RT_FUNCTION bool operator==(const Vector4DTemplate &v) const { return x == v.x && y == v.y && z == v.z && w == v.w; }
        RT_FUNCTION bool operator!=(const Vector4DTemplate &v) const { return x != v.x || y != v.y || z != v.z || w != v.w; }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < 4, "\"index\" is out of range [0, 3].");
            return *(&x + index);
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < 4, "\"index\" is out of range [0, 3].");
            return *(&x + index);
        }

        RT_FUNCTION explicit operator Vector3DTemplate<RealType>() const { return Vector3DTemplate<RealType>(x, y, z); }

        RT_FUNCTION RealType maxValue() const { using std::fmax; return fmax(fmax(x, y), fmax(z, w)); }
        RT_FUNCTION RealType minValue() const { using std::fmin; return fmin(fmin(x, y), fmin(z, w)); }
        RT_FUNCTION bool hasNaN() const { using std::isnan; return isnan(x) || isnan(y) || isnan(z) || isnan(w); }
        RT_FUNCTION bool hasInf() const { using std::isinf; return isinf(x) || isinf(y) || isinf(z) || isinf(w); }
    };


    template <typename RealType>
    RT_FUNCTION inline RealType dot(const Vector4DTemplate<RealType> &vec1, const Vector4DTemplate<RealType> &vec2) {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z + vec1.w * vec2.w;
    }

    template <typename RealType>
    RT_FUNCTION inline Vector4DTemplate<RealType> min(const Vector4DTemplate<RealType> &vec1, const Vector4DTemplate<RealType> &vec2) {
        using std::fmin;
        return Vector4DTemplate<RealType>(fmin(vec1.x, vec2.x), fmin(vec1.y, vec2.y), fmin(vec1.z, vec2.z), fmin(vec1.w, vec2.w));
    }

    template <typename RealType>
    RT_FUNCTION inline Vector4DTemplate<RealType> max(const Vector4DTemplate<RealType> &vec1, const Vector4DTemplate<RealType> &vec2) {
        using std::fmax;
        return Vector4DTemplate<RealType>(fmax(vec1.x, vec2.x), fmax(vec1.y, vec2.y), fmax(vec1.z, vec2.z), fmax(vec1.w, vec2.w));
    }



    template <typename RealType>
    struct Normal3DTemplate {
        RealType x, y, z;

#if defined(VLR_Device)
        RT_FUNCTION constexpr Normal3DTemplate() { }
#else
        constexpr Normal3DTemplate() : x(0), y(0), z(0) { }
#endif
        RT_FUNCTION constexpr Normal3DTemplate(RealType v) : x(v), y(v), z(v) { }
        RT_FUNCTION constexpr Normal3DTemplate(RealType xx, RealType yy, RealType zz) : x(xx), y(yy), z(zz) { }
        RT_FUNCTION constexpr Normal3DTemplate(const Vector3DTemplate<RealType> &v) : x(v.x), y(v.y), z(v.z) { }

        RT_FUNCTION operator Vector3DTemplate<RealType>() const {
            return Vector3DTemplate<RealType>(x, y, z);
        }

        RT_FUNCTION Normal3DTemplate operator+() const { return *this; }
        RT_FUNCTION Normal3DTemplate operator-() const { return Normal3DTemplate(-x, -y, -z); }

        RT_FUNCTION Vector3DTemplate<RealType> operator+(const Vector3DTemplate<RealType> &v) const { return Vector3DTemplate<RealType>(x + v.x, y + v.y, z + v.z); }
        RT_FUNCTION Vector3DTemplate<RealType> operator-(const Vector3DTemplate<RealType> &v) const { return Vector3DTemplate<RealType>(x - v.x, y - v.y, z - v.z); }
        RT_FUNCTION Vector3DTemplate<RealType> operator+(const Normal3DTemplate &n) const { return Vector3DTemplate<RealType>(x + n.x, y + n.y, z + n.z); }
        RT_FUNCTION Vector3DTemplate<RealType> operator-(const Normal3DTemplate &n) const { return Vector3DTemplate<RealType>(x - n.x, y - n.y, z - n.z); }
        RT_FUNCTION Vector3DTemplate<RealType> operator*(RealType s) const { return Vector3DTemplate<RealType>(x * s, y * s, z * s); }
        RT_FUNCTION Vector3DTemplate<RealType> operator/(RealType s) const { RealType r = 1 / s; return Vector3DTemplate<RealType>(x * r, y * r, z * r); }
        RT_FUNCTION friend inline Vector3DTemplate<RealType> operator*(RealType s, const Normal3DTemplate &n) { return Vector3DTemplate<RealType>(s * n.x, s * n.y, s * n.z); }

        RT_FUNCTION Normal3DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; return *this; }

        RT_FUNCTION bool operator==(const Normal3DTemplate &p) const { return x == p.x && y == p.y && z == p.z; }
        RT_FUNCTION bool operator!=(const Normal3DTemplate &p) const { return x != p.x || y != p.y || z != p.z; }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }

        RT_FUNCTION RealType length() const { return std::sqrt(x * x + y * y + z * z); }
        RT_FUNCTION RealType sqLength() const { return x * x + y * y + z * z; }
        RT_FUNCTION Normal3DTemplate& normalize() {
            RealType rcpLength = RealType(1.0) / std::sqrt(x * x + y * y + z * z);
            x *= rcpLength;
            y *= rcpLength;
            z *= rcpLength;
            return *this;
        }

        // References
        // Building an Orthonormal Basis, Revisited
        RT_FUNCTION void makeCoordinateSystem(Vector3DTemplate<RealType>* tangent, Vector3DTemplate<RealType>* bitangent) const {
            RealType sign = z >= 0 ? 1 : -1;
            const RealType a = -1 / (sign + z);
            const RealType b = x * y * a;
            *tangent = Vector3DTemplate<RealType>(1 + sign * x * x * a, sign * b, -sign * x);
            *bitangent = Vector3DTemplate<RealType>(b, sign + y * y * a, -y);
        }
        RT_FUNCTION static Normal3DTemplate fromPolarZUp(RealType phi, RealType theta) {
            RealType sinPhi, cosPhi;
            RealType sinTheta, cosTheta;
            VLR::sincos(phi, &sinPhi, &cosPhi);
            VLR::sincos(theta, &sinTheta, &cosTheta);
            return Normal3DTemplate(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
        }
        RT_FUNCTION static Normal3DTemplate fromPolarYUp(RealType phi, RealType theta) {
            RealType sinPhi, cosPhi;
            RealType sinTheta, cosTheta;
            VLR::sincos(phi, &sinPhi, &cosPhi);
            VLR::sincos(theta, &sinTheta, &cosTheta);
            return Normal3DTemplate(-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
        }
        RT_FUNCTION void toPolarZUp(RealType* theta, RealType* phi) const {
            *theta = std::acos(clamp(z, (RealType)-1, (RealType)1));
            *phi = std::fmod((RealType)(std::atan2(y, x) + 2 * VLR_M_PI), (RealType)(2 * VLR_M_PI));
        }
        RT_FUNCTION void toPolarYUp(RealType* theta, RealType* phi) const {
            *theta = std::acos(clamp(y, (RealType)-1, (RealType)1));
            *phi = std::fmod((RealType)(std::atan2(-x, z) + 2 * VLR_M_PI), (RealType)(2 * VLR_M_PI));
        }

        RT_FUNCTION RealType maxValue() const { using std::fmax; return fmax(x, fmax(y, z)); }
        RT_FUNCTION RealType minValue() const { using std::fmin; return fmin(x, fmin(y, z)); }
        RT_FUNCTION bool hasNaN() const { using std::isnan; return isnan(x) || isnan(y) || isnan(z); }
        RT_FUNCTION bool hasInf() const { using std::isinf; return isinf(x) || isinf(y) || isinf(z); }
    };



    template <typename RealType>
    RT_FUNCTION inline Normal3DTemplate<RealType> normalize(const Normal3DTemplate<RealType> &n) {
        RealType l = n.length();
        return n / l;
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> cross(const Vector3DTemplate<RealType> &vec, const Normal3DTemplate<RealType> &norm) {
        return Vector3DTemplate<RealType>(vec.y * norm.z - vec.z * norm.y,
                                          vec.z * norm.x - vec.x * norm.z,
                                          vec.x * norm.y - vec.y * norm.x);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> cross(const Normal3DTemplate<RealType> &norm, const Vector3DTemplate<RealType> &vec) {
        return Vector3DTemplate<RealType>(norm.y * vec.z - norm.z * vec.y,
                                          norm.z * vec.x - norm.x * vec.z,
                                          norm.x * vec.y - norm.y * vec.x);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> cross(const Normal3DTemplate<RealType> &n1, const Normal3DTemplate<RealType> &n2) {
        return Vector3DTemplate<RealType>(n1.y * n2.z - n1.z * n2.y,
                                          n1.z * n2.x - n1.x * n2.z,
                                          n1.x * n2.y - n1.y * n2.x);
    }

    template <typename RealType>
    RT_FUNCTION inline RealType dot(const Vector3DTemplate<RealType> &vec, const Normal3DTemplate<RealType> &norm) {
        return vec.x * norm.x + vec.y * norm.y + vec.z * norm.z;
    }

    template <typename RealType>
    RT_FUNCTION inline RealType dot(const Normal3DTemplate<RealType> &norm, const Vector3DTemplate<RealType> &vec) {
        return vec.x * norm.x + vec.y * norm.y + vec.z * norm.z;
    }

    template <typename RealType>
    RT_FUNCTION inline RealType absDot(const Normal3DTemplate<RealType> &norm1, const Normal3DTemplate<RealType> &norm2) {
        return std::fabs(norm1.x * norm2.x + norm1.y * norm2.y + norm1.z * norm2.z);
    }

    template <typename RealType>
    RT_FUNCTION inline RealType absDot(const Vector3DTemplate<RealType> &vec, const Normal3DTemplate<RealType> &norm) {
        return std::fabs(vec.x * norm.x + vec.y * norm.y + vec.z * norm.z);
    }

    template <typename RealType>
    RT_FUNCTION inline RealType absDot(const Normal3DTemplate<RealType> &norm, const Vector3DTemplate<RealType> &vec) {
        return std::fabs(vec.x * norm.x + vec.y * norm.y + vec.z * norm.z);
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> min(const Normal3DTemplate<RealType> &p1, const Normal3DTemplate<RealType> &p2) {
        using std::fmin;
        return Vector3DTemplate<RealType>(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
    }

    template <typename RealType>
    RT_FUNCTION inline Vector3DTemplate<RealType> max(const Normal3DTemplate<RealType> &p1, const Normal3DTemplate<RealType> &p2) {
        using std::fmax;
        return Vector3DTemplate<RealType>(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }



    template <typename RealType>
    struct Point3DTemplate {
        RealType x, y, z;

#if defined(VLR_Device)
        RT_FUNCTION constexpr Point3DTemplate() { }
#else
        constexpr Point3DTemplate() : x(0), y(0), z(0) { }
#endif
        RT_FUNCTION constexpr Point3DTemplate(RealType v) : x(v), y(v), z(v) { }
        RT_FUNCTION constexpr Point3DTemplate(RealType xx, RealType yy, RealType zz) : x(xx), y(yy), z(zz) { }
        RT_FUNCTION constexpr Point3DTemplate(const Vector3DTemplate<RealType> &v) : x(v.x), y(v.y), z(v.z) { }

        RT_FUNCTION operator Vector3DTemplate<RealType>() const {
            return Vector3DTemplate<RealType>(x, y, z);
        }

        RT_FUNCTION Point3DTemplate operator+() const { return *this; }
        RT_FUNCTION Point3DTemplate operator-() const { return Point3DTemplate(-x, -y, -z); }

        RT_FUNCTION Point3DTemplate operator+(const Vector3DTemplate<RealType> &v) const { return Point3DTemplate(x + v.x, y + v.y, z + v.z); }
        RT_FUNCTION Point3DTemplate operator-(const Vector3DTemplate<RealType> &v) const { return Point3DTemplate(x - v.x, y - v.y, z - v.z); }
        RT_FUNCTION Point3DTemplate<RealType> operator+(const Point3DTemplate &p) const { return Point3DTemplate<RealType>(x + p.x, y + p.y, z + p.z); }
        RT_FUNCTION Vector3DTemplate<RealType> operator-(const Point3DTemplate &p) const { return Vector3DTemplate<RealType>(x - p.x, y - p.y, z - p.z); }
        RT_FUNCTION Point3DTemplate operator*(RealType s) const { return Point3DTemplate(x * s, y * s, z * s); }
        RT_FUNCTION Point3DTemplate operator/(RealType s) const { RealType r = 1 / s; return Point3DTemplate(x * r, y * r, z * r); }
        RT_FUNCTION friend inline Point3DTemplate operator+(const Vector3DTemplate<RealType> &v, const Point3DTemplate &p) { return Point3DTemplate(p.x + v.x, p.y + v.y, p.z + v.z); }
        RT_FUNCTION friend inline Point3DTemplate operator*(RealType s, const Point3DTemplate &p) { return Point3DTemplate(s * p.x, s * p.y, s * p.z); }

        RT_FUNCTION Point3DTemplate &operator+=(const Vector3DTemplate<RealType> &v) { x += v.x; y += v.y; z += v.z; return *this; }
        RT_FUNCTION Point3DTemplate &operator-=(const Vector3DTemplate<RealType> &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
        RT_FUNCTION Point3DTemplate &operator*=(RealType s) { x *= s; y *= s; z *= s; return *this; }
        RT_FUNCTION Point3DTemplate &operator/=(RealType s) { RealType r = 1 / s; x *= r; y *= r; z *= r; return *this; }

        RT_FUNCTION bool operator==(const Point3DTemplate &p) const { return x == p.x && y == p.y && z == p.z; }
        RT_FUNCTION bool operator!=(const Point3DTemplate &p) const { return x != p.x || y != p.y || z != p.z; }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < 3, "\"index\" is out of range [0, 2].");
            return *(&x + index);
        }

        RT_FUNCTION RealType maxValue() const { return std::fmax(x, std::fmax(y, z)); }
        RT_FUNCTION RealType minValue() const { return std::fmin(x, std::fmin(y, z)); }
        RT_FUNCTION bool hasNaN() const { using std::isnan; return isnan(x) || isnan(y) || isnan(z); }
        RT_FUNCTION bool hasInf() const { using std::isinf; return isinf(x) || isinf(y) || isinf(z); }

        RT_FUNCTION static constexpr Point3DTemplate Zero() { return Point3DTemplate(0); }
    };



    template <typename RealType>
    RT_FUNCTION inline RealType absDot(const Point3DTemplate<RealType> &p1, const Point3DTemplate<RealType> &p2) {
        return std::fabs(p1.x * p2.x + p1.y * p2.y + p1.z * p2.z);
    }

    template <typename RealType>
    RT_FUNCTION inline Point3DTemplate<RealType> min(const Point3DTemplate<RealType> &p1, const Point3DTemplate<RealType> &p2) {
        using std::fmin;
        return Point3DTemplate<RealType>(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
    }

    template <typename RealType>
    RT_FUNCTION inline Point3DTemplate<RealType> max(const Point3DTemplate<RealType> &p1, const Point3DTemplate<RealType> &p2) {
        using std::fmax;
        return Point3DTemplate<RealType>(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }

    template <typename RealType>
    RT_FUNCTION inline Point3DTemplate<RealType> clamp(const Point3DTemplate<RealType> &p, const Point3DTemplate<RealType> &minP, const Point3DTemplate<RealType> &maxP) {
        return max(min(p, maxP), minP);
    }

    template <typename RealType>
    RT_FUNCTION inline RealType sqDistance(const Point3DTemplate<RealType> &p1, const Point3DTemplate<RealType> &p2) {
        return (p2 - p1).sqLength();
    }

    template <typename RealType>
    RT_FUNCTION inline RealType distance(const Point3DTemplate<RealType> &p1, const Point3DTemplate<RealType> &p2) {
        return (p2 - p1).length();
    }



    template <typename RealType>
    struct TexCoord2DTemplate {
        RealType u, v;

#if defined(VLR_Device)
        RT_FUNCTION constexpr TexCoord2DTemplate() { }
#else
        constexpr TexCoord2DTemplate() : u(0), v(0) { }
#endif
        RT_FUNCTION constexpr TexCoord2DTemplate(RealType val) : u(val), v(val) { }
        RT_FUNCTION constexpr TexCoord2DTemplate(RealType uu, RealType vv) : u(uu), v(vv) { }

        RT_FUNCTION TexCoord2DTemplate operator+() const { return *this; }
        RT_FUNCTION TexCoord2DTemplate operator-() const { return TexCoord2DTemplate(-u, -v); }

        RT_FUNCTION TexCoord2DTemplate operator+(const TexCoord2DTemplate<RealType> &t) const { return TexCoord2DTemplate(u + t.u, v + t.v); }
        RT_FUNCTION TexCoord2DTemplate operator-(const TexCoord2DTemplate<RealType> &t) const { return TexCoord2DTemplate(u - t.u, v - t.v); }
        RT_FUNCTION TexCoord2DTemplate operator*(RealType s) const { return TexCoord2DTemplate(u * s, v * s); }
        RT_FUNCTION TexCoord2DTemplate operator/(RealType s) const { RealType r = 1 / s; return TexCoord2DTemplate(u * r, v * r); }
        RT_FUNCTION friend inline TexCoord2DTemplate operator*(RealType s, const TexCoord2DTemplate &t) { return TexCoord2DTemplate(s * t.u, s * t.v); }

        RT_FUNCTION TexCoord2DTemplate &operator+=(const TexCoord2DTemplate<RealType> &t) { u += t.u; v += t.v; return *this; }
        RT_FUNCTION TexCoord2DTemplate &operator-=(const TexCoord2DTemplate<RealType> &t) { u -= t.u; v -= t.v; return *this; }
        RT_FUNCTION TexCoord2DTemplate &operator*=(RealType s) { u *= s; v *= s; return *this; }
        RT_FUNCTION TexCoord2DTemplate &operator/=(RealType s) { RealType r = 1 / s; u *= r; v *= r; return *this; }

        RT_FUNCTION bool operator==(const TexCoord2DTemplate &t) const { return u == t.u && v == t.v; }
        RT_FUNCTION bool operator!=(const TexCoord2DTemplate &t) const { return u != t.u || v != t.v; }

        RT_FUNCTION RealType &operator[](unsigned int index) {
            VLRAssert(index < 2, "\"index\" is out of range [0, 1].");
            return *(&u + index);
        }
        RT_FUNCTION RealType operator[](unsigned int index) const {
            VLRAssert(index < 2, "\"index\" is out of range [0, 1].");
            return *(&u + index);
        }

        RT_FUNCTION RealType maxValue() const { return std::fmax(u, v); }
        RT_FUNCTION RealType minValue() const { return std::fmin(u, v); }
        RT_FUNCTION bool hasNaN() const { using std::isnan; return isnan(u) || isnan(v); }
        RT_FUNCTION bool hasInf() const { using std::isinf; return isinf(u) || isinf(v); }

        RT_FUNCTION static constexpr TexCoord2DTemplate Zero() { return TexCoord2DTemplate(0); }
    };



    template <typename RealType>
    struct BoundingBox3DTemplate {
        enum Axis : uint8_t {
            Axis_X = 0,
            Axis_Y,
            Axis_Z,
        };

        Point3DTemplate<RealType> minP, maxP;

#if defined(VLR_Device)
        RT_FUNCTION constexpr BoundingBox3DTemplate() { }
#else
        constexpr BoundingBox3DTemplate() : minP(INFINITY), maxP(-INFINITY) { }
#endif
        RT_FUNCTION constexpr BoundingBox3DTemplate(const Point3DTemplate<RealType> &p) : minP(p), maxP(p) { }
        RT_FUNCTION constexpr BoundingBox3DTemplate(const Point3DTemplate<RealType> &pmin, const Point3DTemplate<RealType> &pmax) : minP(pmin), maxP(pmax) { }

        RT_FUNCTION Point3DTemplate<RealType> centroid() const {
            return (minP + maxP) * (RealType)0.5;
        }

        RT_FUNCTION RealType surfaceArea() const {
            Vector3DTemplate<RealType> d = maxP - minP;
            return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
        }

        RT_FUNCTION RealType volume() const {
            Vector3DTemplate<RealType> d = maxP - minP;
            return d.x * d.y * d.z;
        }

        RT_FUNCTION Point3DTemplate<RealType> corner(uint32_t c) const {
            VLRAssert(c < 8, "\"c\" is out of range [0, 8]");
            const size_t offset = sizeof(Point3DTemplate<RealType>);
            return Point3DTemplate<RealType>(*(&minP.x + offset * (c & 0x01)),
                                             *(&minP.y + offset * (c & 0x02)),
                                             *(&minP.z + offset * (c & 0x04)));
        }

        RT_FUNCTION RealType centerOfAxis(Axis axis) const {
            return (minP[axis] + maxP[axis]) * (RealType)0.5;
        }

        RT_FUNCTION RealType width(Axis axis) const {
            return maxP[axis] - minP[axis];
        }

        RT_FUNCTION Axis widestAxis() const {
            Vector3DTemplate<RealType> d = maxP - minP;
            if (d.x > d.y && d.x > d.z)
                return Axis_X;
            else if (d.y > d.z)
                return Axis_Y;
            else
                return Axis_Z;
        }

        RT_FUNCTION bool isValid() const {
            Vector3DTemplate<RealType> d = maxP - minP;
            return d.x >= 0 && d.y >= 0 && d.z >= 0;
        }

        RT_FUNCTION BoundingBox3DTemplate &unify(const Point3DTemplate<RealType> &p) {
            minP = min(minP, p);
            maxP = max(maxP, p);
            return *this;
        }

        RT_FUNCTION BoundingBox3DTemplate unify(const BoundingBox3DTemplate &b) {
            minP = min(minP, b.minP);
            maxP = max(maxP, b.maxP);
            return *this;
        }

        RT_FUNCTION bool contains(const Point3DTemplate<RealType> &p) const {
            return ((p.x >= minP.x && p.x < maxP.x) &&
                (p.y >= minP.y && p.y < maxP.y) &&
                    (p.z >= minP.z && p.z < maxP.z));
        }

        RT_FUNCTION void calculateLocalCoordinates(const Point3DTemplate<RealType> &p, Point3DTemplate<RealType>* param) const {
            *param = (p - minP) / (maxP - minP);
        }

        RT_FUNCTION bool hasNaN() const {
            return minP.hasNaN() || maxP.hasNaN();
        }

        RT_FUNCTION bool hasInf() const {
            return minP.hasInf() || maxP.hasInf();
        }
    };



    template <typename RealType>
    RT_FUNCTION inline BoundingBox3DTemplate<RealType> calcUnion(const BoundingBox3DTemplate<RealType> &b0, const BoundingBox3DTemplate<RealType> &b1) {
        return BoundingBox3DTemplate<RealType>(min(b0.minP, b1.minP), max(b0.maxP, b1.maxP));
    }

    template <typename RealType>
    RT_FUNCTION inline BoundingBox3DTemplate<RealType> intersection(const BoundingBox3DTemplate<RealType> &b0, const BoundingBox3DTemplate<RealType> &b1) {
        return BoundingBox3DTemplate<RealType>(max(b0.minP, b1.minP), min(b0.maxP, b1.maxP));
    }



    template <typename RealType>
    struct Matrix3x3Template {
        union {
            struct { RealType m00, m10, m20; };
            Vector3DTemplate<RealType> c0;
        };
        union {
            struct { RealType m01, m11, m21; };
            Vector3DTemplate<RealType> c1;
        };
        union {
            struct { RealType m02, m12, m22; };
            Vector3DTemplate<RealType> c2;
        };

#if defined(VLR_Device)
        RT_FUNCTION constexpr Matrix3x3Template() { }
#else
        constexpr Matrix3x3Template() : c0(), c1(), c2() { }
#endif
        RT_FUNCTION constexpr Matrix3x3Template(const RealType array[9]) :
            m00(array[0]), m10(array[1]), m20(array[2]),
            m01(array[3]), m11(array[4]), m21(array[5]),
            m02(array[6]), m12(array[7]), m22(array[8]) { }
        RT_FUNCTION constexpr Matrix3x3Template(const Vector3DTemplate<RealType> &col0, const Vector3DTemplate<RealType> &col1, const Vector3DTemplate<RealType> &col2) :
            c0(col0), c1(col1), c2(col2)
        { }

        RT_FUNCTION Matrix3x3Template operator+() const { return *this; }
        RT_FUNCTION Matrix3x3Template operator-() const { return Matrix3x3Template(-c0, -c1, -c2); }

        RT_FUNCTION Matrix3x3Template operator+(const Matrix3x3Template &mat) const { return Matrix3x3Template(c0 + mat.c0, c1 + mat.c1, c2 + mat.c2); }
        RT_FUNCTION Matrix3x3Template operator-(const Matrix3x3Template &mat) const { return Matrix3x3Template(c0 - mat.c0, c1 - mat.c1, c2 - mat.c2); }
        RT_FUNCTION Matrix3x3Template operator*(const Matrix3x3Template &mat) const {
            const Vector3DTemplate<RealType> r[] = { row(0), row(1), row(2) };
            return Matrix3x3Template(Vector3DTemplate<RealType>(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0)),
                                     Vector3DTemplate<RealType>(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1)),
                                     Vector3DTemplate<RealType>(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2)));
        }
        RT_FUNCTION Vector3DTemplate<RealType> operator*(const Vector3DTemplate<RealType> &v) const {
            return Vector3DTemplate<RealType>(dot(row(0), v), dot(row(1), v), dot(row(2), v));
        }
        RT_FUNCTION Point3DTemplate<RealType> operator*(const Point3DTemplate<RealType> &p) const {
            Vector3DTemplate<RealType> ph{ p.x, p.y, p.z };
            Vector3DTemplate<RealType> pht = Vector3DTemplate<RealType>(dot(row(0), ph), dot(row(1), ph), dot(row(2), ph));
            return Point3DTemplate<RealType>(pht.x, pht.y, pht.z);
        }
        RT_FUNCTION Matrix3x3Template operator*(RealType s) const { return Matrix3x3Template(c0 * s, c1 * s, c2 * s); }
        RT_FUNCTION Matrix3x3Template operator/(RealType s) const { return Matrix3x3Template(c0 / s, c1 / s, c2 / s); }
        RT_FUNCTION friend inline Matrix3x3Template operator*(RealType s, const Matrix3x3Template &mat) { return Matrix3x3Template(s * mat.c0, s * mat.c1, s * mat.c2); }

        RT_FUNCTION Matrix3x3Template &operator+=(const Matrix3x3Template &mat) { c0 += mat.c0; c1 += mat.c1; c2 += mat.c2; return *this; }
        RT_FUNCTION Matrix3x3Template &operator-=(const Matrix3x3Template &mat) { c0 -= mat.c0; c1 -= mat.c1; c2 -= mat.c2; return *this; }
        RT_FUNCTION Matrix3x3Template &operator*=(const Matrix3x3Template &mat) {
            const Vector3DTemplate<RealType> r[] = { row(0), row(1), row(2) };
            c0 = Vector3DTemplate<RealType>(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0));
            c1 = Vector3DTemplate<RealType>(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1));
            c2 = Vector3DTemplate<RealType>(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2));
            return *this;
        }
        RT_FUNCTION Matrix3x3Template &operator*=(RealType s) { c0 *= s; c1 *= s; c2 *= s; return *this; }
        RT_FUNCTION Matrix3x3Template &operator/=(RealType s) { c0 /= s; c1 /= s; c2 /= s; return *this; }

        RT_FUNCTION bool operator==(const Matrix3x3Template &m) const { return c0 == m.c0 && c1 == m.c1 && c2 == m.c2; }
        RT_FUNCTION bool operator!=(const Matrix3x3Template &m) const { return c0 == m.c0 || c1 != m.c1 || c2 != m.c2; }

        RT_FUNCTION Vector3DTemplate<RealType> &operator[](unsigned int c) {
            VLRAssert(c < 3, "\"c\" is out of range [0, 2].");
            return *(&c0 + c);
        }

        RT_FUNCTION Vector3DTemplate<RealType> operator[](unsigned int c) const {
            VLRAssert(c < 3, "\"c\" is out of range [0, 2].");
            return *(&c0 + c);
        }

        RT_FUNCTION const Vector3DTemplate<RealType> &column(unsigned int c) const {
            VLRAssert(c < 3, "\"c\" is out of range [0, 2].");
            return *(&c0 + c);
        }
        RT_FUNCTION Vector3DTemplate<RealType> row(unsigned int r) const {
            VLRAssert(r < 3, "\"r\" is out of range [0, 2].");
            switch (r) {
            case 0:
                return Vector3DTemplate<RealType>(m00, m01, m02);
            case 1:
                return Vector3DTemplate<RealType>(m10, m11, m12);
            case 2:
                return Vector3DTemplate<RealType>(m20, m21, m22);
            default:
                return Vector3DTemplate<RealType>(0, 0, 0);
            }
        }

        RT_FUNCTION Matrix3x3Template &swapColumns(unsigned int ca, unsigned int cb) {
            if (ca != cb) {
                Vector3DTemplate<RealType> temp = column(ca);
                (*this)[ca] = (*this)[cb];
                (*this)[cb] = temp;
            }
            return *this;
        }

        RT_FUNCTION Matrix3x3Template &swapRows(unsigned int ra, unsigned int rb) {
            if (ra != rb) {
                Vector3DTemplate<RealType> temp = row(ra);
                setRow(ra, row(rb));
                setRow(rb, temp);
            }
            return *this;
        }

        RT_FUNCTION Matrix3x3Template &setRow(unsigned int r, const Vector3DTemplate<RealType> &v) {
            VLRAssert(r < 3, "\"r\" is out of range [0, 2].");
            c0[r] = v[0]; c1[r] = v[1]; c2[r] = v[2];
            return *this;
        }
        RT_FUNCTION Matrix3x3Template &scaleRow(unsigned int r, RealType s) {
            VLRAssert(r < 3, "\"r\" is out of range [0, 2].");
            c0[r] *= s; c1[r] *= s; c2[r] *= s;
            return *this;
        }
        RT_FUNCTION Matrix3x3Template &addRow(unsigned int r, const Vector3DTemplate<RealType> &v) {
            VLRAssert(r < 3, "\"r\" is out of range [0, 2].");
            c0[r] += v[0]; c1[r] += v[1]; c2[r] += v[2];
            return *this;
        }

        RealType determinant() const {
            return (c0[0] * (c1[1] * c2[2] - c2[1] * c1[2]) -
                    c1[0] * (c0[1] * c2[2] - c2[1] * c0[2]) +
                    c2[0] * (c0[1] * c1[2] - c1[1] * c0[2]));
        }

        Matrix3x3Template& transpose() {
            std::swap(m10, m01); std::swap(m20, m02);
            std::swap(m21, m12);
            return *this;
        }

        RT_FUNCTION Matrix3x3Template &invert() {
            VLRAssert_NotImplemented();
            return *this;
        }

        RT_FUNCTION bool isIdentity() const {
            typedef Vector3DTemplate<RealType> V3;
            return c0 == V3(1, 0, 0) && c1 == V3(0, 1, 0) && c2 == V3(0, 0, 1);
        }
        RT_FUNCTION bool hasNaN() const { return c0.hasNaN() || c1.hasNaN() || c2.hasNaN(); }
        RT_FUNCTION bool hasInf() const { return c0.hasInf() || c1.hasInf() || c2.hasInf(); }

        RT_FUNCTION static constexpr Matrix3x3Template Identity() {
            RealType data[] = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            };
            return Matrix3x3Template(data);
        }
    };



    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> transpose(const Matrix3x3Template<RealType> &m) {
        return Matrix3x3Template<RealType>(Vector3DTemplate<RealType>(m.m00, m.m01, m.m02),
                                           Vector3DTemplate<RealType>(m.m10, m.m11, m.m12),
                                           Vector3DTemplate<RealType>(m.m20, m.m21, m.m22));
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> invert(const Matrix3x3Template<RealType> &m) {
        VLRAssert_NotImplemented();
        Matrix3x3Template<RealType> mat;

        return mat;
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> scale3x3(const Vector3DTemplate<RealType> &s) {
        return Matrix3x3Template<RealType>(s.x * Vector3DTemplate<RealType>(1, 0, 0),
                                           s.y * Vector3DTemplate<RealType>(0, 1, 0),
                                           s.z * Vector3DTemplate<RealType>(0, 0, 1));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> scale3x3(RealType sx, RealType sy, RealType sz) {
        return scale3x3(Vector3DTemplate<RealType>(sx, sy, sz));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> scale3x3(RealType s) {
        return scale3x3(Vector3DTemplate<RealType>(s, s, s));
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> rotate3x3(RealType angle, const Vector3DTemplate<RealType> &axis) {
        Matrix3x3Template<RealType> matrix;
        Vector3DTemplate<RealType> nAxis = normalize(axis);
        RealType s, c;
        VLR::sincos(angle, &s, &c);
        RealType oneMinusC = 1 - c;

        matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
        matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
        matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
        matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
        matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
        matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
        matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
        matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
        matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

        return matrix;
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> rotate3x3(RealType angle, RealType ax, RealType ay, RealType az) {
        return rotate3x3(angle, Vector3DTemplate<RealType>(ax, ay, az));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> rotateX3x3(RealType angle) { return rotate3x3(angle, Vector3DTemplate<RealType>(1, 0, 0)); }
    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> rotateY3x3(RealType angle) { return rotate3x3(angle, Vector3DTemplate<RealType>(0, 1, 0)); }
    template <typename RealType>
    RT_FUNCTION inline Matrix3x3Template<RealType> rotateZ3x3(RealType angle) { return rotate3x3(angle, Vector3DTemplate<RealType>(0, 0, 1)); }



    template <typename RealType>
    struct Matrix4x4Template {
        union {
            struct { RealType m00, m10, m20, m30; };
            Vector4DTemplate<RealType> c0;
        };
        union {
            struct { RealType m01, m11, m21, m31; };
            Vector4DTemplate<RealType> c1;
        };
        union {
            struct { RealType m02, m12, m22, m32; };
            Vector4DTemplate<RealType> c2;
        };
        union {
            struct { RealType m03, m13, m23, m33; };
            Vector4DTemplate<RealType> c3;
        };

#if defined(VLR_Device)
        RT_FUNCTION constexpr Matrix4x4Template() { }
#else
        constexpr Matrix4x4Template() : c0(), c1(), c2(), c3() { }
#endif
        RT_FUNCTION constexpr Matrix4x4Template(const RealType array[16]) :
            m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
            m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
            m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
            m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) { }
        RT_FUNCTION constexpr Matrix4x4Template(const Vector3DTemplate<RealType> &col0, const Vector3DTemplate<RealType> &col1, const Vector3DTemplate<RealType> &col2) : c0(col0, 0), c1(col1, 0), c2(col2, 0), c3(Vector4DTemplate<RealType>(0, 0, 0, 1)) { }
        RT_FUNCTION constexpr Matrix4x4Template(const Vector4DTemplate<RealType> &col0, const Vector4DTemplate<RealType> &col1, const Vector4DTemplate<RealType> &col2, const Vector4DTemplate<RealType> &col3) :
            c0(col0), c1(col1), c2(col2), c3(col3)
        { }

        RT_FUNCTION Matrix4x4Template operator+() const { return *this; }
        RT_FUNCTION Matrix4x4Template operator-() const { return Matrix4x4Template(-c0, -c1, -c2, -c3); }

        RT_FUNCTION Matrix4x4Template operator+(const Matrix4x4Template &mat) const { return Matrix4x4Template(c0 + mat.c0, c1 + mat.c1, c2 + mat.c2, c3 + mat.c3); }
        RT_FUNCTION Matrix4x4Template operator-(const Matrix4x4Template &mat) const { return Matrix4x4Template(c0 - mat.c0, c1 - mat.c1, c2 - mat.c2, c3 - mat.c3); }
        RT_FUNCTION Matrix4x4Template operator*(const Matrix4x4Template &mat) const {
            const Vector4DTemplate<RealType> r[] = { row(0), row(1), row(2), row(3) };
            return Matrix4x4Template(Vector4DTemplate<RealType>(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0)),
                                     Vector4DTemplate<RealType>(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1)),
                                     Vector4DTemplate<RealType>(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2)),
                                     Vector4DTemplate<RealType>(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3)));
        }
        RT_FUNCTION Vector3DTemplate<RealType> operator*(const Vector3DTemplate<RealType> &v) const {
            return Vector3DTemplate<RealType>(dot((Vector3DTemplate<RealType>)row(0), v), dot((Vector3DTemplate<RealType>)row(1), v), dot((Vector3DTemplate<RealType>)row(2), v));
        }
        RT_FUNCTION Vector4DTemplate<RealType> operator*(const Vector4DTemplate<RealType> &v) const { return Vector4DTemplate<RealType>(dot(row(0), v), dot(row(1), v), dot(row(2), v), dot(row(3), v)); }
        RT_FUNCTION Point3DTemplate<RealType> operator*(const Point3DTemplate<RealType> &p) const {
            Vector4DTemplate<RealType> ph{ p.x, p.y, p.z, 1 };
            Vector4DTemplate<RealType> pht = Vector4DTemplate<RealType>(dot(row(0), ph), dot(row(1), ph), dot(row(2), ph), dot(row(3), ph));
            if (pht.w != 1)
                pht /= pht.w;
            return Point3DTemplate<RealType>(pht.x, pht.y, pht.z);
        }
        RT_FUNCTION Matrix4x4Template operator*(RealType s) const { return Matrix4x4Template(c0 * s, c1 * s, c2 * s, c3 * s); }
        RT_FUNCTION Matrix4x4Template operator/(RealType s) const { return Matrix4x4Template(c0 / s, c1 / s, c2 / s, c3 / s); }
        RT_FUNCTION friend inline Matrix4x4Template operator*(RealType s, const Matrix4x4Template &mat) { return Matrix4x4Template(s * mat.c0, s * mat.c1, s * mat.c2, s * mat.c3); }

        RT_FUNCTION Matrix4x4Template &operator+=(const Matrix4x4Template &mat) { c0 += mat.c0; c1 += mat.c1; c2 += mat.c2; c3 += mat.c3; return *this; }
        RT_FUNCTION Matrix4x4Template &operator-=(const Matrix4x4Template &mat) { c0 -= mat.c0; c1 -= mat.c1; c2 -= mat.c2; c3 -= mat.c3; return *this; }
        RT_FUNCTION Matrix4x4Template &operator*=(const Matrix4x4Template &mat) {
            const Vector4DTemplate<RealType> r[] = { row(0), row(1), row(2), row(3) };
            c0 = Vector4DTemplate<RealType>(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
            c1 = Vector4DTemplate<RealType>(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
            c2 = Vector4DTemplate<RealType>(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
            c3 = Vector4DTemplate<RealType>(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
            return *this;
        }
        RT_FUNCTION Matrix4x4Template &operator*=(RealType s) { c0 *= s; c1 *= s; c2 *= s; c3 *= s; return *this; }
        RT_FUNCTION Matrix4x4Template &operator/=(RealType s) { c0 /= s; c1 /= s; c2 /= s; c3 /= s; return *this; }

        RT_FUNCTION bool operator==(const Matrix4x4Template &m) const { return c0 == m.c0 && c1 == m.c1 && c2 == m.c2 && c3 == m.c3; }
        RT_FUNCTION bool operator!=(const Matrix4x4Template &m) const { return c0 == m.c0 || c1 != m.c1 || c2 != m.c2 || c3 != m.c3; }

        RT_FUNCTION Vector4DTemplate<RealType> &operator[](unsigned int c) {
            VLRAssert(c < 4, "\"c\" is out of range [0, 3].");
            return *(&c0 + c);
        }

        RT_FUNCTION Vector4DTemplate<RealType> operator[](unsigned int c) const {
            VLRAssert(c < 4, "\"c\" is out of range [0, 3].");
            return *(&c0 + c);
        }

        RT_FUNCTION const Vector4DTemplate<RealType> &column(unsigned int c) const {
            VLRAssert(c < 4, "\"c\" is out of range [0, 3].");
            return *(&c0 + c);
        }
        RT_FUNCTION Vector4DTemplate<RealType> row(unsigned int r) const {
            VLRAssert(r < 4, "\"c\" is out of range [0, 3].");
            switch (r) {
            case 0:
                return Vector4DTemplate<RealType>(m00, m01, m02, m03);
            case 1:
                return Vector4DTemplate<RealType>(m10, m11, m12, m13);
            case 2:
                return Vector4DTemplate<RealType>(m20, m21, m22, m23);
            case 3:
                return Vector4DTemplate<RealType>(m30, m31, m32, m33);
            default:
                return Vector4DTemplate<RealType>(0, 0, 0, 0);
            }
        }

        RT_FUNCTION Matrix4x4Template &swapColumns(unsigned int ca, unsigned int cb) {
            if (ca != cb) {
                Vector4DTemplate<RealType> temp = column(ca);
                (*this)[ca] = (*this)[cb];
                (*this)[cb] = temp;
            }
            return *this;
        }

        RT_FUNCTION Matrix4x4Template &swapRows(unsigned int ra, unsigned int rb) {
            if (ra != rb) {
                Vector4DTemplate<RealType> temp = row(ra);
                setRow(ra, row(rb));
                setRow(rb, temp);
            }
            return *this;
        }

        RT_FUNCTION Matrix4x4Template &setRow(unsigned int r, const Vector4DTemplate<RealType> &v) {
            VLRAssert(r < 4, "\"r\" is out of range [0, 3].");
            c0[r] = v[0]; c1[r] = v[1]; c2[r] = v[2]; c3[r] = v[3];
            return *this;
        }
        RT_FUNCTION Matrix4x4Template &scaleRow(unsigned int r, RealType s) {
            VLRAssert(r < 4, "\"r\" is out of range [0, 3].");
            c0[r] *= s; c1[r] *= s; c2[r] *= s; c3[r] *= s;
            return *this;
        }
        RT_FUNCTION Matrix4x4Template &addRow(unsigned int r, const Vector4DTemplate<RealType> &v) {
            VLRAssert(r < 4, "\"r\" is out of range [0, 3].");
            c0[r] += v[0]; c1[r] += v[1]; c2[r] += v[2]; c3[r] += v[3];
            return *this;
        }

        RT_FUNCTION Matrix4x4Template& transpose() {
            auto swap = [](RealType* v0, RealType* v1) {
                RealType temp = *v0;
                *v0 = *v1;
                *v1 = temp;
            };
            swap(&m10, &m01); swap(&m20, &m02); swap(&m30, &m03);
            swap(&m21, &m12); swap(&m31, &m13);
            swap(&m32, &m23);
            return *this;
        }

        RT_FUNCTION Matrix4x4Template &invert() {
            VLRAssert_NotImplemented();
            return *this;
        }

        RT_FUNCTION bool isIdentity() const {
            typedef Vector4DTemplate<RealType> V4;
            return c0 == V4(1, 0, 0, 0) && c1 == V4(0, 1, 0, 0) && c2 == V4(0, 0, 1, 0) && c3 == V4(0, 0, 0, 1);
        }
        RT_FUNCTION bool hasNaN() const { return c0.hasNaN() || c1.hasNaN() || c2.hasNaN() || c3.hasNaN(); }
        RT_FUNCTION bool hasInf() const { return c0.hasInf() || c1.hasInf() || c2.hasInf() || c3.hasInf(); }

        void getArray(RealType array[16]) const {
            std::memcpy(array, this, sizeof(*this));
        }

        RT_FUNCTION static constexpr Matrix4x4Template Identity() {
            RealType data[] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            };
            return Matrix4x4Template(data);
        }
    };



    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> transpose(const Matrix4x4Template<RealType> &m) {
        return Matrix4x4Template<RealType>(Vector4DTemplate<RealType>(m.m00, m.m01, m.m02, m.m03),
                                           Vector4DTemplate<RealType>(m.m10, m.m11, m.m12, m.m13),
                                           Vector4DTemplate<RealType>(m.m20, m.m21, m.m22, m.m23),
                                           Vector4DTemplate<RealType>(m.m30, m.m31, m.m32, m.m33));
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> invert(const Matrix4x4Template<RealType> &m) {
        Matrix4x4Template<RealType> mat = m;

        bool colDone[] = { false, false, false, false };
        struct SwapPair {
            int a, b;
            RT_FUNCTION SwapPair(int aa, int bb) : a(aa), b(bb) {}
        };
        SwapPair swapPairs[] = { SwapPair(0, 0), SwapPair(0, 0), SwapPair(0, 0), SwapPair(0, 0) };
        for (int pass = 0; pass < 4; ++pass) {
            int pvCol = 0;
            int pvRow = 0;
            RealType maxPivot = -1;
            for (int c = 0; c < 4; ++c) {
                if (colDone[c])
                    continue;
                for (int r = 0; r < 4; ++r) {
                    if (colDone[r])
                        continue;

                    RealType absValue = std::fabs(mat[c][r]);
                    if (absValue > maxPivot) {
                        pvCol = c;
                        pvRow = r;
                        maxPivot = absValue;
                    }
                }
            }

            mat.swapRows(pvRow, pvCol);
            swapPairs[pass] = SwapPair(pvRow, pvCol);

            RealType pivot = mat[pvCol][pvCol];
            if (pivot == 0)
                return Matrix4x4Template<RealType>(NAN, NAN, NAN, NAN);

            mat[pvCol][pvCol] = 1;
            mat.scaleRow(pvCol, 1 / pivot);
            Vector4DTemplate<RealType> addendRow = mat.row(pvCol);
            for (int r = 0; r < 4; ++r) {
                if (r != pvCol) {
                    RealType s = mat[pvCol][r];
                    mat[pvCol][r] = 0;
                    mat.addRow(r, -s * addendRow);
                }
            }

            colDone[pvCol] = true;
        }

        for (int pass = 3; pass >= 0; --pass) {
            const SwapPair &pair = swapPairs[pass];
            mat.swapColumns(pair.a, pair.b);
        }

        return mat;
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> lookAt(const Point3DTemplate<RealType> &eye, const Point3DTemplate<RealType> &tgt, const Vector3DTemplate<RealType> &up) {
        Vector3DTemplate<RealType> z = normalize(eye - tgt);
        Vector3DTemplate<RealType> x = normalize(cross(up, z));
        Vector3DTemplate<RealType> y = cross(z, x);
        Vector4DTemplate<RealType> t = Vector4DTemplate<RealType>(-dot(Vector3DTemplate<RealType>(eye), x),
                                                                  -dot(Vector3DTemplate<RealType>(eye), y),
                                                                  -dot(Vector3DTemplate<RealType>(eye), z), 1);

        return Matrix4x4Template<RealType>(Vector4DTemplate<RealType>(x.x, y.x, z.x, 0),
                                           Vector4DTemplate<RealType>(x.y, y.y, z.y, 0),
                                           Vector4DTemplate<RealType>(x.z, y.z, z.z, 0),
                                           t);
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> lookAt(RealType ex, RealType ey, RealType ez, RealType tx, RealType ty, RealType tz, RealType ux, RealType uy, RealType uz) {
        return lookAt(Point3DTemplate<RealType>(ex, ey, ez), Point3DTemplate<RealType>(tx, ty, tz), Vector3DTemplate<RealType>(ux, uy, uz));
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> scale(const Vector3DTemplate<RealType> &s) {
        return Matrix4x4Template<RealType>(s.x * Vector4DTemplate<RealType>(1, 0, 0, 0),
                                           s.y * Vector4DTemplate<RealType>(0, 1, 0, 0),
                                           s.z * Vector4DTemplate<RealType>(0, 0, 1, 0),
                                           Vector4DTemplate<RealType>(0, 0, 0, 1));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> scale(RealType sx, RealType sy, RealType sz) {
        return scale(Vector3DTemplate<RealType>(sx, sy, sz));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> scale(RealType s) {
        return scale(Vector3DTemplate<RealType>(s, s, s));
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> translate(const Vector3DTemplate<RealType> &t) {
        return Matrix4x4Template<RealType>(Vector4DTemplate<RealType>(1, 0, 0, 0),
                                           Vector4DTemplate<RealType>(0, 1, 0, 0),
                                           Vector4DTemplate<RealType>(0, 0, 1, 0),
                                           Vector4DTemplate<RealType>(t, 1));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> translate(RealType tx, RealType ty, RealType tz) {
        return translate(Vector3DTemplate<RealType>(tx, ty, tz));
    }

    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> rotate(RealType angle, const Vector3DTemplate<RealType> &axis) {
        Matrix4x4Template<RealType> matrix;
        Vector3DTemplate<RealType> nAxis = normalize(axis);
        RealType s, c;
        VLR::sincos(angle, &s, &c);
        RealType oneMinusC = 1 - c;

        matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
        matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
        matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
        matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
        matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
        matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
        matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
        matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
        matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

        matrix.m30 = matrix.m31 = matrix.m32 =
        matrix.m03 = matrix.m13 = matrix.m23 = 0;
        matrix.m33 = 1;

        return matrix;
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> rotate(RealType angle, RealType ax, RealType ay, RealType az) {
        return rotate(angle, Vector3DTemplate<RealType>(ax, ay, az));
    }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> rotateX(RealType angle) { return rotate(angle, Vector3DTemplate<RealType>(1, 0, 0)); }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> rotateY(RealType angle) { return rotate(angle, Vector3DTemplate<RealType>(0, 1, 0)); }
    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> rotateZ(RealType angle) { return rotate(angle, Vector3DTemplate<RealType>(0, 0, 1)); }

    template <typename RealType>
    RT_FUNCTION inline Matrix4x4Template<RealType> camera(RealType aspect, RealType fovY, RealType near, RealType far) {
        Matrix4x4Template<RealType> matrix;
        RealType f = 1 / std::tan(fovY / 2);
        RealType dz = far - near;

        matrix.m00 = f / aspect;
        matrix.m11 = f;
        matrix.m22 = -(near + far) / dz;
        matrix.m32 = -1;
        matrix.m23 = -2 * far * near / dz;
        matrix.m10 = matrix.m20 = matrix.m30 =
            matrix.m01 = matrix.m21 = matrix.m31 =
            matrix.m02 = matrix.m12 =
            matrix.m03 = matrix.m13 = matrix.m33 = 0;

        return matrix;
    }



    template <typename RealType>
    struct QuaternionTemplate {
        union {
            Vector3DTemplate<RealType> v;
            struct { RealType x, y, z; };
        };
        RealType w;

#if defined(VLR_Device)
        RT_FUNCTION QuaternionTemplate() { }
#else
        constexpr QuaternionTemplate() : v(), w(1) { }
#endif
        RT_FUNCTION constexpr QuaternionTemplate(RealType xx, RealType yy, RealType zz, RealType ww) : v(xx, yy, zz), w(ww) { }
        RT_FUNCTION constexpr QuaternionTemplate(const Vector3DTemplate<RealType> &vv, RealType ww) : v(vv), w(ww) { }
        RT_FUNCTION constexpr QuaternionTemplate(const Matrix4x4Template<RealType> &m) {
            RealType trace = m[0][0] + m[1][1] + m[2][2];
            if (trace > 0) {
                RealType s = std::sqrt(trace + 1);
                v = ((RealType)0.5 / s) * Vector3DTemplate<RealType>(m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0]);
                w = s / 2;
            }
            else {
                const int nxt[3] = { 1, 2, 0 };
                RealType q[3];
                int i = 0;
                if (m[1][1] > m[0][0])
                    i = 1;
                if (m[2][2] > m[i][i])
                    i = 2;
                int j = nxt[i];
                int k = nxt[j];
                RealType s = std::sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1);
                q[i] = s * 0;
                if (s != 0)
                    s = (RealType)0.5 / s;
                w = (m[j][k] - m[k][j]) * s;
                q[j] = (m[i][j] + m[j][i]) * s;
                q[k] = (m[i][k] + m[k][i]) * s;
                v = Vector3DTemplate<RealType>(q[0], q[1], q[2]);
            }
        }

        RT_FUNCTION QuaternionTemplate operator+() const { return *this; }
        RT_FUNCTION QuaternionTemplate operator-() const { return QuaternionTemplate(-v, -w); }

        RT_FUNCTION QuaternionTemplate operator+(const QuaternionTemplate &q) const { return QuaternionTemplate(v + q.v, w + q.w); }
        RT_FUNCTION QuaternionTemplate operator-(const QuaternionTemplate &q) const { return QuaternionTemplate(v - q.v, w - q.w); }
        RT_FUNCTION QuaternionTemplate operator*(const QuaternionTemplate &q) const {
            return QuaternionTemplate(cross(v, q.v) + w * q.v + q.w * v, w * q.w - dot(v, q.v));
        }
        RT_FUNCTION QuaternionTemplate operator*(RealType s) const { return QuaternionTemplate(v * s, w * s); }
        RT_FUNCTION QuaternionTemplate operator/(RealType s) const { RealType r = 1 / s; return QuaternionTemplate(v * r, w * r); }
        RT_FUNCTION friend inline QuaternionTemplate operator*(RealType s, const QuaternionTemplate &q) { return QuaternionTemplate(q.v * s, q.w * s); }

        RT_FUNCTION QuaternionTemplate &operator+=(const QuaternionTemplate &q) { v += q.v; w += q.w; return *this; }
        RT_FUNCTION QuaternionTemplate &operator-=(const QuaternionTemplate &q) { v -= q.v; w -= q.w; return *this; }
        RT_FUNCTION QuaternionTemplate &operator*=(RealType s) { v *= s; w *= s; return *this; }
        RT_FUNCTION QuaternionTemplate &operator/=(RealType s) { RealType r = 1 / s; v *= r; w *= r; return *this; }

        RT_FUNCTION Matrix3x3Template<RealType> toMatrix3x3() const {
            RealType xx = x * x, yy = y * y, zz = z * z;
            RealType xy = x * y, yz = y * z, zx = z * x;
            RealType xw = x * w, yw = y * w, zw = z * w;
            return Matrix3x3Template<RealType>(Vector3DTemplate<RealType>(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw)),
                                               Vector3DTemplate<RealType>(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)),
                                               Vector3DTemplate<RealType>(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)));
        }

        RT_FUNCTION bool operator==(const QuaternionTemplate &q) const { return v == q.v && w == q.w; }
        RT_FUNCTION bool operator!=(const QuaternionTemplate &q) const { return v != q.v || w != q.w; }
    };



    template <typename RealType>
    RT_FUNCTION inline RealType dot(const QuaternionTemplate<RealType> &q0, const QuaternionTemplate<RealType> &q1) {
        return dot(q0.v, q1.v) + q0.w * q1.w;
    }

    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> normalize(const QuaternionTemplate<RealType> &q) {
        return q / std::sqrt(dot(q, q));
    }

    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> qRotate(RealType angle, const Vector3DTemplate<RealType> &axis) {
        RealType s, c;
        VLR::sincos(angle / 2, &s, &c);
        return QuaternionTemplate<RealType>(s * normalize(axis), c);
    }
    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> qRotate(RealType angle, RealType ax, RealType ay, RealType az) {
        return qRotate(angle, Vector3DTemplate<RealType>(ax, ay, az));
    }
    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> qRotateX(RealType angle) { return qRotate(angle, Vector3DTemplate<RealType>(1, 0, 0)); }
    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> qRotateY(RealType angle) { return qRotate(angle, Vector3DTemplate<RealType>(0, 1, 0)); }
    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> qRotateZ(RealType angle) { return qRotate(angle, Vector3DTemplate<RealType>(0, 0, 1)); }

    template <typename RealType>
    RT_FUNCTION inline QuaternionTemplate<RealType> Slerp(RealType t, const QuaternionTemplate<RealType> &q0, const QuaternionTemplate<RealType> &q1) {
        RealType cosTheta = dot(q0, q1);
        if (cosTheta > (RealType)0.9995)
            return normalize((1 - t) * q0 + t * q1);
        else {
            RealType theta = std::acos(clamp(cosTheta, (RealType)-1, (RealType)1));
            RealType thetap = theta * t;
            QuaternionTemplate<RealType> qPerp = normalize(q1 - q0 * cosTheta);
            RealType sinThetaP, cosThetaP;
            VLR::sincos(thetap, &sinThetaP, &cosThetaP);
            return q0 * cosThetaP + qPerp * sinThetaP;
        }
    }

    template <typename RealType>
    RT_FUNCTION inline void decompose(const Matrix4x4Template<RealType> &mat, Vector3DTemplate<RealType>* T, QuaternionTemplate<RealType>* R, Matrix4x4Template<RealType>* S) {
        T->x = mat[3][0];
        T->y = mat[3][1];
        T->z = mat[3][2];

        Matrix4x4Template<RealType> matRS = mat;
        for (int i = 0; i < 3; ++i)
            matRS[3][i] = matRS[i][3] = 0;
        matRS[3][3] = 1;

        RealType norm;
        int count = 0;
        Matrix4x4Template<RealType> curR = matRS;
        do {
            Matrix4x4Template<RealType> itR = invert(transpose(curR));
            Matrix4x4Template<RealType> nextR = (RealType)0.5 * (curR + itR);

            norm = 0;
            for (int i = 0; i < 3; ++i) {
                using std::fabs;
                RealType n = fabs(curR[0][i] - nextR[0][i]) + abs(curR[1][i] - nextR[1][i]) + abs(curR[2][i] - nextR[2][i]);
                norm = std::fmax(norm, n);
            }
            curR = nextR;
        } while (++count < 100 && norm > 0.0001);
        *R = QuaternionTemplate<RealType>(curR);

        *S = invert(curR) * matRS;
    }



    using Vector3D = Vector3DTemplate<float>;
    using Vector4D = Vector4DTemplate<float>;
    using Normal3D = Normal3DTemplate<float>;
    using Point3D = Point3DTemplate<float>;
    using TexCoord2D = TexCoord2DTemplate<float>;
    using BoundingBox3D = BoundingBox3DTemplate<float>;
    using Matrix3x3 = Matrix3x3Template<float>;
    using Matrix4x4 = Matrix4x4Template<float>;
    using Quaternion = QuaternionTemplate<float>;



    struct Vertex {
        Point3D position;
        Normal3D normal;
        Vector3D tc0Direction;
        TexCoord2D texCoord;
    };
}
