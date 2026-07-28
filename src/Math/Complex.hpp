#pragma once

#include <Math/Scalar.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <optional>

namespace ysq {

/// Complex numbers.
///
/// Ours rather than std::complex because the standard only specifies
/// std::complex for float, double and long double; instantiating it on
/// anything else is unspecified. Complex<Dual<double>> is a shape this project
/// will want, for a phase that carries its own derivative, and that is simply
/// not something std::complex promises to work for.
///
/// Complex deliberately does not satisfy the Numeric concept: it has no
/// ordering, so it cannot go inside Vector or Matrix, whose pivoting,
/// componentwise min and max, and zero-length checks all compare magnitudes.
/// A complex vector space would need a Hermitian inner product rather than the
/// bilinear one those types use, so it would be a separate design decision
/// rather than a free instantiation.
template <Numeric T>
struct Complex {
    using value_type = T;

    T re{};
    T im{};

    /// So the test comparators and the formatter can treat this as a pair of
    /// components. Index 0 is the real part.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return 2; }

    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? re : im;
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? re : im;
    }

    [[nodiscard]] static constexpr Complex zero() noexcept { return {}; }
    [[nodiscard]] static constexpr Complex one() noexcept { return {T{1}, T{0}}; }
    /// The imaginary unit.
    [[nodiscard]] static constexpr Complex i() noexcept { return {T{0}, T{1}}; }

    [[nodiscard]] static constexpr Complex real(T value) noexcept {
        return {value, T{0}};
    }

    [[nodiscard]] static constexpr Complex imaginary(T value) noexcept {
        return {T{0}, value};
    }

    /// r * (cos(theta) + i sin(theta)).
    [[nodiscard]] static Complex polar(T radius, T angle) {
        using std::cos;
        using std::sin;
        return {radius * cos(angle), radius * sin(angle)};
    }

    constexpr Complex& operator+=(const Complex& other) noexcept {
        re += other.re;
        im += other.im;
        return *this;
    }

    constexpr Complex& operator-=(const Complex& other) noexcept {
        re -= other.re;
        im -= other.im;
        return *this;
    }

    constexpr Complex& operator*=(const Complex& other) noexcept {
        *this = *this * other;
        return *this;
    }

    constexpr Complex& operator/=(const Complex& other) noexcept {
        *this = *this / other;
        return *this;
    }

    constexpr Complex& operator*=(T scalar) noexcept {
        re *= scalar;
        im *= scalar;
        return *this;
    }

    constexpr Complex& operator/=(T scalar) noexcept {
        re /= scalar;
        im /= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr Complex operator+(const Complex& z) noexcept {
        return z;
    }

    [[nodiscard]] friend constexpr Complex operator-(const Complex& z) noexcept {
        return {-z.re, -z.im};
    }

    [[nodiscard]] friend constexpr Complex operator+(const Complex& a,
                                                     const Complex& b) noexcept {
        return {a.re + b.re, a.im + b.im};
    }

    [[nodiscard]] friend constexpr Complex operator-(const Complex& a,
                                                     const Complex& b) noexcept {
        return {a.re - b.re, a.im - b.im};
    }

    [[nodiscard]] friend constexpr Complex operator*(const Complex& a,
                                                     const Complex& b) noexcept {
        return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    }

    /// Smith's formula: divides through by whichever part of the denominator
    /// is larger, so the intermediate squares cannot overflow for a divisor
    /// the result is perfectly able to represent.
    [[nodiscard]] friend constexpr Complex operator/(const Complex& a,
                                                     const Complex& b) noexcept {
        const T magRe = (b.re < T{0}) ? -b.re : b.re;
        const T magIm = (b.im < T{0}) ? -b.im : b.im;
        if (magIm <= magRe) {
            const T ratio = b.im / b.re;
            const T denominator = b.re + b.im * ratio;
            return {(a.re + a.im * ratio) / denominator,
                    (a.im - a.re * ratio) / denominator};
        }
        const T ratio = b.re / b.im;
        const T denominator = b.re * ratio + b.im;
        return {(a.re * ratio + a.im) / denominator, (a.im * ratio - a.re) / denominator};
    }

    [[nodiscard]] friend constexpr Complex operator+(const Complex& z,
                                                     T scalar) noexcept {
        return {z.re + scalar, z.im};
    }

    [[nodiscard]] friend constexpr Complex operator+(T scalar,
                                                     const Complex& z) noexcept {
        return {scalar + z.re, z.im};
    }

    [[nodiscard]] friend constexpr Complex operator-(const Complex& z,
                                                     T scalar) noexcept {
        return {z.re - scalar, z.im};
    }

    [[nodiscard]] friend constexpr Complex operator-(T scalar,
                                                     const Complex& z) noexcept {
        return {scalar - z.re, -z.im};
    }

    [[nodiscard]] friend constexpr Complex operator*(const Complex& z,
                                                     T scalar) noexcept {
        return {z.re * scalar, z.im * scalar};
    }

    [[nodiscard]] friend constexpr Complex operator*(T scalar,
                                                     const Complex& z) noexcept {
        return {scalar * z.re, scalar * z.im};
    }

    [[nodiscard]] friend constexpr Complex operator/(const Complex& z,
                                                     T scalar) noexcept {
        return {z.re / scalar, z.im / scalar};
    }

    [[nodiscard]] friend constexpr bool operator==(const Complex&,
                                                   const Complex&) = default;
};

template <Numeric T>
[[nodiscard]] constexpr Complex<T> conj(const Complex<T>& z) noexcept {
    return {z.re, -z.im};
}

/// |z|^2, without a square root.
template <Numeric T>
[[nodiscard]] constexpr T lengthSquared(const Complex<T>& z) noexcept {
    return z.re * z.re + z.im * z.im;
}

/// The modulus, by hypot rather than sqrt of the sum of squares, so a value
/// near the top of the exponent range does not overflow on the way.
template <Numeric T>
[[nodiscard]] auto abs(const Complex<T>& z) {
    return detail::hypotOf(z.re, z.im);
}

template <Numeric T>
[[nodiscard]] auto length(const Complex<T>& z) {
    return abs(z);
}

/// The principal argument, in (-pi, pi].
template <Numeric T>
[[nodiscard]] auto arg(const Complex<T>& z) {
    return detail::atan2Of(z.im, z.re);
}

template <Numeric T>
[[nodiscard]] constexpr Complex<T> inverse(const Complex<T>& z) noexcept {
    return conj(z) / lengthSquared(z);
}

template <Numeric T>
[[nodiscard]] Complex<T> normalized(const Complex<T>& z) {
    return z / abs(z);
}

template <Numeric T>
[[nodiscard]] std::optional<Complex<T>> tryNormalized(const Complex<T>& z) {
    const T magnitude = abs(z);
    // Negated so a NaN fails too. The modulus goes through hypot, so unlike
    // the vector types it cannot overflow on its own; only an operand that was
    // already non-finite gets here.
    if (!(T{0} < magnitude) || !detail::isFiniteValue(magnitude)) {
        return std::nullopt;
    }
    return z / magnitude;
}

template <Numeric T>
[[nodiscard]] Complex<T> exp(const Complex<T>& z) {
    using std::cos;
    using std::exp;
    using std::sin;
    const T scale = exp(z.re);
    return {scale * cos(z.im), scale * sin(z.im)};
}

/// The principal branch, with the imaginary part in (-pi, pi]. Discontinuous
/// across the negative real axis, as any branch must be.
template <Numeric T>
[[nodiscard]] Complex<T> log(const Complex<T>& z) {
    using std::log;
    return {log(abs(z)), arg(z)};
}

/// The principal square root, the one with non-negative real part.
///
/// Built from the modulus rather than from half the argument: the polar route
/// costs an atan2 and a pair of trig calls and loses precision for a z that is
/// nearly real, where the argument it computes is a small difference of large
/// quantities.
template <Numeric T>
[[nodiscard]] Complex<T> sqrt(const Complex<T>& z) {
    using std::sqrt;
    if (z.re == T{0} && z.im == T{0}) {
        return {T{0}, z.im};  // preserves the sign of a signed zero
    }

    const T magRe = (z.re < T{0}) ? -z.re : z.re;
    const T scale = sqrt((magRe + abs(z)) / T{2});

    if (T{0} <= z.re) {
        return {scale, z.im / (T{2} * scale)};
    }
    const T magIm = (z.im < T{0}) ? -z.im : z.im;
    return {magIm / (T{2} * scale), (z.im < T{0}) ? -scale : scale};
}

/// exp(w log z), on the principal branch of the logarithm.
template <Numeric T>
[[nodiscard]] Complex<T> pow(const Complex<T>& z, const Complex<T>& exponent) {
    return exp(exponent * log(z));
}

template <Numeric T>
[[nodiscard]] Complex<T> pow(const Complex<T>& z, T exponent) {
    return exp(log(z) * exponent);
}

using Cplx = Complex<double>;
using Cplxf = Complex<float>;

}  // namespace ysq
