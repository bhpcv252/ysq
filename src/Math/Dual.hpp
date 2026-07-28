#pragma once

#include <Math/Scalar.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

namespace ysq {

/// Dual numbers: forward-mode automatic differentiation.
///
/// A dual number carries a value and the derivative of that value with respect
/// to whatever was seeded. Every operation below propagates both, so evaluating
/// an ordinary expression in Dual gives the derivative exactly, to machine
/// precision, with no step size to choose and no cancellation to trade off.
///
///     derivative([](auto x) { return x * x + sin(x); }, 2.0);  // 4 + cos(2)
///
/// This is why Vector, Matrix and Tensor are templated on their scalar rather
/// than fixed to double: Vector3<Dual<double>> costs nothing extra and is what
/// gives Physics/Spacetime exact metric derivatives, and therefore exact
/// Christoffel symbols, instead of finite differences.
///
/// Dual<Dual<T>> nests, which is where second derivatives come from. Nothing
/// here assumes T is a built-in type, so the nesting works by construction
/// rather than by special case.
///
/// **Comparisons look at the value only.** This is the usual convention for
/// automatic differentiation and it is what makes generic code behave: a
/// branch on `x < 0`, a clamp, a componentwise min are all asking about
/// magnitude, and none of them should change answer because a derivative
/// differs. The cost is that `a == b` does not imply the two are
/// interchangeable; `identical(a, b)` is the spelling for that.
template <Numeric T>
struct Dual {
    using value_type = T;

    T value{};
    T derivative{};

    constexpr Dual() = default;

    /// Implicit on purpose: a constant folded into a dual expression has
    /// derivative zero, which is exactly right, and requiring it to be spelled
    /// out would make every formula unreadable.
    ///
    /// Templated rather than taking T directly so that a nested dual can still
    /// absorb a plain scalar. `Dual<Dual<double>> * 3.0` would otherwise need
    /// two user-defined conversions in a row, which the language does not
    /// allow, and a second-derivative computation would be unable to contain a
    /// literal. Here the inner conversion happens inside the constructor, so
    /// there is only one for overload resolution to find.
    template <class U>
        requires std::convertible_to<U, T>
    constexpr Dual(U real) : value(static_cast<T>(real)) {}

    constexpr Dual(T real, T tangent) : value(real), derivative(tangent) {}

    /// Index 0 is the value. Present so the test comparators and the formatter
    /// can treat this as a pair of components.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return 2; }

    [[nodiscard]] constexpr T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return (index == 0) ? value : derivative;
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return (index == 0) ? value : derivative;
    }

    /// The seed: a variable whose derivative with respect to itself is one.
    [[nodiscard]] static constexpr Dual variable(T at) noexcept { return {at, T{1}}; }

    /// A constant, whose derivative is zero.
    [[nodiscard]] static constexpr Dual constant(T at) noexcept { return {at, T{0}}; }

    constexpr Dual& operator+=(const Dual& other) noexcept {
        value += other.value;
        derivative += other.derivative;
        return *this;
    }

    constexpr Dual& operator-=(const Dual& other) noexcept {
        value -= other.value;
        derivative -= other.derivative;
        return *this;
    }

    constexpr Dual& operator*=(const Dual& other) noexcept {
        *this = *this * other;
        return *this;
    }

    constexpr Dual& operator/=(const Dual& other) noexcept {
        *this = *this / other;
        return *this;
    }

    // Only Dual-on-Dual overloads are needed: a bare T converts implicitly
    // through the constructor above, on either side.

    [[nodiscard]] friend constexpr Dual operator+(const Dual& a) noexcept { return a; }

    [[nodiscard]] friend constexpr Dual operator-(const Dual& a) noexcept {
        return {-a.value, -a.derivative};
    }

    [[nodiscard]] friend constexpr Dual operator+(const Dual& a, const Dual& b) noexcept {
        return {a.value + b.value, a.derivative + b.derivative};
    }

    [[nodiscard]] friend constexpr Dual operator-(const Dual& a, const Dual& b) noexcept {
        return {a.value - b.value, a.derivative - b.derivative};
    }

    /// The product rule.
    [[nodiscard]] friend constexpr Dual operator*(const Dual& a, const Dual& b) noexcept {
        return {a.value * b.value, a.derivative * b.value + a.value * b.derivative};
    }

    /// The quotient rule.
    [[nodiscard]] friend constexpr Dual operator/(const Dual& a, const Dual& b) noexcept {
        return {a.value / b.value,
                (a.derivative * b.value - a.value * b.derivative) / (b.value * b.value)};
    }

    /// Value only. See the note on the class.
    [[nodiscard]] friend constexpr bool operator==(const Dual& a,
                                                   const Dual& b) noexcept {
        return a.value == b.value;
    }

    [[nodiscard]] friend constexpr bool operator<(const Dual& a, const Dual& b) noexcept {
        return a.value < b.value;
    }

    [[nodiscard]] friend constexpr bool operator>(const Dual& a, const Dual& b) noexcept {
        return b.value < a.value;
    }

    [[nodiscard]] friend constexpr bool operator<=(const Dual& a,
                                                   const Dual& b) noexcept {
        return !(b.value < a.value);
    }

    [[nodiscard]] friend constexpr bool operator>=(const Dual& a,
                                                   const Dual& b) noexcept {
        return !(a.value < b.value);
    }
};

/// Component-for-component equality, which operator== deliberately is not.
template <Numeric T>
[[nodiscard]] constexpr bool identical(const Dual<T>& a, const Dual<T>& b) noexcept {
    return a.value == b.value && a.derivative == b.derivative;
}

template <class T>
inline constexpr bool isDual = false;

template <class T>
inline constexpr bool isDual<Dual<T>> = true;

/// Strips every layer of Dual, down to the underlying scalar.
template <class T>
[[nodiscard]] constexpr auto valueOf(const T& x) {
    if constexpr (isDual<T>) {
        return valueOf(x.value);
    } else {
        return x;
    }
}

// The chain rule, once per function. Each is d/dx f(u) = f'(u) du.

template <Numeric T>
[[nodiscard]] Dual<T> sqrt(const Dual<T>& a) {
    using std::sqrt;
    const T root = sqrt(a.value);
    return {root, a.derivative / (T{2} * root)};
}

template <Numeric T>
[[nodiscard]] Dual<T> exp(const Dual<T>& a) {
    using std::exp;
    const T scale = exp(a.value);
    return {scale, scale * a.derivative};
}

template <Numeric T>
[[nodiscard]] Dual<T> log(const Dual<T>& a) {
    using std::log;
    return {log(a.value), a.derivative / a.value};
}

template <Numeric T>
[[nodiscard]] Dual<T> sin(const Dual<T>& a) {
    using std::cos;
    using std::sin;
    return {sin(a.value), cos(a.value) * a.derivative};
}

template <Numeric T>
[[nodiscard]] Dual<T> cos(const Dual<T>& a) {
    using std::cos;
    using std::sin;
    return {cos(a.value), -sin(a.value) * a.derivative};
}

/// Through 1 + tan^2 rather than 1/cos^2, which needs one call instead of two
/// and does not lose the answer where cos is small.
template <Numeric T>
[[nodiscard]] Dual<T> tan(const Dual<T>& a) {
    using std::tan;
    const T t = tan(a.value);
    return {t, a.derivative * (T{1} + t * t)};
}

template <Numeric T>
[[nodiscard]] Dual<T> asin(const Dual<T>& a) {
    using std::asin;
    using std::sqrt;
    return {asin(a.value), a.derivative / sqrt(T{1} - a.value * a.value)};
}

template <Numeric T>
[[nodiscard]] Dual<T> acos(const Dual<T>& a) {
    using std::acos;
    using std::sqrt;
    return {acos(a.value), -a.derivative / sqrt(T{1} - a.value * a.value)};
}

template <Numeric T>
[[nodiscard]] Dual<T> atan(const Dual<T>& a) {
    using std::atan;
    return {atan(a.value), a.derivative / (T{1} + a.value * a.value)};
}

template <Numeric T>
[[nodiscard]] Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) {
    using std::atan2;
    const T denominator = x.value * x.value + y.value * y.value;
    return {atan2(y.value, x.value),
            (x.value * y.derivative - y.value * x.derivative) / denominator};
}

template <Numeric T>
[[nodiscard]] Dual<T> sinh(const Dual<T>& a) {
    using std::cosh;
    using std::sinh;
    return {sinh(a.value), cosh(a.value) * a.derivative};
}

template <Numeric T>
[[nodiscard]] Dual<T> cosh(const Dual<T>& a) {
    using std::cosh;
    using std::sinh;
    return {cosh(a.value), sinh(a.value) * a.derivative};
}

template <Numeric T>
[[nodiscard]] Dual<T> tanh(const Dual<T>& a) {
    using std::tanh;
    const T t = tanh(a.value);
    return {t, a.derivative * (T{1} - t * t)};
}

template <Numeric T>
[[nodiscard]] Dual<T> asinh(const Dual<T>& a) {
    using std::asinh;
    using std::sqrt;
    return {asinh(a.value), a.derivative / sqrt(a.value * a.value + T{1})};
}

template <Numeric T>
[[nodiscard]] Dual<T> acosh(const Dual<T>& a) {
    using std::acosh;
    using std::sqrt;
    return {acosh(a.value), a.derivative / sqrt(a.value * a.value - T{1})};
}

template <Numeric T>
[[nodiscard]] Dual<T> atanh(const Dual<T>& a) {
    using std::atanh;
    return {atanh(a.value), a.derivative / (T{1} - a.value * a.value)};
}

/// The derivative at zero does not exist. sign() returns zero there, so that
/// is what comes back: a defensible choice, not a correct one.
template <Numeric T>
[[nodiscard]] Dual<T> abs(const Dual<T>& a) {
    using std::abs;
    return {abs(a.value), sign(a.value) * a.derivative};
}

template <Numeric T>
[[nodiscard]] Dual<T> hypot(const Dual<T>& a, const Dual<T>& b) {
    using std::hypot;
    const T magnitude = hypot(a.value, b.value);
    return {magnitude, (a.value * a.derivative + b.value * b.derivative) / magnitude};
}

/// The general case, which needs log and therefore a positive base.
template <Numeric T>
[[nodiscard]] Dual<T> pow(const Dual<T>& a, const Dual<T>& b) {
    using std::log;
    using std::pow;
    const T result = pow(a.value, b.value);
    return {result,
            result * (b.derivative * log(a.value) + b.value * a.derivative / a.value)};
}

/// A constant exponent, which avoids the logarithm entirely and so works for a
/// negative base with an integral power.
template <Numeric T>
[[nodiscard]] Dual<T> pow(const Dual<T>& a, T exponent) {
    using std::pow;
    return {pow(a.value, exponent),
            exponent * pow(a.value, exponent - T{1}) * a.derivative};
}

template <Numeric T>
[[nodiscard]] Dual<T> pow(T base, const Dual<T>& b) {
    using std::log;
    using std::pow;
    const T result = pow(base, b.value);
    return {result, result * log(base) * b.derivative};
}

/// f(x) and f'(x) in one evaluation.
///
/// f has to be a generic callable, since it is invoked on Dual<T> rather than
/// on T. A lambda taking `auto` is the usual way to write one.
template <class F, Numeric T>
[[nodiscard]] constexpr auto valueAndDerivative(F&& f, T at) {
    const auto result = f(Dual<T>::variable(at));
    return std::pair{result.value, result.derivative};
}

template <class F, Numeric T>
[[nodiscard]] constexpr auto derivative(F&& f, T at) {
    return f(Dual<T>::variable(at)).derivative;
}

/// f''(x), by nesting: the outer level differentiates the inner level's
/// derivative. Both levels are seeded, which is what makes the second
/// derivative fall out of the doubly nested tangent.
template <class F, Numeric T>
[[nodiscard]] constexpr auto secondDerivative(F&& f, T at) {
    using Inner = Dual<T>;
    const Dual<Inner> seed{Inner{at, T{1}}, Inner{T{1}, T{0}}};
    return f(seed).derivative.derivative;
}

using DualD = Dual<double>;
using DualF = Dual<float>;
/// Second derivatives.
using Dual2D = Dual<Dual<double>>;

// The contract this header exists to satisfy. If Dual ever stops being a
// Numeric, Vector3<Dual<double>> stops compiling and the exact-derivative path
// through Physics/Spacetime goes with it.
static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);
static_assert(Numeric<Dual<Dual<double>>>);

}  // namespace ysq
