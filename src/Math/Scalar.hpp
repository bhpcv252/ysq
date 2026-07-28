#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>

namespace ysq {

namespace detail {

// Unqualified-call wrappers. A Math type has to work both for the built-in
// floating types, where the operation is std::sqrt, and for our own scalars
// like Dual, where it is found by ADL. The `using` has to sit inside a
// function body: a requires-clause cannot host one, and a bare qualified
// std::sqrt would silently convert a Dual to nothing useful.
//
// None of these are constexpr. std::sqrt is not constexpr before C++26, and
// declaring a wrapper constexpr that can never be constant-evaluated is
// ill-formed with no diagnostic required. lengthSquared() stays constexpr;
// length() does not.

template <class T>
[[nodiscard]] auto sqrtOf(const T& x) {
    using std::sqrt;
    return sqrt(x);
}

template <class T>
[[nodiscard]] auto absOf(const T& x) {
    using std::abs;
    return abs(x);
}

template <class T>
[[nodiscard]] auto atan2Of(const T& y, const T& x) {
    using std::atan2;
    return atan2(y, x);
}

/// Finite, without needing std::isfinite or numeric_limits, so it works for
/// any Numeric including Dual: x - x is zero for a finite value and NaN for an
/// infinity or a NaN, and NaN compares false against everything.
template <class T>
[[nodiscard]] constexpr bool isFiniteValue(const T& x) {
    return (x - x) == T{0};
}

/// sqrt(a^2 + b^2) without the intermediate overflow or underflow that
/// squaring introduces.
template <class T>
[[nodiscard]] auto hypotOf(const T& a, const T& b) {
    using std::hypot;
    return hypot(a, b);
}

}  // namespace detail

/// The element type of a Math container: a field with the usual arithmetic, an
/// ordering, and a square root. Satisfied by float and double, and by Dual and
/// Complex once they land.
///
/// Deliberately not `std::floating_point`. Vector3<Dual<double>> is how the
/// spacetime code gets exact gradients instead of finite differences, and that
/// only works if the vector and tensor templates never assume a built-in type.
template <class T>
concept Numeric = requires(T a, T b) {
    T{0};
    { -a } -> std::convertible_to<T>;
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { a == b } -> std::convertible_to<bool>;
    { a < b } -> std::convertible_to<bool>;
    detail::sqrtOf(a);
};

// Variable templates over the underlying scalar rather than over the container
// element type. std::numbers is only defined for the built-in floating types,
// so there is no pi_v<Dual<double>>; code working in Dual writes kPi<double>
// and relies on Dual's implicit conversion from its value type.

template <std::floating_point T>
inline constexpr T kPi = std::numbers::pi_v<T>;

template <std::floating_point T>
inline constexpr T kTau = T{2} * std::numbers::pi_v<T>;

template <std::floating_point T>
inline constexpr T kE = std::numbers::e_v<T>;

/// Default relative and absolute tolerances for approxEqual. 128 epsilons
/// leaves room for a handful of rounding steps without waving through a real
/// error: at double that is about 2.8e-14.
template <std::floating_point T>
inline constexpr T kDefaultRelTol = std::numeric_limits<T>::epsilon() * T{128};

template <std::floating_point T>
inline constexpr T kDefaultAbsTol = std::numeric_limits<T>::epsilon() * T{128};

template <std::floating_point T>
[[nodiscard]] constexpr T radians(T degrees) noexcept {
    return degrees * (kPi<T> / T{180});
}

template <std::floating_point T>
[[nodiscard]] constexpr T degrees(T radians) noexcept {
    return radians * (T{180} / kPi<T>);
}

/// By value, unlike std::clamp, which returns a reference and so dangles on a
/// temporary. Math arguments are small and passed in registers anyway.
/// Unlike std::clamp this does not require lo <= hi; it returns lo when they
/// are crossed rather than being undefined.
template <Numeric T>
[[nodiscard]] constexpr T clamp(T value, T lo, T hi) noexcept {
    if (value < lo) {
        return lo;
    }
    if (hi < value) {
        return hi;
    }
    return value;
}

/// -1, 0 or +1. NaN gives 0, since it is neither above nor below zero.
template <Numeric T>
[[nodiscard]] constexpr T sign(T value) noexcept {
    if (T{0} < value) {
        return T{1};
    }
    if (value < T{0}) {
        return T{-1};
    }
    return T{0};
}

/// Mixed relative and absolute comparison: |a - b| <= max(absTol, relTol *
/// max(|a|, |b|)).
///
/// The absolute term is what makes it usable near zero, where a relative
/// tolerance collapses to nothing. Equal infinities compare equal because the
/// exact check runs first; any NaN compares unequal, including to itself.
template <std::floating_point T>
[[nodiscard]] constexpr bool approxEqual(T a, T b, T relTol = kDefaultRelTol<T>,
                                         T absTol = kDefaultAbsTol<T>) noexcept {
    if (a == b) {
        return true;
    }
    // Written without std::abs so the whole function stays constexpr.
    const T magA = (a < T{0}) ? -a : a;
    const T magB = (b < T{0}) ? -b : b;

    // Past the exact check, a non-finite operand can only produce a nonsense
    // answer: an infinite scale makes the relative tolerance infinite, which
    // would rate +inf and -inf as approximately equal. NaN is caught by the
    // same test, since it compares false against everything.
    constexpr T largestFinite = std::numeric_limits<T>::max();
    if (!(magA <= largestFinite) || !(magB <= largestFinite)) {
        return false;
    }

    const T diff = (a < b) ? (b - a) : (a - b);
    const T scale = (magA < magB) ? magB : magA;
    const T tol = (absTol < relTol * scale) ? relTol * scale : absTol;
    return diff <= tol;
}

template <std::floating_point T>
[[nodiscard]] constexpr bool isNearZero(T value, T absTol = kDefaultAbsTol<T>) noexcept {
    const T mag = (value < T{0}) ? -value : value;
    return mag <= absTol;
}

}  // namespace ysq
