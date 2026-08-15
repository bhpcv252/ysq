#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Math/Scalar.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>

namespace ysq {

/// Special functions with no home elsewhere in Math: the error function and
/// gamma function (needed by `Math/Random.hpp`'s normal distribution and
/// `Physics/Thermodynamics/StatisticalMechanics.hpp`'s Maxwell-Boltzmann
/// distribution), the associated Legendre polynomials (needed by
/// `Physics/Gravity/SphericalHarmonics.hpp`), and the Bessel functions
/// (general-purpose special functions with no standard-library shortcut
/// here, same as `legendreP` below).
///
/// `erf`, `erfc`, `gamma` and `logGamma` are thin, ADL-dispatched wrappers
/// over the C99 math functions already in `<cmath>` (`std::erf`, `std::erfc`,
/// `std::tgamma`, `std::lgamma`), following `Math/Scalar.hpp`'s own
/// `sqrtOf`/`absOf` pattern so a future `Numeric` type can extend them by ADL
/// exactly as it already can for those.
///
/// `legendreP` has no such standard-library shortcut: the associated
/// Legendre polynomials are part of C++17's special mathematical functions
/// (`std::assoc_legendre`), which libc++ (this project's standard library on
/// macOS) does not implement, so it is computed here from the standard
/// upward recurrence instead.
///
/// `erf`, `erfc`, `gamma`, `logGamma` and `legendreP` each have a batched
/// overload (`std::span<const T> x, std::span<T> result`) alongside the
/// scalar one, evaluating independently at every element with no
/// cross-element dependency; above a size threshold, and only for `T =
/// float`, that overload dispatches through `Compute::defaultBackend()`
/// rather than looping the scalar version, transparently to the caller. See
/// `Compute/ComputeBackend.hpp`'s own comment on this kernel family for why
/// the GPU path uses different (but float32-safe) approximations than the
/// scalar path's C99 calls, and why `besselJ`/`besselY` have no batched
/// overload at all.

namespace detail {

template <class T>
[[nodiscard]] auto erfOf(const T& x) {
    using std::erf;
    return erf(x);
}

template <class T>
[[nodiscard]] auto erfcOf(const T& x) {
    using std::erfc;
    return erfc(x);
}

template <class T>
[[nodiscard]] auto gammaOf(const T& x) {
    using std::tgamma;
    return tgamma(x);
}

template <class T>
[[nodiscard]] auto lgammaOf(const T& x) {
    using std::lgamma;
    return lgamma(x);
}

/// Rational-polynomial approximations of `J0`/`J1` (Abramowitz & Stegun
/// 9.4.1/9.4.3 for `|x| < 8`, 9.4.5/9.4.6 elsewhere), the same coefficients
/// long popularized as Numerical Recipes' `bessj0`/`bessj1`. Computed in
/// double regardless of the caller's `T`: these coefficients are fit to
/// about 1e-8 relative accuracy, already at or past `float`'s own
/// precision, so nothing is gained carrying `T` through the arithmetic —
/// only the final result is cast back to `T`.
[[nodiscard]] inline double besselJ0Double(double x) {
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;

    const double ax = abs(x);
    if (ax < 8.0) {
        const double y = x * x;
        const double num =
            57568490574.0 +
            y * (-13362590354.0 +
                 y * (651619640.7 +
                      y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))));
        const double den =
            57568490411.0 +
            y * (1029532985.0 +
                 y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
        return num / den;
    }

    const double z = 8.0 / ax;
    const double y = z * z;
    const double xx = ax - 0.785398164;
    const double p0 =
        1.0 + y * (-0.1098628627e-2 +
                   y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
    const double q0 =
        -0.1562499995e-1 +
        y * (0.1430488765e-3 +
             y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
    return sqrt(0.636619772 / ax) * (cos(xx) * p0 - z * sin(xx) * q0);
}

[[nodiscard]] inline double besselJ1Double(double x) {
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;

    const double ax = abs(x);
    if (ax < 8.0) {
        const double y = x * x;
        const double num =
            x * (72362614232.0 +
                 y * (-7895059235.0 +
                      y * (242396853.1 +
                           y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))));
        const double den =
            144725228442.0 +
            y * (2300535178.0 +
                 y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        return num / den;
    }

    const double z = 8.0 / ax;
    const double y = z * z;
    const double xx = ax - 2.356194491;
    const double p1 =
        1.0 + y * (0.183105e-2 +
                   y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
    const double q1 =
        0.04687499995 +
        y * (-0.2002690873e-3 +
             y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
    const double result = sqrt(0.636619772 / ax) * (cos(xx) * p1 - z * sin(xx) * q1);
    return (x < 0.0) ? -result : result;
}

/// `Y0`/`Y1` (Abramowitz & Stegun 9.4.2/9.4.4 for `x < 8` — the small-`x`
/// branch has an explicit `ln(x) J0(x)`/`ln(x) J1(x)` term, since `Y_n` has
/// a logarithmic singularity at the origin that no polynomial alone can
/// reproduce — and the same 9.4.5/9.4.6 asymptotic form `J0`/`J1` use for
/// `x >= 8`, since `J_n` and `Y_n` share one asymptotic amplitude and only
/// differ in phase).
[[nodiscard]] inline double besselY0Double(double x) {
    using std::cos;
    using std::log;
    using std::sin;
    using std::sqrt;

    if (x < 8.0) {
        const double y = x * x;
        const double num =
            -2957821389.0 +
            y * (7062834065.0 +
                 y * (-512359803.6 +
                      y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
        const double den =
            40076544269.0 +
            y * (745249964.8 +
                 y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y))));
        return num / den + 0.636619772 * besselJ0Double(x) * log(x);
    }

    const double z = 8.0 / x;
    const double y = z * z;
    const double xx = x - 0.785398164;
    const double p0 =
        1.0 + y * (-0.1098628627e-2 +
                   y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
    const double q0 =
        -0.1562499995e-1 +
        y * (0.1430488765e-3 +
             y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
    return sqrt(0.636619772 / x) * (sin(xx) * p0 + z * cos(xx) * q0);
}

[[nodiscard]] inline double besselY1Double(double x) {
    using std::cos;
    using std::log;
    using std::sin;
    using std::sqrt;

    if (x < 8.0) {
        const double y = x * x;
        const double num =
            x * (-0.4900604943e13 +
                 y * (0.1275274390e13 +
                      y * (-0.5153438139e11 +
                           y * (0.7349264551e9 +
                                y * (-0.4237922726e7 + y * 0.8511937935e4)))));
        const double den =
            0.2499580570e14 +
            y * (0.4244419664e12 +
                 y * (0.3733650367e10 +
                      y * (0.2245904002e8 +
                           y * (0.1020426050e6 + y * (0.3549632885e3 + y)))));
        return num / den + 0.636619772 * (besselJ1Double(x) * log(x) - 1.0 / x);
    }

    const double z = 8.0 / x;
    const double y = z * z;
    const double xx = x - 2.356194491;
    const double p1 =
        1.0 + y * (0.183105e-2 +
                   y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
    const double q1 =
        0.04687499995 +
        y * (-0.2002690873e-3 +
             y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
    return sqrt(0.636619772 / x) * (sin(xx) * p1 + z * cos(xx) * q1);
}

/// `J_n` for `n >= 2` via Miller's algorithm (Numerical Recipes' `bessj`):
/// recurring `J`'s three-term recurrence downward from an order well above
/// `n` is stable, unlike recurring it upward, because downward recurrence
/// damps whatever arbitrary error the arbitrary starting values introduce
/// while upward recurrence amplifies it. The starting order is seeded with
/// an arbitrary value and the whole resulting sequence is rescaled once at
/// the end via the sum rule `J_0(x) + 2 sum_{k>=1} J_{2k}(x) = 1`, since
/// downward recurrence gets every ratio between consecutive orders right
/// immediately but not the overall scale. For `|x| > n`, plain upward
/// recurrence from `J0`/`J1` is already stable and cheaper, so it is used
/// instead.
[[nodiscard]] inline double besselJDouble(unsigned n, double x) {
    using std::abs;
    using std::sqrt;

    if (n == 0) {
        return besselJ0Double(x);
    }
    if (n == 1) {
        return besselJ1Double(x);
    }

    const double ax = abs(x);
    if (ax == 0.0) {
        return 0.0;
    }

    constexpr double kAcc = 40.0;
    constexpr double kBigNo = 1.0e10;
    constexpr double kBigNi = 1.0e-10;

    double result{};
    if (ax > static_cast<double>(n)) {
        const double tox = 2.0 / ax;
        double bjm = besselJ0Double(ax);
        double bj = besselJ1Double(ax);
        for (unsigned j = 1; j < n; ++j) {
            const double bjp = static_cast<double>(j) * tox * bj - bjm;
            bjm = bj;
            bj = bjp;
        }
        result = bj;
    } else {
        const double tox = 2.0 / ax;
        const unsigned m =
            2 * ((n + static_cast<unsigned>(sqrt(kAcc * static_cast<double>(n)))) / 2);
        bool jsum = false;
        double bjp = 0.0;
        double ans = 0.0;
        double sum = 0.0;
        double bj = 1.0;
        for (unsigned j = m; j >= 1; --j) {
            const double bjm = static_cast<double>(j) * tox * bj - bjp;
            bjp = bj;
            bj = bjm;
            if (abs(bj) > kBigNo) {
                bj *= kBigNi;
                bjp *= kBigNi;
                ans *= kBigNi;
                sum *= kBigNi;
            }
            if (jsum) {
                sum += bj;
            }
            jsum = !jsum;
            if (j == n) {
                ans = bjp;
            }
        }
        sum = 2.0 * sum - bj;
        result = ans / sum;
    }

    return (x < 0.0 && (n % 2 == 1)) ? -result : result;
}

/// `Y_n` for `n >= 2` via plain upward recurrence from `Y0`/`Y1`: stable in
/// this direction, the opposite of `J`'s recurrence above, because `Y_n`
/// grows with `n` for fixed `x` rather than shrinking, so accumulated
/// error shrinks relative to the answer instead of swamping it.
[[nodiscard]] inline double besselYDouble(unsigned n, double x) {
    if (n == 0) {
        return besselY0Double(x);
    }
    if (n == 1) {
        return besselY1Double(x);
    }
    const double tox = 2.0 / x;
    double bym = besselY0Double(x);
    double by = besselY1Double(x);
    for (unsigned j = 1; j < n; ++j) {
        const double byp = static_cast<double>(j) * tox * by - bym;
        bym = by;
        by = byp;
    }
    return by;
}

}  // namespace detail

namespace detail {

/// Measured on the development machine (Apple Silicon, Metal backend) by
/// `benchmarks/compute_thresholds.cpp`, using `batchErf` as the
/// representative of this file's whole batch-evaluation family
/// (`erf`/`erfc`/`gamma`/`logGamma`/`legendreP` all share this one
/// dispatch shape and per-call GPU overhead; only `erf` was measured
/// directly). The GPU path never won up to 65536 elements, the largest
/// size tried, so this is a measured floor (double the largest size
/// tried), not an observed crossover: a single dispatch's fixed buffer
/// allocation/upload/readback overhead outweighs this family's
/// per-element cost even at that size. Re-run the benchmark and update
/// this if the reference machine or backend ever changes.
inline constexpr std::size_t kSpecialFunctionsGpuDispatchThreshold = 131072;

inline void erfGpu(std::span<const float> x, std::span<float> result) {
    defaultBackend().batchErf(x, result);
}

inline void erfcGpu(std::span<const float> x, std::span<float> result) {
    defaultBackend().batchErfc(x, result);
}

inline void gammaGpu(std::span<const float> x, std::span<float> result) {
    defaultBackend().batchGamma(x, result);
}

inline void logGammaGpu(std::span<const float> x, std::span<float> result) {
    defaultBackend().batchLogGamma(x, result);
}

inline void legendrePGpu(unsigned n, unsigned m, std::span<const float> x,
                         std::span<float> result) {
    defaultBackend().batchLegendreP(n, m, x, result);
}

}  // namespace detail

/// The error function, `erf(x) = (2/sqrt(pi)) integral_0^x exp(-t^2) dt`: the
/// probability a standard normal variable falls within `x sqrt(2)` of its
/// mean, which is what `Math/Random.hpp`'s normal CDF is built from.
template <std::floating_point T>
[[nodiscard]] T erf(T x) {
    return detail::erfOf(x);
}

/// `1 - erf(x)`, computed directly rather than by subtraction: for large `x`,
/// `erf(x)` rounds to exactly 1 and the subtraction loses every bit `erfc`
/// would have had left, the same cancellation `erfc` exists to avoid in any
/// library that provides it.
template <std::floating_point T>
[[nodiscard]] T erfc(T x) {
    return detail::erfcOf(x);
}

/// Batched `erf`, one result per element of `x`. Above a size threshold, and
/// only for `T = float` (a `double` call always stays on the CPU loop below,
/// gated with `if constexpr`, matching `Math/LinearSolve.hpp`'s own
/// convention for the same reason: no GPU backend offers `float64`),
/// dispatches through `Compute::defaultBackend()`.
template <std::floating_point T>
void erf(std::span<const T> x, std::span<T> result) {
    assert(x.size() == result.size());
    if constexpr (std::same_as<T, float>) {
        if (x.size() >= detail::kSpecialFunctionsGpuDispatchThreshold) {
            detail::erfGpu(x, result);
            return;
        }
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = erf(x[i]);
    }
}

/// Batched `erfc`; same dispatch policy as batched `erf` above.
template <std::floating_point T>
void erfc(std::span<const T> x, std::span<T> result) {
    assert(x.size() == result.size());
    if constexpr (std::same_as<T, float>) {
        if (x.size() >= detail::kSpecialFunctionsGpuDispatchThreshold) {
            detail::erfcGpu(x, result);
            return;
        }
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = erfc(x[i]);
    }
}

/// The gamma function, `Gamma(n) = (n-1)!` for a positive integer `n` and the
/// standard analytic continuation elsewhere, needed by the Maxwell-Boltzmann
/// speed distribution's normalization.
template <std::floating_point T>
[[nodiscard]] T gamma(T x) {
    return detail::gammaOf(x);
}

/// `log(|Gamma(x)|)`, directly rather than via `log(gamma(x))`: `gamma`
/// overflows past `x` in the low hundreds at double precision, while its
/// logarithm stays representable far beyond that.
template <std::floating_point T>
[[nodiscard]] T logGamma(T x) {
    return detail::lgammaOf(x);
}

/// Batched `gamma`; same dispatch policy as batched `erf` above.
template <std::floating_point T>
void gamma(std::span<const T> x, std::span<T> result) {
    assert(x.size() == result.size());
    if constexpr (std::same_as<T, float>) {
        if (x.size() >= detail::kSpecialFunctionsGpuDispatchThreshold) {
            detail::gammaGpu(x, result);
            return;
        }
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = gamma(x[i]);
    }
}

/// Batched `logGamma`; same dispatch policy as batched `erf` above.
template <std::floating_point T>
void logGamma(std::span<const T> x, std::span<T> result) {
    assert(x.size() == result.size());
    if constexpr (std::same_as<T, float>) {
        if (x.size() >= detail::kSpecialFunctionsGpuDispatchThreshold) {
            detail::logGammaGpu(x, result);
            return;
        }
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = logGamma(x[i]);
    }
}

/// The associated Legendre polynomial `P_n^m(x)`, `0 <= m <= n`,
/// `-1 <= x <= 1`, with the Condon-Shortley phase `(-1)^m` included (the
/// convention `Physics/Gravity/SphericalHarmonics.hpp` and most physics
/// references use; some mathematics references omit it).
///
/// Computed by the standard upward recurrence rather than the textbook
/// closed forms, for the same numerical-stability reason `Math/Calculus.hpp`
/// prefers Richardson extrapolation over a single wide-step difference:
/// starting from the closed form for `P_m^m`,
///
/// ```
/// P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
/// ```
///
/// built up one factor at a time so nothing computes `(2m-1)!!` as a separate
/// intermediate that could overflow before the `(1-x^2)^(m/2)` factor brings
/// it back down, then climbing to `P_n^m` via
///
/// ```
/// (n-m) P_n^m(x) = x(2n-1) P_{n-1}^m(x) - (n+m-1) P_{n-2}^m(x)
/// ```
///
/// which only ever multiplies and adds terms already the right size, unlike
/// evaluating a high-degree polynomial from its coefficients directly.
template <std::floating_point T>
[[nodiscard]] T legendreP(unsigned n, unsigned m, T x) {
    assert(m <= n);
    assert(T{-1} <= x && x <= T{1});

    T pmm = T{1};
    if (m > 0) {
        const T oneMinusX2 = (T{1} - x) * (T{1} + x);
        const T somx2 = detail::sqrtOf(oneMinusX2);
        T fact = T{1};
        for (unsigned i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += T{2};
        }
    }
    if (n == m) {
        return pmm;
    }

    T pmmp1 = x * static_cast<T>(2 * m + 1) * pmm;
    if (n == m + 1) {
        return pmmp1;
    }

    T pll{};
    for (unsigned ll = m + 2; ll <= n; ++ll) {
        pll =
            (x * static_cast<T>(2 * ll - 1) * pmmp1 - static_cast<T>(ll + m - 1) * pmm) /
            static_cast<T>(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

/// The ordinary Legendre polynomial `P_n(x)`: the `m = 0` case of
/// `legendreP` above, where the Condon-Shortley phase is moot (`(-1)^0 = 1`).
template <std::floating_point T>
[[nodiscard]] T legendreP(unsigned n, T x) {
    return legendreP(n, 0U, x);
}

/// Batched `legendreP`, `n`/`m` fixed for the whole batch, one result per
/// element of `x`. Same dispatch policy as batched `erf` above.
template <std::floating_point T>
void legendreP(unsigned n, unsigned m, std::span<const T> x, std::span<T> result) {
    assert(x.size() == result.size());
    if constexpr (std::same_as<T, float>) {
        if (x.size() >= detail::kSpecialFunctionsGpuDispatchThreshold) {
            detail::legendrePGpu(n, m, x, result);
            return;
        }
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = legendreP(n, m, x[i]);
    }
}

/// The `m = 0` case of batched `legendreP` above.
template <std::floating_point T>
void legendreP(unsigned n, std::span<const T> x, std::span<T> result) {
    legendreP(n, 0U, x, result);
}

/// The Bessel function of the first kind, order 0: the radially symmetric
/// standing-wave solution of Bessel's equation `x^2 y'' + x y' + x^2 y = 0`
/// regular at the origin, the same equation a vibrating circular membrane
/// or a cylindrical waveguide's radial mode reduces to.
template <std::floating_point T>
[[nodiscard]] T besselJ0(T x) {
    return static_cast<T>(detail::besselJ0Double(static_cast<double>(x)));
}

/// The Bessel function of the first kind, order 1.
template <std::floating_point T>
[[nodiscard]] T besselJ1(T x) {
    return static_cast<T>(detail::besselJ1Double(static_cast<double>(x)));
}

/// The Bessel function of the first kind, order `n`, for any `n >= 0`.
template <std::floating_point T>
[[nodiscard]] T besselJ(unsigned n, T x) {
    return static_cast<T>(detail::besselJDouble(n, static_cast<double>(x)));
}

/// The Bessel function of the second kind (Weber/Neumann function), order
/// 0: the solution of the same equation as `besselJ0` that is singular
/// (logarithmically) at the origin instead of regular there, needed
/// whenever a cylindrical domain excludes the axis itself (an annulus, a
/// waveguide with an inner conductor). Undefined at `x <= 0`.
template <std::floating_point T>
[[nodiscard]] T besselY0(T x) {
    assert(T{0} < x);
    return static_cast<T>(detail::besselY0Double(static_cast<double>(x)));
}

/// The Bessel function of the second kind, order 1. Undefined at `x <= 0`.
template <std::floating_point T>
[[nodiscard]] T besselY1(T x) {
    assert(T{0} < x);
    return static_cast<T>(detail::besselY1Double(static_cast<double>(x)));
}

/// The Bessel function of the second kind, order `n`, for any `n >= 0`.
/// Undefined at `x <= 0`.
template <std::floating_point T>
[[nodiscard]] T besselY(unsigned n, T x) {
    assert(T{0} < x);
    return static_cast<T>(detail::besselYDouble(n, static_cast<double>(x)));
}

}  // namespace ysq
