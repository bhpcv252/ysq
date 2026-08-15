#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Math/RootFinding.hpp>
#include <Math/Scalar.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ysq {

namespace detail {

/// Same dispatch shape and per-call GPU overhead as
/// `Math/SpecialFunctions.hpp`'s batch-evaluation family, whose
/// `kSpecialFunctionsGpuDispatchThreshold` comment explains the
/// measurement (`benchmarks/compute_thresholds.cpp`, a floor rather than
/// an observed crossover). Uniquely named per header, matching
/// `Math/LinearSolve.hpp`'s ODR convention.
inline constexpr std::size_t kPolynomialGpuDispatchThreshold = 131072;

}  // namespace detail

/// A single-variable polynomial, coefficients ascending: `coefficients()[i]`
/// is the coefficient of `x^i`. Evaluation is Horner's method, with a
/// batched `operator()` overload alongside the scalar one (see its own
/// comment for the GPU dispatch policy); root finding is closed-form up to
/// degree 4 (the highest degree with a general formula in radicals) and
/// falls back to numeric deflation above that, via `Math/RootFinding.hpp`.
template <std::floating_point T>
class Polynomial {
public:
    explicit Polynomial(std::vector<T> coefficients)
        : m_coefficients(std::move(coefficients)) {
        assert(!m_coefficients.empty());
    }
    Polynomial(std::initializer_list<T> coefficients) : m_coefficients(coefficients) {
        assert(!m_coefficients.empty());
    }

    [[nodiscard]] const std::vector<T>& coefficients() const noexcept {
        return m_coefficients;
    }

    /// The stored degree: `coefficients().size() - 1`. Not trimmed of
    /// trailing (highest-order) zero coefficients — a caller that built this
    /// from, say, a fixed-size buffer may have deliberately zero-padded it.
    [[nodiscard]] std::size_t degree() const noexcept {
        return m_coefficients.size() - 1;
    }

    /// Horner's method: `n` multiply-adds for a degree-`n` polynomial,
    /// rather than computing each `x^i` separately.
    [[nodiscard]] T operator()(T x) const {
        T result = m_coefficients.back();
        for (std::size_t i = m_coefficients.size() - 1; i-- > 0;) {
            result = result * x + m_coefficients[i];
        }
        return result;
    }

    /// Batched evaluation, one result per element of `x`, this polynomial's
    /// coefficients fixed for the whole batch. Above a size threshold, and
    /// only for `T = float` (a `double` call always stays on the CPU loop
    /// below, gated with `if constexpr`, matching `Math/LinearSolve.hpp`'s
    /// own convention for the same reason: no GPU backend offers
    /// `float64`), dispatches through `Compute::defaultBackend()`
    /// transparently to the caller.
    void operator()(std::span<const T> x, std::span<T> result) const {
        assert(x.size() == result.size());
        if constexpr (std::same_as<T, float>) {
            if (x.size() >= detail::kPolynomialGpuDispatchThreshold) {
                defaultBackend().batchPolynomialEval(m_coefficients, x, result);
                return;
            }
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = (*this)(x[i]);
        }
    }

    [[nodiscard]] Polynomial derivative() const {
        if (m_coefficients.size() <= 1) {
            return Polynomial({T{0}});
        }
        std::vector<T> d(m_coefficients.size() - 1);
        for (std::size_t i = 1; i < m_coefficients.size(); ++i) {
            d[i - 1] = m_coefficients[i] * static_cast<T>(i);
        }
        return Polynomial(std::move(d));
    }

private:
    std::vector<T> m_coefficients;
};

/// The one real root of `a x + b = 0`. `nullopt` if `a` is zero (either no
/// root, if `b` is also nonzero, or every `x`, if it is — neither has a
/// single value to return).
template <std::floating_point T>
[[nodiscard]] std::optional<T> linearRealRoot(T a, T b) {
    if (a == T{0}) {
        return std::nullopt;
    }
    return -b / a;
}

/// The real roots of `a x^2 + b x + c = 0`, ascending, zero, one (a repeated
/// root) or two of them.
///
/// Uses the numerically stable form rather than the textbook quadratic
/// formula applied directly: computing `-b +- sqrt(discriminant)` loses
/// precision to cancellation whenever `b` and the square root are close in
/// magnitude and the same sign, which happens for exactly one of the two
/// roots every time `b != 0`. Computing one root as `q / a` (`q` chosen with
/// the sign that avoids the cancellation) and the other as `c / q` from
/// Vieta's formula (`x1 x2 = c/a`) sidesteps it for both.
template <std::floating_point T>
[[nodiscard]] std::vector<T> quadraticRealRoots(T a, T b, T c) {
    if (a == T{0}) {
        const std::optional<T> root = linearRealRoot(b, c);
        return root ? std::vector<T>{*root} : std::vector<T>{};
    }

    const T discriminant = b * b - T{4} * a * c;
    if (discriminant < T{0}) {
        return {};
    }
    if (discriminant == T{0}) {
        return {-b / (T{2} * a)};
    }

    const T sqrtDisc = detail::sqrtOf(discriminant);
    const T q = (b >= T{0}) ? -(b + sqrtDisc) / T{2} : -(b - sqrtDisc) / T{2};
    T x1 = q / a;
    T x2 = c / q;
    if (x1 > x2) {
        std::swap(x1, x2);
    }
    return {x1, x2};
}

/// The real roots of `a x^3 + b x^2 + c x + d = 0`, ascending: one or three
/// (a repeated root counted once, at the boundary between the two cases).
///
/// Depresses to `t^3 + p t + q = 0` via the standard shift `x = t - b/(3a)`,
/// then branches on the sign of the discriminant `(q/2)^2 + (p/3)^3`:
/// positive means one real root (Cardano's formula, real cube roots
/// throughout); negative means three distinct real roots, found by the
/// trigonometric substitution `t = 2 sqrt(-p/3) cos(...)` rather than
/// Cardano's formula directly, since Cardano's formula needs a complex cube
/// root to reach a real answer in this case (the "casus irreducibilis") and
/// this substitution never leaves the reals.
template <std::floating_point T>
[[nodiscard]] std::vector<T> cubicRealRoots(T a, T b, T c, T d) {
    if (a == T{0}) {
        return quadraticRealRoots(b, c, d);
    }

    const T bOverA = b / a;
    const T cOverA = c / a;
    const T dOverA = d / a;
    const T shift = bOverA / T{3};

    const T p = cOverA - bOverA * bOverA / T{3};
    const T q = T{2} * bOverA * bOverA * bOverA / T{27} - bOverA * cOverA / T{3} + dOverA;

    const T discriminant = (q * q) / T{4} + (p * p * p) / T{27};

    std::vector<T> roots;
    if (discriminant > T{0}) {
        const T sqrtDisc = detail::sqrtOf(discriminant);
        const T u = std::cbrt(-q / T{2} + sqrtDisc);
        const T v = std::cbrt(-q / T{2} - sqrtDisc);
        roots.push_back(u + v - shift);
    } else if (discriminant == T{0}) {
        const T u = std::cbrt(-q / T{2});
        roots.push_back(T{2} * u - shift);
        roots.push_back(-u - shift);
    } else {
        const T radius = T{2} * detail::sqrtOf(-p / T{3});
        const T cosArg = clamp((T{3} * q) / (p * radius), T{-1}, T{1});
        const T phi = std::acos(cosArg);
        for (int k = 0; k < 3; ++k) {
            const T angle = phi / T{3} - T{2} * kPi<T> * static_cast<T>(k) / T{3};
            roots.push_back(radius * std::cos(angle) - shift);
        }
        std::sort(roots.begin(), roots.end());
    }
    return roots;
}

/// The real roots of `a x^4 + b x^3 + c x^2 + d x + e = 0`, ascending, up to
/// four of them.
///
/// Depresses to `t^4 + p t^2 + q t + r = 0` via `x = t - b/(4a)`. A
/// vanishing `q` is the biquadratic special case, solved as a quadratic in
/// `u = t^2`. Otherwise, factors the quartic into two real quadratics,
/// `(t^2 + s t + u)(t^2 - s t + v)`, whose coefficients (matched against
/// `p`, `q`, `r` by expanding the product) reduce to `s^2` being a root of
/// the resolvent cubic `z^3 + 2p z^2 + (p^2 - 4r) z - q^2 = 0`: a real
/// quartic always factors into two real quadratics (its complex roots come
/// in conjugate pairs), so this cubic always has a nonnegative real root,
/// and once `s = sqrt(z)` is known, `u` and `v` follow directly and each
/// quadratic is solved by `quadraticRealRoots` above.
template <std::floating_point T>
[[nodiscard]] std::vector<T> quarticRealRoots(T a, T b, T c, T d, T e) {
    if (a == T{0}) {
        return cubicRealRoots(b, c, d, e);
    }

    const T bOverA = b / a;
    const T cOverA = c / a;
    const T dOverA = d / a;
    const T eOverA = e / a;
    const T shift = bOverA / T{4};
    const T b2 = bOverA * bOverA;

    const T p = cOverA - T{3} * b2 / T{8};
    const T q = dOverA - bOverA * cOverA / T{2} + b2 * bOverA / T{8};
    const T r =
        eOverA - bOverA * dOverA / T{4} + b2 * cOverA / T{16} - T{3} * b2 * b2 / T{256};

    const T epsilon = std::numeric_limits<T>::epsilon() * T{100};
    std::vector<T> tRoots;

    if (detail::absOf(q) < epsilon) {
        for (T u : quadraticRealRoots(T{1}, p, r)) {
            if (u > T{0}) {
                const T su = detail::sqrtOf(u);
                tRoots.push_back(su);
                tRoots.push_back(-su);
            } else if (u == T{0}) {
                tRoots.push_back(T{0});
            }
        }
    } else {
        const std::vector<T> zRoots =
            cubicRealRoots(T{1}, T{2} * p, p * p - T{4} * r, -q * q);
        const T z = std::max(*std::max_element(zRoots.begin(), zRoots.end()), T{0});
        const T s = detail::sqrtOf(z);

        if (s > epsilon) {
            const T u = (p + z) / T{2} - q / (T{2} * s);
            const T v = (p + z) / T{2} + q / (T{2} * s);
            for (T t : quadraticRealRoots(T{1}, s, u)) {
                tRoots.push_back(t);
            }
            for (T t : quadraticRealRoots(T{1}, -s, v)) {
                tRoots.push_back(t);
            }
        }
    }

    std::vector<T> roots;
    roots.reserve(tRoots.size());
    for (T t : tRoots) {
        roots.push_back(t - shift);
    }
    std::sort(roots.begin(), roots.end());
    return roots;
}

namespace detail {

/// Synthetic division of `p` by `(x - root)`: the quotient, one degree
/// lower. The remainder is not returned — `root` is assumed to already be a
/// root to the caller's required precision, so the remainder is expected to
/// be negligible, and a caller checking numeric root quality should check
/// `p(root)` directly rather than the remainder here.
template <std::floating_point T>
[[nodiscard]] Polynomial<T> deflate(const Polynomial<T>& p, T root) {
    const std::vector<T>& c = p.coefficients();
    const std::size_t n = p.degree();
    assert(n >= 1);

    std::vector<T> b(n);
    b[n - 1] = c[n];
    for (std::size_t k = n - 1; k > 0; --k) {
        b[k - 1] = c[k] + root * b[k];
    }
    return Polynomial<T>(std::move(b));
}

/// `1 + max_i |c_i / c_n|` (Cauchy's bound): every real or complex root of
/// `p` has magnitude at most this, so it is a safe search radius for a
/// numeric root finder with no other information about where the roots are.
template <std::floating_point T>
[[nodiscard]] T cauchyBound(const Polynomial<T>& p) {
    const std::vector<T>& c = p.coefficients();
    const std::size_t n = p.degree();
    T bound = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        bound = std::max(bound, detail::absOf(c[i] / c[n]));
    }
    return T{1} + bound;
}

/// One real root of `p`, found by sampling `[-bound, bound]` for a sign
/// change and refining it: bisection first, since it is guaranteed to stay
/// inside the bracket sampling already found, then Newton-Raphson to reach
/// full precision quickly from bisection's already-close estimate.
/// `nullopt` if no sign change turns up in the sampled points — a real root
/// might still exist between two samples that touch zero without crossing
/// it (a double real root), which this deliberately does not chase: the
/// deflation loop above it stops rather than looping forever on a root it
/// cannot bracket.
template <std::floating_point T>
[[nodiscard]] std::optional<T> findOneRealRootBySampling(const Polynomial<T>& p, T bound,
                                                         int samples = 400) {
    const Polynomial<T> derivative = p.derivative();

    T previousX = -bound;
    T previousValue = p(previousX);
    if (previousValue == T{0}) {
        return previousX;
    }

    for (int i = 1; i <= samples; ++i) {
        const T x = -bound + (T{2} * bound) * static_cast<T>(i) / static_cast<T>(samples);
        const T value = p(x);
        if (value == T{0}) {
            return x;
        }
        if ((previousValue < T{0}) != (value < T{0})) {
            const auto f = [&p](T at) { return p(at); };
            const T bracketed = bisection(f, previousX, x);
            const auto fPrime = [&derivative](T at) { return derivative(at); };
            return newtonRaphson(f, fPrime, bracketed);
        }
        previousX = x;
        previousValue = value;
    }
    return std::nullopt;
}

}  // namespace detail

/// Every real root of `p`, ascending. Degree 0 through 4 use the closed
/// forms above directly; above degree 4 (the highest with a general formula
/// in radicals — the Abel-Ruffini theorem is why there is no `quinticRealRoots`
/// to reach for), one real root is found numerically at a time via
/// `detail::findOneRealRootBySampling` and divided out
/// (`detail::deflate`), reducing the degree by one each time, until what
/// remains is degree 4 or lower and finishes via `quarticRealRoots`. Search
/// stops early, returning only the roots found so far, if a sampling pass
/// finds no sign change (a real root the sampling grid straddles without
/// crossing, or fewer real roots than the degree allows for).
template <std::floating_point T>
[[nodiscard]] std::vector<T> realRoots(const Polynomial<T>& p) {
    std::vector<T> trimmed = p.coefficients();
    while (trimmed.size() > 1 && trimmed.back() == T{0}) {
        trimmed.pop_back();
    }
    Polynomial<T> current(std::move(trimmed));

    std::vector<T> roots;
    while (current.degree() > 4) {
        const T bound = detail::cauchyBound(current);
        const std::optional<T> root = detail::findOneRealRootBySampling(current, bound);
        if (!root) {
            return roots;
        }
        roots.push_back(*root);
        current = detail::deflate(current, *root);
    }

    const std::vector<T>& c = current.coefficients();
    std::vector<T> lowDegreeRoots;
    switch (current.degree()) {
        case 0:
            break;
        case 1: {
            const std::optional<T> root = linearRealRoot(c[1], c[0]);
            if (root) {
                lowDegreeRoots = {*root};
            }
            break;
        }
        case 2:
            lowDegreeRoots = quadraticRealRoots(c[2], c[1], c[0]);
            break;
        case 3:
            lowDegreeRoots = cubicRealRoots(c[3], c[2], c[1], c[0]);
            break;
        case 4:
            lowDegreeRoots = quarticRealRoots(c[4], c[3], c[2], c[1], c[0]);
            break;
        default:
            break;
    }

    roots.insert(roots.end(), lowDegreeRoots.begin(), lowDegreeRoots.end());
    std::sort(roots.begin(), roots.end());
    return roots;
}

}  // namespace ysq
