#pragma once

#include <Math/Dual.hpp>
#include <Math/Matrix2.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Matrix4.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>

namespace ysq {

/// Numerical differentiation and integration.
///
/// **The exact and approximate forms are deliberately not called the same
/// thing.** `gradient`, `jacobian` and `hessian` are the dual-number versions
/// and are exact to the last few bits; `numericalGradient` and friends are
/// finite differences and lose roughly half the available digits no matter how
/// the step is chosen. The plain name goes to the better method, so reaching
/// for the worse one has to be deliberate. The finite-difference forms are
/// still worth having: they need only a function of plain doubles, which is
/// what an external table, a measured dataset or a black-box callback gives
/// you.
///
/// The quadrature rules take a function of a plain scalar throughout. Nothing
/// stops them being handed a dual-valued integrand, which differentiates an
/// integral with respect to a parameter, but that is a use rather than a
/// requirement.

namespace detail {

/// Vector2/3/4 with a different element type. The dual-number differentiation
/// below needs to build a Vector3<Dual<T>> from a Vector3<T> generically.
template <class V, class NewT>
struct RebindScalar;

template <class T, class NewT>
struct RebindScalar<Vector2<T>, NewT> {
    using type = Vector2<NewT>;
};

template <class T, class NewT>
struct RebindScalar<Vector3<T>, NewT> {
    using type = Vector3<NewT>;
};

template <class T, class NewT>
struct RebindScalar<Vector4<T>, NewT> {
    using type = Vector4<NewT>;
};

template <class V, class NewT>
using RebindScalarT = typename RebindScalar<V, NewT>::type;

/// The square matrix that pairs with a given vector type.
template <class V>
struct MatrixFor;

template <class T>
struct MatrixFor<Vector2<T>> {
    using type = Matrix2<T>;
};

template <class T>
struct MatrixFor<Vector3<T>> {
    using type = Matrix3<T>;
};

template <class T>
struct MatrixFor<Vector4<T>> {
    using type = Matrix4<T>;
};

template <class V>
using MatrixForT = typename MatrixFor<V>::type;

}  // namespace detail

// --- Step sizes -------------------------------------------------------------

/// The step that balances truncation against cancellation for a one-sided
/// difference: sqrt(epsilon), scaled to the point.
///
/// A one-sided difference has truncation error of order h and rounding error
/// of order epsilon/h, and the sum is smallest where they meet. That is why a
/// forward difference cannot do better than about seven significant digits at
/// double precision, however carefully it is written.
template <std::floating_point T>
[[nodiscard]] T onesidedStep(T at) {
    return std::sqrt(std::numeric_limits<T>::epsilon()) *
           std::max(std::abs(at), T{1});
}

/// The same balance for a central difference, whose truncation error is of
/// order h^2: cbrt(epsilon), worth about eleven digits at double precision.
template <std::floating_point T>
[[nodiscard]] T centralStep(T at) {
    return std::cbrt(std::numeric_limits<T>::epsilon()) *
           std::max(std::abs(at), T{1});
}

// --- Finite differences -----------------------------------------------------

/// First order in the step. Two evaluations.
template <class F, std::floating_point T>
[[nodiscard]] T forwardDifference(F&& f, T at, T step) {
    // Rounding the step to a representable difference makes the denominator
    // exactly the distance actually used, which recovers a bit of accuracy for
    // free.
    const volatile T shifted = at + step;
    const T exactStep = static_cast<T>(shifted) - at;
    return (f(at + exactStep) - f(at)) / exactStep;
}

template <class F, std::floating_point T>
[[nodiscard]] T forwardDifference(F&& f, T at) {
    return forwardDifference(f, at, onesidedStep(at));
}

template <class F, std::floating_point T>
[[nodiscard]] T backwardDifference(F&& f, T at, T step) {
    const volatile T shifted = at - step;
    const T exactStep = at - static_cast<T>(shifted);
    return (f(at) - f(at - exactStep)) / exactStep;
}

template <class F, std::floating_point T>
[[nodiscard]] T backwardDifference(F&& f, T at) {
    return backwardDifference(f, at, onesidedStep(at));
}

/// Second order in the step, because the first-order error terms on the two
/// sides cancel. Same two evaluations as a one-sided difference, for two extra
/// digits.
template <class F, std::floating_point T>
[[nodiscard]] T centralDifference(F&& f, T at, T step) {
    const volatile T shifted = at + step;
    const T exactStep = static_cast<T>(shifted) - at;
    return (f(at + exactStep) - f(at - exactStep)) / (T{2} * exactStep);
}

template <class F, std::floating_point T>
[[nodiscard]] T centralDifference(F&& f, T at) {
    return centralDifference(f, at, centralStep(at));
}

/// The second derivative, second order in the step.
///
/// The step is larger than for a first derivative because the denominator is
/// h^2: cancellation bites twice as hard, so the balance point moves to the
/// fourth root of epsilon.
template <class F, std::floating_point T>
[[nodiscard]] T secondCentralDifference(F&& f, T at, T step) {
    return (f(at + step) - T{2} * f(at) + f(at - step)) / (step * step);
}

template <class F, std::floating_point T>
[[nodiscard]] T secondCentralDifference(F&& f, T at) {
    const T step = std::pow(std::numeric_limits<T>::epsilon(), T{0.25}) *
                   std::max(std::abs(at), T{1});
    return secondCentralDifference(f, at, step);
}

/// Richardson extrapolation of central differences on a halving step.
///
/// Each level cancels the next term in the error expansion, so `levels` of it
/// reaches order 2 * levels in the step, at the cost of two evaluations per
/// level. It runs out of road once rounding dominates, which for double
/// precision is around four levels; asking for more makes the answer worse.
template <class F, std::floating_point T>
[[nodiscard]] T richardsonDerivative(F&& f, T at, T step,
                                     std::size_t levels = 4) {
    std::array<T, 8> table{};
    const std::size_t used = std::min<std::size_t>(levels, table.size());

    T current = step;
    for (std::size_t i = 0; i < used; ++i) {
        table[i] = centralDifference(f, at, current);
        current /= T{2};

        // Neville, in place: each pass folds the previous column in.
        T factor = T{4};
        for (std::size_t j = i; j-- > 0;) {
            table[j] = table[j + 1] + (table[j + 1] - table[j]) / (factor - T{1});
            factor *= T{4};
        }
    }
    return table[0];
}

template <class F, std::floating_point T>
[[nodiscard]] T richardsonDerivative(F&& f, T at) {
    // A deliberately coarse starting step: extrapolation needs room to halve
    // before rounding takes over.
    return richardsonDerivative(f, at, std::cbrt(std::numeric_limits<T>::epsilon()) *
                                           T{100} * std::max(std::abs(at), T{1}));
}

// --- Exact derivatives, through dual numbers --------------------------------

/// The gradient of a scalar field, exactly.
///
/// f is called with the vector's element type replaced by Dual, so it has to
/// be generic: a lambda taking `auto` is the usual way. One evaluation per
/// component.
template <class F, class V>
[[nodiscard]] V gradient(F&& f, const V& at) {
    using T = typename V::value_type;
    using Seeded = detail::RebindScalarT<V, Dual<T>>;

    V result{};
    for (std::size_t i = 0; i < V::size(); ++i) {
        Seeded seeded{};
        for (std::size_t j = 0; j < V::size(); ++j) {
            seeded[j] = Dual<T>{at[j], (i == j) ? T{1} : T{0}};
        }
        result[i] = f(seeded).derivative;
    }
    return result;
}

/// The Jacobian of a vector field: entry (i, j) is d f_i / d x_j.
template <class F, class V>
[[nodiscard]] detail::MatrixForT<V> jacobian(F&& f, const V& at) {
    using T = typename V::value_type;
    using Seeded = detail::RebindScalarT<V, Dual<T>>;

    detail::MatrixForT<V> result{};
    for (std::size_t j = 0; j < V::size(); ++j) {
        Seeded seeded{};
        for (std::size_t k = 0; k < V::size(); ++k) {
            seeded[k] = Dual<T>{at[k], (j == k) ? T{1} : T{0}};
        }
        const Seeded evaluated = f(seeded);
        for (std::size_t i = 0; i < V::size(); ++i) {
            result(i, j) = evaluated[i].derivative;
        }
    }
    return result;
}

/// The Hessian of a scalar field, exactly, by nesting duals two deep.
///
/// Symmetric for any twice-differentiable f, and this computes both triangles
/// rather than assuming it, so a test can check the symmetry instead of
/// getting it by construction.
template <class F, class V>
[[nodiscard]] detail::MatrixForT<V> hessian(F&& f, const V& at) {
    using T = typename V::value_type;
    using Inner = Dual<T>;
    using Seeded = detail::RebindScalarT<V, Dual<Inner>>;

    detail::MatrixForT<V> result{};
    for (std::size_t i = 0; i < V::size(); ++i) {
        for (std::size_t j = 0; j < V::size(); ++j) {
            Seeded seeded{};
            for (std::size_t k = 0; k < V::size(); ++k) {
                seeded[k] = Dual<Inner>{Inner{at[k], (k == i) ? T{1} : T{0}},
                                        Inner{(k == j) ? T{1} : T{0}, T{0}}};
            }
            result(i, j) = f(seeded).derivative.derivative;
        }
    }
    return result;
}

// --- The same three by finite difference -------------------------------------

/// For an f that only accepts plain scalars. Central differences, component by
/// component.
template <class F, class V>
[[nodiscard]] V numericalGradient(F&& f, const V& at) {
    using T = typename V::value_type;

    V result{};
    for (std::size_t i = 0; i < V::size(); ++i) {
        const T step = centralStep(at[i]);
        V forward = at;
        V backward = at;
        forward[i] += step;
        backward[i] -= step;
        result[i] = (f(forward) - f(backward)) / (T{2} * step);
    }
    return result;
}

template <class F, class V>
[[nodiscard]] detail::MatrixForT<V> numericalJacobian(F&& f, const V& at) {
    using T = typename V::value_type;

    detail::MatrixForT<V> result{};
    for (std::size_t j = 0; j < V::size(); ++j) {
        const T step = centralStep(at[j]);
        V forward = at;
        V backward = at;
        forward[j] += step;
        backward[j] -= step;

        const V difference = (f(forward) - f(backward)) / (T{2} * step);
        for (std::size_t i = 0; i < V::size(); ++i) {
            result(i, j) = difference[i];
        }
    }
    return result;
}

template <class F, class V>
[[nodiscard]] detail::MatrixForT<V> numericalHessian(F&& f, const V& at) {
    using T = typename V::value_type;

    detail::MatrixForT<V> result{};
    const T fourthRoot = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});

    for (std::size_t i = 0; i < V::size(); ++i) {
        for (std::size_t j = 0; j < V::size(); ++j) {
            const T stepI = fourthRoot * std::max(std::abs(at[i]), T{1});
            const T stepJ = fourthRoot * std::max(std::abs(at[j]), T{1});

            if (i == j) {
                V forward = at;
                V backward = at;
                forward[i] += stepI;
                backward[i] -= stepI;
                result(i, j) =
                    (f(forward) - T{2} * f(at) + f(backward)) / (stepI * stepI);
                continue;
            }

            V pp = at;
            V pm = at;
            V mp = at;
            V mm = at;
            pp[i] += stepI;
            pp[j] += stepJ;
            pm[i] += stepI;
            pm[j] -= stepJ;
            mp[i] -= stepI;
            mp[j] += stepJ;
            mm[i] -= stepI;
            mm[j] -= stepJ;

            result(i, j) =
                (f(pp) - f(pm) - f(mp) + f(mm)) / (T{4} * stepI * stepJ);
        }
    }
    return result;
}

// --- Quadrature -------------------------------------------------------------

/// The composite trapezoid rule. Second order in the interval width.
template <class F, std::floating_point T>
[[nodiscard]] T trapezoid(F&& f, T lower, T upper, std::size_t intervals) {
    if (intervals == 0) {
        return T{0};
    }
    const T width = (upper - lower) / static_cast<T>(intervals);
    T total = (f(lower) + f(upper)) / T{2};
    for (std::size_t i = 1; i < intervals; ++i) {
        total += f(lower + static_cast<T>(i) * width);
    }
    return total * width;
}

/// The composite Simpson rule. Fourth order, and exact for any cubic, which is
/// one more degree than the quadratic it is derived from.
///
/// `intervals` is rounded up to an even number, since the rule pairs them.
template <class F, std::floating_point T>
[[nodiscard]] T simpson(F&& f, T lower, T upper, std::size_t intervals) {
    if (intervals < 2) {
        intervals = 2;
    }
    if (intervals % 2 != 0) {
        ++intervals;
    }

    const T width = (upper - lower) / static_cast<T>(intervals);
    T total = f(lower) + f(upper);
    for (std::size_t i = 1; i < intervals; ++i) {
        const T at = lower + static_cast<T>(i) * width;
        total += ((i % 2 == 0) ? T{2} : T{4}) * f(at);
    }
    return total * width / T{3};
}

namespace detail {

template <class F, std::floating_point T>
[[nodiscard]] T adaptiveSimpsonStep(F&& f, T lower, T upper, T atLower, T atMid,
                                    T atUpper, T whole, T tolerance, int depth) {
    const T mid = (lower + upper) / T{2};
    const T leftMid = (lower + mid) / T{2};
    const T rightMid = (mid + upper) / T{2};
    const T atLeftMid = f(leftMid);
    const T atRightMid = f(rightMid);

    const T left = (mid - lower) / T{6} * (atLower + T{4} * atLeftMid + atMid);
    const T right = (upper - mid) / T{6} * (atMid + T{4} * atRightMid + atUpper);
    const T refined = left + right;

    // A non-finite estimate can never satisfy the error test below, because
    // every comparison against a NaN is false. Without this the recursion runs
    // to full depth in *every* branch rather than in a few, which is 2^maxDepth
    // calls: about 10^12 at the default, indistinguishable from a hang. An
    // integrand that goes non-finite anywhere is not one this rule can do
    // anything with, so propagate and stop.
    if (!std::isfinite(refined)) {
        return refined;
    }

    // The classic error estimate: the difference between one Simpson panel and
    // two, over fifteen, which is what the next term in the expansion works
    // out to.
    if (depth <= 0 || std::abs(refined - whole) <= T{15} * tolerance) {
        return refined + (refined - whole) / T{15};
    }

    return adaptiveSimpsonStep(f, lower, mid, atLower, atLeftMid, atMid, left,
                               tolerance / T{2}, depth - 1) +
           adaptiveSimpsonStep(f, mid, upper, atMid, atRightMid, atUpper, right,
                               tolerance / T{2}, depth - 1);
}

}  // namespace detail

/// Simpson's rule that puts its evaluations where the integrand needs them.
///
/// A peaked integrand is the case this exists for: a composite rule has to use
/// its finest spacing everywhere, while this refines only the panels that fail
/// their error estimate.
template <class F, std::floating_point T>
[[nodiscard]] T adaptiveSimpson(F&& f, T lower, T upper, T tolerance,
                                int maxDepth = 40) {
    const T mid = (lower + upper) / T{2};
    const T atLower = f(lower);
    const T atMid = f(mid);
    const T atUpper = f(upper);
    const T whole = (upper - lower) / T{6} * (atLower + T{4} * atMid + atUpper);

    return detail::adaptiveSimpsonStep(f, lower, upper, atLower, atMid, atUpper,
                                       whole, tolerance, maxDepth);
}

/// Romberg integration: Richardson extrapolation of the trapezoid rule on a
/// repeatedly halved step. Converges very fast for a smooth integrand and not
/// at all for one with a kink, since the error expansion it extrapolates does
/// not exist there.
///
/// The default tolerance is scaled to the precision of T rather than fixed:
/// 1e-12 is a reasonable ask of a double and unreachable for a float, and
/// hard-coding it would not even compile for the latter.
template <class F, std::floating_point T>
[[nodiscard]] T romberg(F&& f, T lower, T upper, std::size_t maxLevels = 12,
                        T tolerance = std::numeric_limits<T>::epsilon() *
                                      T{1000}) {
    std::array<T, 24> previous{};
    std::array<T, 24> current{};
    const std::size_t levels = std::min<std::size_t>(maxLevels, previous.size());

    previous[0] = (upper - lower) / T{2} * (f(lower) + f(upper));

    std::size_t intervals = 1;
    for (std::size_t level = 1; level < levels; ++level) {
        intervals *= 2;
        const T width = (upper - lower) / static_cast<T>(intervals);

        // Only the newly inserted midpoints need evaluating.
        T added{};
        for (std::size_t i = 1; i < intervals; i += 2) {
            added += f(lower + static_cast<T>(i) * width);
        }
        current[0] = previous[0] / T{2} + width * added;

        T factor = T{4};
        for (std::size_t j = 1; j <= level; ++j) {
            current[j] =
                current[j - 1] + (current[j - 1] - previous[j - 1]) / (factor - T{1});
            factor *= T{4};
        }

        if (level > 1 && std::abs(current[level] - previous[level - 1]) < tolerance) {
            return current[level];
        }
        previous = current;
    }
    return previous[levels - 1];
}

namespace detail {

/// Gauss-Legendre nodes and weights on [-1, 1].
///
/// Transcribed rather than generated. A wrong digit is caught immediately by
/// the exactness test: an n-point rule integrates every polynomial up to
/// degree 2n - 1 exactly, and nothing close to the right table does that by
/// accident.
template <std::size_t N>
struct GaussLegendre;

template <>
struct GaussLegendre<2> {
    static constexpr std::array<double, 2> nodes{-0.5773502691896257,
                                                 0.5773502691896257};
    static constexpr std::array<double, 2> weights{1.0, 1.0};
};

template <>
struct GaussLegendre<3> {
    static constexpr std::array<double, 3> nodes{-0.7745966692414834, 0.0,
                                                 0.7745966692414834};
    static constexpr std::array<double, 3> weights{
        0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
};

template <>
struct GaussLegendre<4> {
    static constexpr std::array<double, 4> nodes{
        -0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
        0.8611363115940526};
    static constexpr std::array<double, 4> weights{
        0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
        0.3478548451374538};
};

template <>
struct GaussLegendre<5> {
    static constexpr std::array<double, 5> nodes{
        -0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831,
        0.9061798459386640};
    static constexpr std::array<double, 5> weights{
        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
        0.4786286704993665, 0.2369268850561891};
};

}  // namespace detail

/// Fixed-order Gauss-Legendre quadrature.
///
/// Exact for polynomials up to degree 2N - 1, which is what the freedom to
/// place the nodes buys over a rule with equally spaced ones. For a smooth
/// integrand this is far and away the most accuracy per evaluation; for one
/// with a kink or a pole it is no better than anything else.
template <std::size_t N, class F, std::floating_point T>
[[nodiscard]] T gaussLegendre(F&& f, T lower, T upper) {
    using Table = detail::GaussLegendre<N>;

    const T halfWidth = (upper - lower) / T{2};
    const T centre = (upper + lower) / T{2};

    T total{};
    for (std::size_t i = 0; i < N; ++i) {
        total += static_cast<T>(Table::weights[i]) *
                 f(centre + halfWidth * static_cast<T>(Table::nodes[i]));
    }
    return total * halfWidth;
}

}  // namespace ysq
