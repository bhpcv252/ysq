#pragma once

#include <Math/Calculus.hpp>
#include <Math/Scalar.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace ysq {

/// Unconstrained minimization of a scalar field `f: V -> T`, `V` one of
/// `Vector2`/`Vector3`/`Vector4` (the same types `Math/Calculus.hpp`'s
/// `gradient`/`numericalGradient` already work over).
///
/// Two families, for two situations. `gradientDescent` needs `f` to be
/// differentiable (its gradient is estimated by `numericalGradient`, so `f`
/// itself only ever needs plain scalars, not a dual-number overload) and
/// converges quickly when it is smooth. `nelderMead` needs nothing but
/// values of `f` at points — no derivative, estimated or exact — at the
/// cost of converging more slowly, and is what to reach for when `f` is
/// noisy, discontinuous, or otherwise not something a finite difference
/// could estimate a useful gradient from.

/// Backtracking line search: starting from `initialStep`, shrinks the step
/// by `shrink` until moving from `x` along `direction` by that step
/// satisfies the Armijo sufficient-decrease condition,
///
/// ```
/// f(x + step * direction) <= f(x) + c1 * step * (gradient . direction)
/// ```
///
/// A plain fixed step size either overshoots a steep region or creeps
/// through a shallow one; this adapts the step to the local shape of `f`
/// along the one direction that matters this iteration, without needing a
/// second derivative.
template <class F, class V>
[[nodiscard]] typename V::value_type backtrackingLineSearch(
    F&& f, const V& x, const V& direction, const V& gradient,
    typename V::value_type initialStep = typename V::value_type{1},
    typename V::value_type c1 = typename V::value_type{1} / typename V::value_type{10000},
    typename V::value_type shrink = typename V::value_type{0.5}, int maxIterations = 50) {
    using T = typename V::value_type;

    const T atX = f(x);
    const T directionalDerivative = dot(gradient, direction);

    T step = initialStep;
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        if (f(x + direction * step) <= atX + c1 * step * directionalDerivative) {
            return step;
        }
        step *= shrink;
    }
    return step;
}

/// Steepest descent: repeatedly steps opposite the numerical gradient, with
/// `backtrackingLineSearch` choosing how far each step goes. Stops once the
/// gradient's length falls below `tolerance` (a stationary point, to that
/// precision) or `maxIterations` is reached.
template <class F, class V>
[[nodiscard]] V
gradientDescent(F&& f, V x,
                typename V::value_type tolerance =
                    std::numeric_limits<typename V::value_type>::epsilon() *
                    typename V::value_type{1000},
                int maxIterations = 1000) {
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        const V gradientAtX = numericalGradient(f, x);
        if (length(gradientAtX) < tolerance) {
            break;
        }
        const V direction = -gradientAtX;
        const auto step = backtrackingLineSearch(f, x, direction, gradientAtX);
        x += direction * step;
    }
    return x;
}

/// The Nelder-Mead simplex method: maintains `n + 1` points (`n` the
/// dimension of `V`) and replaces the worst one each iteration by
/// reflecting, expanding or contracting it through the centroid of the
/// rest, only ever comparing values of `f` against each other — no
/// gradient, estimated or exact, anywhere in the algorithm.
///
/// The initial simplex is `initial` plus one point offset by `initialStep`
/// along each axis. Stops once the spread between the best and worst
/// values in the simplex falls below `tolerance`, or `maxIterations` is
/// reached.
template <class F, class V>
[[nodiscard]] V
nelderMead(F&& f, const V& initial,
           typename V::value_type initialStep = typename V::value_type{1} /
                                                typename V::value_type{10},
           typename V::value_type tolerance =
               std::numeric_limits<typename V::value_type>::epsilon() *
               typename V::value_type{1000},
           int maxIterations = 1000) {
    using T = typename V::value_type;
    const std::size_t n = V::size();

    std::vector<V> simplex;
    simplex.reserve(n + 1);
    simplex.push_back(initial);
    for (std::size_t i = 0; i < n; ++i) {
        V vertex = initial;
        vertex[i] += initialStep;
        simplex.push_back(vertex);
    }

    std::vector<T> values(n + 1);
    for (std::size_t i = 0; i <= n; ++i) {
        values[i] = f(simplex[i]);
    }

    constexpr T kAlpha{1};    // reflection
    constexpr T kGamma{2};    // expansion
    constexpr T kRho{0.5};    // contraction
    constexpr T kSigma{0.5};  // shrink

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        std::vector<std::size_t> order(n + 1);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&values](std::size_t a, std::size_t b) {
            return values[a] < values[b];
        });

        std::vector<V> sortedSimplex;
        std::vector<T> sortedValues;
        sortedSimplex.reserve(n + 1);
        sortedValues.reserve(n + 1);
        for (std::size_t index : order) {
            sortedSimplex.push_back(simplex[index]);
            sortedValues.push_back(values[index]);
        }
        simplex = std::move(sortedSimplex);
        values = std::move(sortedValues);

        if (values[n] - values[0] < tolerance) {
            break;
        }

        V centroid = V::zero();
        for (std::size_t i = 0; i < n; ++i) {
            centroid += simplex[i];
        }
        centroid *= (T{1} / static_cast<T>(n));

        const V& worst = simplex[n];
        const V reflected = centroid + (centroid - worst) * kAlpha;
        const T reflectedValue = f(reflected);

        if (reflectedValue < values[0]) {
            const V expanded = centroid + (reflected - centroid) * kGamma;
            const T expandedValue = f(expanded);
            if (expandedValue < reflectedValue) {
                simplex[n] = expanded;
                values[n] = expandedValue;
            } else {
                simplex[n] = reflected;
                values[n] = reflectedValue;
            }
            continue;
        }

        if (reflectedValue < values[n - 1]) {
            simplex[n] = reflected;
            values[n] = reflectedValue;
            continue;
        }

        bool contracted = false;
        if (reflectedValue < values[n]) {
            const V outside = centroid + (reflected - centroid) * kRho;
            const T outsideValue = f(outside);
            if (outsideValue <= reflectedValue) {
                simplex[n] = outside;
                values[n] = outsideValue;
                contracted = true;
            }
        } else {
            const V inside = centroid + (worst - centroid) * kRho;
            const T insideValue = f(inside);
            if (insideValue < values[n]) {
                simplex[n] = inside;
                values[n] = insideValue;
                contracted = true;
            }
        }
        if (contracted) {
            continue;
        }

        for (std::size_t i = 1; i <= n; ++i) {
            simplex[i] = simplex[0] + (simplex[i] - simplex[0]) * kSigma;
            values[i] = f(simplex[i]);
        }
    }

    const std::size_t bestIndex = static_cast<std::size_t>(
        std::min_element(values.begin(), values.end()) - values.begin());
    return simplex[bestIndex];
}

}  // namespace ysq
