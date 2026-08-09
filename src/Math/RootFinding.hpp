#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>

namespace ysq {

/// Root finding: where a scalar function crosses zero.
///
/// Three methods, in the order they trade convergence speed for how little
/// they demand of the caller. Newton-Raphson converges quadratically but
/// needs a derivative and a starting guess close enough to the root that the
/// tangent line actually points toward it. Secant approximates that
/// derivative from the last two iterates instead, converging slightly slower
/// (order about 1.618, the golden ratio) for one less function to supply.
/// Bisection only needs the function itself and a bracket where it changes
/// sign, converging linearly, but it can never fail to converge to a root
/// inside that bracket and never needs a derivative at all.

/// One Newton-Raphson step: `x - f(x) / fPrime(x)`, iterated from
/// `initialGuess` until the step shrinks below `tolerance` or `maxIterations`
/// is reached, whichever comes first. Returns the last iterate either way,
/// since a caller with a well-posed problem and a reasonable starting guess
/// converges long before the cap, and one that doesn't is better served by
/// examining the returned value than by an exception it would just have to
/// catch.
template <class F, class FPrime, std::floating_point T>
[[nodiscard]] T newtonRaphson(F&& f, FPrime&& fPrime, T initialGuess,
                              T tolerance = std::numeric_limits<T>::epsilon() * T{100},
                              int maxIterations = 50) {
    T x = initialGuess;
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        const T step = f(x) / fPrime(x);
        x -= step;
        if (std::abs(step) < tolerance) {
            break;
        }
    }
    return x;
}

/// Newton-Raphson without an analytic derivative: `fPrime` is approximated by
/// a central difference at each iterate instead. Costs two evaluations of `f`
/// per step rather than one, and loses a little convergence speed to the
/// difference's own truncation error, in exchange for not needing a second
/// callable at all.
template <class F, std::floating_point T>
[[nodiscard]] T newtonRaphson(F&& f, T initialGuess,
                              T tolerance = std::numeric_limits<T>::epsilon() * T{100},
                              int maxIterations = 50) {
    const auto fPrime = [&f](T x) {
        const T step =
            std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(std::abs(x), T{1});
        return (f(x + step) - f(x - step)) / (T{2} * step);
    };
    return newtonRaphson(f, fPrime, initialGuess, tolerance, maxIterations);
}

/// The secant method: the same idea as Newton-Raphson, but the derivative is
/// replaced by the slope through the previous two iterates,
/// `(f(x1) - f(x0)) / (x1 - x0)`, so no derivative of any kind is needed.
/// Two starting points bracket nothing here, unlike bisection below; they
/// only have to be two distinct guesses for where the root might be.
template <class F, std::floating_point T>
[[nodiscard]] T secant(F&& f, T x0, T x1,
                       T tolerance = std::numeric_limits<T>::epsilon() * T{100},
                       int maxIterations = 50) {
    T fx0 = f(x0);
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        const T fx1 = f(x1);
        const T denominator = fx1 - fx0;
        if (denominator == T{0}) {
            break;
        }
        const T x2 = x1 - fx1 * (x1 - x0) / denominator;
        const T step = x2 - x1;
        x0 = x1;
        fx0 = fx1;
        x1 = x2;
        if (std::abs(step) < tolerance) {
            break;
        }
    }
    return x1;
}

/// Bisection: halves `[lower, upper]` toward the root every step, keeping
/// whichever half still has a sign change. Requires `f(lower)` and `f(upper)`
/// to have opposite signs (a precondition, not a check: there is no single
/// sensible value to return for a bracket that does not actually bracket a
/// root), and, unlike the two methods above, is guaranteed to converge to a
/// root inside that bracket no matter how ill-behaved `f` is between the
/// ends, since it never extrapolates past what it already knows.
template <class F, std::floating_point T>
[[nodiscard]] T bisection(F&& f, T lower, T upper,
                          T tolerance = std::numeric_limits<T>::epsilon() * T{100},
                          int maxIterations = 100) {
    T fLower = f(lower);

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        const T mid = (lower + upper) / T{2};
        if ((upper - lower) / T{2} < tolerance) {
            return mid;
        }
        const T fMid = f(mid);
        if ((fLower < T{0}) == (fMid < T{0})) {
            lower = mid;
            fLower = fMid;
        } else {
            upper = mid;
        }
    }
    return (lower + upper) / T{2};
}

}  // namespace ysq
