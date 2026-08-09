#pragma once

#include <Math/Scalar.hpp>
#include <Math/SpecialFunctions.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>

namespace ysq {

/// Randomness, kept to a thin, deterministic-by-construction layer over
/// `<random>`: a fixed engine choice, distribution helpers that take the
/// engine explicitly rather than hiding a global one, and Monte Carlo
/// integration built on top of the uniform helper.
///
/// Every function here takes its engine by reference and returns rather than
/// mutates global state, on purpose: a caller that seeds an engine once and
/// threads it through gets bit-for-bit reproducible output across runs, the
/// same run twice, which a hidden global generator cannot promise (its state
/// depends on every previous call anywhere in the program, not just this
/// one's own history).

/// The one engine this module standardizes on. 64-bit Mersenne Twister:
/// good statistical quality and a long enough period for any simulation this
/// engine runs, and a single fixed choice means a seed alone is enough to
/// reproduce a run, rather than the seed and also which engine type produced
/// it.
using RandomEngine = std::mt19937_64;

[[nodiscard]] inline RandomEngine makeRandomEngine(std::uint64_t seed) {
    return RandomEngine(seed);
}

/// Uniform over `[lo, hi)`.
template <std::floating_point T>
[[nodiscard]] T uniformReal(RandomEngine& engine, T lo = T{0}, T hi = T{1}) {
    std::uniform_real_distribution<T> distribution(lo, hi);
    return distribution(engine);
}

/// Uniform over `[lo, hi]`, inclusive of both ends (`std::uniform_int_distribution`'s
/// own convention, unlike the real-valued overload above).
template <std::integral T>
[[nodiscard]] T uniformInt(RandomEngine& engine, T lo, T hi) {
    std::uniform_int_distribution<T> distribution(lo, hi);
    return distribution(engine);
}

/// Normally distributed with the given mean and standard deviation.
template <std::floating_point T>
[[nodiscard]] T normal(RandomEngine& engine, T mean = T{0}, T stddev = T{1}) {
    std::normal_distribution<T> distribution(mean, stddev);
    return distribution(engine);
}

/// Poisson-distributed event count with the given mean rate.
[[nodiscard]] inline int poisson(RandomEngine& engine, double mean) {
    std::poisson_distribution<int> distribution(mean);
    return distribution(engine);
}

/// The normal distribution's CDF, `P(X <= x)` for `X ~ Normal(mean, stddev)`,
/// via the identity `CDF(x) = (1 + erf((x - mean) / (stddev sqrt(2)))) / 2`:
/// the reason `Math/SpecialFunctions.hpp`'s `erf` exists at all.
template <std::floating_point T>
[[nodiscard]] T normalCdf(T x, T mean = T{0}, T stddev = T{1}) {
    return (T{1} + erf((x - mean) / (stddev * detail::sqrtOf(T{2})))) / T{2};
}

/// Monte Carlo integration of `f` over `[lower, upper]`: the average of `f`
/// at `samples` uniformly distributed points, scaled by the interval width.
///
/// Converges as `1/sqrt(samples)` regardless of dimension, which is worse
/// than `Math/Calculus.hpp`'s deterministic quadrature rules for a smooth
/// one-dimensional integrand (Gauss-Legendre or Romberg reach machine
/// precision in a handful of evaluations, not thousands) — this exists for
/// the cases those rules do not cover well: an integrand sampled from a
/// physical process rather than evaluated as a closed-form function, or (its
/// real strength, not exercised by this one-dimensional signature but the
/// reason to reach for it at all) a high-dimensional integral, where every
/// deterministic rule's cost grows exponentially with dimension and Monte
/// Carlo's does not grow with dimension at all.
template <class F, std::floating_point T>
[[nodiscard]] T monteCarloIntegrate(F&& f, T lower, T upper, std::size_t samples,
                                    RandomEngine& engine) {
    std::uniform_real_distribution<T> distribution(lower, upper);
    T total{};
    for (std::size_t i = 0; i < samples; ++i) {
        total += f(distribution(engine));
    }
    return (total / static_cast<T>(samples)) * (upper - lower);
}

}  // namespace ysq
