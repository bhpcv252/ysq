#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Math/Scalar.hpp>
#include <Math/SpecialFunctions.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

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
///
/// `parallelUniformReal`/`parallelNormal` below are a second, additional
/// family: `RandomEngine` is one strictly sequential stream (call `n` costs
/// depend on every call before it), which cannot be split across GPU
/// threads without a full port of Mersenne Twister's own state machine and
/// a jump-ahead capability it does not straightforwardly offer. The
/// parallel family uses a counter-based generator (Philox4x32-10) instead,
/// where sample `i`'s value depends only on `(seed, offset + i)` with no
/// dependency on any other sample at all, so it is embarrassingly parallel
/// by construction and dispatches through `Compute::defaultBackend()`
/// above a size threshold; see
/// `Compute::ComputeBackend::batchUniformReal`'s own comment for the full
/// rationale. Reach for `RandomEngine` for a single deterministic
/// sequential stream; reach for the parallel family for bulk,
/// order-independent sampling at a size where GPU dispatch pays for
/// itself.

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

namespace detail {

/// Measured on the development machine (Apple Silicon, Metal backend) by
/// `benchmarks/compute_thresholds.cpp`: the smallest sample count at which
/// `batchUniformReal` actually beat the CPU reference. Re-run the
/// benchmark and update this if the reference machine or backend ever
/// changes.
inline constexpr std::size_t kRandomGpuDispatchThreshold = 32768;

/// Philox4x32-10 (Salmon, Moraes, Hadjidoukas & Schulten 2011), duplicated
/// from `Compute/CPU/CpuBackend.cpp`'s own copy of this algorithm rather
/// than shared: `Compute` has no dependency on `Math` at all (see
/// `src/Compute/README.md`), so nothing here can be a shared header, the
/// same reason every GPU backend's kernel source duplicates whatever CPU
/// logic it needs. This is Math's own below-threshold/`double` reference,
/// used exactly like `Math/LinearSolve.hpp`'s plain nested loops or
/// `Math/SpecialFunctions.hpp`'s scalar `erf(x[i])` calls are: never a call
/// into `ysq::CpuBackend` itself, so Math's non-dispatching path never
/// depends on `Compute` being linked in beyond the interface header.
inline void mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi,
                      std::uint32_t& lo) {
    const std::uint64_t product =
        static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    hi = static_cast<std::uint32_t>(product >> 32);
    lo = static_cast<std::uint32_t>(product);
}

inline std::array<std::uint32_t, 4> philox4x32_10(std::uint32_t c0, std::uint32_t c1,
                                                  std::uint32_t c2, std::uint32_t c3,
                                                  std::uint32_t k0, std::uint32_t k1) {
    constexpr std::uint32_t kM0 = 0xD2511F53u;
    constexpr std::uint32_t kM1 = 0xCD9E8D57u;
    constexpr std::uint32_t kW0 = 0x9E3779B9u;
    constexpr std::uint32_t kW1 = 0xBB67AE85u;
    for (int round = 0; round < 10; ++round) {
        std::uint32_t hi0{}, lo0{}, hi1{}, lo1{};
        mulhilo32(kM0, c0, hi0, lo0);
        mulhilo32(kM1, c2, hi1, lo1);
        const std::uint32_t nextC0 = hi1 ^ c1 ^ k0;
        const std::uint32_t nextC1 = lo1;
        const std::uint32_t nextC2 = hi0 ^ c3 ^ k1;
        const std::uint32_t nextC3 = lo0;
        c0 = nextC0;
        c1 = nextC1;
        c2 = nextC2;
        c3 = nextC3;
        k0 += kW0;
        k1 += kW1;
    }
    return {c0, c1, c2, c3};
}

template <std::floating_point T>
[[nodiscard]] T philoxUniform01(std::uint64_t seed, std::uint64_t counter) {
    const auto k0 = static_cast<std::uint32_t>(seed);
    const auto k1 = static_cast<std::uint32_t>(seed >> 32);
    const auto c0 = static_cast<std::uint32_t>(counter);
    const auto c1 = static_cast<std::uint32_t>(counter >> 32);
    const std::array<std::uint32_t, 4> out = philox4x32_10(c0, c1, 0u, 0u, k0, k1);
    return static_cast<T>(out[0]) * static_cast<T>(2.3283064365386963e-10);
}

template <std::floating_point T>
[[nodiscard]] T philoxNormal01(std::uint64_t seed, std::uint64_t counter) {
    const auto k0 = static_cast<std::uint32_t>(seed);
    const auto k1 = static_cast<std::uint32_t>(seed >> 32);
    const auto c0 = static_cast<std::uint32_t>(counter);
    const auto c1 = static_cast<std::uint32_t>(counter >> 32);
    const std::array<std::uint32_t, 4> out = philox4x32_10(c0, c1, 0u, 0u, k0, k1);
    // u1 in (0, 1], never exactly 0, so log(u1) is always finite: offsetting
    // the 32-bit word up by one before scaling avoids a special case for the
    // all-zero output word instead, matching batchNormal's GPU kernels
    // exactly.
    const T u1 = (static_cast<T>(out[0]) + T{1}) * static_cast<T>(2.3283064365386963e-10);
    const T u2 = static_cast<T>(out[1]) * static_cast<T>(2.3283064365386963e-10);
    return sqrtOf(T{-2} * std::log(u1)) * std::cos(T{2} * kPi<T> * u2);
}

}  // namespace detail

/// Parallel, counter-based sampling of `[lo, hi)`. Above a size threshold,
/// and only for `T = float` (`Compute`'s GPU interface is `float`-only, so
/// a `double` call always runs the identical Philox algorithm directly on
/// the CPU below, gated with `if constexpr`, matching every other
/// GPU-dispatching function in this module), dispatches through
/// `Compute::defaultBackend()`. Calling again with `offset += result.size()`
/// draws the next stretch of the same logical stream; see this header's own
/// top comment for the full rationale and how this differs from
/// `uniformReal`/`RandomEngine` above.
template <std::floating_point T>
void parallelUniformReal(std::uint64_t seed, std::uint64_t offset, std::span<T> result,
                         T lo = T{0}, T hi = T{1}) {
    if constexpr (std::same_as<T, float>) {
        if (result.size() >= detail::kRandomGpuDispatchThreshold) {
            defaultBackend().batchUniformReal(seed, offset, result);
            if (lo != T{0} || hi != T{1}) {
                for (T& value : result) {
                    value = lo + value * (hi - lo);
                }
            }
            return;
        }
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = lo + detail::philoxUniform01<T>(seed, offset + i) * (hi - lo);
    }
}

/// Parallel, counter-based normal sampling. Same dispatch policy as
/// `parallelUniformReal` above.
template <std::floating_point T>
void parallelNormal(std::uint64_t seed, std::uint64_t offset, std::span<T> result,
                    T mean = T{0}, T stddev = T{1}) {
    if constexpr (std::same_as<T, float>) {
        if (result.size() >= detail::kRandomGpuDispatchThreshold) {
            defaultBackend().batchNormal(seed, offset, result);
            if (mean != T{0} || stddev != T{1}) {
                for (T& value : result) {
                    value = mean + value * stddev;
                }
            }
            return;
        }
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = mean + detail::philoxNormal01<T>(seed, offset + i) * stddev;
    }
}

/// Monte Carlo integration of `f` over `[lower, upper]`, `1/sqrt(samples)`
/// convergence exactly like `monteCarloIntegrate` above, but drawing its
/// sample points from `parallelUniformReal` rather than a single sequential
/// `RandomEngine`: at a large `samples` count, sample-point generation
/// benefits from the same automatic GPU dispatch `parallelUniformReal`
/// gets. `f` itself is still evaluated on the CPU, one call per sample —
/// only the sample points are parallelized, since this signature has no
/// way to know how to evaluate an arbitrary `f` on a GPU.
template <class F, std::floating_point T>
[[nodiscard]] T parallelMonteCarloIntegrate(F&& f, T lower, T upper, std::size_t samples,
                                            std::uint64_t seed,
                                            std::uint64_t offset = 0) {
    std::vector<T> points(samples);
    parallelUniformReal<T>(seed, offset, points, lower, upper);
    T total{};
    for (T point : points) {
        total += f(point);
    }
    return (total / static_cast<T>(samples)) * (upper - lower);
}

}  // namespace ysq
