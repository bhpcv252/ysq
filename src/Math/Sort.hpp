#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Math/Scalar.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace ysq {

/// Sorting and order statistics: a pure algorithm family with no physical
/// meaning of its own (Rule 1 in the project's Math/Physics/Applications
/// split -- sorting works exactly the same whether the values are
/// particle energies or quarterly revenue figures), so it lives here
/// rather than in Physics even though its one real consumer so far
/// (`Math/Statistics.hpp`'s `quantile`) is statistical in flavor.

namespace detail {

/// Measured on the development machine (Apple Silicon, Metal backend) by
/// `benchmarks/compute_thresholds.cpp`: bitonic sort's O(log^2 n)
/// sequential dispatches (each paying real CPU/GPU synchronization
/// overhead) meant the GPU path never won up to 65536 elements, the
/// largest size tried, so this is a measured floor (double the largest
/// size tried) rather than an observed crossover. Re-run the benchmark
/// and update this if the reference machine or backend ever changes.
inline constexpr std::size_t kSortGpuDispatchThreshold = 131072;

}  // namespace detail

/// Ascending sort, in place. Above a size threshold, and only for `T =
/// float` (`Compute`'s GPU interface is `float`-only, so a `double` call
/// always uses `std::sort` directly, gated with `if constexpr`, matching
/// every other GPU-dispatching function in this module), dispatches
/// through `Compute::defaultBackend()` via a bitonic sort (the standard
/// data-parallel sorting network; see
/// `Compute::ComputeBackend::sortAscending`'s own comment for how it
/// handles a `values.size()` that isn't a power of two). Below the
/// threshold, or for `double`, uses `std::sort`. The result is identical
/// either way -- the choice is purely about which path is faster at a
/// given size -- so a caller never needs to know or care which one ran.
template <std::floating_point T>
void sortInPlace(std::span<T> values) {
    if constexpr (std::same_as<T, float>) {
        if (values.size() >= detail::kSortGpuDispatchThreshold) {
            defaultBackend().sortAscending(values);
            return;
        }
    }
    std::sort(values.begin(), values.end());
}

/// Copies `values`, sorts the copy ascending (via `sortInPlace`), and
/// returns it.
template <std::floating_point T>
[[nodiscard]] std::vector<T> sorted(std::span<const T> values) {
    std::vector<T> result(values.begin(), values.end());
    sortInPlace<T>(result);
    return result;
}

/// The `k`-th smallest element, 0-indexed (`k = 0` is the minimum), via a
/// full sort. Simpler, and no slower in practice at this engine's scale,
/// than a dedicated parallel selection algorithm (median-of-medians, or a
/// GPU radix-select) would be to build and independently verify, and it
/// gets `sortInPlace`'s own GPU dispatch for free rather than needing one
/// of its own. `k < values.size()`.
template <std::floating_point T>
[[nodiscard]] T kthSmallest(std::span<const T> values, std::size_t k) {
    assert(k < values.size());
    return sorted<T>(values)[k];
}

/// The `k`-th largest element, 0-indexed (`k = 0` is the maximum):
/// `kthSmallest` from the other end. `k < values.size()`.
template <std::floating_point T>
[[nodiscard]] T kthLargest(std::span<const T> values, std::size_t k) {
    assert(k < values.size());
    return kthSmallest<T>(values, values.size() - 1 - k);
}

}  // namespace ysq
