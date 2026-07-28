#pragma once

#include <Math/Scalar.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>

namespace ysq {

/// Flat spacetime: special relativity, no gravity. Cartesian chart, (T, x,
/// y, z) with T = c t, and the metric does not depend on position at all.
///
/// The base case every other metric in this module has to reduce to
/// somewhere, and the one whose Christoffel symbols are exactly zero
/// everywhere: there is nothing to differentiate in a constant tensor.
/// spacetime_metric.cpp checks that directly, and a geodesic here is exactly
/// a straight line at constant velocity, which spacetime_geodesic.cpp checks
/// as the solver's own base case.
struct Minkowski {
    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>&) const {
        MetricTensor<T> g{};
        g(0, 0) = T{-1};
        g(1, 1) = T{1};
        g(2, 2) = T{1};
        g(3, 3) = T{1};
        return g;
    }
};

}  // namespace ysq
