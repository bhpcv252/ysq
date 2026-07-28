#pragma once

#include <Math/Integrators/RK4.hpp>
#include <Math/ODE.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Geodesic.hpp>
#include <Physics/Spacetime/Metric.hpp>

#include <cmath>
#include <cstddef>

namespace ysq {

/// Light as a null geodesic. The physical content of "light propagation" in
/// this module is entirely Spacetime's geodesic solver, evaluated with a
/// null initial tangent; nothing here adds new physics. What is here is
/// building the right null tangent, and running the solver over an affine
/// interval, so that does not have to be re-derived at every call site.

/// A null four-velocity at `at`, in the spatial direction `direction`. Only
/// `direction`'s direction matters, not its magnitude: the returned
/// four-velocity's magnitude is fixed by the null condition itself. Solves
/// g_mu_nu u^mu u^nu = 0, quadratic in u^0 given the spatial components, and
/// returns the future-directed root.
template <SpacetimeMetric M>
[[nodiscard]] Vector4<double> nullTangent(const M& metric, const Vector4<double>& at,
                                          const Vector3<double>& direction) {
    const MetricTensor<double> g = metric.components(at);

    const double a = g(0, 0);
    double b = 0.0;
    double c = 0.0;
    for (std::size_t i = 1; i < 4; ++i) {
        b += 2.0 * g(0, i) * direction[i - 1];
        for (std::size_t j = 1; j < 4; ++j) {
            c += g(i, j) * direction[i - 1] * direction[j - 1];
        }
    }

    const double discriminant = b * b - 4.0 * a * c;
    const double root = std::sqrt(discriminant);
    const double timeComponent =
        std::max((-b + root) / (2.0 * a), (-b - root) / (2.0 * a));

    return Vector4<double>{timeComponent, direction.x, direction.y, direction.z};
}

/// Propagates a light ray from `start` across an affine interval in
/// `steps` equal sub-steps, on the same geodesic system a timelike
/// worldline would use. `observe` sees every step, including the zeroth
/// (the starting state), the same convention Math's integrate() uses.
template <SpacetimeMetric M, class Observer>
[[nodiscard]] PhaseState<Vector4<double>>
propagate(const M& metric, const PhaseState<Vector4<double>>& start,
          double affineInterval, std::size_t steps, Observer&& observe) {
    const auto system = geodesicSystem(metric);
    Rk4Stepper<PhaseState<Vector4<double>>> stepper;

    const double h = affineInterval / static_cast<double>(steps);
    PhaseState<Vector4<double>> state = start;
    PhaseState<Vector4<double>> next = start;

    observe(0.0, state);
    for (std::size_t i = 0; i < steps; ++i) {
        stepper.step(system, static_cast<double>(i) * h, state, h, next);
        state = next;
        observe(static_cast<double>(i + 1) * h, state);
    }
    return state;
}

template <SpacetimeMetric M>
[[nodiscard]] PhaseState<Vector4<double>>
propagate(const M& metric, const PhaseState<Vector4<double>>& start,
          double affineInterval, std::size_t steps) {
    return propagate(metric, start, affineInterval, steps, [](double, const auto&) {});
}

}  // namespace ysq
