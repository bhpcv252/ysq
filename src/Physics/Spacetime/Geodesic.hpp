#pragma once

#include <Math/ODE.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Metric.hpp>

#include <cstddef>

namespace ysq {

/// The affine-parametrized geodesic equation,
///
///     d^2 x^mu / dlambda^2 + Gamma^mu_ab (dx^a/dlambda)(dx^b/dlambda) = 0
///
/// as a first-order system in (position, four-velocity), so Math's explicit
/// steppers can run it directly.
///
/// **Why this cannot use the symplectic steppers Gravity does.** Those take
/// an AccelerationField, a(t, q): an acceleration depending on position
/// alone. The geodesic equation's Gamma^mu_ab v^a v^b term is quadratic in
/// velocity, so it is not that shape, and geodesicSystem() below builds a
/// plain OdeSystem instead, meant for Rk4Stepper or the adaptive
/// Dormand-Prince stepper (Math/Integrators/RK4.hpp, Adaptive.hpp).
///
/// **One equation serves both timelike and null geodesics.** Which one you
/// get is entirely a property of the initial four-velocity's normalization,
/// metricProduct(metric, at, u, u) equal to -c^2 for a massive particle's
/// proper time or 0 for light, not of anything here; see Metric.hpp's
/// isTimelike/isNull. The affine parameter is proper time for a timelike
/// geodesic and has no invariant meaning for a null one, the ordinary
/// situation in relativity rather than a limitation of this solver.
template <SpacetimeMetric M>
[[nodiscard]] auto geodesicSystem(M metric) {
    return [metric](double, const PhaseState<Vector4<double>>& state) {
        const ChristoffelSymbols<double> gamma =
            christoffelSymbols(metric, state.position);

        Vector4<double> acceleration{};
        for (std::size_t mu = 0; mu < 4; ++mu) {
            double sum = 0.0;
            for (std::size_t alpha = 0; alpha < 4; ++alpha) {
                for (std::size_t beta = 0; beta < 4; ++beta) {
                    sum += gamma(mu, alpha, beta) * state.velocity[alpha] *
                           state.velocity[beta];
                }
            }
            acceleration[mu] = -sum;
        }
        return PhaseState<Vector4<double>>{state.velocity, acceleration};
    };
}

}  // namespace ysq
