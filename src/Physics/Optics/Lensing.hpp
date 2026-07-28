#pragma once

#include <Math/Integrators/RK4.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Geodesic.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

namespace ysq {

/// Gravitational lensing: not a separate phenomenon from propagation, per
/// docs/architecture.md, just the same null geodesic evaluated past a
/// source massive enough for the bending to be measurable.
///
/// **The radial coordinate convention.** deflectionAngle assumes the
/// metric's chart puts a radial coordinate in position.y (as Schwarzschild,
/// Kerr and FLRW all do), and measures the total azimuth swept as a ray
/// travels from radius startRadius inward, past the source, back out to
/// startRadius.
///
/// **That sweep is not pi, even in flat space, unless startRadius is
/// infinite.** A straight line at perpendicular distance b from the origin
/// crosses a circle of radius R (b < R) at azimuth pi - asin(b/R) inbound
/// and asin(b/R) outbound, for a flat-space sweep of pi - 2 asin(b/R); the
/// excess of the true sweep over that, not over pi outright, is the
/// deflection. Comparing against pi instead is a startRadius-dependent
/// systematic error of exactly that missing term, roughly 2b/R for b << R,
/// which does not shrink with a finer step: it is a geometry mistake, not a
/// truncation one, and no amount of resolution fixes it. Confirmed directly
/// by taking impactParameter to within a few percent of startRadius, deep
/// in the regime where the true GR deflection is negligible: the measured
/// sweep converges to pi - 2 asin(b/R) there, not to pi.

/// The deflection angle of a null geodesic that starts and ends at
/// `startRadius`, given it was launched with impact parameter
/// `impactParameter` (see schwarzschildRayFromImpactParameter). Found by
/// propagating `start` and watching for the crossing back outward through
/// `startRadius`, linearly interpolated between the two straddling steps
/// the same way tests/integration/math_kepler.cpp finds a period crossing,
/// then correcting for the finite-startRadius flat-space sweep above.
/// Returns NaN if no such crossing occurs within `maxSteps` of size
/// `affineStep`, which means the ray never returned, either because it fell
/// in past the horizon or because the run was not long enough.
template <SpacetimeMetric M>
[[nodiscard]] double deflectionAngle(const M& metric,
                                     const PhaseState<Vector4<double>>& start,
                                     double impactParameter, double startRadius,
                                     double affineStep, std::size_t maxSteps) {
    const auto system = geodesicSystem(metric);
    Rk4Stepper<PhaseState<Vector4<double>>> stepper;

    PhaseState<Vector4<double>> state = start;
    PhaseState<Vector4<double>> next = start;

    double previousRadius = state.position.y;
    bool wasInside = false;

    for (std::size_t i = 0; i < maxSteps; ++i) {
        stepper.step(system, static_cast<double>(i) * affineStep, state, affineStep,
                     next);
        const double radius = next.position.y;

        if (radius < startRadius) {
            wasInside = true;
        }
        if (wasInside && previousRadius < startRadius && radius >= startRadius) {
            const double t = (startRadius - previousRadius) / (radius - previousRadius);
            const double azimuth =
                state.position.w + t * (next.position.w - state.position.w);
            const double sweep = std::abs(azimuth - start.position.w);
            const double flatSpaceSweep =
                kPi<double> - 2.0 * std::asin(impactParameter / startRadius);
            return sweep - flatSpaceSweep;
        }

        previousRadius = radius;
        state = next;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/// A null geodesic in Schwarzschild's equatorial plane, launched from
/// `startRadius` with impact parameter `impactParameter`, moving inward.
///
/// Built from the two quantities a static, spherically symmetric spacetime
/// conserves along a geodesic, energy E and angular momentum L (Killing
/// vectors of dT and dazimuth), normalized to E = 1 so impactParameter is
/// exactly L / E = L:
///
///     uT = 1 / (1 - r_s/startRadius)
///     uPhi = impactParameter / startRadius^2
///     ur from the null condition, negative root (inward)
///
/// This is the standard way an impact parameter is turned into an initial
/// condition, and it is exact, not an approximation valid only at large
/// startRadius: docs/physics.md has the full derivation.
[[nodiscard]] inline PhaseState<Vector4<double>>
schwarzschildRayFromImpactParameter(const Schwarzschild& metric, double impactParameter,
                                    double startRadius) {
    const double rs = metric.schwarzschildRadius();
    const double factor = 1.0 - rs / startRadius;

    const double uT = 1.0 / factor;
    const double uPhi = impactParameter / (startRadius * startRadius);
    const double urSquared =
        factor * (factor * uT * uT - startRadius * startRadius * uPhi * uPhi);
    const double ur = -std::sqrt(urSquared);

    return PhaseState<Vector4<double>>{
        Vector4<double>{0.0, startRadius, kPi<double> / 2.0, 0.0},
        Vector4<double>{uT, ur, 0.0, uPhi}};
}

/// The weak-field (impactParameter >> schwarzschildRadius) analytic
/// deflection angle, 4GM/(c^2 b) = 2 r_s / b, what deflectionAngle should
/// approach as impactParameter grows relative to schwarzschildRadius.
[[nodiscard]] inline double weakFieldDeflectionAngle(double schwarzschildRadius,
                                                     double impactParameter) {
    return 2.0 * schwarzschildRadius / impactParameter;
}

}  // namespace ysq
