#pragma once

#include <Math/Vector4.hpp>
#include <Physics/Spacetime/Metric.hpp>
#include <Units/Constants.hpp>

#include <cmath>

namespace ysq {

/// Doppler, gravitational and cosmological frequency shift are one
/// computation: the ratio of a photon's
/// four-momentum contracted with an observer's four-velocity, at emission
/// and at observation. Nothing distinguishes "kinds" of shift here; which
/// name applies is a property of the scenario, static observers near a mass
/// for gravitational, comoving observers in an expanding FLRW background
/// for cosmological, relatively moving observers in flat spacetime for
/// Doppler, not of the formula.

/// The ratio nu_observed / nu_emitted = 1 / (1 + z).
///
/// `photonTangentAtEmission` / `AtObservation` are the photon's four-velocity
/// where it meets each observer (Propagation.hpp's nullTangent or the state
/// propagate() returns); `emitterFourVelocity` / `observerFourVelocity` are
/// each observer's own four-velocity there. Only each photon tangent's
/// direction and relative scale along its own geodesic matter, not an
/// absolute normalization: k is null, and k . u is proportional to the
/// frequency that observer measures for whatever affine parametrization k
/// was given, with the proportionality constant cancelling in the ratio.
template <SpacetimeMetric M>
[[nodiscard]] double frequencyShift(const M& metric, const Vector4<double>& emissionEvent,
                                    const Vector4<double>& photonTangentAtEmission,
                                    const Vector4<double>& emitterFourVelocity,
                                    const Vector4<double>& observationEvent,
                                    const Vector4<double>& photonTangentAtObservation,
                                    const Vector4<double>& observerFourVelocity) {
    const double emitted = -metricProduct(metric, emissionEvent, photonTangentAtEmission,
                                          emitterFourVelocity);
    const double observed = -metricProduct(
        metric, observationEvent, photonTangentAtObservation, observerFourVelocity);
    return observed / emitted;
}

/// The four-velocity of an observer at fixed spatial coordinates: u = (uT,
/// 0, 0, 0), normalized by g_TT(at) uT^2 = -c^2. Defined wherever g_TT < 0,
/// which is every static or comoving observer this module's metrics
/// support: outside the horizon in Schwarzschild and Kerr, and always in
/// FLRW, where g_TT = -1 exactly and this reduces to u = (c, 0, 0, 0), a
/// comoving observer's four-velocity.
template <SpacetimeMetric M>
[[nodiscard]] Vector4<double> staticObserverFourVelocity(const M& metric,
                                                         const Vector4<double>& at) {
    const MetricTensor<double> g = metric.components(at);
    const double c = constants::speedOfLight.value();
    const double timeComponent = c / std::sqrt(-g(0, 0));
    return Vector4<double>{timeComponent, 0.0, 0.0, 0.0};
}

}  // namespace ysq
