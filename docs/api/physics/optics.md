# Physics/Optics API reference

Light propagation, gravitational lensing, and frequency shift: one null
geodesic computation, viewed three ways. Start with
[docs/physics/optics.md](../../physics/optics.md) for why they're the same
idea; [src/Physics/README.md](../../../src/Physics/README.md) has the
`nullTangent` quadratic solve and the flat-space sweep derivation in full.

## `Physics/Optics/Propagation.hpp`

Light as a null geodesic: no new physics beyond `Spacetime`'s geodesic
solver, evaluated with a null initial tangent.

```cpp
template <SpacetimeMetric M>
Vector4<double> nullTangent(const M& metric, const Vector4<double>& at,
                            const Vector3<double>& direction);

template <SpacetimeMetric M, class Observer>
PhaseState<Vector4<double>> propagate(const M& metric, const PhaseState<Vector4<double>>& start,
                                      double affineInterval, std::size_t steps, Observer&& observe);

template <SpacetimeMetric M>
PhaseState<Vector4<double>> propagate(const M& metric, const PhaseState<Vector4<double>>& start,
                                      double affineInterval, std::size_t steps);
```

| Function | Description |
| --- | --- |
| `nullTangent(metric, at, direction)` | A null four-velocity at `at` in spatial direction `direction`. Only `direction`'s *direction* matters, not magnitude: the null condition (`g_mu_nu u^mu u^nu = 0`, quadratic in `u^0`) fixes the magnitude. Returns the future-directed root. |
| `propagate(metric, start, affineInterval, steps, observe)` | Propagates from `start` across `affineInterval` in `steps` equal RK4 sub-steps, on the same geodesic system a timelike worldline uses. `observe` sees every step including the zeroth, matching `Math::integrate`'s convention. The no-observer overload discards it. |

```cpp
const ysq::Vec4 k = ysq::nullTangent(schwarzschild, emissionEvent, direction);
const ysq::PhaseState<ysq::Vec4> end =
    ysq::propagate(schwarzschild, {emissionEvent, k}, affineInterval, steps);
```

## `Physics/Optics/Lensing.hpp`

Gravitational lensing: the same null geodesic, evaluated past a source
massive enough for the bending to be measurable. Assumes the metric's chart
puts the radial coordinate in `position.y` (as `Schwarzschild`, `Kerr`, and
`FLRW` all do).

```cpp
template <SpacetimeMetric M>
double deflectionAngle(const M& metric, const PhaseState<Vector4<double>>& start,
                       double impactParameter, double startRadius,
                       double affineStep, std::size_t maxSteps);

PhaseState<Vector4<double>>
schwarzschildRayFromImpactParameter(const Schwarzschild& metric, double impactParameter,
                                    double startRadius);

double weakFieldDeflectionAngle(double schwarzschildRadius, double impactParameter);
```

| Function | Description |
| --- | --- |
| `deflectionAngle` | Propagates `start` and watches for the ray crossing back outward through `startRadius` (linearly interpolated between straddling steps), then corrects for the finite-`startRadius` flat-space sweep. **Returns NaN** if no such crossing occurs within `maxSteps`: the ray fell past the horizon, or the run wasn't long enough. |
| `schwarzschildRayFromImpactParameter` | Builds the exact initial `PhaseState` for a null geodesic in Schwarzschild's equatorial plane, launched inward from `startRadius` with the given impact parameter. Not a large-`startRadius` approximation: exact. |
| `weakFieldDeflectionAngle` | The weak-field analytic deflection, `4GM/(c^2 b) = 2 r_s / b`: what `deflectionAngle` should approach as `impactParameter` grows relative to `schwarzschildRadius`. |

**The flat-space sweep is not `pi`, even in flat space, unless
`startRadius` is infinite.** A straight line at perpendicular distance `b`
crosses a circle of radius `R` (`b < R`) sweeping `pi - 2 asin(b/R)`, not
`pi`. Comparing `deflectionAngle`'s measured sweep against bare `pi` is a
`startRadius`-dependent systematic error (roughly `2b/R` for `b << R`) that
does **not** shrink with a finer step: it's a geometry mistake, not a
truncation one. `deflectionAngle` compares against the correct flat-space
sweep instead.

```cpp
const auto start = ysq::schwarzschildRayFromImpactParameter(schwarzschild, b, startRadius);
const double deflection = ysq::deflectionAngle(schwarzschild, start, b, startRadius,
                                                affineStep, maxSteps);
// deflection should approach ysq::weakFieldDeflectionAngle(schwarzschild.schwarzschildRadius(), b)
// as b grows relative to the Schwarzschild radius
```

## `Physics/Optics/FrequencyShift.hpp`

Doppler, gravitational, and cosmological frequency shift as one computation:
the ratio of a photon's four-momentum contracted with an observer's
four-velocity, at emission and at observation. Nothing in the formula
distinguishes the three "kinds": which name applies is a property of the
scenario (relatively moving observers for Doppler, static observers near a
mass for gravitational, comoving observers in FLRW for cosmological).

```cpp
template <SpacetimeMetric M>
double frequencyShift(const M& metric, const Vector4<double>& emissionEvent,
                      const Vector4<double>& photonTangentAtEmission,
                      const Vector4<double>& emitterFourVelocity,
                      const Vector4<double>& observationEvent,
                      const Vector4<double>& photonTangentAtObservation,
                      const Vector4<double>& observerFourVelocity);
// returns nu_observed / nu_emitted = 1 / (1 + z)

template <SpacetimeMetric M>
Vector4<double> staticObserverFourVelocity(const M& metric, const Vector4<double>& at);
```

| Function | Description |
| --- | --- |
| `frequencyShift` | `> 1.0` is blueshift, `< 1.0` is redshift, `1.0` is no shift, whatever the actual cause. Only each photon tangent's direction and relative scale along its own geodesic matter, not an absolute normalization (the proportionality constant between `k.u` and observed frequency cancels in the ratio). |
| `staticObserverFourVelocity` | The four-velocity of an observer at fixed spatial coordinates, `u = (uT, 0, 0, 0)` normalized by `g_TT(at) uT^2 = -c^2`. Defined wherever `g_TT < 0`: outside the horizon in `Schwarzschild`/`Kerr`, and always in `FLRW` (where it reduces exactly to a comoving observer's four-velocity). Covers both the gravitational-redshift and cosmological-redshift observer cases from one function. |

```cpp
const double shift = ysq::frequencyShift(
    schwarzschild, emissionEvent, k, emitterFourVelocity,
    end.position, end.velocity, observerFourVelocity);
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/optics)
and let us know.
