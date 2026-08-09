# Physics/Optics API reference

Light propagation, gravitational lensing, frequency shift, refraction and
scattering through a medium, extended-source illumination, radiation
pressure, diffraction/interference, and relativistic aberration. Start with
[docs/physics/optics.md](../../physics/optics.md) for why the geodesic-based
pieces are the same idea; [src/Physics/README.md](../../../src/Physics/README.md)
has the `nullTangent` quadratic solve and the flat-space sweep derivation
in full.

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

## `Physics/Optics/RefractiveMedium.hpp`

A static, spherically symmetric medium of refractive index `n(r)`, exposed
as a `SpacetimeMetric`: not because it curves spacetime (it does not), but
because Fermat's principle makes a static medium's ray paths exactly the
spatial projections of null geodesics of an "ultra-static" optical metric.
`Propagation.hpp`'s `nullTangent`/`propagate` and `Lensing.hpp`'s shooting
technique all apply to this metric exactly as they do to `Schwarzschild`,
since nothing in them assumes which metric it is being handed.

```cpp
class RefractiveMedium {
public:
    RefractiveMedium() = default;   // vacuum: n = 1 everywhere
    RefractiveMedium(double radius, double surfaceRefractivity, double scaleHeight);

    double radius() const noexcept;
    double surfaceRefractivity() const noexcept;
    double scaleHeight() const noexcept;

    template <Numeric T> T refractiveIndex(T r) const;
    template <Numeric T> MetricTensor<T> components(const Vector4<T>& at) const;  // SpacetimeMetric
};

PhaseState<Vector4<double>>
refractiveMediumRayFromImpactParameter(const RefractiveMedium& medium, double impactParameter,
                                       double startRadius);
```

| Member | Description |
| --- | --- |
| `refractiveIndex(r)` | `n(r) = 1 + surfaceRefractivity * exp(-(r - radius) / scaleHeight)`, the same exponential shape Gladstone-Dale gives for an isothermal barometric atmosphere (`Thermodynamics.hpp`'s `isothermalAtmosphereDensity`). Defined for any `r`, not only `r >= radius()`. |
| `refractiveMediumRayFromImpactParameter` | The exact initial `PhaseState` for a null geodesic launched inward from `startRadius`, the same `E=1, L=impactParameter` normalization `Lensing.hpp`'s `schwarzschildRayFromImpactParameter` uses, adapted to this metric's constant `g_TT = -1`. |

```cpp
ysq::RefractiveMedium atmosphere(planetRadius, surfaceRefractivity, scaleHeight);
const auto start = ysq::refractiveMediumRayFromImpactParameter(atmosphere, b, startRadius);
const auto end = ysq::propagate(atmosphere, start, affineInterval, steps);
```

## `Physics/Optics/RayleighScattering.hpp`

A gas's per-molecule scattering cross-section and the exponential
number-density profile any isothermal barometric atmosphere has: why the
light surviving a long grazing path through an atmosphere comes out
reddened, not just dimmer.

```cpp
double rayleighCrossSection(double wavelength, double refractiveIndex, double numberDensity);
double exponentialNumberDensity(double r, double surfaceNumberDensity, double radius,
                                double scaleHeight);
double transmission(double opticalDepth);   // Beer-Lambert
```

| Function | Description |
| --- | --- |
| `rayleighCrossSection` | `sigma(lambda) = (24 pi^3/(lambda^4 N^2)) ((n^2-1)/(n^2+2))^2`, the standard result for particles much smaller than the wavelength. The `1/lambda^4` dependence is why blue scatters far more than red. |
| `exponentialNumberDensity` | The same isothermal barometric shape as `Thermodynamics::isothermalAtmosphereDensity`, in number-density terms. |
| `transmission(opticalDepth)` | `exp(-opticalDepth)`: the fraction of light surviving that optical depth. |

```cpp
const double sigma = ysq::rayleighCrossSection(wavelength, refractiveIndex, numberDensity);
const double survived = ysq::transmission(sigma * columnDensity);
```

## `Physics/Optics/Illumination.hpp`

General extended-source illumination: how much light from a spherical
source reaches a target, through straight occlusion and at most one
refracting/scattering medium. Composes `RefractiveMedium.hpp` and
`RayleighScattering.hpp`; nothing here knows what an eclipse is.

```cpp
struct OpaqueOccluder {
    Vec3 center{};
    double radius{};
};

struct RefractingOccluder {
    Vec3 center{};
    double opaqueRadius{};
    RefractiveMedium medium;
    double surfaceNumberDensity{};
    double scatteringScaleHeight{};
};

struct IlluminationResult {
    Vec3 transmission{1.0, 1.0, 1.0};   // per wavelength, in the caller's own order
    double geometricVisibility = 1.0;   // medium-independent: fraction with a clear line of sight
};

IlluminationResult illuminate(const Vec3& sourceCenter, double sourceRadius,
                              std::span<const OpaqueOccluder> opaqueOccluders,
                              const RefractingOccluder* refractingOccluder, const Vec3& target,
                              const std::array<double, 3>& wavelengths, int sourceSamples,
                              int stepBudget);

double discOcclusionFraction(const Vec3& point, const Vec3& sourceCenter, double sourceRadius,
                             const Vec3& occluderCenter, double occluderRadius);
```

| Function | Description |
| --- | --- |
| `illuminate` | Samples `sourceSamples` points around the source's limb; each unblocked sample contributes full transmission, each blocked only by `refractingOccluder`'s core attempts a bent path (a bisection over impact parameter, the same shooting technique `Lensing.hpp`'s `deflectionAngle` uses), each blocked by anything else contributes nothing. The result is the average over all samples; `sourceSamples`/`stepBudget` trade fidelity for speed. |
| `discOcclusionFraction` | Closed-form, `O(1)`, geometrically exact disc-overlap fraction for a single opaque occluder: real eclipse/transit geometry with no atmosphere or color, cheap enough to call once per particle every rendered frame. `1.0` fully lit, `0.0` fully covered, partial values include the annular-eclipse case. Returns `1.0` outright if `occluder` isn't nearer to `point` than the source is. |

```cpp
const ysq::IlluminationResult result = ysq::illuminate(
    sourceCenter, sourceRadius, opaqueOccluders, &refractingOccluder, target,
    {redWavelength, greenWavelength, blueWavelength}, sourceSamples, stepBudget);
```

## `Physics/Optics/RadiationPressure.hpp`

The force a body feels from the momentum light itself carries: turns
`Illumination.hpp`'s visibility/irradiance into an actual acceleration.

```cpp
Irradiance irradianceFromPointSource(RadiantPower luminosity, Length distance);

Force3 radiationPressureForce(Irradiance irradiance, Area crossSectionalArea,
                              double radiationPressureCoefficient,
                              const Vec3& directionFromSource);

Force3 radiationPressureForce(RadiantPower luminosity, const Length3& sourcePosition,
                              const Length3& bodyPosition, Length bodyRadius,
                              double radiationPressureCoefficient);
```

| Function | Description |
| --- | --- |
| `irradianceFromPointSource` | The inverse-square law: `luminosity` spread over the sphere of `distance`'s radius. |
| `radiationPressureForce(irradiance, ...)` | Momentum flux (`irradiance / c`) times area times a coefficient `Cr`: 1 for a perfect absorber, up to 2 for a perfect reflector. Pushes directly away from the source. |
| `radiationPressureForce(luminosity, ...)` | The "cannonball" model: a spherical body of `bodyRadius` facing an isotropic point source, cross-section approximated by its own silhouette `pi bodyRadius^2` regardless of attitude. Zero if the positions coincide. |

```cpp
const ysq::Force3 pressure = ysq::radiationPressureForce(
    starLuminosity, starPosition, bodyPosition, bodyRadius, radiationPressureCoefficient);
```

## `Physics/Optics/Diffraction.hpp`

Fraunhofer (far-field) diffraction and interference: wave-optics phenomena
none of `Optics`'s ray-propagation headers cover. General for whatever
aperture a caller describes.

```cpp
struct DiffractionPattern {
    std::vector<double> sinTheta;
    std::vector<double> intensity;   // normalized to 1 at its own maximum
};

DiffractionPattern fraunhoferDiffraction(std::span<const double> transmission,
                                         double apertureWidth, double wavelength);

double twoSlitIntensity(double sinTheta, double slitWidth, double slitSeparation,
                        double wavelength);
```

| Function | Description |
| --- | --- |
| `fraunhoferDiffraction` | The far-field pattern is exactly the squared-magnitude Fourier transform of the aperture's own transmission function (via `Math/FFT.hpp`, so `transmission.size()` must be a power of two). A bin whose implied `\|sinTheta\| > 1` is dropped rather than returned as nonphysical. |
| `twoSlitIntensity` | The classical Young's two-slit pattern as a closed form: the single-slit `sinc(beta)^2` envelope times the two-beam `cos(gamma)^2` interference factor. |

```cpp
const ysq::DiffractionPattern pattern =
    ysq::fraunhoferDiffraction(transmission, apertureWidth, wavelength);
```

## `Physics/Optics/Aberration.hpp`

Relativistic aberration: how a light ray's direction changes between two
frames in relative motion, the direction counterpart to
`FrequencyShift.hpp`'s Doppler shift. Exactly `Mechanics/Kinematics.hpp`'s
`relativisticVelocityAdd` applied to a velocity of magnitude `c`.

```cpp
Vec3 aberratedDirection(const Vec3& directionInOriginalFrame, const Velocity3& frameVelocity);

double aberratedCosine(double cosAngleInOriginalFrame, Speed boostSpeed);
```

| Function | Description |
| --- | --- |
| `aberratedDirection` | The general (any angle to the boost) form. Tracks the photon's own direction of *travel*, not the apparent direction to its source (`-directionInOriginalFrame`). |
| `aberratedCosine` | The classic 1D formula, `cos(theta') = (cos(theta) - beta)/(1 - beta cos(theta))`: the special case where the photon's direction and the boost already share an axis. |

```cpp
const ysq::Vec3 seenDirection = ysq::aberratedDirection(emittedDirection, observerVelocity);
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/optics)
and let us know.
