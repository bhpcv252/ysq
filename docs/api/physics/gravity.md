# Physics/Gravity API reference

The gravity ladder: Newtonian direct summation, Barnes-Hut, the 1PN
correction, general spherical-harmonics oblateness, and closed-form
Kepler orbital elements. Start with
[docs/physics/gravity.md](../../physics/gravity.md) for which rung to reach
for; [src/Physics/README.md](../../../src/Physics/README.md) has softening,
the opening-angle tradeoff, and the precession derivation in full.

## `Physics/Gravity/Newtonian.hpp`

`F = G m1 m2 / r^2`, toward the source: the weak-field, slow-motion limit
of general relativity and the rung that drives dynamical many-body systems.

```cpp
using GravitationalConstant = Quantity<dim::GravitationalConstant>;   // L^3 / (M T^2)

namespace constants {
    inline constexpr GravitationalConstant G{6.67430e-11};   // measured, CODATA 2018/2022, ~2.2e-5 relative uncertainty
}

Force3 newtonianForce(const Body& on, const Body& from);

Acceleration3 newtonianAcceleration(const Length3& at, std::span<const Body> sources,
                                    Length softening = Length::zero());

std::vector<Acceleration3> newtonianAccelerations(std::span<const Body> bodies,
                                                   Length softening = Length::zero());
// O(n^2) direct sum; see BarnesHutTree for O(n log n)

Energy newtonianPotentialEnergy(std::span<const Body> bodies,
                                Length softening = Length::zero());

class NewtonianField {
public:
    explicit NewtonianField(std::span<const Body> bodies, Length softening = Length::zero());
    NBodyState operator()(double time, const NBodyState& positions) const;
    // matches AccelerationField<NBodyState>: hand directly to VelocityVerletStepper<NBodyState> etc.
};
```

| Member | Description |
| --- | --- |
| `constants::G` | Measured, not one of `Units`'s seven SI-defining constants: it parameterizes one specific interaction, so it lives with the gravity that uses it, not in `Units/Constants.hpp`. Prefer a source's mass parameter GM directly (e.g. `constants::nominalSolarMassParameter`) where it's known: GM is what an orbit actually measures, and doesn't carry `G`'s uncertainty. |
| `softening` | Every function that takes one uses **Plummer softening**: `a = GM (r_j - r_i) / (\|r_j - r_i\|^2 + softening^2)^(3/2)`, the same force well above the softening length, finite rather than singular as bodies approach each other. Needed for direct-summation N-body integration. |
| `newtonianPotentialEnergy` | Must be called with the **same softening** the acceleration was integrated with: a conservation check is only meaningful for the softened system actually integrated, not the true `1/r` one. |
| `NewtonianField` | The `AccelerationField` to hand a `Math` stepper. Precomputes `G * mass` per body at construction. |

```cpp
ysq::NewtonianField field(bodies, softening);
ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
stepper.step(field, time, state, h, next);
```

### `NewtonianJerkField`

The time-derivative of `NewtonianField`'s own acceleration, one body at a
time: `jerk_i (from j) = G m_j [ v/R^3 - 3 (r.v) r/R^5 ]`, `r`/`v` the
relative position/velocity, `R^2 = |r|^2 + softening^2`. Matches
`IndividualJerkField` (see
[docs/api/physics/mechanics.md](mechanics.md#physicsmechanicshermitehpp)),
what `IndividualTimestepScheduler` needs alongside acceleration to predict
and correct a body's own state and to choose its own next step. **J2 jerk
is not implemented** -- future work, not a limitation for a body whose
`j2` actually is zero (every body the Solar System catalog names).

```cpp
class NewtonianJerkField {
public:
    explicit NewtonianJerkField(std::span<const Body> bodies,
                                Length softening = Length::zero());
    std::pair<Vec3, Vec3> operator()(std::size_t bodyIndex, const NBodyState& positions,
                                     const NBodyState& velocities) const;
};
```

## `Physics/Gravity/BarnesHut.hpp`

Approximate N-body gravity: a distant group of bodies is treated as one
point mass at their center of mass. Barnes & Hut, *A hierarchical O(N log N)
force-calculation algorithm*, Nature 324 (1986), 446-449.

```cpp
class BarnesHutTree {
public:
    explicit BarnesHutTree(std::span<const Body> bodies, double openingAngle = 0.5,
                           Length softening = Length::zero());
    NBodyState operator()(double time, const NBodyState& positions) const;
    // rebuilds the tree from `positions` every call; matches AccelerationField<NBodyState>
};
```

| Parameter | Description |
| --- | --- |
| `openingAngle` (theta) | A tree node of width `s` at distance `d` is accepted as a single mass when `s/d < theta`; otherwise the search recurses into its children. `theta = 0` degenerates to direct summation (every node opened); a larger `theta` accepts coarser, more distant approximations. `0.5` is the conventional default. |

The tree is **monopole only**: each internal node carries a total mass and
center of mass, not higher moments; a quadrupole correction would tighten
the error at fixed `theta` and isn't implemented. Rebuilt from scratch on
every call (the bodies have generally moved since the last one); that
rebuild cost is what buys O(N log N) per force evaluation over direct
summation's O(N^2).

```cpp
ysq::BarnesHutTree tree(bodies, 0.5, softening);
ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
stepper.step(tree, time, state, h, next);   // same AccelerationField shape as NewtonianField
```

## `Physics/Gravity/PostNewtonian.hpp`

The 1PN (first post-Newtonian) correction to the gravitational acceleration
on a test particle orbiting a dominant source: the correction that
produces perihelion precession, standard PPN form with `gamma = beta = 1`
(general relativity):

```cpp
Acceleration3 postNewtonianCorrection(const Body& testParticle, const Body& source);
```

```
a_1PN = (GM / (c^2 r^2)) * [ (4 GM/r - v^2) n + 4 (v . n) v ]
```

where `r`, `v` are the position and velocity of `testParticle` relative to
`source`, `n` is the unit vector from source to test particle, and `GM` uses
only the source's mass.

```cpp
double perihelionPrecessionPerOrbit(double gm, double semiMajorAxis, double eccentricity);
```

The relativistic perihelion advance a bound orbit accumulates over one full
orbit, in radians: `6 pi GM / (c^2 a (1 - e^2))`, the same closed form
`postNewtonianCorrection` is validated against. A caller not running a real
integrator over `postNewtonianCorrection` (a closed-form Kepler propagator,
say) can still show this real effect by dividing this by the orbital
period to get radians per second and feeding that into
`OrbitalElementsAtEpoch::precessionRatePerSecond` below.

**Scope: two bodies, one a test particle.** Exact in the limit that
`source`'s mass dominates (Mercury around the Sun, not two comparable
masses), the same regime the analytic precession formula it's validated
against assumes. A full N-body correction (the Einstein-Infeld-Hoffmann
equations, with cross terms between every pair) is not implemented.

The two rungs of the ladder **compose rather than replace one another**:
add this to the Newtonian acceleration:

```cpp
const ysq::Acceleration3 total =
    ysq::newtonianAcceleration(testParticle.position, {&source, 1}) +
    ysq::postNewtonianCorrection(testParticle, source);
```

### `RelativisticNBodySystem`

The practical N-body extension: direct-summation Newtonian gravity for
every pair (the same kernel `NewtonianField` uses, J2 included) plus this
correction for whichever bodies the caller assigns a primary to -- each
body against its own dominant nearby source (a moon's own planet, not the
Sun), not full Einstein-Infeld-Hoffmann cross terms between every pair.

```cpp
class RelativisticNBodySystem {
public:
    RelativisticNBodySystem(std::span<const Body> bodies, std::vector<int> primaryIndex,
                            Length softening = Length::zero());

    PhaseState<NBodyState> operator()(double time,
                                      const PhaseState<NBodyState>& state) const;
};
```

| Member | Description |
| --- | --- |
| `primaryIndex[i]` | Which body (by index into `bodies`) body `i`'s own 1PN correction is computed against. Negative, or equal to `i`, means no correction for that body. |
| `operator()(time, state)` | The full first-order system `d(position, velocity)/dt = (velocity, acceleration)`, for an explicit stepper. |

**Velocity-dependent, unlike `NewtonianField`**: the 1PN term needs
velocity, so this is an `OdeSystem` over `PhaseState<NBodyState>` for
`Rk4Stepper<PhaseState<NBodyState>>`, not an `AccelerationField` a
symplectic stepper (`VelocityVerletStepper` and the rest of
`Math/Integrators/Symplectic.hpp`) could take. That trade -- the
symplectic bounded-energy-error guarantee, for the real relativistic
physics -- is real and documented, not a limitation to route around.
`Applications/SolarSystem/main.cpp` is the worked example: a "Relativistic
corrections" toggle switches which jerk-providing system
`IndividualTimestepScheduler` is driven with, with `primaryIndex` built
once from each real body's own parent in the loaded catalog.

### `relativisticJerkTerm` and `RelativisticNBodyJerkSystem`

The jerk (accel/velocity/1PN counterparts) `IndividualTimestepScheduler`
needs alongside acceleration. `relativisticJerkTerm` is the
time-derivative of the 1PN acceleration term above -- a real, multi-term
closed-form expression, differentiated through `r`, `v`, `n = r/|r|`, and
the radial speed `s = v.n`, not a formula that can be looked up. Verified
against a finite-difference of the acceleration term
(`tests/unit/physics_gravity.cpp`) rather than trusted on the derivation
alone. `RelativisticNBodyJerkSystem` combines it with `NewtonianJerkField`
the same way `RelativisticNBodySystem` combines the acceleration terms;
matches `IndividualJerkField`.

```cpp
Vec3 relativisticJerkTerm(const Vec3& r, const Vec3& v, double gm);

class RelativisticNBodyJerkSystem {
public:
    RelativisticNBodyJerkSystem(std::span<const Body> bodies, std::vector<int> primaryIndex,
                                Length softening = Length::zero());
    std::pair<Vec3, Vec3> operator()(std::size_t bodyIndex, const NBodyState& positions,
                                     const NBodyState& velocities) const;
};
```

## `Physics/Gravity/SphericalHarmonics.hpp`

A general spherical-harmonics gravity field: the generalization of
`Newtonian.hpp`'s single J2 term to a body whose real shape needs more
than one oblateness coefficient (a tumbling asteroid, north-south or
longitudinal mass asymmetry J2 alone cannot represent).

```cpp
struct SphericalHarmonicsField {
    Mass mass;
    Length referenceRadius;
    std::vector<std::vector<double>> cosine;   // cosine[i][m]: degree n = i+2, order m = 0..n
    std::vector<std::vector<double>> sine;
};

double sphericalHarmonicsPotential(const SphericalHarmonicsField& field, const Vec3& position);
Acceleration3 sphericalHarmonicsAcceleration(const SphericalHarmonicsField& field,
                                             const Length3& position);
```

| Function | Description |
| --- | --- |
| `cosine`/`sine` | Standard geodesy `C_nm`/`S_nm` coefficients, unnormalized convention. `C_2,0 = -J2`, every other coefficient zero, reproduces `Newtonian.hpp`'s oblateness term exactly. Coefficients are expressed in the source body's own (generally rotating) frame; a caller rotates `position` into that frame first. |
| `sphericalHarmonicsPotential` | The geopotential in the geodesy sign convention (positive, increasing toward the source), `U(r) = (GM/r)[1 + sum_n (Re/r)^n sum_m P_n^m(sin phi)(C_nm cos(m lambda) + S_nm sin(m lambda))]`. |
| `sphericalHarmonicsAcceleration` | `+grad(U)`, by central-difference numerical differentiation (`Math/Calculus.hpp`'s `numericalGradient`) rather than an analytic partial derivative, sidestepping the associated Legendre functions' own pole singularities at the cost of finite-difference (about eight-digit) rather than machine precision. |

```cpp
ysq::SphericalHarmonicsField field{mass, referenceRadius, cosineCoefficients, sineCoefficients};
const ysq::Acceleration3 accel = ysq::sphericalHarmonicsAcceleration(field, bodyFramePosition);
```

## `Physics/Gravity/Kepler.hpp`

Classical orbital elements: the standard six-number description of an
unperturbed two-body ellipse, used to build a physically sensible initial
state (or a repeatedly re-evaluated closed-form propagation) for a real
n-body integrator to take over from. Nothing here is re-consulted once a
simulation actually starts.

```cpp
struct OrbitalElements {
    double semiMajorAxis;
    double eccentricity;
    double inclination;               // radians
    double longitudeOfAscendingNode;  // radians
    double argumentOfPeriapsis;       // radians
    double trueAnomaly;               // radians
};

struct KeplerStateVector {
    Vec3 position;
    Vec3 velocity;
};

KeplerStateVector stateVectorFromElements(const OrbitalElements& elements, double gm);

double trueAnomalyFromMeanAnomaly(double meanAnomaly, double eccentricity);
double keplerMeanMotion(double gm, double semiMajorAxis);
double keplerOrbitalPeriod(double gm, double semiMajorAxis);

struct OrbitalElementsAtEpoch {
    double semiMajorAxis;
    double eccentricity;
    double inclination;
    double longitudeOfAscendingNode;
    double argumentOfPeriapsis;
    double meanAnomalyAtEpoch;
    double precessionRatePerSecond = 0.0;
};

KeplerStateVector stateVectorAtTime(const OrbitalElementsAtEpoch& elements, double gm,
                                    double elapsedSeconds);
```

| Function | Description |
| --- | --- |
| `stateVectorFromElements` | The standard perifocal-to-reference-frame rotation applied to the perifocal-plane position/velocity formulas. `gm` is the *central* body's own gravitational parameter; the caller adds the central body's own position/velocity afterward. |
| `trueAnomalyFromMeanAnomaly` | Solves Kepler's equation `M = E - e sin(E)` for the eccentric anomaly (`Math/RootFinding.hpp`'s `newtonRaphson`), then converts to true anomaly by the standard half-angle relation. Published elements (JPL's included) give mean anomaly, not the true anomaly `stateVectorFromElements` needs. |
| `keplerMeanMotion`/`keplerOrbitalPeriod` | The two-body mean motion `n = sqrt(gm/a^3)` and period `2 pi / n`. |
| `OrbitalElementsAtEpoch` | An orbit anchored at an epoch for repeated re-evaluation at any later time. `precessionRatePerSecond` rotates `argumentOfPeriapsis` at a constant rate; zero reproduces a fixed ellipse exactly. `PostNewtonian.hpp`'s `perihelionPrecessionPerOrbit`, divided by the orbital period, gives the physical rate for a real relativistic effect. |
| `stateVectorAtTime` | `stateVectorFromElements` at `elapsedSeconds` after the epoch: advances mean anomaly (and, if nonzero, the periapsis) at constant cost regardless of how large `elapsedSeconds` is, unlike stepping a real integrator forward. |

```cpp
const ysq::KeplerStateVector initial = ysq::stateVectorFromElements(elements, gm);

ysq::OrbitalElementsAtEpoch precessing{elements.semiMajorAxis, elements.eccentricity,
                                       elements.inclination, elements.longitudeOfAscendingNode,
                                       elements.argumentOfPeriapsis, meanAnomalyAtEpoch,
                                       ysq::perihelionPrecessionPerOrbit(gm, elements.semiMajorAxis,
                                                                        elements.eccentricity) /
                                           ysq::keplerOrbitalPeriod(gm, elements.semiMajorAxis)};
const ysq::KeplerStateVector later = ysq::stateVectorAtTime(precessing, gm, elapsedSeconds);
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/gravity)
and let us know.
