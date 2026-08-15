# Physics

The engine's theories: what things are made of, and the laws that move them.
Organized by theory, not by application. The engine encodes what things are
and how they behave; an `Application` encodes a scenario: which things
exist, with what initial conditions, and what to measure. A planet's mass
belongs to an application; the gravitational force law belongs here. That
split is what makes `Mechanics`, `Gravity`, `Spacetime`, `Electromagnetism`,
`Fluids`, `Thermodynamics` and `Optics` siblings rather than one another's
dependencies: none of them needs to know about a specific scenario, so none
of them needs to know about the others' scenarios either.

**Target:** `ysq::Physics` (static)
**Depends on:** `ysq::Math`, `ysq::Units`, both `PUBLIC` since every header
here hands back a `Quantity` or a `Math` type. `ysq::Core` and `ysq::Compute`
are linked `PRIVATE`. `Gravity/Newtonian.hpp`'s `newtonianAccelerations()`
and `NewtonianField`, and `Electromagnetism/Field.hpp`'s `electricFields()`
and `magneticFields()`, all dispatch through `Compute::defaultBackend()`
above a size threshold (point-mass bodies only for gravity; see
`Newtonian.cpp`), falling back to CPU below it or when nothing better is
available. `Mechanics/Hermite.hpp`'s `IndividualTimestepScheduler` does the
same for finding the next body due to update, and `Fluids/SPH.hpp`'s
`computeDensityAndPressure`/`pressureAccelerations` for the SPH kernel sum
(direct O(n^2) with a distance cutoff on GPU, `Math/SpatialPartition::KdTree3`
on CPU). `Thermodynamics/HeatEquation3D.hpp`'s `step()`,
`Acoustics/Acoustic3D.hpp`'s `step()`, `Electromagnetism/Maxwell3D.hpp`'s
`step()`, and `Fluids/Eulerian3D.hpp`'s `step()` (three dispatches, one per
dimensional-split sweep axis) dispatch the same way for their grid stencil
update, each above its own cell-count threshold; the GPU kernel computes
periodic wraparound with modular index arithmetic rather than the CPU
path's ghost cells. See src/Compute/README.md.

This module is being built in stages; only what is listed below exists so
far.

## Contents

| Header | Purpose |
| --- | --- |
| `Physics/Body.hpp` | Matter state: mass, charge, position, momentum |
| `Physics/Mechanics/Frame.hpp` | Inertial reference frames, Galilean transform |
| `Physics/Mechanics/Kinematics.hpp` | Lorentz factor, four-velocity, proper time, relativistic velocity addition |
| `Physics/Mechanics/Dynamics.hpp` | `NBodyState`, the boundary between a span of `Body` and Math's integrators |
| `Physics/Mechanics/RigidBody.hpp` | Gravity-gradient torque, Euler's rotation equation, and diagonalizing an arbitrary inertia tensor |
| `Physics/Mechanics/Hermite.hpp` | 4th-order predictor-corrector integration and `IndividualTimestepScheduler`: each body its own step size, force-law-agnostic |
| `Physics/Mechanics/Spring.hpp` | Hooke's law with linear damping, anchor- or body-to-body |
| `Physics/Mechanics/Drag.hpp` | Linear (Stokes) and quadratic drag through a medium |
| `Physics/Mechanics/Collision.hpp` | Sphere-sphere and sphere-box overlap, impulse-based resolution |
| `Physics/Mechanics/Friction.hpp` | Coulomb friction: kinetic (sliding) and static (holding) |
| `Physics/Mechanics/Constraints.hpp` | Sequential-impulse distance and point constraints, Baumgarte-stabilized |
| `Physics/Continuum/Elasticity.hpp` | Hooke's law for a continuum; the bridge to Mechanics/Spring.hpp's spring constant |
| `Physics/Continuum/ElasticChain1D.hpp` | A 1D elastic bar-chain solved for static equilibrium via a direct linear solve |
| `Physics/Gravity/Newtonian.hpp` | Pairwise gravity, direct-sum N-body, potential energy, J2 oblateness, and `NewtonianJerkField` for `IndividualTimestepScheduler` |
| `Physics/Gravity/PostNewtonian.hpp` | The 1PN two-body correction (perihelion precession), `RelativisticNBodySystem`'s per-body-primary N-body extension of it, and their jerk counterparts |
| `Physics/Gravity/BarnesHut.hpp` | O(N log N) approximate N-body gravity |
| `Physics/Gravity/Kepler.hpp` | Closed-form two-body orbits: classical elements to state vectors, Kepler's equation, mean motion and period |
| `Physics/Gravity/SphericalHarmonics.hpp` | General geopotential gravity field: J2's generalization to arbitrary degree and order |
| `Physics/Spacetime/Metric.hpp` | The `SpacetimeMetric` concept, Christoffel symbols, causal character |
| `Physics/Spacetime/Minkowski.hpp` | Flat spacetime |
| `Physics/Spacetime/Schwarzschild.hpp` | Non-rotating mass |
| `Physics/Spacetime/Kerr.hpp` | Rotating mass |
| `Physics/Spacetime/FLRW.hpp` | Expanding universe, single-component analytic solutions |
| `Physics/Spacetime/Geodesic.hpp` | Timelike and null geodesics |
| `Physics/Spacetime/ADM.hpp` | The primitive 3+1 variables as `Grid3D` fields: a spacetime that evolves, not a prescribed one |
| `Physics/Spacetime/Bssn.hpp` | The BSSN reformulation, its evolution equations, and the moving-puncture gauge |
| `Physics/Spacetime/PunctureInitialData.hpp` | Brandt-Brügmann puncture data with Bowen-York extrinsic curvature |
| `Physics/Optics/Propagation.hpp` | Light as a null geodesic |
| `Physics/Optics/Lensing.hpp` | Gravitational deflection angle |
| `Physics/Optics/FrequencyShift.hpp` | Doppler, gravitational and cosmological shift, one formula |
| `Physics/Optics/RefractiveMedium.hpp` | A graded-index medium, exposed as a metric: refraction is a null geodesic too |
| `Physics/Optics/RayleighScattering.hpp` | Scattering cross-section and optical depth, general for any gas |
| `Physics/Optics/Illumination.hpp` | Extended-source visibility through occlusion, refraction and scattering |
| `Physics/Optics/RadiationPressure.hpp` | The force of light's own momentum: inverse-square irradiance, absorber/reflector pressure |
| `Physics/Optics/Diffraction.hpp` | Fraunhofer diffraction (via FFT, any aperture) and two-slit interference |
| `Physics/Optics/Aberration.hpp` | Relativistic aberration of a light ray's direction between two frames |
| `Physics/Electromagnetism/Field.hpp` | Point-charge E and B fields, quasi-static |
| `Physics/Electromagnetism/Lorentz.hpp` | Force on a charged body |
| `Physics/Electromagnetism/Maxwell.hpp` | 1D FDTD: a field that actually propagates |
| `Physics/Electromagnetism/Maxwell3D.hpp` | The full 3D Yee grid: six field components, two curl terms per update |
| `Physics/Acoustics/Acoustic.hpp` | 1D FDTD linear acoustic wave equation: structurally identical to Maxwell's own |
| `Physics/Acoustics/Acoustic3D.hpp` | 3D staggered (MAC) grid linear acoustics |
| `Physics/Fluids/SPH.hpp` | Smoothed Particle Hydrodynamics, Lagrangian, no grid |
| `Physics/Fluids/Eulerian.hpp` | 1D compressible flow on a `Math/Grid.hpp` mesh |
| `Physics/Fluids/Eulerian3D.hpp` | 3D compressible flow via dimensional (Godunov) splitting |
| `Physics/Thermodynamics/Thermodynamics.hpp` | Ideal gas law, adiabatic relation, black-body radiation |
| `Physics/Thermodynamics/StatisticalMechanics.hpp` | Maxwell-Boltzmann speed distribution: density, CDF, moments, characteristic speeds |
| `Physics/Thermodynamics/RadiativeTransfer.hpp` | Radiative exchange between two finite surfaces; the coaxial-disk view factor |
| `Physics/QuantumMechanics/Schrodinger.hpp` | Time-independent Schrödinger equation as a symmetric eigenvalue problem |
| `Physics/QuantumMechanics/Schrodinger3D.hpp` | The same, on the 3D 7-point Laplacian stencil |
| `Physics/QuantumMechanics/WavePacket.hpp` | Time-dependent evolution via split-step Fourier |
| `Physics/QuantumMechanics/WavePacket3D.hpp` | The same, via `Math/FFT.hpp`'s `fft3D`/`ifft3D` |
| `Physics/Thermodynamics/HeatEquation.hpp` | 1D heat diffusion on a `Math/Grid.hpp` mesh |
| `Physics/Thermodynamics/HeatEquation3D.hpp` | 3D heat diffusion on a `Math/Grid3D.hpp` mesh |

## Body stores momentum, not velocity

```cpp
struct Body {
    Mass mass;
    ElectricCharge charge;
    Length3 position;
    Momentum3 momentum;
};
```

`velocity()` is derived, `momentum / mass`, and exact only while v << c.
Momentum rather than velocity is the primitive because it is the one that
stays correct once Mechanics/Kinematics's relativistic momentum, `p = gamma m
v`, applies: velocity alone cannot be turned back into that, while momentum
already is it.

## Units cross the boundary once

Math's integrators need a vector space over a scalar and nothing more (see
`Math/ODE.hpp`'s `OdeState`), which a dimensioned `Quantity` deliberately is
not. So the N-body state a stepper actually runs on, `NBodyState`, is a plain
array of unitless `Vec3`, one per body, and `positionsOf` / `velocitiesOf` /
`applyState` in `Mechanics/Dynamics.hpp` are the one place a span of `Body`
crosses into it and back. This is the same boundary
`tests/integration/units_kinematics.cpp` draws for a single body; every force
law under `Gravity` that plugs into a stepper produces and consumes
`NBodyState` rather than reaching for `Body` in its inner loop.

```cpp
std::vector<Body> bodies = /* ... */;
ysq::NewtonianField field(bodies);

ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                       ysq::velocitiesOf(bodies)};
ysq::PhaseState<ysq::NBodyState> next;
stepper.step(field, 0.0, state, stepSize, next);

ysq::applyState(bodies, next.position, next.velocity);
```

## General-purpose mechanics: forces any scenario can reach for

Gravity is a ladder of approximations to one interaction; the rest of
`Mechanics` is the opposite shape, a growing set of independent, general
force laws with nothing to do with each other beyond all taking a `Body`
(or a pair of them) and returning a `Force3`. None of them assume anything
about what scenario is using them.

`Spring.hpp` is Hooke's law, `F = -k(|r| - restLength) rHat`, with an
optional linear damping term along the spring's own axis,
`-c(v . rHat) rHat`. Two overloads: one anchored to a fixed point (a point
in the inertial frame, not itself simulated, so it contributes no velocity
of its own to the damping term), and one between two bodies, where the
relative velocity of both feeds the damping term and Newton's third law
means the second body's force is exactly the negation of the first's,
`springForce(a, b, ...) == -springForce(b, a, ...)`.

`Drag.hpp` is two regimes of one phenomenon, resistance to motion through a
medium, that do not share a formula. `linearDragForce` (Stokes drag,
`F = -b v_rel`) is correct at low Reynolds number: a small or slow object in
a viscous medium, where drag scales with speed. `quadraticDragForce`
(`F = -(1/2) rho Cd A |v_rel| v_rel`) is correct at high Reynolds number:
most solid objects moving through air or water at everyday speed, where
drag scales with speed *squared*. Both take an optional medium velocity
(a wind, a current), defaulting to a stationary medium, so what actually
enters either formula is always the body's velocity relative to whatever
it is moving through, not its velocity in the inertial frame.

`Collision.hpp` splits into the same two concerns any collision system
does: detection (`detectCollision`, sphere-sphere and sphere-box, each
returning a `Contact` -- where, along which axis, how deep) and response
(`resolveCollision`, the standard point-mass impulse formula,
`j = -(1 + e) v_rel . n / (1/m_a + 1/m_b)`, applied along the contact
normal). `e` (`restitution`) of 0 is perfectly inelastic, both bodies end
up moving at the same velocity along the normal; 1 is perfectly elastic,
kinetic energy along the normal is conserved exactly, which
`physics_mechanics_collision.cpp` checks directly (along with total
momentum, always conserved regardless of `e`) rather than trusting the
formula's derivation alone. Resolution is a no-op whenever the two shapes
are already separating along the normal, so resolving the same contact
twice -- or a contact left over from a previous step that has since
resolved itself -- does not pull them back together. Neither detection nor
resolution moves anything to fix the overlap itself; `Contact::penetrationDepth`
is there for whichever positional-correction scheme a caller's own
integrator prefers.

`Friction.hpp` is Coulomb friction, split into the same two regimes real
friction has and taking the same `contactNormal` convention as
`Contact::normal`. `kineticFrictionForce` opposes whichever direction a
body is actually sliding relative to a surface (stationary by default),
`F = -mu_k N tHat`, projected onto the plane perpendicular to the normal so
it never fights the collision response along the normal itself.
`staticFrictionForce` is the complementary regime, holding a body in place
against a driving force (gravity along an incline, an applied push) as
long as that force's tangential component stays within the Coulomb limit
`mu_s N` -- it returns exactly the negation of that component below the
limit (equilibrium) and the capped value above it (the point past which
the body actually starts to slide, where `kineticFrictionForce` becomes
the correct law instead). Neither function knows where its normal-force
magnitude comes from; a collision impulse, a resting contact, and `m g
cos(theta)` on an incline are all equally valid callers.

`Constraints.hpp` keeps two bodies (or a body and a fixed anchor) at a
fixed distance, or pins a body to a fixed point exactly: a pendulum's rod,
a chain's links, a rope's fixed end. Each is a sequential-impulse solve --
one instantaneous impulse per call that drives the constrained quantity
toward being satisfied at the velocity level, plus a Baumgarte
stabilization term (`baumgarteFactor * error / dt`) that also corrects
whatever positional drift has already accumulated. Calling one of these
once per constraint per timestep is one Gauss-Seidel sweep; a chain of
several links converges toward simultaneous satisfaction over a handful of
sweeps, the same trade real-time constraint solvers make generally, and
the reason this is sequential impulses rather than an exact simultaneous
solve through `Math/LinearSolve.hpp`. `solvePointConstraint` is the vector
generalization of `solveDistanceConstraint` at zero target distance,
constraining all three axes at once rather than only the radial one, since
"zero distance" alone does not pin a direction.
`physics_mechanics_constraints.cpp`'s `PendulumStaysNearItsRodLengthOverManySteps`
integrates a pendulum under gravity for 2000 steps and checks the rod
length stays within 2.5% of target throughout, which is what the
stabilization term is actually for: without it, this kind of small
per-step error compounds into visible drift over exactly this many steps.

## Continuum mechanics: Spring.hpp generalized

`Spring.hpp`'s Hooke's law concentrates all of a connection's compliance
at one point; `Continuum/Elasticity.hpp` is the same law for a
*distributed* elastic medium, one that stretches proportionally to load
along its own length rather than at a single spring. `hookeStress`
(`stress = E strain`) and `axialExtension` (the same law solved for how
far a rod of given length, cross-section and Young's modulus stretches
under a load) are the closed forms; `equivalentSpringConstant`
(`k = E A / L0`, the standard finite-element "bar element" stiffness) is
the bridge back to `Spring.hpp`'s own `SpringConstant` -- the reason that
type lives where it does, since a continuum discretized into segments is
exactly a chain of springs of this stiffness.

`Continuum/ElasticChain1D` is that discretization carried through: a 1D
chain of elastic bar elements, solved for static equilibrium displacement
under a load via `Math/LinearSolve.hpp`'s direct `solve` rather than
`Constraints.hpp`'s sequential impulses. The difference in approach is
deliberate, not incidental -- a static problem has no time-stepping to
iterate over, so one exact simultaneous linear solve is both the natural
and the cheaper choice, where a dynamic constraint chain would have to
reassemble and re-solve its own system every single timestep instead. See
[The elastic chain](#the-elastic-chain) below for the tridiagonal
assembly and the two closed-form checks that validate it independently of
the solver's own internals.

## The gravity ladder

Newtonian gravity, its 1PN correction, and Barnes-Hut summation are three
different things: a force law, a relativistic correction to that force law,
and an approximate way of summing it over many bodies. An application
composes the rungs it needs rather than picking one exclusive "gravity"
entry point.

- **`newtonianForce` / `newtonianAcceleration` / `newtonianAccelerations`**:
  `F = G m1 m2 / r^2`, direct summation, softened. `constants::G` lives here
  rather than in `Units/Constants.hpp` because it is measured, not
  definitional, and parameterizes one specific interaction; see
  [../Units/README.md](../Units/README.md) and [Newtonian gravity](#newtonian-gravity)
  below.
- **`postNewtonianCorrection`**: the 1PN acceleration correction for a test
  particle orbiting a dominant source, added to the Newtonian acceleration
  rather than replacing it. Scope is two bodies, one of them a test particle;
  see the header for why, and [The 1PN correction](#the-1pn-correction) below
  for the derivation.
- **`BarnesHutTree`**: an octree built fresh every call, trading exact
  pairwise summation for O(N log N) at an accuracy set by its opening angle.
  Monopole only.
- **Oblateness**: every function above adds a source's J2 term when its
  `Body::j2` is nonzero, not a second law but the same integral of Newton's
  law evaluated for a non-point mass distribution to its first non-trivial
  order; see [Oblateness (J2)](#oblateness-j2) below. `Mechanics/RigidBody.hpp`
  is its rotational counterpart: the gravity-gradient torque an external mass
  exerts on that same asymmetry, and Euler's rotation equation to integrate
  it, general for any body with a nonzero `principalMomentsOfInertia`; see
  [Gravity-gradient torque and rigid-body rotation](#gravity-gradient-torque-and-rigid-body-rotation).

`NewtonianField` and `BarnesHutTree` both implement
`AccelerationField<NBodyState>` (see `Math/ODE.hpp`), so either drops
directly into `VelocityVerletStepper<NBodyState>` or any of the other
steppers in `Math/Integrators/`. Both also take a softening length, and
`newtonianPotentialEnergy` has to be given the same value the acceleration
was computed with: an energy conservation check is a statement about the
system actually being integrated, the softened one, not the unsoftened one
it approximates. Full derivations, the softening rationale, and Barnes-Hut's
error-versus-theta tradeoff are in [Softening](#softening) and
[Barnes-Hut](#barnes-hut) below.

**`Gravity/Kepler.hpp`** is the closed-form counterpart to all of the
above: the exact, unperturbed two-body solution, rather than a force law
to integrate. `stateVectorFromElements` converts classical orbital
elements straight to a Cartesian position/velocity state; `stateVectorAtTime`
does the same at any later time, propagating through Kepler's equation
(`trueAnomalyFromMeanAnomaly`, solved by Newton-Raphson) at a cost that
does not grow with how far forward the query is, unlike stepping a real
integrator that far. `keplerMeanMotion`/`keplerOrbitalPeriod` give the same
two-body mean motion `n = sqrt(gm / a^3)` every other rung above already
needs, in one place rather than re-derived at each call site. See
[Kepler orbit](#kepler-orbit) below for the full derivation.

## Spacetime: a metric is a concept, not a base class

`SpacetimeMetric` requires one thing: `components(Vector4<T>) const` for any
`Numeric T`, returning `Tensor<T,2,4>`. Being templated rather than virtual
is what lets `christoffelSymbols` differentiate a metric exactly, by seeding
`Dual<double>` through that same method in each coordinate direction and
reading the derivative back off, the same trick `Math/Calculus.hpp` uses for
`gradient` and `jacobian`. `Minkowski`, `Schwarzschild`, `Kerr` and the three
`FLRW` variants each just implement that one method.

```cpp
const ysq::Schwarzschild schwarzschild{gm};
const ysq::Vec4 at{0.0, r, polar, azimuth};        // (c t, r, polar, azimuth)

const ysq::MetricTensor<double> g = schwarzschild.components(at);
const ysq::ChristoffelSymbols<double> gamma = ysq::christoffelSymbols(schwarzschild, at);

const auto system = ysq::geodesicSystem(schwarzschild);
ysq::Rk4Stepper<ysq::PhaseState<ysq::Vec4>> stepper;
stepper.step(system, lambda, state, step, next);
```

**Every coordinate is in metres.** Four-position is `(x0, x1, x2, x3)` with
`x0 = c t`, so a metric needs no unit conversion internally; a
`Schwarzschild` or `Kerr` still takes a `GravitationalParameter` at
construction; it just reduces it to a length once rather than on every
evaluation. This is the same boundary Gravity draws between `Body` and
`NBodyState`, drawn once more here. Which coordinates `x1..x3` are, Cartesian
or spherical, is each metric's own chart; see [Schwarzschild](#schwarzschild),
[Kerr](#kerr) and [FLRW](#flrw) below for each one's.

**Geodesics do not use Gravity's steppers.** The geodesic equation's
`Gamma^mu_ab v^a v^b` term is quadratic in velocity, unlike an
`AccelerationField`'s `a(t, q)`, so `geodesicSystem` builds a plain
`OdeSystem` for `Rk4Stepper` or the adaptive Dormand-Prince stepper instead.
One system serves both timelike and null geodesics: which one a run is
depends only on whether the initial four-velocity is normalized to `-c^2` or
to `0`, checked with `isTimelike` / `isNull`.

Full derivations, each metric's line element, and the geodesic tests'
reasoning are in [Derivations](#derivations) below.

**Every metric above is prescribed, not solved for.** `ADM.hpp`, `Bssn.hpp`
and `PunctureInitialData.hpp` are this module's other case: a spacetime the
engine itself evolves, one timestep at a time, from Einstein's field
equations rather than from a closed form someone solved by hand. See
[3+1 and BSSN](#3+1-and-bssn) below.

## Optics: light propagation, lensing and frequency shift are one geodesic

Because a metric defines how anything moves through it, light propagation,
gravitational lensing, and frequency shift are not three features but one
computation, a null geodesic through a metric, looked at three ways:

```cpp
const ysq::Vec4 k = ysq::nullTangent(schwarzschild, emissionEvent, direction);
const ysq::PhaseState<ysq::Vec4> end =
    ysq::propagate(schwarzschild, {emissionEvent, k}, affineInterval, steps);

const double shift = ysq::frequencyShift(
    schwarzschild, emissionEvent, k, emitterFourVelocity,
    end.position, end.velocity, observerFourVelocity);
```

`Lensing.hpp`'s `deflectionAngle` measures the total azimuth a ray sweeps
between two crossings of a fixed radius and compares it against the
**flat-space** sweep at that same radius, `pi - 2 asin(b/R)`, not against
`pi` outright: the two agree only as the starting radius goes to infinity,
and the gap between them does not shrink under a finer step, since it is a
geometry fact rather than a numerical error. See
[Gravitational lensing](#gravitational-lensing) below for the derivation and
how getting this wrong actually surfaced during testing.

`FrequencyShift.hpp`'s `staticObserverFourVelocity` gives the four-velocity
of an observer at fixed spatial coordinates in any metric with `g_TT < 0`
there, covering the gravitational (Schwarzschild, Kerr) and cosmological
(FLRW, where it reduces to a comoving observer) cases; a relatively moving
Doppler observer instead reuses `Mechanics/Kinematics.hpp`'s `fourVelocity`
directly, since a Minkowski four-velocity and this module's four-velocity
convention are the same object once Kinematics's `x0 = ct` and this
module's agree, which they do by construction.

**Aberration is the direction counterpart to Doppler shift.** The same
relative motion that changes a photon's frequency changes its apparent
direction too; `Aberration.hpp`'s `aberratedDirection` gets there by
reusing `relativisticVelocityAdd` again, this time applied to a velocity
of magnitude `c` rather than to a massive body's velocity, since a
photon's direction transforming between frames is exactly relativistic
velocity addition applied to *its own* velocity. `aberratedCosine` is the
1D textbook special case for a photon and boost that already share an
axis. Both track the photon's own direction of travel, not the apparent
direction to its source (the negation of the travel direction): the
relativistic beaming that concentrates an isotropic field's apparent
sources toward a fast frame's forward direction is this same shift, seen
through that negation.

**Refraction is the same computation once more, against a different
metric.** `RefractiveMedium.hpp` exposes a static, spherically symmetric
graded-index medium (n(r)) as a `SpacetimeMetric`, not because it curves
spacetime, it does not, but because a null geodesic of its particular
"ultra-static" form has exactly the spatial path Fermat's principle gives
for that medium. Everything above, `nullTangent`, `propagate`,
`deflectionAngle`, works on it unchanged; see
[The optical metric](#the-optical-metric) below. `RayleighScattering.hpp`
supplies the one genuinely new law this needed, a gas's scattering
cross-section and the resulting wavelength-dependent transmission along a
path, general for any gas; see [Rayleigh scattering](#rayleigh-scattering).
`Illumination.hpp` is a thin orchestrator over both: how much of an
extended light source is visible from a point, through a scene of opaque
and refracting-and-scattering bodies. It carries no scenario knowledge of
its own; what a caller does with the color and geometric visibility it
returns is entirely up to `Applications/`.

`discOcclusionFraction`, in the same header, is the cheap sibling `illuminate()`
does not replace: the closed-form circle-circle overlap of a light
source's own apparent disc and a single opaque occluder's, as seen from a
point (each disc's own apparent angular radius and the angular separation
between their centers -- real eclipse/transit geometry), not a sampled or
ray-marched approximation. No atmosphere, no color, no per-wavelength
transmission -- exactly what a real shadow needs and nothing an
`illuminate()`-style caller (`LunarEclipse`) needs on top, which is what
makes it cheap enough to call once per particle for a large population
every rendered frame. `KeplerSolarSystem/main.cpp` is the worked example:
a moon's own real shadow from its planet, a ring particle's own real
shadow from its planet, both feeding `Renderer::Material::lightMultiplier`
(a single draw) or `Mesh::setInstanceLightMultipliers` (an instanced one) --
see `src/Renderer/README.md`'s own section on both.

`RadiationPressure.hpp` is the piece that turns light's own presence into
an actual force, rather than only a color and a geometric fact about what
is lit: `irradianceFromPointSource` is the inverse-square law a source's
luminosity falls off by, and `radiationPressureForce` is momentum flux
(irradiance over `c`) times a surface's area and a reflectivity
coefficient `Cr` (1 for a perfect absorber, up to 2 for a perfect
reflector, since reversing a photon's momentum transfers it twice). A
"cannonball" overload composes both for the common astrodynamics case, a
spherical body facing a point source. Nothing here decides visibility or
color -- a caller combines this with `Illumination.hpp`'s own occlusion
and scattering results for a source that is only partially visible or
passing through a medium.

`Diffraction.hpp` is the one place in `Optics` that is genuinely wave
optics rather than ray optics: everything above (propagation, lensing,
frequency shift, refraction, scattering) is a null geodesic looked at some
way, but a diffraction pattern is not a ray's path at all, it is what
happens because light is a wave. `fraunhoferDiffraction` computes the
far-field pattern of an arbitrary 1D aperture as the (squared magnitude of
the) Fourier transform of its own transmission function, via
`Math/FFT.hpp` -- general for whatever aperture a sampled array describes,
not a formula specific to a single slit. `twoSlitIntensity` is the
classical closed form for the specific two-slit case instead, the
single-slit diffraction envelope times the two-beam interference factor,
useful directly without needing to sample an aperture array for the most
common textbook case.

## Electromagnetism is a ladder too

`Field.hpp` is the first rung: `electricField` / `magneticField` superpose
Coulomb's law and the point-charge form of Biot-Savart over a span of
`Body`, quasi-static (each source's present state, not its retarded one).
`constants::vacuumPermeability` lives here rather than in
`Units/Constants.hpp` for the same reason `Gravity`'s `G` does: measured,
not one of the seven constants that define the SI, and specific to this
interaction. `Lorentz.hpp`'s `lorentzForce` is `F = q(E + v x B)`, and takes
the fields `Field.hpp` produces directly.

`Maxwell.hpp`'s `MaxwellField1D` is that next rung: a leapfrog FDTD solver
for the vacuum Maxwell equations, restricted to one spatial dimension (per
`Math/Grid.hpp`'s own scope), so a field that genuinely propagates at `c`
rather than one assumed instantaneous. At the scheme's "magic" time step,
`spacing / c`, it has no numerical dispersion at all and exactly conserves
the field's energy; see [Maxwell FDTD](#maxwell-fdtd) below for the
staggering and the conservation argument.

`Maxwell3D.hpp`'s `MaxwellField3D` is the full 3D Yee grid this rung's own
doc comment used to name as future work: the same leapfrog idea, six field
components instead of two, two curl terms per update instead of one. See
[3D Maxwell: the full Yee grid](#3d-maxwell-the-full-yee-grid) below.

## Acoustics: the same wave equation once more

`Acoustics/Acoustic.hpp`'s `AcousticField1D` is `Maxwell.hpp`'s own
leapfrog FDTD scheme under a different physical interpretation, not a
second derivation: the linearized 1D acoustic wave equations,
`dp/dt = -rho0 c^2 du/dx`, `du/dt = -(1/rho0) dp/dx` (pressure `p`,
particle velocity `u`, medium density `rho0`, sound speed `c`), are
`Maxwell.hpp`'s `dEy/dt = -c^2 dBz/dx`, `dBz/dt = -dEy/dx` with
`p <-> Ey`, `u <-> Bz`, `rho0 c^2 <-> c^2`, `1/rho0 <-> 1`. Every property
that makes `MaxwellField1D` correct -- the Yee/leapfrog staggering, the
CFL limit, the exact zero-dispersion "magic" time step, the
symplectic-style energy conservation -- is a property of that discretized
linear wave operator itself, not of which fields are involved, so it
carries over unchanged; see [Acoustic wave equation](#acoustic-wave-equation)
below. Unlike vacuum light speed, sound speed and medium density are not
universal constants, so `AcousticField1D` takes both as constructor
parameters rather than reaching for one from `Units/Constants.hpp`.

`Acoustic3D.hpp`'s `AcousticField3D` generalizes the same way `Maxwell3D.hpp`
does, on a staggered (marker-and-cell) grid instead of a full Yee cell,
since divergence and gradient (what linear acoustics needs) have no
cross-axis coupling the way curl (what Maxwell needs) does; see
[3D acoustics](#3d-acoustics) below.

## Fluids: Lagrangian and Eulerian, another ladder

`SPH.hpp` (rung 1, Lagrangian, particles carry the fluid, no grid) and
`Eulerian.hpp` (rung 2, the fluid moves through a fixed `Math/Grid.hpp`
mesh) solve the same physics from opposite ends: SPH suits smooth,
low-Mach flows and has no artificial viscosity to handle a shock; the
Eulerian solver, a first-order finite-volume scheme with the Rusanov flux,
is built for exactly that regime. Neither replaces the other, the same
relationship as every other ladder in this module. `Eulerian3D.hpp`
extends the Eulerian rung to 3D by dimensional splitting; see
[3D Eulerian flow: dimensional splitting](#3d-eulerian-flow-dimensional-splitting)
below.

```cpp
std::vector<ysq::SPHParticle> particles = /* ... */;
ysq::computeDensityAndPressure(particles, smoothingLength, k, gamma);
const std::vector<ysq::Vec3> accel = ysq::pressureAccelerations(particles, smoothingLength);

ysq::EulerianFluid1D fluid(cellCount, spacing, gamma);
fluid.setState(cell, density, velocity, pressure);
fluid.step(fluid.stableTimeStep(0.4));
```

Both are validated primarily through exact structural conservation (SPH's
pairwise pressure force is exactly momentum-conserving the same way
Newtonian gravity's is; the Eulerian solver's periodic domain makes mass,
momentum and energy exactly conserved) rather than through matching a
closed-form solution pointwise, since neither an SPH kernel sum on a
discrete lattice nor a first-order shock capture is meant to be exact. See
[SPH](#sph) and
[The Eulerian solver and the Sod shock tube](#the-eulerian-solver-and-the-sod-shock-tube)
below for the details, including a genuine pitfall the shock tube test
walked into and how it was diagnosed: a periodic domain split into two
half-states is two shock tubes, not one.

## Thermodynamics: closed-form, then the heat equation

`Thermodynamics.hpp` (rung 1) is relations with no space or time
dependence: the ideal gas law (mass form, `p = rho R_specific T`), the
adiabatic relation (`p V^gamma = const`), and black-body radiation
(Stefan-Boltzmann, Wien's displacement law). `HeatEquation.hpp` (rung 2) is
`Physics/Fluids`' and `Physics/Electromagnetism`'s third use of
`Math/Grid.hpp`, this time for 1D diffusion, `dT/dt = alpha d^2T/dx^2`.

```cpp
const ysq::Pressure p = ysq::idealGasPressure(density, specificGasConstant, temperature);
const ysq::Power luminosity = ysq::blackBodyLuminosity(starRadius, starTemperature);

ysq::HeatEquation1D heat(cellCount, spacing, diffusivity);
heat.setTemperature(cell, value);
heat.step(heat.stableTimeStep(0.9));
```

`StatisticalMechanics.hpp` sits underneath both: the ideal gas law's
pressure and the heat equation's diffusion are macroscopic, bulk
descriptions of what is really a distribution of individual molecular
speeds. The Maxwell-Boltzmann speed distribution is that underlying
distribution, general for any ideal gas: its probability density, its
cumulative distribution (closed form, via the error function), and its
characteristic speeds (most probable, mean, root-mean-square) via a
general `<v^n>` moment formula through the gamma function, of which those
three closed forms are special cases.

`RadiativeTransfer.hpp` generalizes the Stefan-Boltzmann law the other
direction: `blackBodyLuminosity` is radiation into free space, a view
factor of 1 and a sink at absolute zero; `netRadiativeExchange` is the same
law between two finite surfaces, each at its own temperature, only the
fraction `viewFactor` of one surface's radiation ever reaching the other.
`coaxialDiskViewFactor` supplies that fraction in closed form for one
general geometric configuration (two coaxial, parallel disks) with an
exact analytic view factor, rather than needing a numerical
double-surface integral every other configuration does.

The heat equation rung is validated against its fundamental solution, a
Gaussian's variance growing linearly in time, the same kind of exact
closed-form check `MaxwellField1D`'s propagation speed and
`EulerianFluid1D`'s shock-tube structure are; see
[The heat equation](#the-heat-equation) and
[Closed-form thermodynamics](#closed-form-thermodynamics) below for the
derivation and why the Stefan-Boltzmann constant above is computed from the
SI-defining constants rather than typed as its own measured value.
`HeatEquation3D.hpp`'s `HeatEquation3D` is the same FTCS scheme on
`Math/Grid3D.hpp`'s 7-point Laplacian; see
[3D heat diffusion](#3d-heat-diffusion) below.

## Quantum mechanics: the last theory

Every other theory in this module describes something classical; quantum
mechanics is the one place the engine's state is a wavefunction rather
than a position and momentum. Two pieces, matching the two forms the
Schrödinger equation actually comes in.

`QuantumMechanics/Schrodinger.hpp` is the time-*independent* equation as
what it actually is once discretized: a real symmetric eigenvalue problem,
solved directly by `Math/Eigen.hpp`'s `jacobiEigenSymmetric` -- the exact
tool that module was built for, applied here to a Hamiltonian instead of
an inertia tensor or a covariance matrix. `solveTimeIndependentSchrodinger`
returns every bound state a confining potential has, ascending by energy.

`QuantumMechanics/WavePacket.hpp`'s `TimeDependentWavefunction1D` is the
time-*dependent* equation, evolved by the split-step Fourier method: half
a potential-phase step, a full kinetic-phase step done in momentum space
via `Math/FFT.hpp` (diagonal there, where it would be an expensive
convolution in position space), then the other half potential-phase step.
**This is the reason `Math/FFT.hpp` exists in this engine at all**, ahead
of whichever other consumer asked for it first.

`Schrodinger3D.hpp` and `WavePacket3D.hpp` are the same two pieces in
three spatial dimensions: the eigenvalue problem on the 3D 7-point
Laplacian stencil (still `jacobiEigenSymmetric`, only a bigger matrix),
and split-step evolution via `Math/FFT.hpp`'s `fft3D`/`ifft3D` (the
kinetic operator stays diagonal in momentum space in any dimension, since
`kx^2+ky^2+kz^2` separates additively). See
[3D quantum mechanics](#3d-quantum-mechanics) below, including why the
eigenvalue solver stays practical only for modest grids.

```cpp
const std::vector<double> potential = /* V(x_i), one per grid point */;
const ysq::QuantumEigenstates states =
    ysq::solveTimeIndependentSchrodinger(potential, spacing, mass, hbar);

ysq::TimeDependentWavefunction1D psi(initialWavefunction, potential, spacing, mass, hbar);
psi.step(dt);
const double probability = psi.totalProbability();  // stays 1: unitarity
```

Neither piece assumes a unit system: `hbar` and `mass` are always
explicit parameters, since real SI quantum mechanics (`hbar ~ 1e-34`) and
natural/atomic units (`hbar = 1`) are equally valid choices a caller might
make. See [The discretized Schrödinger equation](#the-discretized-schrödinger-equation)
and [Split-step Fourier evolution](#split-step-fourier-evolution) below for
the discretization, the validation against textbook closed forms, and a
cross-check linking the two files together.

## Derivations

The analytic results the test suite validates against: the material too
long for a header comment.

### Newtonian gravity

`F = G m1 m2 / r^2`, directed along the line joining the two bodies. `G`
lives in `Physics/Gravity/Newtonian.hpp` rather than in `Units/Constants.hpp`:
since the 2019 SI redefinition, `Units/Constants.hpp` holds only the seven
constants that *define* the SI, and `G` is not one of them. It is measured,
CODATA 2018/2022, with a relative standard uncertainty of about 2.2e-5, and
it parameterizes one specific interaction rather than the vocabulary physics
is written in. See [../Units/README.md](../Units/README.md).

Wherever a source's mass parameter `GM` is already known directly, IAU
nominal solar and terrestrial values for instance, prefer that over `G *
mass`: `GM` is what an orbit actually measures, to far more digits than `G`
is known to, and multiplying a mass in kilograms by `G` reintroduces `G`'s
uncertainty into a number that need not carry it. `Body` stores mass in
kilograms because that is the primitive the module's target layout specifies,
so `Gravity` pays that cost internally; an application that already has a
body's `GM` can skip it by constructing a `Body` from `mass = GM / G`, which
still costs the conversion once rather than on every force evaluation, or
`Gravity`'s API can grow a `GM`-based entry point if that turns out to matter
in practice.

### Softening

Direct summation and Barnes-Hut both accept a softening length `epsilon`:

```
a = GM (r_j - r_i) / (|r_j - r_i|^2 + epsilon^2)^(3/2)
```

At separations well above `epsilon` this is indistinguishable from the exact
`1/r^2` law; as two bodies approach each other it stays finite rather than
diverging. An N-body integrator with a fixed step size has no way to resolve
an arbitrarily close encounter, and an unsoftened force there produces an
acceleration spike the stepper cannot integrate accurately, which shows up as
spurious energy injection rather than a physical effect. `epsilon = 0`
recovers the exact law; `nbody_energy`'s random cluster uses a small nonzero
value for exactly this reason.

`newtonianPotentialEnergy` takes the same `epsilon` for the matching reason
in the other direction: a conservation check compares the energy of the
system actually being integrated against itself over time, and that system is
the softened one whenever `epsilon != 0`. Comparing the softened trajectory
against the unsoftened potential would report a discrepancy that is really
just softening's own effect at close range, not an integration error.

### Oblateness (J2)

A point mass is the zeroth term of the multipole expansion of Newton's law
integrated over an extended mass distribution. J2 is the next, non-trivial
term, for a body whose distribution is an oblate spheroid rather than a
sphere: not a second law, the same integral, carried one term further. The
standard result (Vallado, *Fundamentals of Astrodynamics and Applications*),
for a query point at separation `r` from an oblate source's center, `rHat`
the unit vector toward it, `spinAxis` the source's polar axis, and
`s = rHat . spinAxis` the sine of the point's latitude above the source's
equatorial plane:

```
a = -(3/2) J2 mu Req^2 / r^4 * [(1 - 5 s^2) rHat + 2 s spinAxis]
```

`mu = G * sourceMass`, `Req` the source's equatorial radius. Two closed-form
checks validate this directly: in the source's equatorial plane (`s = 0`)
the bracket reduces to `rHat` alone, a purely radial correction that adds to
the point-mass attraction (`Physics/Gravity/Newtonian.hpp`'s equatorial
bulge sits closer to a satellite there than a point mass would);
on the polar axis (`s = 1`) the bracket becomes `-4 rHat + 2 spinAxis`, a
net outward correction, a *weaker* pull than the point-mass term, since the
bulge's extra mass sits away from the poles.

**Newton's third law is not automatic here.** The monopole term is
symmetric in the two bodies (the same `G m1 m2 / r^2` regardless of which
one is called the source), so a per-source accumulation loop conserves
momentum structurally. J2 comes from one specific body's shape, so it is
not symmetric that way: the force an oblate body's bulge exerts on a point
mass has an equal-and-opposite reaction the point mass exerts back on the
oblate body, and a loop that only ever asks "is the source oblate" misses
that reaction whenever only one of a pair is. `newtonianForce`,
`newtonianAccelerations` and `NewtonianField` all check *both* bodies in a
pair, applying the reaction explicitly when only one carries a nonzero
`j2`; `tests/unit/physics_gravity.cpp`'s
`ForceIsEqualAndOppositeWhenOnlyOneBodyIsOblate` and
`MutualAccelerationsConserveMomentumWhenOnlyOneBodyIsOblate` are exactly
the cases a per-source loop would fail.

The matching potential energy, needed for any energy-conservation check on
a J2-perturbed orbit, is `(1/2) G Mquery Msource J2 (Req/r)^2 (3 s^2 - 1) /
r`, verified by differentiating it component-by-component against the
acceleration above rather than derived independently:
`EnergyIsConservedForAJ2PerturbedOrbit` catches the sign this formula is
easy to get backwards (the corrected version conserves energy to floating-point
precision over many orbits; the flipped sign drifted by parts in 10^4 and
did not improve with a finer step, the signature of a formula error rather
than truncation error).

### General spherical harmonics

J2 is one term (degree 2, order 0, "zonal": latitude-dependent but not
longitude-dependent) of a general expansion that has no reason to stop
there. `Gravity/SphericalHarmonics.hpp` is that general expansion, the
standard geodesy geopotential:

```
U(r) = (GM/r) [1 + sum_n (Re/r)^n sum_m P_n^m(sin(phi))
                     (C_nm cos(m lambda) + S_nm sin(m lambda))]
```

`phi`/`lambda` the query point's own latitude/longitude in the source's
body frame, `P_n^m` `Math/SpecialFunctions.hpp`'s associated Legendre
polynomial, and `C_nm`/`S_nm` the body's own shape coefficients ("tesseral"
once `m > 0`, since those terms vary with longitude too, not just
latitude, which a purely zonal `J2` term cannot). `C_2,0 = -J2`, every
other coefficient zero, reproduces `Newtonian.hpp`'s own J2 acceleration
exactly; `physics_gravity_sphericalharmonics.cpp`'s
`J2OnlyMatchesNewtonianHppsClosedFormJ2Term` checks this directly, at
several positions, against the already independently-validated closed
form, which is what actually pins down this general expansion's sign
convention rather than trusting the derivation alone.

`U` is defined positive and increasing toward the source (the geodesy
convention, unlike a potential *energy*), so the acceleration is `+grad(U)`
with no extra sign: `U = GM/r` alone already has `grad(U) = -(GM/r^2) rHat`,
the correct attractive pull, and every harmonic term adds to that same `U`.

**The gradient is numerical, not analytic.** The associated Legendre
functions' own derivatives need a separate recurrence relation with
singularities at the poles to get right; a central-difference gradient
(`Math/Calculus.hpp`'s `numericalGradient`) of the perfectly ordinary,
pole-free Cartesian potential above sidesteps needing it, at the cost of
finite-difference precision (about eight digits) rather than machine
precision. Worth revisiting with an exact analytic gradient only once a
real consumer needs more than that.

### Gravity-gradient torque and rigid-body rotation

The rotational consequence of the identical asymmetry J2 is: an external
mass does not just pull an oblate body's bulge translationally, it exerts a
torque on it, tending to align the bulge with the line to that mass. The
standard result (Hughes, *Spacecraft Attitude Dynamics*), for a body with
principal moments of inertia `I` (diagonal, in its own frame) and an
external mass `M` at distance `r` along unit vector `rHat` from the body's
center:

```
tau = (3 GM / r^3) rHat x (I . rHat)
```

evaluated in the body's own frame (where `I` is diagonal) and rotated back
to the inertial frame by the body's current orientation.
`Mechanics/RigidBody.hpp::gravityGradientTorque` computes exactly this,
reusing the same `principalMomentsOfInertia` the rotational state carries,
not a value derived from `j2` (the two are related for a body in
hydrostatic equilibrium by MacCullagh's formula, but this engine treats
them as two independently-supplied real physical constants, the same way a
scenario supplies both a real mass and a real radius rather than deriving
one from the other).

**Angular momentum, not angular velocity, kept in the inertial frame.**
`dL/dt = tau` is exactly true in an inertial frame for any rigid body, no
extra term. Keeping angular *velocity* as the integrated state instead
would mean working in the body's own rotating frame, where the identical
physics picks up an `omega x (I omega)` term purely from the frame's own
rotation, the textbook form of Euler's equations. Both describe the same
physics; `RigidBody.cpp`'s internal `RotationalState` (orientation quaternion
plus inertial-frame angular momentum) is the simpler one to integrate,
because nothing in it ever needs that cross term. The angular velocity
Euler's equations actually needs, to build `dq/dt = (1/2) q (0, omega_body)`,
comes from rotating the state's angular momentum into the body frame and
dividing through the diagonal `I` there.

RK4 does not preserve the unit-quaternion constraint exactly over a step,
the standard reason quaternion integration renormalizes the orientation
after each accepted step rather than never or at every intermediate stage;
`stepRigidBody` does this once per call.

Two closed-form checks validate the whole chain independently of each
other: a torque-free axisymmetric top's body-frame angular velocity
precesses at the textbook rate `(I_axial - I_equatorial) / I_equatorial *
omega_spin` (Goldstein, *Classical Mechanics*), and the same torque-free
case conserves angular momentum magnitude and rotational kinetic energy
exactly over many steps; `gravityGradientTorque` itself is checked against
the closed form directly, including that it vanishes for a spherically
symmetric body (no asymmetry, no torque) and on an oblate body's own
equatorial limb (a perturber exactly in the equatorial plane pulls straight
along a principal axis, no torque either).

### Diagonalizing an arbitrary inertia tensor

Everything above assumes `principalMomentsOfInertia` is already diagonal in
the body's own frame -- true by construction for a symmetric body whose
frame was chosen to match its symmetry, but not for an arbitrarily shaped
one (an irregular asteroid, a spacecraft with off-axis instruments) whose
inertia tensor, computed directly from its mass distribution, has nonzero
off-diagonal products of inertia in whatever frame it was computed in.

`diagonalizeInertia` closes that gap by eigendecomposing the tensor
(`Math/Eigen.hpp`'s `jacobiEigenSymmetric`): the eigenvalues are the
principal moments, and the eigenvectors, read off as a rotation matrix's
columns, are the principal axes expressed in the tensor's own original
frame. An eigenvector is only defined up to sign, so nothing guarantees the
three Jacobi happens to return form a proper rotation rather than a
reflection (determinant -1); the fix is the standard one, negate the third
axis whenever the determinant comes out negative, which flips a reflection
into a rotation without disturbing the other two (mutually orthogonal)
axes it did not touch. `physics_rigidbody.cpp` checks the defining property
directly, `R diag(moments) Rᵀ` reproduces the original tensor, rather than
only checking the eigensolver's own guarantees, since diagonalization and
correct eigendecomposition are related but distinct claims.

### The elastic chain

`Continuum/ElasticChain1D` assembles the standard finite-element "bar
element" stiffness matrix for a chain of `n` nodes (`n - 1` segments,
node 0 fixed) and solves it directly with `Math/LinearSolve.hpp`.

**Assembly.** Each segment `i` (connecting nodes `i` and `i + 1`, stiffness
`k_i`) contributes the elementary 2x2 stiffness `[[k, -k], [-k, k]]` to the
global system: `K[i][i] += k`, `K[i+1][i+1] += k`, `K[i][i+1] -= k`,
`K[i+1][i] -= k`. Fixing node 0 at zero displacement removes its row and
column from the system entirely, rather than needing a separate
correction term: a segment's coupling term to node 0 would otherwise move
`-k u_0` to the load vector, and that term is exactly zero since `u_0` is
zero by construction. The result is a reduced, `(n - 1) x (n - 1)`
tridiagonal system for nodes `1..n-1`, solved with
`Math/LinearSolve.hpp`'s `solve` (LU with partial pivoting).

**Two closed-form checks, independent of the assembly's own correctness
claims.** A single segment must reduce to plain Hooke's law inverted,
`displacement = load / k`, checked directly. A uniform chain of `n`
identical segments, each of stiffness `E A n / L` (so the whole chain
represents one continuous rod of length `L`, cross-section `A`, modulus
`E`), loaded only at the free end with force `F`: springs in series add
reciprocal stiffness, so `1 / k_effective = n / (E A n / L) = L / (E A)`,
meaning the total end displacement must equal the continuum formula
`F L / (E A)` *exactly*, independent of how finely the rod is
discretized. `physics_continuum_elasticchain.cpp` checks this at four
different segment counts (1, 2, 5, 20) against the same closed form,
which is what actually exercises the discretization-independence claim
rather than merely checking one arbitrary mesh size.

### Barnes-Hut

Barnes, J. and Hut, P., "A hierarchical O(N log N) force-calculation
algorithm", *Nature* 324 (1986), 446-449.

Bodies are stored in an octree: each node is a cube, subdivided into eight
octants only where more than one body would otherwise share a node, so a
sparse distribution produces a shallow tree and a dense cluster a deep one in
the same structure. Every node caches its total mass parameter and center of
mass, computed bottom-up as bodies are inserted.

**The opening-angle criterion.** To find the acceleration at a point, walk
the tree from the root. A node of width `s` at distance `d` from the query
point is accepted, its mass and center of mass used as a single point source,
when

```
s / d < theta
```

Otherwise the walk recurses into the node's children. `theta = 0` forces
every node open, which is direct summation with extra bookkeeping; a large
`theta` accepts coarse approximations more readily and each evaluation costs
less. `theta = 0.5` is the conventional default and is what
`nbody_energy`'s baseline run uses.

A leaf is always evaluated exactly regardless of `theta`, since there is
nothing left to approximate once a node holds one body (or an unresolved
merged aggregate; see the depth guard in the header). A node's own leaf is
excluded from its own acceleration by index, not by distance, since two
distinct bodies can legitimately sit at the same position under softening.

**Monopole only.** Each node's approximation is a single point mass at the
center of mass; it does not carry the distribution's quadrupole moment. A
quadrupole term tightens the error at fixed `theta`, at the cost of a more
expensive node, and is a documented possible refinement rather than
implemented here.

**Error scales with theta.** `nbody_energy` checks this directly: the
root-mean-square deviation between Barnes-Hut's accelerations and direct
summation's, on the same configuration, shrinks as `theta` shrinks, and
`theta = 0` (forcing every node open) reproduces direct summation to
floating-point rounding.

### The 1PN correction

The first post-Newtonian correction to the acceleration of a test particle
orbiting a dominant mass, in the standard parametrized post-Newtonian form
with `gamma = beta = 1` (general relativity):

```
a_1PN = (GM / (c^2 r^2)) * [ (4 GM/r - v^2) n + 4 (v . n) v ]
```

where `r` and `v` are the test particle's position and velocity relative to
the source, `n = r / |r|`, and `GM` uses only the source's mass. Added to the
Newtonian acceleration, `-GM/r^2 n`, this reproduces the relativistic
perihelion advance of a bound orbit, in radians per orbit:

```
delta_phi = 6 pi GM / (c^2 a (1 - e^2))
```

where `a` is the semi-major axis and `e` the eccentricity of the
(now slowly precessing) ellipse. This is the formula
`physics_gravity.cpp`'s `RelativisticNBodySystemPerihelionPrecessionMatchesTheAnalyticRate`
validates the correction against (a Mercury-like case: negligible-mass
planet, dominant star), and the same formula `lensing_deflection.cpp`
validates a full Schwarzschild geodesic against once `Physics/Spacetime`
exists, confirming the two rungs of the gravity ladder agree in the regime
they overlap.

`perihelionPrecessionPerOrbit(gm, semiMajorAxis, eccentricity)` is this
same closed form as a real, reusable function rather than inline test
math: for a caller that wants the real precession rate without running a
real integrator over `postNewtonianCorrection` at all, dividing the
result by `keplerOrbitalPeriod` (see [Kepler orbit](#kepler-orbit) above)
to get radians per second. `KeplerSolarSystem`'s own closed-form Kepler
propagator (`Physics/Gravity/Kepler.hpp`'s
`stateVectorAtTime`) is the worked example: it rotates a body's own
argument of periapsis at that rate directly, which is how a caller with no
n-body integration at all still shows Mercury's real perihelion advance.

**Scope.** The correction itself, `postNewtonianCorrection`, is the
two-body, test-particle form: exact where the source's mass dominates,
which is the regime the precession formula itself assumes. The full N-body
generalization is the Einstein-Infeld-Hoffmann equations, which add cross
terms between every pair of bodies and are not implemented here.

**`RelativisticNBodySystem`** is the practical middle ground a real
hierarchical system (planets around a star, moons around their own
planet) actually needs: direct-summation Newtonian gravity for every
pair, exactly as `NewtonianField` already computes it (J2 included), plus
this correction for whichever bodies the caller names a primary for --
each body against its *own* dominant nearby source, not full EIH cross
terms between every pair. A moon's primary is its own planet, not the
Sun; `Applications/SolarSystem/main.cpp` is the worked example, one
`primaryIndex` entry per real body in the catalog.

This is also the one place in the gravity ladder that is
**velocity-dependent**: the 1PN term needs velocity, not only position, so
`RelativisticNBodySystem` is a full `OdeSystem` over
`PhaseState<NBodyState>` for an explicit stepper
(`Rk4Stepper<PhaseState<NBodyState>>`), not an `AccelerationField` a
symplectic stepper (`VelocityVerletStepper` and the rest of
`Math/Integrators/Symplectic.hpp`) could take. Turning this on is a real,
documented trade: the symplectic bounded-energy-error guarantee for the
actual relativistic physics (visible perihelion precession), not a
limitation to work around.

### Kepler orbit

The unperturbed two-body problem has an exact, closed-form solution, which
is what makes it worth having alongside a numerical integrator rather than
only ever stepping toward it: given the central body's own gravitational
parameter `gm` and the six classical orbital elements (semi-major axis
`a`, eccentricity `e`, inclination, longitude of ascending node, argument
of periapsis, and either true or mean anomaly), the state is known at any
instant with no accumulated integration error at all.

`stateVectorFromElements` builds the position and velocity in the
perifocal frame (the ellipse's own plane, periapsis along +x) from the
standard formulas

```
r = a (1 - e^2) / (1 + e cos(nu))
h = sqrt(gm a (1 - e^2))
x_pf = r cos(nu),  y_pf = r sin(nu)
vx_pf = -(gm/h) sin(nu),  vy_pf = (gm/h) (e + cos(nu))
```

then rotates both into the reference frame by the standard
`Rz(Omega) Rx(i) Rz(omega)` composition (longitude of ascending node,
inclination, argument of periapsis).

Real published data (JPL's included) gives the mean anomaly `M`, the angle
a body *would* have moving at the constant mean motion `n = sqrt(gm/a^3)`
(`keplerMeanMotion`), not the true anomaly the state-vector formula above
needs. The two are related by Kepler's equation, `M = E - e sin(E)`, for
the eccentric anomaly `E`; `trueAnomalyFromMeanAnomaly` solves it by
Newton-Raphson (a handful of iterations reach double precision for any
bound orbit) starting from `E0 = M + e sin(M)`, then converts `E` to true
anomaly by the numerically robust atan2 form of
`tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)`.

`stateVectorAtTime` composes both: it advances the mean anomaly (and,
if `precessionRatePerSecond` is nonzero, the argument of periapsis) to the
requested time, solves Kepler's equation, and converts to a state vector --
at a cost that does not grow with how far forward the query is, unlike
stepping a real integrator that far. `precessionRatePerSecond` is how a
caller with no real integrator at all still shows a real effect like
relativistic perihelion advance: convert
[`postNewtonianCorrection`](#the-1pn-correction)'s
`perihelionPrecessionPerOrbit` (radians per orbit) to radians per second by
dividing by `keplerOrbitalPeriod`.

### Individual timesteps (the Hermite scheme)

A shared global step forces every body through whichever one moves
fastest: correct for that body, wasteful for every slower one, and there
is no single step size that is both cheap and accurate for bodies whose
dynamical timescales differ by orders of magnitude (an inner ring moon
under a day, an outer irregular moon decades). `Physics/Mechanics/Hermite.hpp`
gives each body its own step instead.

**The predictor** extrapolates a body's position and velocity forward by
`dt` using its own last known acceleration `a` and jerk (`ȧ`, the rate
`a` itself is changing) via a third-order Taylor expansion:

    x_pred = x + v dt + (1/2) a dt^2 + (1/6) ȧ dt^3
    v_pred = v + a dt + (1/2) ȧ dt^2

This is what lets one body's force evaluation use a physically reasonable
estimate of another body's position, even when that other body has not
been updated in a while.

**The corrector** fits the acceleration's own 2nd and 3rd derivatives
("snap" and "crackle") from the (acceleration, jerk) pair known at both
the start and the predicted end of the step (Makino & Aarseth 1992):

    ä = [-6 (a - a_new) - dt (4 ȧ + 2 ȧ_new)] / dt^2
    ⃛a = [12 (a - a_new) + 6 dt (ȧ + ȧ_new)] / dt^3
    x = x_pred + (1/24) ä dt^4 + (1/120) ⃛a dt^5
    v = v_pred + (1/6) ä dt^3 + (1/24) ⃛a dt^4

Verified by measuring the observed convergence order on a circular
Kepler orbit (`tests/unit/physics_mechanics_hermite.cpp`), the same
methodology `tests/unit/math_integrators.cpp` uses for RK4 and Verlet:
4th order, as the derivation claims.

**Each body's own step** comes from the Aarseth (1985) criterion --
`dt = sqrt(eta * |a| / |ȧ|)`, shrinking automatically where a body's
acceleration is large and rapidly changing (a close encounter, any sharp
perturbation) and growing where it is calm and slowly varying, with
nothing naming an orbit or a parent anywhere in the formula -- rounded
down to the nearest power-of-two fraction of a caller-chosen base
interval, so bodies with similar timescales land on shared update times
rather than each drifting to a time nothing else ever lines up with.

**`IndividualTimestepScheduler`** owns every body's own (last-update
time, step, position, velocity, acceleration, jerk) and, each cycle,
advances whichever body's own next update is soonest: every other body
is predicted to that instant, the caller's jerk field (see below) is
asked for the mover's own new (acceleration, jerk), the corrector
refines its position and velocity, and its next step is re-chosen. This
lives in Mechanics, not Gravity: nothing here knows what jerk *is*
physically, the same way `Dynamics.hpp`'s `NBodyState` does not know
what force law produced an acceleration. `Physics/Gravity/Newtonian.hpp`'s
`NewtonianJerkField` and `Physics/Gravity/PostNewtonian.hpp`'s
`RelativisticNBodyJerkSystem` supply the concrete gravity (and 1PN) jerk.

**Jerk formulas.** The pairwise Newtonian jerk (`NewtonianJerkField`) is
the direct time-derivative of the same softened term
`NewtonianField` already computes:

    r = position_j - position_i,  v = velocity_j - velocity_i
    R = sqrt(|r|^2 + softening^2)
    jerk_i (from j) = G m_j [ v / R^3 - 3 (r . v) r / R^5 ]

The 1PN correction's own jerk (`relativisticJerkTerm`) is the
time-derivative of `relativisticAccelerationTerm`, differentiated
term-by-term through `r`, `v`, the unit vector `n = r / |r|`, and the
radial speed `s = v . n` -- a real, multi-term closed-form expression,
not a formula that can be looked up the way the base 1PN term itself
was. Verified against a finite-difference of the already-tested
acceleration term rather than trusted on the derivation alone
(`tests/unit/physics_gravity.cpp`), exactly the kind of closed-form
derivative that is easy to get subtly wrong. Its own relative
acceleration term uses the two-body Newtonian relative acceleration,
`-gm r / |r|^3` -- the same test-particle-around-a-dominant-source scope
`relativisticAccelerationTerm` itself already assumes, so this pulls in
no more of the full N-body picture than that formula already does.

**J2 jerk is not implemented.** Differentiating the quadrupole term
through a rotating spin axis is a separate derivation, future work
rather than a limitation of this being wrong for a body whose `j2`
actually is zero -- every body the Solar System catalog names. A future
consumer with an oblate body under this scheduler needs that term added
first.

**Not symplectic.** This is 4th-order accurate, the same category RK4
is in, not `VelocityVerletStepper`'s: there is no guarantee against slow
energy drift over an arbitrarily long run, only a small per-step error
and each body resolved at its own appropriate scale rather than everyone
paying for the fastest one.

**What this actually buys, measured against the real catalog.** For
`Applications/SolarSystem`'s real ~175-body dataset, advancing 1
simulated month takes about 30 seconds of real compute with individual
timesteps, against roughly 40 seconds the old single-global-step
approach would need to do the same amount of work (both uncapped, i.e.
the total cost of the work itself, not throttled by a per-frame cap). A
real improvement, but a modest one (roughly 1.3x): about a fifth of this
catalog's ~175 bodies have their own period under a day, not just the
one fastest moon, so most of the benefit an individual-timestep scheme
gets from letting *slow* bodies skip needless updates is diluted by how
many bodies in a real solar system are not slow.

### Spacetime conventions

Fixed once, here, and used without restatement by every metric in
`Physics/Spacetime`.

**Signature (-,+,+,+).** A timelike interval has negative squared length; a
spacelike one, positive. `metricProduct`, `isTimelike`, `isSpacelike` and
`isNull` in `Metric.hpp` all read against this.

**Four-position is `(x0, x1, x2, x3)` with `x0 = c t`.** Every component is
then in metres, and no metric needs a unit conversion of its own: a
`Schwarzschild` or `Kerr` object still takes a `GravitationalParameter` at
construction and reduces it to a length (`r_s = 2GM/c^2`) once, but the
`components()` that actually runs per evaluation is unitless throughout, the
same boundary Gravity draws between `Body` and the raw `NBodyState` an
integrator runs on. What `x1, x2, x3` mean is each metric's own choice of
chart, Cartesian for `Minkowski`, spherical `(r, polar, azimuth)` for
`Schwarzschild`, `Kerr` and `FLRW`, documented on the metric itself. The
geodesic solver only ever calls `components()`, so it never needs to know
which.

### Christoffel symbols

```
Gamma^lambda_mu_nu = (1/2) g^lambda_sigma ( d_mu g_sigma_nu
                                            + d_nu g_sigma_mu
                                            - d_sigma g_mu_nu )
```

computed exactly rather than by finite difference: `components()` is
evaluable at any `Numeric` scalar, so seeding `Dual<double>` through it in
each of the four coordinate directions gives `d g_mu_nu / d x^alpha` to
machine precision, one seeded evaluation per direction, the same trick
`Math/Calculus.hpp`'s `gradient` and `jacobian` use for `Vector`-shaped
results. `Tensor` is not one of those, which is why `christoffelSymbols` in
`Metric.hpp` does its own seeding rather than reusing them. The metric
inverse `g^lambda_sigma` costs one `Matrix4::inverse`, through the
`Tensor`/`Matrix4` conversions already in `Math/Tensor.hpp`.

`spacetime_metric.cpp` checks the result is symmetric in its lower two
indices, `Gamma^lambda_mu_nu = Gamma^lambda_nu_mu`, which the formula
guarantees structurally and is worth pinning as a regression check anyway;
and that Minkowski's Christoffel symbols vanish exactly everywhere, since a
constant tensor has nothing to differentiate.

`spacetime_schwarzschild.cpp` checks one component, `Gamma^r_TT`, against a
form derived directly from the metric rather than transcribed from a
reference table: since `g` is diagonal and `g_TT` does not depend on `T`,

```
Gamma^r_TT = -(1/2) g^rr d_r g_TT = (1/2) r_s (1 - r_s/r) / r^2
```

which is also the check that this reduces to Newtonian gravity: a particle
released from rest has four-velocity `u = (u^T, 0, 0, 0)` with `u^T
approx c` far from the source, so its initial coordinate acceleration is
`d^2 r / dtau^2 = -Gamma^r_TT (u^T)^2 approx -GM/r^2` as `r_s/r -> 0`.

### The geodesic equation

```
d^2 x^mu / dlambda^2 + Gamma^mu_ab (dx^a/dlambda)(dx^b/dlambda) = 0
```

as a first-order system in `(position, four-velocity)`,
`Physics/Spacetime/Geodesic.hpp`'s `geodesicSystem`. **This cannot use the
symplectic steppers Gravity's rungs do.** Those take an `AccelerationField`,
`a(t, q)`: an acceleration depending on position alone. The
`Gamma^mu_ab v^a v^b` term above is quadratic in velocity, so a geodesic
rides `Rk4Stepper` or the adaptive Dormand-Prince stepper instead, both
already in `Math/Integrators/`.

One system serves both timelike and null geodesics. Which one a given run is
depends only on the initial four-velocity's normalization,
`metricProduct(metric, at, u, u)` equal to `-c^2` for a massive particle's
proper time, or `0` for light: nothing in the equation itself distinguishes
them. The affine parameter is proper time for a timelike geodesic and has no
invariant meaning for a null one, the ordinary situation in relativity.

`spacetime_geodesic.cpp` checks two cases. In Minkowski, where `Gamma` is
identically zero, a geodesic is an exactly straight line at constant
velocity, and RK4 integrates that exactly regardless of step size, since
every term past the first in its truncation error involves a derivative of
an acceleration that is identically zero. In Schwarzschild, a null geodesic
launched tangentially at the photon sphere, `r = 1.5 r_s`, holds that radius:
the unstable circular photon orbit, and the test that actually exercises
`christoffelSymbols` and the stepper together on curved spacetime.

### 3+1 and BSSN

Every metric above is a closed-form solution someone already solved
Einstein's equations to get. `ADM.hpp`, `Bssn.hpp` and
`PunctureInitialData.hpp` are what lets this engine solve them itself:
given a spatial slice's own geometry and matter content, evolve what the
next slice looks like, rather than only tracing a test particle through a
geometry handed to it. This is genuine engine infrastructure, general for
any consumer who needs a spacetime that responds to what's in it -- not
built for, or limited to, any one scenario.

**The 3+1 (ADM) split.** Foliate spacetime into a stack of spatial slices,
each with its own metric `gamma_ij`, and describe how one slice is embedded
in spacetime by its extrinsic curvature `K_ij` -- how the normal vector to
the slice changes as you move within it. The lapse `alpha` and shift
`beta^i` say how to step from one slice to the next: `alpha` is how much
proper time passes for an observer moving normal to the slice, `beta^i` is
how the spatial coordinates themselves shift. Einstein's equations become
an initial-value problem in these variables: two evolution equations
(for `gamma_ij` and `K_ij`) and two constraint equations (Hamiltonian and
momentum) that must hold on every slice, evolution preserving them if they
held on the last one. See Gourgoulhon, "3+1 formalism and bases of
numerical relativity" (arXiv:gr-qc/0703035) for the full derivation from
the 4D field equations.

**Why BSSN, not raw ADM.** The plain ADM evolution equations are only
weakly hyperbolic: small, high-frequency errors are not damped, and a
numerical evolution built on them is unstable beyond the shortest runs.
Baumgarte & Shapiro (Phys. Rev. D 59, 024007, 1998) and, independently,
Shibata & Nakamura (Phys. Rev. D 52, 5428, 1995) found a reformulation that
fixes this:

- **Conformal decomposition.** `gamma_ij = e^{4 phi} gammaTilde_ij`, with
  `phi` chosen so `det(gammaTilde_ij) = 1`.
- **Trace/trace-free split.** `K_ij = Atilde_ij / e^{-4 phi} + (1/3) gamma_ij K`
  (equivalently, `Atilde_ij = e^{-4 phi} (K_ij - (1/3) gamma_ij K)`), `K`
  evolved as its own variable.
- **Promoting the contracted Christoffel symbols to an independent
  variable**, `GammaTilde^i = gammaTilde^jk GammaTilde^i_jk`, evolved by its
  own equation rather than recomputed from second derivatives of
  `gammaTilde_ij` every step. This is the specific change that makes the
  system strongly hyperbolic -- substituting the momentum constraint into
  this variable's own evolution equation is what removes the weakly
  hyperbolic terms raw ADM has.

`Bssn.hpp`'s evolved state is exactly `(phi, gammaTilde_ij, K, Atilde_ij,
GammaTilde^i, alpha, beta^i, B^i)`, the last an auxiliary variable the
shift condition below needs. The evolution equations themselves (cited
individually in `Bssn.cpp`'s own comments, term by term, since this is the
single most index-heavy derivation in the codebase and worth being able to
check against the source directly):

```
dt phi        = beta^i d_i phi - (1/6) alpha K
dt gammaTilde_ij = -2 alpha Atilde_ij + Lie_beta(gammaTilde_ij)
dt K          = -gamma^ij D_i D_j alpha + alpha (Atilde_ij Atilde^ij + K^2/3) + beta^i d_i K
dt Atilde_ij  = e^{-4 phi} [-D_i D_j alpha + alpha R_ij]^TF
              + alpha (K Atilde_ij - 2 Atilde_ik Atilde^k_j) + Lie_beta(Atilde_ij)
dt GammaTilde^i = gammaTilde^jk d_j d_k beta^i + (1/3) gammaTilde^ij d_j d_k beta^k
                + beta^j d_j GammaTilde^i - GammaTilde^j d_j beta^i
                + (2/3) GammaTilde^i d_j beta^j - 2 Atilde^ij d_j alpha
                + 2 alpha (GammaTilde^i_jk Atilde^jk - (2/3) gammaTilde^ij d_j K
                           - 6 Atilde^ij d_j phi)
```

`R_ij` is the physical Ricci tensor, split as `R_ij = RtildeIJ + R^phi_ij`:
`RtildeIJ` (the conformal metric's own Ricci tensor, built from
`GammaTilde^i` rather than directly from second derivatives of
`gammaTilde_ij`, per the strong-hyperbolicity point above) and `R^phi_ij`
(the standard conformal-transformation correction from `phi`'s own
gradient and Hessian). `conformalRicciAt` in `Bssn.cpp` implements
`RtildeIJ` as literally as possible -- direct nested summation over every
term, no hand-simplified closed form -- specifically so it can be checked
term by term against Baumgarte & Shapiro's own equations rather than
trusted on the strength of this description alone.

**Moving-puncture gauge.** 1+log slicing for the lapse (Bona-Masso family,
`dt alpha = beta^i d_i alpha - 2 alpha K`) and a Gamma-driver for the shift
(`dt beta^i = (3/4) B^i`, `dt B^i = dt GammaTilde^i - eta B^i`): the
combination that made dynamical black-hole evolution numerically tractable
(Campanelli, Lousto, Marronetti & Zlochower 2006; Baker, Centrella, Choi,
Koppitz & van Meter 2006; van Meter et al. 2006). This omits the advective
(`beta^j d_j B^i` / `beta^j d_j GammaTilde^i`) terms some implementations
add to the Gamma-driver: those mainly help track a puncture moving quickly
across the grid (an orbiting or merging binary), not the stability of
evolution itself, so this module does not need them yet -- a future
consumer whose scenario does can add them.

**Time integration reuses `Rk4Stepper`.** BSSN's evolution is a
Method-of-Lines system, `dt(state) = RHS(state, spatial derivatives)` --
exactly `Math/ODE.hpp`'s general `OdeSystem` shape. `BssnState` implements
`OdeState` by delegating its vector-space operations field by field to
`Grid3D`'s own (added there for exactly this reason), so it hands straight
to `Rk4Stepper<BssnState>` with no new integrator needed, unlike
`Physics/Gravity/PostNewtonian.hpp`'s velocity-dependent correction, which
genuinely cannot use any of the symplectic steppers.

**Puncture initial data.** A slice's `gamma_ij` and `K_ij` cannot be chosen
freely -- they must satisfy the Hamiltonian and momentum constraints before
evolution even starts. Brandt & Brügmann's puncture construction
(arXiv:gr-qc/9711015), with Bowen-York extrinsic curvature (Bowen & York,
Phys. Rev. D 21, 2047, 1980):

```
AbarIJ = (3 / 2 r^2) [P^i n^j + P^j n^i - (delta^ij - n^i n^j) P.n]
       + (3 / r^3) [(S x n)^i n^j + (S x n)^j n^i]
```

satisfies the momentum constraint identically, for any puncture
mass/momentum/spin, in a conformally flat background -- an exact, closed-form
property of this particular ansatz, not a numerical coincidence. That
leaves only the Hamiltonian constraint, one elliptic equation for the
conformal factor's correction `u` (`psi = 1 + sum m_i/(2 r_i) + u`, the
singular Brill-Lindquist part factored out so `u` itself is smooth
everywhere, including at every puncture):

```
flatLaplacian(u) = -(1/8) AbarIJ AbarIJ (1 + sum m_i/(2 r_i) + u)^{-7}
```

solved by `Math/Multigrid.hpp`'s general FAS (Full Approximation Scheme)
nonlinear multigrid solver, not a plain relaxation loop. For a single,
momentarily-static puncture, `AbarIJ` is identically zero and `u = 0` is
the exact solution regardless of which solver finds it; a binary's is not
(see "Binary initial data" below), which is the actual reason this moved
off plain Gauss-Seidel -- plain relaxation converges far too slowly once
there is a genuinely nonzero, spatially varying source term to resolve.

**Binary initial data.** Two punctures, momentarily on a quasi-circular
orbit, need nonzero momentum: `PunctureSpec::momentum` opposite and equal
in magnitude on each, estimated by the Newtonian circular-orbit relation
`P = mu sqrt(M / D)` (`mu` the reduced mass, `M` the total mass, `D` the
coordinate separation) --
`newtonianCircularMomentum` in `PunctureInitialData.hpp`. This is the
standard, simple starting point real numerical-relativity papers themselves
use before iterative refinement (Cook's effective-potential method,
post-Newtonian momenta, eccentricity reduction); genuinely accurate
quasi-circular data construction is its own research topic and not
approximated further here. With nonzero momentum, `AbarIJ` is genuinely
nonzero and spatially varying, and its square enters the Hamiltonian
constraint's source term steeply near each puncture -- exactly the case
multigrid exists for, and exactly what a single static puncture's `u = 0`
answer never exercised.

**Validation.** `tests/unit/bssn.cpp` builds a static Schwarzschild puncture
directly (not via the solver) and checks that `hamiltonianConstraint` is
small in the bulk (away from the handful of cells immediately around the
puncture, where any discretization of `1 + m/2r` has large local truncation
error regardless of correctness) and shrinks under grid refinement -- the
actual signature of a correct implementation, not merely a small number at
one resolution. `tests/integration/single_puncture_stability.cpp` is Stage
1's full validated milestone: evolve, confirm the run stays finite and the
constraint stays bounded (not growing without limit) through the initial
"wormhole to trumpet" transient the moving-puncture gauge produces (Hannam,
Husa, Pollney, Brügmann & O'Murchadha, arXiv:0804.0628), and confirm the
ADM mass recovered from `psi = e^{phi}`'s own asymptotic falloff is
consistent with the puncture's input mass.

`tests/integration/binary_puncture_stability.cpp` is Stage 2's own
milestone: two equal-mass, non-spinning punctures at a wide (`D = 6M`)
separation, with the quasi-circular momentum above. Multigrid converges in
18 V-cycles on this genuinely nontrivial source term (down from the
default cap of 50, and the test itself asserts under 30), with a bulk
Hamiltonian-constraint violation around 4e-3 on the raw initial data.
Both the Hamiltonian and momentum constraints, excluding both the
near-puncture cells (the same well-understood, expected large local error
as the static case) and a domain-edge margin (this test's simplified
outflow boundary condition does not itself satisfy either constraint, by
construction, near the edge -- see that test's own doc comment), stay
small and bounded through a short evolution.

**Scope, honestly.** At the resolution and step count this suite runs on
every push (16 cells across, 16 RK4 steps), the run covers a negligible
fraction of one Newtonian orbital period and is not itself evidence the
orbit tracks the Newtonian estimate `newtonianCircularMomentum` provides --
only that initial data construction and short-term evolution are stable.
A longer, higher-resolution run to actually verify orbital dynamics is a
real gap, not something already checked elsewhere in this repository; it
needs its own follow-up before this stage's initial data can be trusted
for anything beyond the stability this suite verifies.

### Schwarzschild

A non-rotating mass, the unique spherically symmetric vacuum solution, in
the chart `(T, r, polar, azimuth)`:

```
ds^2 = -(1 - r_s/r) dT^2 + dr^2 / (1 - r_s/r) + r^2 dpolar^2
       + r^2 sin^2(polar) dazimuth^2
```

`r_s = 2GM/c^2`. Components diverge at `r = r_s`, the coordinate singularity
of this chart rather than a physical one, and at `r = 0`, the genuine
curvature singularity; neither is guarded against, since a geodesic that
reaches either has left the regime this chart can say anything about.

### Kerr

A rotating mass, the Boyer-Lindquist form of the unique stationary,
axisymmetric vacuum solution, same chart as Schwarzschild, which this
reduces to exactly at `spin = 0`. Writing `sigma = r^2 + a^2 cos^2(polar)`
and `delta = r^2 - r_s r + a^2`, with `a = J/(Mc)` the spin length:

```
ds^2 = -(1 - r_s r / sigma) dT^2
       - (2 r_s r a sin^2(polar) / sigma) dT dazimuth
       + (sigma / delta) dr^2 + sigma dpolar^2
       + (r^2 + a^2 + r_s r a^2 sin^2(polar) / sigma) sin^2(polar) dazimuth^2
```

The `g_T,azimuth` cross term is frame dragging: an observer at fixed `r` and
`polar` cannot hold `azimuth` fixed and remain timelike close enough to the
horizon, since the term forces `dT` and `dazimuth` to mix.

### FLRW

An expanding, homogeneous, isotropic universe, comoving coordinates
`(T, r, polar, azimuth)`:

```
ds^2 = -dT^2 + a(T)^2 [ dr^2 / (1 - k r^2) + r^2 dpolar^2
                        + r^2 sin^2(polar) dazimuth^2 ]
```

`a` is dimensionless, usually normalized to 1 at the present; `k` is `+1`,
`0` or `-1` for closed, flat and open. What determines `a(T)` in general is
the Friedmann equations, sourced by however much matter, radiation and dark
energy the universe holds, a coupled system that is not solved here. What is
implemented are the three standard single-component analytic solutions, for
the regimes where one component's stress-energy dominates the rest:

| Class | `a(T)` | Equation of state |
| --- | --- | --- |
| `MatterDominatedFLRW` | `(T/T0)^(2/3)` | `w = 0` (dust) |
| `RadiationDominatedFLRW` | `(T/T0)^(1/2)` | `w = 1/3` |
| `LambdaDominatedFLRW` | `exp(H (T - T0))` | cosmological constant alone (de Sitter) |

Each is its own class rather than one `FLRW` taking a callable scale factor,
because a callable would have to differentiate correctly through `Dual` to
give correct Christoffel symbols, and `Dual` supports `log`, `exp` and `sqrt`
but not a general `pow`. `T^(2/3)` is written as `exp((2/3) log T)` for
exactly that reason: both `log` and `exp` differentiate exactly through
`Dual`, so the composition does too, once, here, rather than asking every
caller who might supply a scale factor to know the same substitution.

### Light propagation

Light is a null geodesic (see [Optics](#optics-light-propagation-lensing-and-frequency-shift-are-one-geodesic)
above); `Optics/Propagation.hpp` adds nothing to the geodesic solver beyond
building the right initial tangent and running it. `nullTangent(metric, at,
direction)` fixes the tangent's magnitude by solving `g_mu_nu u^mu u^nu = 0`,
quadratic in `u^0` given the spatial components `direction`:

```
A (u^0)^2 + B u^0 + C = 0
A = g_00,  B = 2 sum_i g_0i direction_i,  C = sum_ij g_ij direction_i direction_j
```

taking whichever root is future-directed. `propagate` then runs the same
`geodesicSystem` a timelike worldline would, over an affine interval, with
an observer callback on the same convention as `Math::integrate`.

### Gravitational lensing

**The finite-radius correction, and why it exists.** `deflectionAngle`
measures the total azimuth swept by a ray that starts and ends at some
radius `startRadius`, and the natural first guess is that this sweep is `pi`
in flat space, with the excess being the deflection. That guess is wrong
for any finite `startRadius`, and the error is not small: a straight line at
perpendicular distance `b` from the origin crosses a circle of radius `R`
(`b < R`) at azimuth `pi - asin(b/R)` inbound and `asin(b/R)` outbound, for a
flat-space sweep of

```
pi - 2 asin(b / R)
```

not `pi`. The missing term is order `2b/R`, and critically, **it does not
shrink under a finer integration step**, since it is a fact about the
geometry of a fixed-radius sphere, not a property of how the geodesic was
integrated. Comparing against bare `pi` produces a deflection estimate that
converges under refinement to a wrong, `startRadius`-dependent constant
rather than to zero, which is what first exposed this: taking
`impactParameter` to within a few percent of `startRadius`, deep in the
regime where the true GR deflection is negligible, the measured sweep
converged cleanly to `pi - 2 asin(b/R)`, not to `pi`, and did not move at
all under a ten-fold finer step. `deflectionAngle` subtracts the correct
flat-space sweep, not `pi`, for exactly this reason.

**Building a ray of known impact parameter.**
`schwarzschildRayFromImpactParameter` uses the two quantities a static,
spherically symmetric spacetime conserves along a geodesic: energy `E` and
angular momentum `L`, from the Killing vectors of `dT` and `dazimuth`.
Normalizing `E = 1` makes `impactParameter` exactly `L`:

```
uT = 1 / (1 - r_s / startRadius)
uPhi = impactParameter / startRadius^2
ur from the null condition, negative root (inward)
```

exact for any `startRadius`, not an approximation that only holds far from
the source.

`lensing_deflection.cpp` validates the result against
`4GM/(c^2 b) = 2 r_s / b`, the standard weak-field deflection formula, deep
in the regime `b >> r_s` where that leading-order formula is accurate.

### The optical metric

Fermat's principle says a ray through a medium of refractive index `n(x)`
extremizes the optical path length `integral n dl`. That is a spatial
geodesic problem for the Riemannian metric `h_ij = n(x)^2 delta_ij`, a
different mathematical object from this module's 4D pseudo-Riemannian
`(-,+,+,+)` metrics, at first. The standard "optical metric" construction
(Gordon 1923's moving-medium metric, specialized to a medium at rest)
embeds it as one of those after all:

```
ds^2 = -dT^2 + n(r)^2 (dr^2 + r^2 dpolar^2 + r^2 sin^2(polar) dazimuth^2)
```

`g_TT = -1` exactly, so `T` never appears in any component: there is a
Killing vector along `T`, and the null condition `(dT)^2 = n(r)^2 dl^2`
makes `T`, along any null geodesic, equal to the optical path traveled so
far. For a metric of the "ultra-static" form `-dT^2 + h_ij(x) dx^i dx^j`
(`h` depending on space alone), the spatial projections of its null
geodesics are exactly the spatial geodesics of `h_ij`, a standard result
also used the other direction, describing Schwarzschild's own light
bending as an effective refractive index. `RefractiveMedium.hpp` uses it
to go from a real, ordinary medium to something `Spacetime/Geodesic.hpp`'s
unmodified solver already knows how to propagate through: `nullTangent`,
`propagate`, `deflectionAngle` and `refractiveMediumRayFromImpactParameter`
(the same `E = 1`, `L = impactParameter` shooting construction
`schwarzschildRayFromImpactParameter` uses, adapted to this metric's
constant `g_TT`) all work on it unchanged, because none of them assume
which metric they are given.

`n(r) = 1 + surfaceRefractivity * exp(-(r - radius) / scaleHeight)`, the
same exponential shape Gladstone-Dale (`n - 1` proportional to density)
gives for any isothermal barometric atmosphere
(`Thermodynamics::isothermalAtmosphereDensity`); `RefractiveMedium.hpp`
does not depend on `Thermodynamics` itself, `surfaceRefractivity` and
`scaleHeight` are two plain numbers a scenario supplies however it derived
them.

`optics_refractivemedium.cpp` validates the whole construction against a
real, measured number, not merely an internal consistency check: a ray
grazing about half a percent above an Earth-radius, Earth-atmosphere
medium's surface bends by about 38 arcminutes, close to the real
horizontal (grazing) atmospheric refraction of 34-35 arcminutes, the
remaining gap being this model's isothermal simplification of the real,
non-isothermal atmosphere, not a construction error. A practical numerical
note the same test found: unlike Schwarzschild's `1/r` tail, this medium is
exactly flat (`n = 1` to well beyond double precision) a few dozen scale
heights up, so `startRadius` only needs to clear the atmosphere, not be
"far away" in Schwarzschild's sense, which keeps the affine range
`deflectionAngle` has to search across small enough to run in well under a
second rather than minutes.

### Rayleigh scattering

The standard result for scattering by particles much smaller than the
wavelength (Bohren & Huffman, *Absorption and Scattering of Light by Small
Particles*), a gas's per-molecule cross-section:

```
sigma(lambda) = (24 pi^3 / (lambda^4 N^2)) * ((n^2 - 1) / (n^2 + 2))^2
```

`n` and `N` (number density) at whatever single reference condition they
were measured at. The `1/lambda^4` dependence is the entire reason a
grazing atmospheric path reddens whatever light survives it: blue
scatters far more than red over the same path, exactly the mechanism
behind both a blue sky and a red sunset (or a grazing path into a lunar
eclipse's umbra). `optics_rayleighscattering.cpp` checks this ratio
directly, `(650/450)^4`, exactly, and the cross-section's order of
magnitude against real air (about 5e-31 m^2 at 532 nm).

Optical depth along a path is this cross-section times the integral of
number density along it; `RayleighScattering.hpp` supplies
`exponentialNumberDensity`, the same barometric shape
`RefractiveMedium.hpp`'s refractivity and `Thermodynamics`'s density share,
and leaves the actual path integral to whoever is walking one already
(`Illumination.hpp`, alongside the bending calculation, rather than a
second pass over the same trajectory). `transmission(tau) = exp(-tau)`,
Beer-Lambert.

### Frequency shift

Doppler, gravitational and cosmological shift are the same computation, the
ratio of a photon's four-momentum contracted against an observer's
four-velocity at emission and at observation:

```
nu_observed / nu_emitted = (k . u)_observed / (k . u)_emitted
```

(both dot products negated, since a future-null `k` and future-timelike `u`
give a negative product in this module's `-,+,+,+` signature). Nothing
distinguishes "kinds" of shift here; which name applies is a property of the
scenario, not of the formula: static observers near a mass for
gravitational shift, comoving observers in an expanding FLRW background for
cosmological, relatively moving inertial observers in flat spacetime for
Doppler.

`staticObserverFourVelocity(metric, at)` is `u = (u^T, 0, 0, 0)`, normalized
by `g_TT u_T^2 = -c^2`: the four-velocity of an observer at fixed spatial
coordinates, defined wherever `g_TT < 0`. In FLRW, where `g_TT = -1`
identically, this reduces to `u = (c, 0, 0, 0)`, a comoving observer.

Validated three ways in `optics_frequencyshift.cpp`: longitudinal Doppler in
Minkowski against `sqrt((1-beta)/(1+beta))`; radial gravitational redshift
between two static Schwarzschild observers against
`sqrt((1 - r_s/r_1)/(1 - r_s/r_2))`, using an actually-propagated photon
rather than the endpoints alone; and cosmological redshift in
`MatterDominatedFLRW` against `1/a(T_observed)`, likewise from a propagated
photon.

### Electromagnetic fields

Electromagnetism is a ladder, the same shape as Gravity's: `Field.hpp` is
its first rung, analytic point-charge superposition, quasi-static. Each
source's *present* position and velocity are what matter, not where it was
one light-travel-time ago. `Maxwell.hpp`/`Maxwell3D.hpp` (below) are the
second rung, a field genuinely evolved from Maxwell's equations rather
than assumed instantaneous -- but as an initial-value wave solve, not a
source term: neither takes a moving charge's trajectory and produces the
field it retards into. A field sourced that way (full Liénard-Wiechert
retarded potentials, replacing `Field.hpp`'s quasi-static approximation
with the exact relativistic one) is not implemented.

```
E(at) = k_e sum_i  q_i (at - r_i) / |at - r_i|^3
B(at) = (mu0/4pi) sum_i  q_i v_i x (at - r_i) / |at - r_i|^3
```

Coulomb's law for E; the point-charge form of Biot-Savart for B. `mu0`
(vacuum permeability) and `k_e` (Coulomb's constant, `1/(4 pi epsilon0)`)
live in `Physics/Electromagnetism/Field.hpp` rather than
`Units/Constants.hpp`, on the same grounds `G` does: since the 2019 SI
redefinition tied the ampere to the elementary charge, `mu0` is measured,
not exact, and it parameterizes this interaction rather than the vocabulary
physics is written in. `epsilon0` and `k_e` are computed from `mu0` and the
(exact) speed of light rather than typed independently, so retyping one
cannot let them drift apart from each other.

A source exactly at the query point is skipped rather than producing an
infinite or NaN contribution: `em_field.cpp`'s
`SkipsASourceExactlyAtTheQueryPoint` checks that querying at one charge's
own position still returns the well-defined field from every other charge.

### The Lorentz force

`F = q (E + v x B)`, `Physics/Electromagnetism/Lorentz.hpp`. Nothing beyond
the definition: `em_lorentz.cpp` checks the electric and magnetic terms
individually (the magnetic term is always perpendicular to velocity, and so
does no work, which is the check that catches a transposed cross product),
their sum, and that an uncharged body feels no force regardless of the
fields present.

### Maxwell FDTD

Electromagnetism's second rung: a field that actually propagates, evolved
from Maxwell's equations rather than assumed instantaneous the way
`Field.hpp`'s superposition is. `MaxwellField1D` restricts to a transverse
electromagnetic wave travelling along x, `Ey` and `Bz`:

```
dEy/dt = -c^2 dBz/dx
dBz/dt = -dEy/dx
```

**Scope: one spatial dimension**, the same restriction `Math/Grid.hpp`
states for itself; what this validates is what 1D vacuum electrodynamics
actually predicts, not a toy simplification of it. The full 3D Yee-grid
solver, what a genuine radiating source needs, is
[3D Maxwell: the full Yee grid](#3d-maxwell-the-full-yee-grid) below.

**The Yee staggering.** `Ey` lives at the grid's integer points, `Bz` at the
half-integer points between them, and time is staggered the same way: one
`step()` advances `Bz` by half a time step from the current `Ey`, then `Ey`
by a full step from the updated `Bz`. This is a leapfrog, the same
structure `Math/Integrators/Symplectic.hpp` uses for a particle, applied
here to a field instead, and for the same reason: it is what makes the
scheme conserve energy rather than merely approximate its conservation.

**The magic time step.** At Courant number 1, `dt = spacing / c`, this
particular 1D scheme has no numerical dispersion at all: a traveling wave's
peak advances by exactly one grid cell per step, not approximately one.
`em_maxwell.cpp`'s `ARightMovingPulsePeakTravelsAtExactlyC` checks the peak
position directly rather than the whole waveform, since a Gaussian pulse is
not a single discrete eigenmode of the scheme and picks up a small
non-propagating residual that a per-cell comparison would (correctly) flag
as motion the scheme's own energy-conservation property does not forbid;
the peak location is a cleaner, still-rigorous signature of the propagation
speed. `EnergyIsConservedOverManySteps` covers the finer-grained claim, that
the field's total energy holds over hundreds of steps.

**Boundary condition: periodic.** The simplest choice for validating
propagation and conservation in a closed system; an absorbing boundary
(so a wave leaves the domain rather than wrapping around) is not
implemented.

### 3D Maxwell: the full Yee grid

`Maxwell3D.hpp`'s `MaxwellField3D` generalizes the leapfrog scheme above to
the full vacuum Maxwell curl equations,

```
dEx/dt = c^2 (dBz/dy - dBy/dz)     dBx/dt = -(dEz/dy - dEy/dz)
dEy/dt = c^2 (dBx/dz - dBz/dx)     dBy/dt = -(dEx/dz - dEz/dx)
dEz/dt = c^2 (dBy/dx - dBx/dy)     dBz/dt = -(dEy/dx - dEx/dy)
```

on the standard Yee cell: each of the six field components keeps its own
staggered position (`Ex` at `(i+1/2,j,k)`, `Ey` at `(i,j+1/2,k)`, `Ez` at
`(i,j,k+1/2)`; `Bx` at `(i,j+1/2,k+1/2)`, `By` at `(i+1/2,j,k+1/2)`, `Bz`
at `(i+1/2,j+1/2,k)`), chosen precisely so every curl term any component
needs is a plain adjacent-index difference of another component already
staggered to exactly the right position -- no interpolation, the same
"only ever a 2-point difference" property the 1D scheme has, just with two
curl terms per update instead of one. `B` updates read a forward
difference of `E` (matching the 1D scheme's `Bz -= dt/dx (Ey[i+1] -
Ey[i])`); `E` updates then read a backward difference of the
just-updated `B`.

**No exact "magic timestep" in 3D.** The 1D scheme's zero-dispersion step
is a genuine coincidence of that specific 1D discretization exactly
matching the continuum wave equation; no analogous step exists on a
Cartesian Yee grid in 3D, where numerical dispersion becomes
direction-dependent (a textbook FDTD fact, not a gap to close).
`MaxwellField3D` exposes `stableTimeStep(courantFactor)`, the CFL limit
`h / (c sqrt(3))`, only.

**Validation.** Beyond
[the dimensional-reduction cross-check](#the-dimensional-reduction-cross-check)
below (a field uniform in `y` and `z`, only `Ey`/`Bz` populated, must
reduce exactly to `MaxwellField1D`) two genuinely 3D checks:
energy staying bounded (not exactly conserved, per the point above) over
thousands of steps for a transverse plane wave travelling along the grid
diagonal; and that same plane wave's actual simulated angular frequency
matching the continuum `omega = c|k|` to the tolerance a second-order
scheme's `O((k h)^2)` dispersion error predicts. A single Fourier mode
is used for both, rather than a localized pulse, since it is an exact
eigenmode of the periodic finite-difference curl operator regardless of
any numerical dispersion in its time evolution, where a pulse (a
superposition of many modes, each dispersing at a slightly different
rate) is not.

### The dimensional-reduction cross-check

Every 3D solver below (`MaxwellField3D`, `AcousticField3D`, `HeatEquation3D`,
`EulerianFluid3D`, and `TimeDependentWavefunction3D`) shares one validation
technique in addition to whatever closed form or conservation law is
specific to its own physics: a 3D field built uniform (translationally
invariant) along two of its three axes, with periodic boundaries, must
evolve *identically*, to floating-point precision, to the corresponding
1D solver run on the remaining axis. This is not an approximation to check
against a loose tolerance -- it is an exact identity for every scheme
here, since each one's spatial operator (a curl, a divergence, a
Laplacian, a flux difference, a Fourier mode) reduces its "flat" axes'
contributions to differences between identical neighboring values, which
are exactly zero in floating point, not merely small. Each 3D test file
builds its field with a small cell count (typically 4) on the two flat
axes and checks every 3D cell against the 1D solver's own state after the
same number of identical steps, which is what actually ties a brand new
3D implementation back to its already-validated 1D sibling rather than
only to itself.

Where a genuinely 3D case is also needed (a diagonal plane wave, an
off-center blast, a product wavefunction along all three axes, all of
which no flat-axis reduction can exercise), each solver's own subsection
below covers it separately.

### Acoustic wave equation

`Physics/Acoustics/Acoustic.hpp`'s `AcousticField1D` is Maxwell FDTD's own
scheme above, reused rather than re-derived. Linearizing the continuity
equation, `d(rho')/dt + rho0 du/dx = 0`; Euler's equation,
`rho0 du/dt = -d(p')/dx`; and the adiabatic relation between pressure and
density perturbations, `p' = c^2 rho'`, gives:

```
dp/dt = -rho0 c^2 du/dx
du/dt = -(1/rho0) dp/dx
```

for pressure perturbation `p`, particle velocity `u`, medium density
`rho0`, and sound speed `c`. This is `dEy/dt = -c^2 dBz/dx`,
`dBz/dt = -dEy/dx` with `p <-> Ey`, `u <-> Bz`, `rho0 c^2 <-> c^2`,
`1/rho0 <-> 1`: the identical linear wave operator, so the Yee/leapfrog
staggering, the CFL stability limit, the exact "magic" time step
`spacing / c` (`c` the *sound* speed here, a per-medium constructor
parameter rather than a universal constant), and the symplectic-style
energy conservation all carry over unchanged -- properties of the
discretized operator, not of which physical fields it is stepping.

**Energy density** is kinetic plus compressional potential,
`(1/2) rho0 u^2 + (1/2) p^2 / (rho0 c^2)`, the direct analog of Maxwell's
electric-plus-magnetic energy density (`(1/2) epsilon0 Ey^2 + Bz^2 /
(2 mu0)`) under the same field correspondence.

`acoustics_acoustic.cpp` mirrors `em_maxwell.cpp`'s validation exactly: a
right-moving Gaussian pressure pulse (`u = p / (rho0 c)`, the plane-wave
impedance relation, the acoustic analog of `Bz = Ey / c`) advances by
precisely one grid cell per step at the magic time step, and the field's
total energy holds over hundreds of steps.

### 3D acoustics

`Acoustic3D.hpp`'s `AcousticField3D` generalizes the same way
`MaxwellField3D` does, but on a staggered (marker-and-cell) grid rather
than a full Yee cell: `p` at cell centers, `ux`/`uy`/`uz` each at their own
face centers, updated by the direct 3D generalization of
`dp/dt = -rho0 c^2 div(u)`, `du/dt = -(1/rho0) grad(p)`. Simpler than
`MaxwellField3D` because divergence and gradient have no cross-axis
coupling the way curl does -- each velocity component's update reads only
pressure differences along its *own* axis, not the other two. Same "no
exact magic timestep in 3D" point as `MaxwellField3D`, and the same
validation shape: [the dimensional-reduction cross-check](#the-dimensional-reduction-cross-check)
against `AcousticField1D`, energy bounded over many steps for a diagonal
plane wave, and that wave's measured frequency matching `c|k|` to
`O((k h)^2)`.

### SPH

Smoothed Particle Hydrodynamics: fluid rung 1, Lagrangian (particles carry
the fluid with them), no grid. The cubic spline kernel (Monaghan and
Lattanzio 1985), compact support at `2h`:

```
W(r, h) = sigma / h^3 * { 1 - 1.5 q^2 + 0.75 q^3   0 <= q < 1
                           0.25 (2 - q)^3           1 <= q < 2
                           0                        q >= 2 }
q = r / h,  sigma = 1 / pi   (3D normalization)
```

Density is the kernel sum, `rho_i = sum_j m_j W(|r_i - r_j|, h)`, including
`j = i`; pressure is a polytropic equation of state,
`P = k rho^gamma`. The pressure-gradient acceleration is the symmetric form

```
a_i = - sum_j  m_j (P_i / rho_i^2 + P_j / rho_j^2) grad_i W_ij
```

which conserves momentum exactly, the same reasoning as Newtonian gravity's
pairwise antisymmetry: `grad_i W_ij = -grad_j W_ji`, since `W` depends only
on `|r_i - r_j|`, so every pairwise contribution to the total momentum
cancels regardless of configuration or equation of state.
`fluids_sph.cpp`'s `PressureAccelerationsConserveMomentumExactly` checks
this directly, on a perturbed (deliberately non-symmetric) lattice, so the
cancellation is not hidden behind an artificially symmetric setup.

**Kernel normalization** is checked by numeric quadrature,
`integral W(r,h) 4 pi r^2 dr` over `[0, 2h]` equal to 1, and density
against the known bulk value on a uniform lattice: SPH's kernel-sum density
estimate is not exact even in the bulk of a perfect lattice, a few percent
of scatter is the kernel's own discretization error at this `h / spacing`
ratio, not a bug, and the test's tolerance reflects that.

**Neighbor search** uses `Math/SpatialPartition/KdTree.hpp`'s `radiusQuery`
at `2h` rather than a loop over every other particle: the kernel is exactly
zero past that radius, so restricting the sum to it is not an approximation,
only skipping pairs that would have contributed zero anyway. Both
`computeDensityAndPressure` and `pressureAccelerations` build a fresh tree
from the current positions each call, since a tree built once and reused
across timesteps would silently miss neighbors that moved into range since
it was built.

**Scope.** No artificial viscosity, so this rung suits smooth, low
Mach-number flows; the Eulerian rung below is built for shocks instead. No
self-gravity: an application couples this to Physics/Gravity's own
accelerations if a scenario needs both.

### The Eulerian solver and the Sod shock tube

Fluid rung 2: Eulerian (the fluid moves through a fixed Math/Grid.hpp
mesh), the standard shape for compressible gas dynamics and shocks, which
SPH resolves poorly without artificial viscosity. The compressible Euler
equations for an ideal gas, adiabatic index `gamma`, one spatial dimension:

```
d(rho)/dt   + d(rho u)/dx       = 0
d(rho u)/dt + d(rho u^2 + p)/dx = 0
d(E)/dt     + d(u (E + p))/dx   = 0
```

solved by a first-order finite-volume method with the Rusanov (local
Lax-Friedrichs) flux: the average of the two neighbouring cells' physical
fluxes, stabilized by a diffusive term scaled to the fastest signal speed
either cell can carry, `|u| + c` where `c = sqrt(gamma P / rho)`. Periodic
boundaries, the same choice `MaxwellField1D` makes, so mass, momentum and
energy are exactly conserved: whatever flux leaves one edge enters the
other, and the sum telescopes to zero regardless of what is happening
inside the domain.

**Scope.** First-order in space and time: robust and simple to verify
exactly, at the cost of smearing a shock or contact discontinuity over
several cells instead of resolving it sharply, the standard tradeoff a
first-order Godunov-type scheme makes. `fluids_shocktube.cpp` validates
against the Sod problem's qualitative structure, not an exact Riemann
solution: a rarefaction fan, a plateau of constant pressure and velocity
(the defining property of a contact discontinuity, where only density
jumps), and a shock, rather than the precise analytic values an exact
Riemann solver would need to produce.

**A periodic domain laid out as two half-states is two shock tubes, not
one.** Splitting the domain into a left half and a right half produces the
intended discontinuity at the midpoint, and, because the domain wraps, an
identical second one where index `count - 1` meets index `0`. Both launch
the same rarefaction-contact-shock structure, so "still at the initial,
undisturbed state" has to be checked at the points equidistant from both
discontinuities, the quarter and three-quarter marks, not at the domain's
own edges, which sit immediately next to the wrap-around discontinuity and
are disturbed from the first step. This is what
`fluids_shocktube.cpp`'s header comment walks through: it is what the test
actually measured before the sampling points were corrected, not a
hypothetical.

**The shock outruns the rarefaction.** For this classic configuration
(`rho_L = 1, p_L = 1, rho_R = 0.125, p_R = 0.1, gamma = 1.4`), the shock
that propagates into the low-pressure gas moves faster than the
rarefaction's leading edge moves into the high-pressure gas. That is a
genuine physical asymmetry of the Sod problem, not a bug, and it is why the
three-quarter mark (in the shock's path) needs a shorter run than the
quarter mark (in the rarefaction's) to stay undisturbed.

### 3D Eulerian flow: dimensional splitting

`Eulerian3D.hpp`'s `EulerianFluid3D` extends the scheme above to 3D by
**dimensional (Godunov) splitting**: a full x-sweep, then a full y-sweep,
then a full z-sweep, each `dt`, each being the identical 1D Rusanov update
generalized to carry the two transverse momentum components through every
flux passively (`flux = rho u v`, `rho u w`, no pressure term -- a
1D-normal Riemann problem along one axis has nothing to say about the
other two) while the Rusanov dissipation term still applies uniformly to
all five conserved quantities. Splitting adds no accuracy and removes
none: both the split-off 1D pieces and the original scheme are already
first order, so this stays exactly the "robust, simple to verify, at the
cost of smearing a shock" tradeoff the 1D scope note above already makes,
just per-axis.

**Validation.** [The dimensional-reduction cross-check](#the-dimensional-reduction-cross-check)
against `EulerianFluid1D` (a Sod-tube-style discontinuity uniform in `y`
and `z` -- the y- and z-sweeps are provably no-ops on uniform data, since
a Rusanov flux between two identical states is exactly the physical flux
with zero dissipation, and consecutive equal fluxes cancel exactly);
exact conservation of mass, all three momentum components, and energy for
a genuinely 3D case (an off-center high-pressure region inside a moving
ambient medium, the nonzero background velocity making each momentum
component's own conservation a real check rather than "stays near zero");
no exact Riemann-structure check for the 3D case, since no simple closed
form exists for a genuinely multi-dimensional shock.

### Closed-form thermodynamics

Thermodynamics' first rung: relations with no space or time dependence, in
`Physics/Thermodynamics/Thermodynamics.hpp`.

**The ideal gas law**, mass form: `p = rho R_specific T`, where
`R_specific = R / M` is particular to the gas in question (about
287 J/(kg K) for dry air), not the universal gas constant. This module
works in densities rather than moles, which is why it is the specific form
rather than `pV = nRT`.

**The adiabatic relation**, `p V^gamma = const` for a reversible
(isentropic) compression or expansion, `gamma` the ratio of specific heats:
5/3 for a monatomic ideal gas, 7/5 for diatomic.

**The Stefan-Boltzmann constant**, `sigma = 2 pi^5 k^4 / (15 h^3 c^2)`,
computed from the SI-defining constants in `Units/Constants.hpp` rather
than typed independently, so it cannot drift from them: since the 2019
redefinition fixed `h`, `k` and `c` exactly, `sigma` is exact, not
measured, the same reasoning `Units/Constants.hpp`'s own
`reducedPlanckConstant` already uses. `blackBodyLuminosity` is then
`4 pi r^2 sigma T^4`, the total power radiated by a sphere at uniform
temperature, treated as an ideal black body.

**Wien's displacement constant**, `lambda_max T = b`, is transcribed rather
than computed, unlike `sigma`: `b = (h c / k) / x`, where `x` is the root
of `x = 5 (1 - e^-x)`, the condition for Planck's law to peak, and that
equation has no closed form. `b` is still exact in the same sense `sigma`
is, since it follows from `x` and the exact SI constants alone; it is the
transcription of `x` itself that cannot be avoided, not an approximation
of the physics.

### Statistical mechanics

The Maxwell-Boltzmann speed distribution, in
`Physics/Thermodynamics/StatisticalMechanics.hpp`: the distribution of
molecular speeds a gas in thermal equilibrium actually has, which the
ideal gas law's pressure is a bulk average over.

Every closed form here is expressed in units of the distribution's own
scale parameter, `a = sqrt(k T / m)`, the speed at which the exponential
factor `exp(-v^2 / (2a^2))` falls to `1/e`. The density,

```
f(v) = sqrt(2/pi) (v/a)^2 exp(-v^2 / (2a^2)) / a
```

is a chi distribution with three degrees of freedom (speed is the
magnitude of a 3D normally-distributed velocity), which is why its
cumulative distribution has the standard chi-3 closed form,
`erf(v / (sqrt(2) a)) - sqrt(2/pi) (v/a) exp(-v^2 / (2a^2))`, via
`Math/SpecialFunctions.hpp`'s `erf`.

**The general moment formula**, via `Math/SpecialFunctions.hpp`'s `gamma`:

```
<v^n> = (2^(n/2 + 1) / sqrt(pi)) a^n Gamma((n + 3) / 2)
```

is what the three named characteristic speeds are special cases of: `n = 1`
gives the mean speed `sqrt(8 k T / (pi m))`, `n = 2` gives the
mean-square speed `3 k T / m` (so `sqrt` of it is the RMS speed, the
speed whose kinetic energy `(1/2) m v_rms^2 = (3/2) k T` is the
equipartition result for three translational degrees of freedom). The
most probable speed, `sqrt(2) a`, is not a moment at all, the density's
own peak rather than an average, and is derived directly rather than
through this formula. `physics_thermodynamics_statisticalmechanics.cpp`
checks the general moment formula against direct numerical integration of
`v^n f(v)` for `n = 1, 2` (not merely against itself for the closed forms
that reduce to it), and the standard textbook ratios `v_p : v_mean :
v_rms = sqrt(2) : sqrt(8/pi) : sqrt(3)`, plus a real-world number:
nitrogen at room temperature has an RMS speed of about 517 m/s.

### Radiative transfer

`Physics/Thermodynamics/RadiativeTransfer.hpp` generalizes
`blackBodyLuminosity` from radiating into free space to exchanging with a
second finite surface:

```
Q = sigma A1 F_12 (T1^4 - T2^4)
```

`F_12`, the view factor, is the fraction of surface 1's own radiation that
reaches surface 2; `blackBodyLuminosity` is the special case `F_12 = 1`,
`T2 = 0`, everything emitted escaping to a sink at absolute zero.

**The coaxial-disk view factor** is one of the few surface-pair
configurations with an exact closed form (Incropera et al.):

```
R_i = radius1 / distance,  R_j = radius2 / distance
S = 1 + (1 + R_j^2) / R_i^2
F_ij = (1/2) [S - sqrt(S^2 - 4 (R_j/R_i)^2)]
```

implemented instead as `2x / (S + sqrt(S^2 - 4x))` (`x = (R_j/R_i)^2`), the
algebraically equivalent form reached by multiplying through by the
textbook expression's own conjugate: the textbook form subtracts two
nearly equal large numbers whenever the disks are far apart relative to
their radii (`S` large), losing most of its own precision to cancellation
exactly where the view factor is smallest and that precision matters
most, which `physics_thermodynamics_radiativetransfer.cpp`'s
`CoaxialDiskViewFactorVanishesAtLargeSeparation` is what actually caught --
the textbook form failed that check at 1000 radii of separation, and the
division form does not.

Three geometric limits and one physical identity are checked directly:
the view factor approaches 1 as two equal disks are brought together
(nearly everything one emits reaches the other), vanishes at large
separation, stays within `[0, 1]` generally, and satisfies the reciprocity
relation `A1 F_12 = A2 F_21` regardless of which disk is named first.

### The heat equation

Thermodynamics' second rung: `dT/dt = alpha d^2T/dx^2` in one spatial
dimension, `alpha` the thermal diffusivity `k / (rho c_p)`, solved by the
explicit forward-time, centred-space (FTCS) scheme:

```
T_i^(n+1) = T_i^n + alpha dt/dx^2 (T_(i+1)^n - 2 T_i^n + T_(i-1)^n)
```

stable for `alpha dt / dx^2 <= 0.5`, the standard FTCS limit;
`stableTimeStep` applies a safety factor below that. Explicit rather than
an unconditionally stable implicit scheme (Crank-Nicolson, say): the
tradeoff is a stricter step-size limit in exchange for a scheme simple
enough to validate directly against an exact solution, rather than trusting
an implicit solve's own correctness as a second unverified layer.

**Validated against the fundamental solution.** A Gaussian temperature
profile stays Gaussian under 1D diffusion, with its variance growing
linearly, `sigma^2(t) = sigma^2(0) + 2 alpha t`. `thermodynamics_heatequation.cpp`
measures the profile's second moment directly (temperature-weighted, about
its own measured centroid rather than an assumed fixed one) after a known
elapsed time and compares it to that closed form, which is a substantially
more direct check than comparing pointwise profiles, since it does not
depend on exactly where a Gaussian's tails have spread to being resolved
by the grid. Periodic boundaries, the same choice `MaxwellField1D` and
`EulerianFluid1D` make, so total heat, `sum T dx`, is exactly conserved,
checked separately from the spreading-rate measurement.

### 3D heat diffusion

`HeatEquation3D.hpp`'s `HeatEquation3D` sums the same centred-difference
idea over all three axes, the standard 7-point Laplacian:

```
T_ijk^(n+1) = T_ijk^n + alpha dt/h^2 (T_(i+1)jk + T_(i-1)jk + T_i(j+1)k
               + T_i(j-1)k + T_ij(k+1) + T_ij(k-1) - 6 T_ijk)
```

stable for `alpha dt / h^2 <= 1/6` -- three times tighter than 1D's `1/2`,
since the Laplacian now picks up two neighbor terms per axis across three
axes instead of one. Still second order: this is a dimensional extension
of the 1D scheme, not also an accuracy upgrade, so it does not reach for
`Math/FiniteDifference.hpp`'s fourth-order stencils (built for
`Physics/Spacetime`'s BSSN evolution, a different accuracy target).

**Validated three ways**: total heat conservation (as in 1D); the 3D heat
kernel, which separates exactly into a product of three 1D Gaussians
(`T(x,y,z,t) = T0 prod_axis gaussian(axis, t)`, since the Laplacian is a
sum of independent per-axis second derivatives), checked by marginalizing
the 3D field onto one axis and measuring that axis's own variance growth
rate against the identical 1D closed form; and
[the dimensional-reduction cross-check](#the-dimensional-reduction-cross-check)
against `HeatEquation1D`.

### The discretized Schrödinger equation

`Physics/QuantumMechanics/Schrodinger.hpp` discretizes
`-hbar^2/(2m) d^2(psi)/dx^2 + V(x) psi = E psi` with the standard 3-point
central-difference stencil for the second derivative, turning it into a
real symmetric matrix eigenvalue problem:

```
H[i][i]   = hbar^2 / (m dx^2) + V(x_i)
H[i][i-1] = H[i][i+1] = -hbar^2 / (2 m dx^2)
```

(Dirichlet boundary, `psi = 0` just outside the grid: the standard
"particle confined to this domain" condition), solved directly by
`Math/Eigen.hpp`'s `jacobiEigenSymmetric` -- the Hamiltonian discretizes to
exactly the kind of matrix that solver exists for, no adaptation needed.
Each returned eigenvector is normalized so `sum psi_i^2 dx = 1`.

**Validated against two textbook closed forms**, not only against the
eigensolver's own guarantees (which `physics_quantummechanics_schrodinger.cpp`'s
`EigenstatesAreOrthonormal` checks separately): the infinite square well
(`V = 0` inside, confined by the Dirichlet boundary itself acting as
infinite walls), `E_n = n^2 pi^2 hbar^2 / (2 m L^2)`; and the quantum
harmonic oscillator (`V(x) = (1/2) m omega^2 x^2`),
`E_n = (n + 1/2) hbar omega`. Both are standard results with no free
parameters to tune into agreement, so matching them is a genuine test of
the discretization rather than a circular one.

### Split-step Fourier evolution

`Physics/QuantumMechanics/WavePacket.hpp`'s `TimeDependentWavefunction1D`
evolves `i hbar d(psi)/dt = [-hbar^2/(2m) d^2/dx^2 + V(x)] psi` by the
second-order Strang-split split-step Fourier method (Feit, Fleck &
Steiger 1982):

```
psi <- exp(-i V dt / (2 hbar)) psi                          [position space]
psi_k <- FFT(psi); psi_k <- exp(-i hbar k^2 dt / (2m)) psi_k; psi <- IFFT(psi_k)
psi <- exp(-i V dt / (2 hbar)) psi                          [position space]
```

`k` follows the same spatial-frequency mapping
`Optics/Diffraction.hpp`'s `fraunhoferDiffraction` already uses,
`k = 2 pi j / (N dx)`, negative for `j > N/2`. Every operator applied is a
pure phase, so **unitarity is structural, not merely approximate** --
`sum |psi_i|^2 dx` cannot drift at any single step, which
`physics_quantummechanics_wavepacket.cpp`'s
`TotalProbabilityStaysNormalizedOverManySteps` confirms holds over
hundreds of steps in practice, not only in principle.

**Energy conservation is checked by a deliberately independent
measurement.** `expectationEnergy()` computes `<H>` with the *same*
3-point finite-difference Hamiltonian `Schrodinger.hpp` uses, not the
FFT's spectral kinetic operator `step()` actually propagates with: two
different discretizations of the same continuous operator, exact for each
other's own eigenstates only in the continuum limit. A small,
non-accumulating offset between what each discretization reports is the
expected signature of that mismatch (and shrinks as the grid refines), not
drift; `physics_quantummechanics_wavepacket.cpp`'s energy-conservation
test tolerance is set to comfortably cover it while still catching genuine
drift, which would keep growing over further steps rather than saturate.

**The cross-check linking both files.** Initializing
`TimeDependentWavefunction1D` with `solveTimeIndependentSchrodinger`'s own
ground state (for the harmonic oscillator) and evolving it must leave the
probability density unchanged: a stationary state only picks up an
overall phase `exp(-i E t / hbar)`, invisible in `|psi|^2`. This only
holds at all if the eigenvalue solver and the propagator agree on the same
underlying physics -- `AnEigenstateStaysStationaryUnderEvolution`'s
tolerance, like the energy check's, is set relative to the wavefunction's
own peak density to absorb the same finite-difference-versus-spectral
discretization mismatch discussed above, not to hide an error.

### 3D quantum mechanics

`Schrodinger3D.hpp` discretizes the 3D Laplacian the same way
`HeatEquation3D.hpp` does, the standard 7-point stencil, turning
`-hbar^2/(2m)(d^2/dx^2+d^2/dy^2+d^2/dz^2) psi + V psi = E psi` into a
real symmetric eigenvalue problem exactly like the 1D one, only bigger
(`N = nx ny nz` instead of one point per grid cell), solved by the same
`jacobiEigenSymmetric`.

**Scope: dense eigensolver, so only modest grids.** `Math/Eigen.hpp` has
no sparse/iterative eigensolver (Lanczos, say) -- building one is separate
scope well beyond a 1D-to-3D extension, flagged rather than silently
worked around. `jacobiEigenSymmetric` is `O(N^3)`, so this stays correct
and general but practical only up to about `8x8x8 = 512`, not a
production-scale simulation.

**Validated by separability, not a resolution-dependent closed form.**
Both the 3D infinite well (potential zero everywhere) and the 3D isotropic
harmonic oscillator (`V(x,y,z) = Vx(x)+Vy(y)+Vz(z)`, itself a sum of three
identical 1D potentials) have Hamiltonians that are exactly the Kronecker
sum of three copies of the identical 1D discretized Hamiltonian
`Schrodinger.hpp` already solves. A Kronecker sum's eigenvalues are
exactly every sum of three eigenvalues of its summands, so
`physics_quantummechanics_schrodinger3d.cpp` computes every such triple
sum from the *already-validated 1D solver's own output* and checks the 3D
solver's eigenvalues against that list directly -- an exact
cross-check tied to existing, proven code, not a comparison to the
continuum closed form loosened to fit whatever resolution the dense
eigensolver's cost allows.

`WavePacket3D.hpp`'s `TimeDependentWavefunction3D` is `WavePacket.hpp`'s
split-step method with `Math/FFT.hpp`'s `fft3D`/`ifft3D` in place of
`fft`/`ifft`: the kinetic operator stays diagonal in momentum space in any
dimension, since `kx^2+ky^2+kz^2` separates additively into one phase
factor per axis. Storage is a flat `std::vector<Complex<double>>`, not a
`Grid3D`, for the same reason `TimeDependentWavefunction1D` is not a
`Grid1D`: `Complex` does not satisfy `Numeric`.

**Validated the same four ways as the 1D version**, at a wider tolerance
where three axes compound the same finite-difference-versus-spectral
mismatch `WavePacket.hpp`'s own derivation above discusses: unitarity;
energy conservation via the 7-point finite-difference Hamiltonian; a
stationary eigenstate staying stationary; and
[the dimensional-reduction cross-check](#the-dimensional-reduction-cross-check)
against `TimeDependentWavefunction1D`. The stationary-state check builds
its eigenstate as a product of three 1D ground states from
`Schrodinger.hpp`'s fast solver rather than calling the 3D dense
eigensolver at the same resolution — exactly valid, since a separable
Hamiltonian's ground state is exactly the product of its 1D pieces' own
ground states, and far cheaper than an `O(N^3)` solve would be at a
matching grid size.
