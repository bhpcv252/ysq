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
are linked `PRIVATE` and not yet used by anything: the `Compute` edge exists
so the dependency graph in the root `README.md`'s Project structure section
is true from the start, ahead of the GPU-accelerated N-body kernel that will
actually use it; see src/Compute/README.md.

This module is being built in stages; only what is listed below exists so
far.

## Contents

| Header | Purpose |
| --- | --- |
| `Physics/Body.hpp` | Matter state: mass, charge, position, momentum |
| `Physics/Mechanics/Frame.hpp` | Inertial reference frames, Galilean transform |
| `Physics/Mechanics/Kinematics.hpp` | Lorentz factor, four-velocity, proper time, relativistic velocity addition |
| `Physics/Mechanics/Dynamics.hpp` | `NBodyState`, the boundary between a span of `Body` and Math's integrators |
| `Physics/Gravity/Newtonian.hpp` | Pairwise gravity, direct-sum N-body, potential energy |
| `Physics/Gravity/PostNewtonian.hpp` | The 1PN two-body correction (perihelion precession) |
| `Physics/Gravity/BarnesHut.hpp` | O(N log N) approximate N-body gravity |
| `Physics/Spacetime/Metric.hpp` | The `SpacetimeMetric` concept, Christoffel symbols, causal character |
| `Physics/Spacetime/Minkowski.hpp` | Flat spacetime |
| `Physics/Spacetime/Schwarzschild.hpp` | Non-rotating mass |
| `Physics/Spacetime/Kerr.hpp` | Rotating mass |
| `Physics/Spacetime/FLRW.hpp` | Expanding universe, single-component analytic solutions |
| `Physics/Spacetime/Geodesic.hpp` | Timelike and null geodesics |
| `Physics/Optics/Propagation.hpp` | Light as a null geodesic |
| `Physics/Optics/Lensing.hpp` | Gravitational deflection angle |
| `Physics/Optics/FrequencyShift.hpp` | Doppler, gravitational and cosmological shift, one formula |
| `Physics/Electromagnetism/Field.hpp` | Point-charge E and B fields, quasi-static |
| `Physics/Electromagnetism/Lorentz.hpp` | Force on a charged body |
| `Physics/Electromagnetism/Maxwell.hpp` | 1D FDTD: a field that actually propagates |
| `Physics/Fluids/SPH.hpp` | Smoothed Particle Hydrodynamics, Lagrangian, no grid |
| `Physics/Fluids/Eulerian.hpp` | 1D compressible flow on a `Math/Grid.hpp` mesh |
| `Physics/Thermodynamics/Thermodynamics.hpp` | Ideal gas law, adiabatic relation, black-body radiation |
| `Physics/Thermodynamics/HeatEquation.hpp` | 1D heat diffusion on a `Math/Grid.hpp` mesh |

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
staggering, the conservation argument, and what a full 3D solver would still
need to add.

## Fluids: Lagrangian and Eulerian, another ladder

`SPH.hpp` (rung 1, Lagrangian, particles carry the fluid, no grid) and
`Eulerian.hpp` (rung 2, the fluid moves through a fixed `Math/Grid.hpp`
mesh) solve the same physics from opposite ends: SPH suits smooth,
low-Mach flows and has no artificial viscosity to handle a shock; the
Eulerian solver, a first-order finite-volume scheme with the Rusanov flux,
is built for exactly that regime. Neither replaces the other, the same
relationship as every other ladder in this module.

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

The heat equation rung is validated against its fundamental solution, a
Gaussian's variance growing linearly in time, the same kind of exact
closed-form check `MaxwellField1D`'s propagation speed and
`EulerianFluid1D`'s shock-tube structure are; see
[The heat equation](#the-heat-equation) and
[Closed-form thermodynamics](#closed-form-thermodynamics) below for the
derivation and why the Stefan-Boltzmann constant above is computed from the
SI-defining constants rather than typed as its own measured value.

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
(now slowly precessing) ellipse. This is the formula the Mercury-like case in
`physics_gravity.cpp` validates the correction against, and the same formula
`lensing_deflection.cpp` validates a full Schwarzschild geodesic against once
`Physics/Spacetime` exists, confirming the two rungs of the gravity ladder
agree in the regime they overlap.

**Scope.** This is the two-body, test-particle form: exact where the source's
mass dominates, which is the regime the precession formula itself assumes.
The N-body generalization is the Einstein-Infeld-Hoffmann equations, which
add cross terms between every pair of bodies and are not implemented here.

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
one light-travel-time ago; a field actually sourced by Maxwell's equations,
with that retardation built in rather than assumed away, is the second
rung and is not implemented yet.

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
states for itself. A full 3D Yee-grid solver, which is what a genuine
radiating source needs, is future work; what this validates is what 1D
vacuum electrodynamics actually predicts, not a toy simplification of it.

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
