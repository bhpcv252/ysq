# Physics

The engine's theories: what things are made of, and the laws that move them.
Organized by theory, not by application; see docs/architecture.md for why
`Mechanics`, `Gravity`, `Spacetime`, `Electromagnetism`, `Fluids`,
`Thermodynamics` and `Optics` are siblings rather than one another's
dependencies.

**Target:** `ysq::Physics` (static)
**Depends on:** `ysq::Math`, `ysq::Units`, both `PUBLIC` since every header
here hands back a `Quantity` or a `Math` type. `ysq::Core` and `ysq::Compute`
are linked `PRIVATE` and not yet used by anything: the `Compute` edge exists
so the dependency graph in docs/architecture.md is true from the start,
ahead of the GPU-accelerated N-body kernel that will actually use it; see
src/Compute/README.md.

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
  docs/units.md and docs/physics.md.
- **`postNewtonianCorrection`**: the 1PN acceleration correction for a test
  particle orbiting a dominant source, added to the Newtonian acceleration
  rather than replacing it. Scope is two bodies, one of them a test particle;
  see the header for why, and docs/physics.md for the derivation.
- **`BarnesHutTree`**: an octree built fresh every call, trading exact
  pairwise summation for O(N log N) at an accuracy set by its opening angle.
  Monopole only.

`NewtonianField` and `BarnesHutTree` both implement
`AccelerationField<NBodyState>` (see `Math/ODE.hpp`), so either drops
directly into `VelocityVerletStepper<NBodyState>` or any of the other
steppers in `Math/Integrators/`.

## Softening

Direct summation and Barnes-Hut both take a softening length: `a = GM
(r_j - r_i) / (|r_j - r_i|^2 + softening^2)^(3/2)`, the exact Newtonian force
at separations well above `softening` and finite rather than singular as two
bodies approach each other. `newtonianPotentialEnergy` takes the same
parameter, and it has to be the same value the acceleration was computed
with: an energy conservation check is a statement about the system actually
being integrated, the softened one, not the unsoftened one it approximates.

Derivations, the softening rationale, and Barnes-Hut's error-versus-theta
tradeoff are in [docs/physics.md](../../docs/physics.md).

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
or spherical, is each metric's own chart; `docs/physics.md` states each
one's.

**Geodesics do not use Gravity's steppers.** The geodesic equation's
`Gamma^mu_ab v^a v^b` term is quadratic in velocity, unlike an
`AccelerationField`'s `a(t, q)`, so `geodesicSystem` builds a plain
`OdeSystem` for `Rk4Stepper` or the adaptive Dormand-Prince stepper instead.
One system serves both timelike and null geodesics: which one a run is
depends only on whether the initial four-velocity is normalized to `-c^2` or
to `0`, checked with `isTimelike` / `isNull`.

Full derivations, each metric's line element, and the geodesic tests'
reasoning are in [docs/physics.md](../../docs/physics.md).

## Optics: light propagation, lensing and frequency shift are one geodesic

Per docs/architecture.md, these are not three features but one computation,
a null geodesic through a metric, looked at three ways:

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
geometry fact rather than a numerical error. `docs/physics.md` has the
derivation and how getting this wrong actually surfaced during testing.

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
the field's energy; see `docs/physics.md` for the staggering, the
conservation argument, and what a full 3D solver would still need to add.

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
discrete lattice nor a first-order shock capture is meant to be exact.
`docs/physics.md` has the details, including a genuine pitfall the shock
tube test walked into and how it was diagnosed: a periodic domain split
into two half-states is two shock tubes, not one.

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
`EulerianFluid1D`'s shock-tube structure are; `docs/physics.md` has the
derivation and why the Stefan-Boltzmann constant above is computed from
the SI-defining constants rather than typed as its own measured value.
