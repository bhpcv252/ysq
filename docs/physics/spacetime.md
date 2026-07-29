# Spacetime

What a metric is, what a geodesic is, and the four shapes of spacetime YSQ
can put something in.

## The idea

A **metric** is a rule for measuring distance, at every point in space (and
time). In flat, empty space, that rule is the ordinary one from school
geometry, the Pythagorean theorem. General relativity's central claim is
that mass and energy change that rule: near a massive object, the metric is
different than it is far away, and "curved spacetime" is just a name for a
metric that varies from point to point.

Once you have a metric, "what path does something with no forces acting on
it actually follow through this space" has one answer: a **geodesic**, the
straightest possible path the metric allows. In flat spacetime that's an
ordinary straight line at constant velocity. Near a mass, it curves, and
that curving *is* gravity in general relativity: not a force reaching out
and pulling, but the straightest available path itself bending. A free-fall
trajectory and a beam of light both follow geodesics; the only difference
is which kind (see [Optics](optics.md)).

Actually computing a geodesic needs the metric's local rate of change at
every point (the Christoffel symbols), and YSQ gets these *exactly*, not
approximately, by reusing the automatic-differentiation trick from
[docs/math/algebra.md](../math/algebra.md): evaluate the metric with a dual
number seeded in each coordinate direction, and its derivative comes back
exact, to the last bit, with no finite-difference error at all.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Spacetime/Metric.hpp` | The metric concept, Christoffel symbols, causal character (timelike/spacelike/null) |
| `Spacetime/Minkowski.hpp` | Flat spacetime: no gravity, special relativity only |
| `Spacetime/Schwarzschild.hpp` | Spacetime around one non-rotating mass |
| `Spacetime/Kerr.hpp` | The same, but rotating (includes frame dragging) |
| `Spacetime/FLRW.hpp` | An expanding, homogeneous universe: cosmology |
| `Spacetime/Geodesic.hpp` | The solver: one system serves both massive particles and light |

A metric in YSQ is a concept, not a base class: anything with one method,
`components(position) -> a 4x4 tensor`, qualifies. That's what lets
`christoffelSymbols` differentiate any of them the same exact way, and
what makes adding a fifth metric someday a matter of writing one method.

**Which metric models what:**

- **Minkowski**: no mass anywhere, flat spacetime, special relativity.
  Christoffel symbols are exactly zero everywhere, so a geodesic is just an
  ordinary straight line.
- **Schwarzschild**: the spacetime around a single non-rotating mass, a
  star or a non-spinning black hole. Diverges at the Schwarzschild radius
  (a feature of this particular coordinate chart, not a real singularity
  there) and at the true singularity at the center.
- **Kerr**: the same, but rotating. Frame dragging shows up directly: close
  enough to the horizon, an observer can't hold a fixed angular position
  and still be moving forward in time.
- **FLRW**: an expanding universe. Three ready-made solutions for matter-,
  radiation-, and cosmological-constant-dominated expansion.

Geodesics can't use the same symplectic steppers `Physics/Gravity` does
(see [docs/math/integrators.md](../math/integrators.md)): the geodesic
equation is quadratic in velocity, not a plain acceleration, so it rides
`Rk4Stepper` or the adaptive Dormand-Prince stepper instead.

## Using it

```cpp
#include <Physics/Spacetime/Geodesic.hpp>
#include <Physics/Spacetime/Schwarzschild.hpp>
#include <Math/Integrators/RK4.hpp>

const ysq::Schwarzschild schwarzschild{gravitationalParameter};
const ysq::Vec4 at{0.0, r, polar, azimuth};  // (c*t, r, polar, azimuth)

const ysq::MetricTensor<double> g = schwarzschild.components(at);
const ysq::ChristoffelSymbols<double> gamma = ysq::christoffelSymbols(schwarzschild, at);

const auto system = ysq::geodesicSystem(schwarzschild);
ysq::Rk4Stepper<ysq::PhaseState<ysq::Vec4>> stepper;
stepper.step(system, lambda, state, step, next);
```

One system serves both a massive particle's worldline and a photon's path;
which one a run computes depends only on how the initial four-velocity is
normalized (`isTimelike` versus `isNull`), nothing else in the equation
distinguishes them. A concrete, checkable prediction: a light ray launched
tangentially at `r = 1.5 * r_s` around a Schwarzschild mass holds that exact
radius forever, the unstable circular photon orbit, which is what
`spacetime_geodesic.cpp` actually tests.

## Go deeper

[docs/api/physics/spacetime.md](../api/physics/spacetime.md) has every
signature: the `SpacetimeMetric` concept, `christoffelSymbols`, each
metric's constructor and line element, and `geodesicSystem`.

[src/Physics/README.md](../../src/Physics/README.md) has every metric's
full line element, the Christoffel symbol formula and how it's evaluated
exactly, and the spacetime conventions (signature, four-position in metres)
fixed once and used by every metric.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/spacetime)
and let us know.
