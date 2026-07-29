# Tutorial 3: choosing a gravity model

[Tutorial 1](01-your-first-simulation.md) simulated one planet around one
star with direct-summation Newtonian gravity. This tutorial covers the two
directions you'll actually need to move in from there: many more bodies,
and more relativistic accuracy for one. Read
[docs/physics/gravity.md](../physics/gravity.md) alongside this.

## Many bodies: swap in Barnes-Hut

Direct summation costs `O(N^2)`: fine for a handful of planets, far too
slow once `N` reaches into the thousands. `BarnesHutTree` implements the
exact same `AccelerationField<NBodyState>` concept `NewtonianField` does
(see [docs/math/integrators.md](../math/integrators.md)), so swapping one
for the other is a one-line change, nothing else in the simulation loop
moves:

```cpp
#include <Physics/Gravity/BarnesHut.hpp>

// Before: exact, O(n^2)
// ysq::NewtonianField gravity(bodies);

// After: approximate, O(n log n)
ysq::BarnesHutTree gravity(bodies, /*openingAngle=*/0.5);

ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
// everything else from Tutorial 1 is unchanged
```

`openingAngle` (often called `theta`) is the only new knob: `0.0` forces
every node open, which is direct summation with extra bookkeeping and no
speed benefit; `0.5` is the conventional default; larger values trade more
accuracy for more speed. If you're not sure whether the approximation is
hurting a particular scenario, run it at `0.0` and at your chosen `theta`
and compare, the same check `nbody_energy`'s own test does.

## One body, more accuracy: add the 1PN correction

The 1PN correction is a different kind of upgrade: not faster, more
*correct*, for a test particle orbiting one dominant mass (a planet around
a star, not a general many-body system; see
[docs/physics/gravity.md](../physics/gravity.md) for the scope). It's what
produces the extra perihelion precession Mercury's orbit shows and plain
Newtonian gravity can't explain:

```cpp
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Gravity/PostNewtonian.hpp>

const ysq::Acceleration3 newtonianAccel =
    ysq::newtonianAcceleration(testParticle.position, std::array{source});
const ysq::Acceleration3 correction =
    ysq::postNewtonianCorrection(testParticle, source);
const ysq::Acceleration3 total = newtonianAccel + correction;
```

This computes a dimensioned `Acceleration3` directly from `Body`s, one step
below `NBodyState`, the same boundary
[docs/physics/mechanics.md](../physics/mechanics.md) covers. Wiring `total`
into a full multi-step run means writing a small custom acceleration field
for this specific two-body case, the same shape `NewtonianField` itself
has internally: capture the source's mass once, and on each call convert
the raw unitless position back into a `Length3`, evaluate `total` above,
and convert the result back to a raw `Vec3` for the stepper. That's a
handful of lines, not a new abstraction; it's exactly the "cross the units
boundary once" pattern from [docs/physics/mechanics.md](../physics/mechanics.md)
applied to one extra force term.

## Which rung, when

| Situation | Use |
| --- | --- |
| A handful of bodies, exact answer matters | `NewtonianField`, direct summation |
| Thousands of bodies, speed matters more than exactness | `BarnesHutTree` |
| One test particle around one dominant mass, relativistic precession matters | Newtonian plus the 1PN correction |
| A photon, or anything near a compact object where 1PN isn't enough | A fixed background metric; see [Tutorial 4](04-spacetime-and-light.md) |

These compose rather than exclude each other: nothing stops a scenario
from running `BarnesHutTree` for a star cluster's bulk dynamics while
treating one especially close binary pair inside it with the 1PN
correction, if that's what the scenario actually needs.

## Next

[Tutorial 4](04-spacetime-and-light.md) leaves the gravity ladder
entirely, for the regime where a fixed background metric is exact rather
than approximate: light bending around a black hole.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+tutorials/03-choosing-a-gravity-model)
and let us know.
