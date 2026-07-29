# Fluids

Two different answers to the same question: how do you represent something
that has no fixed shape?

## The idea

A rigid body has an obvious representation: a position and an orientation.
A fluid doesn't; it's constantly deforming, splitting, merging. YSQ gives
you two different ways to represent one, each better suited to a different
kind of flow.

**SPH** (smoothed particle hydrodynamics) represents the fluid as a cloud of
particles that carry it with them, like tracking individual blobs of water
as they move. Each particle's local density comes from how crowded its
nearby particles are (a weighted sum called a kernel), and pressure pushes
particles apart from crowded regions toward sparse ones. It's a natural fit
for smooth, slow flow, and it has no way to represent a shock: a sudden
jump in pressure and density, like the front of an explosion.

**The Eulerian solver** takes the opposite approach: the fluid flows
*through* a fixed grid, like reading wind speed off an array of stationary
weather stations rather than following the air itself. This is built for
exactly the case SPH struggles with: the classic test problem here is the
Sod shock tube, a tube of gas at high pressure on one side and low pressure
on the other, released at the midpoint, which develops a rarefaction wave,
a contact surface, and a genuine shock, all of which the Eulerian solver
resolves and SPH would badly smear.

Neither replaces the other; they solve the same physics for different
regimes, the same relationship as the gravity ladder in
[docs/physics/gravity.md](gravity.md).

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Fluids/SPH.hpp` | `SPHParticle`, the cubic-spline kernel, density/pressure, the symmetric pressure-gradient force |
| `Fluids/Eulerian.hpp` | `EulerianFluid1D`: the compressible Euler equations on a fixed 1D grid |

SPH's pressure force is written in a symmetric form that conserves momentum
*exactly*, regardless of how the particles are arranged, the same kind of
structural guarantee Newtonian gravity's pairwise force has. The Eulerian
solver's periodic boundary makes mass, momentum, and energy exactly
conserved too: whatever leaves one edge of the grid enters the other.

## Using it

```cpp
#include <Physics/Fluids/SPH.hpp>

std::vector<ysq::SPHParticle> particles = /* positions, masses, velocities */;
ysq::computeDensityAndPressure(particles, smoothingLength, equationOfStateK,
                                polytropicIndex);
const std::vector<ysq::Vec3> accel =
    ysq::pressureAccelerations(particles, smoothingLength);
```

```cpp
#include <Physics/Fluids/Eulerian.hpp>

ysq::EulerianFluid1D fluid(cellCount, spacing, adiabaticIndex);
fluid.setState(cell, density, velocity, pressure);
fluid.step(fluid.stableTimeStep(/*courantNumber=*/0.4));
```

## Go deeper

[src/Physics/README.md](../../src/Physics/README.md) has the cubic spline
kernel's exact form, the Rusanov flux the Eulerian solver uses, and a real
pitfall its own test walked into: a periodic domain split into a left half
and a right half actually creates *two* shock tubes, not one, since the
domain wraps around at the edges too.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/fluids)
and let us know.
