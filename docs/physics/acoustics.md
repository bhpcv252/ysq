# Acoustics

Sound is a wave equation you've already seen.

## The idea

A sound wave is a small pressure perturbation and a small particle
velocity, propagating through a medium (air, water, a solid) at the
medium's own sound speed. Linearize the fluid equations for a small
perturbation and you get a pair of coupled first-order equations for
pressure `p` and particle velocity `u`:

```
dp/dt = -rho0 c^2 du/dx
du/dt = -(1/rho0) dp/dx
```

`rho0` is the medium's rest density, `c` its sound speed. This is not a new
kind of physics for the engine to solve: it's **exactly**
`Physics/Electromagnetism`'s 1D vacuum Maxwell equations,
`dEy/dt = -c^2 dBz/dx`, `dBz/dt = -dEy/dx`, with `p` standing in for `Ey`,
`u` for `Bz`, and `rho0 c^2`/`1/rho0` standing in for the vacuum
equations' bare `c^2`/`1`. Same linear wave operator, different physical
label on each field. Everything that makes `MaxwellField1D` correct, the
Yee/leapfrog staggering, the CFL stability limit, the exact
zero-numerical-dispersion "magic" time step, carries over unchanged,
because those are properties of the discretized operator itself, not of
which fields happen to be involved.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Acoustics/Acoustic.hpp` | `AcousticField1D`: the 1D linear acoustic wave equation, Maxwell's own FDTD scheme relabeled |
| `Acoustics/Acoustic3D.hpp` | `AcousticField3D`: the same equations in 3D, on a staggered (MAC) grid |

Unlike vacuum electromagnetism's fixed `c`, sound speed and medium density
vary by material, so `mediumDensity`/`soundSpeed` are constructor
parameters here rather than a single physical constant to reach for.
`AcousticField3D` has no cross-axis coupling the way `Maxwell3D`'s curl
terms do: divergence and gradient are separable per component, so each
velocity component only ever talks to pressure along its own axis. As with
every other 3D solver in this engine, there is **no exact "magic
timestep"** in 3D; `stableTimeStep` gives the CFL limit only.

## Using it

```cpp
#include <Physics/Acoustics/Acoustic.hpp>

ysq::AcousticField1D field(cellCount, spacing, mediumDensity, soundSpeed);
field.setPressure(sourceCell, initialPulse);

field.step(ysq::magicAcousticTimeStep(spacing, soundSpeed));  // exact propagation, no dispersion
const double energy = field.totalEnergy();  // conserved, checked over many steps
```

The same physics in 3D, stepped at the CFL limit:

```cpp
#include <Physics/Acoustics/Acoustic3D.hpp>

ysq::AcousticField3D field(nx, ny, nz, spacing, mediumDensity, soundSpeed);
field.setPressure(i, j, k, initialPulse);

field.step(field.stableTimeStep(/*courantFactor=*/0.9));
const double energy = field.totalEnergy();  // bounded, not exactly conserved off the magic step
```

## Go deeper

[docs/api/physics/acoustics.md](../api/physics/acoustics.md) has every
signature: both `AcousticField1D`'s and `AcousticField3D`'s full
interfaces, and `magicAcousticTimeStep`.

[src/Physics/README.md](../../src/Physics/README.md) has the full
correspondence with `Maxwell.hpp`'s own equations, the exact energy-density
formula both dimensions use, and how `AcousticField3D` is validated: an
exact reduction to `AcousticField1D` when the field is held uniform along
two axes, energy bounded over many steps, and a diagonal plane wave
checked against the discrete (not continuum) dispersion relation for that
direction.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/acoustics)
and let us know.
