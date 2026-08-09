# Electromagnetism

Coulomb's law, Biot-Savart, and the one step up from both: a field that
actually propagates.

## The idea

Every charge creates an electric field around it, pulling opposite charges
together and pushing like charges apart; that's Coulomb's law, and it falls
off with the square of the distance, the same shape as gravity. A *moving*
charge additionally creates a magnetic field (Biot-Savart), which is what
makes a current-carrying wire magnetic. The **Lorentz force** is how a field
pushes back on a charge sitting in it: `F = q(E + v x B)`, electric force
plus the sideways push a magnetic field gives anything moving through it.

YSQ implements this as a ladder too. The first rung (`Field.hpp`) assumes
each charge's field reaches everywhere *instantly*, using each source's
present position and velocity with no delay, which is an excellent
approximation whenever nothing is changing very fast. The second rung
(`Maxwell.hpp`/`Maxwell3D.hpp`) drops that assumption: it solves Maxwell's
equations directly, so a field genuinely propagates at the speed of light
rather than updating everywhere at once, which is what you actually need
once radiation or wave propagation matters, not just the pull between
charges. (Neither rung takes a moving charge's own trajectory and produces
the field it radiates with the correct retardation baked in — that's a
further, still-unbuilt piece; see `src/Physics/README.md`.)

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Electromagnetism/Field.hpp` | `electricField`/`magneticField`: instantaneous superposition over a span of `Body` |
| `Electromagnetism/Lorentz.hpp` | `lorentzForce`: `F = q(E + v x B)` |
| `Electromagnetism/Maxwell.hpp` | `MaxwellField1D`: a field that actually propagates, one spatial dimension |
| `Electromagnetism/Maxwell3D.hpp` | `MaxwellField3D`: the same scheme on the full 3D Yee grid, six field components |

`MaxwellField1D` is a leapfrog scheme (the same structure as the symplectic
integrators in [docs/math/integrators.md](../math/integrators.md), applied
to a field instead of a particle), staggered in both space and time. At its
"magic" time step, `magicTimeStep(spacing)`, this particular scheme has
*zero* numerical dispersion in 1D vacuum: a wave keeps its exact shape as it
propagates, not just approximately. `MaxwellField3D` is the direct 3D
generalization — the same staggering idea, two curl terms per update
instead of one — but has no such magic step: a Cartesian grid's numerical
dispersion becomes direction-dependent once there's more than one
dimension to propagate through, a textbook FDTD fact rather than a gap.

## Using it

```cpp
#include <Physics/Electromagnetism/Field.hpp>
#include <Physics/Electromagnetism/Lorentz.hpp>

const ysq::ElectricField3 e = ysq::electricField(queryPoint, sources);
const ysq::MagneticFluxDensity3 b = ysq::magneticField(queryPoint, sources);
const ysq::Force3 force = ysq::lorentzForce(chargedBody, e, b);
```

A field that genuinely propagates, one spatial dimension:

```cpp
#include <Physics/Electromagnetism/Maxwell.hpp>

ysq::MaxwellField1D field(cellCount, spacing);
field.setElectricField(sourceCell, initialPulse);

field.step(ysq::magicTimeStep(spacing));  // exact propagation, no dispersion
const double energy = field.totalEnergy();  // conserved, checked over many steps
```

The same field, in 3D, stepped at the CFL limit rather than a magic step:

```cpp
#include <Physics/Electromagnetism/Maxwell3D.hpp>

ysq::MaxwellField3D field(nx, ny, nz, spacing);
field.setElectricFieldY(i, j, k, initialValue);

field.step(field.stableTimeStep(/*courantFactor=*/0.9));
const double energy = field.totalEnergy();  // bounded, not exactly conserved off the magic step
```

## Go deeper

[docs/api/physics/electromagnetism.md](../api/physics/electromagnetism.md)
has every signature: `electricField`/`magneticField`, `lorentzForce`, and
both `MaxwellField1D`'s and `MaxwellField3D`'s full interfaces.

[src/Physics/README.md](../../src/Physics/README.md) has the full field
formulas (Coulomb's constant and vacuum permeability, and why both are
computed rather than typed independently), the Yee staggering both Maxwell
solvers use, and how `MaxwellField3D` is validated: energy bounded over
many steps, a plane wave's measured propagation speed against the
continuum prediction, and an exact reduction to `MaxwellField1D` when the
field is held uniform along two axes.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/electromagnetism)
and let us know.
