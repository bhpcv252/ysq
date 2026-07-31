# Gravity

Newton's law, sped up for many bodies, and corrected toward general
relativity where the correction is affordable.

## The idea

Newton's law of universal gravitation says every mass pulls every other
mass toward it, with a force that grows with both masses and falls off with
the square of the distance between them: `F = G*m1*m2/r^2`. That's enough
to hold a solar system together, spin a galaxy, and match everyday
intuition about weight, and it's cheap to compute. It's also not exactly
right: general relativity is the more complete theory, but solving it
exactly for many bodies pulling on each other is far too expensive to
simulate. YSQ's answer is a ladder: use Newtonian gravity for the dynamics,
and add relativistic corrections or switch to full general relativity only
where each is actually affordable.

Two problems come up as soon as you try to simulate Newtonian gravity for
real, though. First, **two bodies passing very close together** demand a
force that blows up toward infinity as the distance shrinks toward zero,
which no integrator with a fixed step size can follow accurately; the fix
is **softening**, smoothing the force law at very short range so it stays
finite instead of spiking. Second, **computing every pair of forces**
between N bodies costs `N^2` work, which is fine for a handful of planets
and far too slow for a galaxy of a million stars; the fix is
**Barnes-Hut**, which groups distant clusters of bodies into one averaged
"body" at their combined center of mass, trading a small, controllable
amount of accuracy for `N log N` instead of `N^2`.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Gravity/Newtonian.hpp` | The force law, softened, direct pairwise summation, potential energy |
| `Gravity/BarnesHut.hpp` | The same physics, `O(N log N)`, via an octree |
| `Gravity/PostNewtonian.hpp` | The 1PN correction: the first step toward general relativity |
| `Body::j2`, `Body::radius` | Oblateness: a body's own shape, read by every function above |

**Softening** adds one parameter, `epsilon`, to the force law:

```
a = GM * (r_j - r_i) / (|r_j - r_i|^2 + epsilon^2)^(3/2)
```

Well above `epsilon` this is indistinguishable from the exact `1/r^2` law;
as two bodies approach, it stays finite instead of diverging. `epsilon = 0`
recovers the exact law exactly.

**Barnes-Hut** walks a tree of nested cubes, each caching the total mass and
center of mass of everything inside it. A node of width `s` at distance `d`
from the point you're evaluating is accepted as a single point source when
`s/d < theta`; otherwise the walk recurses into that node's children.
`theta = 0` forces every node open, which is direct summation with extra
bookkeeping; `theta = 0.5` is the conventional default, trading a small,
measured amount of accuracy for a large speedup. Every leaf (a single body)
is always evaluated exactly, whatever `theta` is.

**The 1PN correction** adds the leading relativistic term to the
acceleration of a test particle orbiting one dominant mass, on top of the
Newtonian force rather than replacing it. It's what produces the extra
perihelion precession that Mercury's orbit famously shows and Newtonian
gravity alone can't explain.

**Oblateness (J2)** is not a fourth rung, it's the same Newtonian force law
carried one term further: a point mass is the zeroth term of integrating
gravity over a mass distribution, and J2 is the first correction for a body
shaped like an oblate spheroid rather than a sphere. Set `body.j2`,
`body.radius` (and `body.orientation` for which way the bulge points), and
every function above, `newtonianForce`, `newtonianAcceleration`,
`NewtonianField`, picks it up automatically; a body with `j2 == 0.0` (the
default) is unaffected. `Mechanics/RigidBody.hpp` is the rotational
counterpart: the torque that same asymmetry feels from an external mass,
and Euler's rotation equation to spin the body under it, useful for
anything from a satellite's precessing orbit to why Earth's own axis slowly
precesses. See [docs/physics/mechanics.md](mechanics.md).

## Using it

`NewtonianField` and `BarnesHutTree` both implement the same
`AccelerationField` concept `Math`'s steppers expect (see
[docs/math/integrators.md](../math/integrators.md)), so either one drops
directly into the same stepper without the rest of the simulation caring
which:

```cpp
#include <Physics/Gravity/BarnesHut.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Gravity/PostNewtonian.hpp>

// A handful of bodies: direct summation, exact.
ysq::NewtonianField exact(bodies);

// Many thousands of bodies: approximate, much faster.
ysq::BarnesHutTree tree(bodies, /*openingAngle=*/0.5);

// Either one plugs into the same stepper:
ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
stepper.step(exact, time, state, stepSize, next);

// A Mercury-like precessing orbit: Newtonian plus the relativistic correction,
// composed rather than one replacing the other.
const ysq::Acceleration3 newtonianAccel =
    ysq::newtonianAcceleration(testParticle.position, std::array{source});
const ysq::Acceleration3 correction =
    ysq::postNewtonianCorrection(testParticle, source);
const ysq::Acceleration3 total = newtonianAccel + correction;
```

An application composes the rungs it needs; nothing forces an exclusive
choice of "the" gravity model. See
[docs/physics/mechanics.md](mechanics.md) for how a span of `Body` gets
into `NBodyState` in the first place, and
[docs/tutorials/03-choosing-a-gravity-model.md](../tutorials/03-choosing-a-gravity-model.md)
for a worked example moving between rungs.

## Go deeper

[docs/api/physics/gravity.md](../api/physics/gravity.md) has every
signature: `newtonianForce`/`Acceleration`/`Accelerations`, `NewtonianField`,
`BarnesHutTree`, and `postNewtonianCorrection`.

[src/Physics/README.md](../../src/Physics/README.md) has the full
derivations: the exact 1PN acceleration formula and the perihelion
precession it predicts, the Barnes-Hut opening-angle error analysis, and
why `G` lives here rather than among `Units`' defining constants (it's
measured, not definitional).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/gravity)
and let us know.
