# Mechanics

The primitive matter of the engine, and the boundary where it crosses into
something an integrator can actually run.

## The idea

A `Body` is the simplest thing YSQ knows how to simulate: something with a
mass, maybe a charge, a position, and momentum. Every other theory in
`Physics` works on `Body`s or on collections of them.

`Body` stores **momentum, not velocity**, and that choice matters once
speeds get relativistic. Velocity is `distance / time`; momentum for a
massive particle is `p = gamma * m * v`, where `gamma` (the Lorentz factor)
grows without bound as speed approaches the speed of light. At low speed,
`gamma` is almost exactly 1 and the two are almost the same thing; at high
speed they diverge, and only momentum can be turned back into velocity
(`v = p / (gamma * m)`, solved for `gamma`). Velocity alone can't be turned
back into momentum once `gamma` matters, so momentum is the one that stays
correct as a scenario moves from everyday speeds into relativistic ones.

**Proper time** is what a moving clock actually reads. A stationary
observer sees a moving clock tick more slowly than their own (time
dilation); "proper time" is the time that clock, riding along with
whatever's moving, experiences for itself. It's the same idea for a body
moving through curved spacetime in `Physics/Spacetime`, generalized: proper
time is what makes "how long has this body existed, from its own point of
view" a well-defined question independent of who's watching.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Physics/Body.hpp` | The primitive: mass, charge, position, momentum |
| `Mechanics/Frame.hpp` | Inertial reference frames, Galilean transforms |
| `Mechanics/Kinematics.hpp` | Lorentz factor, four-velocity, proper time, relativistic velocity addition |
| `Mechanics/Dynamics.hpp` | `NBodyState`: the boundary between a span of `Body` and `Math`'s integrators |

That last one is worth understanding on its own. `Math`'s integrators only
need a plain vector space, values that can be added, subtracted, and scaled
(see [docs/math/integrators.md](../math/integrators.md)); a dimensioned
`Quantity` from `Units` deliberately isn't that. So the actual state an
integrator runs on, `NBodyState`, is a plain array of unitless positions and
velocities, and `Mechanics/Dynamics.hpp` is the one place a span of `Body`
crosses into that array and back. Every force law in `Gravity` reads and
writes `NBodyState`, never `Body`, in its inner loop; units cross the
boundary exactly once, not once per force evaluation.

## Using it

```cpp
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Math/Integrators/Symplectic.hpp>

std::vector<ysq::Body> bodies = /* masses, positions, momenta */;
ysq::NewtonianField field(bodies);

ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                        ysq::velocitiesOf(bodies)};
ysq::PhaseState<ysq::NBodyState> next;
stepper.step(field, 0.0, state, stepSize, next);

ysq::applyState(bodies, next.position, next.velocity);
```

`positionsOf`/`velocitiesOf` read a span of `Body` into the plain form the
stepper needs; `applyState` writes the result back. Every gravity model in
[docs/physics/gravity.md](gravity.md) plugs into a stepper exactly this way.

## Go deeper

[docs/api/physics/mechanics.md](../api/physics/mechanics.md) has every
signature: `Body`, `Frame`'s Galilean transform, the relativistic
`Kinematics` functions, and `NBodyState`.

[src/Physics/README.md](../../src/Physics/README.md) has the full
interface, including the exact `Body` layout and the concept boundary
(`NBodyState` satisfying `OdeState` while `Body` itself doesn't need to).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/mechanics)
and let us know.
