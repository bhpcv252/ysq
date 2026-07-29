# Tutorial 1: your first simulation

A planet orbiting a star, printed to the console. No window, no rendering,
just the physics loop every YSQ simulation is built from underneath.

This draws on [docs/units.md](../units.md), [docs/physics/mechanics.md](../physics/mechanics.md),
[docs/physics/gravity.md](../physics/gravity.md), and
[docs/math/integrators.md](../math/integrators.md); skim those first if
anything below is unfamiliar.

## Set up the bodies

A `Body` is a mass, a charge, a position, and a momentum (see
[docs/physics/mechanics.md](../physics/mechanics.md) for why momentum
rather than velocity). A circular orbit's speed comes from balancing
gravity against the centripetal acceleration it needs: `v = sqrt(GM/r)`.

```cpp
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Units/Mass.hpp>
#include <Units/Length.hpp>

std::vector<ysq::Body> bodies(2);

ysq::Body& star = bodies[0];
star.mass = ysq::units::solarMass;
// star stays at the origin, at rest

ysq::Body& planet = bodies[1];
planet.mass = ysq::units::earthMass;

const ysq::Length orbitalRadius = ysq::units::astronomicalUnit;
planet.position = orbitalRadius * ysq::Vec3{1.0, 0.0, 0.0};

const ysq::Speed orbitalSpeed =
    ysq::sqrt(ysq::constants::G * star.mass / orbitalRadius);
planet.momentum = planet.mass * (orbitalSpeed * ysq::Vec3{0.0, 1.0, 0.0});
```

`orbitalSpeed * ysq::Vec3{0.0, 1.0, 0.0}` is the "magnitude times direction"
pattern from [docs/units.md](../units.md): a scalar `Speed` and a plain,
dimensionless direction combine into a `Velocity3`. `planet.mass * velocity`
is a `Mass` times a `Velocity3`, which is exactly a `Momentum3`, by
construction, the same way every other product of `Quantity` types is.

## Cross into the integrator

`Math`'s steppers only know about plain vector spaces (see
[docs/math/integrators.md](../math/integrators.md)); `NBodyState` is the
unitless array `Mechanics/Dynamics.hpp` converts a span of `Body` into and
back, crossing the units boundary exactly once:

```cpp
#include <Math/Integrators/Symplectic.hpp>
#include <Physics/Mechanics/Dynamics.hpp>

ysq::NewtonianField gravity(bodies);
ysq::VelocityVerletStepper<ysq::NBodyState> stepper;

ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                        ysq::velocitiesOf(bodies)};
```

`VelocityVerletStepper` is the workhorse for orbits: see
[Order isn't the whole story](../math/integrators.md#order-isnt-the-whole-story)
for why it, not the more accurate-looking RK4, is the right default for
something you're going to watch orbit for a long time.

## Run it, and watch energy

```cpp
#include <Math/ODE.hpp>

#include <iostream>

constexpr double julianYear = 365.25 * 86400.0;
constexpr double step = 3600.0;  // one hour, in seconds

const auto energy = [&](const std::vector<ysq::Body>& current) {
    ysq::Energy kinetic{};
    for (const ysq::Body& body : current) {
        kinetic += 0.5 * body.mass * ysq::dot(body.velocity(), body.velocity());
    }
    return kinetic + ysq::newtonianPotentialEnergy(current);
};

ysq::PhaseState<ysq::NBodyState> next = state;
for (int year = 0; year < 5; ++year) {
    state = ysq::integrate(stepper, gravity, state, 0.0, julianYear, step);
    ysq::applyState(bodies, state.position, state.velocity);
    std::cout << "year " << year << ": energy = " << energy(bodies).value() << "\n";
}
```

Run this and the printed energy should barely move from year to year,
oscillating in a tiny band rather than drifting: that's Verlet's symplectic
guarantee actually holding, not a coincidence, and it's the same check
[Tutorial 3](03-choosing-a-gravity-model.md) leans on when you switch to
Barnes-Hut for many bodies.

## Next

[Tutorial 2](02-adding-visualization.md) takes this exact simulation and
adds a window, a camera, and a live energy chart, so you can watch the
orbit instead of reading numbers.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+tutorials/01-your-first-simulation)
and let us know.
