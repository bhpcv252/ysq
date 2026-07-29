# Tutorial 4: spacetime and light

Every earlier tutorial stayed on the gravity ladder's Newtonian rungs. This
one leaves it entirely: build a Schwarzschild spacetime around a black
hole, launch a light ray past it, and measure how much it bends, compared
against the textbook weak-field formula. Read
[docs/physics/spacetime.md](../physics/spacetime.md) and
[docs/physics/optics.md](../physics/optics.md) alongside this.

## Build the spacetime

A `Schwarzschild` metric needs one number: the gravitational parameter
(`GM`) of whatever it's describing. Nothing else about the metric is
configurable, because a non-rotating mass's spacetime is fully determined
by that one quantity.

```cpp
#include <Physics/Spacetime/Schwarzschild.hpp>

const ysq::Schwarzschild blackHole{someGravitationalParameter};
const double rs = blackHole.schwarzschildRadius();
```

## Launch a ray at a chosen impact parameter

The **impact parameter** is the perpendicular distance the ray would pass
the black hole at if spacetime were flat, the everyday notion of "how close
does this beam of light aim to pass." `schwarzschildRayFromImpactParameter`
builds the exact initial state for a ray launched inward from some starting
radius with a given impact parameter; "exact" here means exact for any
starting radius, not just one far enough away that an approximation holds.

```cpp
#include <Physics/Optics/Lensing.hpp>

const double startRadius = 100.0 * rs;   // far from the black hole
const double impactParameter = 20.0 * rs;  // how close the ray aims to pass

const ysq::PhaseState<ysq::Vec4> ray =
    ysq::schwarzschildRayFromImpactParameter(blackHole, impactParameter, startRadius);
```

## Measure the deflection

`deflectionAngle` propagates the ray, watches for it crossing back out
through `startRadius`, and returns how much its total sweep exceeded the
correct flat-space value (not a bare `pi`; see
[docs/physics/optics.md](../physics/optics.md) for why that distinction
matters):

```cpp
const double measured =
    ysq::deflectionAngle(blackHole, ray, impactParameter, startRadius,
                          /*affineStep=*/0.01 * rs, /*maxSteps=*/1'000'000);

const double weakFieldEstimate = ysq::weakFieldDeflectionAngle(rs, impactParameter);
```

With `impactParameter` well above `rs`, as it is here, `measured` and
`weakFieldEstimate` should agree closely: that's the weak-field deflection
formula, `4GM/(c^2 b)`, being recovered from the full geodesic calculation
in the regime where both are supposed to agree. Shrink `impactParameter`
toward `rs` and watch them diverge: that divergence is the actual point of
having a real geodesic solver rather than only the weak-field formula, the
same relationship the gravity ladder has between Newtonian gravity and its
relativistic corrections (see
[Tutorial 3](03-choosing-a-gravity-model.md)).

## Next

You've now used every rung of the ladder from
[docs/physics/index.md](../physics/index.md): Newtonian gravity, Barnes-Hut,
the 1PN correction, and a full geodesic through curved spacetime. From
here, [docs/applications.md](../applications.md) covers turning any of
these into a real windowed `Application` of your own, the same shape
[Tutorial 2](02-adding-visualization.md) built by hand.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+tutorials/04-spacetime-and-light)
and let us know.
