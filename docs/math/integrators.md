# Numerical integration

How YSQ advances a simulation from one instant to the next, and which of the
seven ways to do it you actually want.

## The idea

A lot of physics is stated as a rate of change rather than a formula for the
answer. Velocity is the rate of change of position; acceleration is the rate
of change of velocity. Newton's second law, `F = ma`, tells you the
acceleration a body has *right now*, given where everything is *right now*.
It does not hand you a formula for where the body will be in ten seconds.
That's an ordinary differential equation (ODE): an equation about rates, not
values.

For a single planet orbiting a fixed sun, calculus can solve that equation
exactly, by hand. For a dozen mutually-attracting bodies, or a photon curving
past a black hole, it can't: nobody has a formula for that. So instead of
solving the equation, you approximate it: take a small step forward in time,
use the rate of change you know *now* to guess the state a little later,
then repeat, thousands or millions of times. Each repeat is one call to a
**stepper**. The whole simulation loop (planets moving, orbits precessing,
light bending) is that loop, over and over, one small step at a time.

Two things distinguish one stepper from another:

- **Order.** Roughly, how fast the error shrinks as you shrink the step.
  A first-order method halves its error when you halve the step; a
  fourth-order method's error shrinks sixteen-fold. Higher order means more
  accuracy per step, usually at the cost of more work per step.
- **Structure.** Order describes accuracy over a short run. It says nothing
  about what happens over a long one. A method can be "accurate" and still
  be the wrong choice for a simulation you intend to watch orbit for a
  thousand years; see [Order isn't the whole story](#order-isnt-the-whole-story)
  below.

## What YSQ gives you

Every stepper is a small object, not a free function (see
[src/Math/README.md](../../src/Math/README.md#integrators) for why), with the
same shape: construct it, then call `step(system, time, state, stepSize,
out)` repeatedly, or hand it to a driver that does the repeating for you.

| Stepper | Order | Takes | Use it for |
| --- | --- | --- | --- |
| `ExplicitEulerStepper` | 1 | `dy/dt = f(t, y)` | Nothing. See below: it exists to be measured against. |
| `MidpointStepper`, `HeunStepper` | 2 | `dy/dt = f(t, y)` | Rarely reached for directly; stepping stones between Euler and RK4. |
| `Rk4Stepper` | 4 | `dy/dt = f(t, y)` | The default for a general system over a bounded number of steps. |
| `DormandPrince54Stepper` | 5 (4 embedded) | `dy/dt = f(t, y)` | When you want to specify accuracy, not step size: it picks its own step. |
| `SemiImplicitEulerStepper` | 1 | acceleration `a(t, q)` | Cheapest orbit-safe method; same cost as explicit Euler, none of the blow-up. |
| `VelocityVerletStepper` | 2 | acceleration `a(t, q)` | The workhorse for orbits and N-body gravity. Start here. |
| `ForestRuthStepper`, `PefrlStepper` | 4 | acceleration `a(t, q)` | Fourth-order accuracy without giving up the long-run stability Verlet has. `PefrlStepper` is the better of the two per unit of work. |

Two families, split by what they take. `dy/dt = f(t, y)` steppers (Euler,
RK4, Dormand-Prince) move any state that has a rate of change. Acceleration
steppers (semi-implicit Euler, Verlet, Forest-Ruth, PEFRL) are specialized
for `PhaseState<S>` (position and velocity together) and only need the
acceleration, not the full derivative. `asPhaseSystem(acceleration)` converts
between the two, so a `dy/dt` method can run on a position/velocity problem
too, which is what lets you put Euler and Verlet on the *same* problem and
watch the difference for yourself, below.

Two drivers run any stepper without you writing the loop:

- **`integrate(stepper, system, state, from, to, step)`**: fixed step size,
  optionally taking an observer callback fired after every step.
- **`integrateAdaptive(stepper, system, state, from, to, initialStep,
  settings)`**: for `DormandPrince54Stepper` only (it's the one method that
  produces its own error estimate). Grows or shrinks the step to hit a
  target accuracy, returning how many steps it took and whether it
  succeeded.

## Using it

A mass on a spring, `F = -x` (so acceleration `= -position`), is the
simplest oscillating system there is, and oscillation is exactly where the
difference between these methods stops being academic:

```cpp
#include <Math/Integrators/Euler.hpp>
#include <Math/Integrators/RK4.hpp>
#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>

auto acceleration = [](double /*time*/, double position) { return -position; };
const ysq::PhaseState<double> start{1.0, 0.0};  // displaced, at rest
constexpr double step = 0.1;
constexpr double duration = 200.0;  // roughly 30 periods

// Explicit Euler wants dy/dt = f(t, y), so wrap the acceleration first.
ysq::ExplicitEulerStepper<ysq::PhaseState<double>> euler;
const auto asFirstOrder = ysq::asPhaseSystem(acceleration);
const ysq::PhaseState<double> eulerEnd =
    ysq::integrate(euler, asFirstOrder, start, 0.0, duration, step);

// Semi-implicit Euler: same order, same cost, takes the acceleration directly.
ysq::SemiImplicitEulerStepper<double> symplecticEuler;
const ysq::PhaseState<double> symplecticEnd =
    ysq::integrate(symplecticEuler, acceleration, start, 0.0, duration, step);

// Velocity Verlet: same cost as symplectic Euler, one order more accurate.
ysq::VelocityVerletStepper<double> verlet;
const ysq::PhaseState<double> verletEnd =
    ysq::integrate(verlet, acceleration, start, 0.0, duration, step);

// Energy of this system is (velocity^2 + position^2) / 2, conserved exactly
// by the real physics, so it's a direct check on the integrator, not the model.
auto energy = [](const ysq::PhaseState<double>& s) {
    return (s.velocity * s.velocity + s.position * s.position) / 2.0;
};
```

Run that and compare `energy(eulerEnd)` against the starting energy of
`0.5`: explicit Euler's has grown, visibly, and would keep growing however
long you ran it. This is the method that "is here to be measured against,"
per the table above. `energy(symplecticEnd)` and `energy(verletEnd)` are
both close to `0.5`, and stay close no matter how long `duration` gets,
because both are *symplectic*: see
[Order isn't the whole story](#order-isnt-the-whole-story). `verletEnd` is
noticeably closer than `symplecticEnd` at the same step size and cost; that's
the one order of accuracy Verlet buys over semi-implicit Euler.

For a system whose acceleration only needs a handful of evaluations and
whose accuracy matters more than its step count, reach for
`DormandPrince54Stepper` through `integrateAdaptive` instead of picking a
step size by hand:

```cpp
#include <Math/Integrators/Adaptive.hpp>

ysq::DormandPrince54Stepper<ysq::PhaseState<double>> dp54;
ysq::AdaptiveSettings<double> settings;
settings.relativeTolerance = 1e-9;

const ysq::AdaptiveResult<ysq::PhaseState<double>, double> result =
    ysq::integrateAdaptive(dp54, asFirstOrder, start, 0.0, duration, 0.01, settings);
// result.succeeded, result.state, result.acceptedSteps, result.rejectedSteps
```

### Order isn't the whole story

Explicit Euler and semi-implicit Euler are both order 1: by the numbers,
equally accurate, equally cheap. Run both on the spring above and they
diverge completely: Euler's energy climbs without bound, semi-implicit's
oscillates in a narrow band forever. The difference isn't accuracy, it's
**structure**. A drift (moving position at fixed velocity) and a kick
(changing velocity at fixed position) are each an exactly solvable,
area-preserving step; a method built entirely out of drifts and kicks
(semi-implicit Euler, Verlet, Forest-Ruth, PEFRL) inherits that
area-preservation and is called *symplectic*. A symplectic method doesn't
conserve the energy of the system you actually gave it; it exactly conserves
the energy of a very slightly different one, and that difference is fixed by
the step size rather than growing with time. That's why the energy error
*oscillates* instead of *drifting*.

RK4 is not symplectic, and is far more accurate than any of these over a
handful of orbits regardless. Which matters, order or structure, depends
entirely on how long you intend to run: a quick trajectory calculation wants
RK4; a solar system you're going to watch for a thousand simulated years
wants Verlet or PEFRL. That tradeoff, worked through in more detail with the
actual numbers, is in [Go deeper](#go-deeper) below.

## Go deeper

[src/Math/README.md](../../src/Math/README.md) has the full interface
(every stepper's exact signature, `OdeState`/`OdeSystem`/`AccelerationField`
as concepts rather than base classes, why steppers are objects), the
"Which one to use" decision table with the reasoning behind each row, and a
**Derivations** section with the Butcher tableaux, the PI step controller
Dormand-Prince uses, the symplectic coefficients for Forest-Ruth and PEFRL
with their sources, and the numerical pitfalls (compensated summation,
`atan2` instead of `acos`, and more) that the rest of `Math` runs into and
had to solve once, carefully, rather than per call site.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+math/integrators)
and let us know.
