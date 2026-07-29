# Math API reference: ODE integrators

The stepper interface, state types, and drivers in `Math/ODE.hpp` and
`Math/Integrators/`. Start with
[docs/math/integrators.md](../../math/integrators.md) for which one to reach
for; [src/Math/README.md](../../../src/Math/README.md#integrators) has the
Butcher tableaux, the PI controller derivation, and the symplectic
coefficients with their sources.

Every stepper is a small object (not a free function: a Dormand-Prince
stepper carries state between calls, and a heap-allocated state needs
scratch that must not reallocate every step), exposing `State`, `Scalar`,
`order`, and `step(system, time, state, h, out)`. `evaluations()` tracks the
cumulative call count on `system`/`acceleration`.

## `Math/ODE.hpp`

The interface and the two drivers.

```cpp
template <class S> using StateScalarT = /* recurses to the innermost floating-point type */;

template <class S>
concept OdeState = /* a vector space over its scalar: + - * and in-place forms */;

template <class F, class S>
concept OdeSystem = OdeState<S> && requires(F f, Scalar t, S y) { { f(t, y) } -> S; };

template <class F, class S>
concept AccelerationField = OdeSystem<F, S>;  // structurally identical, deliberately distinct name
```

`OdeSystem` and `AccelerationField` cannot tell each other apart at the type
level (nothing in the signature distinguishes `dy/dt = f(t, y)` from
`a = a(t, q)`), so handing an acceleration to an explicit method compiles
and integrates the wrong problem. What they do catch is a callable of the
wrong shape entirely, reported at the call site.

```cpp
template <OdeState S> struct PhaseState {
    S position{}, velocity{};
    // += -= *= /=, + - * /, ==; itself an OdeState
};

template <class Acceleration>
constexpr auto asPhaseSystem(Acceleration acceleration);
// wraps a(t, q) into d(q,v)/dt = (v, a(t,q)), so an explicit method (RK4, Euler)
// can integrate the same problem a symplectic one does, for direct comparison
```

```cpp
template <std::floating_point T> class StateVector {
public:
    StateVector();
    explicit StateVector(std::size_t count, T fill = T{});
    StateVector(std::initializer_list<T> values);

    std::size_t size() const noexcept;
    void resize(std::size_t count);
    T& operator[](std::size_t index) noexcept;   // asserted
    // begin/end, += -= *= /=, + - * /, ==
};
```

A run-time-sized state, for an N-body problem or anything else sized by its
input. Mismatched-size arithmetic is asserted, not checked, in release.

Drivers:

```cpp
template <class Stepper, class System>
auto integrate(Stepper&, const System&, State state, Scalar from, Scalar to, Scalar step)
    -> State;
// with an optional Observer callback: integrate(..., step, observer)
// observer(time, state) fires after every step, including the initial one

template <std::floating_point T> std::size_t stepCount(T from, T to, T step);
// how many fixed steps integrate() will actually take

template <class Stepper, class System>
auto integrateAdaptive(Stepper&, const System&, State state, Scalar from, Scalar to,
                        Scalar initialStep, const AdaptiveSettings<Scalar>& settings)
    -> AdaptiveResult<State, Scalar>;
// with an optional Observer callback, same shape as integrate()
```

| Type/Member | Description |
| --- | --- |
| `integrate` | Fixed step. Rounds the step **down** to the nearest divisor of the interval so the run lands exactly on `to` with uniform steps: asking for 0.3 over a unit interval gets four steps of 0.25. |
| `AdaptiveSettings<T>` | `absoluteTolerance`/`relativeTolerance` (default `sqrt(epsilon)`, scaled to `T`'s precision), `minimumStep`/`maximumStep`, `safety` (default 0.9), `minimumScale`/`maximumScale` (controller clamp), `maximumRejections` (default 20), `maximumSteps` (default 1e7). |
| `AdaptiveResult<S, T>` | `state`, `time`, `acceptedSteps`, `rejectedSteps`, `evaluations`, `succeeded` (false if rejections or the step floor were hit before reaching `to`). |
| `integrateAdaptive` | For `DormandPrince54Stepper` only: the one method with its own error estimate. `initialStep` is a starting guess; the PI controller replaces it after one step. Runs in either direction (`from > to` is valid). |
| `errorNorm(error, current, next, absTol, relTol)` | Root-mean-square of the per-component error, each weighed against its own tolerance: what the adaptive controller checks against 1. |

```cpp
ysq::Rk4Stepper<ysq::Vec3> stepper;
const ysq::Vec3 end = ysq::integrate(stepper, system, start, 0.0, 10.0, 0.01);

ysq::DormandPrince54Stepper<ysq::Vec3> dp54;
ysq::AdaptiveSettings<double> settings;
settings.relativeTolerance = 1e-9;
const auto result = ysq::integrateAdaptive(dp54, system, start, 0.0, 10.0, 0.01, settings);
// result.succeeded, result.state, result.acceptedSteps, result.rejectedSteps
```

## `Math/Integrators/Euler.hpp`

```cpp
template <OdeState S> class ExplicitEulerStepper {
public: using State = S; using Scalar = StateScalarT<S>;
    static constexpr int order = 1;
    template <OdeSystem<State> System> void step(system, time, state, h, out);
};

template <OdeState S> class MidpointStepper   { /* order 2, dy/dt = f(t,y) */ };
template <OdeState S> class HeunStepper       { /* order 2, dy/dt = f(t,y) */ };

template <OdeState S> class SemiImplicitEulerStepper {
public: using State = PhaseState<S>; using Scalar = StateScalarT<S>;
    static constexpr int order = 1;
    template <AccelerationField<S> Acceleration> void step(acceleration, time, state, h, out);
};
```

| Stepper | Takes | Notes |
| --- | --- | --- |
| `ExplicitEulerStepper` | `dy/dt = f(t, y)` | Exists to be measured against, not used: unstable on anything oscillatory (a harmonic oscillator's energy grows without bound at any step size). |
| `MidpointStepper` | `dy/dt = f(t, y)` | One Euler probe to the half step, then the derivative there for the whole step. |
| `HeunStepper` | `dy/dt = f(t, y)` | An Euler probe to the far end, then the average of the two derivatives. Same order and cost as `MidpointStepper`, different error constant and stability region. |
| `SemiImplicitEulerStepper` | acceleration `a(t, q)` | One line different from explicit Euler (velocity updates first, position moves with the *new* velocity) and a completely different method: symplectic, so energy oscillates in a band instead of growing. Same order (1) and cost as explicit Euler. |

## `Math/Integrators/RK4.hpp`

```cpp
template <OdeState S> class Rk4Stepper {
public: using State = S; using Scalar = StateScalarT<S>;
    static constexpr int order = 4;
    template <OdeSystem<State> System> void step(system, time, state, h, out);
};
```

Classical fourth-order Runge-Kutta, four evaluations per step. The default
for a general system over a bounded number of steps: not symplectic, so its
energy error drifts slowly over very long runs, but far more accurate than
any second-order symplectic method over a handful of orbits.

## `Math/Integrators/Adaptive.hpp`

```cpp
template <OdeState S> class DormandPrince54Stepper {
public:
    using State = S; using Scalar = StateScalarT<S>;
    static constexpr int order = 5;            // propagated solution's order
    static constexpr int embeddedOrder = 4;    // the estimate's order

    template <OdeSystem<State> System>
    void step(system, time, state, h, out, error&);   // fills error, for the adaptive driver

    template <OdeSystem<State> System>
    void step(system, time, state, h, out);            // discards the estimate, for the fixed-step driver

    void reset();   // drops the carried (FSAL) derivative
};
```

Seven-stage embedded pair producing a 5th-order solution and a 4th-order
one from the same stages, so their difference estimates the error at
(almost) no extra cost. **FSAL** (First Same As Last): the seventh stage is
evaluated at the new state with exactly the propagated solution's weights,
so it becomes the first stage of the next step, and an accepted step costs
six evaluations instead of seven. The cache is keyed on the time it belongs
to, so a rejected step re-evaluates rather than reusing a stale derivative.
Use `integrateAdaptive` from `Math/ODE.hpp`, not this stepper's `step`
directly, unless you're writing your own driver.

## `Math/Integrators/Symplectic.hpp`

```cpp
template <OdeState S> class VelocityVerletStepper {
public: using State = PhaseState<S>; using Scalar = StateScalarT<S>;
    static constexpr int order = 2;
    template <AccelerationField<S> Acceleration> void step(acceleration, time, state, h, out);
};

template <OdeState S> class ForestRuthStepper { /* order 4, three evaluations */ };
template <OdeState S> class PefrlStepper      { /* order 4, four evaluations */ };
```

All three take an acceleration `a(t, q)` and advance a `PhaseState`, built
from drifts (move position at fixed velocity) and kicks (change velocity at
fixed position), each exactly solvable and area-preserving, so any
composition of them is too.

| Stepper | Order | Evaluations/step | Notes |
| --- | --- | --- | --- |
| `VelocityVerletStepper` | 2 | 2 | The workhorse: kick-drift-kick, time-reversible. Start here for orbits and N-body gravity. |
| `ForestRuthStepper` | 4 | 3 | Yoshida composition of three Verlet-like steps with a negative middle coefficient (the middle step runs backward in time, unavoidable for 4th order from forward-only steps). |
| `PefrlStepper` | 4 | 4 | One more evaluation than Forest-Ruth, considerably smaller error constant: the better 4th-order choice per unit of work. |

```cpp
ysq::VelocityVerletStepper<ysq::Vec3> verlet;
const auto result = ysq::integrate(
    verlet, acceleration, ysq::PhaseState<ysq::Vec3>{position, velocity},
    0.0, 10.0, 0.01,
    [&](double t, const auto& state) { record(t, state); });
```

**What symplectic buys, and what it doesn't.** A symplectic method doesn't
conserve the energy of the system it was given; it exactly conserves the
energy of a nearby one, and the gap is fixed by the step size rather than
growing with elapsed time, so the energy error oscillates in a band
instead of drifting. It says nothing about trajectory error, which grows
with time regardless. Right choice for a long run whose invariants matter;
wrong choice for a short run that has to land in exactly the right place.
The guarantee is strictly for an autonomous system; time is threaded through
the substages so an explicitly time-dependent force still integrates, but
the bounded-energy-error promise doesn't extend to a driven system.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/math/integrators)
and let us know.
