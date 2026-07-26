# Math

Vectors, matrices, quaternions, complex and dual numbers, tensors, statistics,
interpolation, calculus, and the ODE integrators. Everything the engine
computes with, and nothing that knows what it is computing about.

**Target:** `ysq::Math` (INTERFACE, header-only)
**Depends on:** nothing. Not even `Core`: Math is usable without a logger.

## Contents

| Header                          | Purpose                                                     |
| ------------------------------- | ----------------------------------------------------------- |
| `Math/Scalar.hpp`               | The `Numeric` concept, constants, tolerances, `approxEqual`  |
| `Math/Vector2.hpp` `3` `4`      | Fixed-size vectors                                           |
| `Math/Matrix2.hpp` `3` `4`      | Fixed-size matrices, and a pivoting linear solve             |
| `Math/Quaternion.hpp`           | Rotations, and both conversions with `Matrix3`               |
| `Math/Complex.hpp`              | Complex numbers                                              |
| `Math/Dual.hpp`                 | Dual numbers: forward-mode automatic differentiation         |
| `Math/Tensor.hpp`               | Fixed rank and dimension, with the index algebra             |
| `Math/Statistics.hpp`           | Summaries, compensated summation, an online accumulator      |
| `Math/Interpolation.hpp`        | Lerp through natural cubic splines                           |
| `Math/Calculus.hpp`             | Differentiation and quadrature                               |
| `Math/CoordinateSystems.hpp`    | Spherical, cylindrical and polar, with their local bases     |
| `Math/ODE.hpp`                  | The integrator interface, state types, and the drivers       |
| `Math/Integrators/Euler.hpp`    | Explicit, semi-implicit, midpoint, Heun                      |
| `Math/Integrators/RK4.hpp`      | Classical fourth-order Runge-Kutta                           |
| `Math/Integrators/Adaptive.hpp` | Dormand-Prince 5(4) with a PI step controller                |
| `Math/Integrators/Symplectic.hpp` | Velocity Verlet, Forest-Ruth, PEFRL                        |
| `Math/Format.hpp`               | `std::formatter` for the value types above                   |

Derivations, coefficient tables and their sources are in
[docs/math.md](../../docs/math.md).

## Everything is templated on its scalar

`Vector3<T>`, `Matrix4<T>`, `Tensor<T, Rank, Dim>`, with `Vec3`, `Mat4` and so
on as the `double` aliases and `Vec3f`, `Mat4f` for `float`.

This is not about supporting both precisions, though it does. It is what makes
`Vector3<Dual<double>>` an instantiation rather than a second implementation,
and that composition is how `Physics/Spacetime` will get exact metric
derivatives, and therefore exact Christoffel symbols, instead of finite
differences. `Dual` satisfying the `Numeric` concept is asserted in `Dual.hpp`
itself, and `tests/smoke/math_strict_warnings.cpp` instantiates
`Vector3<Dual<double>>`, `Matrix3<Dual<double>>` and `Complex<Dual<double>>` so
the composition cannot quietly stop compiling.

`Numeric` is deliberately weaker than `std::floating_point`: a field with an
ordering and a square root. `Complex` does not satisfy it, on purpose, and so
cannot go inside `Vector` or `Matrix`. Those types compare magnitudes for
pivoting, for componentwise min and max, and for zero-length checks, and
complex numbers have no ordering. A complex vector space also wants a Hermitian
inner product rather than the bilinear one they use, so that would be a design
decision rather than a free instantiation.

## Conventions

These are the ones that produce a plausible wrong answer rather than an error
when they are got wrong, so each is stated in the header, in the member names,
and in a test.

**Matrices are column-major with column vectors.** `columns[j]` is the j-th
column, transforms compose right to left (`M = T * R * S`) and apply as
`v' = M * v`. That is GLSL's convention, so a `Matrix4` uploads to a uniform
with `transpose = GL_FALSE` and the shader reads the same as the C++.
`operator[]` indexes a column; `operator()(row, col)` indexes an element in
reading order. `math_matrix.cpp` memcpys a `Mat4` and asserts the sixteen
doubles come out in column order, which is the only test that would catch a
transposed layout.

**Spherical coordinates use the physics convention.** `polar` is the angle down
from +z in [0, π]; `azimuth` is the angle round from +x in (-π, π]. Mathematics
texts routinely swap the two names.

**Quaternions are stored scalar part first**, `(w, x, y, z)`. The other common
layout puts w last, the two are indistinguishable at the type level, and mixing
them silently produces a wrong rotation.

**Euler angles carry their order in the name**: `fromEulerZYX`, `toEulerZYX`.
There are twelve conventions and picking the wrong one produces a
plausible-looking rotation that is simply not the one asked for.

**Tensors are row-major**, so the last index varies fastest.

## Failure is reported, not invented

Following `Core`: `tryX` returns `std::optional`, plain `X` is the unchecked
form.

- `tryInverse` is nullopt for a singular matrix; `inverse` yields NaN.
- `tryNormalized` is nullopt for a zero or non-finite vector; `normalized`
  yields NaN.
- `CubicSpline::natural` is nullopt unless its knots are strictly increasing.
- Statistics on insufficient data return NaN, not zero. A variance of "no
  samples" is not zero, and a NaN propagates into whatever it feeds instead of
  reading as a good result.
- A NaN in a dataset comes back as a NaN from every statistic, including
  `minimum` and `median`. That costs a scan, and it buys two things: `median`
  cannot sort past a NaN without undefined behaviour, and `minimum` would
  otherwise skip it and report a smallest value that is not the smallest
  anything. `histogram` drops NaN instead of binning it, since a count cannot
  be NaN and a silently binned one would make the totals lie.

The unchecked forms propagate NaN rather than returning something plausible,
which is the cheaper failure to trace.

## Bounds are asserted, not checked

Every indexed accessor asserts its argument: `operator[]` on the vectors,
quaternion, complex, dual and coordinate types, `operator[]` and
`operator()` on the matrices, both on `Tensor`, and `operator[]` plus the
size match on `StateVector` arithmetic. `assert` compiles out under `NDEBUG`,
so a release build pays nothing and behaves exactly as before.

The two hazards behind that one policy are different. The ternary-based
accessors were already safe, returning the last component for an out-of-range
index; they were merely silent. The array-backed ones were undefined. An
assertion turns both into a message naming the file and line of the mistake,
which is the only form of either that is any use.

A checked accessor returning `std::optional` was the alternative and is the
wrong trade here: these sit inside integrator inner loops, the index is almost
always a loop variable that is correct by construction, and the cost would be
paid on every element of every operation to catch a class of bug that a debug
run finds immediately.

## The representable range, and where it stops

`length` is `sqrt(dot(v, v))`, so a component beyond about **1.3e154** at
double precision overflows the square and a component below about **1.5e-162**
underflows it, in both cases before the square root can recover. The direction
is perfectly representable in either case; the intermediate is not.

`tryNormalized` reports that rather than dividing through, which used to give a
zero vector inside a successful result. `tryInverse` does the same for a
determinant that overflowed, and `solve` for a pivot that did. All four used to
return a success holding zeros, which is the one thing this module is not
supposed to do.

`Complex::abs` has no such limit, because a complex modulus goes through
`hypot`. Doing the same for `length` would widen the range to the full exponent
at the cost of a `hypot` on every normalisation on the integration inner path,
to cover magnitudes that no physical quantity approaches: the largest distances
in the project are around 1e21 metres and the largest masses around 1e41
kilograms, both squaring comfortably inside the range. The asymmetry is
deliberate, and `math_vector.cpp` pins where each one gives out.

## A stepper may write over its own input

`stepper.step(system, t, y, h, y)` is valid and gives bit-identical results to
stepping into a separate object, for every stepper here. Each either finishes
reading the input before writing the output or copies first. `math_ode.cpp`
checks all seven, because it is an easy property to lose and nothing else would
notice.

## Non-finite values propagate, and never hang

A NaN or an infinity is a value the module carries through rather than a state
it special-cases: `mean`, `median`, a table lookup, an integrated state and a
quadrature all return NaN when their input holds one, and none of them invents
a finite answer instead.

The two places that took work are the ones where propagation is not the
default behaviour of the obvious code:

- **A comparison against a NaN is false**, so a convergence or error test
  written as "stop when the error is small" never stops. `adaptiveSimpson`
  checks its estimate is finite before recursing; without it, a non-finite
  integrand drives every branch to full depth, which at the default is 2^40
  calls. The adaptive ODE driver reaches its rejection limit for the same
  reason, which is the behaviour wanted there.
- **Converting a NaN to an integer is undefined**, so anything that turns a
  value into a count or an index has to reject it first. `histogram` tests its
  range positively rather than by negation, and the fixed-step driver's step
  count cannot be reached with a non-finite step.

## Accuracy, where it was a choice

Several functions are written the long way because the short way loses most of
its significant digits somewhere that matters. Each has a test that fails if
someone simplifies it back.

- **Angles come from `atan2`, never from `acos`.** `angleBetween` on vectors,
  `toAxisAngle` on quaternions, and the polar angle in `toSpherical` all use
  `atan2` of a perpendicular component against a parallel one. For a small
  angle the `acos` argument has already rounded to exactly 1 and `acos` returns
  exactly 0. `math_vector.cpp` resolves angles down to 1e-8 and asserts the
  `acos` route gives literally zero there.
- **`angleBetween` for quaternions forms the relative rotation** rather than
  recovering a half-angle sine from the dot product, which would bottom out
  around 3e-8. It resolves 1e-13.
- **`toEulerZYX` handles gimbal lock.** Within 2⁻⁴⁰ of a pole the general
  expressions become `atan2(0, 0)`; there the function pins roll to zero and
  puts the whole determined quantity into yaw, and takes pitch as exactly a
  right angle rather than through `asin`, whose endpoint error is `sqrt(eps)`.
  The threshold is far tighter than is comfortable on purpose: the general
  branch degrades gracefully and this one is only exactly right at the pole.
- **`Quaternion::fromRotationMatrix` uses Shepperd's method**, branching on the
  largest component. The trace-only derivation divides by something that goes
  to zero at a half turn.
- **Complex modulus uses `hypot` and division uses Smith's formula**, so
  neither overflows for an operand whose answer is perfectly representable.
- **`Statistics::sum` is Neumaier-compensated**, and `RunningStatistics` is
  Welford. Both matter for an energy accumulator over a long run; there are
  tests showing what the naive forms lose.

## Exact derivatives, and approximate ones

`derivative`, `gradient`, `jacobian` and `hessian` are the dual-number
versions. They are exact to the last few bits, need no step size, and cost one
evaluation per component. `numericalGradient`, `centralDifference`,
`richardsonDerivative` and the rest are finite differences and lose roughly
half the available digits however carefully the step is chosen.

The plain names go to the better method, so reaching for the worse one has to
be deliberate. The finite-difference forms are still worth having: they need
only a function of plain doubles, which is what a measured dataset or a
black-box callback gives you.

```cpp
ysq::derivative([](auto x) { return exp(sin(x)); }, 1.3);   // exact
ysq::gradient([](const auto& v) { return dot(v, v); }, at); // 2 * at, exactly
ysq::hessian(potential, at);                                // nested duals
```

## Integrators

Steppers are objects, not free functions, for two independent reasons.
Dormand-Prince is FSAL, so it carries its last stage derivative into the next
step and costs six evaluations per accepted step instead of seven, which is
state between calls. And a heap-allocated state needs four temporaries per RK4
step, which as free functions would be four allocations per step in an N-body
inner loop.

Every stepper exposes `State`, `Scalar`, `order` and
`step(system, time, state, h, out)`. What `system` means is the stepper's
business: an explicit method wants `dy/dt = f(t, y)`, a symplectic one wants an
acceleration `a(t, q)`. That is why the same drivers run both.

```cpp
ysq::Rk4Stepper<ysq::Vec3> stepper;
const ysq::Vec3 end = ysq::integrate(stepper, system, start, 0.0, 10.0, 0.01);

ysq::VelocityVerletStepper<ysq::Vec3> verlet;
ysq::integrate(verlet, acceleration, ysq::PhaseState<ysq::Vec3>{q, v}, 0.0,
               10.0, 0.01, [&](double t, const auto& state) { record(t, state); });

ysq::DormandPrince54Stepper<ysq::Vec3> adaptive;
const auto result = ysq::integrateAdaptive(adaptive, system, start, 0.0, 10.0,
                                           0.01, settings);
```

`asPhaseSystem(acceleration)` wraps an acceleration into a first-order system
so an explicit method can integrate the same problem, which is what lets RK4
and Verlet be compared directly.

`AccelerationField` and `OdeSystem` are structurally identical concepts with
deliberately different names, and both constrain the `step()` they belong to.
They cannot tell each other apart, so handing an acceleration to RK4 still
compiles and still integrates the wrong problem; what they do catch is a
callable of genuinely the wrong shape, reported against the call site rather
than ten frames inside a Runge-Kutta stage.

### Which one to use

| Situation | Method |
| --- | --- |
| A general system, a few thousand steps | `Rk4Stepper` |
| A general system, accuracy specified rather than step size | `DormandPrince54Stepper` |
| A separable system over many orbits, invariants matter | `VelocityVerletStepper` |
| The same, and second order is not enough | `PefrlStepper` |
| Demonstrating what a method's order does not tell you | `ExplicitEulerStepper` |

The choice between RK4 and Verlet is not about order. RK4 is two orders better
per step and far more accurate over a few orbits. Over a million steps its
energy error, which is not bounded, has drifted past Verlet's, which is. A
symplectic method does not conserve the energy of the system it was given; it
exactly conserves that of a nearby one, and the difference is set by the step
size rather than by elapsed time.

`ExplicitEulerStepper` is there to be measured against. It is unstable on
anything oscillatory: the energy of a harmonic oscillator grows without bound
however small the step. `SemiImplicitEulerStepper` differs by one line, costs
exactly the same, and does not. That pair is the cheapest demonstration that
the structure of a method matters more than its order.

## The fixed-step driver adjusts your step

`integrate` rounds the step down to the nearest divisor of the interval so the
run lands exactly on the end time with uniform steps. Asking for 0.3 over a
unit interval gets four steps of 0.25.

This matters more than it sounds. An order-of-accuracy measurement that halves
a step size which does not divide the span is not refining by two, and the
resulting error ratios are meaningless. `math_integrators.cpp` doubles the step
*count* for exactly this reason, and the comment there records what the
alternative looked like.

## Warnings

`ysq::Math` links no warning flags. On an INTERFACE target they would propagate
to every consumer's own sources, which would put `-Wconversion`
`-Wsign-conversion` `-Wdouble-promotion` on `Renderer`, `UI` and
`Applications`. `docs/architecture.md` rules that out: those layers talk to
OpenGL and ImGui, which are float/int APIs.

The strict set is applied instead in `tests/smoke/math_strict_warnings.cpp`,
which includes every header here and **explicitly instantiates every template
for both `float` and `double`**. The instantiations are the point: an
uninstantiated template is barely checked, and without them the file would
compile clean whatever the headers said. `float` is not optional coverage,
since `-Wdouble-promotion` only has anything to say below double precision.

## Tests

Twelve unit files, one integration file, and the smoke check above.
`math_integrators.cpp` is the one that decides whether the module is correct:
it measures each method's observed order from how its error falls under
refinement, and refuses to report a number at all if the errors have fallen
into rounding noise or have not yet reached the asymptotic regime. A wrong
Butcher tableau does not crash. It produces a method that still converges, just
more slowly than advertised, and every result downstream is then quietly less
accurate than the simulation claims.
