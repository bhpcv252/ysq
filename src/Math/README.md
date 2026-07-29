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
| `Math/Grid.hpp`                 | A uniform 1D grid with ghost cells, for the PDE rungs in `Physics` |

Derivations, coefficient tables and their sources are in
[Derivations](#derivations) below.

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
`Applications`. The root `README.md`'s Warnings section rules that out: those
layers talk to OpenGL and ImGui, which are float/int APIs.

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

## Derivations

Coefficient tables, proofs and their sources: the material too long for a
header comment and too important to leave implicit.

### Automatic differentiation

A dual number is `a + bε` with `ε² = 0`. Multiplying two of them,

```
(a + bε)(c + dε) = ac + (ad + bc)ε
```

the ε part is exactly the product rule. That is the whole idea: carry a value
and a derivative together, define every operation to propagate both, and an
ordinary expression evaluated in dual arithmetic produces its own derivative.

Every function in `Dual.hpp` is one line of the chain rule,
`d/dx f(u) = f'(u) du`:

| Function | Value | Tangent |
| --- | --- | --- |
| `a * b` | `a.v * b.v` | `a.d * b.v + a.v * b.d` |
| `a / b` | `a.v / b.v` | `(a.d * b.v - a.v * b.d) / b.v²` |
| `sqrt` | `√a.v` | `a.d / (2√a.v)` |
| `exp` | `e^a.v` | `e^a.v · a.d` |
| `log` | `ln a.v` | `a.d / a.v` |
| `sin` | `sin a.v` | `cos(a.v) · a.d` |
| `tan` | `tan a.v` | `(1 + tan² a.v) · a.d` |
| `atan2(y, x)` | `atan2(y.v, x.v)` | `(x.v·y.d − y.v·x.d) / (x.v² + y.v²)` |
| `hypot(a, b)` | `h` | `(a.v·a.d + b.v·b.d) / h` |

`tan` goes through `1 + tan²` rather than `1/cos²`: one call instead of two,
and it does not lose the answer where cosine is small.

This is *forward* mode. It costs one evaluation per input variable, so it suits
a function of a few variables producing many outputs. The reverse mode, which
suits the opposite shape, is not implemented and is not needed: the derivatives
this project wants are of metric components with respect to four coordinates.

**Nesting, and second derivatives.** Nothing in `Dual` assumes `T` is a
built-in type, so `Dual<Dual<T>>` works by construction. Seeding both levels
gives the second derivative. For `f(x) = x²` with `x = ((x, 1), (1, 0))`:

```
value      = (x, 1)·(x, 1) = (x², 2x)          f  and  f'
derivative = (1, 0)·(x, 1) + (x, 1)·(1, 0) = (2x, 2)
```

so `f(seed).derivative.derivative` is `2`, which is `f''`. Seeding the inner
level in one variable and the outer in another gives a mixed partial the same
way, and that is how `hessian` is built.

**Comparisons look at the value only.** `a == b` and `a < b` compare values
and ignore tangents. This is the usual convention for automatic
differentiation and it is what makes generic code behave: a branch on
`x < 0`, a `clamp`, a componentwise `min` are all asking about magnitude, and
none of them should change answer because a derivative differs.

The cost is that `a == b` no longer implies the two are interchangeable.
`identical(a, b)` is the spelling for component-for-component equality.

**The converting constructor is templated.** `Dual<T>` is implicitly
constructible from anything convertible to `T`, not just from `T`. Without
that, `Dual<Dual<double>> * 3.0` needs two user-defined conversions in a row,
which the language does not allow, and a second-derivative computation cannot
contain a numeric literal. Doing the inner conversion inside the constructor
makes it one conversion for overload resolution to find.

### Runge-Kutta methods

An explicit s-stage method evaluates

```
k_i = f(t + c_i h,  y + h Σ_j a_ij k_j)
y'  = y + h Σ_i b_i k_i
```

and the tableau `(a, b, c)` is the method. The order conditions are polynomial
equations in those coefficients; a tableau that fails one of them still
produces a convergent method, just a lower-order one. That failure mode is why
`math_integrators.cpp` measures the order rather than asserting it by name.

**Classical RK4:**

```
0   |
1/2 | 1/2
1/2 | 0    1/2
1   | 0    0    1
----+---------------------
    | 1/6  1/3  1/3  1/6
```

The weights 1, 2, 2, 1 over 6 are Simpson's rule, which is where the method
comes from: for `f` independent of `y` it *is* Simpson's rule on the interval.

### The Dormand-Prince 5(4) pair

Seven stages producing a fifth-order solution and a fourth-order one from the
same evaluations, so their difference estimates the error at no extra cost.

Coefficients from J. R. Dormand and P. J. Prince, "A family of embedded
Runge-Kutta formulae", *Journal of Computational and Applied Mathematics* 6
(1980), 19-26. This is the pair behind MATLAB's `ode45` and SciPy's `RK45`.

```
0    |
1/5  | 1/5
3/10 | 3/40         9/40
4/5  | 44/45        -56/15        32/9
8/9  | 19372/6561   -25360/2187   64448/6561   -212/729
1    | 9017/3168    -355/33       46732/5247   49/176      -5103/18656
1    | 35/384       0             500/1113     125/192     -2187/6784    11/84
-----+--------------------------------------------------------------------------------
b    | 35/384       0             500/1113     125/192     -2187/6784    11/84     0
b*   | 5179/57600   0             7571/16695   393/640     -92097/339200 187/2100  1/40
```

`b` is the fifth-order solution that gets propagated, `b*` the embedded
fourth-order one used only for the estimate. The second stage has weight zero
in both, which is a property of the pair rather than an omission.

The implementation stores `b` and `b*` and forms their difference at run time
rather than keeping precomputed differences, so there is one set of numbers to
get right instead of two. They are written as exact integer ratios: a mistyped
digit in a decimal expansion collapses the order, and a ratio is checkable by
eye against the paper.

**First Same As Last.** The seventh row of `a` is exactly `b`. So the seventh
stage is evaluated at `(t + h, y_next)`, which means it *is* the first stage
of the next step. Cache it and an accepted step costs six evaluations instead
of seven, a saving of one in seven over a long run.

That carried derivative is state between calls, and it is one of the two
reasons steppers in this module are objects rather than free functions. The
cache is keyed on the time it belongs to, so a rejected step misses and
re-evaluates rather than reusing a derivative from a step that was discarded.
`math_integrators.cpp` asserts the evaluation count is exactly
`6 × accepted + 1` on a run with no rejections.

### Adaptive step control

**The error norm.** A single norm of the whole state would let its largest
component set the step for all of them, which is wrong the moment a position
in metres shares a state with a velocity in metres per second. So each
component is weighed against its own tolerance:

```
err = sqrt( (1/n) Σ_i ( e_i / (atol + rtol · max(|y_i|, |y'_i|)) )² )
```

An answer at or below 1 means the step is acceptable. `errorNorm` walks a
nested state recursively down to its scalar leaves in one pass, so a
`PhaseState` of `Vector3` contributes six components rather than two.

**The PI controller.** Given an error estimate, the naive step update is

```
h_next = h · safety · err^(-1/(p+1))
```

A controller that reacts to each estimate alone oscillates: it overshoots, gets
rejected, overcorrects, and wastes evaluations. Adding a term in the *previous*
error damps that:

```
h_next = h · safety · err^(-α) · err_prev^(β)
α = 1/p − 0.75 β,     β = 0.04
```

after Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*,
2nd ed., section II.4. The exponents are matched to the method's order; `p` is
`Stepper::order`.

Two details that are easy to get wrong and are handled explicitly:

- On a **rejection** the integral term is dropped. The previous error describes
  a step that was not taken.
- An error of **exactly zero** would divide by zero, so it is floored at one
  epsilon before entering the controller.

The scale factor is clamped to `[minimumScale, maximumScale]`, which keeps the
controller from chasing a single unlucky estimate off a cliff.

### Symplectic integrators

For a separable Hamiltonian `H = T(p) + V(q)`, each of

- a **drift**, `q += v·h` at fixed velocity
- a **kick**, `v += a(q)·h` at fixed position

is exactly solvable and area-preserving. Any composition of them is therefore
symplectic, whatever the coefficients, which is what makes these methods
structurally rather than approximately conservative.

**What that buys.** A symplectic method does not conserve the energy of the
system it was given. It exactly conserves the energy of a nearby one, and the
difference between the two is fixed by the step size. So the energy error
oscillates within a band forever rather than accumulating.

**What it does not buy.** Nothing about the trajectory error, which grows with
time for these exactly as for anything else. A symplectic method is right for a
long run whose invariants matter, not for a short run that has to end up in
precisely the right place.

**Velocity Verlet**, order 2, kick-drift-kick:

```
v_½ = v + (h/2)·a(t, q)
q'  = q + h·v_½
v'  = v_½ + (h/2)·a(t + h, q')
```

The symmetry of that sandwich is what makes it time-reversible, and
second-order rather than first.

**Forest-Ruth**, order 4, three kicks with a negative middle coefficient:

```
θ = 1 / (2 − 2^(1/3)) ≈ 1.3512071919596578

drift: θ/2,  (1−θ)/2,  (1−θ)/2,  θ/2      (sums to 1)
kick:  θ,    1−2θ,     θ                   (sums to 1)
```

θ is the root of the condition that the third-order error terms cancel; the
implementation computes it from `cbrt` rather than transcribing it. The middle
step runs backwards in time, which is unavoidable: no composition of
forward-only symplectic steps reaches fourth order. From H. Yoshida,
"Construction of higher order symplectic integrators", *Physics Letters A* 150
(1990), 262-268.

**PEFRL**, Position Extended Forest-Ruth Like, order 4, four kicks:

```
ξ = 0.1786178958448091
λ = −0.2123418310626054
χ = −0.06626458266981849

drift: ξ,  χ,  1−2(χ+ξ),  χ,  ξ
kick:  (1−2λ)/2,  λ,  λ,  (1−2λ)/2
```

One more evaluation per step than Forest-Ruth and a considerably smaller error
constant, so it is the better fourth-order choice per unit of work. The
coefficients have no closed form; they are the numerical solution of the order
conditions. From I. P. Omelyan, I. M. Mryglod and R. Folk, "Optimized
Forest-Ruth- and Suzuki-like algorithms for integration of motion in many-body
systems", *Computer Physics Communications* 146 (2002), 188-202.

**Time dependence.** Strictly, the symplectic guarantee is for an autonomous
system. The time is threaded through the substages so an explicitly
time-dependent force still integrates correctly, but a driven system is not a
Hamiltonian one and the bounded energy error is not promised there.

**Angular momentum.** For a central force, velocity Verlet conserves angular
momentum *exactly*, to rounding, at any step size. A drift changes `r × v` by
`h(v × v) = 0`; a kick changes it by `h(r × a(r)) = 0` because the force is
parallel to `r`. Neither half can touch it. This is a stronger statement than
the bounded energy error and it comes from a different mechanism, which
`tests/integration/math_kepler.cpp` checks by asserting the angular momentum
holds to 1e-12 at a step where the energy error is already 1e-6.

### Coordinate conventions

The physics convention, stated here as well as in the header because a swapped
pair produces a plausible wrong point rather than an error:

```
x = r · sin(polar) · cos(azimuth)
y = r · sin(polar) · sin(azimuth)
z = r · cos(polar)
```

`polar` runs down from +z in [0, π]; `azimuth` runs round from +x in (-π, π].
Mathematics texts routinely swap the two names.

Converting a *point* is only half of what a vector quantity needs. A velocity
or a field has components against a local basis that changes from place to
place. `sphericalBasis` returns that basis as the columns of a matrix, so the
matrix itself is the change of basis: multiplying by it takes local components
to Cartesian ones, and since it is orthonormal its transpose takes them back.

```
ê_r      = ( sinθ cosφ,  sinθ sinφ,  cosθ )
ê_polar  = ( cosθ cosφ,  cosθ sinφ, −sinθ )
ê_azimuth= (     −sinφ,       cosφ,     0 )
```

The Jacobian is the same three directions unnormalised, scaled by how far the
point actually moves per unit of each coordinate: `1`, `r`, and `r sin(polar)`.
Its determinant is `r² sin(polar)`, the volume element every spherical integral
carries.

**Degeneracies.** At the origin neither angle is defined; on the axis the
azimuth is not. Both come back as zero, which is a documented choice rather
than an answer.

### Quadrature

| Rule | Order | Exact for |
| --- | --- | --- |
| Trapezoid | 2 | degree 1 |
| Simpson | 4 | degree 3 |
| Gauss-Legendre, n points | n/a | degree 2n − 1 |

Simpson being exact for cubics is one degree better than the quadratic it is
derived from, which is the fact worth knowing about it.

Gauss-Legendre gets degree `2n − 1` from `n` evaluations because the node
positions are free as well as the weights. That exactness is also the sharpest
available test of a node table: nothing close to the right values integrates
every polynomial up to that degree correctly, so a mistyped digit fails
immediately. Nodes and weights for n = 2, 3, 4, 5 are transcribed rather than
generated, and `math_calculus.cpp` checks each against every monomial up to
degree `2n − 1`, and confirms it stops being exact at `2n`.

**Adaptive Simpson** refines only the panels that fail their local estimate,

```
I ≈ I_refined + (I_refined − I_whole) / 15
```

where 15 is what the next term in the expansion works out to. It is worth
reaching for when the integrand has a feature much narrower than its domain;
for a smooth one a composite rule at the same cost is at least as good, and
`math_calculus.cpp` compares them at a matched evaluation budget rather than at
a matched tolerance for that reason.

**Romberg** is Richardson extrapolation of the trapezoid rule on a repeatedly
halved step. It converges very fast for a smooth integrand and not at all for
one with a kink, since the error expansion it extrapolates does not exist
there.

### Numerical notes

**Why angles never come from acos.** For a small angle θ between unit
vectors, `cos θ = 1 − θ²/2`. At double precision, once θ falls below
`√eps ≈ 1.5e-8` that expression rounds to exactly 1, and `acos(1)` is exactly
0. The angle is gone, and no tolerance recovers it.

`atan2(|a × b|, a · b)` computes the same angle from a ratio whose numerator
stays proportional to θ, and is accurate across the whole range. The same
substitution appears in `toAxisAngle` (`atan2` of the vector part against `w`)
and in `toSpherical` (`atan2` of the distance from the axis against `z`).

`math_vector.cpp` resolves angles down to 1e-8 and asserts that the `acos`
route returns literally zero there, so the contrast is recorded rather than
merely claimed.

**Why summation is compensated.** Naive summation loses the low bits of every
addend once the running total grows large, and the error accumulates with the
number of terms. Neumaier's compensation carries the lost part in a second
register:

```
next = total + value
comp += (|total| ≥ |value|) ? (total − next) + value
                            : (value − next) + total
total = next
```

Neumaier rather than plain Kahan because Kahan silently drops the correction
when an addend is larger than the running total, which is exactly what happens
when a total passes through zero. Summing `{1e16, 1, −1e16}` naively gives 0;
compensated it gives 1.

**Why variance is two-pass, and Welford online.** `E[x²] − E[x]²` is one
subtraction of two nearly equal large numbers and loses everything once the
mean dwarfs the spread. On `1e9 + {1, 2, 3, 4, 5}` it returns a value off by
more than the answer itself.

Welford's update subtracts the running mean before squaring:

```
count += 1
delta  = value − mean
mean  += delta / count
m2    += delta · (value − mean)      ← second deviation against the *new* mean
```

The second deviation being taken against the updated mean is what makes it
stable rather than merely correct. `RunningStatistics::merge` uses Chan's
parallel combination for statistics gathered separately.

**Why the fixed-step driver adjusts your step.** `integrate` rounds the step
down to the nearest divisor of the interval, so the run lands exactly on the
end time and every step is the same size.

Without that, a loop that adds `h` until it passes the end accumulates a
rounding error in the final time and takes one short step. Both corrupt an
order-of-accuracy measurement, which is most of what the driver is used for
here. The consequence for a test is that halving a step size which does not
already divide the span does *not* halve the step actually taken: 0.4 and 0.3
over a unit interval both become 0.25. An order test written that way reports
whatever the quantisation produces. `math_integrators.cpp` doubles the step
count instead.

**The order measurement.** The observed order of a method is read off how its
global error falls under refinement:

```
p ≈ log₂( e(h) / e(h/2) )
```

Three things make that a real measurement rather than a ritual:

- **Refine by doubling the step count**, so each halving is exact.
- **Take the median** of the consecutive ratios. One sample can land badly,
  from FMA contraction on one platform or from a step near a zero of the error
  term, and a median absorbs that where a least-squares fit would be dragged by
  it.
- **Check every error is in the window where truncation dominates**: above
  `100·eps`, below `1e-3`. Below the floor the ratio measures rounding noise;
  above the ceiling the leading term of the expansion has not taken over yet.
  Without this guard an order test quietly becomes a test of nothing, which is
  the usual way it goes wrong.
