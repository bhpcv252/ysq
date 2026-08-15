# Math

Vectors, matrices, quaternions, complex and dual numbers, tensors, statistics,
interpolation, calculus, and the ODE integrators. Everything the engine
computes with, and nothing that knows what it is computing about.

**Target:** `ysq::Math` (INTERFACE, header-only)
**Depends on:** `ysq::Compute`. `Math/Multigrid.hpp`'s `detail::restrictGrid`/
`detail::prolongateAndAdd` (the geometric-multigrid transfer operators),
`Math/FFT.hpp`'s `fft`/`ifft`/`fft3D`/`ifft3D`, `Math/LinearSolve.hpp`'s
matrix-vector/matrix-matrix `operator*` and `luDecompose`/`choleskyDecompose`,
`Math/Eigen.hpp`'s `qrDecompose`/`jacobiEigenSymmetric`/`svd`,
`Math/SpecialFunctions.hpp`'s batched `erf`/`erfc`/`gamma`/`logGamma`/
`legendreP` overloads, `Math/Polynomial.hpp`'s `Polynomial::operator()`
batched overload, `Math/Interpolation.hpp`'s `CubicSpline::operator()`
batched overload, `Math/Random.hpp`'s `parallelUniformReal`/
`parallelNormal`, and `Math/Sort.hpp`'s `sortInPlace` all route through
`Compute::defaultBackend()` above a size threshold, the same
automatic-dispatch pattern `Physics` uses; below it, or on a machine with
no GPU, the plain CPU path runs instead. All dispatch only when
instantiated for `float` (`Complex<float>` for FFT, `MatrixN<float>`/
`VectorN<float>` for LinearSolve and Eigen, `float` for the batched
evaluation, parallel-RNG and sort families): `Compute`'s GPU interface
is `float`-only
throughout, so a `double` call stays on the CPU path always, gated with
`if constexpr`, rather than silently narrowing through a `float` kernel.
`LinearSolve.hpp`'s `luDecompose`/`choleskyDecompose` and `Eigen.hpp`'s
`qrDecompose`/`jacobiEigenSymmetric`/`svd` are the cases here where the GPU
path is a full host-side loop of dispatches (one iteration per pivot
column/factorization column/sweep round, not a small fixed number), since
elimination and Jacobi rotation are inherently sequential across steps or
rounds; see `src/Compute/README.md`'s own section on why that's still just
the existing kernel building blocks (a reduction, an elementwise update)
chained by a host loop, not a different kind of thing.
`Math/Eigen.hpp`'s `realSchur`/`generalEigenvalues` have no kernel of their
own at all: their repeated internal `qrDecompose` calls (the actual
O(n^3)-per-iteration cost) dispatch transitively once `qrDecompose` itself
does, with no extra code needed. The batched-evaluation family
(`SpecialFunctions.hpp`/`Polynomial.hpp`/`Interpolation.hpp`) is the
simplest dispatch shape here: one GPU thread per element, no reduction, no
sequential host loop, matching the scalar function/method these overloads
sit alongside exactly except for `besselJ`/`besselY`, which deliberately
have no batched overload at all — see `src/Compute/README.md`'s own
section on why float32 cannot carry that specific rational approximation's
internal cancellation. `Math/Sort.hpp`'s `sortInPlace` dispatches via a
bitonic sort, accepting any size (not just a power of two) since each GPU
backend pads with `+infinity` and truncates back internally, invisibly to
the caller; `Math/Sort.hpp`'s own `kthSmallest`/`kthLargest` and
`Math/Statistics.hpp`'s `quantile`/`median` need no kernel of their own,
the same "composes an already-dispatching building block" design
`generalEigenvalues` uses for `qrDecompose`. `Compute`'s kernels are
domain-neutral or
equation-independent, "how do we compute" primitives, not physical
quantities, so this is not a layering violation; see
`src/Compute/README.md`. Not `Core`: Math is usable without a logger, and
never needed one for this either. Every consumer of Math picks up `Compute`
transitively, since Math is `INTERFACE`; that is the accepted cost of a
header-only module gaining a real dependency.

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
| `Math/Statistics.hpp`           | Summaries, compensated summation, an online accumulator, GPU-dispatching quantile/median via `Math/Sort.hpp` |
| `Math/Sort.hpp`                 | Ascending sort (GPU-dispatching above a size threshold) and the order statistics built on it |
| `Math/Interpolation.hpp`        | Lerp through natural cubic splines, with a batched (GPU-dispatching) spline evaluation overload |
| `Math/Geometry/Primitives.hpp`  | `Ray3`, `Sphere3`, `Plane3`, `Segment3`, `AABB3`, `Triangle3`, `OBB3` |
| `Math/Geometry/Intersection.hpp` | Ray/sphere/plane/triangle/AABB crossings, sphere/AABB/plane overlap |
| `Math/Geometry/Queries.hpp`     | Closest point on a segment/plane/AABB/triangle, point-in-polygon |
| `Math/Geometry/ConvexHull.hpp`  | 2D (monotone chain) and 3D (incremental) convex hull          |
| `Math/SpatialPartition/KdTree.hpp` | Point-based radius and k-nearest-neighbor queries          |
| `Math/SpatialPartition/Bvh.hpp` | Box-based overlap and ray queries over a set of `AABB3`s (object partitioning) |
| `Math/SpatialPartition/Octree.hpp` | The same queries via region octants instead (space partitioning) |
| `Math/FFT.hpp`                  | Iterative radix-2 Cooley-Tukey FFT/IFFT over `Complex`         |
| `Math/Calculus.hpp`             | Differentiation and quadrature                               |
| `Math/RootFinding.hpp`          | Newton-Raphson, secant and bisection                          |
| `Math/LinearSolve.hpp`          | Dynamic `MatrixN`/`VectorN`, LU and Cholesky solves           |
| `Math/Eigen.hpp`                | Symmetric (cyclic Jacobi) and general (Schur/QR-algorithm) eigenvalues, QR decomposition, SVD |
| `Math/SpecialFunctions.hpp`     | Error, gamma, associated Legendre and Bessel functions, with batched (GPU-dispatching) overloads for all but Bessel |
| `Math/Polynomial.hpp`           | Evaluation (with a batched, GPU-dispatching overload), differentiation, closed-form roots to degree 4, deflation above it |
| `Math/Random.hpp`               | Uniform/normal/Poisson sampling, normal CDF, Monte Carlo integration, plus a second GPU-dispatching parallel (Philox-based) RNG family |
| `Math/Optimization.hpp`         | Gradient descent with backtracking line search, Nelder-Mead      |
| `Math/CoordinateSystems.hpp`    | Spherical, cylindrical and polar, with their local bases     |
| `Math/ODE.hpp`                  | The integrator interface, state types, and the drivers       |
| `Math/Integrators/Euler.hpp`    | Explicit, semi-implicit, midpoint, Heun                      |
| `Math/Integrators/RK4.hpp`      | Classical fourth-order Runge-Kutta                           |
| `Math/Integrators/Adaptive.hpp` | Dormand-Prince 5(4) with a PI step controller                |
| `Math/Integrators/Symplectic.hpp` | Velocity Verlet, Forest-Ruth, PEFRL                        |
| `Math/Format.hpp`               | `std::formatter` for the value types above                   |
| `Math/Grid.hpp`                 | A uniform 1D grid with ghost cells, for the PDE rungs in `Physics` |
| `Math/Grid3D.hpp`               | The 3D sibling: same role, one more dimension, an `OdeState` in its own right |
| `Math/FiniteDifference.hpp`     | Fourth-order stencils and Kreiss-Oliger dissipation, on a `Grid3D` |
| `Math/Multigrid.hpp`            | A general nonlinear (FAS) geometric multigrid V-cycle solver, on a `Grid3D` |

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

### Root finding

Three methods, ordered by what they demand of the caller and what they give
back in return.

**Newton-Raphson** iterates `x ← x - f(x) / f'(x)`: the tangent line at the
current guess, extended to where it crosses zero. Convergence is quadratic
near a simple root (the number of correct digits roughly doubles each step),
but a bad starting guess can send it anywhere, including nowhere. The
derivative-free overload approximates `f'` with a central difference at each
iterate instead, at the cost of one extra evaluation of `f` per step.

`Physics/Gravity/Kepler.cpp`'s `trueAnomalyFromMeanAnomaly` is the reference
consumer: it solves Kepler's equation `M = E - e sin(E)` for the eccentric
anomaly `E`, starting from the standard `E0 = M + e sin(M)` guess, which is
close enough that a handful of iterations reach double precision for any
bound orbit.

**Secant** replaces the derivative with the slope through the two most recent
iterates, `(f(x1) - f(x0)) / (x1 - x0)`, converging at order φ ≈ 1.618 (the
golden ratio) rather than quadratically, in exchange for never needing `f'`
at all, analytic or approximated.

**Bisection** only needs `f` and a bracket `[lower, upper]` where it changes
sign, halving the bracket every step. Convergence is linear, the slowest of
the three, but it is the only one of the three guaranteed to converge to a
root inside the bracket regardless of how `f` behaves between the endpoints,
since it only ever narrows a region it already knows contains a sign change
rather than extrapolating past what it knows.

### General linear solving

`MatrixN`/`VectorN` are the dynamically sized counterpart to `Matrix2`/`3`/`4`:
those three are fixed-size because they carry geometry (a transform, a
metric), sized to what that geometry needs. A system whose size is only
known at run time — one row per data point in a fit, one row per constraint
in a solver — needs a matrix that doesn't know its size at compile time
either.

**LU decomposition** (Doolittle form, partial pivoting) is Gaussian
elimination stopped halfway: instead of reducing all the way to the
identity the way `Matrix2.hpp`'s `detail::solveByElimination` does, it stops
at a lower-triangular `L` (unit diagonal, so never stored) and an
upper-triangular `U`, packed together into one matrix. The payoff is that a
caller solving `A x = b` for many different `b` against the same `A` pays
the O(n³) factorization once and each subsequent solve is only the O(n²)
pair of triangular substitutions. `solve` is the one-shot convenience for a
single right-hand side; `luDecompose` + `luSolve` is the two-step form for
repeated ones.

Partial pivoting exists for the same reason `Matrix2.hpp`'s elimination
pivots: without it, a small (or exactly zero) pivot either amplifies
rounding error or halts the algorithm outright, and swapping in whichever
remaining row has the largest entry in that column keeps every elimination
factor at most 1 in magnitude.

**Cholesky decomposition** (`A = L Lᵀ`) is the specialization for a symmetric
positive-definite `A`: a normal-equations system `AᵀA x = Aᵀb`, a covariance
matrix, a stiffness matrix from a physically stable system, all guaranteed
positive-definite by construction. It costs about half the arithmetic of LU
and needs no pivoting at all, because positive-definiteness alone already
guarantees every pivot along the way is both positive and the largest
available. `choleskyDecompose` returning `nullopt` the moment a diagonal
entry it needs to take the square root of is non-positive doubles as the
positive-definiteness test: a matrix that is symmetric but not
positive-definite has no real Cholesky factor at all, so there is nothing
further to check.

### Eigendecomposition and related factorizations

#### Symmetric eigendecomposition

The one case where the eigenvalues are guaranteed real and the
eigenvectors guaranteed to form an orthonormal basis, which every consumer
this engine's own physics has needed so far (an inertia tensor, a
covariance matrix) happens to be.

The **cyclic Jacobi** method gets there by repeated plane rotations, each
chosen to zero exactly one off-diagonal entry `a(p, q)`:

```
theta = (a_qq - a_pp) / (2 a_pq)
t = sign(theta) / (|theta| + sqrt(theta^2 + 1))
c = 1 / sqrt(t^2 + 1), s = t c
```

applied as a rotation in the `(p, q)` plane to both the matrix and an
accumulator that starts as the identity. Zeroing `(p, q)` generally
disturbs entries a previous rotation already zeroed, but the total
off-diagonal energy strictly decreases every rotation, so cycling through
every `(p, q)` pair above the diagonal, sweep after sweep, converges to a
diagonal matrix: the diagonal is the eigenvalues, and the accumulator is the
eigenvectors, one per column.

The `t` formula is the numerically stable form (Golub & Van Loan), not the
textbook `theta = 0.5 atan2(2 a_pq, a_qq - a_pp)`: the textbook form divides
by `a_qq - a_pp`, which is exactly zero whenever the two diagonal entries
already agree, precisely the case a real matrix hits often. `math_eigen.cpp`
checks the result three independent ways: against a diagonal matrix's own
entries, against a matrix with closed-form eigenvalues, and by
reconstructing `V diag(lambda) Vᵀ` and comparing it to the original.

#### QR decomposition

Householder reflections, one per column: for column `k`, a reflection is
chosen that zeros every entry below the diagonal in that column without
disturbing the columns already zeroed, the same "reduce one column,
preserve the rest" shape `luDecompose`'s elimination has, but orthogonal
(and so unconditionally stable) rather than merely invertible. The
reflection targets `-sign(a_kk) * ||column||` rather than `+||column||`,
for the same cancellation-avoidance reason `quadraticRealRoots` picks its
own sign: subtracting two nearly equal numbers when `a_kk` is already
close to the column norm would lose precision right where the reflection
vector is built. `qrDecompose` works for any `a` with at least as many
rows as columns; `math_eigen.cpp` checks `q` is orthogonal, `r` is upper
triangular, and `q * r` reconstructs `a`, on both a square and a
non-square input.

#### General eigenvalues via the real Schur form

A general (non-symmetric) matrix does not always have real eigenvalues or
orthogonal eigenvectors, so it cannot always be brought to a fully
triangular form by a real orthogonal similarity transform: a
complex-conjugate eigenvalue pair has no real eigenvector to triangularize
around. The **real Schur form** is what a real orthogonal transform can
always reach instead: quasi-upper-triangular, meaning upper triangular
except for isolated 2x2 blocks straddling the diagonal, each block holding
one complex-conjugate pair's own two-dimensional invariant subspace.

`realSchur` gets there in two stages. First, **Hessenberg reduction**:
the same Householder-reflection idea as `qrDecompose`, but applied on both
sides at once (`a <- qᵀ a q`, a similarity transform, so the eigenvalues
are unchanged) to zero everything below the first subdiagonal. This is a
one-time O(n^3) cost that pays for itself immediately, since the second
stage then runs in O(n^2) per iteration on a Hessenberg matrix instead of
O(n^3) on a dense one.

Second, the **shifted QR algorithm with deflation**: each iteration on the
still-active leading block factors `(a - shift I) = q r` and recombines it
as `r q + shift I`, a similarity transform that drives subdiagonal entries
toward zero from the bottom right upward (the shift is Wilkinson's choice,
computed from the trailing 2x2 block's own eigenvalues). Once a subdiagonal
entry is negligible relative to its neighboring diagonal entries, it is
deflated to exactly zero and the active block shrinks. A block that
shrinks to size 2 without deflating further is left as-is: a real 2x2 with
genuinely complex eigenvalues can never be reduced further by a real
orthogonal similarity, so `eigenvaluesFromSchur` reads it directly instead
of iterating on it — a 1x1 block is a real eigenvalue on its own diagonal
entry, a 2x2 block is solved via the quadratic formula on its own trace
and determinant.

`generalEigenvalues` is `eigenvaluesFromSchur(realSchur(a).t)`.
Eigenvectors are deliberately not part of this round: a numerically solid
eigenvector for a complex eigenvalue needs a complex linear solve (inverse
iteration against a complex-shifted system), and neither
`Math/LinearSolve.hpp` nor `Math/Complex.hpp` has a complex counterpart to
build that on yet. `math_eigen.cpp` checks a companion matrix (whose
eigenvalues are its characteristic polynomial's roots by construction)
against known real roots, a scaled rotation matrix against its known
complex-conjugate pair, and a symmetric matrix against
`jacobiEigenSymmetric`'s own answer for the same input.

#### Singular value decomposition

Computed by **one-sided Jacobi** (Hestenes 1958): the same plane-rotation
idea as `jacobiEigenSymmetric`, aimed at a different target. Instead of
zeroing a symmetric matrix's off-diagonal entry, each rotation orthogonalizes
a pair of columns `p, q` of a working copy of `a`:

```
zeta = (beta - alpha) / (2 gamma)   [alpha, beta: column norms^2; gamma: their dot product]
t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))
c = 1 / sqrt(1 + t^2), s = c t
```

the identical stable-tangent form `jacobiEigenSymmetric` uses, applied to
a different pair of quantities. At convergence the working copy's columns
are exactly `u` scaled by the singular values, so the singular values fall
out as the converged column norms (sorted descending) and `u` as those
columns renormalized; a second matrix accumulates the same rotations
column-for-column into `v`, exactly mirroring how `jacobiEigenSymmetric`
accumulates its own rotations into eigenvectors. Requires `a.rows() >=
a.cols()`; a caller with fewer rows than columns transposes first and swaps
`u`/`v` back afterward. `math_eigen.cpp` checks the reconstruction `u
diag(s) vᵀ = a`, orthonormality of `u` and `v`, and the defining relationship
to the symmetric case: `sigma_i = sqrt(lambda_i)` where `lambda_i` are
`jacobiEigenSymmetric(aᵀ a)`'s own eigenvalues.

### Special functions

`erf`, `erfc`, `gamma` and `logGamma` are thin wrappers over `<cmath>`'s own
`std::erf`, `std::erfc`, `std::tgamma` and `std::lgamma` — those have been in
the standard library since C++11 and need no reimplementation, only the same
ADL-dispatched wrapper pattern (`sqrtOf`, `absOf`) `Math/Scalar.hpp` already
uses, so a future `Numeric` type could extend them the same way.

`legendreP` does not have that shortcut: the associated Legendre
polynomials are part of C++17's special mathematical functions
(`std::assoc_legendre`), which libc++ — the standard library this project
builds against on macOS — does not implement. Computed here instead by the
standard upward recurrence: the closed form for `P_m^m`,

```
P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
```

built up one factor of the double factorial at a time rather than computed
as a separate intermediate (which risks overflowing before the `(1-x^2)^
(m/2)` factor brings the product back down for large `m`), followed by the
three-term recurrence

```
(n-m) P_n^m(x) = x(2n-1) P_{n-1}^m(x) - (n+m-1) P_{n-2}^m(x)
```

up to the requested `n`. `math_specialfunctions.cpp` checks the low-order
terms (`P_0`, `P_1`, `P_2`, `P_1^1`, `P_2^1`, `P_2^2`) against their closed
forms directly, since those are also the terms most consumers actually use.

**Bessel functions** (`besselJ0`/`besselJ1`/`besselJ` for the first kind,
`besselY0`/`besselY1`/`besselY` for the second) have the same
no-standard-library-shortcut problem as `legendreP` (`std::cyl_bessel_j`
and `std::cyl_neumann` are C++17 special math functions libc++ does not
implement), solved by the classic Abramowitz & Stegun 9.4 rational
approximations for orders 0 and 1 (one polynomial fit for `|x| < 8`, an
asymptotic amplitude/phase fit beyond it — the small-`x` branch of `Y0`/`Y1`
carries an explicit `ln(x) J0(x)`/`ln(x) J1(x)` term, since `Y_n` has a
logarithmic singularity at the origin no polynomial alone reproduces), then
recurrence for every higher order. The two families recur in opposite
stable directions: `J`'s three-term recurrence is unstable running upward
(errors amplify) but stable running downward, so `besselJ` for order 2 and
above uses **Miller's algorithm** — seed an arbitrary value at an order
well above the one requested, recur down to order 0, then rescale the
whole sequence at the end via the sum rule `J_0(x) + 2 sum_{k>=1} J_{2k}(x)
= 1`, since downward recurrence gets every ratio between consecutive
orders right immediately but not the overall scale. `Y`'s recurrence is
stable in the opposite direction, so `besselY` just recurs upward from
`besselY0`/`besselY1` directly. `math_specialfunctions.cpp` checks `J0`/`J1`
at the origin and at their known first zeros, the three-term recurrence
independently for both families (at an `x` on each side of the
downward/upward branch threshold `besselJ` itself switches on), and the
Wronskian identity `J_n(x) Y_{n+1}(x) - J_{n+1}(x) Y_n(x) = -2/(pi x)`
linking the two families together.

### Polynomial roots

Closed forms exist in radicals up to degree 4 and no further — the
Abel-Ruffini theorem is why there is no general formula for degree 5, and
so no `quinticRealRoots` to reach for. `realRoots` uses the closed form
directly through degree 4, and above that finds one real root numerically
at a time and divides it out, reducing the degree by one, until degree 4 is
reached and the closed form finishes the job.

**Quadratic** uses the numerically stable form rather than the textbook
formula applied literally: `(-b +- sqrt(disc)) / 2a` cancels badly for
whichever root has the same sign as `b`. Computing one root as
`q / a` (`q` chosen with the sign that avoids the cancellation) and the
other from Vieta's formula, `c / q`, keeps both roots accurate.
`math_polynomial.cpp` includes a case (`a=1, b=1e8, c=1`) chosen so the
naive formula would lose most of its precision on the smaller root, and
checks the returned roots by substitution rather than against a literal, to
confirm the stable form actually earns its keep there.

**Cubic** depresses to `t^3 + p t + q = 0` and branches on the sign of
`(q/2)^2 + (p/3)^3`. Positive means one real root, reached by Cardano's
formula with real cube roots throughout. Negative means three distinct real
roots — the "casus irreducibilis", where Cardano's formula technically
still works but only by passing through a complex intermediate value that
happens to have zero imaginary part at the end, which is worse to compute
with than the trigonometric substitution `t = 2 sqrt(-p/3) cos(theta)`, whose
`cos(3 theta) = 3q / (2 p sqrt(-p/3))` stays real the entire way.

**Quartic** depresses to `t^4 + p t^2 + q t + r = 0`. A vanishing `q` is the
biquadratic special case, a quadratic in `t^2`. Otherwise it factors into
two real quadratics, `(t^2 + st + u)(t^2 - st + v)`; matching coefficients
against `p`, `q`, `r` shows `s^2` must be a root of the resolvent cubic
`z^3 + 2p z^2 + (p^2 - 4r) z - q^2 = 0`. A real quartic's complex roots
always come in conjugate pairs, so it always factors into two real
quadratics, which means this cubic always has a real, nonnegative root; once
`s = sqrt(z)` is in hand, `u` and `v` fall out algebraically and each
quadratic is solved by the stable quadratic form above.

**Above degree 4**, `detail::findOneRealRootBySampling` samples
`[-bound, bound]` (`bound` from Cauchy's bound, `1 + max_i |c_i / c_n|`, safe
for any root) looking for a sign change, brackets it with bisection, and
polishes with Newton-Raphson; `detail::deflate` then divides the found root
out by synthetic division, one degree lower, and the loop repeats. This
stops (returning whatever roots it already found) if a sampling pass turns
up no sign change, which happens for a real root the grid straddles without
crossing (a double root) or simply fewer real roots than the degree allows.
`math_polynomial.cpp` verifies this path two ways: the closed forms and the
deflation path agree on the same cubic given through both routes, and a
degree-5 polynomial's roots (found only by deflation) satisfy the *original*
un-deflated polynomial, not just the reduced one at the point each was
found.

### Euclidean geometry

`Math/Geometry/` holds shapes and operations with no physical meaning —
nothing here has a mass, a material, or an owner, the same separation
`Math/Intersection.hpp` (now folded into this directory alongside its new
siblings) always drew, just with more company now.

**Ray-triangle** uses the Möller-Trumbore algorithm: it never computes the
triangle's plane explicitly, instead solving directly for the barycentric
coordinates `u`, `v` (and the ray parameter `t`) that express the hit point
as `(1-u-v) a + u b + v c`. A hit is inside the triangle exactly when
`u >= 0`, `v >= 0` and `u + v <= 1`, so those three comparisons are the
entire inside/outside test, with no separate point-in-triangle check
needed afterward.

**Ray-AABB** uses the slab method: an axis-aligned box is the intersection
of three axis-aligned slabs (the region between a pair of parallel planes),
so intersecting the ray against each pair in turn and narrowing
`[tMin, tMax]` every time finds the overall entry and exit parameters
directly, with no need to test individual faces.

**Closest-point-on-triangle** (Ericson, *Real-Time Collision Detection*)
classifies the query point against the triangle's Voronoi regions: three
vertex regions, three edge regions, and the face region, in that order,
each ruled out by a couple of dot products before falling through to the
next. Whichever region the point lands in fixes the answer immediately —
a vertex, a point along an edge, or a barycentric combination of all three
vertices for the face region — without needing the general (and more
expensive) closest-point-on-a-plane-then-clamp approach.

**2D convex hull** uses Andrew's monotone chain: sort every point by `x`
(then `y`), then build the lower and upper chains of the hull in one linear
pass each, popping the most recently kept point whenever the next point
would make a clockwise turn. Every popped point is provably inside the
hull of its neighbors, so nothing discarded ever needs revisiting — the
whole algorithm is `O(n log n)`, dominated by the initial sort.

**3D convex hull** uses the incremental algorithm: start from a tetrahedron
built from four well-separated points (found by farthest-pair, then
farthest-from-that-line, then farthest-from-that-plane, so the starting
volume is never degenerate unless every point actually is coplanar), then
add every remaining point in turn. Adding a point removes every face it can
see past (a face whose outward side the point is on) and patches the
resulting hole with new faces connecting the point to the *horizon*, the
boundary between removed and kept faces. The horizon is found without
maintaining a face-adjacency graph at all: collect every visible face's
three directed edges, and an edge survives as a horizon edge exactly when
its reverse does not also appear in that collection (an edge shared by two
visible faces cancels out; one on the boundary does not). Reusing each
horizon edge's own direction for its new face keeps every face's outward
orientation correct automatically, the same property the initial
tetrahedron's four faces get by checking directly against the one
tetrahedron vertex each excludes. `math_geometry_convexhull.cpp` checks a
cube's hull three ways: exactly 12 triangles, an interior point contributing
no vertex to any of them, and every face's normal pointing away from the
hull's own centroid.

**Oriented bounding boxes** (`OBB3`) get their overlap tests from the
**Separating Axis Theorem** (SAT): two convex shapes are disjoint if and
only if some axis exists onto which their projections do not overlap, and
for two boxes the only candidate axes that can ever be the one that
separates them are each box's own three face normals, plus the nine
pairwise cross products of one box's axes with the other's — fifteen axes
in total (Gottschalk, Lin & Manocha 1996). The six face-normal axes alone
catch every *face-to-face* separation; the nine cross-product axes are
what catch a separation that is genuinely *edge-to-edge*, the case two
boxes can overlap on every face normal and still not actually touch.
`intersects(OBB3, AABB3)` reuses the same routine by building the AABB's
equivalent degenerate OBB (axis-aligned, so its own three face normals are
already among the standard basis) rather than re-deriving SAT for a case
it already covers. Ray-OBB and closest-point-on-OBB both work by
transforming into the box's own local frame (a pure change of basis, since
`axes` is orthonormal) and reusing the AABB versions of the same
operations there. `math_geometry_intersection.cpp` includes a
purpose-built case — two thin rods with generically skew long axes — where
the true minimal separating direction is a cross-product axis and every
face-normal axis alone would wrongly report overlap, the case a naive
six-axis test would get wrong.

### Spatial partitioning

Two different strategies for the same "what's near this?" question, over
objects with their own extent (each given as an `AABB3`): `Bvh3` splits
the *objects* (object partitioning), `Octree3` splits the *space* they
live in (space partitioning). `KdTree3`, below, is the third structure,
for dimensionless points rather than extended objects.

**`KdTree3`** splits points by their median along an axis that cycles `x`,
`y`, `z` with tree depth, so every level halves the remaining search space
along whichever axis still discriminates. `radiusQuery` descends the near
side of each splitting plane unconditionally and the far side only if the
plane itself is within the query radius — the minimum distance any far-side
point could possibly be, regardless of where exactly it sits, so this never
misses a real neighbor while still pruning most of the tree. `nearestNeighbors`
prunes the same way against a shrinking bound: once a bounded max-heap holds
`k` candidates, its own worst distance becomes the radius, so it tightens as
better candidates are found. `Physics/Fluids/SPH.cpp`'s neighbor search is
the reference consumer: the cubic spline kernel has compact support at
`2h`, so a radius query at that distance finds exactly the particles that
contribute anything, replacing what was a brute-force all-pairs loop.

**`Bvh3`** builds top-down, splitting each range of objects along whichever
axis its combined bounding box is longest on, at the median object center
along that axis — not the surface-area heuristic a production renderer
would use to minimize expected query cost, but enough to give every query
the `O(log n)` depth a balanced tree provides over testing every object.
`overlapQuery` and `rayQuery` both descend only into children whose own
bounds the query region or ray actually touches
(`Math/Geometry/Intersection.hpp`'s `intersects`/`intersect` do the actual
test), which is the entire pruning strategy: a box that misses a node's
bounds cannot possibly overlap anything inside it.

**`Octree3`** instead fixes a `worldBounds` region up front and splits it
into eight equal octants at its center, recursively, stopping once a
node's own object count is small enough or a maximum depth is reached. An
object is handed down to a single child only if its box fits entirely
inside that child's octant; a box straddling the split (spanning both
halves on some axis) stays at the node doing the splitting, rather than
being duplicated into every octant it touches or forcing an ambiguous
choice of one. Because a cell's own bounds are then a fixed geometric
region rather than a tight fit around whatever objects it holds (unlike a
`Bvh3` node, whose bounds are always exactly the union of its objects'
boxes), a query still has to test each candidate object's own box, not
just trust that overlapping the cell means overlapping the object inside
it — the cell bounds are only good enough to prune whole subtrees, the
same role `Bvh3`'s tighter bounds play. Fixed cells (rather than
object-derived ones) are the reason to reach for an octree over a BVH:
the space partitioning stays meaningful even as objects move or get
added, which an object-partitioned tree cannot promise without a full
rebuild.

All three structures are validated against brute force directly, not just
against small hand-picked cases: `math_spatialpartition_kdtree.cpp`,
`math_spatialpartition_bvh.cpp` and `math_spatialpartition_octree.cpp`
each build a random cloud of points or boxes and check that the
structure's answer matches an all-pairs search over the same data, which
is the property that actually matters (the optimization changes nothing
about which answer is correct).

### The Fast Fourier Transform

The discrete Fourier transform by definition costs `O(n^2)`: every output
bin is a sum over every input sample. The iterative radix-2 Cooley-Tukey
algorithm gets the same answer in `O(n log n)` by exploiting that an
`n`-point DFT decomposes exactly into two `n/2`-point DFTs (over the even-
and odd-indexed samples) plus `O(n)` combining work, recursively, down to
one-point transforms (which are trivial: the identity).

The *iterative* form used here runs that decomposition bottom-up instead of
by recursive call: `detail::bitReversalPermute` reorders the input so that
index `i` holds what belongs at `i`'s bit-reversal, which is exactly the
order the bottom level's trivial one-point "transforms" need to already be
in for the first real butterfly stage (combining pairs into two-point
results) to combine the correct pairs. Each stage after that doubles the
group size — pairs, then groups of four, then eight, and so on up to `n` —
which is why `data.size()` must be a power of two.

Forward and inverse share one routine, differing only in the sign of the
twiddle factors' angle (the DFT and its inverse are the same sum with the
sign of the exponent flipped) and a final `1/n` scale the inverse applies
and the forward transform does not, the usual convention for which
direction carries the normalization. `math_fft.cpp` checks the transform
four independent ways: a constant signal's energy lands entirely in bin
zero, a pure sinusoid's energy lands entirely in its own frequency bin (and
its mirror), a round trip through `fft` then `ifft` recovers the original
signal, and — the check least dependent on anything about this
implementation's own internal structure — the result matches the `O(n^2)`
direct sum over the DFT's textbook definition on a small case.

**`fft3D`/`ifft3D`** add no new transform algorithm: a 3D DFT factors
exactly into three passes of the existing 1D transform, one per axis (the
row-column algorithm), since each 1D pass along one axis only mixes
points that already share both other coordinates, and running the three
passes in any order reaches the same result. Each pass extracts a line
(strided for `x` and `y`, already contiguous for `z`) into a scratch
buffer, transforms it with the existing `fftImpl`, and writes it back.
Normalization falls out for free: each pass's own `1/n` on the inverse
multiplies together across all three passes into exactly the
`1/(nx ny nz)` a direct 3D inverse definition would apply once.
`math_fft.cpp` checks the 3D transform the same independent ways as the
1D one (constant signal, round trip on random data) plus one more specific
to being separable: with the two extra axes both of size 1, `fft3D` must
agree exactly with plain `fft` along the one real axis, since there is
nowhere else for a genuinely 3D transform to mix into.

### Randomness

Every function here takes an `RandomEngine&` explicitly rather than reading
a hidden global generator, so a caller who seeds one engine and threads it
through gets the same sequence on every run — bit-for-bit reproducibility
that a global generator cannot offer, since its state depends on every
other call anywhere in the program, not just this caller's own history.
`RandomEngine` is fixed to `std::mt19937_64`: one engine choice for the
whole project, so a seed alone reproduces a run rather than a seed and also
which engine type produced it.

`normalCdf` is the one place outside `Math/SpecialFunctions.hpp` itself
that `erf` earns its keep: `CDF(x) = (1 + erf((x - mean) / (stddev
sqrt(2)))) / 2` is the standard identity relating the two.

`monteCarloIntegrate` converges as `1/sqrt(samples)` regardless of
dimension, worse than `Math/Calculus.hpp`'s deterministic rules for a smooth
one-dimensional integrand (Gauss-Legendre reaches machine precision in a
handful of evaluations, not thousands) — its actual advantage, a
high-dimensional integral where a deterministic rule's cost grows
exponentially with dimension while Monte Carlo's does not grow with
dimension at all, is not exercised by this module's one-dimensional
signature, but is the reason to reach for it once a real consumer needs
more than one dimension.

### Optimization

Two families, for two situations, both minimizing a scalar field `f: V ->
T` over `V` = `Vector2`/`3`/`4`.

**`gradientDescent`** needs `f` to be differentiable: it estimates the
gradient with `numericalGradient` (the same finite-difference machinery
`Math/Calculus.hpp` already has, not reimplemented here) and steps opposite
it. A fixed step size either overshoots a steep region or crawls through a
shallow one — `backtrackingLineSearch` picks the step adaptively each
iteration instead, shrinking a starting guess until it satisfies the Armijo
sufficient-decrease condition,

```
f(x + step * direction) <= f(x) + c1 * step * (gradient . direction)
```

which only accepts a step that actually decreases `f` by a fraction of what
the (linear) gradient prediction promises, ruling out both an overshoot
that overall increases `f` and a step so large it exploits how the linear
prediction breaks down.

**`nelderMead`** needs nothing but values of `f` at points: no derivative,
estimated or exact, anywhere in it. It carries `n + 1` points (`n` the
dimension of `V`) and each iteration replaces the worst one by reflecting,
expanding or contracting it through the centroid of the rest, falling back
to shrinking the whole simplex toward the best point only when none of
those improve on the worst. This is the one to reach for once `f` is noisy,
discontinuous, or otherwise not something a finite difference could
estimate a useful gradient from — `math_optimization.cpp` includes exactly
that case, minimizing a sum of absolute values (not differentiable at its
own minimum), which `gradientDescent` has no good answer for and
`nelderMead` does not need one for.

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
