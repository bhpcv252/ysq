# Math

Derivations, coefficient tables and their sources. Interface documentation is
in [src/Math/README.md](../src/Math/README.md); this is the material that is
too long for a header comment and too important to leave implicit.

## Contents

- [Automatic differentiation](#automatic-differentiation)
- [Runge-Kutta methods](#runge-kutta-methods)
- [The Dormand-Prince 5(4) pair](#the-dormand-prince-54-pair)
- [Adaptive step control](#adaptive-step-control)
- [Symplectic integrators](#symplectic-integrators)
- [Coordinate conventions](#coordinate-conventions)
- [Quadrature](#quadrature)
- [Numerical notes](#numerical-notes)

## Automatic differentiation

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

### Nesting, and second derivatives

Nothing in `Dual` assumes `T` is a built-in type, so `Dual<Dual<T>>` works by
construction. Seeding both levels gives the second derivative. For `f(x) = x²`
with `x = ((x, 1), (1, 0))`:

```
value      = (x, 1)·(x, 1) = (x², 2x)          f  and  f'
derivative = (1, 0)·(x, 1) + (x, 1)·(1, 0) = (2x, 2)
```

so `f(seed).derivative.derivative` is `2`, which is `f''`. Seeding the inner
level in one variable and the outer in another gives a mixed partial the same
way, and that is how `hessian` is built.

### Comparisons look at the value only

`a == b` and `a < b` compare values and ignore tangents. This is the usual
convention for automatic differentiation and it is what makes generic code
behave: a branch on `x < 0`, a `clamp`, a componentwise `min` are all asking
about magnitude, and none of them should change answer because a derivative
differs.

The cost is that `a == b` no longer implies the two are interchangeable.
`identical(a, b)` is the spelling for component-for-component equality.

### The converting constructor is templated

`Dual<T>` is implicitly constructible from anything convertible to `T`, not
just from `T`. Without that, `Dual<Dual<double>> * 3.0` needs two user-defined
conversions in a row, which the language does not allow, and a second-
derivative computation cannot contain a numeric literal. Doing the inner
conversion inside the constructor makes it one conversion for overload
resolution to find.

## Runge-Kutta methods

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

## The Dormand-Prince 5(4) pair

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

### First Same As Last

The seventh row of `a` is exactly `b`. So the seventh stage is evaluated at
`(t + h, y_next)`, which means it *is* the first stage of the next step. Cache
it and an accepted step costs six evaluations instead of seven, a saving of one
in seven over a long run.

That carried derivative is state between calls, and it is one of the two
reasons steppers in this module are objects rather than free functions. The
cache is keyed on the time it belongs to, so a rejected step misses and
re-evaluates rather than reusing a derivative from a step that was discarded.
`math_integrators.cpp` asserts the evaluation count is exactly
`6 × accepted + 1` on a run with no rejections.

## Adaptive step control

### The error norm

A single norm of the whole state would let its largest component set the step
for all of them, which is wrong the moment a position in metres shares a state
with a velocity in metres per second. So each component is weighed against its
own tolerance:

```
err = sqrt( (1/n) Σ_i ( e_i / (atol + rtol · max(|y_i|, |y'_i|)) )² )
```

An answer at or below 1 means the step is acceptable. `errorNorm` walks a
nested state recursively down to its scalar leaves in one pass, so a
`PhaseState` of `Vector3` contributes six components rather than two.

### The PI controller

Given an error estimate, the naive step update is

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

## Symplectic integrators

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

### Time dependence

Strictly, the symplectic guarantee is for an autonomous system. The time is
threaded through the substages so an explicitly time-dependent force still
integrates correctly, but a driven system is not a Hamiltonian one and the
bounded energy error is not promised there.

### Angular momentum

For a central force, velocity Verlet conserves angular momentum *exactly*, to
rounding, at any step size. A drift changes `r × v` by `h(v × v) = 0`; a kick
changes it by `h(r × a(r)) = 0` because the force is parallel to `r`. Neither
half can touch it. This is a stronger statement than the bounded energy error
and it comes from a different mechanism, which
`tests/integration/math_kepler.cpp` checks by asserting the angular momentum
holds to 1e-12 at a step where the energy error is already 1e-6.

## Coordinate conventions

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

## Quadrature

| Rule | Order | Exact for |
| --- | --- | --- |
| Trapezoid | 2 | degree 1 |
| Simpson | 4 | degree 3 |
| Gauss-Legendre, n points | — | degree 2n − 1 |

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

## Numerical notes

### Why angles never come from acos

For a small angle θ between unit vectors, `cos θ = 1 − θ²/2`. At double
precision, once θ falls below `√eps ≈ 1.5e-8` that expression rounds to exactly
1, and `acos(1)` is exactly 0. The angle is gone, and no tolerance recovers it.

`atan2(|a × b|, a · b)` computes the same angle from a ratio whose numerator
stays proportional to θ, and is accurate across the whole range. The same
substitution appears in `toAxisAngle` (`atan2` of the vector part against `w`)
and in `toSpherical` (`atan2` of the distance from the axis against `z`).

`math_vector.cpp` resolves angles down to 1e-8 and asserts that the `acos`
route returns literally zero there, so the contrast is recorded rather than
merely claimed.

### Why summation is compensated

Naive summation loses the low bits of every addend once the running total grows
large, and the error accumulates with the number of terms. Neumaier's
compensation carries the lost part in a second register:

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

### Why variance is two-pass, and Welford online

`E[x²] − E[x]²` is one subtraction of two nearly equal large numbers and loses
everything once the mean dwarfs the spread. On `1e9 + {1, 2, 3, 4, 5}` it
returns a value off by more than the answer itself.

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

### Why the fixed-step driver adjusts your step

`integrate` rounds the step down to the nearest divisor of the interval, so the
run lands exactly on the end time and every step is the same size.

Without that, a loop that adds `h` until it passes the end accumulates a
rounding error in the final time and takes one short step. Both corrupt an
order-of-accuracy measurement, which is most of what the driver is used for
here. The consequence for a test is that halving a step size which does not
already divide the span does *not* halve the step actually taken: 0.4 and 0.3
over a unit interval both become 0.25. An order test written that way reports
whatever the quantisation produces. `math_integrators.cpp` doubles the step
count instead.

### The order measurement

The observed order of a method is read off how its global error falls under
refinement:

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
