# Math API reference: vectors, matrices, and everything scalar

Every public type and function in `Math` except the ODE integrators, which
have their own page: [docs/api/math/integrators.md](integrators.md). Start
with [docs/math/algebra.md](../../math/algebra.md) for the ideas; this page
is the lookup table. [src/Math/README.md](../../../src/Math/README.md) has
the conventions, derivations and coefficient tables in full.

Everything here is templated on its scalar `T`, constrained by the `Numeric`
concept below. `Vec3`/`Mat4`/etc. are the `double` aliases; `Vec3f`/`Mat4f`
the `float` ones. `Math` has no dependency on `Core`, not even for logging.

Two policies apply throughout and aren't repeated per entry:

- **Bounds are asserted, not checked.** Every `operator[]`/`operator()`
  asserts its index; this compiles out under `NDEBUG`.
- **Failure is reported, not invented.** The unchecked form (`normalized`,
  `inverse`) propagates NaN on a degenerate input; the `tryX` form returns
  `std::optional` instead.

## `Math/Scalar.hpp`

The concept everything else is built on, plus constants and tolerant
comparison.

```cpp
template <class T>
concept Numeric = /* a field: +, -, *, /, ==, <, and a square root */;

template <std::floating_point T> inline constexpr T kPi;
template <std::floating_point T> inline constexpr T kTau;   // 2*pi
template <std::floating_point T> inline constexpr T kE;
template <std::floating_point T> inline constexpr T kDefaultRelTol;  // 128 epsilons
template <std::floating_point T> inline constexpr T kDefaultAbsTol;

template <std::floating_point T> constexpr T radians(T degrees) noexcept;
template <std::floating_point T> constexpr T degrees(T radians) noexcept;

template <Numeric T> constexpr T clamp(T value, T lo, T hi) noexcept;
template <Numeric T> constexpr T sign(T value) noexcept;  // -1, 0, +1; NaN -> 0

template <std::floating_point T>
constexpr bool approxEqual(T a, T b, T relTol = kDefaultRelTol<T>,
                           T absTol = kDefaultAbsTol<T>) noexcept;
template <std::floating_point T>
constexpr bool isNearZero(T value, T absTol = kDefaultAbsTol<T>) noexcept;
```

`Numeric` is deliberately weaker than `std::floating_point`: satisfied by
`float`, `double`, and by `Dual<T>`. `Complex<T>` does **not** satisfy it (no
ordering), which is why it cannot go inside `Vector`/`Matrix`.

| Function | Description |
| --- | --- |
| `clamp(value, lo, hi)` | By value, unlike `std::clamp`. Returns `lo` if `hi < lo` rather than being undefined. |
| `sign(value)` | `-1`, `0`, or `+1`; NaN gives `0`. |
| `approxEqual(a, b, relTol, absTol)` | `\|a - b\| <= max(absTol, relTol * max(\|a\|, \|b\|))`. Equal infinities compare equal; any NaN compares unequal, including to itself. |
| `isNearZero(value, absTol)` | `\|value\| <= absTol`. |

```cpp
ysq::approxEqual(computed, expected, 1e-9);
const double angle = ysq::radians(90.0);
```

## `Math/Vector2.hpp`, `Vector3.hpp`, `Vector4.hpp`

Fixed-size vectors: position, direction, velocity. Aggregates with public
members (`v.x`, `v.y`, ...), standard layout, trivially copyable, so an
array of these uploads to a GPU buffer as-is.

```cpp
template <Numeric T> struct Vector3 {
    T x{}, y{}, z{};
    static constexpr std::size_t size() noexcept;   // 2, 3, or 4
    T& operator[](std::size_t index) noexcept;       // asserted
    static constexpr Vector3 zero() noexcept;
    static constexpr Vector3 splat(T value) noexcept;
    static constexpr Vector3 unitX/unitY/unitZ() noexcept;
    // += -= *= /=, unary +/-, + - * / (vector-vector, vector-scalar)
    // == is exact, component by component
};
using Vec3 = Vector3<double>;
using Vec3f = Vector3<float>;
```

Free functions, shared shape across `Vector2`/`3`/`4` (all constexpr and
`noexcept` unless noted):

| Function | Description |
| --- | --- |
| `dot(a, b)` | Euclidean inner product. |
| `lengthSquared(v)` | `dot(v, v)`. |
| `length(v)` | Not `noexcept`: goes through a `sqrt`. |
| `normalized(v)` | Undefined direction at zero length: every component comes back NaN. |
| `tryNormalized(v)` | `nullopt` for a zero, non-finite, or overflowing-length vector. Use this where the input can legitimately be zero. |
| `distanceSquared(a, b)` / `distance(a, b)` | `lengthSquared`/`length` of `a - b`. |
| `lerp(a, b, t)` | Exact at both endpoints; extrapolates outside `[0, 1]`. |
| `project(a, onto)` / `reject(a, from)` | Component along a direction / the orthogonal remainder. `project + reject` reconstructs `a`. |
| `reflect(v, n)` | Mirror of `v` in the plane whose unit normal is `n`. |
| `hadamard(a, b)` | Componentwise product. |
| `min(a, b)` / `max(a, b)` / `abs(v)` | Componentwise. |
| `angleBetween(a, b)` | Via `atan2`, never `acos` (see below). |

Per-dimension additions:

- **`Vector2`**: `cross(a, b)` (the scalar 2D cross / signed parallelogram
  area), `perpendicular(v)` (quarter turn CCW). `angleBetween` is *signed*,
  in `(-pi, pi]`.
- **`Vector3`**: `cross(a, b)` (the 3D cross product), `scalarTriple(a, b,
  c)` (`a . (b x c)`, the signed parallelepiped volume), `rotateAbout(v,
  axis, angle)` (Rodrigues' formula, `axis` must be unit), `v.xy()`.
  `angleBetween` is *unsigned*, in `[0, pi]`.
- **`Vector4`**: no cross product (the 4D analogue is a wedge product, in
  `Tensor`). `Vector4::point(v)` / `Vector4::direction(v)` build homogeneous
  coordinates (`w = 1` / `w = 0`) for use with `Matrix4`.
  `perspectiveDivide(v)` divides `x, y, z` by `w`, undefined for `w = 0`.
  `v.xy()`, `v.xyz()`.

```cpp
const ysq::Vec3 a{1, 0, 0};
const ysq::Vec3 b{0, 1, 0};
ysq::cross(a, b);                    // (0, 0, 1)
ysq::angleBetween(a, b);             // pi/2

const std::optional<ysq::Vec3> dir = ysq::tryNormalized(velocity);
if (dir) { /* velocity was non-zero and finite */ }
```

**Why `angleBetween` never uses `acos`.** For a small angle, `acos` of the
normalized dot product has already rounded to exactly 1 and returns exactly
0; `atan2` of the cross magnitude over the dot stays accurate across the
whole range. The same reasoning applies to `Quaternion::toAxisAngle` and
`toSpherical`'s polar angle below.

**The representable range.** `length` overflows for a component beyond
about `1.3e154` at double precision, and underflows below about `1.5e-162`;
`tryNormalized`/`tryInverse`/`solve` all detect this rather than silently
returning a zero result. This is wide enough for the project's scales
(distances to ~1e21 m, masses to ~1e41 kg).

## `Math/Matrix2.hpp`, `Matrix3.hpp`, `Matrix4.hpp`

Fixed-size matrices: rotation, scale, projection. **Column-major storage,
column-vector convention**: `columns[j]` is the j-th column, transforms
compose right to left (`M = T * R * S`), and apply as `v' = M * v`. This is
GLSL's convention, so a `Matrix4` uploads to a uniform with `transpose =
GL_FALSE`. `operator[]` indexes a column; `operator()(row, col)` indexes an
element in reading order, and mixing the two up is the classic bug here.

```cpp
template <Numeric T> struct Matrix3 {
    using Column = Vector3<T>;
    std::array<Column, 3> columns{};

    static constexpr std::size_t rows() noexcept;
    static constexpr std::size_t cols() noexcept;
    Column& operator[](std::size_t col) noexcept;
    T& operator()(std::size_t row, std::size_t col) noexcept;
    Column row(std::size_t index) const noexcept;

    static constexpr Matrix3 zero/identity() noexcept;
    static constexpr Matrix3 fromColumns(c0, c1, c2) noexcept;
    static constexpr Matrix3 fromRows(r0, r1, r2) noexcept;        // written as on paper
    static constexpr Matrix3 diagonal(d) / scale(s) noexcept;
    static constexpr Matrix3 outerProduct(a, b) noexcept;           // a b^T
    static constexpr Matrix3 crossMatrix(a) noexcept;               // [a]_x, cross(a, v) == this * v
    static Matrix3 rotationX/rotationY/rotationZ(angle);
    static Matrix3 rotation(axis, angle);                           // Rodrigues'
    // += -= *= /= (scalar and matrix), + - * /, M * v, M * M, ==
};
using Mat3 = Matrix3<double>;
```

Free functions, shared across `Matrix2`/`3`/`4`:

| Function | Description |
| --- | --- |
| `transpose(m)` | |
| `determinant(m)` | `Matrix2`: `2x2` formula. `Matrix3`: scalar triple of the columns. `Matrix4`: Laplace expansion (readable, not the fastest form; this isn't an inner loop). |
| `trace(m)` | Sum of the diagonal. |
| `tryInverse(m)` | `nullopt` if singular or NaN. **Near-singular is not detected**; use `solve` for a matrix that came from measurement rather than a transform you built. |
| `inverse(m)` | Unchecked: a singular matrix yields infinities or NaN. |
| `solve(m, b)` | Solves `M x = b` by Gauss-Jordan elimination **with partial pivoting**. `nullopt` if singular. Slower than `inverse` but bounded error near-singular. |

Per-size additions:

- **`Matrix2`**: `rotation(angle)`, counter-clockwise.
- **`Matrix3`**: `upperLeft2x2()`; `inverseOrthogonal(m)` (transpose, correct
  only for an orthogonal matrix, wrong silently otherwise).
- **`Matrix4`**: `upperLeft3x3()`, `translationPart()`;
  `translation(offset)`, `fromLinear(linear)`,
  `fromLinearTranslation(linear, offset)`; `rotationX/Y/Z(angle)`,
  `rotation(axis, angle)`; `lookAt(eye, center, up)` (right-handed view
  matrix); `perspective(fovY, aspect, nearPlane, farPlane)` and
  `orthographic(left, right, bottom, top, nearPlane, farPlane)` (both target
  OpenGL clip space, `z` in `[-1, 1]`); `adjugate(m)`; `inverseAffine(m)`
  (cheap inverse for a matrix whose last row is `(0,0,0,1)`, silently wrong
  for a projective matrix); `transformPoint`/`transformDirection`/
  `projectPoint(m, v)` (the last does the perspective divide).

```cpp
const ysq::Mat4 view = ysq::Mat4::lookAt(eye, target, ysq::Vec3::unitY());
const ysq::Mat4 proj = ysq::Mat4::perspective(ysq::radians(60.0), aspect, 0.1, 1000.0);
const ysq::Vec3 world = ysq::transformPoint(model, localPoint);

const std::optional<ysq::Vec3> x = ysq::solve(matrixFromMeasurement, rhs);
```

## `Math/Quaternion.hpp`

Rotations, without a matrix's drift or gimbal lock. Stored `(w, x, y, z)`,
scalar part first. Default-constructs to the **identity** rotation, not
zero; `Quaternion::zero()` is the additive identity, for accumulating a
derivative.

```cpp
template <Numeric T> struct Quaternion {
    T w{1}, x{}, y{}, z{};

    static constexpr Quaternion identity() noexcept;
    static constexpr Quaternion zero() noexcept;
    static constexpr Quaternion fromScalarVector(scalar, vector) noexcept;
    static Quaternion fromAxisAngle(axis, angle);         // axis must be unit
    static Quaternion fromEulerZYX(yaw, pitch, roll);     // intrinsic Z-Y-X
    static Quaternion fromRotationMatrix(const Matrix3<T>&);  // Shepperd's method
    Vector3<T> xyz() const noexcept;                       // the vector part
    // += -= *= /= (scalar and quaternion, Hamilton product), + - * /, ==
};
using Quat = Quaternion<double>;

struct AxisAngle { Vector3<T> axis; T angle; };
struct EulerZYX { T yaw, pitch, roll; };
```

| Function | Description |
| --- | --- |
| `dot(a, b)` / `lengthSquared(q)` / `length(q)` | |
| `conjugate(q)` | `(w, -x, -y, -z)`. |
| `normalized(q)` / `tryNormalized(q)` | |
| `inverse(q)` | General inverse, any non-zero quaternion. |
| `inverseUnit(q)` | `conjugate(q)`; correct only for a unit quaternion. |
| `rotate(q, v)` | Rotates `v` by unit quaternion `q`. Requires `q` unit. |
| `toMatrix3(q)` | |
| `toAxisAngle(q)` | Angle in `[0, pi]`. At the identity the axis comes back as `+X`, arbitrarily. |
| `toEulerZYX(q)` | Handles gimbal lock near `pitch = +-pi/2` by pinning roll to zero. |
| `angleBetween(a, b)` | Angle of the relative rotation, in `[0, pi]`. `q` and `-q` are the same rotation and compare equal here even though `operator==` says no. |
| `nlerp(a, b, t)` | Cheap, shortest arc, non-constant angular rate. |
| `slerp(a, b, t)` | Constant angular rate along the shortest arc; falls back to `nlerp` when `a`, `b` are nearly parallel. |

```cpp
const ysq::Quat spin = ysq::Quat::fromAxisAngle(ysq::Vec3::unitZ(), std::numbers::pi / 2.0);
const ysq::Vec3 rotated = ysq::rotate(spin, ysq::Vec3::unitX());  // approx (0, 1, 0)
```

Twelve Euler-angle conventions exist; `fromEulerZYX`/`toEulerZYX` are named
for the one they implement (intrinsic yaw-Z, pitch-Y, roll-X) rather than
leaving it to be assumed.

## `Math/Complex.hpp`

Ours rather than `std::complex`, because the standard only guarantees
`std::complex<float|double|long double>`; `Complex<Dual<double>>` (a phase
carrying its own derivative) needs a type that works for any `Numeric`.
**Does not satisfy `Numeric`** (no ordering), so it cannot go inside
`Vector`/`Matrix`.

```cpp
template <Numeric T> struct Complex {
    T re{}, im{};
    static constexpr Complex zero/one/i() noexcept;
    static constexpr Complex real(T) / imaginary(T) noexcept;
    static Complex polar(T radius, T angle);
    // += -= *= /= (complex and scalar), + - * / (complex-complex, complex-scalar), ==
};
using Cplx = Complex<double>;
```

| Function | Description |
| --- | --- |
| `conj(z)` | |
| `lengthSquared(z)` | `\|z\|^2`, no square root. |
| `abs(z)` / `length(z)` | Modulus, via `hypot`: does not overflow where `sqrt(re^2 + im^2)` would. |
| `arg(z)` | Principal argument, `(-pi, pi]`. |
| `inverse(z)` / `normalized(z)` / `tryNormalized(z)` | |
| `exp(z)` / `log(z)` | `log` is the principal branch, discontinuous across the negative real axis. |
| `sqrt(z)` | Principal root (non-negative real part), built from the modulus rather than half the argument for precision near the real axis. |
| `pow(z, exponent)` | Complex or real exponent; `exp(exponent * log(z))`. |

Division uses Smith's formula (divides through by whichever part of the
denominator is larger) so it never overflows for a divisor whose quotient
is representable.

## `Math/Dual.hpp`

Forward-mode automatic differentiation: a value paired with its exact
derivative. This is the mechanism behind `Vector3<Dual<double>>` and behind
`gradient`/`jacobian`/`hessian` below.

```cpp
template <Numeric T> struct Dual {
    T value{}, derivative{};

    constexpr Dual() = default;
    template <class U> constexpr Dual(U real);           // implicit; derivative 0
    constexpr Dual(T real, T tangent);

    static constexpr Dual variable(T at) noexcept;        // seed: d/d(itself) = 1
    static constexpr Dual constant(T at) noexcept;        // derivative 0
    // += -= *= /=, + - * / (product/quotient rule), ==, < <= > >= (value only)
};
using DualD = Dual<double>;
using Dual2D = Dual<Dual<double>>;   // second derivatives
```

`sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `abs`, `hypot`, `pow` are
all overloaded for `Dual<T>`, each one line of the chain rule.

| Function | Description |
| --- | --- |
| `identical(a, b)` | Component-for-component equality (value **and** derivative); `operator==` compares value only. |
| `valueOf(x)` | Strips every layer of `Dual`, down to the underlying scalar. |
| `valueAndDerivative(f, at)` | `f(x)` and `f'(x)` in one evaluation, as a `std::pair`. |
| `derivative(f, at)` | Just `f'(x)`. |
| `secondDerivative(f, at)` | `f''(x)`, by nesting `Dual<Dual<T>>` seeded at both levels. |

```cpp
const double slope = ysq::derivative([](auto x) { return exp(sin(x)); }, 1.3);  // exact
const auto [value, tangent] = ysq::valueAndDerivative([](auto x) { return x * x; }, 3.0);
```

**Comparisons look at the value only.** A branch on `x < 0`, a `clamp`, a
componentwise `min` all ask about magnitude and should not change answer
because a derivative differs; `identical` is the spelling when the tangent
matters too.

## `Math/Tensor.hpp`

A dense tensor of fixed rank, every index running over the same dimension:
`Tensor<T, Rank, Dim>`. In relativity every index is a spacetime index, so
`Dim` is 4 throughout: `MetricTensor<T> = Tensor<T, 2, 4>`,
`ChristoffelSymbols<T> = Tensor<T, 3, 4>`, `RiemannTensor<T> = Tensor<T, 4,
4>`. Storage is row-major (last index varies fastest). Raising/lowering
indices needs a metric and lives in `Physics/Spacetime`; what's here is
index algebra that doesn't care what the numbers mean.

```cpp
template <Numeric T, std::size_t Rank, std::size_t Dim> struct Tensor {
    static constexpr std::size_t kComponents = /* Dim^Rank */;
    std::array<T, kComponents> components{};

    static constexpr std::size_t rank() / dimension() / size() noexcept;
    template <class... Indices> T& operator()(Indices... indices) noexcept;  // exactly Rank of them
    T& operator[](std::size_t flat) noexcept;         // flat storage
    static constexpr Tensor zero/filled(T) noexcept;
    static constexpr Tensor delta() noexcept requires(Rank == 2);  // Kronecker delta
    // += -= *= /=, + - * /, ==
};
```

| Function | Description |
| --- | --- |
| `outerProduct(a, b)` | Tensor product; ranks add. |
| `contract<I, J>(a, b)` | Contracts index `I` of `a` against index `J` of `b`, summing the shared value. `I`/`J` are template parameters, so contracting an index the tensor doesn't have is a compile error. |
| `traceOver<I, J>(t)` | Contracts two indices of the **same** tensor, dropping rank by two: how Ricci comes from Riemann. |
| `trace(t)` | The rank-2 case, as a scalar. |
| `transposeIndices<I, J>(t)` | Swaps two indices. |
| `symmetrize<I, J>(t)` / `antisymmetrize<I, J>(t)` | The part unchanged / sign-flipped under swapping `I`, `J`. |
| `toTensor(v)` / `toVector3/4(t)` / `toMatrix3/4(t)` | Conversions with `Vector3/4` and `Matrix3/4` at matching `Dim`. |

```cpp
using Metric = ysq::MetricTensor<double>;  // Tensor<double, 2, 4>
const auto ricci = ysq::traceOver<0, 2>(riemann);
```

## `Math/Statistics.hpp`

Summaries over a run, with the two pieces that matter for a long
simulation: compensated summation and an online accumulator. Insufficient
data yields **NaN**, not zero or an exception: a variance of "no samples"
is not zero. A NaN anywhere in the input comes back as NaN from every
statistic, including `minimum` and `median`.

```cpp
template <FloatRange R> auto sum(const R&) -> value_type;       // Neumaier-compensated
template <FloatRange R> auto naiveSum(const R&) -> value_type;  // for comparison only
template <FloatRange R> auto mean(const R&) -> value_type;
template <FloatRange R> auto variance(const R&) -> value_type;         // divides by n
template <FloatRange R> auto sampleVariance(const R&) -> value_type;   // divides by n-1
template <FloatRange R> auto standardDeviation(const R&) -> value_type;
template <FloatRange R> auto sampleStandardDeviation(const R&) -> value_type;
template <FloatRange R> auto minimum/maximum/range(const R&) -> value_type;
template <FloatRange R> auto quantile(const R&, T p) -> value_type;    // linear interpolation, R's type-7 definition
template <FloatRange R> auto median(const R&) -> value_type;
template <FloatRange R> auto covariance(const R&, const R&) -> value_type;   // divides by n
template <FloatRange R> auto correlation(const R&, const R&) -> value_type;  // Pearson's, [-1, 1]

struct LinearFit { T slope, intercept, rSquared; };
template <FloatRange R> LinearFit<T> linearFit(const R& x, const R& y);

std::vector<std::size_t> histogram(const R&, binCount, low, high);  // NaN dropped, not binned
```

`RunningStatistics<T>`: Welford's online algorithm, mean, variance, min,
max in one pass, without keeping the samples.

```cpp
class RunningStatistics<T> {
public:
    void add(T value);
    void merge(const RunningStatistics& other);   // Chan's parallel combination
    void reset();
    std::size_t count() const noexcept;
    T mean/variance/sampleVariance/standardDeviation/sampleStandardDeviation() const;
    T minimum/maximum() const noexcept;
    T range() const noexcept;   // peak-to-peak; is the conserved-quantity error bounded or growing?
};
```

```cpp
ysq::RunningStatistics<double> energy;
// each step:
energy.add(currentEnergy);
// later:
ysq::logging::info("energy range over run: {}", energy.range());
```

## `Math/Interpolation.hpp`

Lerp up to a natural cubic spline. Curve functions are generic in the value
type (`Vector3` or a plain scalar) and take the parameter separately.

```cpp
template <Numeric T> constexpr T lerp(T a, T b, T t) noexcept;
template <Numeric T> constexpr T inverseLerp(T a, T b, T value) noexcept;  // undefined for a == b
template <Numeric T> constexpr T remap(T value, fromLow, fromHigh, toLow, toHigh) noexcept;
template <std::floating_point T> constexpr T smoothstep(edgeLow, edgeHigh, at) noexcept;
template <std::floating_point T> constexpr T smootherstep(edgeLow, edgeHigh, at) noexcept;

template <class V, Numeric T> constexpr V bilinear(v00, v10, v01, v11, tx, ty) noexcept;
template <class V, Numeric T> constexpr V trilinear(/* 8 corners */, tx, ty, tz) noexcept;
template <class V, Numeric T> constexpr V cubicHermite(p0, m0, p1, m1, t) noexcept;
template <class V, Numeric T> constexpr V catmullRom(p0, p1, p2, p3, t) noexcept;  // passes through p1, p2
template <class V, Numeric T> constexpr V cubicBezier(p0, p1, p2, p3, t) noexcept; // passes through p0, p3 only

template <class R> std::optional<T> interpolateTable(xRange, yRange, at);  // piecewise linear; xs strictly increasing
```

| Function | Description |
| --- | --- |
| `smoothstep`/`smootherstep` | Ease a clamped, normalized parameter; `smootherstep`'s second derivative also vanishes at the ends. |
| `catmullRom` | Interpolates between `p1`/`p2` using `p0`/`p3` only to pick tangents; the usual choice for a path through measured positions. |
| `cubicBezier` | Passes through only the first and last control points; the middle two pull at it. |
| `interpolateTable` | Held **flat** outside the table (no extrapolation); `nullopt` if the ranges don't form a valid table of >= 2 points. |

`CubicSpline<T>`: a natural cubic spline through tabulated points, C2
continuous with the second derivative pinned to zero at both ends.

```cpp
class CubicSpline<T> {
public:
    static std::optional<CubicSpline> natural(std::span<const T> xs, std::span<const T> ys);
    // nullopt unless same length, >= 3 points, xs strictly increasing
    T operator()(T at) const;     // held flat outside the table
    T derivative(T at) const;     // zero outside the table
    std::size_t size() const noexcept;
    T lowerBound() / upperBound() const noexcept;
};
```

Built once (solves a tridiagonal system, O(n)), evaluated many times (binary
search plus a few multiplies).

## `Math/Calculus.hpp`

Differentiation and quadrature. **The exact and approximate forms are
deliberately named differently.** `gradient`/`jacobian`/`hessian` (via
`Dual`) are exact to the last few bits; `numericalGradient` and friends are
finite differences and lose roughly half the available digits regardless of
step size. The plain name always goes to the better method.

```cpp
template <class F, class V> V gradient(F&& f, const V& at);              // exact
template <class F, class V> Matrix jacobian(F&& f, const V& at);          // exact
template <class F, class V> Matrix hessian(F&& f, const V& at);           // exact, nested duals

template <class F, class V> V numericalGradient(F&& f, const V& at);      // central differences
template <class F, class V> Matrix numericalJacobian(F&& f, const V& at);
template <class F, class V> Matrix numericalHessian(F&& f, const V& at);

template <class F, std::floating_point T> T forwardDifference(F&& f, T at, T step = onesidedStep(at));
template <class F, std::floating_point T> T backwardDifference(F&& f, T at, T step = onesidedStep(at));
template <class F, std::floating_point T> T centralDifference(F&& f, T at, T step = centralStep(at));
template <class F, std::floating_point T> T secondCentralDifference(F&& f, T at, T step = /* h^1/4 */);
template <class F, std::floating_point T> T richardsonDerivative(F&& f, T at, T step, std::size_t levels = 4);
```

Quadrature, all taking a function of a plain scalar:

```cpp
template <class F, std::floating_point T> T trapezoid(F&& f, lower, upper, intervals);       // order 2
template <class F, std::floating_point T> T simpson(F&& f, lower, upper, intervals);          // order 4, exact for cubics
template <class F, std::floating_point T> T adaptiveSimpson(F&& f, lower, upper, tolerance, maxDepth = 40);
template <class F, std::floating_point T> T romberg(F&& f, lower, upper, maxLevels = 12, tolerance = /* scaled to T */);
template <std::size_t N, class F, std::floating_point T> T gaussLegendre(F&& f, lower, upper);  // N in {2,3,4,5}; exact to degree 2N-1
```

```cpp
ysq::derivative([](auto x) { return exp(sin(x)); }, 1.3);      // exact
ysq::gradient([](const auto& v) { return dot(v, v); }, at);    // 2 * at, exactly
ysq::hessian(potential, at);                                    // nested duals

const double area = ysq::gaussLegendre<4>(integrand, 0.0, 1.0);
```

`adaptiveSimpson` refines only the panels that fail their local error
estimate, worth it when the integrand has a feature much narrower than its
domain; for a smooth integrand `simpson` at the same evaluation cost is at
least as good. `romberg` converges very fast for a smooth integrand and not
at all for one with a kink.

## `Math/CoordinateSystems.hpp`

Spherical, cylindrical, and polar coordinates, and the bases that go with
them. **The convention is the physics one**: `polar` is the angle down from
`+z` in `[0, pi]`; `azimuth` is the angle round from `+x` in `(-pi, pi]`
(mathematics texts routinely swap the two names).

```cpp
template <Numeric T> struct Spherical { T radius, polar, azimuth; };
template <Numeric T> struct Cylindrical { T radius, azimuth, height; };
template <Numeric T> struct Polar { T radius, angle; };

Vector3<T> toCartesian(const Spherical<T>&);
Spherical<T> toSpherical(const Vector3<T>&);
Vector3<T> toCartesian(const Cylindrical<T>&);
Cylindrical<T> toCylindrical(const Vector3<T>&);
Vector2<T> toCartesian(const Polar<T>&);
Polar<T> toPolar(const Vector2<T>&);

Matrix3<T> sphericalBasis(const Spherical<T>&);      // columns e_r, e_polar, e_azimuth
Matrix3<T> cylindricalBasis(const Cylindrical<T>&);  // columns e_radius, e_azimuth, e_height
Matrix2<T> polarBasis(const Polar<T>&);               // columns e_radius, e_angle

Matrix3<T> sphericalJacobian(const Spherical<T>&);    // d(x,y,z)/d(radius,polar,azimuth); det = r^2 sin(polar)
Matrix3<T> cylindricalJacobian(const Cylindrical<T>&); // det = radius

Vector3<T> sphericalComponentsToCartesian(at, components);
Vector3<T> cartesianComponentsToSpherical(at, vector);   // basis is orthonormal: a transpose, not an inversion
Vector3<T> cylindricalComponentsToCartesian(at, components);
Vector3<T> cartesianComponentsToCylindrical(at, vector);
```

Converting a *point* is only half of what a vector quantity needs: a
velocity or a field has components against a local basis that changes from
point to point, which is what the basis/Jacobian functions are for. Both
bases are degenerate on the axis (`sphericalBasis` also at the origin),
where they return an arbitrary but defined column rather than failing.

## `Math/Format.hpp`

`std::formatter` specializations for every value type above, so they drop
into `Core::logging` and `std::format` directly. A separate header from the
value types on purpose: `<format>` is heavy, and the vector headers sit on
the integration inner path.

```cpp
#include <Math/Format.hpp>

ysq::logging::info("v = {:.3f}", velocity);   // v = (1.000, 0.000, -9.810)
```

Vectors, quaternions, coordinate triples, and tensors print as `(a, b,
...)`; matrices print by row (`[[m00, m01], [m10, m11]]`) regardless of the
column-major storage; `Complex`/`Dual` print as a binomial (`1 - 2i`, `1 +
3eps`). The format spec is forwarded to the component type, so any spec
valid for a `double` applies to every component. Does **not** cover
`PhaseState`/`StateVector` from `Math/ODE.hpp`: printing an integrator
state is a debugging concern and its formatter lives with test support.

## `Math/Grid.hpp`

A uniform one-dimensional grid of cell values with ghost cells, the shared
storage `Physics/Electromagnetism`'s FDTD rung, `Physics/Fluids`' Eulerian
rung, and `Physics/Thermodynamics`' heat-equation rung all build on.
**Scope: one dimension**; a 3D solver is future work for each of those.

```cpp
template <Numeric T> class Grid1D {
public:
    Grid1D(std::size_t cellCount, double spacing, std::size_t ghostCells = 1);

    std::size_t cellCount() const noexcept;
    std::size_t ghostCells() const noexcept;
    double spacing() const noexcept;

    T& operator[](std::ptrdiff_t index) noexcept;
    // index 0 is the first interior cell; negative reaches the left ghost
    // region, >= cellCount() the right, both up to ghostCells() past the interior

    void applyPeriodicBoundary();
    // copies each side's interior edge into the opposite side's ghost cells
};
```

```cpp
ysq::Grid1D<double> field(100, 0.01);
field.applyPeriodicBoundary();
const double leftNeighbor = field[-1];  // reaches into the ghost region
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/math/algebra)
and let us know.
