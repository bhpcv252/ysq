# Physics/Spacetime API reference

The metric concept, Christoffel symbols, the four spacetimes, and the
geodesic solver. Start with
[docs/physics/spacetime.md](../../physics/spacetime.md) for the ideas;
[src/Physics/README.md](../../../src/Physics/README.md) has every line
element's derivation and the fixed conventions in full.

**Convention, fixed across every metric here.** Signature `(-,+,+,+)`.
Four-position components are `(x0, x1, x2, x3)` with `x0 = c*t`, so every
component is in metres and no metric needs its own unit conversion. Spatial
coordinates `x1..x3` (Cartesian vs. spherical) are each metric's own choice
of chart.

## `Physics/Spacetime/Metric.hpp`

The metric concept, the inner product, causal character, and Christoffel
symbols, computed exactly via dual numbers, not finite differences.

```cpp
template <class M>
concept SpacetimeMetric = requires(const M& metric, const Vector4<double>& at,
                                   const Vector4<Dual<double>>& dualAt) {
    { metric.components(at) } -> std::same_as<MetricTensor<double>>;
    { metric.components(dualAt) } -> std::same_as<MetricTensor<Dual<double>>>;
};

double metricProduct(const M& metric, const Vector4<double>& at,
                     const Vector4<double>& u, const Vector4<double>& v);
// g_mu_nu(at) u^mu v^nu; negative timelike, positive spacelike, zero null

bool isTimelike(const M&, const Vector4<double>& at, const Vector4<double>& u);
bool isSpacelike(const M&, const Vector4<double>& at, const Vector4<double>& u);
bool isNull(const M&, const Vector4<double>& at, const Vector4<double>& u,
           double tolerance = 1e-9);   // tolerance test, not exact-zero

ChristoffelSymbols<double> christoffelSymbols(const M& metric, const Vector4<double>& at);
```

A metric is a **concept, not a base class**: anything with a `components`
method callable at both `double` and `Dual<double>` qualifies, which is what
lets `christoffelSymbols` differentiate any of them the same way, and what
makes adding a fifth metric a matter of writing one method.
`christoffelSymbols` seeds a `Dual<double>` in each of the four coordinate
directions and reads the derivative back exactly (no finite-difference
truncation), the same automatic-differentiation trick as
`Math::gradient`/`jacobian`, reimplemented here because `Tensor` isn't one
of the shapes those functions handle.

```cpp
const ysq::MetricTensor<double> g = metric.components(at);
const ysq::ChristoffelSymbols<double> gamma = ysq::christoffelSymbols(metric, at);
const bool photon = ysq::isNull(metric, at, fourVelocity);
```

## `Physics/Spacetime/Minkowski.hpp`

Flat spacetime, no gravity: special relativity only. The base case every
other metric reduces to somewhere; Christoffel symbols are exactly zero
everywhere.

```cpp
struct Minkowski {
    template <Numeric T> MetricTensor<T> components(const Vector4<T>&) const;
    // diag(-1, 1, 1, 1); does not depend on position at all
};
```

## `Physics/Spacetime/Schwarzschild.hpp`

A non-rotating mass: the unique spherically symmetric vacuum solution.
Chart `(T, r, polar, azimuth)`, the same physics convention
`Math/CoordinateSystems.hpp` uses.

```cpp
class Schwarzschild {
public:
    explicit Schwarzschild(GravitationalParameter mass);   // GM, not mass in kilograms
    double schwarzschildRadius() const noexcept;             // r_s = 2GM/c^2, computed once
    template <Numeric T> MetricTensor<T> components(const Vector4<T>& at) const;
};
```

```
ds^2 = -(1 - r_s/r) dT^2 + dr^2/(1 - r_s/r) + r^2 dpolar^2 + r^2 sin^2(polar) dazimuth^2
```

`mass` takes the source's gravitational parameter GM (as `Physics/Gravity`
prefers, for the same reason: GM carries none of `G`'s measurement
uncertainty). Components diverge at `r = r_s` (a coordinate artifact of this
chart, not a physical singularity) and at `r = 0` (the genuine one);
neither is guarded against; a geodesic that reaches either has left the
regime this chart can describe.

```cpp
const ysq::Schwarzschild schwarzschild{gravitationalParameter};
const ysq::Vec4 at{0.0, r, polar, azimuth};   // (c*t, r, polar, azimuth)
```

## `Physics/Spacetime/Kerr.hpp`

A rotating mass: the Boyer-Lindquist form of the unique stationary,
axisymmetric vacuum solution. Reduces exactly to `Schwarzschild` at
`spin = 0`.

```cpp
class Kerr {
public:
    explicit Kerr(GravitationalParameter mass, Length spin);   // spin: a = J/(Mc), a length
    double schwarzschildRadius() const noexcept;
    double spin() const noexcept;
    template <Numeric T> MetricTensor<T> components(const Vector4<T>& at) const;
};
```

```
sigma = r^2 + a^2 cos^2(polar),  delta = r^2 - r_s r + a^2

ds^2 = -(1 - r_s r/sigma) dT^2 - (2 r_s r a sin^2(polar)/sigma) dT dazimuth
       + (sigma/delta) dr^2 + sigma dpolar^2
       + (r^2 + a^2 + r_s r a^2 sin^2(polar)/sigma) sin^2(polar) dazimuth^2
```

The `g_T,azimuth` cross term **is** frame dragging: close enough to the
horizon, an observer at fixed `r` and `polar` cannot hold `azimuth` fixed
and still be on a timelike worldline, since the term forces `dT` and
`dazimuth` to mix.

## `Physics/Spacetime/FLRW.hpp`

An expanding, homogeneous, isotropic universe: comoving coordinates,
`ds^2 = -dT^2 + a(T)^2 [dr^2/(1 - k r^2) + r^2 dpolar^2 + r^2 sin^2(polar) dazimuth^2]`.
Three ready-made single-component analytic solutions for `a(T)`, each its
own class (a general Friedmann-equation solve for a mixed-component universe
is not implemented):

```cpp
class MatterDominatedFLRW {      // dust, w = 0: a(T) = (T/T0)^(2/3)
public:
    MatterDominatedFLRW(double referenceTime, double curvature);   // curvature k: +1, 0, -1
    template <Numeric T> MetricTensor<T> components(const Vector4<T>& at) const;
};

class RadiationDominatedFLRW {   // w = 1/3: a(T) = (T/T0)^(1/2)
public:
    RadiationDominatedFLRW(double referenceTime, double curvature);
    template <Numeric T> MetricTensor<T> components(const Vector4<T>& at) const;
};

class LambdaDominatedFLRW {      // cosmological constant alone (de Sitter): a(T) = exp(H(T - T0))
public:
    LambdaDominatedFLRW(double hubbleRate, double referenceTime, double curvature);
    // hubbleRate H is in inverse-length units, since T = c*t
    template <Numeric T> MetricTensor<T> components(const Vector4<T>& at) const;
};
```

`referenceTime` `T0` normalizes `a(T0) = 1`. `curvature` `k` is `+1`
(closed), `0` (flat), or `-1` (open).

## `Physics/Spacetime/Geodesic.hpp`

The solver: one system serves both massive particles and light.

```cpp
template <SpacetimeMetric M>
auto geodesicSystem(M metric);
// returns an OdeSystem<PhaseState<Vector4<double>>> for Rk4Stepper / DormandPrince54Stepper
```

```
d^2 x^mu / dlambda^2 + Gamma^mu_ab (dx^a/dlambda)(dx^b/dlambda) = 0
```

**Cannot use the symplectic steppers `Physics/Gravity` does**: those take
an `AccelerationField` (acceleration depends on position alone), but the
`Gamma^mu_ab v^a v^b` term here is quadratic in *velocity*, so
`geodesicSystem` builds a plain `OdeSystem` for `Rk4Stepper` or the adaptive
Dormand-Prince stepper instead. **One equation serves both timelike and
null geodesics**: which one you get depends entirely on the initial
four-velocity's normalization (`metricProduct(metric, at, u, u)` equal to
`-c^2` for a massive particle's proper time, or `0` for light), nothing else
in the equation distinguishes them.

```cpp
const auto system = ysq::geodesicSystem(schwarzschild);
ysq::Rk4Stepper<ysq::PhaseState<ysq::Vec4>> stepper;
stepper.step(system, lambda, state, step, next);
```

The affine parameter `lambda` is proper time for a timelike geodesic and has
no invariant meaning for a null one, the ordinary situation in relativity,
not a limitation of this solver.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/spacetime)
and let us know.
