# Vectors, matrices, and exact derivatives

The computational vocabulary everything else in YSQ is built from: what a
position is, what a rotation is, and one genuinely surprising trick for
getting an exact derivative without doing calculus by hand.

## The idea

A **vector** is a position, a direction, or a velocity: a handful of numbers
(two, three, or four of them here) that together have both a size and a
direction. A **matrix** is something that transforms vectors: multiply a
vector by the right matrix and you rotate it, scale it, or move it. That's
almost the whole of classical linear algebra as YSQ uses it, and it's what
`Vector2/3/4` and `Matrix2/3/4` are for.

A **quaternion** is a more compact, better-behaved way to represent a
rotation than a matrix (no risk of a rotation slowly drifting out of
orthogonality after repeated multiplication, and no gimbal lock). A
**tensor** generalizes vectors and matrices to more indices than "one" or
"two", which is exactly what's needed once you get to a spacetime metric in
`Physics/Spacetime`, a rank-2 tensor with four indices rather than three.

The one idea here worth real explanation is the **dual number**, because
it answers a question that looks like it shouldn't have a clean answer: how
do you compute the *exact* derivative of a function, in code, without
either doing the calculus by hand first or settling for the small error a
finite-difference approximation (`(f(x+h) - f(x)) / h`) always carries?

The trick: extend every number to a pair, `a + b*epsilon`, where `epsilon`
is defined to satisfy `epsilon^2 = 0` (it isn't zero itself, it just squares
to zero, the same way `i^2 = -1` defines a complex number). Multiply two of
these out:

```
(a + b*epsilon)(c + d*epsilon) = ac + (ad + bc)*epsilon
```

and the `epsilon` part is exactly the product rule. Define every arithmetic
operation, every `sin`, `sqrt`, `exp`, the same way, each following its own
derivative rule, and evaluating an ordinary function on `Dual<T>` instead of
`T` produces its own exact derivative as a side effect, to the last bit of
precision `T` can hold. No finite step size, no calculus by hand: seed the
input with a derivative of `1`, run the function, read the derivative back
off the output. `Physics/Spacetime` uses exactly this to get exact
Christoffel symbols; see [docs/physics/spacetime.md](../physics/spacetime.md).

## What YSQ gives you

| Type | What it's for |
| --- | --- |
| `Vector2<T>`, `Vector3<T>`, `Vector4<T>` | Position, direction, velocity |
| `Matrix2<T>`, `Matrix3<T>`, `Matrix4<T>` | Rotation, scale, projection; a pivoting linear solve |
| `Quaternion<T>` | Rotations, without a matrix's drift or gimbal lock |
| `Complex<T>` | Complex numbers |
| `Dual<T>` | Exact derivatives, per the idea above |
| `Tensor<T, Rank, Dim>` | Fixed rank and dimension, with index algebra, for `Physics/Spacetime`'s metrics |
| `Statistics.hpp` | Mean, variance, and an online accumulator, both compensated against rounding error |
| `Interpolation.hpp` | Lerp, and natural cubic splines |
| `Calculus.hpp` | `derivative`/`gradient`/`jacobian`/`hessian` (exact, via `Dual`) and quadrature rules |
| `CoordinateSystems.hpp` | Spherical, cylindrical, and polar conversions, with their local bases |
| `Scalar.hpp` | The `Numeric` concept every type above is templated on, plus `clamp`/`approxEqual`/constants |
| `Format.hpp` | `std::formatter` for every type above, so they drop into `Core`'s logger and `std::format` |
| `Grid.hpp` | A uniform 1D grid with ghost cells, the shared storage `Physics`' PDE rungs (fluids, electromagnetism, thermodynamics) build on |

Every one of these is templated on its scalar type (`Vector3<double>` versus
`Vector3<float>`, aliased `Vec3`/`Vec3f`), and that's not just about
choosing precision: it's what lets `Vector3<Dual<double>>` exist at all,
composing the derivative trick above with ordinary vector math for free.

## Using it

A rotation, built from an axis and an angle, applied to a vector:

```cpp
#include <Math/Quaternion.hpp>
#include <Math/Vector3.hpp>

#include <numbers>

const ysq::Quaternion<double> spin =
    ysq::Quaternion<double>::fromAxisAngle(ysq::Vec3{0.0, 0.0, 1.0},
                                            std::numbers::pi / 2.0);
const ysq::Vec3 rotated = ysq::rotate(spin, ysq::Vec3{1.0, 0.0, 0.0});
// rotated is approximately (0, 1, 0): a quarter turn about the z axis
```

And an exact derivative, no finite difference in sight:

```cpp
#include <Math/Dual.hpp>

#include <cmath>

const double slope =
    ysq::derivative([](auto x) { return exp(sin(x)); }, 1.3);
// exact to the last bit double can hold, not approximate
```

## Go deeper

[docs/api/math/algebra.md](../api/math/algebra.md) has every signature in
this file: the shared vector/matrix free-function table, every named
constructor, and the exact failure behavior of `tryNormalized`/`tryInverse`/
`solve`.

[src/Math/README.md](../../src/Math/README.md) has the full interface for
every type above, the conventions worth knowing before they cost you a
plausible-looking wrong answer (column-major matrices, the physics
convention for spherical coordinates, quaternions stored scalar-first), and
a **Derivations** section covering the full dual-number chain-rule table,
Gauss-Legendre quadrature, and why some routines (angles from `atan2`
instead of `acos`, compensated summation) are written the longer way on
purpose.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+math/algebra)
and let us know.
