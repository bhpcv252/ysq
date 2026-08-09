# Numerical methods

The tools that turn "I have an equation" into "I have a number": finding
where a function crosses zero, solving a system of equations too big for
`Matrix2`/`3`/`4`, finding the natural vibration modes of a physical
system, fitting a curve, and computing a spectrum with the FFT.

## The idea

A lot of physics doesn't hand you a closed-form answer. Where does a
projectile's trajectory cross the ground? There's no algebra formula for
that in general — you find it numerically, by narrowing in on the root of
an equation. What are the natural oscillation frequencies of a system of
coupled springs? That's an eigenvalue problem: you're not solving for a
single number, you're solving for the *whole set* of special directions
and rates a linear system wants to move in. What's the area under a curve
that has no elementary antiderivative? You approximate it by sampling the
curve cleverly and adding up the pieces (that one's actually over in
[Vectors, matrices, and exact derivatives](algebra.md), alongside
`Calculus.hpp`'s quadrature rules).

`Math`'s numerical methods are this project's answer to all of that: a
toolbox of general-purpose algorithms that don't know or care whether
they're being used for a physics simulation, an economics model, or a
puzzle. That's the whole point of keeping them in `Math` rather than
`Physics` — a root finder doesn't know what root it's finding.

## What YSQ gives you

| Header | What it's for |
| --- | --- |
| `RootFinding.hpp` | Newton-Raphson, secant, and bisection: where a function crosses zero |
| `LinearSolve.hpp` | `MatrixN`/`VectorN` (dynamically sized, unlike `Matrix2/3/4`), LU and Cholesky solves |
| `Eigen.hpp` | Eigenvalues and eigenvectors (symmetric and general), singular value decomposition, QR decomposition |
| `SpecialFunctions.hpp` | The error function, gamma function, associated Legendre polynomials, Bessel functions |
| `Polynomial.hpp` | Evaluation, differentiation, and closed-form roots up to degree 4 |
| `Random.hpp` | Uniform/normal/Poisson sampling, and Monte Carlo integration |
| `Optimization.hpp` | Finding the minimum of a scalar field, with or without a gradient |
| `FFT.hpp` | The Fast Fourier Transform, in one dimension and in three |

## Using it

Finding a root, no calculus required:

```cpp
#include <Math/RootFinding.hpp>

const auto shifted = [](double x) { return x * x - 2.0; };
const double root = ysq::newtonRaphson(shifted, /*initialGuess=*/1.0);
// root is sqrt(2), to machine precision
```

Finding the natural vibration modes of a system — the eigenvalues of its
stiffness matrix, in this toy case a symmetric 3x3:

```cpp
#include <Math/Eigen.hpp>
#include <Math/LinearSolve.hpp>

ysq::MatrixN<double> stiffness(3, 3);
// ... fill in the coupling between three masses ...
const ysq::EigenDecomposition<double> modes = ysq::jacobiEigenSymmetric(stiffness);
// modes.eigenvalues: the squared natural frequencies, ascending
// modes.eigenvectors: column i is the i-th mode shape
```

**Why a *general* eigensolver exists alongside the symmetric one.** Every
consumer this engine's own physics has needed so far (an inertia tensor, a
covariance matrix) is symmetric, which is the one case where eigenvalues
are guaranteed real and eigenvectors guaranteed orthogonal — simpler and
more numerically robust, which is why `jacobiEigenSymmetric` is a
different function rather than a special case of a general one.
`generalEigenvalues` is what's there for a matrix that isn't symmetric (a
state-space stability analysis, say), where eigenvalues can come out as
complex-conjugate pairs.

## Go deeper

[docs/api/math/numerics.md](../api/math/numerics.md) has every signature:
every default parameter, every precondition, and which closed form each
polynomial-root function actually implements.

[src/Math/README.md](../../src/Math/README.md) has the full derivation for
each method — the numerically stable rotation angle `jacobiEigenSymmetric`
uses and why the textbook formula loses precision, why `qrDecompose`
reflects toward `-sign(a) * norm` rather than `+norm`, the closed forms
behind `Polynomial.hpp`'s cubic and quartic roots including the
trigonometric case Cardano's formula needs complex intermediates for —
and the cross-checks each one is validated against (a companion matrix's
own roots, a Gram matrix's relationship between SVD and symmetric
eigendecomposition, the direct `O(n^2)` DFT sum for the FFT).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+math/numerics)
and let us know.
