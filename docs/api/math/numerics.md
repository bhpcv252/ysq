# Math API reference: numerical methods

Root-finding, linear algebra beyond fixed-size matrices, eigendecomposition,
special functions, closed-form polynomial roots, randomness, gradient-based
optimization, and the FFT. Start with
[docs/math/numerics.md](../../math/numerics.md) for the ideas; this page is
the lookup table. [src/Math/README.md](../../../src/Math/README.md) has
every derivation, coefficient table, and validation in full.

## `Math/RootFinding.hpp`

Where a scalar function crosses zero, three ways, trading convergence speed
for how little each demands of the caller.

```cpp
template <class F, class FPrime, std::floating_point T>
T newtonRaphson(F&& f, FPrime&& fPrime, T initialGuess,
               T tolerance = /* 100 epsilons */, int maxIterations = 50);

template <class F, std::floating_point T>
T newtonRaphson(F&& f, T initialGuess, T tolerance = /* 100 epsilons */,
                int maxIterations = 50);
// derivative-free: fPrime approximated by a central difference each step

template <class F, std::floating_point T>
T secant(F&& f, T x0, T x1, T tolerance = /* 100 epsilons */, int maxIterations = 50);

template <class F, std::floating_point T>
T bisection(F&& f, T lower, T upper, T tolerance = /* 100 epsilons */,
           int maxIterations = 100);
```

| Function | Description |
| --- | --- |
| `newtonRaphson` (with `fPrime`) | Quadratic convergence, but needs a derivative and a starting guess close enough that the tangent line actually points toward the root. |
| `newtonRaphson` (without `fPrime`) | The same iteration with the derivative estimated by a central difference: two evaluations of `f` per step instead of one, no second callable needed. |
| `secant` | Approximates the derivative from the last two iterates instead of a formula; two starting guesses, not required to bracket the root. |
| `bisection` | Needs only a bracket where `f` changes sign (a precondition, not checked). Converges linearly but is guaranteed to land inside that bracket, unlike the other two. |

Every one of these returns its last iterate regardless of whether it
converged before `maxIterations`, on purpose: a well-posed problem
converges long before the cap, and one that doesn't is better served by
inspecting the returned value than by an exception it would just have to
catch.

```cpp
const auto shifted = [](double x) { return x * x - 2.0; };
const double root = ysq::newtonRaphson(shifted, 1.0);   // sqrt(2), to machine precision
```

## `Math/LinearSolve.hpp`

Dense linear algebra for a system whose size is only known at run time — a
normal-equations fit, a constraint system with one row per contact — where
`Matrix2`/`3`/`4` are the wrong shape entirely.

```cpp
template <std::floating_point T> class VectorN {
public:
    explicit VectorN(std::size_t size, T fill = T{0});
    VectorN(std::initializer_list<T> values);
    std::size_t size() const noexcept;
    T& operator[](std::size_t i);
    // += -= *= (scalar), + - * (vector-vector, vector-scalar)
};
template <std::floating_point T> T dot(const VectorN<T>&, const VectorN<T>&);
template <std::floating_point T> T norm(const VectorN<T>&);

template <std::floating_point T> class MatrixN {
public:
    MatrixN(std::size_t rows, std::size_t cols, T fill = T{0});
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    T& operator()(std::size_t row, std::size_t col);
    static MatrixN identity(std::size_t n);
    // += -= *= (scalar and matrix), + - *
};
template <std::floating_point T> MatrixN<T> transpose(const MatrixN<T>&);
template <std::floating_point T> VectorN<T> operator*(const MatrixN<T>&, const VectorN<T>&);
template <std::floating_point T> MatrixN<T> operator*(const MatrixN<T>&, const MatrixN<T>&);

template <std::floating_point T> struct LuDecomposition { MatrixN<T> lu; std::vector<std::size_t> pivot; };
template <std::floating_point T> std::optional<LuDecomposition<T>> luDecompose(MatrixN<T> a);
template <std::floating_point T> VectorN<T> luSolve(const LuDecomposition<T>&, const VectorN<T>& b);
template <std::floating_point T> std::optional<VectorN<T>> solve(const MatrixN<T>& a, const VectorN<T>& b);

template <std::floating_point T> std::optional<MatrixN<T>> choleskyDecompose(const MatrixN<T>& a);
template <std::floating_point T> std::optional<VectorN<T>> choleskySolve(const MatrixN<T>& a, const VectorN<T>& b);
```

| Function | Description |
| --- | --- |
| `luDecompose`/`luSolve`/`solve` | Gaussian elimination with partial pivoting. `solve` is the one-shot convenience; use `luDecompose` directly when solving against many right-hand sides, to pay the `O(n^3)` factorization once. `nullopt` if singular. |
| `choleskyDecompose`/`choleskySolve` | Symmetric positive-definite only, about half the work of LU, no pivoting needed. `nullopt` the moment a diagonal entry would be non-positive — the same condition as "not actually positive-definite." Only the lower triangle of `a` is read. |

```cpp
ysq::MatrixN<double> a(3, 3);
// ... fill a ...
const ysq::VectorN<double> b{1.0, 2.0, 3.0};
const std::optional<ysq::VectorN<double>> x = ysq::solve(a, b);
```

## `Math/Eigen.hpp`

Symmetric and general eigendecomposition, singular value decomposition, and
QR decomposition.

```cpp
template <std::floating_point T> struct EigenDecomposition {
    VectorN<T> eigenvalues;     // ascending
    MatrixN<T> eigenvectors;    // column i is the unit eigenvector for eigenvalues[i]
};
template <std::floating_point T>
EigenDecomposition<T> jacobiEigenSymmetric(MatrixN<T> a, int maxSweeps = 100, T tolerance = T{0});
// only the lower triangle of a is read
```

| Function | Description |
| --- | --- |
| `jacobiEigenSymmetric` | The cyclic Jacobi method: real eigenvalues, orthonormal eigenvectors, always — the one case an inertia tensor, a covariance matrix, or a Gram matrix always is. |

```cpp
template <std::floating_point T> struct QrDecomposition { MatrixN<T> q; MatrixN<T> r; };
template <std::floating_point T> QrDecomposition<T> qrDecompose(MatrixN<T> a);   // a.rows() >= a.cols()

template <std::floating_point T> struct SchurDecomposition { MatrixN<T> q; MatrixN<T> t; };
template <std::floating_point T> SchurDecomposition<T> realSchur(MatrixN<T> a, int maxIterations = 500);
template <std::floating_point T> std::vector<Complex<T>> eigenvaluesFromSchur(const MatrixN<T>& t);
template <std::floating_point T>
std::vector<Complex<T>> generalEigenvalues(const MatrixN<T>& a, int maxIterations = 500);

template <std::floating_point T> struct SvdDecomposition {
    MatrixN<T> u; VectorN<T> singularValues; MatrixN<T> v;   // a = u * diag(singularValues) * v^T
};
template <std::floating_point T> SvdDecomposition<T> svd(MatrixN<T> a, int maxSweeps = 60);
// a.rows() >= a.cols(); for fewer rows than columns, svd(transpose(a)) and swap u/v back
```

| Function | Description |
| --- | --- |
| `qrDecompose` | Householder reflections, any `a` with at least as many rows as columns. |
| `realSchur` | Hessenberg reduction plus the shifted QR algorithm: `q` orthogonal, `t` quasi-upper-triangular (1x1 blocks are real eigenvalues, 2x2 blocks are complex-conjugate pairs, since a real matrix can't always be triangularized by a real orthogonal transform). |
| `generalEigenvalues` | `eigenvaluesFromSchur(realSchur(a).t)` in one call. **Eigenvectors are not returned**: a solid eigenvector for a complex eigenvalue needs a complex linear solve, which `LinearSolve.hpp` doesn't have yet. |
| `svd` | One-sided Jacobi (Hestenes): the same plane-rotation idea as `jacobiEigenSymmetric`, aimed at orthogonalizing columns instead of zeroing an off-diagonal entry. |

```cpp
const auto eigen = ysq::jacobiEigenSymmetric(inertiaTensorAsMatrixN);
const std::vector<ysq::Complex<double>> spectrum = ysq::generalEigenvalues(stateMatrix);
const auto decomposition = ysq::svd(designMatrix);   // least-squares, PCA, and the like
```

## `Math/SpecialFunctions.hpp`

The error function, gamma function, associated Legendre polynomials, and
Bessel functions — special functions with no home elsewhere in `Math`.

```cpp
template <std::floating_point T> T erf(T x);
template <std::floating_point T> T erfc(T x);         // 1 - erf(x), computed directly
template <std::floating_point T> T gamma(T x);
template <std::floating_point T> T logGamma(T x);      // log(|gamma(x)|), doesn't overflow where gamma(x) would

template <std::floating_point T> T legendreP(unsigned n, unsigned m, T x);   // 0 <= m <= n, -1 <= x <= 1
template <std::floating_point T> T legendreP(unsigned n, T x);               // m = 0

template <std::floating_point T> T besselJ0(T x);
template <std::floating_point T> T besselJ1(T x);
template <std::floating_point T> T besselJ(unsigned n, T x);
template <std::floating_point T> T besselY0(T x);   // x > 0
template <std::floating_point T> T besselY1(T x);   // x > 0
template <std::floating_point T> T besselY(unsigned n, T x);   // x > 0
```

| Function | Description |
| --- | --- |
| `erf`/`erfc`/`gamma`/`logGamma` | Thin ADL-dispatched wrappers over `<cmath>`'s `std::erf`/`std::erfc`/`std::tgamma`/`std::lgamma`. |
| `legendreP` | The associated Legendre polynomial, Condon-Shortley phase included. Computed by upward recurrence: libc++ has no `std::assoc_legendre`. |
| `besselJ0`/`besselJ1`/`besselJ` | Bessel function of the first kind. Orders 0/1 via Abramowitz & Stegun rational approximations (no `std::cyl_bessel_j` in libc++ either); order `n >= 2` via Miller's algorithm (stable downward recurrence, rescaled by the sum rule `J_0 + 2 sum J_{2k} = 1`). |
| `besselY0`/`besselY1`/`besselY` | Bessel function of the second kind (logarithmically singular at the origin). Orders 0/1 via the same style of rational approximation; order `n >= 2` via plain upward recurrence, stable in this direction (the opposite of `J`). |

```cpp
const double probability = 0.5 * (1.0 + ysq::erf(x / std::sqrt(2.0)));   // standard normal CDF
const double firstZero = 2.4048255577;
ysq::besselJ0(firstZero);   // approximately 0
```

## `Math/Polynomial.hpp`

Evaluation, differentiation, and roots — closed-form through degree 4 (the
highest with a general formula in radicals), numeric deflation above that.

```cpp
template <std::floating_point T> class Polynomial {
public:
    explicit Polynomial(std::vector<T> coefficients);   // ascending: coefficients()[i] is x^i
    Polynomial(std::initializer_list<T> coefficients);
    const std::vector<T>& coefficients() const noexcept;
    std::size_t degree() const noexcept;                 // coefficients().size() - 1, not trimmed
    T operator()(T x) const;                              // Horner's method
    Polynomial derivative() const;
};

template <std::floating_point T> std::optional<T> linearRealRoot(T a, T b);
template <std::floating_point T> std::vector<T> quadraticRealRoots(T a, T b, T c);
template <std::floating_point T> std::vector<T> cubicRealRoots(T a, T b, T c, T d);
template <std::floating_point T> std::vector<T> quarticRealRoots(T a, T b, T c, T d, T e);
template <std::floating_point T> std::vector<T> realRoots(const Polynomial<T>& p);
```

| Function | Description |
| --- | --- |
| `quadraticRealRoots` | Citardauq form (multiply by the conjugate), avoiding the catastrophic cancellation the textbook quadratic formula suffers when one root is much smaller than the other. |
| `cubicRealRoots` | Cardano's formula for one real root plus a complex-conjugate pair; the trigonometric substitution for three distinct real roots (the "casus irreducibilis," where Cardano's formula needs complex intermediates to produce a real answer). |
| `quarticRealRoots` | Biquadratic special case, or the resolvent-cubic factorization into two quadratics in general. |
| `realRoots(p)` | Dispatches to the closed forms up to degree 4; above that, deflates one numeric root at a time (bisection bracket from Cauchy's bound, polished by Newton-Raphson) and recurses on the reduced polynomial. |

```cpp
const ysq::Polynomial<double> p{-6.0, 11.0, -6.0, 1.0};   // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
for (double root : ysq::realRoots(p)) { /* 1, 2, 3, in some order */ }
```

## `Math/Random.hpp`

Uniform, normal, and Poisson sampling, the normal CDF, and Monte Carlo
integration, all over an explicit engine — never a hidden global generator.

```cpp
using RandomEngine = std::mt19937_64;
RandomEngine makeRandomEngine(std::uint64_t seed);

template <std::floating_point T> T uniformReal(RandomEngine&, T lo = T{0}, T hi = T{1});
template <std::integral T> T uniformInt(RandomEngine&, T lo, T hi);
template <std::floating_point T> T normal(RandomEngine&, T mean = T{0}, T stddev = T{1});
int poisson(RandomEngine&, double mean);
template <std::floating_point T> T normalCdf(T x, T mean = T{0}, T stddev = T{1});

template <class F, std::floating_point T>
T monteCarloIntegrate(F&& f, T lower, T upper, std::size_t samples, RandomEngine& engine);
```

| Function | Description |
| --- | --- |
| `makeRandomEngine(seed)` | One engine choice project-wide (`std::mt19937_64`): a seed alone reproduces a run. |
| `normalCdf` | `(1 + erf((x - mean) / (stddev sqrt(2)))) / 2`, `Math/SpecialFunctions.hpp`'s `erf` doing the real work. |
| `monteCarloIntegrate` | Plain Monte Carlo (uniform sampling, no variance reduction): `(upper - lower) * mean(f(sample))`. Converges as `O(1/sqrt(samples))` regardless of dimension, the property that makes it worth having alongside `Calculus.hpp`'s much faster-converging 1D quadrature rules. |

```cpp
ysq::RandomEngine engine = ysq::makeRandomEngine(42);
const double sample = ysq::normal(engine, 0.0, 1.0);
```

## `Math/Optimization.hpp`

Unconstrained minimization of a scalar field over `Vector2`/`3`/`4`, two
ways: with a gradient (estimated numerically) or without one at all.

```cpp
template <class F, class V>
typename V::value_type backtrackingLineSearch(
    F&& f, const V& x, const V& direction, const V& gradient,
    typename V::value_type initialStep = 1, typename V::value_type c1 = /* 1e-4 */,
    typename V::value_type shrink = 0.5, int maxIterations = 50);

template <class F, class V>
V gradientDescent(F&& f, V x, typename V::value_type tolerance = /* 1000 epsilons */,
                  int maxIterations = 1000);

template <class F, class V>
V nelderMead(F&& f, const V& initial, typename V::value_type initialStep = 0.1,
            typename V::value_type tolerance = /* 1000 epsilons */, int maxIterations = 1000);
```

| Function | Description |
| --- | --- |
| `backtrackingLineSearch` | Shrinks a step from `initialStep` until it satisfies the Armijo sufficient-decrease condition. What `gradientDescent` uses to pick each step's length rather than a fixed size. |
| `gradientDescent` | Steepest descent: steps opposite `numericalGradient(f, x)`, sized by `backtrackingLineSearch`. Needs `f` to be reasonably smooth. |
| `nelderMead` | The simplex method: reflects/expands/contracts/shrinks a simplex of `n + 1` points, comparing only values of `f`, never a gradient. For `f` that's noisy, discontinuous, or otherwise not something a finite difference could usefully differentiate. |

```cpp
const auto field = [](const ysq::Vec3& v) { return dot(v, v); };
const ysq::Vec3 minimum = ysq::gradientDescent(field, ysq::Vec3{1.0, 1.0, 1.0});  // -> near zero
```

## `Math/FFT.hpp`

The discrete Fourier transform, iterative radix-2 Cooley-Tukey: `O(n log n)`
rather than the `O(n^2)` direct sum, at the one precondition that makes the
recursive halving exact — every size involved a power of two.

```cpp
template <std::floating_point T> void fft(std::vector<Complex<T>>& data);    // in place, forward
template <std::floating_point T> void ifft(std::vector<Complex<T>>& data);   // in place, inverse, normalized by 1/n
template <std::floating_point T> std::vector<Complex<T>> fftReal(std::span<const T> samples);

template <std::floating_point T>
void fft3D(std::vector<Complex<T>>& data, std::size_t nx, std::size_t ny, std::size_t nz);
template <std::floating_point T>
void ifft3D(std::vector<Complex<T>>& data, std::size_t nx, std::size_t ny, std::size_t nz);
```

| Function | Description |
| --- | --- |
| `fft`/`ifft` | Forward and inverse share one routine, differing only in the twiddle factors' sign and the inverse's final `1/n` scale. |
| `fftReal` | The convenience for plain real samples (an audio buffer, a field sampled on a grid), rather than a `Complex` sequence with every imaginary part already zero. |
| `fft3D`/`ifft3D` | The row-column algorithm: a 3D DFT factors exactly into three passes of the 1D transform, one per axis, over a flat row-major buffer (`data[(i * ny + j) * nz + k]`, `x` slowest, `z` fastest). Each of `nx`/`ny`/`nz` independently a power of two. This is what `Physics/QuantumMechanics/WavePacket3D.hpp`'s split-step Fourier method is built on: the kinetic operator stays diagonal in momentum space in any dimension, since `kx^2+ky^2+kz^2` separates additively. |

```cpp
std::vector<ysq::Complex<double>> signal = /* ... */;
ysq::fft(signal);        // now the spectrum
ysq::ifft(signal);       // back to the original signal
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/math/numerics)
and let us know.
