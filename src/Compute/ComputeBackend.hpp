#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace ysq {

/// Which compute backend produced a result, and the priority selectComputeBackend
/// tries them in: Metal, then Cuda, then Vulkan, then OpenGL, then Cpu, which
/// always succeeds. See src/Compute/README.md.
enum class ComputeBackendKind { Cpu, OpenGL, Cuda, Vulkan, Metal };

[[nodiscard]] std::string_view toString(ComputeBackendKind kind) noexcept;

/// Where kernels run. Physics dispatches through this and never talks to a GPU
/// directly.
///
/// The CPU backend is the reference implementation, not a degraded mode: it
/// defines what correct means, and every other backend is validated against it
/// within tolerance, never for exact equality, because consumer GPUs are
/// commonly weak at float64 and this interface runs float32 uniformly across
/// every backend. See src/Compute/README.md.
///
/// This is the one interface in the engine core built on runtime polymorphism.
/// Everywhere else a compile-time concept is enough, because the choice is
/// known at compile time (Numeric, for instance); here it genuinely is not,
/// since which backend a machine can offer is discovered at run time.
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    [[nodiscard]] virtual ComputeBackendKind kind() const noexcept = 0;

    /// y[i] = a * x[i] + y[i]. x and y must be the same length.
    virtual void saxpy(std::span<const float> x, std::span<float> y, float a) const = 0;

    /// The sum of every element, zero for an empty span.
    [[nodiscard]] virtual float sum(std::span<const float> x) const = 0;

    /// A fused multi-term linear combine, `y[i] = sum_k coefficients[k] *
    /// terms[k][i]`, for one to four terms: what an RK4-family integrator
    /// step decomposes into (a weighted sum of several stage derivatives; see
    /// Math/Integrators/RK4.hpp), in one dispatch rather than one saxpy per
    /// term. `terms.size()` must equal `coefficients.size()`, in `[1, 4]`;
    /// every span in `terms`, and `y` itself, must be the same length.
    /// Domain-neutral, the same as saxpy/sum: nothing here is shaped around
    /// any one integrator or physical quantity.
    virtual void linearCombine(std::span<const std::span<const float>> terms,
                               std::span<const float> coefficients,
                               std::span<float> y) const = 0;

    /// Direct-sum (O(n^2)) Newtonian gravitational acceleration on every
    /// body from every other body, Plummer-softened:
    ///
    ///   a_i = sum_{j != i} gm[j] * (p_j - p_i) / (|p_j - p_i|^2 +
    ///   softeningSquared)^(3/2)
    ///
    /// Point masses only: this does not model J2/oblateness, since that
    /// needs the momentum-conserving reaction terms between an oblate
    /// source and the body it perturbs that Physics/Gravity/Newtonian.cpp's
    /// pairwise CPU loop carries and this one-sided per-target shape does
    /// not; a caller with any oblate body stays on that CPU path instead.
    /// positionsX/Y/Z and gm must all be the same length, as must the three
    /// (output) acceleration spans.
    virtual void gravitationalNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> gm, float softeningSquared,
                                    std::span<float> accelerationsX,
                                    std::span<float> accelerationsY,
                                    std::span<float> accelerationsZ) const = 0;

    /// Direct-sum electric field at every point charge's own position, from
    /// every other point charge, Coulomb's law:
    ///
    ///   E_i = coulombConstant * sum_{j != i} charge[j] * (p_i - p_j) /
    ///   |p_i - p_j|^3
    ///
    /// Unsoftened, skipping (contributing nothing from) a source at zero
    /// separation, matching Physics/Electromagnetism/Field.cpp exactly.
    virtual void electricFieldNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> charge, float coulombConstant,
                                    std::span<float> fieldX, std::span<float> fieldY,
                                    std::span<float> fieldZ) const = 0;

    /// Direct-sum magnetic flux density at every point charge's own
    /// position, from every other moving point charge, the point-charge
    /// form of Biot-Savart:
    ///
    ///   B_i = permeabilityOver4Pi * sum_{j != i} charge[j] * (v_j x
    ///   (p_i - p_j)) / |p_i - p_j|^3
    ///
    /// Unsoftened, matching Physics/Electromagnetism/Field.cpp exactly.
    virtual void magneticFieldNBody(
        std::span<const float> positionsX, std::span<const float> positionsY,
        std::span<const float> positionsZ, std::span<const float> velocitiesX,
        std::span<const float> velocitiesY, std::span<const float> velocitiesZ,
        std::span<const float> charge, float permeabilityOver4Pi, std::span<float> fieldX,
        std::span<float> fieldY, std::span<float> fieldZ) const = 0;

    /// SPH density and pressure at every particle's own position: the
    /// cubic spline kernel sum (Physics/Fluids/SPH.hpp's cubicSplineKernel)
    /// over every particle, including itself, within the kernel's compact
    /// support (2 * smoothingLength), then a polytropic equation of state,
    /// `pressure[i] = equationOfStateK * density[i]^polytropicIndex`.
    /// Direct O(n^2) with a distance cutoff, not a spatial acceleration
    /// structure: see src/Compute/README.md on why. positionsX/Y/Z and
    /// mass must all be the same length, as must density/pressure (output).
    virtual void sphDensityPressure(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> mass, float smoothingLength,
                                    float equationOfStateK, float polytropicIndex,
                                    std::span<float> density,
                                    std::span<float> pressure) const = 0;

    /// The symmetric SPH pressure-gradient acceleration on every particle,
    /// `a_i = -sum_{j != i} m_j (P_i/rho_i^2 + P_j/rho_j^2) grad_i W_ij`,
    /// over every particle within the kernel's compact support. density
    /// and pressure must already be current (a prior sphDensityPressure
    /// call, or its CPU equivalent). Matches
    /// Physics/Fluids/SPH.hpp's pressureAccelerations() exactly.
    virtual void sphPressureAcceleration(
        std::span<const float> positionsX, std::span<const float> positionsY,
        std::span<const float> positionsZ, std::span<const float> mass,
        std::span<const float> density, std::span<const float> pressure,
        float smoothingLength, std::span<float> accelerationsX,
        std::span<float> accelerationsY, std::span<float> accelerationsZ) const = 0;

    /// One explicit-Euler diffusion step over a periodic 3D grid, flat
    /// row-major (`(i * ny + j) * nz + k`), interior cells only (no ghost
    /// cells: periodic wraparound is computed directly in the kernel).
    /// `next[p] = temperature[p] + factor * laplacian(p)`, `factor =
    /// diffusivity * dt / spacing^2`. Matches
    /// Physics/Thermodynamics/HeatEquation3D.cpp's step() exactly.
    virtual void heatEquation3DStep(std::span<const float> temperature, std::size_t nx,
                                    std::size_t ny, std::size_t nz, float factor,
                                    std::span<float> next) const = 0;

    /// One leapfrog FDTD step of the 3D acoustic wave equation over a
    /// periodic grid, flat row-major, interior cells only (periodic
    /// wraparound via modular index arithmetic, the same convention
    /// heatEquation3DStep uses). Two passes internally: velocity is
    /// updated from the input pressure first, then pressure is updated
    /// from the just-updated velocity, matching
    /// Physics/Acoustics/Acoustic3D.cpp's step() exactly (`velocityFactor
    /// = (dt/dx)/density`, `pressureFactor = density*soundSpeed^2*(dt/dx)`).
    virtual void
    acoustic3DStep(std::span<const float> pressure, std::span<const float> velocityX,
                   std::span<const float> velocityY, std::span<const float> velocityZ,
                   std::size_t nx, std::size_t ny, std::size_t nz, float velocityFactor,
                   float pressureFactor, std::span<float> nextPressure,
                   std::span<float> nextVelocityX, std::span<float> nextVelocityY,
                   std::span<float> nextVelocityZ) const = 0;

    /// One leapfrog FDTD step of the 3D Maxwell curl equations over a
    /// periodic grid, flat row-major, interior cells only (periodic
    /// wraparound via modular index arithmetic, the same convention every
    /// other grid kernel here uses). Two passes internally: B is updated
    /// from the input E first, then E is updated from the just-updated B,
    /// matching Physics/Electromagnetism/Maxwell3D.cpp's step() exactly
    /// (`bFactor = dt/spacing`, `eFactor = speedOfLight^2 * dt/spacing`).
    virtual void maxwell3DStep(std::span<const float> ex, std::span<const float> ey,
                               std::span<const float> ez, std::span<const float> bx,
                               std::span<const float> by, std::span<const float> bz,
                               std::size_t nx, std::size_t ny, std::size_t nz,
                               float bFactor, float eFactor, std::span<float> nextEx,
                               std::span<float> nextEy, std::span<float> nextEz,
                               std::span<float> nextBx, std::span<float> nextBy,
                               std::span<float> nextBz) const = 0;

    /// One dimensional-split finite-volume sweep of the compressible Euler
    /// equations along one axis, first-order Rusanov flux, over a periodic
    /// grid, flat row-major, interior cells only (periodic wraparound via
    /// modular index arithmetic, the same convention every other grid
    /// kernel here uses). `momentumNormal` is whichever momentum component
    /// is normal to `axis` (0 = x, 1 = y, 2 = z); `momentumTangent1/2` are
    /// the other two, carried passively (no pressure term in their own
    /// flux, matching a 1D-normal Riemann problem's actual physics).
    /// `dtOverSpacing = dt / spacing`. Matches
    /// Physics/Fluids/Eulerian3D.cpp's `sweep()` exactly for one axis; a
    /// caller doing the full x-then-y-then-z step calls this three times.
    virtual void eulerianFluid3DSweep(
        std::span<const float> density, std::span<const float> momentumNormal,
        std::span<const float> momentumTangent1, std::span<const float> momentumTangent2,
        std::span<const float> energy, std::size_t nx, std::size_t ny, std::size_t nz,
        int axis, float gamma, float dtOverSpacing, std::span<float> nextDensity,
        std::span<float> nextMomentumNormal, std::span<float> nextMomentumTangent1,
        std::span<float> nextMomentumTangent2, std::span<float> nextEnergy) const = 0;

    /// A batched, power-of-two, radix-2 Cooley-Tukey FFT: `batchCount`
    /// independent sequences of `length` complex elements each (`length`
    /// must be a power of two), stored as separate flat real/imaginary
    /// arrays with each batch contiguous (`data[batch * length + i]`).
    /// Matches Math/FFT.hpp's `detail::fftImpl` exactly, per batch:
    /// bit-reversal permutation, then one butterfly pass per stage `len =
    /// 2, 4, ..., length`, then, if `inverse`, a final `1/length` scale.
    /// Internally several GPU dispatches run in sequence inside one call
    /// (`log2(length) + 2`: bit-reversal, one per butterfly stage, and the
    /// scale when `inverse`), the same multi-pass-inside-one-call shape
    /// `acoustic3DStep`/`maxwell3DStep` already use for two passes,
    /// extended here to as many as `length` requires. `real`/`imag` and
    /// `nextReal`/`nextImag` must all be `length * batchCount` long.
    virtual void fftBatched(std::span<const float> real, std::span<const float> imag,
                            std::size_t length, std::size_t batchCount, bool inverse,
                            std::span<float> nextReal,
                            std::span<float> nextImag) const = 0;

    /// Dense matrix-vector multiply, row-major: `result[r] = sum_c
    /// matrix[r * cols + c] * vector[c]`. `matrix` is `rows * cols`,
    /// `vector` is `cols`, `result` is `rows`.
    virtual void matVec(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                        std::span<const float> vector, std::span<float> result) const = 0;

    /// Dense matrix-matrix multiply, row-major, naive O(rows*aCols*bCols):
    /// `result[r * bCols + c] = sum_k a[r * aCols + k] * b[k * bCols + c]`.
    /// `a` is `aRows * aCols`, `b` is `aCols * bCols` (its row count is
    /// implied), `result` is `aRows * bCols`.
    virtual void matMul(std::span<const float> a, std::size_t aRows, std::size_t aCols,
                        std::span<const float> b, std::size_t bCols,
                        std::span<float> result) const = 0;

    /// Dense LU decomposition with partial pivoting, Doolittle form,
    /// matching Math/LinearSolve.hpp's `luDecompose` exactly: `lu` packs
    /// both factors (L's unit diagonal implicit), `pivot[i]` is the
    /// original row now sitting in row `i` after the swaps elimination
    /// made. Returns `false` (leaving `lu`/`pivot` unspecified) if `matrix`
    /// is singular or non-finite at some pivot, matching `luDecompose`'s
    /// `nullopt`. `n` sequential steps internally, one GPU dispatch
    /// sequence per pivot column: a reduction finds the pivot row (the
    /// same reduce-then-finish-on-CPU shape `minIndex` already uses, here
    /// restricted to one column's remaining rows), then a row swap, then a
    /// trailing-submatrix elimination update -- all three read the
    /// previous step's complete output, so, like `fftBatched`, this
    /// ping-pongs between two buffers across steps.
    [[nodiscard]] virtual bool luDecomposeGpu(std::span<const float> matrix,
                                              std::size_t n, std::span<float> lu,
                                              std::span<std::uint32_t> pivot) const = 0;

    /// Dense Cholesky factorization `a = L L^T` for a symmetric
    /// positive-definite `a`, matching Math/LinearSolve.hpp's
    /// `choleskyDecompose` exactly (only the lower triangle of `a` is
    /// read). Returns `false` (leaving `l` unspecified) the moment a
    /// diagonal entry would be non-positive, matching
    /// `choleskyDecompose`'s `nullopt`. `n` sequential steps internally,
    /// one column at a time: a small dispatch computes that column's
    /// diagonal entry (no pivoting needed, unlike LU, since
    /// positive-definiteness alone keeps every pivot valid), then a second
    /// dispatch computes every entry below it in parallel, each thread's
    /// own row independently reading that column's now-complete diagonal.
    [[nodiscard]] virtual bool choleskyDecomposeGpu(std::span<const float> a,
                                                    std::size_t n,
                                                    std::span<float> l) const = 0;

    /// Dense QR decomposition via Householder reflections, matching
    /// Math/Eigen.hpp's `qrDecompose` exactly: `q` (`rows * rows`,
    /// orthogonal) and `r` (`rows * cols`, upper triangular in its top
    /// `cols * cols` block) for any `matrix` (`rows * cols`, `rows >=
    /// cols`). `min(rows, cols)` sequential steps internally, one per
    /// column: a small dispatch reduces that column's remaining norm (and
    /// reads back the one diagonal entry needed to pick the reflection's
    /// sign, the same small-per-step-readback shape `luDecomposeGpu`'s
    /// pivot search and `choleskyDecomposeGpu`'s diagonal already use),
    /// then two dispatches apply the same reflection to the working `r`
    /// and accumulate it into `q` -- both read the *pre-this-step* `r`
    /// (the reflection vector is captured once per step, then applied to
    /// both, exactly matching `qrDecompose`'s own order of operations, not
    /// two independent reflections).
    virtual void qrDecomposeGpu(std::span<const float> matrix, std::size_t rows,
                                std::size_t cols, std::span<float> q,
                                std::span<float> r) const = 0;

    /// The cyclic Jacobi eigenvalue algorithm for a symmetric `matrix`
    /// (`n * n`), matching Math/Eigen.hpp's `jacobiEigenSymmetric`'s
    /// numerical iteration exactly (sorting into ascending
    /// `EigenDecomposition` order happens in Math/Eigen.hpp itself, not
    /// here). Up to `maxSweeps` sweeps, each `n - 1` *rounds* of the
    /// standard round-robin/tournament pairing (Brent & Luk 1985): within
    /// one round every pair of indices is disjoint, so all `n / 2`
    /// rotations that round commute and can run as one GPU dispatch
    /// instead of `n(n-1)/2` individually sequential ones (what the cyclic
    /// CPU version below actually does; a full round-robin sweep reaches
    /// the same set of pairs in a different order, converging to the same
    /// diagonalization). Each round is *two* dispatches, not one: applying
    /// a two-sided similarity `Q^T A Q` for several simultaneous disjoint
    /// rotations is not simply each pair independently overwriting its own
    /// rows and columns (the cross terms connecting one round's pairs
    /// depend on both rotations at once) -- the correct decomposition is a
    /// column-mixing pass computing `B = A Q` first (also mixing the
    /// accumulated eigenvectors' own columns the same way in the same
    /// dispatch, since that update is a pure right-multiply too), then a
    /// row-mixing pass computing `A' = Q^T B` from that pass's complete
    /// output. `resultDiagonal` (`n * n`, though only the diagonal is meaningful
    /// once converged) and `resultEigenvectors` (`n * n`) are both
    /// pre-sized by the caller; convergence is checked once per sweep by
    /// reading the working matrix back and summing its squared
    /// off-diagonal entries, mirroring the CPU version's own per-sweep
    /// check.
    virtual void jacobiEigenSymmetricGpu(std::span<const float> matrix, std::size_t n,
                                         int maxSweeps, float tolerance,
                                         std::span<float> resultDiagonal,
                                         std::span<float> resultEigenvectors) const = 0;

    /// One-sided Jacobi SVD, matching Math/Eigen.hpp's `svd`'s numerical
    /// iteration exactly (extracting singular values as column norms,
    /// sorting, and normalizing `u` all happen in Math/Eigen.hpp itself,
    /// same division of labor as `jacobiEigenSymmetricGpu`). Same
    /// round-robin pairing and per-sweep convergence check as
    /// `jacobiEigenSymmetricGpu`, but only *one* dispatch per round: this
    /// rotation is one-sided (`A <- A Q`, no corresponding left
    /// multiplication), so, unlike the symmetric case, disjoint pairs
    /// never share a cross term and every round-robin round genuinely is
    /// `n / 2` independent column pairs read from and written to
    /// directly. `resultA` (`rows * cols`) and `resultV` (`cols * cols`)
    /// are both pre-sized by the caller.
    virtual void jacobiSvdGpu(std::span<const float> matrix, std::size_t rows,
                              std::size_t cols, int maxSweeps, float tolerance,
                              std::span<float> resultA,
                              std::span<float> resultV) const = 0;

    /// Batched, independent, pointwise evaluation: the same scalar function
    /// (or, for `batchPolynomialEval`/`batchCubicSplineEval`, the same
    /// fixed polynomial/spline) evaluated at every element of `x`
    /// independently, one thread per element, no reduction and no
    /// multi-pass structure at all -- the simplest kernel shape in this
    /// file. Matches Math/SpecialFunctions.hpp's `erf`/`erfc`/`gamma`/
    /// `logGamma`/`legendreP`, Math/Polynomial.hpp's `Polynomial::operator()`,
    /// and Math/Interpolation.hpp's `CubicSpline::operator()`, respectively.
    /// `erf`/`erfc` use the Abramowitz & Stegun 7.1.26 rational
    /// approximation and `gamma`/`logGamma` the Lanczos approximation
    /// (reflected via `gamma(x) = pi / (sin(pi x) gamma(1-x))` for `x <=
    /// 0`) rather than `Math/SpecialFunctions.hpp`'s own `std::erf`/
    /// `std::tgamma`/`std::lgamma` calls, since there is no GPU-side C99
    /// math library to call into; both are well-conditioned in `float32`
    /// (no intermediate cancellation between differently-scaled terms).
    /// `Math/SpecialFunctions.hpp`'s `besselJ`/`besselY` deliberately have
    /// no batched kernel here: their existing rational-polynomial
    /// approximation (Abramowitz & Stegun 9.4.1-9.4.6) sums terms that
    /// individually reach ~1e11 in magnitude to produce an O(1) result --
    /// numerically fine in the `double` it already runs in, but this
    /// cancellation would consume `float32`'s entire ~7 digits of
    /// precision and leave nothing for the answer itself. A `float32`-safe
    /// Bessel approximation (a power series for small `x`, say) is a
    /// distinct, independently-verified numerical method, not a mechanical
    /// port of the existing one, and is out of scope for this pass; see
    /// src/Compute/README.md.
    virtual void batchErf(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchErfc(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchGamma(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchLogGamma(std::span<const float> x,
                               std::span<float> result) const = 0;

    /// `n`/`m` fixed for the whole batch (`0 <= m <= n`, `-1 <= x[i] <= 1`),
    /// matching `legendreP`'s own preconditions.
    virtual void batchLegendreP(unsigned n, unsigned m, std::span<const float> x,
                                std::span<float> result) const = 0;

    /// Horner's method, `coefficients` ascending (`coefficients[i]` is the
    /// coefficient of `x^i`), matching `Polynomial::operator()` exactly.
    virtual void batchPolynomialEval(std::span<const float> coefficients,
                                     std::span<const float> x,
                                     std::span<float> result) const = 0;

    /// Matches `CubicSpline::operator()` exactly, including its held-flat
    /// behavior outside `[knotsX.front(), knotsX.back()]`. `knotsX` strictly
    /// increasing; `knotsX`, `knotsY`, and `secondDerivatives` all the same
    /// length (`CubicSpline::natural`'s own solved second derivatives,
    /// computed on the CPU once when the spline itself is built, not
    /// recomputed here).
    virtual void batchCubicSplineEval(std::span<const float> knotsX,
                                      std::span<const float> knotsY,
                                      std::span<const float> secondDerivatives,
                                      std::span<const float> queryX,
                                      std::span<float> result) const = 0;

    /// Parallel, counter-based random number generation (Philox4x32-10;
    /// Salmon, Moraes, Hadjidoukas & Schulten 2011): thread `i`'s output
    /// depends only on `(seed, offset + i)`, with no shared state and no
    /// sequential dependency between threads at all -- embarrassingly
    /// parallel by construction, unlike `Math/Random.hpp`'s own
    /// `std::mt19937_64`-based `RandomEngine`, a single strictly sequential
    /// stream that cannot be split across GPU threads without a full port
    /// of Mersenne Twister's own state machine (and, even then, a
    /// jump-ahead capability MT19937 does not straightforwardly offer).
    /// This is deliberately a second, additional RNG family, not a
    /// replacement: `RandomEngine`'s engine-threaded API is for a single,
    /// deterministic, bit-reproducible sequential stream; this one is for
    /// bulk, order-independent sampling (a batch Monte Carlo integral, a
    /// particle system's initial velocities) at a size where GPU dispatch
    /// pays for itself. `offset` lets repeated calls with the same `seed`
    /// draw successive, non-overlapping stretches of the same logical
    /// stream (`offset = 0`, then `offset = result.size()`, and so on).
    /// `batchUniformReal` produces `[0, 1)`; `batchNormal` produces
    /// standard-normal (mean 0, stddev 1) via the Box-Muller transform,
    /// consuming two of Philox's four output words per thread. Matching
    /// `Math/Random.hpp`'s own division of labor for `uniformReal`/`normal`,
    /// affine rescaling to an arbitrary range/mean/stddev happens at the
    /// `Math` level, not here.
    virtual void batchUniformReal(std::uint64_t seed, std::uint64_t offset,
                                  std::span<float> result) const = 0;
    virtual void batchNormal(std::uint64_t seed, std::uint64_t offset,
                             std::span<float> result) const = 0;

    /// Ascending sort, in place, matching `Math/Sort.hpp`'s `sortInPlace`.
    /// Every backend accepts any `values.size()`, not just a power of two:
    /// GPU backends implement this via bitonic sort, the standard
    /// data-parallel sorting network, which only naturally handles
    /// power-of-two sizes, so they pad internally with `+infinity`
    /// sentinels (guaranteed to sort to the very end and never displace any
    /// of the first `values.size()` elements, unlike zero-padding an FFT,
    /// which does change the answer) and truncate back before returning,
    /// invisibly to the caller.
    virtual void sortAscending(std::span<float> values) const = 0;

    /// Geometric-multigrid restriction: the straight average of the 8 fine
    /// cells each coarse cell exactly contains, for a cell-centered grid
    /// coarsened by exactly a factor of 2 on every axis. `fine` is flat
    /// row-major (`(i * ny + j) * nz + k`) with dimensions `nx, ny, nz`
    /// (each must be even); `coarse` is the same layout at `nx/2, ny/2,
    /// nz/2`. Equation-independent: matches
    /// Math/Multigrid.hpp's `detail::restrictGrid` exactly, which is what
    /// every FAS multigrid solve built on it uses regardless of which
    /// elliptic equation it's solving.
    virtual void multigridRestrict3D(std::span<const float> fine, std::size_t nx,
                                     std::size_t ny, std::size_t nz,
                                     std::span<float> coarse) const = 0;

    /// Geometric-multigrid prolongation: adds each coarse cell's correction
    /// to the 8 fine cells it came from, `nextFine[p] = fine[p] +
    /// coarseCorrection[p/2]` for the coarse cell containing fine cell `p`.
    /// Same layout convention as multigridRestrict3D (`nx, ny, nz` are the
    /// fine grid's dimensions, each even; `coarseCorrection` is `nx/2,
    /// ny/2, nz/2`). Equation-independent, matching
    /// Math/Multigrid.hpp's `detail::prolongateAndAdd` exactly.
    virtual void multigridProlongateAndAdd3D(std::span<const float> fine,
                                             std::span<const float> coarseCorrection,
                                             std::size_t nx, std::size_t ny,
                                             std::size_t nz,
                                             std::span<float> nextFine) const = 0;

    /// The index of the minimum element, or `x.size()` for an empty span:
    /// the same reduction shape as `sum`, a different operator, for
    /// Mechanics/Hermite.hpp's IndividualTimestepScheduler::nextMover
    /// (which body's next update time is soonest). Ties resolve to the
    /// lowest index, matching a plain linear scan.
    [[nodiscard]] virtual std::size_t minIndex(std::span<const float> x) const = 0;
};

/// Whether `kind` can actually be used right now on this machine: SDK, driver
/// and hardware all present. Safe to call before selecting anything; for
/// OpenGL, CUDA and Vulkan this opens and discards a context or device to find
/// out, so it costs more than a flag check.
[[nodiscard]] bool computeBackendAvailable(ComputeBackendKind kind);

/// The first available backend, tried in priority order. `forceBackend` skips
/// the probing and returns exactly that backend, or nullptr if it is not
/// available; for debugging and benchmarking. With no override this only
/// returns nullptr if Cpu itself failed, which does not happen.
[[nodiscard]] std::unique_ptr<ComputeBackend>
selectComputeBackend(std::optional<ComputeBackendKind> forceBackend = std::nullopt);

/// The process-wide default backend: selected once, via selectComputeBackend()
/// with no override, on first call, and cached for every call after that.
///
/// selectComputeBackend() itself reprobes every time it's called, which is
/// deliberate there (debugging and benchmarking want a fresh probe) but too
/// expensive to pay on every dispatch from Math or Physics, since probing
/// Metal/CUDA/Vulkan/OpenGL opens and discards a real device or context. This
/// is what Math and Physics call instead: once resolved, the same backend
/// answers every automatic-dispatch call for the life of the process.
[[nodiscard]] ComputeBackend& defaultBackend();

}  // namespace ysq
