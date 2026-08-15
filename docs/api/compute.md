# Compute API reference

Every public type and function in `Compute`: the backend interface `Math`
and `Physics` dispatch through, plus the CPU, Metal, OpenGL, CUDA and Vulkan
implementations. Start with [docs/compute.md](../compute.md) for backend
selection and the fallback ladder; [src/Compute/README.md](../../src/Compute/README.md)
covers platform-by-platform availability and the precision tradeoffs in
depth. `Compute` depends only on `Core` (and, for the OpenGL backend,
`Platform`), deliberately not on `Math`/`Units`: every reference kernel
operates on plain `float`/`double` spans.

## `Compute/ComputeBackend.hpp`

The interface, and backend selection.

```cpp
enum class ComputeBackendKind { Cpu, OpenGL, Cuda, Vulkan, Metal };
std::string_view toString(ComputeBackendKind kind) noexcept;

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual ComputeBackendKind kind() const noexcept = 0;

    // y[i] = a * x[i] + y[i]. x and y must be the same length.
    virtual void saxpy(std::span<const float> x, std::span<float> y, float a) const = 0;

    // Sum of every element; zero for an empty span.
    virtual float sum(std::span<const float> x) const = 0;

    // y[i] = sum_k coefficients[k] * terms[k][i], for one to four terms.
    virtual void linearCombine(std::span<const std::span<const float>> terms,
                               std::span<const float> coefficients,
                               std::span<float> y) const = 0;

    // Direct-sum (O(n^2)) Plummer-softened Newtonian gravity, point masses only.
    virtual void gravitationalNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> gm, float softeningSquared,
                                    std::span<float> accelerationsX,
                                    std::span<float> accelerationsY,
                                    std::span<float> accelerationsZ) const = 0;

    // Direct-sum Coulomb electric field at every point charge's own position.
    virtual void electricFieldNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> charge, float coulombConstant,
                                    std::span<float> fieldX, std::span<float> fieldY,
                                    std::span<float> fieldZ) const = 0;

    // Direct-sum point-charge Biot-Savart magnetic flux density.
    virtual void magneticFieldNBody(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> velocitiesX,
                                    std::span<const float> velocitiesY,
                                    std::span<const float> velocitiesZ,
                                    std::span<const float> charge, float permeabilityOver4Pi,
                                    std::span<float> fieldX, std::span<float> fieldY,
                                    std::span<float> fieldZ) const = 0;

    // SPH cubic-spline density/pressure and pressure-gradient acceleration,
    // direct O(n^2) with a distance cutoff at the compact support (2h).
    virtual void sphDensityPressure(std::span<const float> positionsX,
                                    std::span<const float> positionsY,
                                    std::span<const float> positionsZ,
                                    std::span<const float> mass, float smoothingLength,
                                    float equationOfStateK, float polytropicIndex,
                                    std::span<float> density,
                                    std::span<float> pressure) const = 0;
    virtual void sphPressureAcceleration(
        std::span<const float> positionsX, std::span<const float> positionsY,
        std::span<const float> positionsZ, std::span<const float> mass,
        std::span<const float> density, std::span<const float> pressure,
        float smoothingLength, std::span<float> accelerationsX,
        std::span<float> accelerationsY, std::span<float> accelerationsZ) const = 0;

    // Index of the minimum element, or x.size() for an empty span.
    virtual std::size_t minIndex(std::span<const float> x) const = 0;

    // Explicit-Euler diffusion step on a periodic nx*ny*nz grid, flattened
    // (i*ny+j)*nz+k. factor = alpha*dt/h^2.
    virtual void heatEquation3DStep(std::span<const float> temperature, std::size_t nx,
                                    std::size_t ny, std::size_t nz, float factor,
                                    std::span<float> next) const = 0;

    // Leapfrog FDTD step for the linear acoustic wave equation on a
    // periodic grid: velocity half-step from the pressure gradient, then
    // pressure full-step from the updated velocity's divergence.
    virtual void acoustic3DStep(std::span<const float> pressure,
                                std::span<const float> velocityX,
                                std::span<const float> velocityY,
                                std::span<const float> velocityZ, std::size_t nx,
                                std::size_t ny, std::size_t nz, float velocityFactor,
                                float pressureFactor, std::span<float> nextPressure,
                                std::span<float> nextVelocityX,
                                std::span<float> nextVelocityY,
                                std::span<float> nextVelocityZ) const = 0;

    // Leapfrog FDTD step for vacuum Maxwell's equations on a periodic
    // grid: B half-step from curl(E), then E full-step from curl(B).
    virtual void maxwell3DStep(std::span<const float> ex, std::span<const float> ey,
                               std::span<const float> ez, std::span<const float> bx,
                               std::span<const float> by, std::span<const float> bz,
                               std::size_t nx, std::size_t ny, std::size_t nz,
                               float bFactor, float eFactor, std::span<float> nextEx,
                               std::span<float> nextEy, std::span<float> nextEz,
                               std::span<float> nextBx, std::span<float> nextBy,
                               std::span<float> nextBz) const = 0;

    // One dimensional-split finite-volume sweep of the compressible Euler
    // equations along one axis (0/1/2), first-order Rusanov flux, periodic.
    // momentumNormal is whichever momentum component is normal to axis;
    // momentumTangent1/2 are the other two, carried passively.
    // dtOverSpacing = dt / spacing. Call three times (once per axis) for a
    // full dimensional-split step.
    virtual void eulerianFluid3DSweep(
        std::span<const float> density, std::span<const float> momentumNormal,
        std::span<const float> momentumTangent1, std::span<const float> momentumTangent2,
        std::span<const float> energy, std::size_t nx, std::size_t ny, std::size_t nz,
        int axis, float gamma, float dtOverSpacing, std::span<float> nextDensity,
        std::span<float> nextMomentumNormal, std::span<float> nextMomentumTangent1,
        std::span<float> nextMomentumTangent2, std::span<float> nextEnergy) const = 0;

    // A batched, power-of-two, radix-2 Cooley-Tukey FFT: batchCount
    // independent length-element complex sequences, data[batch*length+i],
    // real/imaginary as separate flat arrays. Internally several GPU
    // dispatches in sequence (bit-reversal, one per butterfly stage, and a
    // final 1/length scale when inverse). Used by Math/FFT.hpp, not
    // Physics.
    virtual void fftBatched(std::span<const float> real, std::span<const float> imag,
                            std::size_t length, std::size_t batchCount, bool inverse,
                            std::span<float> nextReal, std::span<float> nextImag) const = 0;

    // Dense matrix-vector / matrix-matrix multiply, row-major, naive
    // (one thread per output element). Used by Math/LinearSolve.hpp, not
    // Physics.
    virtual void matVec(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                        std::span<const float> vector, std::span<float> result) const = 0;
    virtual void matMul(std::span<const float> a, std::size_t aRows, std::size_t aCols,
                        std::span<const float> b, std::size_t bCols,
                        std::span<float> result) const = 0;

    // Dense LU decomposition with partial pivoting (Doolittle form) and
    // dense Cholesky factorization (a = L L^T, symmetric positive-definite
    // only), matching Math/LinearSolve.hpp's luDecompose/choleskyDecompose
    // exactly. n sequential GPU dispatches internally (one per pivot
    // column / factorization column); false (output spans unspecified) on
    // singular/non-positive-definite, matching the CPU functions' nullopt.
    [[nodiscard]] virtual bool luDecomposeGpu(std::span<const float> matrix, std::size_t n,
                                              std::span<float> lu,
                                              std::span<std::uint32_t> pivot) const = 0;
    [[nodiscard]] virtual bool choleskyDecomposeGpu(std::span<const float> a, std::size_t n,
                                                    std::span<float> l) const = 0;

    // Dense QR decomposition via Householder reflections, matching
    // Math/Eigen.hpp's qrDecompose. min(rows,cols) sequential steps
    // internally, the same shape as luDecomposeGpu.
    virtual void qrDecomposeGpu(std::span<const float> matrix, std::size_t rows,
                                std::size_t cols, std::span<float> q,
                                std::span<float> r) const = 0;

    // The cyclic Jacobi eigenvalue algorithm (symmetric matrices) and
    // one-sided Jacobi SVD, matching Math/Eigen.hpp's jacobiEigenSymmetric
    // and svd numerically (sorting/normalizing happens in Math/Eigen.hpp
    // itself). Both use round-robin/tournament pair ordering internally,
    // not the CPU reference's cyclic order, converging to the same result.
    virtual void jacobiEigenSymmetricGpu(std::span<const float> matrix, std::size_t n,
                                         int maxSweeps, float tolerance,
                                         std::span<float> resultDiagonal,
                                         std::span<float> resultEigenvectors) const = 0;
    virtual void jacobiSvdGpu(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                              int maxSweeps, float tolerance, std::span<float> resultA,
                              std::span<float> resultV) const = 0;

    // Batched, independent, pointwise evaluation: one thread per element of
    // x, no reduction, no multi-pass structure. Matches
    // Math/SpecialFunctions.hpp's erf/erfc/gamma/logGamma/legendreP,
    // Math/Polynomial.hpp's Polynomial::operator(), and
    // Math/Interpolation.hpp's CubicSpline::operator(), respectively.
    // erf/erfc/gamma/logGamma use well-conditioned float32 approximations
    // (Abramowitz & Stegun 7.1.26 for erf/erfc, Lanczos for gamma/logGamma)
    // rather than a C99 math library, which no GPU shading language has.
    // besselJ/besselY have no batched kernel here: their existing
    // rational-polynomial approximation is numerically fine in double but
    // would exhaust float32's precision in intermediate cancellation.
    virtual void batchErf(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchErfc(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchGamma(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchLogGamma(std::span<const float> x, std::span<float> result) const = 0;
    virtual void batchLegendreP(unsigned n, unsigned m, std::span<const float> x,
                                std::span<float> result) const = 0;
    virtual void batchPolynomialEval(std::span<const float> coefficients,
                                     std::span<const float> x,
                                     std::span<float> result) const = 0;
    virtual void batchCubicSplineEval(std::span<const float> knotsX,
                                      std::span<const float> knotsY,
                                      std::span<const float> secondDerivatives,
                                      std::span<const float> queryX,
                                      std::span<float> result) const = 0;

    // Parallel, counter-based random number generation (Philox4x32-10): a
    // second, additional RNG family alongside Math/Random.hpp's sequential
    // RandomEngine, embarrassingly parallel by construction since thread
    // i's output depends only on (seed, offset + i), no shared state.
    // batchUniformReal produces [0, 1); batchNormal produces standard
    // normal via Box-Muller. offset lets repeated calls with the same seed
    // draw successive, non-overlapping stretches of the same stream.
    virtual void batchUniformReal(std::uint64_t seed, std::uint64_t offset,
                                  std::span<float> result) const = 0;
    virtual void batchNormal(std::uint64_t seed, std::uint64_t offset,
                             std::span<float> result) const = 0;

    // Ascending sort, in place, via bitonic sort (Batcher 1968), matching
    // Math/Sort.hpp's sortInPlace. Accepts any values.size(), not just a
    // power of two: GPU backends pad internally with +infinity and
    // truncate back, invisibly to the caller.
    virtual void sortAscending(std::span<float> values) const = 0;

    // Geometric-multigrid restriction: the straight average of the 8 fine
    // cells each coarse cell exactly contains. fine is nx*ny*nz (each
    // even), coarse is nx/2 * ny/2 * nz/2. Equation-independent: used by
    // Math/Multigrid.hpp, not Physics.
    virtual void multigridRestrict3D(std::span<const float> fine, std::size_t nx,
                                     std::size_t ny, std::size_t nz,
                                     std::span<float> coarse) const = 0;

    // Geometric-multigrid prolongation: nextFine[p] = fine[p] +
    // coarseCorrection[p's coarse cell]. Equation-independent.
    virtual void multigridProlongateAndAdd3D(std::span<const float> fine,
                                             std::span<const float> coarseCorrection,
                                             std::size_t nx, std::size_t ny, std::size_t nz,
                                             std::span<float> nextFine) const = 0;
};

bool computeBackendAvailable(ComputeBackendKind kind);
std::unique_ptr<ComputeBackend>
selectComputeBackend(std::optional<ComputeBackendKind> forceBackend = std::nullopt);
ComputeBackend& defaultBackend();
```

| Function | Description |
| --- | --- |
| `computeBackendAvailable(kind)` | Whether `kind` can genuinely be used right now (SDK, driver, hardware all present). Safe before selecting anything. For Metal/OpenGL/CUDA/Vulkan this opens and discards a real device or context, so it costs more than a flag check; call it at startup, not per frame. |
| `selectComputeBackend(forceBackend)` | Probes `Metal -> Cuda -> Vulkan -> OpenGL -> Cpu` in that order and returns the first available. `forceBackend` skips probing and returns exactly that backend (or `nullptr` if unavailable), for debugging/benchmarking. With no override, only returns `nullptr` if `Cpu` itself failed, which doesn't happen. Reprobes on every call, so calling this per-frame is wasteful; see `defaultBackend()`. |
| `defaultBackend()` | The process-wide default: `selectComputeBackend()` run once, on first call, and cached for every call after. What `Math` and `Physics` route through for automatic dispatch. |

```cpp
std::unique_ptr<ysq::ComputeBackend> backend = ysq::selectComputeBackend();
backend->saxpy(x, y, 2.0f);   // y[i] = 2*x[i] + y[i]
const float total = backend->sum(x);

// Or, for repeated automatic dispatch without reprobing every call:
ysq::ComputeBackend& fast = ysq::defaultBackend();
```

**The interface is `float` in, `float` out on every backend**, deliberately
uniform: consumer GPUs are commonly weak at `float64` (often 1/32 to 1/64
`float32` throughput), so a `double` interface isn't something a GPU backend
could implement well even in principle. The CPU backend is the **reference
implementation**, not a fallback: it defines what "correct" means, and
every other backend is checked against it within tolerance, never for exact
equality.

## `Compute/CPU/CpuBackend.hpp`

Always available; what every other backend is validated against.

```cpp
class CpuBackend final : public ComputeBackend {
public:
    static std::unique_ptr<ComputeBackend> create();
    ComputeBackendKind kind() const noexcept override;   // Cpu
    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    float sum(std::span<const float> x) const override;
    void linearCombine(std::span<const std::span<const float>> terms,
                       std::span<const float> coefficients, std::span<float> y) const override;

    // Outside ComputeBackend: no GPU backend can offer float64, so these aren't virtual.
    void saxpyD(std::span<const double> x, std::span<double> y, double a) const;
    double sumD(std::span<const double> x) const;
};
```

`sum`/`sumD` accumulate internally at `double` before narrowing back to
`float` for the interface method: a naive `float` accumulator loses far
more over a long run than the internal `double` costs.
`saxpyD`/`sumD` are CPU-specific, outside the polymorphic interface
entirely, for scenarios that must stay on CPU for accuracy regardless of
what hardware is available (long-baseline orbital integration, in
particular):

```cpp
ysq::CpuBackend cpu;
cpu.saxpyD(xd, yd, a);   // double throughout, no GPU equivalent exists
```

## `Compute/Metal/MetalBackend.hpp`

The priority backend on Apple platforms: Apple's native Metal API, the only
one of the four GPU backends that can reach macOS's actual GPU hardware
(OpenGL compute needs a 4.3 context, which Apple never shipped; CUDA needs an
NVIDIA GPU, absent from Apple Silicon and every recent Mac). Compiled in only
when `YSQ_BUILD_COMPUTE_METAL` is set (default `ON`, Apple platforms only).

```cpp
class MetalBackend final : public ComputeBackend {
public:
    static std::unique_ptr<ComputeBackend> create();
    // nullptr if MTLCreateSystemDefaultDevice() returns nil, or any kernel fails to compile
    ComputeBackendKind kind() const noexcept override;   // Metal
    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    float sum(std::span<const float> x) const override;
    void linearCombine(std::span<const std::span<const float>> terms,
                       std::span<const float> coefficients, std::span<float> y) const override;
};
```

`create()` does a genuine device probe (`MTLCreateSystemDefaultDevice()`) and
compiles all three kernels from one inline Metal Shading Language source
string at runtime (`newLibraryWithSource:options:error:`), the same
no-shader-embedding-step convention the OpenGL backend uses for GLSL. `sum`
is the same two-pass tree reduction the OpenGL kernel uses, one threadgroup
reduction plus a CPU finish over the (tiny) per-threadgroup partials.
Objective-C++ (`MetalBackend.mm`); the header stays plain C++ via a pimpl, so
it can be included from any translation unit regardless of language mode.
`MTLResourceStorageModeShared` throughout, the simplest correct choice for a
reference kernel on both unified-memory Apple Silicon and discrete-GPU Intel
Macs, not the fastest one.

## `Compute/OpenGL/OpenGLBackend.hpp` and `ComputeShader.hpp`

A `ComputeBackend` over a 4.3+ offscreen OpenGL context. Compiled in only
under `YSQ_BUILD_GRAPHICS`.

```cpp
class OpenGLBackend final : public ComputeBackend {
public:
    static std::unique_ptr<ComputeBackend> create();
    // nullptr if no 4.3 context is available, or either kernel fails to compile
    ComputeBackendKind kind() const noexcept override;   // OpenGL
    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    float sum(std::span<const float> x) const override;
    void linearCombine(std::span<const std::span<const float>> terms,
                       std::span<const float> coefficients, std::span<float> y) const override;
};
```

`create()` requests a 4.3 offscreen context through `Window::createOffscreen`
(compute shaders are a 4.3 feature). **Always fails on macOS**, which is
capped at OpenGL 4.1: see [docs/api/compute.md](#backend-availability-by-platform)
below and `MetalBackend` above, which is what macOS actually uses.

```cpp
class ComputeShader {
public:
    static std::optional<ComputeShader> compile(std::string_view source, std::string* error = nullptr);
    // move-only

    void use() const;                                          // makes this the active program
    void bindBuffer(unsigned binding, unsigned bufferHandle) const;  // layout(std430, binding = N)
    void setUniform(std::string_view name, float value) const;
    void dispatch(unsigned groupsX, unsigned groupsY = 1, unsigned groupsZ = 1) const;
    // waits for every shader storage write to become visible before returning
};
```

`ComputeShader` never makes its context current itself: the context it was
compiled under must already be current for every method here;
`OpenGLBackend` owns that decision. All three reference kernels are compiled
from inline GLSL string literals in `OpenGLBackend.cpp`, not separate
`.comp` files; `sum` is a two-pass tree reduction, finishing the (tiny)
remaining reduction over per-workgroup partials on the CPU.

## `Compute/CUDA/CudaBackend.hpp` and `Compute/Vulkan/VulkanBackend.hpp`

Real device/instance probes; **kernels not yet implemented**. Compiled in
only when their SDK is found at configure time
(`YSQ_BUILD_COMPUTE_CUDA`/`YSQ_BUILD_COMPUTE_VULKAN`, both default `ON`
meaning "use it if found," never a hard requirement).

```cpp
class CudaBackend final : public ComputeBackend {
public:
    static std::unique_ptr<ComputeBackend> create();
    // nullptr unless cudaGetDeviceCount() reports at least one device
    ComputeBackendKind kind() const noexcept override;   // Cuda
    // saxpy/sum/linearCombine exist to satisfy the interface; not implemented
};

class VulkanBackend final : public ComputeBackend {
public:
    static std::unique_ptr<ComputeBackend> create();
    // nullptr unless a VkInstance can be created with a compute-capable physical device
    ComputeBackendKind kind() const noexcept override;   // Vulkan
    // saxpy/sum/linearCombine exist to satisfy the interface; not implemented
};
```

`create()` on both does a genuine runtime probe (`cudaGetDeviceCount()` for
CUDA; a real `VkInstance` and a physical-device query for Vulkan). Compiling
either backend in only means its SDK was found at configure time, which says
nothing about the machine the binary actually runs on. **Do not call
`saxpy`/`sum`/`linearCombine` on either** expecting a result: no kernel is
written, and since `create()` only succeeds where a device genuinely exists,
the only verified path is "SDK absent, backend compiled out entirely." Use
`selectComputeBackend()` rather than constructing these directly, and it
will never hand you one whose `create()` didn't succeed.

### Backend availability by platform

| Platform | Best compute available |
| --- | --- |
| macOS | **Metal**, Apple's native API — the only one of the four GPU backends that reaches this platform's hardware (OpenGL is capped at 4.1, no compute shaders; no NVIDIA drivers, so CUDA is out) |
| Linux/Windows, NVIDIA | CUDA, then Vulkan, then GL compute |
| Linux/Windows, AMD/Intel | Vulkan, then GL compute |
| Headless server, CI | CPU |

The CPU rung has no hardware requirement, so there is no machine where the
engine fails to start; what varies is throughput, never capability.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/compute)
and let us know.
