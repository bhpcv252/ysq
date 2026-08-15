# Compute: CPU and GPU backends

Where the heavy, repetitive arithmetic actually runs, and why the CPU is
the answer that's always right even when it isn't the fastest.

## The idea

Some calculations are the same small operation repeated over and over, once
per body, once per grid cell, once per pixel: computing the force on every
one of ten thousand particles from every other particle, say. That shape,
"do this one simple thing many times, mostly independently", is exactly
what a GPU is built to do fast, in parallel, and exactly what a CPU is
comparatively slow at once the count gets large.

Not every machine has a capable GPU, though, and `Math` and `Physics` still
have to run correctly everywhere. So neither talks to a GPU directly; each
asks `Compute` for whichever backend is available, and treats every backend
the same way through one interface. The **CPU backend is the reference
implementation, not a fallback of last resort**: it defines what "correct"
means for every calculation, and every GPU backend's result is checked
against it, within a tolerance, never expected to match exactly. That's
also why the entire engine and its test suite build and run correctly on a
machine with no GPU at all.

## What YSQ gives you

`selectComputeBackend()` probes in priority order and returns the first one
that's actually available:

```
Metal        -> Apple platforms, ships with the OS
CUDA         -> NVIDIA GPU and toolkit present
Vulkan       -> Vulkan 1.1+ loader with a compute queue
OpenGL 4.3+  -> context reports 4.3 or higher (compute shaders)
CPU          -> always succeeds
```

The bottom rung has no hardware requirement, so there's no machine this
fails to start on. What changes with the backend is throughput, never
correctness.

| Platform | Best compute available |
| --- | --- |
| macOS | Metal, Apple's native API |
| Linux / Windows, NVIDIA | CUDA, then Vulkan, then OpenGL compute |
| Linux / Windows, AMD/Intel | Vulkan, then OpenGL compute |
| Headless server, CI | CPU |

macOS is worth being precise about: Apple capped OpenGL at 4.1 and never
shipped compute shaders (a 4.3 feature), and NVIDIA drivers haven't been
available there in years, so both OpenGL compute and CUDA are permanently
out. Metal is Apple's own native GPU API, ships with the OS on every Mac
since 2012, and is the only one of the four GPU backends that can actually
reach macOS hardware, so it's both the answer and, in practice, always
there.

## Using it

```cpp
#include <Compute/ComputeBackend.hpp>

std::unique_ptr<ysq::ComputeBackend> backend = ysq::selectComputeBackend();
backend->saxpy(x, y, 2.0f);          // y[i] = 2*x[i] + y[i]
const float total = backend->sum(x); // a parallel reduction
```

Every backend takes `float` in, `float` out, deliberately: consumer GPUs
are commonly far weaker at `float64` than `float32`, often by a factor of
32 to 64, so a `double` interface isn't something a GPU backend could
implement well even in principle. For the cases that genuinely need
`double` regardless of hardware, long-baseline orbital integration in
particular, `CpuBackend` exposes `saxpyD`/`sumD` outside the shared
interface entirely, since no GPU backend could offer them anyway.

`selectComputeBackend` also takes an optional `ComputeBackendKind` to skip
probing and force a specific backend, for debugging and benchmarking:
`selectComputeBackend(ysq::ComputeBackendKind::Cpu)` returns `nullptr` if
that backend genuinely isn't available rather than silently substituting
another one. For repeated automatic dispatch, `defaultBackend()` resolves
once and caches the result, rather than reprobing (opening and discarding a
real device or context) on every call:

```cpp
ysq::ComputeBackend& backend = ysq::defaultBackend();
```

This is what `Math` and `Physics` route through internally, so a caller
never has to think about which backend is running underneath.
`Gravity::newtonianAccelerations()`/`NewtonianField`,
`Electromagnetism::electricFields()`/`magneticFields()`,
`Mechanics::Hermite`'s per-body scheduler, `Fluids::SPH`'s
`computeDensityAndPressure()`/`pressureAccelerations()`, the grid-stencil
timesteppers `Thermodynamics::HeatEquation3D::step()`,
`Acoustics::AcousticField3D::step()`, `Electromagnetism::MaxwellField3D::step()`,
and `Fluids::EulerianFluid3D::step()` (three dispatches per step, one per
sweep axis), and `Math`'s own
`Multigrid.hpp` transfer operators (`detail::restrictGrid`/`prolongateAndAdd`,
the equation-independent part of a multigrid V-cycle), `FFT.hpp`'s
`fft`/`ifft`/`fft3D`/`ifft3D`, `LinearSolve.hpp`'s matrix-vector/matrix-
matrix `operator*` and `luDecompose`/`choleskyDecompose`, `Eigen.hpp`'s
`qrDecompose`/`jacobiEigenSymmetric`/`svd`, `SpecialFunctions.hpp`'s
batched `erf`/`erfc`/`gamma`/`logGamma`/`legendreP` overloads,
`Polynomial::operator()`'s batched overload, `CubicSpline::operator()`'s
batched overload, `Random.hpp`'s `parallelUniformReal`/`parallelNormal`
(a second, additional RNG family built on a counter-based generator,
Philox4x32-10, that's embarrassingly parallel by construction, alongside
the existing sequential `RandomEngine`), and `Sort.hpp`'s `sortInPlace`
(via bitonic sort, which `Statistics.hpp`'s `quantile`/`median` and
`Sort.hpp`'s own `kthSmallest`/`kthLargest` build on top of rather than
needing any GPU dispatch of their own) (only when instantiated for
`Complex<float>`/`MatrixN<float>`/`float`; a `double` call never dispatches,
since the GPU interface is `float`-only) already do this today: above a size
threshold, they dispatch through Compute automatically; below it, or
whenever a body has properties (like gravitational
oblateness) the GPU kernel doesn't model, they fall back to the plain CPU
path, with no change to what a caller sees or calls. Multigrid's own
relaxation step never dispatches, regardless of grid size: it's a
caller-supplied lambda closing over whatever equation is being solved, so
there is no one fixed piece of math to compile into a GPU kernel for it.

## Go deeper

[docs/api/compute.md](api/compute.md) has every signature: `ComputeBackend`,
each backend's constructor and availability, and the platform availability
table.

[src/Compute/README.md](../src/Compute/README.md) has the full interface,
the Metal and OpenGL backends' two-pass reduction, and the current state of
the CUDA and Vulkan backends (real device detection, kernels not yet
implemented, since no machine this was built on has either SDK installed to
test against).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+compute)
and let us know.
