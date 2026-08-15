# Compute

Backends `Math` and `Physics` dispatch to: a CPU reference implementation plus
GPU acceleration (Metal, OpenGL compute shaders, CUDA, Vulkan). Neither talks
to a GPU directly.

**Target:** `ysq::Compute` (static)
**Depends on:** `ysq::Core` for logging, linked `PRIVATE`. The OpenGL backend
additionally links `ysq::Platform` and `glad`, `PRIVATE`, and exists only in a
graphics build. The Metal backend links Apple's `Metal`/`Foundation`
frameworks, Apple platforms only. The CUDA and Vulkan backends link
`CUDA::cudart` / `Vulkan::Vulkan` when those are found at configure time.
Deliberately **not** linked to `Math` or `Units`: every reference kernel here
operates on plain `float`/`double` spans, so Compute has no dependency on
either. (`Math` links `Compute`, the other direction; see
`src/Math/README.md`.)

## Contents

| Header                              | Purpose                                             |
| ------------------------------------ | --------------------------------------------------- |
| `Compute/ComputeBackend.hpp`         | The backend interface, `ComputeBackendKind`, selection, availability, and `defaultBackend()` |
| `Compute/CPU/CpuBackend.hpp`         | Reference implementation, always available           |
| `Compute/Metal/MetalBackend.hpp`     | `ComputeBackend` over Apple's native Metal API — the priority backend on Apple platforms |
| `Compute/OpenGL/ComputeShader.hpp`   | Compile/link/dispatch a GLSL compute program          |
| `Compute/OpenGL/OpenGLBackend.hpp`   | `ComputeBackend` over a 4.3+ offscreen context          |
| `Compute/CUDA/CudaBackend.hpp`       | Device probe; kernels are stubbed, see below          |
| `Compute/Vulkan/VulkanBackend.hpp`   | Device probe; kernels are stubbed, see below          |

## `ComputeBackend`

```cpp
std::unique_ptr<ysq::ComputeBackend> backend = ysq::selectComputeBackend();
backend->saxpy(x, y, 2.0f);         // y[i] = 2*x[i] + y[i]
const float total = backend->sum(x);
```

The one interface in the engine core built on runtime polymorphism. Everywhere
else a compile-time concept is enough (`Numeric`, for instance) because the
choice is known at compile time; here it genuinely is not, since which backend
a machine can offer is discovered at run time.

Three domain-neutral kernels, taking no `Math` vector or `Physics::Body`,
shaped around no one consumer: `saxpy` (`y[i] = a*x[i] + y[i]`) checks the
dispatch plumbing itself, one invocation per element with no cross-lane
communication; `sum` is a parallel reduction, a meaningfully different shape
(workgroup-local reduction on OpenGL/Metal, block reduction on CUDA/Vulkan);
`linearCombine` is a fused multi-term weighted sum, `y[i] = sum_k
coefficients[k] * terms[k][i]` for one to four terms, added for the
RK4-family integrator steps `Math/Integrators` decomposes into, one dispatch
instead of one `saxpy` per stage. `minIndex` is a fourth reduction,
`sum`'s two-pass shape with a different operator: the index of the smallest
element, for `Physics/Mechanics/Hermite.hpp`'s per-body scheduler.

Three more kernels are physics-shaped, one pairwise direct-summation (SoA
positions in, a per-body field or acceleration out) per force law:
`gravitationalNBody` (Plummer-softened Newtonian gravity, point masses only;
see `Physics/Gravity/Newtonian.cpp` for why an oblate body stays on the CPU
path), `electricFieldNBody` (Coulomb's law), and `magneticFieldNBody` (the
point-charge form of Biot-Savart). All three are `Physics/Gravity/Newtonian.hpp`
and `Physics/Electromagnetism/Field.hpp`'s actual GPU-dispatch path, not
placeholders: `newtonianAccelerations()`, `NewtonianField`, `electricFields()`
and `magneticFields()` all route through `Compute::defaultBackend()` above a
size threshold. Two more are neighbor-bounded rather than all-to-all: `sphDensityPressure`
and `sphPressureAcceleration`, matching `Physics/Fluids/SPH.hpp`'s cubic
spline kernel sum exactly. Direct O(n^2) with a distance cutoff at the
kernel's compact support, deliberately **not** accelerated by a spatial
hash grid: a real one (binning, sorted traversal) is a legitimate future
upgrade for large particle counts, not a correctness gap in what's here now
— the physics is exact either way, only the asymptotic scaling differs.

Three more are grid stencils: one dispatch per timestep, a fixed 3D array in,
the same-shaped array out, each cell updated from its immediate neighbors.
`heatEquation3DStep` is the explicit-Euler diffusion update (one buffer in,
one out), matching `Physics/Thermodynamics/HeatEquation3D.cpp`'s CPU loop
exactly. `acoustic3DStep` and `maxwell3DStep` are both leapfrog FDTD
updates with two coupled fields (pressure/velocity, E/B): each is one
`ComputeBackend` method that dispatches two kernels in sequence inside a
single call, the second reading the first's output, matching
`Physics/Acoustics/Acoustic3D.cpp` and `Physics/Electromagnetism/Maxwell3D.cpp`'s
own two-pass CPU update. `eulerianFluid3DSweep` is a fourth: one dimensional
-split finite-volume sweep of the compressible Euler equations (first-order
Rusanov flux), matching `Physics/Fluids/Eulerian3D.cpp`'s own `sweep()` for
one axis exactly. Unlike the other three, it takes an `axis` parameter
rather than fixing the stencil shape at compile time, since
`EulerianFluid3D::step()` calls the same kernel three times (x, then y,
then z, dimensional/Godunov splitting) with the momentum components
permuted each time; it is also the one kernel here that recomputes each
face's flux independently at both of the two cells that share it rather
than writing it once to a shared buffer, an embarrassingly-parallel-per-cell
shape that avoids a race on that shared face without needing a second
kernel dispatch. All four grid-stencil kernels are periodic on every axis;
unlike the CPU path's `Grid3D` ghost cells, the GPU kernels compute periodic
wraparound directly with modular index arithmetic (`(i + n - 1) % n`),
since allocating and refreshing ghost layers on the GPU side would cost
more than the modulo. `Physics/Spacetime`'s BSSN evolution is a grid stencil too but is
deliberately not GPU-accelerated here: its update touches far more state
per cell (the full set of ADM/BSSN variables plus constraint terms) than a
single-scalar or two-field FDTD step, and transcribing it correctly from
the CPU reference was judged too large and too risky to do by hand
alongside the other kernels in this pass; it stays CPU-only until that can
be done as its own focused piece of work. Likewise, `Physics/Thermodynamics`
and `Physics/Fluids`' 1D stencil variants stay CPU-only: the grid sizes
they run at rarely cross a size where GPU dispatch overhead pays for
itself, so a dedicated kernel for them was not worth adding speculatively.

Two more are the geometric-multigrid transfer operators, `multigridRestrict3D`
and `multigridProlongateAndAdd3D`, one thread per coarse (restrict) or fine
(prolongate) cell: the straight 8-cell average and the matching correction
broadcast a coarsening-by-2 cell-centered grid needs, matching
`Math/Multigrid.hpp`'s `detail::restrictGrid`/`detail::prolongateAndAdd`
exactly. Unlike every other kernel above, these are not Physics-shaped: they
live in `Math` because they are equation-independent, the same restriction
and prolongation any FAS multigrid V-cycle needs regardless of which
elliptic equation it's solving, not something Physics content (`Compute`
still has no dependency on `Math`/`Units` itself; `Math/Multigrid.hpp` is
the consumer, dispatching through `Compute::defaultBackend()` the same way
Physics' own kernels do). The relaxation step (`solveFAS`'s `relaxPoint`) is
deliberately **not** a kernel here: it is a caller-supplied lambda closing
over whichever equation is being solved (`Physics/Spacetime/PunctureInitialData.cpp`'s
Hamiltonian constraint, today), so there is no single fixed piece of math to
compile into a kernel the way there is for `saxpy` or `heatEquation3DStep` --
a GPU kernel has to be one concrete calculation, and "call an arbitrary
caller-supplied function" is not one. It stays CPU-only, always, regardless
of grid size, for every equation `Math/Multigrid.hpp` is ever used to solve.

One more is `fftBatched`: a batched, power-of-two, radix-2 Cooley-Tukey
FFT, matching `Math/FFT.hpp`'s `detail::fftImpl` exactly per batch. Like
the multigrid transfer operators, this lives in `Math`, not Physics: an
FFT has no physical meaning of its own, and `Math/FFT.hpp`'s own consumers
(`Physics/QuantumMechanics/WavePacket.cpp`/`WavePacket3D.cpp`,
`Physics/Optics/Diffraction.cpp`) are exactly that, consumers, the same
relationship `Math/Multigrid.hpp` has with `PunctureInitialData.cpp`.
Internally the interface's one method dispatches `log2(length) + 2` GPU
kernels in sequence for one call (a bit-reversal permutation, one butterfly
pass per stage, and, for an inverse transform, a final `1/length` scale),
ping-ponging between two buffers across stages since each stage's kernel
is out-of-place and reads the previous stage's complete output; every
other multi-pass kernel in this file (`acoustic3DStep`, `maxwell3DStep`)
fixes its pass count at exactly two, so this is the one kernel whose
internal dispatch count depends on its own input size. `Math::fft`/`ifft`
(1D) and `fft3D`/`ifft3D` (the row-column algorithm, three batched 1D
passes: `fft3D`'s own `x` and `y` passes extract their strided lines into
a contiguous batch first and scatter the result back, the same reason
`detail::fft3DImpl` uses scratch line buffers on the CPU side; the `z`
pass needs no extraction, since the flat array layout already *is* a
batch-of-lines for that axis) dispatch through this above a size
threshold, but **only when instantiated for `Complex<float>`**: the
interface is `float`-only throughout this file, so a `Complex<double>`
call would silently narrow if it dispatched, the same reason `CpuBackend`
keeps `saxpyD`/`sumD` outside the shared interface rather than routing a
`double` caller through a `float` kernel. `Math/FFT.hpp` gates this with
`if constexpr` so the `double` instantiation (`Complex<double>`, what
every current real consumer above actually uses) never mentions Compute
at all; a future `Complex<float>` consumer gets the same automatic
dispatch every other kernel here provides.

The last four are dense linear algebra, matching `Math/LinearSolve.hpp`'s
`MatrixN`/`VectorN` operations exactly; like the FFT and multigrid
kernels, these live in `Math`, not Physics, and dispatch only for
`MatrixN<float>`/`VectorN<float>`, gated the same `if constexpr` way, for
the same reason. `matVec` and `matMul` are ordinary one-thread-per-output-
element kernels, no different in shape from `gravitationalNBody` or any
other reduction-per-output kernel here; `matMul` is a naive O(rows*cols*cols)
kernel, not a tiled/shared-memory GEMM, the same reference-over-throughput
choice made everywhere else in this file. `luDecomposeGpu` and
`choleskyDecomposeGpu` are the two genuinely different kernels in this
catalog: LU/Cholesky elimination is `n` *sequential* steps (step `k+1`
cannot start before step `k`'s full trailing-submatrix update is done, a
property of the algorithm itself, not an implementation shortcut), so
unlike every dispatch above -- which is either one kernel or a small fixed
number run once -- these run a full host-side loop of GPU dispatches, one
loop iteration per row/column of the matrix. Within one LU step, three
things happen: a reduction finds the pivot row (the largest-magnitude
remaining entry in the current column -- structurally `minIndex`'s own
two-pass reduce-then-finish-on-CPU shape, adapted to a strided column and
a max instead of a min, and the reason `luDecomposeGpu` needs one small
GPU-to-CPU readback per step, the same as `minIndex` needs one per call),
a row swap, and a trailing-submatrix elimination update, each an ordinary
per-element kernel; all three ping-pong between two buffers across steps,
like `fftBatched`. Cholesky needs no pivoting (positive-definiteness alone
keeps every pivot valid), so it has no reduction step and one fewer kernel
per iteration: a small dispatch computes one column's diagonal entry
(reading back that single value to check the non-positive-definite case,
matching the CPU reference's own check), then a second dispatch computes
every entry below it in parallel; since Cholesky's writes never depend on
being read back later by an *earlier* step, both dispatches accumulate
into one resident buffer in place across the whole decomposition rather
than ping-ponging. Both factor `n` sequential steps of otherwise-ordinary
kernels, the same building blocks already proven correct elsewhere in this
file (a reduction, a passthrough-guarded elementwise update) chained by a
host loop, not a fundamentally different kind of kernel -- more dispatches
than anything else here, not a riskier design.

The last three are dense eigendecomposition, matching `Math/Eigen.hpp`'s
`qrDecompose`, `jacobiEigenSymmetric`, and `svd` exactly; like every other
`Math`-owned kernel above, they dispatch only for `MatrixN<float>`.
`qrDecomposeGpu` is structurally the same `n`-sequential-steps-of-
otherwise-ordinary-kernels shape as `luDecomposeGpu`: a small reduction
finds a column's norm, then two per-column/per-row kernels apply the same
Householder reflection to the working `R` and accumulate it into `Q`,
reading the reflection vector from `R`'s own (pre-this-step) column rather
than a separate buffer. `jacobiEigenSymmetricGpu` and `jacobiSvdGpu`
replace `Math/Eigen.hpp`'s *cyclic* Jacobi ordering (one rotation at a
time, each depending on the last) with the standard round-robin/tournament
schedule (Brent & Luk 1985): every sweep is `n - 1` rounds, each pairing
every index with exactly one other via a schedule computed on the host and
uploaded as a small buffer, so the `n / 2` rotations within one round touch
entirely disjoint rows and columns and can run as a single dispatch. The
two kernels are not equally simple, though: SVD's rotation is one-sided
(`a <- a * q`, columns only), so disjoint pairs never share a cross term
and one dispatch per round is genuinely independent column updates.
Symmetric eigendecomposition's rotation is two-sided (`a <- q^T a q`), and
naively applying several simultaneous disjoint rotations by having each
thread mix its own rows *and* columns in one pass computes the wrong
answer for the four entries connecting one round's two pairs (this was
caught, not theorized -- an earlier draft did exactly that and failed
integration tests against the CPU reference before being corrected). The
fix is two passes: a column-mix computing `b = a * q` first (also mixing
the accumulated eigenvectors the same way, a pure right-multiply, in the
same dispatch), then a row-mix computing `a' = q^T * b` from that pass's
complete output, both independently recomputing the same rotation angle
from the pair's original pre-round values rather than communicating it.

The last seven are batched, independent, pointwise evaluation: `batchErf`,
`batchErfc`, `batchGamma`, `batchLogGamma`, `batchLegendreP`,
`batchPolynomialEval`, and `batchCubicSplineEval`, matching
`Math/SpecialFunctions.hpp`'s `erf`/`erfc`/`gamma`/`logGamma`/`legendreP`,
`Math/Polynomial.hpp`'s `Polynomial::operator()`, and
`Math/Interpolation.hpp`'s `CubicSpline::operator()` respectively, each
gaining a batched overload (`std::span<const T> x, std::span<T> result`)
alongside its existing scalar one that dispatches through this above a size
threshold, the same `if constexpr`-gated, `float`-only policy as every
other `Math`-owned kernel here. Structurally these are the simplest kernel
shape in this file: one thread per element of `x`, no reduction, no
multi-pass structure, and (for `batchPolynomialEval`/`batchCubicSplineEval`)
the same fixed polynomial/spline evaluated at every element. `erf`/`erfc`
use the Abramowitz & Stegun 7.1.26 rational approximation and
`gamma`/`logGamma` the Lanczos approximation (reflected via `gamma(x) = pi
/ (sin(pi x) gamma(1-x))` for `x <= 0`) rather than
`Math/SpecialFunctions.hpp`'s own `std::erf`/`std::tgamma`/`std::lgamma`
calls, since neither Metal Shading Language nor GLSL has a C99 math
library to call into; both are well-conditioned in `float32` (no
intermediate cancellation between differently-scaled terms). Near a pole of
`gamma`/`logGamma` (a negative integer), the reflection formula's `1 /
sin(pi x)` amplifies `x`'s ordinary float32 rounding into a large swing in
an already-large result -- expected and unavoidable in any float32
implementation, not a bug, which is why
`tests/integration/compute_backends_agree.cpp`'s agreement checks for these
two use a relative tolerance rather than the fixed absolute one every other
kernel here uses. `Math/SpecialFunctions.hpp`'s `besselJ`/`besselY`
deliberately have **no** batched kernel here at all: their existing
rational-polynomial approximation (Abramowitz & Stegun 9.4.1-9.4.6) sums
terms that individually reach ~1e11 in magnitude to produce an O(1) result
-- numerically fine in the `double` it already runs in, but this
cancellation would consume `float32`'s entire ~7 digits of precision and
leave nothing for the answer itself. A `float32`-safe Bessel approximation
(a power series for small `x`, say) is a distinct, independently-verified
numerical method, not a mechanical port of the existing one, and is out of
scope for this pass.

The last two are parallel, counter-based random number generation:
`batchUniformReal` and `batchNormal`, both built on Philox4x32-10 (Salmon,
Moraes, Hadjidoukas & Schulten 2011). This is a second, additional RNG
family alongside `Math/Random.hpp`'s existing `RandomEngine`
(`std::mt19937_64`), not a replacement: `RandomEngine` is one strictly
sequential stream, useful for exactly that reason (a caller threading one
engine through a run gets bit-for-bit reproducible output), but a
sequential stream is exactly what cannot be split across GPU threads
without a full port of Mersenne Twister's own state machine and a
jump-ahead capability it does not straightforwardly offer. Philox sidesteps
the problem by being counter-based rather than sequential-state-based at
all: thread `tid`'s output depends only on `(seed, offset + tid)`, so
generating `n` values is `n` completely independent evaluations, one per
thread, no different in shape from the batch-evaluation family above,
except that neither kernel takes a per-element input buffer at all (there
is nothing to read; `seed` and `offset` alone determine every output).
`batchUniformReal` produces `[0, 1)`; `batchNormal` produces standard
normal via the Box-Muller transform, spending two of Philox's four 32-bit
output words on one normal value. `offset` is what makes this a genuine
*stream*, not just an independent draw per call: calling
`batchUniformReal(seed, 0, firstHalf)` then
`batchUniformReal(seed, firstHalf.size(), secondHalf)` produces exactly the
same values as one call for the concatenation, verified directly in
`tests/unit/compute_cpu.cpp`. Philox itself is pure 32-bit bit arithmetic
(a Feistel-style multiply-and-permute over a counter and key), so unlike
the batch-evaluation family there is no `float32`-suitability question at
all -- `mulhi32` (the high 32 bits of an unsigned 32x32 multiply) is
computed via 16-bit-limb schoolbook long multiplication rather than a
64-bit intermediate specifically so the identical kernel source compiles
unchanged on both Metal (which does support 64-bit integers on Apple GPUs)
and GLSL 430 (which does not, without an extension this engine does not
require), and this makes GPU and CPU outputs agree to a tight tolerance
(`tests/integration/compute_backends_agree.cpp` uses `1e-6f`, not the
looser tolerances the batch-evaluation family needs) rather than an
algorithmic-agreement tolerance. `seed`/`offset` travel as a small 4-element
`uint` SSBO on OpenGL rather than uniforms, since `ComputeShader` has no
`uint` uniform setter and this reuses the uint-buffer upload path
`jacobiEigenSymmetricGpu`/`jacobiSvdGpu`'s round-robin schedule already
established, rather than introducing an int/uint bit-reinterpretation
uniform trick for the first time here.

The last one is `sortAscending`: ascending sort, in place, via bitonic sort
(Batcher 1968), the standard data-parallel sorting network, matching
`Math/Sort.hpp`'s `sortInPlace`. Unlike every other kernel here,
`sortAscending`'s interface accepts *any* `values.size()`, not just a
power of two, even though bitonic sort itself only naturally handles
power-of-two sizes: each GPU backend pads internally with `+infinity`
sentinels up to the next power of two, runs the sort, and truncates back
before returning, entirely invisibly to the caller. This is safe in a way
zero-padding an FFT is not: `+infinity` sentinels are guaranteed to sort
to the very end and never displace any of the first `values.size()`
elements, so the padding has zero effect on the answer, not merely a
negligible one, which is why the padding lives inside each backend rather
than being pushed up to the caller as a documented restriction the way
`fftBatched`'s power-of-two-length requirement is. One
`bitonic_compare_exchange_kernel` compare-exchange pass per dispatch,
`log2(n) * (log2(n) + 1) / 2` dispatches total (`sortAscending`'s own host
loop) -- one more nested loop level than `fftBatched`'s own `log2(length)`
butterfly passes, since a bitonic sort builds increasingly large bitonic
sequences (`stageSize = 2, 4, 8, ..., n`) and then merges each one down
through decreasing compare distances (`stepSize = stageSize/2, ..., 1`),
where an FFT's butterfly network has only the one loop level. Thread `tid`
only acts when it is the lower index of its compare-exchange pair (`tid <
tid ^ stepSize`); ascending/descending direction alternates in blocks of
`stageSize`, exactly Batcher's original construction. `Math/Sort.hpp`'s own
`kthSmallest`/`kthLargest` and `Math/Statistics.hpp`'s `quantile`/`median`
need no kernel of their own at all: they're built entirely on top of
`sortInPlace`, the same "no bespoke kernel, composes an already-dispatching
building block" design `Math/Eigen.hpp`'s `generalEigenvalues` uses for
`qrDecompose`.

When a new Physics-specific shape is needed beyond these, it's added the
same way all of the above were: against the data model its consumer
actually settled on, not speculatively ahead of one.

`Math/Eigen.hpp`'s `realSchur`/`generalEigenvalues` (the non-symmetric
eigenvalue case) have no kernel of their own here, deliberately: Hessenberg
reduction is a single one-time pass (not repeated, so not worth a bespoke
kernel the way LU's repeated trailing-submatrix update was), and the
dominant cost is up to `maxIterations` repeated *internal* calls to
`qrDecompose` on a shrinking leading block -- which already dispatch
through `qrDecomposeGpu` transitively, above its own threshold, with no
extra code. `tests/unit/math_eigen.cpp` has a test exercising exactly this
composition at a size that starts above `qrDecompose`'s threshold and
shrinks below it within the same `generalEigenvalues` call.

`selectComputeBackend()` probes `Metal -> Cuda -> Vulkan -> OpenGL -> Cpu` in
that order and returns the first available; see
[Backend selection and fallback](#backend-selection-and-fallback) below.
`forceBackend` skips probing for debugging and benchmarking, returning nullptr
if that backend genuinely is not available. `computeBackendAvailable(kind)` is
the same probe exposed standalone; for Metal/OpenGL/CUDA/Vulkan it opens and
discards a context or device to answer, so it costs more than a flag check and
is meant to be called at startup, not per frame. `defaultBackend()` is the
process-wide cached version of `selectComputeBackend()`: resolved once, on
first call, and reused after that, which is what `Math` and `Physics` route
automatic dispatch through rather than reprobing on every call.

## Backend selection and fallback

Selection probes candidates in priority order and takes the first that
reports available, with a manual override (`forceBackend`) for debugging and
benchmarking:

```
Metal        -> Apple platforms, effectively always available (ships with the OS)
CUDA         -> NVIDIA GPU and toolkit present
Vulkan       -> Vulkan 1.1+ loader with a compute queue
OpenGL 4.3+  -> context reports 4.3 or higher (compute shaders)
CPU          -> always succeeds
```

The bottom rung has no hardware requirement, so there is no machine where the
engine fails to start. What varies is throughput, not capability.

| Platform                    | Best compute available          | Rendering    |
| --------------------------- | -------------------------------- | ------------ |
| macOS                       | Metal, Apple's native API         | OpenGL 4.1 |
| Linux / Windows, NVIDIA     | CUDA, then Vulkan, then GL compute | OpenGL 4.6 |
| Linux / Windows, AMD/Intel  | Vulkan, then GL compute          | OpenGL 4.5+  |
| Headless server, CI         | CPU                              | none needed  |

macOS is worth being precise about, since three of the four GPU backends
cannot reach it at all. Apple deprecated OpenGL at 4.1 and never shipped
compute shaders, which are a 4.3 feature, so the OpenGL compute backend
cannot run there (see [OpenGL backend](#opengl-backend) below). NVIDIA
drivers have not been available for years, so CUDA is permanently out.
Metal is Apple's own native GPU API and ships with the OS on every Mac since
2012, so it is both the only backend that reaches macOS hardware and, in
practice, always available there; see [Metal backend](#metal-backend) below.

Rendering is a separate axis and is not constrained the same way. OpenGL 4.1
is sufficient for the real-time rasterized visualizer with ImGui panels, so
macOS is a first-class rendering target.

## Precision: float32 uniformly, with a CPU-only float64 escape hatch

The interface is `float` in, `float` out on every backend, so they are
interchangeable at the call site regardless of which one was selected. This
is a deliberate choice: consumer GPUs are commonly weak at `float64`,
often 1/32 to 1/64 of `float32` throughput, so a `double` interface would not
be something a GPU backend could implement well even in principle.

`CpuBackend::sum` still accumulates internally at `double` before narrowing
back to `float`; that is a private implementation detail; the reference has to
be trustworthy, and a naive `float` accumulator loses far more over a long run
than the `double` form costs. `tests/unit/compute_cpu.cpp` has a case
demonstrating the naive form actually going wrong so this claim is not
untested.

`CpuBackend` additionally exposes `saxpyD`/`sumD` over `double`, outside the
`ComputeBackend` interface entirely, since no GPU backend could offer it:

```cpp
ysq::CpuBackend cpu;
cpu.saxpyD(xd, yd, a);      // double throughout, no GPU equivalent exists
```

This is for scenarios that must stay on CPU for accuracy regardless of
hardware (long-baseline orbital integration in particular may need it), and
it is CPU-specific, not part of backend selection.

## Metal backend

`MetalBackend` requests Apple's default device (`MTLCreateSystemDefaultDevice()`)
and compiles all three reference kernels from one inline Metal Shading
Language source string at runtime (`newLibraryWithSource:options:error:`),
the same no-shader-embedding-step convention the OpenGL backend below uses
for GLSL. Compiled in under `YSQ_BUILD_COMPUTE_METAL` (default `ON`,
`if(APPLE)`); the implementation is Objective-C++ (`MetalBackend.mm`) behind
a plain-C++ header via a pimpl (`struct Impl`, defined entirely inside the
`.mm`), so `ComputeBackend.cpp` and every other plain-C++ translation unit
can include `MetalBackend.hpp` without caring about the language mode.

`sum` is the same two-pass tree reduction the OpenGL kernel uses: one
dispatch reduces each threadgroup's slice to a partial sum, finishing the
(tiny) remaining reduction over those partials on the CPU.
`MTLResourceStorageModeShared` is used throughout for buffers, the simplest
correct choice on both unified-memory Apple Silicon and discrete-GPU Intel
Macs, not the fastest one, matching this codebase's stated preference for
reference-kernel correctness over peak throughput elsewhere in this file.

This is the one GPU backend actually verified against real hardware while
writing it (an Apple Silicon Mac): `tests/integration/compute_backends_agree.cpp`
runs for real here, not skipped, and passes.

## OpenGL backend

Requests a 4.3 offscreen context through `Window::createOffscreen` (compute
shaders are a 4.3 feature; see the comment on `ContextSettings` in
`Platform/Window.hpp`, which reserves this exact version for this exact
purpose). `create()` returns `nullptr` cleanly if that context can't be had,
which is the normal case on macOS: OpenGL is capped at 4.1 there, so this
backend is compiled in under `YSQ_BUILD_GRAPHICS` but `available()` always
answers false. See [Metal backend](#metal-backend) above for the path that
exists on macOS instead.

All three kernels are compiled from GLSL source held as inline string
literals in `OpenGLBackend.cpp`, not separate `.comp` files. There is no
shader-embedding build step in the project yet (`Renderer` will need one for
its own, likely larger, shaders), and adding one for three small kernels was
more infrastructure than this stage needed. If `Renderer` establishes that
convention later, these three are small enough to move into it at no cost.

The `sum` kernel is a two-pass tree reduction: one dispatch reduces each
workgroup's slice to a partial sum, and `OpenGLBackend::sum()` finishes the
(tiny) remaining reduction over those partials on the CPU rather than earning
a second shader for it. `linearCombine` always binds four term buffers,
whichever of them the caller didn't supply bound to `y`'s own buffer purely
to give the binding point something legal; a `termCount` uniform gates which
ones the shader body actually reads, so the unused slots are never touched.

## CUDA and Vulkan: real detection, stubbed kernels

Both are compiled only when their SDK is found at configure time
(`YSQ_BUILD_COMPUTE_CUDA` / `YSQ_BUILD_COMPUTE_VULKAN`, both default `ON`
meaning "use it if found," never a hard requirement). Both `create()`
functions do a genuine runtime probe: `CudaBackend` calls
`cudaGetDeviceCount()`, `VulkanBackend` creates a real `VkInstance` and checks
for a physical device with a compute queue family. Neither backend's
`saxpy`, `sum`, or `linearCombine` is implemented; every body is
`assert(false)`, unreachable in practice because `create()` only succeeds
where a device was actually found.

This machine has neither the CUDA Toolkit nor the Vulkan SDK installed, so
only the "SDK absent, compile it out entirely" path has actually been
exercised. The "SDK present, probe runs against real hardware" path, and the
kernels themselves, are unverified and are a follow-up task once there is
hardware to check them against.

## Dispatch threshold calibration

Every `kGpuDispatchThreshold`-family constant across both the `Math`
headers listed in [Tests](#tests) below and the eight Physics files with
their own copy of the same pattern (`Gravity/Newtonian.cpp`,
`Electromagnetism/Field.cpp`, `Fluids/SPH.cpp`,
`Thermodynamics/HeatEquation3D.cpp`, `Acoustics/Acoustic3D.cpp`,
`Electromagnetism/Maxwell3D.cpp`, `Fluids/Eulerian3D.cpp`,
`Mechanics/Hermite.cpp`) was, for a long stretch of this engine's
development, a provisional guess (`4096` or `200000`, mostly, picked with
no measurement behind it at all). `benchmarks/compute_thresholds.cpp`
replaces every one of them with a number derived from actually timing the
CPU reference against the fastest available GPU backend (Metal, on the
machine this was run on) across a range of sizes, for every GPU-dispatching
operation, and reporting the smallest size at which the GPU path won.
Build and run it with:

```sh
cmake -B build-bench -DYSQ_BUILD_BENCHMARKS=ON -DYSQ_BUILD_TESTS=OFF \
    -DYSQ_BUILD_GRAPHICS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench --target benchmark_compute_thresholds
./build-bench/bin/benchmark_compute_thresholds
```

`YSQ_BUILD_BENCHMARKS` defaults `OFF`, the same "off unless you ask for
it" default `YSQ_BUILD_TESTS` already uses: this is a developer tool for
occasionally re-deriving these constants, not something a normal build or
test run needs. It measures at the `Compute` level directly (`CpuBackend`
against `defaultBackend()`) for every `Math`-owned threshold and most
Physics ones, the same size metric each header's own `if constexpr`/`if`
gate compares against (`rows * cols`, `n * n`, `x.size()`, and so on),
rather than going through `MatrixN`/`Polynomial`/`Body`/etc.: the
threshold is fundamentally about kernel compute cost plus GPU dispatch
overhead against CPU compute cost, which the `Compute`-level call captures
directly, and the Math-level marshaling on top of that is already covered
by each header's own `...AtLargeNAgreesWithTheComputeCpuReference`-style
test (see [Tests](#tests)).

**`Compute::CpuBackend`'s reference kernel is not always what the
Physics-level below-threshold path actually runs**, though, and blindly
comparing `Compute::CpuBackend` against `defaultBackend()` gives the wrong
answer wherever it differs: `Gravity/Newtonian.cpp`'s own CPU fallback is
a pairwise (`i < j`) loop that visits each body pair once, roughly half
the arithmetic `CpuBackend::gravitationalNBody`'s straightforward
every-body-against-every-other loop does, and `Fluids/SPH.cpp`'s own CPU
fallback queries a `Math::KdTree3` for each particle's neighbors rather
than scanning every other particle at all, an entirely different
asymptotic shape from `CpuBackend::sphDensityPressure`'s direct O(n^2)
sum. `benchmarks/compute_thresholds.cpp` accounts for both: it doubles
the raw `matVec`-shaped measurement for `Newtonian.cpp` specifically
(documented at the call site) rather than using it directly, and it
relinks against `ysq::Physics`/`ysq::Math` to replicate `SPH.cpp`'s own
`KdTree3` + `cubicSplineKernel` loop verbatim for that one case, rather
than reusing the simpler `CpuBackend` reference the rest of the benchmark
uses. Every other Physics threshold's below-threshold path (`Field.cpp`'s
electric/magnetic fields, the whole grid-stencil family, `Hermite.cpp`'s
`minIndex`-based scheduler) matches its `CpuBackend` kernel's shape
exactly, so those needed no such adjustment.

Results on the development machine (Apple Silicon, Metal backend), and
what changed:

| Constant | Old (guessed) | New (measured) | Note |
| --- | --- | --- | --- |
| `LinearSolve.hpp`'s `kMatVecGpuDispatchThreshold` | 4096 (shared) | 4194304 | split out of one shared constant -- see below |
| `LinearSolve.hpp`'s `kMatMulGpuDispatchThreshold` | 4096 (shared) | 2097152 | split out of one shared constant -- see below |
| `LinearSolve.hpp`'s `kLuCholeskyGpuDispatchThreshold` | 4096 (shared) | 524288 | split out of one shared constant -- see below |
| `Eigen.hpp`'s `kEigenGpuDispatchThreshold` | 4096 | 131072 | floor: GPU never won up to 256x256 |
| `FFT.hpp`'s `kGpuDispatchThreshold` | 4096 | 16384 | a real observed crossover |
| `Multigrid.hpp`'s `kMultigridGpuDispatchThreshold` | 200000 | 4194304 | floor, and higher than the original guess |
| `SpecialFunctions.hpp`'s `kSpecialFunctionsGpuDispatchThreshold` | 4096 | 131072 | floor: GPU never won up to 65536 elements |
| `Polynomial.hpp`'s `kPolynomialGpuDispatchThreshold` | 4096 | 131072 | shares the batch-evaluation family's measured floor |
| `Interpolation.hpp`'s `kInterpolationGpuDispatchThreshold` | 4096 | 131072 | shares the batch-evaluation family's measured floor |
| `Random.hpp`'s `kRandomGpuDispatchThreshold` | 4096 | 32768 | a real observed crossover |
| `Sort.hpp`'s `kSortGpuDispatchThreshold` | 4096 | 131072 | floor: GPU never won up to 65536 elements |
| `Gravity/Newtonian.cpp`'s `kGpuDispatchThreshold` | 1024 | 1024 | raw measurement (512) doubled to account for the real pairwise loop's ~2x advantage -- see above |
| `Electromagnetism/Field.cpp`'s `kGpuDispatchThreshold` | 1024 | 512 | a real observed crossover; shared by electric and magnetic fields |
| `Fluids/SPH.cpp`'s `kGpuDispatchThreshold` | 1024 | 512 | measured against the real `KdTree3`-accelerated CPU path -- see above; lower than a naive comparison would have predicted |
| `Thermodynamics/HeatEquation3D.cpp`'s `kGpuDispatchThreshold` | 200000 | 4194304 | floor: GPU never won up to 128^3 (~2.1M) cells |
| `Acoustics/Acoustic3D.cpp`'s `kGpuDispatchThreshold` | 200000 | 4194304 | floor: GPU never won up to 128^3 cells |
| `Electromagnetism/Maxwell3D.cpp`'s `kGpuDispatchThreshold` | 200000 | 2097152 | a real observed crossover (six field arrays read and written, more per-cell work than the rest of the grid-stencil family) |
| `Fluids/Eulerian3D.cpp`'s `kGpuDispatchThreshold` | 200000 | 262144 | a real observed crossover (equation-of-state and Rusanov-flux work per face, not just neighbor differencing) |
| `Mechanics/Hermite.cpp`'s `kGpuDispatchThreshold` | 4096 | 32768 | floor: `minIndex` never won up to 16384 bodies |

Three things fell out of actually measuring rather than guessing:

**`LinearSolve.hpp` split one shared threshold into three.** `matVec`
(O(n^2)), `matMul` (O(n^3)), and `luDecompose`/`choleskyDecompose` (`n`
sequential host-loop dispatches) turned out to have measurably different
crossover points -- `matVec`'s is roughly 8x `luDecompose`/
`choleskyDecompose`'s -- so the one constant these four dispatch sites
shared since Phase 8 would have forced the most conservative of the three
onto operations that benefit from GPU dispatch far earlier. `Eigen.hpp`'s
three operations (`qrDecompose`, `jacobiEigenSymmetric`, `jacobiSvdGpu`)
measured to the *same* crossover, so that one stayed shared: splitting is
warranted by what the numbers actually show, not applied uniformly on
principle.

**Several of these are measured floors, not observed crossovers.** For
`Eigen.hpp`, `Multigrid.hpp`'s restriction, the batch-evaluation family,
and `Sort.hpp`, the GPU path never actually won at any size the benchmark
tried (up to 256x256 matrices, 128^3 grid cells, 65536 elements, and
65536 elements respectively) -- for the three host-loop-of-many-dispatches
families (`Eigen`, `Sort`) this is per-dispatch CPU/GPU synchronization
overhead compounding across dozens to hundreds of round trips; for
`Multigrid`'s restriction and the batch-evaluation family (both a single
dispatch) it's simply that the per-element work is cheap enough that a
single dispatch's fixed buffer allocation/upload/readback overhead
dominates even at tens of thousands of elements. In both cases the
recorded threshold is double the largest size actually tried, a
deliberately conservative floor rather than an extrapolated guess at
where the true crossover might be -- re-running the benchmark at larger
sizes would narrow these, but at real cost (`Eigen.hpp`'s own
`jacobiEigenSymmetricGpu` and `jacobiSvdGpu` sweeps already took
20-25 seconds *each* at 256x256 in this run). The same is true of
`HeatEquation3D.cpp`, `Acoustic3D.cpp` and `Hermite.cpp` on the Physics
side.

**A CPU reference kernel that doesn't match the real below-threshold
code path gives a wrong answer, not just an imprecise one.** This is the
one genuinely important finding out of the whole exercise, not a detail:
`Gravity/Newtonian.cpp`'s raw measurement (comparing `defaultBackend()`
against `CpuBackend::gravitationalNBody`, a full double-loop) landed at
512, a size where the *real* below-threshold path -- a pairwise loop
doing roughly half that arithmetic -- would very plausibly still have
been faster, not slower; using the raw number unadjusted would have made
the engine dispatch to GPU in cases where the CPU path it actually has
was still the better choice. `Fluids/SPH.cpp` cut the other way: the
intuition going in (see the git history of this constant's own comment,
before this benchmark existed) was that a `KdTree3`-accelerated CPU path
would push the crossover *higher* than a naive O(n^2)-vs-O(n^2)
comparison would suggest, and measuring the real tree-based path directly
showed that intuition was itself wrong -- at the particle densities
tested, the tree's own traversal overhead and double-precision per-pair
kernel cost kept its advantage from mattering much, and the measured
crossover came out lower than `Newtonian.cpp`'s, not higher. Neither
direction was guessable in advance with any confidence; both needed the
real, shape-correct measurement to get right.

Re-run the benchmark and update the affected header(s) if the reference
machine or GPU backend ever changes; each constant's own doc comment
names which benchmark case it came from.

## Warnings

`ysq::Compute` is a real `STATIC` target, unlike `Math` and `Units`, so it
links `ysq::warnings_strict` directly rather than needing a smoke test to
apply the strict set the way an `INTERFACE` library does.

## Tests

`tests/unit/compute_cpu.cpp` checks `CpuBackend` in isolation,
`tests/unit/compute_backend.cpp` checks selection, availability, and
`defaultBackend()`'s caching (including that probing for a backend that is
not there degrades to `nullptr` cleanly rather than crashing, which is
testable on any machine).

`tests/integration/compute_backends_agree.cpp` is the test that realises
"a kernel produces matching results on every available backend": it runs
every kernel above on every backend that reports itself available and
checks agreement against the CPU reference within a tolerance, never exact
equality. On a machine with no GPU and no CUDA/Vulkan SDK it finds nothing
to compare and reports itself skipped rather than passing on having checked
nothing; on Apple hardware, Metal is always exercised for real.

Each Math/Physics consumer that dispatches through Compute (`Multigrid.hpp`,
`FFT.hpp`, `LinearSolve.hpp`, `Eigen.hpp`, `SpecialFunctions.hpp`,
`Polynomial.hpp`, `Interpolation.hpp`, `Random.hpp`, `Sort.hpp`,
`Newtonian.cpp`, `Field.hpp`, `Hermite.cpp`, `SPH.cpp`, `HeatEquation3D.cpp`,
`Acoustic3D.cpp`, `Maxwell3D.cpp`, `Eulerian3D.cpp`) additionally has its own
`...AtLargeNAgreesWithTheComputeCpuReference`-style unit test above that
consumer's own `kGpuDispatchThreshold`, checking not the kernel's math
(already covered above) but that the consumer's marshaling — the data it
packs into the flat `float` buffers it hands `defaultBackend()`, and the
data it reads back out — is wired correctly.
