# Compute

Backends `Physics` dispatches to: a CPU reference implementation plus GPU
acceleration (OpenGL compute shaders, CUDA, Vulkan). `Physics` never talks to a
GPU directly.

**Target:** `ysq::Compute` (static)
**Depends on:** `ysq::Core` for logging, linked `PRIVATE`. The OpenGL backend
additionally links `ysq::Platform` and `glad`, `PRIVATE`, and exists only in a
graphics build. The CUDA and Vulkan backends link `CUDA::cudart` /
`Vulkan::Vulkan` when those are found at configure time. Deliberately **not**
linked to `Math` or `Units`: the two reference kernels here operate on plain
`float`/`double` spans, so Compute has no dependency on either.

## Contents

| Header                              | Purpose                                             |
| ------------------------------------ | --------------------------------------------------- |
| `Compute/ComputeBackend.hpp`         | The backend interface, `ComputeBackendKind`, selection and availability |
| `Compute/CPU/CpuBackend.hpp`         | Reference implementation, always available           |
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

Two reference kernels, both domain-neutral: `saxpy` (`y[i] = a*x[i] + y[i]`)
checks the dispatch plumbing itself, one invocation per element with no
cross-lane communication; `sum` is a parallel reduction, which is a
meaningfully different shape and the pattern GPU backends actually have to get
right (workgroup-local reduction on OpenGL/Vulkan, block reduction on CUDA).
Neither takes a `Math` vector or a `Physics::Body`, on purpose: `Physics`
doesn't exist yet, so nothing here is shaped around it. When `Physics/Gravity`
lands and needs GPU-accelerated N-body summation, that kernel is added then,
against a data model `Physics` actually settles on.

`selectComputeBackend()` probes `Cuda -> Vulkan -> OpenGL -> Cpu` in that order
and returns the first available; see
[Backend selection and fallback](#backend-selection-and-fallback) below.
`forceBackend` skips probing for debugging and benchmarking, returning nullptr
if that backend genuinely is not available. `computeBackendAvailable(kind)` is
the same probe exposed standalone; for OpenGL/CUDA/Vulkan it opens and
discards a context or device to answer, so it costs more than a flag check and
is meant to be called at startup, not per frame.

## Backend selection and fallback

Selection probes candidates in priority order and takes the first that
reports available, with a manual override (`forceBackend`) for debugging and
benchmarking:

```
CUDA         -> NVIDIA GPU and toolkit present
Vulkan       -> Vulkan 1.1+ loader with a compute queue
OpenGL 4.3+  -> context reports 4.3 or higher (compute shaders)
CPU          -> always succeeds
```

The bottom rung has no hardware requirement, so there is no machine where the
engine fails to start. What varies is throughput, not capability.

| Platform                    | Best compute available          | Rendering    |
| --------------------------- | -------------------------------- | ------------ |
| macOS                       | CPU today; Vulkan via MoltenVK, or a native Metal backend | OpenGL 4.1 |
| Linux / Windows, NVIDIA     | CUDA, then Vulkan, then GL compute | OpenGL 4.6 |
| Linux / Windows, AMD/Intel  | Vulkan, then GL compute          | OpenGL 4.5+  |
| Headless server, CI         | CPU                              | none needed  |

macOS is the constrained case and it is worth being precise about why. Apple
deprecated OpenGL at 4.1 and never shipped compute shaders, which are a 4.3
feature, so the OpenGL compute backend cannot run there (see
[OpenGL backend](#opengl-backend) below). NVIDIA drivers have not been
available for years, so CUDA is permanently out. Vulkan is not native
either, but MoltenVK translates it to Metal and does support compute. So
macOS has a GPU compute path; it just runs through Vulkan or Metal rather
than OpenGL or CUDA.

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

## OpenGL backend

Requests a 4.3 offscreen context through `Window::createOffscreen` (compute
shaders are a 4.3 feature; see the comment on `ContextSettings` in
`Platform/Window.hpp`, which reserves this exact version for this exact
purpose). `create()` returns `nullptr` cleanly if that context can't be had,
which is the normal case on macOS: OpenGL is capped at 4.1 there, so this
backend is compiled in under `YSQ_BUILD_GRAPHICS` but `available()` always
answers false. See [Backend selection and fallback](#backend-selection-and-fallback)
above for the Vulkan/Metal path that exists on macOS instead.

Both kernels are compiled from GLSL source held as inline string literals in
`OpenGLBackend.cpp`, not separate `.comp` files. There is no shader-embedding
build step in the project yet (`Renderer` will need one for its own, likely
larger, shaders), and adding one for two small kernels was more infrastructure
than this stage needed. If `Renderer` establishes that convention later,
these two are small enough to move into it at no cost.

The `sum` kernel is a two-pass tree reduction: one dispatch reduces each
workgroup's slice to a partial sum, and `OpenGLBackend::sum()` finishes the
(tiny) remaining reduction over those partials on the CPU rather than earning
a second shader for it.

## CUDA and Vulkan: real detection, stubbed kernels

Both are compiled only when their SDK is found at configure time
(`YSQ_BUILD_COMPUTE_CUDA` / `YSQ_BUILD_COMPUTE_VULKAN`, both default `ON`
meaning "use it if found," never a hard requirement). Both `create()`
functions do a genuine runtime probe: `CudaBackend` calls
`cudaGetDeviceCount()`, `VulkanBackend` creates a real `VkInstance` and checks
for a physical device with a compute queue family. Neither backend's `saxpy`
or `sum` is implemented; both bodies are `assert(false)`, unreachable in
practice because `create()` only succeeds where a device was actually found.

This machine has neither the CUDA Toolkit nor the Vulkan SDK installed, so
only the "SDK absent, compile it out entirely" path has actually been
exercised. The "SDK present, probe runs against real hardware" path, and the
kernels themselves, are unverified and are a follow-up task once there is
hardware to check them against.

## Warnings

`ysq::Compute` is a real `STATIC` target, unlike `Math` and `Units`, so it
links `ysq::warnings_strict` directly rather than needing a smoke test to
apply the strict set the way an `INTERFACE` library does.

## Tests

`tests/unit/compute_cpu.cpp` checks `CpuBackend` in isolation,
`tests/unit/compute_backend.cpp` checks selection and availability
(including that probing for a backend that is not there degrades to
`nullptr` cleanly rather than crashing, which is testable on any machine).

`tests/integration/compute_backends_agree.cpp` is the test that realises
"a kernel produces matching results on every available backend": it runs
`saxpy` and `sum` on every backend that reports itself available and checks
agreement against the CPU reference within a tolerance, never exact equality.
On a machine with no GPU and no CUDA/Vulkan SDK it finds nothing to compare
and reports itself skipped rather than passing on having checked nothing.
