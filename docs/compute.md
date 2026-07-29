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

Not every machine has a capable GPU, though, and `Physics` still has to run
correctly everywhere. So `Physics` never talks to a GPU directly; it asks
`Compute` for whichever backend is available, and treats every backend the
same way through one interface. The **CPU backend is the reference
implementation, not a fallback of last resort**: it defines what "correct"
means for every calculation, and every GPU backend's result is checked
against it, within a tolerance, never expected to match exactly. That's
also why the entire engine and its test suite build and run correctly on a
machine with no GPU at all.

## What YSQ gives you

`selectComputeBackend()` probes in priority order and returns the first one
that's actually available:

```
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
| macOS | CPU today; Vulkan via MoltenVK, or a native Metal backend |
| Linux / Windows, NVIDIA | CUDA, then Vulkan, then OpenGL compute |
| Linux / Windows, AMD/Intel | Vulkan, then OpenGL compute |
| Headless server, CI | CPU |

macOS is the constrained case: Apple capped OpenGL at 4.1 and never shipped
compute shaders (a 4.3 feature), and NVIDIA drivers haven't been available
there in years, so both OpenGL compute and CUDA are out. Vulkan still works,
translated to Metal by MoltenVK, so macOS does have a GPU compute path, just
not through OpenGL or CUDA.

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
another one.

## Go deeper

[src/Compute/README.md](../src/Compute/README.md) has the full interface,
the OpenGL backend's two-pass reduction, and the current state of the CUDA
and Vulkan backends (real device detection, kernels not yet implemented,
since no machine this was built on has either SDK installed to test
against).

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+compute)
and let us know.
