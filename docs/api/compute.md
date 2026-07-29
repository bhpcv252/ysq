# Compute API reference

Every public type and function in `Compute`: the backend interface `Physics`
dispatches through, plus the CPU, OpenGL, CUDA and Vulkan implementations.
Start with [docs/compute.md](../compute.md) for backend selection and the
fallback ladder; [src/Compute/README.md](../../src/Compute/README.md) covers
platform-by-platform availability and the precision tradeoffs in depth.
`Compute` depends only on `Core`, deliberately not on `Math`/`Units`: the
two reference kernels operate on plain `float`/`double` spans.

## `Compute/ComputeBackend.hpp`

The interface, and backend selection.

```cpp
enum class ComputeBackendKind { Cpu, OpenGL, Cuda, Vulkan };
std::string_view toString(ComputeBackendKind kind) noexcept;

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual ComputeBackendKind kind() const noexcept = 0;

    // y[i] = a * x[i] + y[i]. x and y must be the same length.
    virtual void saxpy(std::span<const float> x, std::span<float> y, float a) const = 0;

    // Sum of every element; zero for an empty span.
    virtual float sum(std::span<const float> x) const = 0;
};

bool computeBackendAvailable(ComputeBackendKind kind);
std::unique_ptr<ComputeBackend>
selectComputeBackend(std::optional<ComputeBackendKind> forceBackend = std::nullopt);
```

| Function | Description |
| --- | --- |
| `computeBackendAvailable(kind)` | Whether `kind` can genuinely be used right now (SDK, driver, hardware all present). Safe before selecting anything. For OpenGL/CUDA/Vulkan this opens and discards a real context or device, so it costs more than a flag check; call it at startup, not per frame. |
| `selectComputeBackend(forceBackend)` | Probes `Cuda -> Vulkan -> OpenGL -> Cpu` in that order and returns the first available. `forceBackend` skips probing and returns exactly that backend (or `nullptr` if unavailable), for debugging/benchmarking. With no override, only returns `nullptr` if `Cpu` itself failed, which doesn't happen. |

```cpp
std::unique_ptr<ysq::ComputeBackend> backend = ysq::selectComputeBackend();
backend->saxpy(x, y, 2.0f);   // y[i] = 2*x[i] + y[i]
const float total = backend->sum(x);
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
};
```

`create()` requests a 4.3 offscreen context through `Window::createOffscreen`
(compute shaders are a 4.3 feature). **Always fails on macOS**, which is
capped at OpenGL 4.1: see [docs/api/compute.md](#backend-availability-by-platform)
below and the Vulkan/Metal path that exists there instead.

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
`OpenGLBackend` owns that decision. Both reference kernels are compiled from
inline GLSL string literals in `OpenGLBackend.cpp`, not separate `.comp`
files; `sum` is a two-pass tree reduction, finishing the (tiny) remaining
reduction over per-workgroup partials on the CPU.

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
    // saxpy/sum exist to satisfy the interface; not implemented
};

class VulkanBackend final : public ComputeBackend {
public:
    static std::unique_ptr<ComputeBackend> create();
    // nullptr unless a VkInstance can be created with a compute-capable physical device
    ComputeBackendKind kind() const noexcept override;   // Vulkan
    // saxpy/sum exist to satisfy the interface; not implemented
};
```

`create()` on both does a genuine runtime probe (`cudaGetDeviceCount()` for
CUDA; a real `VkInstance` and a physical-device query for Vulkan). Compiling
either backend in only means its SDK was found at configure time, which says
nothing about the machine the binary actually runs on. **Do not call
`saxpy`/`sum` on either** expecting a result: neither kernel is written, and
since `create()` only succeeds where a device genuinely exists, the only
verified path is "SDK absent, backend compiled out entirely." Use
`selectComputeBackend()` rather than constructing these directly, and it
will never hand you one whose `create()` didn't succeed.

### Backend availability by platform

| Platform | Best compute available |
| --- | --- |
| macOS | CPU today; Vulkan via MoltenVK is the GPU compute path (OpenGL is capped at 4.1, no compute shaders; no NVIDIA drivers, so CUDA is out) |
| Linux/Windows, NVIDIA | CUDA, then Vulkan, then GL compute |
| Linux/Windows, AMD/Intel | Vulkan, then GL compute |
| Headless server, CI | CPU |

The CPU rung has no hardware requirement, so there is no machine where the
engine fails to start; what varies is throughput, never capability.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/compute)
and let us know.
