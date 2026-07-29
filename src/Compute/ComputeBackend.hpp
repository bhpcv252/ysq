#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace ysq {

/// Which compute backend produced a result, and the priority selectComputeBackend
/// tries them in: Cuda, then Vulkan, then OpenGL, then Cpu, which always
/// succeeds. See src/Compute/README.md.
enum class ComputeBackendKind { Cpu, OpenGL, Cuda, Vulkan };

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

}  // namespace ysq
