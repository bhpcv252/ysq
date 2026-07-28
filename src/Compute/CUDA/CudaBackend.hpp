#pragma once

#include <Compute/ComputeBackend.hpp>

namespace ysq {

/// Compiled only when CUDAToolkit is found (src/Compute/CMakeLists.txt).
/// create() does a real device probe (cudaGetDeviceCount), but neither kernel
/// is implemented yet: no CUDA hardware or toolkit was available to verify
/// against while writing this. See src/Compute/README.md.
class CudaBackend final : public ComputeBackend {
public:
    /// Nullptr unless cudaGetDeviceCount() reports at least one device.
    /// Compiling this in only means the toolkit was found; it says nothing
    /// about the machine the binary actually runs on.
    [[nodiscard]] static std::unique_ptr<ComputeBackend> create();

    [[nodiscard]] ComputeBackendKind kind() const noexcept override {
        return ComputeBackendKind::Cuda;
    }

    // Unreachable in practice: create() only succeeds where a device exists,
    // and the kernels themselves are not written yet.
    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    [[nodiscard]] float sum(std::span<const float> x) const override;
};

}  // namespace ysq
