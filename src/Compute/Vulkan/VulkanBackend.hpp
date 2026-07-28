#pragma once

#include <Compute/ComputeBackend.hpp>

namespace ysq {

/// Compiled only when the Vulkan SDK is found (src/Compute/CMakeLists.txt).
/// create() does a real instance-and-device probe, but neither kernel is
/// implemented yet: no SDK was available to verify against while writing
/// this. See src/Compute/README.md.
class VulkanBackend final : public ComputeBackend {
public:
    /// Nullptr unless a VkInstance can be created and reports at least one
    /// physical device with a compute-capable queue family.
    [[nodiscard]] static std::unique_ptr<ComputeBackend> create();

    [[nodiscard]] ComputeBackendKind kind() const noexcept override {
        return ComputeBackendKind::Vulkan;
    }

    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    [[nodiscard]] float sum(std::span<const float> x) const override;
};

}  // namespace ysq
