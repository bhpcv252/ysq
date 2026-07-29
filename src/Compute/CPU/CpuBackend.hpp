#pragma once

#include <Compute/ComputeBackend.hpp>

namespace ysq {

/// The reference implementation. Always available, and correctness is defined
/// by this: every other backend is validated against it within tolerance,
/// never the reverse. See src/Compute/README.md.
class CpuBackend final : public ComputeBackend {
public:
    [[nodiscard]] static std::unique_ptr<ComputeBackend> create();

    [[nodiscard]] ComputeBackendKind kind() const noexcept override {
        return ComputeBackendKind::Cpu;
    }

    void saxpy(std::span<const float> x, std::span<float> y, float a) const override;
    [[nodiscard]] float sum(std::span<const float> x) const override;

    /// Outside ComputeBackend: no GPU backend can offer float64, so these are
    /// not virtual. For scenarios that must stay on CPU for accuracy
    /// regardless of hardware; see src/Compute/README.md.
    void saxpyD(std::span<const double> x, std::span<double> y, double a) const;
    [[nodiscard]] double sumD(std::span<const double> x) const;
};

}  // namespace ysq
