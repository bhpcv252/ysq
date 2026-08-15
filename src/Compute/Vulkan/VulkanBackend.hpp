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
    void linearCombine(std::span<const std::span<const float>> terms,
                       std::span<const float> coefficients,
                       std::span<float> y) const override;
    void gravitationalNBody(std::span<const float> positionsX,
                            std::span<const float> positionsY,
                            std::span<const float> positionsZ, std::span<const float> gm,
                            float softeningSquared, std::span<float> accelerationsX,
                            std::span<float> accelerationsY,
                            std::span<float> accelerationsZ) const override;
    void electricFieldNBody(std::span<const float> positionsX,
                            std::span<const float> positionsY,
                            std::span<const float> positionsZ,
                            std::span<const float> charge, float coulombConstant,
                            std::span<float> fieldX, std::span<float> fieldY,
                            std::span<float> fieldZ) const override;
    void magneticFieldNBody(
        std::span<const float> positionsX, std::span<const float> positionsY,
        std::span<const float> positionsZ, std::span<const float> velocitiesX,
        std::span<const float> velocitiesY, std::span<const float> velocitiesZ,
        std::span<const float> charge, float permeabilityOver4Pi, std::span<float> fieldX,
        std::span<float> fieldY, std::span<float> fieldZ) const override;
    void sphDensityPressure(std::span<const float> positionsX,
                            std::span<const float> positionsY,
                            std::span<const float> positionsZ,
                            std::span<const float> mass, float smoothingLength,
                            float equationOfStateK, float polytropicIndex,
                            std::span<float> density,
                            std::span<float> pressure) const override;
    void sphPressureAcceleration(
        std::span<const float> positionsX, std::span<const float> positionsY,
        std::span<const float> positionsZ, std::span<const float> mass,
        std::span<const float> density, std::span<const float> pressure,
        float smoothingLength, std::span<float> accelerationsX,
        std::span<float> accelerationsY, std::span<float> accelerationsZ) const override;
    void heatEquation3DStep(std::span<const float> temperature, std::size_t nx,
                            std::size_t ny, std::size_t nz, float factor,
                            std::span<float> next) const override;
    void acoustic3DStep(std::span<const float> pressure, std::span<const float> velocityX,
                        std::span<const float> velocityY,
                        std::span<const float> velocityZ, std::size_t nx, std::size_t ny,
                        std::size_t nz, float velocityFactor, float pressureFactor,
                        std::span<float> nextPressure, std::span<float> nextVelocityX,
                        std::span<float> nextVelocityY,
                        std::span<float> nextVelocityZ) const override;
    void maxwell3DStep(std::span<const float> ex, std::span<const float> ey,
                       std::span<const float> ez, std::span<const float> bx,
                       std::span<const float> by, std::span<const float> bz,
                       std::size_t nx, std::size_t ny, std::size_t nz, float bFactor,
                       float eFactor, std::span<float> nextEx, std::span<float> nextEy,
                       std::span<float> nextEz, std::span<float> nextBx,
                       std::span<float> nextBy, std::span<float> nextBz) const override;
    void eulerianFluid3DSweep(
        std::span<const float> density, std::span<const float> momentumNormal,
        std::span<const float> momentumTangent1, std::span<const float> momentumTangent2,
        std::span<const float> energy, std::size_t nx, std::size_t ny, std::size_t nz,
        int axis, float gamma, float dtOverSpacing, std::span<float> nextDensity,
        std::span<float> nextMomentumNormal, std::span<float> nextMomentumTangent1,
        std::span<float> nextMomentumTangent2,
        std::span<float> nextEnergy) const override;
    void fftBatched(std::span<const float> real, std::span<const float> imag,
                    std::size_t length, std::size_t batchCount, bool inverse,
                    std::span<float> nextReal, std::span<float> nextImag) const override;
    void matVec(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                std::span<const float> vector, std::span<float> result) const override;
    void matMul(std::span<const float> a, std::size_t aRows, std::size_t aCols,
                std::span<const float> b, std::size_t bCols,
                std::span<float> result) const override;
    [[nodiscard]] bool luDecomposeGpu(std::span<const float> matrix, std::size_t n,
                                      std::span<float> lu,
                                      std::span<std::uint32_t> pivot) const override;
    [[nodiscard]] bool choleskyDecomposeGpu(std::span<const float> a, std::size_t n,
                                            std::span<float> l) const override;
    void qrDecomposeGpu(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                        std::span<float> q, std::span<float> r) const override;
    void jacobiEigenSymmetricGpu(std::span<const float> matrix, std::size_t n,
                                 int maxSweeps, float tolerance,
                                 std::span<float> resultDiagonal,
                                 std::span<float> resultEigenvectors) const override;
    void jacobiSvdGpu(std::span<const float> matrix, std::size_t rows, std::size_t cols,
                      int maxSweeps, float tolerance, std::span<float> resultA,
                      std::span<float> resultV) const override;
    void batchErf(std::span<const float> x, std::span<float> result) const override;
    void batchErfc(std::span<const float> x, std::span<float> result) const override;
    void batchGamma(std::span<const float> x, std::span<float> result) const override;
    void batchLogGamma(std::span<const float> x, std::span<float> result) const override;
    void batchLegendreP(unsigned n, unsigned m, std::span<const float> x,
                        std::span<float> result) const override;
    void batchPolynomialEval(std::span<const float> coefficients,
                             std::span<const float> x,
                             std::span<float> result) const override;
    void batchCubicSplineEval(std::span<const float> knotsX,
                              std::span<const float> knotsY,
                              std::span<const float> secondDerivatives,
                              std::span<const float> queryX,
                              std::span<float> result) const override;
    void batchUniformReal(std::uint64_t seed, std::uint64_t offset,
                          std::span<float> result) const override;
    void batchNormal(std::uint64_t seed, std::uint64_t offset,
                     std::span<float> result) const override;
    void sortAscending(std::span<float> values) const override;
    void multigridRestrict3D(std::span<const float> fine, std::size_t nx, std::size_t ny,
                             std::size_t nz, std::span<float> coarse) const override;
    void multigridProlongateAndAdd3D(std::span<const float> fine,
                                     std::span<const float> coarseCorrection,
                                     std::size_t nx, std::size_t ny, std::size_t nz,
                                     std::span<float> nextFine) const override;
    [[nodiscard]] std::size_t minIndex(std::span<const float> x) const override;
};

}  // namespace ysq
