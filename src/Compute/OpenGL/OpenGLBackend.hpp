#pragma once

#include <Compute/ComputeBackend.hpp>
#include <Compute/OpenGL/ComputeShader.hpp>

#include <Platform/Window.hpp>

namespace ysq {

/// Dispatches through a 4.3+ offscreen context, which is where compute shaders
/// start; see the comment on ContextSettings in Window.hpp. Compiled in only
/// under YSQ_BUILD_GRAPHICS.
///
/// macOS caps OpenGL at 4.1, so create() always fails there: there is no
/// route to a 4.3 context through OpenGL on that platform at all. See
/// src/Compute/README.md for the Vulkan/Metal path that exists instead.
class OpenGLBackend final : public ComputeBackend {
public:
    /// Nullptr if no 4.3 context is available, or either reference kernel
    /// fails to compile, which should not happen for shaders shipped with the
    /// engine but is reported rather than assumed.
    [[nodiscard]] static std::unique_ptr<ComputeBackend> create();

    [[nodiscard]] ComputeBackendKind kind() const noexcept override {
        return ComputeBackendKind::OpenGL;
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

private:
    OpenGLBackend(Window window, ComputeShader saxpy, ComputeShader sum,
                  ComputeShader linearCombine, ComputeShader gravitationalNBody,
                  ComputeShader electricFieldNBody, ComputeShader magneticFieldNBody,
                  ComputeShader sphDensityPressure, ComputeShader sphPressureAcceleration,
                  ComputeShader heatEquation3DStep, ComputeShader acoustic3DVelocity,
                  ComputeShader acoustic3DPressure, ComputeShader maxwell3DB,
                  ComputeShader maxwell3DE, ComputeShader eulerianFluid3DSweep,
                  ComputeShader fftBitReversalPermute, ComputeShader fftButterflyStage,
                  ComputeShader fftScale, ComputeShader matVec, ComputeShader matMul,
                  ComputeShader argMaxAbsColumn, ComputeShader swapRows,
                  ComputeShader luEliminationStep, ComputeShader choleskyDiagonal,
                  ComputeShader choleskyColumn, ComputeShader qrColumnNormSquared,
                  ComputeShader qrApplyLeft, ComputeShader qrAccumulateQ,
                  ComputeShader jacobiEigenColumnMix, ComputeShader jacobiEigenRowMix,
                  ComputeShader jacobiSvdRound, ComputeShader batchErf,
                  ComputeShader batchErfc, ComputeShader batchGamma,
                  ComputeShader batchLogGamma, ComputeShader batchLegendreP,
                  ComputeShader batchPolynomialEval, ComputeShader batchCubicSplineEval,
                  ComputeShader batchUniformReal, ComputeShader batchNormal,
                  ComputeShader bitonicCompareExchange, ComputeShader multigridRestrict3D,
                  ComputeShader multigridProlongateAndAdd3D,
                  ComputeShader minIndex) noexcept;

    /// Held for its lifetime: destroying it takes the context that every
    /// buffer and program here belongs to. Mutable because making it current
    /// is implementation state, not logical state a caller observes: saxpy()
    /// and sum() are const on the ComputeBackend interface, since dispatching
    /// a kernel does not change what this backend logically is.
    mutable Window m_window;
    ComputeShader m_saxpy;
    ComputeShader m_sum;
    ComputeShader m_linearCombine;
    ComputeShader m_gravitationalNBody;
    ComputeShader m_electricFieldNBody;
    ComputeShader m_magneticFieldNBody;
    ComputeShader m_sphDensityPressure;
    ComputeShader m_sphPressureAcceleration;
    ComputeShader m_heatEquation3DStep;
    ComputeShader m_acoustic3DVelocity;
    ComputeShader m_acoustic3DPressure;
    ComputeShader m_maxwell3DB;
    ComputeShader m_maxwell3DE;
    ComputeShader m_eulerianFluid3DSweep;
    ComputeShader m_fftBitReversalPermute;
    ComputeShader m_fftButterflyStage;
    ComputeShader m_fftScale;
    ComputeShader m_matVec;
    ComputeShader m_matMul;
    ComputeShader m_argMaxAbsColumn;
    ComputeShader m_swapRows;
    ComputeShader m_luEliminationStep;
    ComputeShader m_choleskyDiagonal;
    ComputeShader m_choleskyColumn;
    ComputeShader m_qrColumnNormSquared;
    ComputeShader m_qrApplyLeft;
    ComputeShader m_qrAccumulateQ;
    ComputeShader m_jacobiEigenColumnMix;
    ComputeShader m_jacobiEigenRowMix;
    ComputeShader m_jacobiSvdRound;
    ComputeShader m_batchErf;
    ComputeShader m_batchErfc;
    ComputeShader m_batchGamma;
    ComputeShader m_batchLogGamma;
    ComputeShader m_batchLegendreP;
    ComputeShader m_batchPolynomialEval;
    ComputeShader m_batchCubicSplineEval;
    ComputeShader m_batchUniformReal;
    ComputeShader m_batchNormal;
    ComputeShader m_bitonicCompareExchange;
    ComputeShader m_multigridRestrict3D;
    ComputeShader m_multigridProlongateAndAdd3D;
    ComputeShader m_minIndex;
};

}  // namespace ysq
