#include <Physics/Thermodynamics/HeatEquation3D.hpp>

#include <Compute/CPU/CpuBackend.hpp>
#include <Physics/Thermodynamics/HeatEquation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

constexpr std::size_t kCellCount = 24;
constexpr double kSpacing = 0.05;
constexpr double kDiffusivity = 0.05;

TEST(ThermodynamicsHeatEquation3D, SetTemperatureRoundTripsExactly) {
    ysq::HeatEquation3D heat(4, 4, 4, 0.1, kDiffusivity);
    heat.setTemperature(1, 2, 3, 373.15);
    EXPECT_NEAR(heat.temperature(1, 2, 3), 373.15, 1e-12);
}

TEST(ThermodynamicsHeatEquation3D, AUniformTemperatureStaysUniform) {
    ysq::HeatEquation3D heat(kCellCount, kCellCount, kCellCount, kSpacing, kDiffusivity);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                heat.setTemperature(i, j, k, 300.0);
            }
        }
    }

    const double dt = heat.stableTimeStep(0.9);
    for (int s = 0; s < 20; ++s) {
        heat.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                EXPECT_NEAR(heat.temperature(i, j, k), 300.0, 1e-9);
            }
        }
    }
}

namespace {

double gaussianPulse(std::size_t cellCount, double spacing, double sigma,
                     std::size_t index) {
    const double center = static_cast<double>(cellCount / 2) * spacing;
    const double x = static_cast<double>(index) * spacing;
    const double normalized = (x - center) / sigma;
    return std::exp(-0.5 * normalized * normalized);
}

}  // namespace

TEST(ThermodynamicsHeatEquation3D, TotalHeatIsConservedExactly) {
    ysq::HeatEquation3D heat(kCellCount, kCellCount, kCellCount, kSpacing, kDiffusivity);
    constexpr double sigma = 0.1;
    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                heat.setTemperature(i, j, k,
                                    100.0 *
                                        gaussianPulse(kCellCount, kSpacing, sigma, i) *
                                        gaussianPulse(kCellCount, kSpacing, sigma, j) *
                                        gaussianPulse(kCellCount, kSpacing, sigma, k));
            }
        }
    }

    const double initialHeat = heat.totalHeat();
    const double dt = heat.stableTimeStep(0.9);
    for (int s = 0; s < 50; ++s) {
        heat.step(dt);
    }

    EXPECT_NEAR(heat.totalHeat(), initialHeat, initialHeat * 1e-9);
}

TEST(ThermodynamicsHeatEquation3D,
     ASeparableGaussianPulseSpreadsAtTheAnalyticRateOnEveryAxis) {
    // The 3D heat kernel is the product of three 1D heat kernels along each
    // axis, since the Laplacian is a sum of independent per-axis second
    // derivatives: a Gaussian that starts as a product of three 1D
    // Gaussians stays one, with each axis's own variance growing exactly at
    // the 1D rate, independent of the other two axes.
    // Its own, finer-resolved parameters rather than the file-level
    // constants: this check needs enough grid points per sigma and enough
    // domain margin past the *final*, spread-out sigma to keep periodic
    // wraparound from biasing the measured variance, which the coarser
    // grid the cheaper conservation/uniformity checks above use does not
    // give enough of.
    constexpr std::size_t cellCount = 48;
    constexpr double spacing = 0.02;
    constexpr double diffusivity = 0.05;
    ysq::HeatEquation3D heat(cellCount, cellCount, cellCount, spacing, diffusivity);
    constexpr double initialSigma = 0.08;
    for (std::size_t i = 0; i < cellCount; ++i) {
        for (std::size_t j = 0; j < cellCount; ++j) {
            for (std::size_t k = 0; k < cellCount; ++k) {
                heat.setTemperature(
                    i, j, k,
                    gaussianPulse(cellCount, spacing, initialSigma, i) *
                        gaussianPulse(cellCount, spacing, initialSigma, j) *
                        gaussianPulse(cellCount, spacing, initialSigma, k));
            }
        }
    }

    const double dt = heat.stableTimeStep(0.9);
    constexpr int steps = 80;
    for (int s = 0; s < steps; ++s) {
        heat.step(dt);
    }
    const double elapsed = steps * dt;
    const double expectedVariance =
        initialSigma * initialSigma + 2.0 * diffusivity * elapsed;
    const double center = static_cast<double>(cellCount / 2) * spacing;

    // Marginalize onto the x axis (sum over y, z) and measure its variance;
    // by symmetry of the setup, y and z behave identically.
    double totalWeight = 0.0;
    double weightedPosition = 0.0;
    for (std::size_t i = 0; i < cellCount; ++i) {
        double marginal = 0.0;
        for (std::size_t j = 0; j < cellCount; ++j) {
            for (std::size_t k = 0; k < cellCount; ++k) {
                marginal += heat.temperature(i, j, k);
            }
        }
        const double x = static_cast<double>(i) * spacing;
        totalWeight += marginal;
        weightedPosition += marginal * x;
    }
    const double centroid = weightedPosition / totalWeight;

    double weightedVarianceSum = 0.0;
    for (std::size_t i = 0; i < cellCount; ++i) {
        double marginal = 0.0;
        for (std::size_t j = 0; j < cellCount; ++j) {
            for (std::size_t k = 0; k < cellCount; ++k) {
                marginal += heat.temperature(i, j, k);
            }
        }
        const double x = static_cast<double>(i) * spacing;
        weightedVarianceSum += marginal * (x - centroid) * (x - centroid);
    }
    const double measuredVariance = weightedVarianceSum / totalWeight;

    EXPECT_NEAR(centroid, center, spacing);
    EXPECT_NEAR(measuredVariance, expectedVariance, expectedVariance * 0.03);
}

TEST(ThermodynamicsHeatEquation3D,
     ReducesExactlyToTheOneDimensionalSolverWhenFlatInYAndZ) {
    constexpr std::size_t kFlatCount = 4;
    ysq::HeatEquation3D heat3D(kCellCount, kFlatCount, kFlatCount, kSpacing,
                               kDiffusivity);
    ysq::HeatEquation1D heat1D(kCellCount, kSpacing, kDiffusivity);

    constexpr double sigma = 0.12;
    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double value = 50.0 * gaussianPulse(kCellCount, kSpacing, sigma, i);
        heat1D.setTemperature(i, value);
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                heat3D.setTemperature(i, j, k, value);
            }
        }
    }

    const double dt = heat1D.stableTimeStep(0.9);
    for (int s = 0; s < 40; ++s) {
        heat1D.step(dt);
        heat3D.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                EXPECT_NEAR(heat3D.temperature(i, j, k), heat1D.temperature(i), 1e-9)
                    << "cell " << i << "," << j << "," << k;
            }
        }
    }
}

TEST(ThermodynamicsHeatEquation3D, StepAtLargeNAgreesWithTheComputeCpuReference) {
    // Above HeatEquation3D.cpp's own GPU dispatch threshold, so step()
    // exercises whichever compute backend is actually available.
    // ysq::CpuBackend is called directly as an independent reference (its
    // own agreement with every GPU backend is already covered by
    // tests/integration/compute_backends_agree.cpp; this test only checks
    // that HeatEquation3D.cpp's dispatch wiring feeds it the right data
    // and reads the result back correctly), with a float32-appropriate
    // tolerance rather than the double-precision exactness the
    // below-threshold tests above expect.
    // 162^3 = 4251528, above kGpuDispatchThreshold (4194304; measured by
    // benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 162;
    ysq::HeatEquation3D heat(n, n, n, kSpacing, kDiffusivity);

    std::vector<float> temperature(n * n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const double value = 50.0 * gaussianPulse(n, kSpacing, 0.1, i) *
                                     gaussianPulse(n, kSpacing, 0.1, j) *
                                     gaussianPulse(n, kSpacing, 0.1, k);
                heat.setTemperature(i, j, k, value);
                temperature[(i * n + j) * n + k] = static_cast<float>(value);
            }
        }
    }

    const double dt = heat.stableTimeStep(0.9);
    heat.step(dt);

    const ysq::CpuBackend cpu;
    const float factor = static_cast<float>(kDiffusivity * dt / (kSpacing * kSpacing));
    std::vector<float> nextReference(n * n * n);
    cpu.heatEquation3DStep(temperature, n, n, n, factor, nextReference);

    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        for (const std::size_t j : {std::size_t{0}, n / 2, n - 1}) {
            for (const std::size_t k : {std::size_t{0}, n / 2, n - 1}) {
                EXPECT_NEAR(heat.temperature(i, j, k), nextReference[(i * n + j) * n + k],
                            1e-4)
                    << "cell " << i << "," << j << "," << k;
            }
        }
    }
}

}  // namespace
