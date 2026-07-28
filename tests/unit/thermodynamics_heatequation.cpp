#include <Physics/Thermodynamics/HeatEquation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr std::size_t kCellCount = 400;
constexpr double kSpacing = 0.01;  // domain length 4.0
constexpr double kDiffusivity = 0.05;

TEST(ThermodynamicsHeatEquation, SetTemperatureRoundTripsExactly) {
    ysq::HeatEquation1D heat(4, 0.1, kDiffusivity);
    heat.setTemperature(2, 373.15);
    EXPECT_NEAR(heat.temperature(2), 373.15, 1e-12);
}

TEST(ThermodynamicsHeatEquation, AUniformTemperatureStaysUniform) {
    ysq::HeatEquation1D heat(kCellCount, kSpacing, kDiffusivity);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        heat.setTemperature(i, 300.0);
    }

    const double dt = heat.stableTimeStep(0.9);
    for (int s = 0; s < 50; ++s) {
        heat.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        EXPECT_NEAR(heat.temperature(i), 300.0, 1e-9);
    }
}

TEST(ThermodynamicsHeatEquation, TotalHeatIsConservedExactly) {
    ysq::HeatEquation1D heat(kCellCount, kSpacing, kDiffusivity);
    const double center = static_cast<double>(kCellCount / 2) * kSpacing;
    constexpr double width = 0.1;
    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        const double normalized = (x - center) / width;
        heat.setTemperature(i, 100.0 * std::exp(-0.5 * normalized * normalized));
    }

    const double initialHeat = heat.totalHeat();
    const double dt = heat.stableTimeStep(0.9);
    for (int s = 0; s < 200; ++s) {
        heat.step(dt);
    }

    // Periodic boundaries: nothing leaves the domain, so this holds to
    // floating-point rounding regardless of how far the pulse has spread.
    EXPECT_NEAR(heat.totalHeat(), initialHeat, initialHeat * 1e-9);
}

TEST(ThermodynamicsHeatEquation, AGaussianPulseSpreadsAtTheAnalyticRate) {
    // The fundamental solution of the 1D heat equation: a Gaussian stays a
    // Gaussian, with variance growing as sigma^2(t) = sigma^2(0) + 2 alpha t.
    ysq::HeatEquation1D heat(kCellCount, kSpacing, kDiffusivity);
    const double center = static_cast<double>(kCellCount / 2) * kSpacing;
    constexpr double initialSigma = 0.08;

    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        const double normalized = (x - center) / initialSigma;
        heat.setTemperature(i, std::exp(-0.5 * normalized * normalized));
    }

    const double dt = heat.stableTimeStep(0.9);
    constexpr int steps = 300;
    for (int s = 0; s < steps; ++s) {
        heat.step(dt);
    }
    const double elapsed = steps * dt;

    // The second moment of the (temperature-weighted) distribution about
    // its own centroid is the variance, measured directly from the field
    // rather than assumed to still be centred at `center`.
    double totalWeight = 0.0;
    double weightedPosition = 0.0;
    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        const double weight = heat.temperature(i);
        totalWeight += weight;
        weightedPosition += weight * x;
    }
    const double centroid = weightedPosition / totalWeight;

    double weightedVarianceSum = 0.0;
    for (std::size_t i = 0; i < kCellCount; ++i) {
        const double x = static_cast<double>(i) * kSpacing;
        const double weight = heat.temperature(i);
        weightedVarianceSum += weight * (x - centroid) * (x - centroid);
    }
    const double measuredVariance = weightedVarianceSum / totalWeight;

    const double expectedVariance =
        initialSigma * initialSigma + 2.0 * kDiffusivity * elapsed;

    EXPECT_NEAR(centroid, center, kSpacing);
    EXPECT_NEAR(measuredVariance, expectedVariance, expectedVariance * 0.02);
}

}  // namespace
