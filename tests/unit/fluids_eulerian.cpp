#include <Physics/Fluids/Eulerian.hpp>

#include <gtest/gtest.h>

#include <cstddef>

namespace {

constexpr std::size_t kCellCount = 100;
constexpr double kSpacing = 0.01;
constexpr double kGamma = 1.4;

ysq::EulerianFluid1D makeSodShockTube() {
    ysq::EulerianFluid1D fluid(kCellCount, kSpacing, kGamma);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        if (i < kCellCount / 2) {
            fluid.setState(i, 1.0, 0.0, 1.0);
        } else {
            fluid.setState(i, 0.125, 0.0, 0.1);
        }
    }
    return fluid;
}

TEST(FluidsEulerian, SetStateRoundTripsExactly) {
    ysq::EulerianFluid1D fluid(4, 0.1, kGamma);
    fluid.setState(0, 1.2, 0.5, 2.0);

    EXPECT_NEAR(fluid.density(0), 1.2, 1e-12);
    EXPECT_NEAR(fluid.velocity(0), 0.5, 1e-12);
    EXPECT_NEAR(fluid.pressure(0), 2.0, 1e-12);
}

TEST(FluidsEulerian, AUniformStateStaysUniform) {
    ysq::EulerianFluid1D fluid(kCellCount, kSpacing, kGamma);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        fluid.setState(i, 1.0, 0.3, 1.0);
    }

    const double dt = fluid.stableTimeStep(0.5);
    for (int s = 0; s < 20; ++s) {
        fluid.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        EXPECT_NEAR(fluid.density(i), 1.0, 1e-10);
        EXPECT_NEAR(fluid.velocity(i), 0.3, 1e-9);
        EXPECT_NEAR(fluid.pressure(i), 1.0, 1e-9);
    }
}

TEST(FluidsEulerian, MassMomentumAndEnergyAreConservedExactly) {
    ysq::EulerianFluid1D fluid = makeSodShockTube();
    const double initialMass = fluid.totalMass();
    const double initialMomentum = fluid.totalMomentum();
    const double initialEnergy = fluid.totalEnergy();

    const double dt = fluid.stableTimeStep(0.4);
    for (int s = 0; s < 200; ++s) {
        fluid.step(dt);
    }

    // Periodic boundaries: whatever leaves one edge enters the other, so
    // these are conserved to floating-point rounding, not approximately.
    EXPECT_NEAR(fluid.totalMass(), initialMass, std::abs(initialMass) * 1e-10);
    EXPECT_NEAR(fluid.totalEnergy(), initialEnergy, std::abs(initialEnergy) * 1e-9);
    EXPECT_NEAR(fluid.totalMomentum(), initialMomentum, 1e-9);
}

}  // namespace
