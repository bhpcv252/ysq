#include <Physics/Fluids/Eulerian3D.hpp>

#include <Physics/Fluids/Eulerian.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr std::size_t kCellCount = 16;
constexpr double kSpacing = 0.01;
constexpr double kGamma = 1.4;

}  // namespace

TEST(FluidsEulerian3D, SetStateRoundTripsExactly) {
    ysq::EulerianFluid3D fluid(4, 4, 4, 0.1, kGamma);
    fluid.setState(1, 2, 3, 1.2, 0.5, -0.3, 0.2, 2.0);

    EXPECT_NEAR(fluid.density(1, 2, 3), 1.2, 1e-12);
    EXPECT_NEAR(fluid.velocityX(1, 2, 3), 0.5, 1e-12);
    EXPECT_NEAR(fluid.velocityY(1, 2, 3), -0.3, 1e-12);
    EXPECT_NEAR(fluid.velocityZ(1, 2, 3), 0.2, 1e-12);
    EXPECT_NEAR(fluid.pressure(1, 2, 3), 2.0, 1e-12);
}

TEST(FluidsEulerian3D, AUniformStateStaysUniform) {
    ysq::EulerianFluid3D fluid(kCellCount, kCellCount, kCellCount, kSpacing, kGamma);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                fluid.setState(i, j, k, 1.0, 0.3, -0.1, 0.2, 1.0);
            }
        }
    }

    const double dt = fluid.stableTimeStep(0.4);
    for (int s = 0; s < 20; ++s) {
        fluid.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                EXPECT_NEAR(fluid.density(i, j, k), 1.0, 1e-9);
                EXPECT_NEAR(fluid.velocityX(i, j, k), 0.3, 1e-8);
                EXPECT_NEAR(fluid.velocityY(i, j, k), -0.1, 1e-8);
                EXPECT_NEAR(fluid.velocityZ(i, j, k), 0.2, 1e-8);
                EXPECT_NEAR(fluid.pressure(i, j, k), 1.0, 1e-8);
            }
        }
    }
}

TEST(FluidsEulerian3D,
     MassAllThreeMomentumComponentsAndEnergyAreConservedForAGenuineBlast) {
    // A genuinely 3D case the dimensional-reduction test below cannot
    // exercise: an off-center high-pressure cube inside a moving ambient
    // medium (the nonzero background velocity makes each momentum
    // component's own conservation a real check, not a trivial "stays
    // near zero").
    ysq::EulerianFluid3D fluid(kCellCount, kCellCount, kCellCount, kSpacing, kGamma);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kCellCount; ++j) {
            for (std::size_t k = 0; k < kCellCount; ++k) {
                const bool inBlast =
                    i >= 6 && i < 10 && j >= 6 && j < 10 && k >= 6 && k < 10;
                if (inBlast) {
                    fluid.setState(i, j, k, 1.0, 0.1, -0.05, 0.2, 10.0);
                } else {
                    fluid.setState(i, j, k, 0.125, 0.1, -0.05, 0.2, 0.1);
                }
            }
        }
    }

    const double initialMass = fluid.totalMass();
    const double initialMomentumX = fluid.totalMomentumX();
    const double initialMomentumY = fluid.totalMomentumY();
    const double initialMomentumZ = fluid.totalMomentumZ();
    const double initialEnergy = fluid.totalEnergy();

    const double dt = fluid.stableTimeStep(0.4);
    for (int s = 0; s < 100; ++s) {
        fluid.step(dt);
    }

    // Periodic boundaries: whatever leaves one edge enters the other, so
    // these are conserved to floating-point rounding, not approximately.
    EXPECT_NEAR(fluid.totalMass(), initialMass, std::abs(initialMass) * 1e-9);
    EXPECT_NEAR(fluid.totalEnergy(), initialEnergy, std::abs(initialEnergy) * 1e-8);
    EXPECT_NEAR(fluid.totalMomentumX(), initialMomentumX,
                std::abs(initialMomentumX) * 1e-8);
    EXPECT_NEAR(fluid.totalMomentumY(), initialMomentumY,
                std::abs(initialMomentumY) * 1e-8);
    EXPECT_NEAR(fluid.totalMomentumZ(), initialMomentumZ,
                std::abs(initialMomentumZ) * 1e-8);
}

TEST(FluidsEulerian3D, ReducesExactlyToTheOneDimensionalSolverWhenFlatInYAndZ) {
    constexpr std::size_t kFlatCount = 4;
    ysq::EulerianFluid3D fluid3D(kCellCount, kFlatCount, kFlatCount, kSpacing, kGamma);
    ysq::EulerianFluid1D fluid1D(kCellCount, kSpacing, kGamma);

    for (std::size_t i = 0; i < kCellCount; ++i) {
        double density = 0.0;
        double pressure = 0.0;
        if (i < kCellCount / 2) {
            density = 1.0;
            pressure = 1.0;
        } else {
            density = 0.125;
            pressure = 0.1;
        }
        fluid1D.setState(i, density, 0.0, pressure);
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                fluid3D.setState(i, j, k, density, 0.0, 0.0, 0.0, pressure);
            }
        }
    }

    const double dt = fluid1D.stableTimeStep(0.4);
    for (int s = 0; s < 100; ++s) {
        fluid1D.step(dt);
        fluid3D.step(dt);
    }

    for (std::size_t i = 0; i < kCellCount; ++i) {
        for (std::size_t j = 0; j < kFlatCount; ++j) {
            for (std::size_t k = 0; k < kFlatCount; ++k) {
                EXPECT_NEAR(fluid3D.density(i, j, k), fluid1D.density(i), 1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(fluid3D.velocityX(i, j, k), fluid1D.velocity(i), 1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(fluid3D.pressure(i, j, k), fluid1D.pressure(i), 1e-9)
                    << "cell " << i << "," << j << "," << k;
                EXPECT_NEAR(fluid3D.velocityY(i, j, k), 0.0, 1e-9);
                EXPECT_NEAR(fluid3D.velocityZ(i, j, k), 0.0, 1e-9);
            }
        }
    }
}
