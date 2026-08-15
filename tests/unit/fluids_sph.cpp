#include <Compute/CPU/CpuBackend.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Fluids/SPH.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace {

using ysq::SPHParticle;
using ysq::Vec3;

TEST(FluidsSPH, KernelHasCompactSupportAtTwoSmoothingLengths) {
    constexpr double h = 1.0;
    EXPECT_GT(ysq::cubicSplineKernel(0.0, h), 0.0);
    EXPECT_GT(ysq::cubicSplineKernel(1.5, h), 0.0);
    EXPECT_NEAR(ysq::cubicSplineKernel(2.0, h), 0.0, 1e-12);
    EXPECT_NEAR(ysq::cubicSplineKernel(3.0, h), 0.0, 1e-12);
}

TEST(FluidsSPH, KernelIsContinuousAtTheSplineJoin) {
    constexpr double h = 1.0;
    // The two branches, evaluated from either side of q = 1, must agree:
    // the classic bug in a piecewise kernel is a mismatched constant.
    constexpr double epsilon = 1e-9;
    const double justBelow = ysq::cubicSplineKernel(h * (1.0 - epsilon), h);
    const double justAbove = ysq::cubicSplineKernel(h * (1.0 + epsilon), h);
    EXPECT_NEAR(justBelow, justAbove, 1e-6);
}

TEST(FluidsSPH, KernelIntegratesToOneOverAllSpace) {
    // integral of W(r,h) * 4 pi r^2 dr from 0 to 2h must be 1: this is what
    // "normalized smoothing kernel" means. Checked by a fine composite
    // trapezoid rule over the kernel's compact support.
    constexpr double h = 1.0;
    constexpr int steps = 20000;
    const double dr = 2.0 * h / steps;

    double integral = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double r = static_cast<double>(i) * dr;
        const double weight = (i == 0 || i == steps) ? 0.5 : 1.0;
        integral +=
            weight * ysq::cubicSplineKernel(r, h) * 4.0 * ysq::kPi<double> * r * r;
    }
    integral *= dr;

    EXPECT_NEAR(integral, 1.0, 1e-4);
}

TEST(FluidsSPH, KernelGradientPointsTowardTheOtherParticle) {
    // grad_i W(r_i - r_j) has to point toward decreasing separation, i.e.
    // toward particle j, since W is largest at zero separation.
    constexpr double h = 1.0;
    const Vec3 separation{0.5, 0.0, 0.0};  // i is at +0.5x relative to j
    const Vec3 gradient = ysq::cubicSplineKernelGradient(separation, h);
    EXPECT_LT(gradient.x, 0.0);
}

std::vector<SPHParticle> cubicLattice(int perSide, double spacing, double mass) {
    std::vector<SPHParticle> particles;
    particles.reserve(static_cast<std::size_t>(perSide * perSide * perSide));
    for (int i = 0; i < perSide; ++i) {
        for (int j = 0; j < perSide; ++j) {
            for (int k = 0; k < perSide; ++k) {
                SPHParticle particle{};
                particle.mass = mass;
                particle.position = Vec3{static_cast<double>(i) * spacing,
                                         static_cast<double>(j) * spacing,
                                         static_cast<double>(k) * spacing};
                particles.push_back(particle);
            }
        }
    }
    return particles;
}

TEST(FluidsSPH, DensityOfAUniformLatticeMatchesTheBulkDensity) {
    constexpr double spacing = 0.2;
    constexpr double mass = 1.0;
    constexpr double h = 2.5 * spacing;
    std::vector<SPHParticle> particles = cubicLattice(9, spacing, mass);

    ysq::computeDensityAndPressure(particles, h, 1.0, 1.0);

    // The centre particle of a large enough lattice sees a full
    // neighbourhood in every direction, unlike one near an edge.
    const std::size_t centreIndex = particles.size() / 2;
    const double expectedDensity = mass / (spacing * spacing * spacing);

    // SPH's kernel-sum density estimate on a discrete lattice is not exact
    // even in the bulk; a few percent is the normal size of that error for
    // this kernel and this h / spacing ratio, not a bug.
    EXPECT_NEAR(particles[centreIndex].density, expectedDensity, expectedDensity * 0.1);
}

TEST(FluidsSPH, PressureAccelerationsConserveMomentumExactly) {
    // Newton's third law, structurally: grad_i W_ij = -grad_j W_ji, so every
    // pairwise contribution to the total momentum cancels regardless of the
    // configuration or the equation of state.
    std::vector<SPHParticle> particles = cubicLattice(4, 0.15, 1.0);
    // Perturb off the lattice so the configuration is not artificially
    // symmetric, which could hide a sign error.
    particles[0].position += Vec3{0.02, -0.01, 0.03};
    particles[5].position += Vec3{-0.02, 0.01, -0.02};
    particles[9].mass = 1.5;

    constexpr double h = 0.4;
    ysq::computeDensityAndPressure(particles, h, 2.0, 1.4);
    const std::vector<Vec3> accelerations = ysq::pressureAccelerations(particles, h);

    Vec3 totalMomentumRate{};
    for (std::size_t i = 0; i < particles.size(); ++i) {
        totalMomentumRate += accelerations[i] * particles[i].mass;
    }
    EXPECT_VEC_NEAR(totalMomentumRate, Vec3{}, 1e-9);
}

TEST(FluidsSPH, TwoCompressedParticlesRepelEachOther) {
    constexpr double h = 1.0;
    SPHParticle a{};
    a.mass = 1.0;
    a.position = Vec3{-0.2, 0.0, 0.0};
    SPHParticle b{};
    b.mass = 1.0;
    b.position = Vec3{0.2, 0.0, 0.0};

    std::vector<SPHParticle> particles{a, b};
    ysq::computeDensityAndPressure(particles, h, 1.0, 1.0);
    ASSERT_GT(particles[0].pressure, 0.0);

    const std::vector<Vec3> accelerations = ysq::pressureAccelerations(particles, h);
    // a is pushed further in -x, b further in +x: they separate.
    EXPECT_LT(accelerations[0].x, 0.0);
    EXPECT_GT(accelerations[1].x, 0.0);
}

TEST(FluidsSPH, DensityAndAccelerationsAtLargeNAgreeWithTheComputeCpuReference) {
    // Above SPH.cpp's own GPU dispatch threshold, so
    // computeDensityAndPressure/pressureAccelerations exercise whichever
    // compute backend is actually available. ysq::CpuBackend is called
    // directly here as an independent reference (its own agreement with
    // every GPU backend is already covered by
    // tests/integration/compute_backends_agree.cpp; this test is only
    // checking that Physics/Fluids/SPH.cpp's dispatch wiring feeds it the
    // right data and reads the result back correctly).
    constexpr std::size_t kCount = 1200;
    constexpr double h = 1.0;
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> posDist(-3.0, 3.0);
    std::uniform_real_distribution<double> massDist(0.5, 2.0);

    std::vector<SPHParticle> particles(kCount);
    std::vector<float> posX(kCount);
    std::vector<float> posY(kCount);
    std::vector<float> posZ(kCount);
    std::vector<float> mass(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        const Vec3 p{posDist(rng), posDist(rng), posDist(rng)};
        const double m = massDist(rng);
        particles[i].position = p;
        particles[i].mass = m;
        posX[i] = static_cast<float>(p.x);
        posY[i] = static_cast<float>(p.y);
        posZ[i] = static_cast<float>(p.z);
        mass[i] = static_cast<float>(m);
    }

    ysq::computeDensityAndPressure(particles, h, 1.0, 2.0);

    const ysq::CpuBackend cpu;
    std::vector<float> densityReference(kCount);
    std::vector<float> pressureReference(kCount);
    cpu.sphDensityPressure(posX, posY, posZ, mass, static_cast<float>(h), 1.0f, 2.0f,
                           densityReference, pressureReference);

    for (const std::size_t i : {std::size_t{0}, kCount / 2, kCount - 1}) {
        EXPECT_NEAR(particles[i].density, densityReference[i], densityReference[i] * 1e-2)
            << "particle " << i;
    }

    const std::vector<Vec3> accelerations = ysq::pressureAccelerations(particles, h);
    std::vector<float> density(kCount);
    std::vector<float> pressure(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        density[i] = static_cast<float>(particles[i].density);
        pressure[i] = static_cast<float>(particles[i].pressure);
    }
    std::vector<float> accXReference(kCount);
    std::vector<float> accYReference(kCount);
    std::vector<float> accZReference(kCount);
    cpu.sphPressureAcceleration(posX, posY, posZ, mass, density, pressure,
                                static_cast<float>(h), accXReference, accYReference,
                                accZReference);

    for (const std::size_t i : {std::size_t{0}, kCount / 2, kCount - 1}) {
        const double tolerance = std::max(std::abs(accXReference[i]), 1.0f) * 2e-2;
        EXPECT_NEAR(accelerations[i].x, accXReference[i], tolerance) << "particle " << i;
        EXPECT_NEAR(accelerations[i].y, accYReference[i], tolerance) << "particle " << i;
        EXPECT_NEAR(accelerations[i].z, accZReference[i], tolerance) << "particle " << i;
    }
}

}  // namespace
