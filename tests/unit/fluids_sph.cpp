#include <Math/Vector3.hpp>
#include <Physics/Fluids/SPH.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cstddef>
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

}  // namespace
