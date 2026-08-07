#include <Applications/Helper/KeplerPopulation.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::applications::generateKeplerPopulation;
using ysq::applications::KeplerParticle;

}  // namespace

TEST(ApplicationsHelperKeplerPopulation, ProducesExactlyTheRequestedCount) {
    const std::vector<KeplerParticle> particles = generateKeplerPopulation(
        /*parentIndex=*/0, /*parentGm=*/1.0, /*minSemiMajorAxis=*/1.0,
        /*maxSemiMajorAxis=*/2.0, /*maxEccentricity=*/0.1, /*maxInclination=*/0.1,
        /*count=*/250, /*seed=*/1, /*realRadiusMeters=*/100.0, ysq::Vec3f::splat(1.0f));
    EXPECT_EQ(particles.size(), 250u);
}

TEST(ApplicationsHelperKeplerPopulation, ZeroOrNegativeCountProducesNoParticles) {
    EXPECT_TRUE(
        generateKeplerPopulation(0, 1.0, 1.0, 2.0, 0.1, 0.1, 0, 1, 100.0, ysq::Vec3f::splat(1.0f))
            .empty());
    EXPECT_TRUE(generateKeplerPopulation(0, 1.0, 1.0, 2.0, 0.1, 0.1, -5, 1, 100.0,
                                         ysq::Vec3f::splat(1.0f))
                   .empty());
}

TEST(ApplicationsHelperKeplerPopulation, EveryParticleFallsWithinTheRequestedRanges) {
    constexpr double minA = 2.1;
    constexpr double maxA = 3.3;
    constexpr double maxE = 0.3;
    constexpr double maxI = ysq::radians(20.0);
    constexpr double realRadiusMeters = 2000.0;

    const std::vector<KeplerParticle> particles = generateKeplerPopulation(
        3, 42.0, minA, maxA, maxE, maxI, 2000, 7, realRadiusMeters, ysq::Vec3f::splat(0.7f));
    ASSERT_EQ(particles.size(), 2000u);

    for (const KeplerParticle& particle : particles) {
        EXPECT_EQ(particle.parentIndex, 3);
        EXPECT_DOUBLE_EQ(particle.parentGm, 42.0);
        EXPECT_DOUBLE_EQ(particle.realRadiusMeters, realRadiusMeters);
        EXPECT_GE(particle.elements.semiMajorAxis, minA);
        EXPECT_LE(particle.elements.semiMajorAxis, maxA);
        EXPECT_GE(particle.elements.eccentricity, 0.0);
        EXPECT_LE(particle.elements.eccentricity, maxE);
        EXPECT_GE(particle.elements.inclination, 0.0);
        EXPECT_LE(particle.elements.inclination, maxI);
        EXPECT_GE(particle.elements.longitudeOfAscendingNode, 0.0);
        EXPECT_LT(particle.elements.longitudeOfAscendingNode, ysq::kTau<double>);
        EXPECT_GE(particle.elements.meanAnomalyAtEpoch, 0.0);
        EXPECT_LT(particle.elements.meanAnomalyAtEpoch, ysq::kTau<double>);
        EXPECT_DOUBLE_EQ(particle.elements.precessionRatePerSecond, 0.0);
    }
}

TEST(ApplicationsHelperKeplerPopulation, TheSameSeedAlwaysProducesTheSameParticles) {
    const std::vector<KeplerParticle> first = generateKeplerPopulation(
        0, 1.0, 1.0, 2.0, 0.2, 0.2, 500, 99, 100.0, ysq::Vec3f::splat(1.0f));
    const std::vector<KeplerParticle> second = generateKeplerPopulation(
        0, 1.0, 1.0, 2.0, 0.2, 0.2, 500, 99, 100.0, ysq::Vec3f::splat(1.0f));

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_DOUBLE_EQ(first[i].elements.semiMajorAxis, second[i].elements.semiMajorAxis);
        EXPECT_DOUBLE_EQ(first[i].elements.eccentricity, second[i].elements.eccentricity);
        EXPECT_DOUBLE_EQ(first[i].elements.meanAnomalyAtEpoch,
                        second[i].elements.meanAnomalyAtEpoch);
    }
}

TEST(ApplicationsHelperKeplerPopulation, DifferentSeedsProduceDifferentParticles) {
    const std::vector<KeplerParticle> first = generateKeplerPopulation(
        0, 1.0, 1.0, 2.0, 0.2, 0.2, 500, 1, 100.0, ysq::Vec3f::splat(1.0f));
    const std::vector<KeplerParticle> second = generateKeplerPopulation(
        0, 1.0, 1.0, 2.0, 0.2, 0.2, 500, 2, 100.0, ysq::Vec3f::splat(1.0f));

    ASSERT_EQ(first.size(), second.size());
    int differences = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (first[i].elements.semiMajorAxis != second[i].elements.semiMajorAxis) {
            ++differences;
        }
    }
    EXPECT_GT(differences, 0) << "two different seeds produced an identical population";
}
