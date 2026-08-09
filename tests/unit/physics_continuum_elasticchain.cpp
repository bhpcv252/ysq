#include <Physics/Continuum/ElasticChain1D.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace {

using ysq::ElasticChain1D;
using ysq::Force;
using ysq::Length;
using ysq::SpringConstant;

}  // namespace

TEST(PhysicsContinuumElasticChain, NodeCountIsOneMoreThanSegmentCount) {
    const ElasticChain1D chain(std::vector<SpringConstant>(4, SpringConstant{1.0}));
    EXPECT_EQ(chain.nodeCount(), 5U);
}

TEST(PhysicsContinuumElasticChain, SingleSegmentReducesToPlainHookesLaw) {
    const SpringConstant k{500.0};
    const ElasticChain1D chain(std::vector<SpringConstant>{k});

    const std::vector<Force> load{Force{0.0}, Force{20.0}};
    const std::optional<std::vector<Length>> displacements =
        chain.solveDisplacements(load);

    ASSERT_TRUE(displacements.has_value());
    EXPECT_NEAR((*displacements)[0].value(), 0.0, 1e-12);
    EXPECT_NEAR((*displacements)[1].value(), 20.0 / 500.0, 1e-9);
}

TEST(PhysicsContinuumElasticChain, ZeroLoadProducesZeroDisplacementEverywhere) {
    const ElasticChain1D chain(std::vector<SpringConstant>(3, SpringConstant{100.0}));
    const std::vector<Force> load(4, Force{0.0});

    const std::optional<std::vector<Length>> displacements =
        chain.solveDisplacements(load);
    ASSERT_TRUE(displacements.has_value());
    for (const Length& d : *displacements) {
        EXPECT_NEAR(d.value(), 0.0, 1e-12);
    }
}

TEST(PhysicsContinuumElasticChain, AZeroStiffnessSegmentMakesTheSystemSingular) {
    const ElasticChain1D chain(
        std::vector<SpringConstant>{SpringConstant{100.0}, SpringConstant{0.0}});
    const std::vector<Force> load{Force{0.0}, Force{0.0}, Force{10.0}};

    EXPECT_FALSE(chain.solveDisplacements(load).has_value());
}

TEST(PhysicsContinuumElasticChain,
     UniformChainEndDisplacementMatchesTheContinuumFormulaRegardlessOfSegmentCount) {
    // A rod of total length L, cross-section A, Young's modulus E,
    // discretized into M identical segments (each of stiffness E A M / L,
    // i.e. length L/M), loaded only at the free end with force F: springs
    // in series add reciprocal stiffness, so the total end displacement
    // must equal the continuum formula F L / (E A) exactly, regardless of
    // how finely the rod is discretized.
    constexpr double youngsModulus = 200e9;
    constexpr double crossSection = 0.0005;
    constexpr double totalLength = 3.0;
    constexpr double endForce = 5000.0;
    const double expectedEndDisplacement =
        endForce * totalLength / (youngsModulus * crossSection);

    for (std::size_t segmentCount : {1U, 2U, 5U, 20U}) {
        const double segmentStiffness = youngsModulus * crossSection *
                                        static_cast<double>(segmentCount) / totalLength;
        const std::vector<SpringConstant> stiffness(segmentCount,
                                                    SpringConstant{segmentStiffness});
        const ElasticChain1D chain(stiffness);

        std::vector<Force> load(segmentCount + 1, Force{0.0});
        load[segmentCount] = Force{endForce};

        const std::optional<std::vector<Length>> displacements =
            chain.solveDisplacements(load);
        ASSERT_TRUE(displacements.has_value());

        EXPECT_NEAR(displacements->back().value(), expectedEndDisplacement,
                    expectedEndDisplacement * 1e-9)
            << "segmentCount = " << segmentCount;
    }
}

TEST(PhysicsContinuumElasticChain, IntermediateNodesShareTheLoadInAUniformChain) {
    // With the load only at the free end, every segment carries the same
    // internal force, so each intermediate node's displacement should be
    // proportional to its own distance from the fixed end.
    constexpr std::size_t segmentCount = 4;
    const SpringConstant k{1000.0};
    const std::vector<SpringConstant> stiffness(segmentCount, k);
    const ElasticChain1D chain(stiffness);

    std::vector<Force> load(segmentCount + 1, Force{0.0});
    load[segmentCount] = Force{40.0};

    const std::optional<std::vector<Length>> displacements =
        chain.solveDisplacements(load);
    ASSERT_TRUE(displacements.has_value());

    const double perSegmentStretch = 40.0 / k.value();
    for (std::size_t i = 0; i <= segmentCount; ++i) {
        EXPECT_NEAR((*displacements)[i].value(),
                    static_cast<double>(i) * perSegmentStretch, 1e-9);
    }
}
