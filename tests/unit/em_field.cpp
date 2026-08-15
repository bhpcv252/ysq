#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Electromagnetism/Field.hpp>
#include <Units/Constants.hpp>
#include <Units/Electromagnetism.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace {

using ysq::Body;
using ysq::Vec3;

Body makePointCharge(double charge, Vec3 position, Vec3 velocity = Vec3{}) {
    Body body{};
    body.mass = ysq::Mass{1.0};
    body.charge = ysq::ElectricCharge{charge};
    body.position = ysq::Length3{position};
    body.momentum = ysq::Momentum3{velocity};
    return body;
}

TEST(ElectromagnetismField, CoulombFieldMagnitudeMatchesTheInverseSquareLaw) {
    const Body source = makePointCharge(2.0e-6, Vec3{0.0, 0.0, 0.0});
    const ysq::Length3 at{Vec3{3.0, 0.0, 0.0}};

    const ysq::ElectricField3 field = ysq::electricField(at, std::array<Body, 1>{source});

    const double expectedMagnitude =
        ysq::constants::coulombConstant.value() * 2.0e-6 / (3.0 * 3.0);
    EXPECT_NEAR(length(field.value()), expectedMagnitude, expectedMagnitude * 1e-12);
    // Points away from a positive source.
    EXPECT_GT(field.value().x, 0.0);
    EXPECT_NEAR(field.value().y, 0.0, 1e-20);
}

TEST(ElectromagnetismField, TwoEqualChargesCancelAtTheMidpoint) {
    const Body a = makePointCharge(1.0e-6, Vec3{-1.0, 0.0, 0.0});
    const Body b = makePointCharge(1.0e-6, Vec3{1.0, 0.0, 0.0});
    const std::array<Body, 2> sources{a, b};

    const ysq::ElectricField3 field =
        ysq::electricField(ysq::Length3{Vec3{0.0, 0.0, 0.0}}, sources);
    EXPECT_VEC_NEAR(field.value(), Vec3{}, 1e-9);
}

TEST(ElectromagnetismField, SkipsASourceExactlyAtTheQueryPoint) {
    const Body a = makePointCharge(1.0e-6, Vec3{0.0, 0.0, 0.0});
    const Body b = makePointCharge(1.0e-6, Vec3{2.0, 0.0, 0.0});
    const std::array<Body, 2> sources{a, b};

    // Querying exactly at a's position: a's own (undefined) contribution is
    // skipped, and only b's well-defined contribution remains.
    const ysq::ElectricField3 field = ysq::electricField(a.position, sources);
    const ysq::ElectricField3 fromBOnly =
        ysq::electricField(a.position, std::array<Body, 1>{b});
    EXPECT_VEC_NEAR(field.value(), fromBOnly.value(), 1e-30);
}

TEST(ElectromagnetismField, AStationaryChargeProducesNoMagneticField) {
    const Body source = makePointCharge(1.0e-6, Vec3{0.0, 0.0, 0.0});
    const ysq::MagneticFluxDensity3 field = ysq::magneticField(
        ysq::Length3{Vec3{1.0, 0.0, 0.0}}, std::array<Body, 1>{source});
    EXPECT_VEC_NEAR(field.value(), Vec3{}, 1e-30);
}

TEST(ElectromagnetismField, MovingChargeMagneticFieldFollowsTheRightHandRule) {
    // A positive charge moving in +x, field point along +y from it: v x
    // r-hat = x-hat x y-hat = z-hat, so B should point in +z.
    const Body source = makePointCharge(1.0e-6, Vec3{0.0, 0.0, 0.0}, Vec3{5.0, 0.0, 0.0});
    const ysq::MagneticFluxDensity3 field = ysq::magneticField(
        ysq::Length3{Vec3{0.0, 2.0, 0.0}}, std::array<Body, 1>{source});

    EXPECT_GT(field.value().z, 0.0);
    EXPECT_NEAR(field.value().x, 0.0, 1e-25);
    EXPECT_NEAR(field.value().y, 0.0, 1e-25);

    const double expectedMagnitude = ysq::constants::vacuumPermeability.value() /
                                     (4.0 * ysq::kPi<double>)*1.0e-6 * 5.0 / (2.0 * 2.0);
    EXPECT_NEAR(field.value().z, expectedMagnitude, expectedMagnitude * 1e-9);
}

TEST(ElectromagnetismField, ElectricFieldsAtLargeNAgreesWithTheSinglePointApi) {
    // Above Field.cpp's own GPU dispatch threshold, so this exercises
    // whichever compute backend is actually available. electricField()
    // (singular) is safe to call with the full body list including the
    // query body itself: it skips a source at zero separation without
    // producing a NaN (unlike Gravity's softened term at zero softening),
    // so it is a genuine independent reference here, not a restatement.
    constexpr std::size_t kBodyCount = 1200;
    std::vector<Body> bodies;
    bodies.reserve(kBodyCount);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> positionDist(-10.0, 10.0);
    std::uniform_real_distribution<double> chargeDist(-1.0e-6, 1.0e-6);
    for (std::size_t i = 0; i < kBodyCount; ++i) {
        bodies.push_back(
            makePointCharge(chargeDist(rng), Vec3{positionDist(rng), positionDist(rng),
                                                  positionDist(rng)}));
    }

    const std::vector<ysq::ElectricField3> fields = ysq::electricFields(bodies);
    ASSERT_EQ(fields.size(), bodies.size());

    for (const std::size_t i : {std::size_t{0}, kBodyCount / 2, kBodyCount - 1}) {
        const ysq::ElectricField3 reference =
            ysq::electricField(bodies[i].position, bodies);
        EXPECT_QUANTITY_VEC_NEAR(fields[i], reference,
                                 length(reference.value()) * 1e-2 *
                                     ysq::units::voltPerMetre)
            << "body " << i;
    }
}

TEST(ElectromagnetismField, MagneticFieldsAtLargeNAgreesWithTheSinglePointApi) {
    constexpr std::size_t kBodyCount = 1200;
    std::vector<Body> bodies;
    bodies.reserve(kBodyCount);
    std::mt19937 rng(8);
    std::uniform_real_distribution<double> positionDist(-10.0, 10.0);
    std::uniform_real_distribution<double> velocityDist(-1.0e3, 1.0e3);
    std::uniform_real_distribution<double> chargeDist(-1.0e-6, 1.0e-6);
    for (std::size_t i = 0; i < kBodyCount; ++i) {
        bodies.push_back(makePointCharge(
            chargeDist(rng),
            Vec3{positionDist(rng), positionDist(rng), positionDist(rng)},
            Vec3{velocityDist(rng), velocityDist(rng), velocityDist(rng)}));
    }

    const std::vector<ysq::MagneticFluxDensity3> fields = ysq::magneticFields(bodies);
    ASSERT_EQ(fields.size(), bodies.size());

    for (const std::size_t i : {std::size_t{0}, kBodyCount / 2, kBodyCount - 1}) {
        const ysq::MagneticFluxDensity3 reference =
            ysq::magneticField(bodies[i].position, bodies);
        const double referenceMagnitude = length(reference.value());
        // A tesla-scale absolute floor as well as a relative one: this
        // configuration's field values span many orders of magnitude
        // (random charges/velocities can nearly cancel), and a purely
        // relative tolerance is meaningless once the reference itself is
        // close to zero.
        EXPECT_QUANTITY_VEC_NEAR(fields[i], reference,
                                 std::max(referenceMagnitude * 1e-2, 1e-20) *
                                     ysq::units::tesla)
            << "body " << i;
    }
}

}  // namespace
