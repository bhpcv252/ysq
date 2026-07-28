#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Electromagnetism/Lorentz.hpp>
#include <Units/Electromagnetism.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Body;
using ysq::Vec3;

Body makeChargedBody(double mass, double charge, Vec3 velocity) {
    Body body{};
    body.mass = ysq::Mass{mass};
    body.charge = ysq::ElectricCharge{charge};
    body.momentum = ysq::Momentum3{velocity * mass};
    return body;
}

TEST(ElectromagnetismLorentz, PureElectricForceIsChargeTimesField) {
    const Body body = makeChargedBody(1.0, 3.0e-6, Vec3{});
    const ysq::ElectricField3 e{Vec3{100.0, 0.0, 0.0}};
    const ysq::MagneticFluxDensity3 b{};

    const ysq::Force3 force = ysq::lorentzForce(body, e, b);
    const Vec3 expected{3.0e-4, 0.0, 0.0};
    EXPECT_VEC_NEAR(force.value(), expected, 1e-18);
}

TEST(ElectromagnetismLorentz, PureMagneticForceIsPerpendicularToVelocity) {
    const Body body = makeChargedBody(1.0, 2.0e-6, Vec3{10.0, 0.0, 0.0});
    const ysq::ElectricField3 e{};
    const ysq::MagneticFluxDensity3 b{Vec3{0.0, 0.0, 1.0}};

    const ysq::Force3 force = ysq::lorentzForce(body, e, b);
    // v x B = (10,0,0) x (0,0,1) = (0*1-0*0, 0*0-10*1, 0) = (0,-10,0)
    const Vec3 expected{0.0, -2.0e-5, 0.0};
    EXPECT_VEC_NEAR(force.value(), expected, 1e-18);
    EXPECT_NEAR(dot(force.value(), body.velocity().value()), 0.0, 1e-18);
}

TEST(ElectromagnetismLorentz, ForceIsTheSumOfBothTerms) {
    const Body body = makeChargedBody(1.0, 1.0, Vec3{1.0, 0.0, 0.0});
    const ysq::ElectricField3 e{Vec3{0.0, 5.0, 0.0}};
    const ysq::MagneticFluxDensity3 b{Vec3{0.0, 0.0, 2.0}};

    const ysq::Force3 force = ysq::lorentzForce(body, e, b);
    // magnetic: (1,0,0) x (0,0,2) = (0*2-0*0, 0*0-1*2, 0) = (0,-2,0)
    // total = (0,5,0) + (0,-2,0) = (0,3,0)
    const Vec3 expected{0.0, 3.0, 0.0};
    EXPECT_VEC_NEAR(force.value(), expected, 1e-12);
}

TEST(ElectromagnetismLorentz, ANeutralBodyFeelsNoForce) {
    const Body body = makeChargedBody(1.0, 0.0, Vec3{5.0, 5.0, 5.0});
    const ysq::ElectricField3 e{Vec3{1.0, 1.0, 1.0}};
    const ysq::MagneticFluxDensity3 b{Vec3{1.0, 1.0, 1.0}};

    const ysq::Force3 force = ysq::lorentzForce(body, e, b);
    EXPECT_VEC_NEAR(force.value(), Vec3{}, 1e-30);
}

}  // namespace
