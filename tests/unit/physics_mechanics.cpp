#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Mechanics/Frame.hpp>
#include <Physics/Mechanics/Kinematics.hpp>
#include <Units/Constants.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Body;
using ysq::Frame;
using ysq::Vec3;
using ysq::Vec4;

// --- Frame -------------------------------------------------------------

TEST(PhysicsFrame, TheLabFrameIsTheIdentityTransform) {
    Body body{};
    body.mass = ysq::Mass{3.0};
    body.position = ysq::Length3{Vec3{1.0, 2.0, 3.0}};
    body.momentum = ysq::Momentum3{Vec3{4.0, 5.0, 6.0}};

    const Body transformed = transformTo(Frame::lab(), body);
    EXPECT_EQ(transformed.position, body.position);
    EXPECT_EQ(transformed.momentum, body.momentum);
}

TEST(PhysicsFrame, TransformToAndFromRoundTrip) {
    Frame frame{};
    frame.origin = ysq::Length3{Vec3{10.0, -5.0, 2.0}};
    frame.velocity = ysq::Velocity3{Vec3{100.0, 0.0, -50.0}};

    Body body{};
    body.mass = ysq::Mass{7.0};
    body.position = ysq::Length3{Vec3{1.0, 2.0, 3.0}};
    body.momentum = ysq::Momentum3{Vec3{700.0, 1400.0, 2100.0}};

    const Body seenFromFrame = transformTo(frame, body);
    const Body roundTripped = transformFrom(frame, seenFromFrame);

    EXPECT_QUANTITY_VEC_APPROX(roundTripped.position, body.position);
    EXPECT_QUANTITY_VEC_APPROX(roundTripped.momentum, body.momentum);
}

TEST(PhysicsFrame, AMovingFrameShiftsMomentumByMassTimesItsVelocity) {
    Frame frame{};
    frame.velocity = ysq::Velocity3{Vec3{20.0, 0.0, 0.0}};

    Body body{};
    body.mass = ysq::Mass{5.0};
    body.momentum = ysq::Momentum3{Vec3{0.0, 0.0, 0.0}};

    const Body seenFromFrame = transformTo(frame, body);
    // The body is at rest in the lab, so in a frame moving at +x it appears
    // to move at -x: momentum shifts by -mass * frame.velocity.
    const ysq::Momentum3 expected{Vec3{-100.0, 0.0, 0.0}};
    EXPECT_QUANTITY_VEC_APPROX(seenFromFrame.momentum, expected);
}

// --- Kinematics ----------------------------------------------------------

TEST(PhysicsKinematics, LorentzFactorIsOneAtRest) {
    EXPECT_APPROX(static_cast<double>(ysq::lorentzFactor(ysq::Speed{0.0})), 1.0);
}

TEST(PhysicsKinematics, LorentzFactorAtSixTenthsCIsFiveQuarters) {
    // The classic textbook value: beta = 0.6, gamma = 1 / sqrt(1 - 0.36) = 1.25.
    const ysq::Speed v = 0.6 * ysq::constants::speedOfLight;
    EXPECT_NEAR(static_cast<double>(ysq::lorentzFactor(v)), 1.25, 1e-12);
}

TEST(PhysicsKinematics, FourVelocityHasMinkowskiNormMinusCSquared) {
    const ysq::Velocity3 v{Vec3{0.6, 0.0, 0.0} * ysq::constants::speedOfLight.value()};
    const ysq::Velocity4 u = ysq::fourVelocity(v);

    const double c = ysq::constants::speedOfLight.value();
    // Components are (t, x, y, z) against Vec4's (x, y, z, w) fields, per
    // Vector4's own documented convention for a four-vector: the time
    // component sits in .x, not .w.
    const Vec4 components = u.value();
    const double minkowskiNormSquared =
        -components.x * components.x + components.y * components.y +
        components.z * components.z + components.w * components.w;

    EXPECT_NEAR(minkowskiNormSquared, -(c * c), (c * c) * 1e-12);
}

TEST(PhysicsKinematics, ProperTimeRateIsOneAtRestAndFallsWithSpeed) {
    EXPECT_APPROX(static_cast<double>(ysq::properTimeRate(ysq::Speed{0.0})), 1.0);

    const ysq::Speed v = 0.6 * ysq::constants::speedOfLight;
    EXPECT_NEAR(static_cast<double>(ysq::properTimeRate(v)), 0.8, 1e-12);
}

TEST(PhysicsKinematics, ProperTimeElapsedAtConstantSpeedMatchesTOverGamma) {
    const ysq::Speed v = 0.6 * ysq::constants::speedOfLight;
    const ysq::Time duration = 1000.0 * ysq::units::second;

    const auto constantSpeed = [&](ysq::Time) { return v; };
    const ysq::Time elapsed =
        ysq::properTimeElapsed(constantSpeed, ysq::Time{0.0}, duration, 200);

    EXPECT_QUANTITY_NEAR(elapsed, duration * 0.8, 1e-6 * ysq::units::second);
}

TEST(PhysicsKinematics, RelativisticVelocityAdditionAtZeroFrameVelocityIsIdentity) {
    const ysq::Velocity3 u{Vec3{1.0, 2.0, 3.0}};
    const ysq::Velocity3 result = ysq::relativisticVelocityAdd(u, ysq::Velocity3{});
    EXPECT_QUANTITY_VEC_APPROX(result, u);
}

TEST(PhysicsKinematics, RelativisticVelocityAdditionOfCOverTwoAndCOverTwoIsFourFifthsC) {
    // The standard textbook composition, c/2 boosted by c/2 collinear, gives
    // 0.8c rather than the Galilean c. relativisticVelocityAdd transforms a
    // velocity FROM the original frame INTO one moving at frameVelocity
    // relative to it, the same direction Frame's Galilean transform goes
    // (u - frameVelocity at low speed). So reproducing "c/2 composed with
    // c/2" needs the frame receding at -c/2: an object moving at c/2 in the
    // lab, seen from a frame moving at -c/2, is a boost of +c/2 applied to
    // it, and comes out at 0.8c.
    const double c = ysq::constants::speedOfLight.value();
    const ysq::Velocity3 u{Vec3{0.5 * c, 0.0, 0.0}};
    const ysq::Velocity3 frameVelocity{Vec3{-0.5 * c, 0.0, 0.0}};

    const ysq::Velocity3 result = ysq::relativisticVelocityAdd(u, frameVelocity);
    EXPECT_NEAR(result.value().x, 0.8 * c, c * 1e-9);
    EXPECT_NEAR(result.value().y, 0.0, c * 1e-9);
}

TEST(PhysicsKinematics, RelativisticVelocityAdditionReducesToGalileanAtLowSpeed) {
    const ysq::Velocity3 u{Vec3{300.0, -150.0, 0.0}};
    const ysq::Velocity3 frameVelocity{Vec3{100.0, 0.0, 50.0}};

    const ysq::Velocity3 result = ysq::relativisticVelocityAdd(u, frameVelocity);
    const ysq::Velocity3 galilean = u - frameVelocity;

    // The correction is order (v/c)^2, utterly negligible at these speeds.
    EXPECT_QUANTITY_VEC_NEAR(result, galilean, 1e-6 * ysq::units::metrePerSecond);
}

}  // namespace
