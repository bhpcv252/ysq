#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/RigidBody.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using ysq::Body;
using ysq::Vec3;

Vec3 angularVelocityBodyFrame(const Body& body) {
    const Vec3 momentumBody =
        rotate(conjugate(body.orientation), body.angularMomentum.value());
    const Vec3 moments = body.principalMomentsOfInertia.value();
    return {momentumBody.x / moments.x, momentumBody.y / moments.y,
            momentumBody.z / moments.z};
}

double rotationalKineticEnergy(const Body& body) {
    const Vec3 omega = angularVelocityBodyFrame(body);
    const Vec3 moments = body.principalMomentsOfInertia.value();
    return 0.5 * (moments.x * omega.x * omega.x + moments.y * omega.y * omega.y +
                  moments.z * omega.z * omega.z);
}

TEST(PhysicsRigidBody, ZeroMomentsOfInertiaIsANoOp) {
    Body body{};
    body.orientation = ysq::Quat::fromAxisAngle(Vec3::unitX(), 0.3);
    body.angularMomentum = ysq::AngularMomentum3{Vec3{1.0, 2.0, 3.0}};
    const ysq::Quat orientationBefore = body.orientation;
    const ysq::AngularMomentum3 momentumBefore = body.angularMomentum;

    ysq::stepRigidBody(body, std::array<Body, 0>{}, 1.0);

    EXPECT_EQ(body.orientation, orientationBefore);
    EXPECT_EQ(body.angularMomentum, momentumBefore);
}

TEST(PhysicsRigidBody, TorqueFreeMotionConservesAngularMomentumMagnitudeAndEnergy) {
    Body body{};
    body.principalMomentsOfInertia = ysq::MomentOfInertia3{Vec3{1.0, 1.5, 2.0}};
    body.orientation = ysq::Quat::identity();
    body.angularMomentum = ysq::AngularMomentum3{Vec3{0.3, 0.2, 1.0}};

    const double initialMagnitude = length(body.angularMomentum.value());
    const double initialEnergy = rotationalKineticEnergy(body);

    for (int i = 0; i < 5000; ++i) {
        ysq::stepRigidBody(body, std::array<Body, 0>{}, 0.001);
    }

    EXPECT_NEAR(length(body.angularMomentum.value()), initialMagnitude,
                initialMagnitude * 1e-6);
    EXPECT_NEAR(rotationalKineticEnergy(body), initialEnergy, initialEnergy * 1e-5);
    EXPECT_NEAR(length(body.orientation.xyz()) * length(body.orientation.xyz()) +
                    body.orientation.w * body.orientation.w,
                1.0, 1e-9)
        << "orientation must stay a unit quaternion after many steps";
}

TEST(PhysicsRigidBody, TorqueFreeSymmetricTopPrecessesAtTheClosedFormRate) {
    // Goldstein's standard result for an axisymmetric torque-free top
    // (Ixx = Iyy != Izz): the body-frame angular velocity's component
    // perpendicular to the symmetry axis rotates at
    // Omega = (Izz - Ixx) / Ixx * omegaZ, omegaZ itself constant.
    const double ixx = 1.0;
    const double izz = 2.0;
    const double omegaZ = 1.0;
    const double omegaPerp0 = 0.1;

    Body body{};
    body.principalMomentsOfInertia = ysq::MomentOfInertia3{Vec3{ixx, ixx, izz}};
    body.orientation = ysq::Quat::identity();
    body.angularMomentum =
        ysq::AngularMomentum3{Vec3{ixx * omegaPerp0, 0.0, izz * omegaZ}};

    const double dt = 0.0005;
    const int steps = 4000;
    for (int i = 0; i < steps; ++i) {
        ysq::stepRigidBody(body, std::array<Body, 0>{}, dt);
    }
    const double elapsed = dt * steps;

    const Vec3 omega = angularVelocityBodyFrame(body);
    const double expectedOmega = (izz - ixx) / ixx * omegaZ;
    const double expectedAngle =
        std::fmod(expectedOmega * elapsed, 2.0 * ysq::kPi<double>);

    EXPECT_NEAR(omega.z, omegaZ, omegaZ * 1e-6);
    EXPECT_NEAR(std::sqrt(omega.x * omega.x + omega.y * omega.y), omegaPerp0,
                omegaPerp0 * 1e-4);
    EXPECT_NEAR(std::atan2(omega.y, omega.x), expectedAngle, 1e-3);
}

TEST(PhysicsRigidBody, GravityGradientTorqueVanishesForASphericallySymmetricBody) {
    Body sphere{};
    sphere.mass = ysq::Mass{1.0e10};
    sphere.principalMomentsOfInertia = ysq::MomentOfInertia3{Vec3{5.0, 5.0, 5.0}};
    sphere.orientation = ysq::Quat::fromAxisAngle(Vec3::unitY(), 0.7);

    Body perturber{};
    perturber.mass = ysq::Mass{5.0e24};
    perturber.position = ysq::Length3{Vec3{1.0e7, 2.0e6, 3.0e6}};

    const ysq::Torque3 torque =
        ysq::gravityGradientTorque(sphere, perturber.position, perturber.mass);
    EXPECT_QUANTITY_VEC_APPROX(torque, ysq::Torque3{});
}

TEST(PhysicsRigidBody, GravityGradientTorqueMatchesTheClosedFormOnTheEquatorialLimb) {
    // A perturber in the oblate body's own equatorial plane (perpendicular
    // to spin axis +Z): rHat . zHat = 0, so I . rHat = Ixx rHat exactly
    // (Ixx = Iyy for the axisymmetric case here), and rHat x (Ixx rHat) is
    // zero: the torque vanishes on the equatorial limb, same as the polar
    // axis, and is only nonzero off both.
    Body oblate{};
    const double ixx = 8.0e37;
    const double izz = 8.1e37;
    oblate.principalMomentsOfInertia = ysq::MomentOfInertia3{Vec3{ixx, ixx, izz}};
    oblate.orientation = ysq::Quat::identity();

    Body perturber{};
    perturber.mass = ysq::Mass{7.342e22};
    const double r = 3.844e8;
    perturber.position = ysq::Length3{Vec3{r, 0.0, 0.0}};

    const ysq::Torque3 torque =
        ysq::gravityGradientTorque(oblate, perturber.position, perturber.mass);
    EXPECT_NEAR(length(torque.value()), 0.0, 1e-6);
}

TEST(PhysicsRigidBody, GravityGradientTorqueMatchesTheClosedFormOffAxis) {
    Body oblate{};
    const double ixx = 8.0e37;
    const double izz = 8.1e37;
    oblate.principalMomentsOfInertia = ysq::MomentOfInertia3{Vec3{ixx, ixx, izz}};
    oblate.orientation = ysq::Quat::identity();

    Body perturber{};
    const double gm = 3.986004418e14;  // Earth's own GM, reused as a stand-in mass here
    perturber.mass = ysq::Mass{gm / ysq::constants::G.value()};
    const double r = 3.844e8;
    // 45 degrees above the equatorial plane: rHat = (1, 0, 1) / sqrt(2).
    perturber.position = ysq::Length3{Vec3{r / std::sqrt(2.0), 0.0, r / std::sqrt(2.0)}};

    const ysq::Torque3 torque =
        ysq::gravityGradientTorque(oblate, perturber.position, perturber.mass);

    // rHat = (s, 0, s), s = 1/sqrt2; I . rHat = (ixx s, 0, izz s);
    // cross(a, b).y = a.z*b.x - a.x*b.z, so
    // rHat x (I . rHat) = (0, s*(ixx s) - s*(izz s), 0) = (0, s^2 (ixx - izz), 0).
    const double s = 1.0 / std::sqrt(2.0);
    const double expectedY = (3.0 * gm / (r * r * r)) * s * s * (ixx - izz);

    EXPECT_NEAR(torque.value().y, expectedY, std::abs(expectedY) * 1e-6);
    EXPECT_NEAR(torque.value().x, 0.0, std::abs(expectedY) * 1e-6 + 1e-30);
    EXPECT_NEAR(torque.value().z, 0.0, std::abs(expectedY) * 1e-6 + 1e-30);
}

}  // namespace
