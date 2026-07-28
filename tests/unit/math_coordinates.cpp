#include <Math/CoordinateSystems.hpp>

#include <Math/Format.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>

namespace {

using ysq::Cylindrical;
using ysq::Mat3;
using ysq::Polar;
using ysq::Spherical;
using ysq::Vec2;
using ysq::Vec3;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kPi = ysq::kPi<double>;

double zeroTolerance(double scale) {
    return 64.0 * kEps * scale;
}

/// Points in every octant, none of them on an axis or a plane, so a swapped
/// pair of angles cannot coincide with the right answer.
constexpr std::array<Vec3, 6> kSamples{
    Vec3{1.0, 2.0, 3.0},    Vec3{-1.5, 0.5, -2.0},  Vec3{3.0, -4.0, 1.0},
    Vec3{-2.0, -1.0, -0.5}, Vec3{0.25, 0.75, -3.0}, Vec3{5.0, 1.0, 2.0},
};

// --- The convention itself --------------------------------------------------

TEST(MathCoordinates, PolarIsMeasuredFromPlusZAndAzimuthFromPlusX) {
    // Stated as a test because the mathematics convention swaps the two names,
    // and a swapped pair gives a plausible-looking wrong point rather than an
    // obvious failure.
    const Spherical<double> onPlusZ = ysq::toSpherical(Vec3{0.0, 0.0, 2.0});
    EXPECT_APPROX(onPlusZ.radius, 2.0);
    EXPECT_APPROX(onPlusZ.polar, 0.0) << "straight up is polar = 0";

    const Spherical<double> onMinusZ = ysq::toSpherical(Vec3{0.0, 0.0, -2.0});
    EXPECT_APPROX(onMinusZ.polar, kPi) << "straight down is polar = pi";

    const Spherical<double> inPlane = ysq::toSpherical(Vec3{2.0, 0.0, 0.0});
    EXPECT_APPROX(inPlane.polar, kPi / 2.0) << "the equator is polar = pi/2";
    EXPECT_APPROX(inPlane.azimuth, 0.0) << "+x is azimuth = 0";

    EXPECT_APPROX(ysq::toSpherical(Vec3{0.0, 2.0, 0.0}).azimuth, kPi / 2.0);
    EXPECT_APPROX(ysq::toSpherical(Vec3{-2.0, 0.0, 0.0}).azimuth, kPi);
    EXPECT_APPROX(ysq::toSpherical(Vec3{0.0, -2.0, 0.0}).azimuth, -kPi / 2.0);
}

TEST(MathCoordinates, TheForwardFormulaMatchesTheDocumentedOne) {
    const Spherical<double> at{2.5, 0.7, -1.1};
    const Vec3 expected{at.radius * std::sin(at.polar) * std::cos(at.azimuth),
                        at.radius * std::sin(at.polar) * std::sin(at.azimuth),
                        at.radius * std::cos(at.polar)};
    EXPECT_VEC_NEAR(ysq::toCartesian(at), expected, zeroTolerance(10.0));
}

// --- Round trips ------------------------------------------------------------

TEST(MathCoordinates, SphericalRoundTripsThroughCartesian) {
    for (const Vec3& at : kSamples) {
        EXPECT_VEC_NEAR(ysq::toCartesian(ysq::toSpherical(at)), at, zeroTolerance(20.0));

        const Spherical<double> spherical = ysq::toSpherical(at);
        EXPECT_APPROX(spherical.radius, length(at));
        EXPECT_GE(spherical.polar, 0.0);
        EXPECT_LE(spherical.polar, kPi);
        EXPECT_GT(spherical.azimuth, -kPi - 1e-15);
        EXPECT_LE(spherical.azimuth, kPi);
    }
}

TEST(MathCoordinates, CylindricalRoundTripsThroughCartesian) {
    for (const Vec3& at : kSamples) {
        EXPECT_VEC_NEAR(ysq::toCartesian(ysq::toCylindrical(at)), at,
                        zeroTolerance(20.0));

        const Cylindrical<double> cylindrical = ysq::toCylindrical(at);
        EXPECT_APPROX(cylindrical.height, at.z);
        EXPECT_APPROX(cylindrical.radius, length(at.xy()));
        EXPECT_GE(cylindrical.radius, 0.0);
    }
}

TEST(MathCoordinates, PolarRoundTripsInTwoDimensions) {
    for (const Vec2& at : {Vec2{1.0, 2.0}, Vec2{-3.0, 0.5}, Vec2{0.0, -4.0}}) {
        EXPECT_VEC_NEAR(ysq::toCartesian(ysq::toPolar(at)), at, zeroTolerance(20.0));
        EXPECT_APPROX(ysq::toPolar(at).radius, length(at));
    }
    EXPECT_APPROX(ysq::toPolar(Vec2{1.0, 1.0}).angle, kPi / 4.0);
}

TEST(MathCoordinates, TheThreeSystemsAgreeWithEachOther) {
    for (const Vec3& at : kSamples) {
        const Spherical<double> spherical = ysq::toSpherical(at);
        const Cylindrical<double> cylindrical = ysq::toCylindrical(at);

        // Cylindrical radius is the spherical radius times sin(polar).
        EXPECT_NEAR(cylindrical.radius, spherical.radius * std::sin(spherical.polar),
                    zeroTolerance(10.0));
        EXPECT_NEAR(cylindrical.height, spherical.radius * std::cos(spherical.polar),
                    zeroTolerance(10.0));
        EXPECT_NEAR(cylindrical.azimuth, spherical.azimuth, zeroTolerance(10.0));
    }
}

// --- Degeneracies -----------------------------------------------------------

TEST(MathCoordinates, TheOriginAndThePolesHaveNoWellDefinedAngles) {
    // Documented behaviour rather than a correct answer: at the origin neither
    // angle exists, and on the axis the azimuth does not.
    const Spherical<double> origin = ysq::toSpherical(Vec3::zero());
    EXPECT_APPROX(origin.radius, 0.0);
    EXPECT_APPROX(origin.polar, 0.0);
    EXPECT_APPROX(origin.azimuth, 0.0);
    EXPECT_VEC_APPROX(ysq::toCartesian(origin), Vec3::zero());

    const Spherical<double> pole = ysq::toSpherical(Vec3{0.0, 0.0, 5.0});
    EXPECT_APPROX(pole.azimuth, 0.0);
    EXPECT_VEC_NEAR(ysq::toCartesian(pole), (Vec3{0.0, 0.0, 5.0}), zeroTolerance(10.0));

    const Cylindrical<double> onAxis = ysq::toCylindrical(Vec3{0.0, 0.0, 5.0});
    EXPECT_APPROX(onAxis.radius, 0.0);
    EXPECT_APPROX(onAxis.height, 5.0);
}

TEST(MathCoordinates, ThePolarAngleStaysAccurateNearTheAxis) {
    // This is why the polar angle comes from atan2 of the distance from the
    // axis against z, and not from acos(z / r). For a point this close to the
    // axis, z / r has already rounded to exactly 1 and acos returns exactly
    // zero, losing the answer completely.
    for (const double offset : {1e-4, 1e-6, 1e-8, 1e-10}) {
        const Vec3 nearAxis{offset, 0.0, 1.0};
        EXPECT_NEAR(ysq::toSpherical(nearAxis).polar, offset, offset * 1e-5)
            << "at offset " << offset;
    }

    const Vec3 veryNear{1e-10, 0.0, 1.0};
    EXPECT_EQ(std::acos(ysq::clamp(veryNear.z / length(veryNear), -1.0, 1.0)), 0.0)
        << "the acos route gives exactly zero here, which is why it is not used";
}

// --- Local bases ------------------------------------------------------------

TEST(MathCoordinates, TheSphericalBasisIsOrthonormalAndRightHanded) {
    for (const Vec3& at : kSamples) {
        const Mat3 basis = ysq::sphericalBasis(ysq::toSpherical(at));

        EXPECT_MAT_NEAR(transpose(basis) * basis, Mat3::identity(), zeroTolerance(10.0));
        EXPECT_NEAR(determinant(basis), 1.0, zeroTolerance(10.0))
            << "a determinant of -1 would mean the triple is left-handed";

        // e_polar cross e_azimuth is e_radius, in that cyclic order.
        EXPECT_VEC_NEAR(cross(basis[1], basis[2]), basis[0], zeroTolerance(10.0));

        // And the radial direction points at the point itself.
        EXPECT_VEC_NEAR(basis[0], normalized(at), zeroTolerance(10.0));
    }
}

TEST(MathCoordinates, TheCylindricalAndPolarBasesAreOrthonormalToo) {
    for (const Vec3& at : kSamples) {
        const Mat3 basis = ysq::cylindricalBasis(ysq::toCylindrical(at));
        EXPECT_MAT_NEAR(transpose(basis) * basis, Mat3::identity(), zeroTolerance(10.0));
        EXPECT_NEAR(determinant(basis), 1.0, zeroTolerance(10.0));
        EXPECT_VEC_NEAR(basis[2], Vec3::unitZ(), zeroTolerance(10.0));
        // The radial direction is the position projected into the xy plane.
        EXPECT_VEC_NEAR(basis[0], normalized(Vec3{at.x, at.y, 0.0}), zeroTolerance(10.0));
    }

    const auto polarBasis = ysq::polarBasis(Polar<double>{2.0, 0.7});
    EXPECT_MAT_NEAR(transpose(polarBasis) * polarBasis, ysq::Matrix2<double>::identity(),
                    zeroTolerance(10.0));
    EXPECT_NEAR(determinant(polarBasis), 1.0, zeroTolerance(10.0));
}

TEST(MathCoordinates, TheBasisAtAKnownPointIsTheExpectedTriple) {
    // On the +x axis at the equator the three directions are exactly the
    // Cartesian ones, permuted.
    const Spherical<double> onPlusX{1.0, kPi / 2.0, 0.0};
    const Mat3 basis = ysq::sphericalBasis(onPlusX);

    EXPECT_VEC_NEAR(basis[0], Vec3::unitX(), zeroTolerance(10.0));
    EXPECT_VEC_NEAR(basis[1], -Vec3::unitZ(), zeroTolerance(10.0))
        << "increasing polar angle moves down, away from +z";
    EXPECT_VEC_NEAR(basis[2], Vec3::unitY(), zeroTolerance(10.0))
        << "increasing azimuth moves toward +y";
}

// --- Vector components ------------------------------------------------------

TEST(MathCoordinates, ComponentTransformationPreservesLengthAndRoundTrips) {
    for (const Vec3& at : kSamples) {
        const Spherical<double> spherical = ysq::toSpherical(at);
        const Vec3 components{1.0, -2.0, 0.5};

        const Vec3 cartesian = ysq::sphericalComponentsToCartesian(spherical, components);
        EXPECT_APPROX(length(cartesian), length(components))
            << "an orthonormal change of basis is a rotation";

        EXPECT_VEC_NEAR(ysq::cartesianComponentsToSpherical(spherical, cartesian),
                        components, zeroTolerance(20.0));

        const Cylindrical<double> cylindrical = ysq::toCylindrical(at);
        const Vec3 inCylinder =
            ysq::cylindricalComponentsToCartesian(cylindrical, components);
        EXPECT_APPROX(length(inCylinder), length(components));
        EXPECT_VEC_NEAR(ysq::cartesianComponentsToCylindrical(cylindrical, inCylinder),
                        components, zeroTolerance(20.0));
    }
}

TEST(MathCoordinates, CircularMotionIsPurelyAzimuthalInTheLocalBasis) {
    // The point of having the basis at all: a velocity has components against
    // a frame that changes from place to place, and converting the position is
    // not enough.
    //
    // A point going anticlockwise round the equator at unit angular rate has
    // Cartesian velocity (-y, x, 0), which in the local spherical frame should
    // be purely along e_azimuth.
    for (const double azimuth : {0.0, 0.7, 2.5, -1.9}) {
        const double radius = 3.0;
        const Spherical<double> at{radius, kPi / 2.0, azimuth};
        const Vec3 position = ysq::toCartesian(at);
        const Vec3 velocity{-position.y, position.x, 0.0};

        const Vec3 local = ysq::cartesianComponentsToSpherical(at, velocity);
        EXPECT_NEAR(local.x, 0.0, zeroTolerance(20.0)) << "no radial component";
        EXPECT_NEAR(local.y, 0.0, zeroTolerance(20.0)) << "no polar component";
        EXPECT_NEAR(local.z, radius, zeroTolerance(20.0))
            << "and the azimuthal component is r times the angular rate";
    }
}

// --- Jacobians --------------------------------------------------------------

TEST(MathCoordinates, TheSphericalJacobianDeterminantIsTheVolumeElement) {
    for (const Vec3& at : kSamples) {
        const Spherical<double> spherical = ysq::toSpherical(at);
        const double expected =
            spherical.radius * spherical.radius * std::sin(spherical.polar);
        EXPECT_NEAR(determinant(ysq::sphericalJacobian(spherical)), expected,
                    zeroTolerance(100.0));
    }
}

TEST(MathCoordinates, TheCylindricalJacobianDeterminantIsTheRadius) {
    for (const Vec3& at : kSamples) {
        const Cylindrical<double> cylindrical = ysq::toCylindrical(at);
        EXPECT_NEAR(determinant(ysq::cylindricalJacobian(cylindrical)),
                    cylindrical.radius, zeroTolerance(20.0));
    }
}

TEST(MathCoordinates, TheJacobianMatchesADifferenceOfThePointMapping) {
    // The Jacobian columns are the unnormalised coordinate directions, so each
    // one has to match how the Cartesian point actually moves per unit of that
    // coordinate.
    const Spherical<double> at{2.5, 0.9, -0.4};
    const Mat3 jacobian = ysq::sphericalJacobian(at);

    constexpr double h = 1e-6;
    for (std::size_t i = 0; i < 3; ++i) {
        Spherical<double> forward = at;
        Spherical<double> backward = at;
        forward[i] += h;
        backward[i] -= h;

        const Vec3 numerical =
            (ysq::toCartesian(forward) - ysq::toCartesian(backward)) / (2.0 * h);
        EXPECT_VEC_NEAR(jacobian[i], numerical, 1e-8) << "column " << i;
    }
}

TEST(MathCoordinates, TheJacobianColumnsAreTheBasisDirectionsScaled) {
    const Spherical<double> at{2.5, 0.9, -0.4};
    const Mat3 basis = ysq::sphericalBasis(at);
    const Mat3 jacobian = ysq::sphericalJacobian(at);

    EXPECT_VEC_NEAR(jacobian[0], basis[0], zeroTolerance(10.0));
    EXPECT_VEC_NEAR(jacobian[1], basis[1] * at.radius, zeroTolerance(10.0));
    EXPECT_VEC_NEAR(jacobian[2], basis[2] * (at.radius * std::sin(at.polar)),
                    zeroTolerance(10.0));
}

// --- Plumbing ---------------------------------------------------------------

TEST(MathCoordinates, CoordinatesIndexAndFormatInDeclarationOrder) {
    Spherical<double> spherical{1.0, 2.0, 3.0};
    EXPECT_EQ(spherical[0], spherical.radius);
    EXPECT_EQ(spherical[1], spherical.polar);
    EXPECT_EQ(spherical[2], spherical.azimuth);
    spherical[1] = 9.0;
    EXPECT_EQ(spherical.polar, 9.0);

    const Cylindrical<double> cylindrical{1.0, 2.0, 3.0};
    EXPECT_EQ(cylindrical[2], cylindrical.height);

    EXPECT_EQ(std::format("{}", (Spherical<double>{1.0, 2.0, 3.0})), "(1, 2, 3)");
    EXPECT_EQ(std::format("{:.1f}", (Polar<double>{1.0, 2.0})), "(1.0, 2.0)");

    EXPECT_EQ((Spherical<double>{1.0, 2.0, 3.0}), (Spherical<double>{1.0, 2.0, 3.0}));
    EXPECT_NE((Spherical<double>{1.0, 2.0, 3.0}), (Spherical<double>{1.0, 2.0, 4.0}));
}

TEST(MathCoordinates, WorksAtSinglePrecision) {
    using ysq::Vec3f;
    constexpr float tol = 1e-5f;

    const Vec3f at{1.0f, 2.0f, 3.0f};
    EXPECT_VEC_NEAR(ysq::toCartesian(ysq::toSpherical(at)), at, tol);

    const auto basis = ysq::sphericalBasis(ysq::toSpherical(at));
    EXPECT_MAT_NEAR(transpose(basis) * basis, ysq::Mat3f::identity(), tol);
}

}  // namespace
