#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Gravity/SphericalHarmonics.hpp>
#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <span>

namespace {

using ysq::Body;
using ysq::Length3;
using ysq::SphericalHarmonicsField;
using ysq::Vec3;

}  // namespace

TEST(PhysicsGravitySphericalHarmonics, MonopoleOnlyMatchesPlainNewtonianGravity) {
    SphericalHarmonicsField field{};
    field.mass = ysq::Mass{5.972e24};
    field.referenceRadius = ysq::Length{6.378e6};
    // No harmonic terms at all: the field is exactly GM/r.

    const Length3 position{Vec3{2.0e7, 1.0e7, 0.5e7}};
    const ysq::Acceleration3 acceleration =
        ysq::sphericalHarmonicsAcceleration(field, position);

    const double gm = ysq::constants::G.value() * field.mass.value();
    const double r = ysq::length(position).value();
    const Vec3 expected = -position.value() * (gm / (r * r * r));

    EXPECT_NEAR(acceleration.value().x, expected.x, std::abs(expected.x) * 1e-6);
    EXPECT_NEAR(acceleration.value().y, expected.y, std::abs(expected.y) * 1e-6);
    EXPECT_NEAR(acceleration.value().z, expected.z, std::abs(expected.z) * 1e-6);
}

TEST(PhysicsGravitySphericalHarmonics, J2OnlyMatchesNewtonianHppsClosedFormJ2Term) {
    // C_2,0 = -J2, every other coefficient zero, should reproduce
    // Gravity/Newtonian.hpp's own analytic J2 oblateness term exactly (up
    // to numerical-differentiation precision): the defining cross-check
    // that this general implementation's convention and sign agree with
    // the special case already independently validated elsewhere.
    const double j2 = 1.08263e-3;  // Earth's own J2
    const double mass = 5.972e24;
    const double radius = 6.378e6;

    Body source{};
    source.mass = ysq::Mass{mass};
    source.radius = ysq::Length{radius};
    source.j2 = j2;
    source.orientation =
        ysq::Quat::identity();  // spin axis is +z, matching latitude = z/r

    SphericalHarmonicsField field{};
    field.mass = ysq::Mass{mass};
    field.referenceRadius = ysq::Length{radius};
    field.cosine = {{-j2, 0.0, 0.0}};
    field.sine = {{0.0, 0.0, 0.0}};

    for (const Vec3& raw : {Vec3{2.0e7, 1.0e7, 0.5e7}, Vec3{-1.5e7, 3.0e7, 1.0e7},
                            Vec3{0.0, 0.0, 3.0e7}, Vec3{4.0e7, 0.0, 0.0}}) {
        const Length3 position{raw};
        const ysq::Acceleration3 viaHarmonics =
            ysq::sphericalHarmonicsAcceleration(field, position);
        const ysq::Acceleration3 viaNewtonian =
            ysq::newtonianAcceleration(position, std::span<const Body>{&source, 1});

        const double magnitude = ysq::length(viaNewtonian).value();
        EXPECT_NEAR(viaHarmonics.value().x, viaNewtonian.value().x, magnitude * 1e-5);
        EXPECT_NEAR(viaHarmonics.value().y, viaNewtonian.value().y, magnitude * 1e-5);
        EXPECT_NEAR(viaHarmonics.value().z, viaNewtonian.value().z, magnitude * 1e-5);
    }
}

TEST(PhysicsGravitySphericalHarmonics, PotentialReducesToGMOverRWithNoHarmonicTerms) {
    SphericalHarmonicsField field{};
    field.mass = ysq::Mass{1.0e24};
    field.referenceRadius = ysq::Length{1.0e6};

    const Vec3 position{1.0e7, 0.0, 0.0};
    const double potential = ysq::sphericalHarmonicsPotential(field, position);
    const double expected = ysq::constants::G.value() * field.mass.value() / 1.0e7;

    EXPECT_NEAR(potential, expected, expected * 1e-12);
}

TEST(PhysicsGravitySphericalHarmonics,
     PotentialIsSymmetricUnderEquatorialReflectionForJ2Only) {
    // A pure J2 (zonal, degree-2) field has no north-south asymmetry: the
    // potential at a point and its mirror image across the equatorial
    // plane must be identical.
    SphericalHarmonicsField field{};
    field.mass = ysq::Mass{5.972e24};
    field.referenceRadius = ysq::Length{6.378e6};
    field.cosine = {{-1.08263e-3, 0.0, 0.0}};
    field.sine = {{0.0, 0.0, 0.0}};

    const Vec3 above{2.0e7, 1.0e7, 0.5e7};
    const Vec3 below{2.0e7, 1.0e7, -0.5e7};

    EXPECT_NEAR(ysq::sphericalHarmonicsPotential(field, above),
                ysq::sphericalHarmonicsPotential(field, below), 1e-3);
}

TEST(PhysicsGravitySphericalHarmonics, TesseralTermBreaksLongitudinalSymmetry) {
    // A nonzero C_22 term (a longitudinal, "tesseral" asymmetry) must make
    // the potential depend on longitude, unlike a purely zonal field.
    SphericalHarmonicsField field{};
    field.mass = ysq::Mass{5.972e24};
    field.referenceRadius = ysq::Length{6.378e6};
    field.cosine = {{0.0, 0.0, 1.0e-6}};  // C_2,2 nonzero
    field.sine = {{0.0, 0.0, 0.0}};

    const double r = 1.0e7;
    const Vec3 alongX{r, 0.0, 0.0};
    const Vec3 alongY{0.0, r, 0.0};

    const double potentialX = ysq::sphericalHarmonicsPotential(field, alongX);
    const double potentialY = ysq::sphericalHarmonicsPotential(field, alongY);

    EXPECT_GT(std::abs(potentialX - potentialY), 1e-6);
}
