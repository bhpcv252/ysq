#include <Math/Quaternion.hpp>

#include <Math/Format.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>

namespace {

using ysq::Mat3;
using ysq::Quat;
using ysq::Vec3;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kPi = ysq::kPi<double>;

double zeroTolerance(double scale) {
    return 64.0 * kEps * scale;
}

const Vec3 kAxis = normalized(Vec3{1.0, -2.0, 0.5});

/// Rotations that are not about an axis direction, not small, and not related
/// to each other by anything tidy.
const std::array<Quat, 4> kRotations{
    Quat::fromAxisAngle(kAxis, 0.7),
    Quat::fromAxisAngle(normalized(Vec3{3.0, 1.0, -1.0}), 2.1),
    Quat::fromAxisAngle(Vec3::unitZ(), -1.3),
    Quat::fromEulerZYX(0.4, -0.9, 1.7),
};

const std::array<Vec3, 3> kVectors{Vec3{1.0, 0.0, 0.0}, Vec3{2.0, -3.0, 0.5},
                                   Vec3{-1.0, 1.0, 4.0}};

/// Two quaternions describe the same rotation when they are equal up to sign.
::testing::AssertionResult sameRotation(const char* aExpr, const char* bExpr,
                                        const Quat& a, const Quat& b) {
    const double angle = angleBetween(a, b);
    if (angle < 1e-9) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << std::format(
               "\n  {} = {}\n  {} = {}\ndiffer by {:.6g} rad", aExpr, a, bExpr, b, angle);
}

#define EXPECT_SAME_ROTATION(a, b) EXPECT_PRED_FORMAT2(sameRotation, a, b)

// --- Basics -----------------------------------------------------------------

TEST(MathQuaternion, DefaultConstructionIsTheIdentityRotation) {
    // Deliberately not zero, which is not a rotation at all.
    EXPECT_EQ(Quat{}, Quat::identity());
    EXPECT_EQ(Quat{}.w, 1.0);
    EXPECT_APPROX(length(Quat{}), 1.0);
    EXPECT_EQ(Quat::zero(), (Quat{0.0, 0.0, 0.0, 0.0}));
    EXPECT_NE(Quat::zero(), Quat::identity());
}

TEST(MathQuaternion, IndexingIsScalarPartFirst) {
    Quat q{1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(q[0], q.w);
    EXPECT_EQ(q[1], q.x);
    EXPECT_EQ(q[2], q.y);
    EXPECT_EQ(q[3], q.z);
    EXPECT_EQ(q.xyz(), (Vec3{2.0, 3.0, 4.0}));
    EXPECT_EQ(Quat::size(), 4u);

    q[0] = 9.0;
    EXPECT_EQ(q.w, 9.0);

    EXPECT_EQ(Quat::fromScalarVector(1.0, Vec3{2.0, 3.0, 4.0}),
              (Quat{1.0, 2.0, 3.0, 4.0}));
}

// --- Algebra ----------------------------------------------------------------

TEST(MathQuaternion, ProductIsAssociativeButNotCommutative) {
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];
    const Quat& c = kRotations[2];

    EXPECT_VEC_NEAR((a * b) * c, a * (b * c), zeroTolerance(4.0));
    EXPECT_NE(a * b, b * a) << "quaternion multiplication does not commute";

    EXPECT_EQ(a * Quat::identity(), a);
    EXPECT_EQ(Quat::identity() * a, a);
}

TEST(MathQuaternion, TheBasisUnitsMultiplyLikeHamiltonSaid) {
    // i^2 = j^2 = k^2 = ijk = -1, which is the whole definition.
    const Quat one{1.0, 0.0, 0.0, 0.0};
    const Quat i{0.0, 1.0, 0.0, 0.0};
    const Quat j{0.0, 0.0, 1.0, 0.0};
    const Quat k{0.0, 0.0, 0.0, 1.0};

    EXPECT_EQ(i * i, -one);
    EXPECT_EQ(j * j, -one);
    EXPECT_EQ(k * k, -one);
    EXPECT_EQ(i * j * k, -one);

    EXPECT_EQ(i * j, k);
    EXPECT_EQ(j * k, i);
    EXPECT_EQ(k * i, j);
    EXPECT_EQ(j * i, -k) << "and the other way round picks up a sign";
}

TEST(MathQuaternion, ConjugationReversesProductsAndNormIsMultiplicative) {
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];

    EXPECT_EQ(conjugate(conjugate(a)), a);
    EXPECT_VEC_NEAR(conjugate(a * b), conjugate(b) * conjugate(a), zeroTolerance(4.0));
    EXPECT_APPROX(length(a * b), length(a) * length(b));
    EXPECT_APPROX(length(conjugate(a)), length(a));
    EXPECT_APPROX(lengthSquared(a), dot(a, a));
}

TEST(MathQuaternion, InverseUndoesTheRotationFromBothSides) {
    const Quat scaled = kRotations[0] * 3.0;  // deliberately not unit

    EXPECT_VEC_NEAR(scaled * inverse(scaled), Quat::identity(), zeroTolerance(4.0));
    EXPECT_VEC_NEAR(inverse(scaled) * scaled, Quat::identity(), zeroTolerance(4.0));

    // For a unit quaternion the conjugate is already the inverse.
    const Quat& unit = kRotations[0];
    EXPECT_VEC_NEAR(inverseUnit(unit), inverse(unit), zeroTolerance(4.0));
    EXPECT_VEC_NEAR(inverseUnit(scaled) / 9.0, inverse(scaled), zeroTolerance(4.0))
        << "inverseUnit skips the norm, which is exactly why it is unit-only";
}

TEST(MathQuaternion, VectorSpaceOperationsActComponentwise) {
    const Quat a{1.0, 2.0, 3.0, 4.0};
    const Quat b{5.0, -1.0, 0.5, 2.0};

    EXPECT_EQ(a + b, (Quat{6.0, 1.0, 3.5, 6.0}));
    EXPECT_EQ(a - b, (Quat{-4.0, 3.0, 2.5, 2.0}));
    EXPECT_EQ(a * 2.0, (Quat{2.0, 4.0, 6.0, 8.0}));
    EXPECT_EQ(2.0 * a, a * 2.0);
    EXPECT_EQ(a / 2.0, (Quat{0.5, 1.0, 1.5, 2.0}));
    EXPECT_EQ(-a, (Quat{-1.0, -2.0, -3.0, -4.0}));
    EXPECT_EQ(+a, a);

    Quat m = a;
    m *= b;
    EXPECT_EQ(m, a * b) << "*= must right-multiply";
}

// --- Rotation ---------------------------------------------------------------

TEST(MathQuaternion, RotationPreservesLengthsAndAngles) {
    for (const Quat& q : kRotations) {
        for (const Vec3& v : kVectors) {
            EXPECT_APPROX(length(rotate(q, v)), length(v));
        }
        EXPECT_NEAR(dot(rotate(q, kVectors[0]), rotate(q, kVectors[1])),
                    dot(kVectors[0], kVectors[1]), zeroTolerance(20.0));
    }
}

TEST(MathQuaternion, RotateAgreesWithTheMatrixItConvertsTo) {
    // Two separate derivations: the short cross-product form against the 3x3
    // expansion. Nothing is shared between them, so a sign error in either one
    // shows up here.
    for (const Quat& q : kRotations) {
        const Mat3 m = toMatrix3(q);
        for (const Vec3& v : kVectors) {
            EXPECT_VEC_NEAR(rotate(q, v), m * v, zeroTolerance(20.0));
        }
    }
}

TEST(MathQuaternion, RotateAgreesWithRodrigues) {
    // And a third derivation, from Vector3, reached without quaternions at all.
    for (const double angle : {0.3, 1.0, 2.5, -1.2, kPi}) {
        const Quat q = Quat::fromAxisAngle(kAxis, angle);
        for (const Vec3& v : kVectors) {
            EXPECT_VEC_NEAR(rotate(q, v), rotateAbout(v, kAxis, angle),
                            zeroTolerance(20.0))
                << "at angle " << angle;
        }
    }
}

TEST(MathQuaternion, NegatingAQuaternionLeavesTheRotationUnchanged) {
    // The double cover: q and -q are different quaternions and the same
    // rotation. Anything comparing rotations has to know that.
    for (const Quat& q : kRotations) {
        EXPECT_NE(q, -q);
        for (const Vec3& v : kVectors) {
            EXPECT_VEC_NEAR(rotate(q, v), rotate(-q, v), zeroTolerance(20.0));
        }
        EXPECT_NEAR(angleBetween(q, -q), 0.0, 1e-12);
    }
}

TEST(MathQuaternion, CompositionAppliesTheRightHandFactorFirst) {
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];
    for (const Vec3& v : kVectors) {
        EXPECT_VEC_NEAR(rotate(a * b, v), rotate(a, rotate(b, v)), zeroTolerance(20.0));
    }
    // Matching the matrix convention exactly.
    EXPECT_MAT_NEAR(toMatrix3(a * b), toMatrix3(a) * toMatrix3(b), zeroTolerance(20.0));
}

TEST(MathQuaternion, RotationAboutAnAxisLeavesThatAxisAlone) {
    for (const double angle : {0.3, 1.0, kPi}) {
        const Quat q = Quat::fromAxisAngle(kAxis, angle);
        EXPECT_VEC_NEAR(rotate(q, kAxis), kAxis, zeroTolerance(10.0));
    }
}

TEST(MathQuaternion, QuarterTurnsGoTheRightWay) {
    const Quat aboutZ = Quat::fromAxisAngle(Vec3::unitZ(), kPi / 2.0);
    EXPECT_VEC_NEAR(rotate(aboutZ, Vec3::unitX()), Vec3::unitY(), zeroTolerance(10.0));
    const Quat aboutX = Quat::fromAxisAngle(Vec3::unitX(), kPi / 2.0);
    EXPECT_VEC_NEAR(rotate(aboutX, Vec3::unitY()), Vec3::unitZ(), zeroTolerance(10.0));
}

// --- Matrix conversion ------------------------------------------------------

TEST(MathQuaternion, ToMatrix3ProducesAProperRotation) {
    for (const Quat& q : kRotations) {
        const Mat3 m = toMatrix3(q);
        EXPECT_MAT_NEAR(transpose(m) * m, Mat3::identity(), zeroTolerance(10.0));
        EXPECT_NEAR(determinant(m), 1.0, zeroTolerance(10.0));
    }
    EXPECT_MAT_NEAR(toMatrix3(Quat::identity()), Mat3::identity(), zeroTolerance(4.0));
}

TEST(MathQuaternion, MatrixConversionRoundTrips) {
    for (const Quat& q : kRotations) {
        EXPECT_SAME_ROTATION(Quat::fromRotationMatrix(toMatrix3(q)), q);
    }
    for (const double angle : {0.1, 1.0, 2.0, -2.8}) {
        const Mat3 m = Mat3::rotation(kAxis, angle);
        EXPECT_MAT_NEAR(toMatrix3(Quat::fromRotationMatrix(m)), m, zeroTolerance(20.0));
    }
}

TEST(MathQuaternion, MatrixConversionSurvivesHalfTurns) {
    // The reason fromRotationMatrix branches on the largest component. A half
    // turn has trace -1, so the divisor in the textbook trace-only derivation
    // goes to zero and the result is noise. Small rotations would never show
    // it.
    const std::array<Vec3, 5> axes{Vec3::unitX(), Vec3::unitY(), Vec3::unitZ(),
                                   normalized(Vec3{1.0, 1.0, 1.0}), kAxis};

    for (const Vec3& axis : axes) {
        const Mat3 m = Mat3::rotation(axis, kPi);
        const Quat q = Quat::fromRotationMatrix(m);

        EXPECT_APPROX(length(q), 1.0) << "half turn about " << axis;
        EXPECT_MAT_NEAR(toMatrix3(q), m, zeroTolerance(20.0));
        for (const Vec3& v : kVectors) {
            EXPECT_VEC_NEAR(rotate(q, v), rotateAbout(v, axis, kPi), zeroTolerance(20.0));
        }
    }

    // And just short of a half turn, where the trace is near its minimum.
    const Mat3 nearly = Mat3::rotation(kAxis, kPi - 1e-7);
    EXPECT_APPROX(length(Quat::fromRotationMatrix(nearly)), 1.0);
    EXPECT_MAT_NEAR(toMatrix3(Quat::fromRotationMatrix(nearly)), nearly, 1e-7);
}

// --- Axis and angle ---------------------------------------------------------

TEST(MathQuaternion, AxisAngleRoundTrips) {
    for (const double angle : {0.3, 1.0, 2.5, 3.0}) {
        const auto decomposed = toAxisAngle(Quat::fromAxisAngle(kAxis, angle));
        EXPECT_NEAR(decomposed.angle, angle, zeroTolerance(10.0));
        EXPECT_VEC_NEAR(decomposed.axis, kAxis, zeroTolerance(10.0));
    }

    // A negative angle comes back as a positive one about the flipped axis,
    // which is the same rotation.
    const auto flipped = toAxisAngle(Quat::fromAxisAngle(kAxis, -1.0));
    EXPECT_NEAR(flipped.angle, 1.0, zeroTolerance(10.0));
    EXPECT_VEC_NEAR(flipped.axis, -kAxis, zeroTolerance(10.0));
}

TEST(MathQuaternion, AxisAngleOfTheIdentityIsZeroWithAnArbitraryAxis) {
    const auto decomposed = toAxisAngle(Quat::identity());
    EXPECT_NEAR(decomposed.angle, 0.0, 1e-15);
    EXPECT_APPROX(length(decomposed.axis), 1.0)
        << "the axis is arbitrary here, but it still has to be a unit vector";
}

TEST(MathQuaternion, AxisAngleStaysAccurateForTinyRotations) {
    // Same reason Vector3::angleBetween uses atan2: near the identity, w
    // rounds to exactly 1 and acos(w) returns exactly 0, losing the angle.
    for (const double angle : {1e-4, 1e-6, 1e-8}) {
        const auto decomposed = toAxisAngle(Quat::fromAxisAngle(kAxis, angle));
        EXPECT_NEAR(decomposed.angle, angle, angle * 1e-5) << "at " << angle;
        EXPECT_VEC_NEAR(decomposed.axis, kAxis, 1e-6);
    }

    EXPECT_EQ(std::acos(ysq::clamp(Quat::fromAxisAngle(kAxis, 1e-9).w, -1.0, 1.0)), 0.0)
        << "the acos form loses this completely, which is why it is not used";
}

// --- Euler angles -----------------------------------------------------------

TEST(MathQuaternion, EulerZyxMatchesTheMatrixProductItNames) {
    const double yaw = 0.4;
    const double pitch = -0.9;
    const double roll = 1.7;

    const Mat3 expected =
        Mat3::rotationZ(yaw) * Mat3::rotationY(pitch) * Mat3::rotationX(roll);
    EXPECT_MAT_NEAR(toMatrix3(Quat::fromEulerZYX(yaw, pitch, roll)), expected,
                    zeroTolerance(20.0));
}

TEST(MathQuaternion, EulerZyxRoundTripsAwayFromGimbalLock) {
    const std::array<ysq::EulerZYX<double>, 4> angles{
        ysq::EulerZYX<double>{0.4, -0.9, 1.7},
        ysq::EulerZYX<double>{0.0, 0.0, 0.0},
        ysq::EulerZYX<double>{2.0, 0.5, -1.0},
        ysq::EulerZYX<double>{-1.5, 1.2, 3.0},
    };

    for (const auto& e : angles) {
        const auto back = toEulerZYX(Quat::fromEulerZYX(e.yaw, e.pitch, e.roll));
        EXPECT_NEAR(back.yaw, e.yaw, 1e-12);
        EXPECT_NEAR(back.pitch, e.pitch, 1e-12);
        EXPECT_NEAR(back.roll, e.roll, 1e-12);
    }
}

TEST(MathQuaternion, EulerZyxIsDegenerateAtGimbalLockButStillTheSameRotation) {
    // At either pole only one combination of yaw and roll is determined, so
    // the individual angles cannot round trip. The rotation they describe
    // still must, which is the property that actually matters, and it is what
    // the degenerate branch in toEulerZYX exists to preserve.
    for (const double pole : {kPi / 2.0, -kPi / 2.0}) {
        for (const double yaw : {0.4, -2.0, 0.0}) {
            for (const double roll : {1.7, 0.5, -1.1}) {
                const Quat q = Quat::fromEulerZYX(yaw, pole, roll);
                const auto back = toEulerZYX(q);

                EXPECT_NEAR(back.pitch, pole, 1e-7);
                EXPECT_EQ(back.roll, 0.0) << "roll is pinned, not guessed";
                EXPECT_SAME_ROTATION(Quat::fromEulerZYX(back.yaw, back.pitch, back.roll),
                                     q);
            }
        }
    }
}

TEST(MathQuaternion, EulerZyxDegradesGracefullyApproachingGimbalLock) {
    // Approaching a pole the general branch stays usable and simply loses
    // accuracy, roughly as epsilon over the distance from the pole, because
    // that is asin's conditioning near its endpoints. The tolerance below says
    // exactly that, and it is why the degenerate branch switches in only at
    // 2^-40 rather than somewhere comfortable: switching early would replace a
    // graceful loss with a hard one.
    for (const double offset : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}) {
        const Quat q = Quat::fromEulerZYX(0.4, kPi / 2.0 - offset, 1.7);
        const auto back = toEulerZYX(q);
        const double tolerance = 1e-11 + 100.0 * kEps / offset;

        EXPECT_NEAR(angleBetween(Quat::fromEulerZYX(back.yaw, back.pitch, back.roll), q),
                    0.0, tolerance)
            << "at offset " << offset;
    }
}

// --- Interpolation ----------------------------------------------------------

TEST(MathQuaternion, SlerpIsExactAtTheEndpointsAndStaysUnit) {
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];

    EXPECT_SAME_ROTATION(slerp(a, b, 0.0), a);
    EXPECT_SAME_ROTATION(slerp(a, b, 1.0), b);

    for (const double t : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
        EXPECT_NEAR(length(slerp(a, b, t)), 1.0, 1e-12) << "at t = " << t;
        EXPECT_NEAR(length(nlerp(a, b, t)), 1.0, 1e-12) << "at t = " << t;
    }
}

TEST(MathQuaternion, SlerpTurnsAtAConstantRate) {
    // This is the whole point of slerp over nlerp: the angle from the start
    // grows linearly in t.
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];
    const double total = angleBetween(a, b);
    ASSERT_GT(total, 0.5) << "the fixture needs a wide arc to say anything";

    for (const double t : {0.1, 0.25, 0.5, 0.75, 0.9}) {
        EXPECT_NEAR(angleBetween(a, slerp(a, b, t)), total * t, 1e-9) << "at t = " << t;
    }
}

TEST(MathQuaternion, NlerpMatchesSlerpAtTheMidpointAndNotElsewhere) {
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];

    // By symmetry both land on the bisector at the halfway point.
    EXPECT_SAME_ROTATION(nlerp(a, b, 0.5), slerp(a, b, 0.5));
    // Away from it they genuinely differ, so the two are not the same function
    // with different spelling.
    EXPECT_GT(angleBetween(nlerp(a, b, 0.25), slerp(a, b, 0.25)), 1e-3);
}

TEST(MathQuaternion, InterpolationTakesTheShortWayRound) {
    const Quat& a = kRotations[0];
    const Quat& b = kRotations[1];

    // b and -b are the same rotation, so interpolating toward either has to
    // trace the same path rather than going the long way round the sphere.
    for (const double t : {0.25, 0.5, 0.75}) {
        EXPECT_SAME_ROTATION(slerp(a, b, t), slerp(a, -b, t));
        EXPECT_SAME_ROTATION(nlerp(a, b, t), nlerp(a, -b, t));
    }
}

TEST(MathQuaternion, SlerpHandlesNearlyIdenticalRotations) {
    // The sin(theta) divisor goes to zero here; the implementation falls back
    // to nlerp rather than dividing by it.
    const Quat& a = kRotations[0];
    const Quat b = Quat::fromAxisAngle(kAxis, 1e-12) * a;

    for (const double t : {0.0, 0.5, 1.0}) {
        const Quat mid = slerp(a, b, t);
        EXPECT_TRUE(std::isfinite(mid.w));
        EXPECT_NEAR(length(mid), 1.0, 1e-12);
        // Tight on purpose: the separation is 1e-12 rad, so anything that
        // bottoms out around sqrt(epsilon) fails here rather than passing on a
        // tolerance wide enough to hide it.
        EXPECT_NEAR(angleBetween(a, mid), 0.0, 1e-11);
    }
}

TEST(MathQuaternion, AngleBetweenResolvesAnglesFarBelowSquareRootEpsilon) {
    // The dot-product route to this number cannot resolve anything under about
    // 3e-8, because it recovers a half-angle sine from a cosine that has
    // already rounded to 1.
    //
    // What can be resolved is bounded from the other side. Two order-one
    // quaternions separated by theta differ in their components by about
    // theta/2, so the difference is carried in roughly log2(theta / epsilon)
    // bits and the relative accuracy available is about epsilon / theta. At
    // theta = 1e-13 that is 2e-3, and no arrangement of the arithmetic beats
    // it: the information is not in the inputs. So the tolerance below is an
    // absolute floor of a hundred epsilons rather than a relative accuracy,
    // and it is still a real assertion, since it excludes both zero and the
    // 3e-8 the acos route would return.
    //
    // Asking for a tighter relative accuracy here used to pass on arm64 and
    // fail on x86-64. Not a compiler bug: fused multiply-add is baseline on
    // arm64 and clang contracts the quaternion product into it, which keeps
    // extra bits through the cancellation, while x86-64 without an explicit
    // -march has no FMA instruction to contract into. A tolerance that a
    // contraction can move is a tolerance below the floor.
    constexpr double kAbsoluteFloor = 100.0 * kEps;

    for (const double angle : {1e-7, 1e-9, 1e-11, 1e-13}) {
        const Quat a = kRotations[0];
        const Quat b = Quat::fromAxisAngle(kAxis, angle) * a;
        const double measured = angleBetween(a, b);

        EXPECT_NEAR(measured, angle, std::max(angle * 1e-6, kAbsoluteFloor))
            << "at " << angle;
    }

    // The contrast. Below about 1e-8 the cosine of the half angle has rounded
    // to within an ulp of one, so acos returns either exactly zero or its own
    // floor of sqrt(epsilon), and which of the two depends on whether the
    // compiler contracted the dot product into an FMA. Either way the answer
    // has stopped depending on the angle, which is the whole point.
    for (const double angle : {1e-9, 1e-11, 1e-13}) {
        const Quat a = kRotations[0];
        const Quat b = Quat::fromAxisAngle(kAxis, angle) * a;

        const double cosHalf = std::abs(dot(normalized(a), normalized(b)));
        const double viaAcos = 2.0 * std::acos(ysq::clamp(cosHalf, -1.0, 1.0));

        EXPECT_TRUE(viaAcos == 0.0 || viaAcos > 1e-8)
            << "the acos route is pinned at its floor rather than tracking the "
               "angle, at "
            << angle << ": it returned " << viaAcos;
        EXPECT_GT(angleBetween(a, b), angle / 2.0)
            << "while this one still returns an answer, at " << angle;
    }
}

// --- Angle between ----------------------------------------------------------

TEST(MathQuaternion, AngleBetweenIsTheRotationTakingOneToTheOther) {
    EXPECT_NEAR(angleBetween(Quat::identity(), Quat::identity()), 0.0, 1e-15);

    for (const double angle : {0.3, 1.0, 2.5, 3.0}) {
        EXPECT_NEAR(angleBetween(Quat::identity(), Quat::fromAxisAngle(kAxis, angle)),
                    angle, zeroTolerance(10.0))
            << "at " << angle;
    }

    // Scale invariant, since it normalises internally.
    EXPECT_NEAR(angleBetween(kRotations[0] * 3.0, kRotations[1] * 0.25),
                angleBetween(kRotations[0], kRotations[1]), 1e-12);
}

// --- Formatting -------------------------------------------------------------

TEST(MathQuaternion, FormattingPrintsScalarPartFirst) {
    EXPECT_EQ(std::format("{}", Quat{1.0, 2.0, 3.0, 4.0}), "(1, 2, 3, 4)");
    EXPECT_EQ(std::format("{:.2f}", Quat::identity()), "(1.00, 0.00, 0.00, 0.00)");
}

// --- Single precision -------------------------------------------------------

TEST(MathQuaternion, IdentitiesHoldAtSinglePrecision) {
    using ysq::Quatf;
    using ysq::Vec3f;
    constexpr float tol = 1e-5f;

    const Vec3f axis = normalized(Vec3f{1.0f, -2.0f, 0.5f});
    const Quatf q = Quatf::fromAxisAngle(axis, 0.7f);
    const Vec3f v{2.0f, -3.0f, 0.5f};

    EXPECT_NEAR(length(q), 1.0f, tol);
    EXPECT_NEAR(length(rotate(q, v)), length(v), tol);
    EXPECT_VEC_NEAR(rotate(q, v), toMatrix3(q) * v, tol);
    EXPECT_VEC_NEAR(rotate(inverse(q), rotate(q, v)), v, tol);
}

}  // namespace
