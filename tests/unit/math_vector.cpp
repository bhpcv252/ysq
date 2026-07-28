#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <Math/Format.hpp>
#include <Math/Scalar.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>

namespace {

using ysq::Vec2;
using ysq::Vec3;
using ysq::Vec4;

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// Vectors that are not axis-aligned, not unit, not orthogonal to one another
/// and not all the same magnitude, so an identity that holds only by accident
/// in a tidy case has somewhere to fail. Magnitudes stay within an order of
/// each other; the cancellation-sensitive identities are hard enough to assert
/// without a 1e6 dynamic range on top.
constexpr std::array<Vec3, 5> kSamples{
    Vec3{1.0, 2.0, 3.0},   Vec3{-4.0, 0.5, 7.25},  Vec3{0.0, 0.0, 1.0},
    Vec3{2.5, -3.5, 0.75}, Vec3{-1.0, -1.0, -1.0},
};

/// An identity whose two sides cancel to (near) zero has its error set by the
/// size of the intermediate terms, not by the size of the answer, so a bare
/// absolute tolerance is either useless or meaningless. `scale` is the
/// magnitude of the largest intermediate product.
double zeroTolerance(double scale) {
    return 64.0 * kEps * scale;
}

double maxLength(const Vec3& a, const Vec3& b) {
    return std::max({length(a), length(b), 1.0});
}

double maxLength(const Vec3& a, const Vec3& b, const Vec3& c) {
    return std::max({length(a), length(b), length(c), 1.0});
}

// --- Vector space axioms ----------------------------------------------------

TEST(MathVector, AdditionIsCommutativeAndAssociative) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_EQ(a + b, b + a);
            for (const Vec3& c : kSamples) {
                EXPECT_VEC_APPROX((a + b) + c, a + (b + c));
            }
        }
    }
}

TEST(MathVector, ZeroIsTheAdditiveIdentityAndNegationTheInverse) {
    for (const Vec3& a : kSamples) {
        EXPECT_EQ(a + Vec3::zero(), a);
        EXPECT_EQ(a - a, Vec3::zero());
        EXPECT_EQ(a + (-a), Vec3::zero());
        EXPECT_EQ(+a, a);
        EXPECT_EQ(-(-a), a);
    }
}

TEST(MathVector, ScalarMultiplicationDistributesAndAssociates) {
    constexpr double s = 3.25;
    constexpr double t = -0.5;

    for (const Vec3& a : kSamples) {
        EXPECT_EQ(a * 1.0, a);
        EXPECT_EQ(a * 0.0, Vec3::zero());
        EXPECT_EQ(a * s, s * a);
        EXPECT_VEC_APPROX(a * (s * t), (a * s) * t);
        EXPECT_VEC_APPROX(a * (s + t), a * s + a * t);
        EXPECT_VEC_APPROX(a / s, a * (1.0 / s));

        for (const Vec3& b : kSamples) {
            EXPECT_VEC_APPROX((a + b) * s, a * s + b * s);
        }
    }
}

TEST(MathVector, CompoundAssignmentMatchesTheBinaryOperators) {
    const Vec3 a{1.0, 2.0, 3.0};
    const Vec3 b{-4.0, 0.5, 7.25};

    Vec3 v = a;
    EXPECT_EQ(&(v += b), &v) << "compound assignment must return *this";
    EXPECT_EQ(v, a + b);

    v = a;
    v -= b;
    EXPECT_EQ(v, a - b);

    v = a;
    v *= 3.0;
    EXPECT_EQ(v, a * 3.0);

    v = a;
    v /= 3.0;
    EXPECT_EQ(v, a / 3.0);
}

// --- Access ----------------------------------------------------------------

TEST(MathVector, IndexingMatchesTheNamedComponents) {
    Vec2 a2{1.0, 2.0};
    Vec3 a3{1.0, 2.0, 3.0};
    Vec4 a4{1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(a2[0], a2.x);
    EXPECT_EQ(a2[1], a2.y);
    EXPECT_EQ(a3[0], a3.x);
    EXPECT_EQ(a3[1], a3.y);
    EXPECT_EQ(a3[2], a3.z);
    EXPECT_EQ(a4[0], a4.x);
    EXPECT_EQ(a4[1], a4.y);
    EXPECT_EQ(a4[2], a4.z);
    EXPECT_EQ(a4[3], a4.w);

    a3[1] = 9.0;
    EXPECT_EQ(a3.y, 9.0);

    EXPECT_EQ(Vec2::size(), 2u);
    EXPECT_EQ(Vec3::size(), 3u);
    EXPECT_EQ(Vec4::size(), 4u);
}

TEST(MathVector, SwizzlesAndFactoriesProduceTheExpectedComponents) {
    const Vec4 a{1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(a.xy(), (Vec2{1.0, 2.0}));
    EXPECT_EQ(a.xyz(), (Vec3{1.0, 2.0, 3.0}));
    EXPECT_EQ((Vec3{1.0, 2.0, 3.0}.xy()), (Vec2{1.0, 2.0}));

    EXPECT_EQ(Vec3::splat(2.0), (Vec3{2.0, 2.0, 2.0}));
    EXPECT_EQ(Vec3::unitX(), (Vec3{1.0, 0.0, 0.0}));
    EXPECT_EQ(Vec3::unitY(), (Vec3{0.0, 1.0, 0.0}));
    EXPECT_EQ(Vec3::unitZ(), (Vec3{0.0, 0.0, 1.0}));
    EXPECT_EQ(Vec3{}, Vec3::zero()) << "default construction must zero";
}

// --- Dot product -----------------------------------------------------------

TEST(MathVector, DotIsSymmetricAndBilinear) {
    constexpr double s = 2.75;

    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_EQ(dot(a, b), dot(b, a));
            EXPECT_APPROX(dot(a * s, b), s * dot(a, b));

            for (const Vec3& c : kSamples) {
                EXPECT_NEAR_REL(dot(a + b, c), dot(a, c) + dot(b, c),
                                zeroTolerance(maxLength(a, b, c)));
            }
        }
    }
}

TEST(MathVector, DotWithSelfIsLengthSquared) {
    for (const Vec3& a : kSamples) {
        EXPECT_EQ(dot(a, a), lengthSquared(a));
        EXPECT_APPROX(length(a) * length(a), lengthSquared(a));
    }
}

TEST(MathVector, CauchySchwarzHolds) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_LE(std::abs(dot(a, b)),
                      length(a) * length(b) + zeroTolerance(maxLength(a, b)));
        }
    }
}

TEST(MathVector, TriangleInequalityHolds) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_LE(length(a + b),
                      length(a) + length(b) + zeroTolerance(maxLength(a, b)));
        }
    }
}

// --- Cross product ---------------------------------------------------------

TEST(MathVector, CrossIsAnticommutative) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_EQ(cross(a, b), -cross(b, a));
        }
        EXPECT_EQ(cross(a, a), Vec3::zero());
    }
}

TEST(MathVector, CrossFollowsTheRightHandRule) {
    EXPECT_EQ(cross(Vec3::unitX(), Vec3::unitY()), Vec3::unitZ());
    EXPECT_EQ(cross(Vec3::unitY(), Vec3::unitZ()), Vec3::unitX());
    EXPECT_EQ(cross(Vec3::unitZ(), Vec3::unitX()), Vec3::unitY());
}

TEST(MathVector, CrossIsOrthogonalToBothOperands) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            const Vec3 n = cross(a, b);
            const double tol = zeroTolerance(maxLength(a, b) * maxLength(a, b));
            EXPECT_NEAR(dot(n, a), 0.0, tol);
            EXPECT_NEAR(dot(n, b), 0.0, tol);
        }
    }
}

TEST(MathVector, CrossOfParallelVectorsVanishes) {
    for (const Vec3& a : kSamples) {
        const Vec3 parallel = a * -2.5;
        EXPECT_VEC_NEAR(cross(a, parallel), Vec3::zero(),
                        zeroTolerance(lengthSquared(a)));
    }
}

TEST(MathVector, CrossSatisfiesTheJacobiIdentity) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            for (const Vec3& c : kSamples) {
                const Vec3 sum =
                    cross(a, cross(b, c)) + cross(b, cross(c, a)) + cross(c, cross(a, b));
                const double s = maxLength(a, b, c);
                EXPECT_VEC_NEAR(sum, Vec3::zero(), zeroTolerance(s * s * s));
            }
        }
    }
}

TEST(MathVector, CrossSatisfiesTheLagrangeIdentity) {
    // |a x b|^2 + (a . b)^2 == |a|^2 |b|^2
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            const double left = lengthSquared(cross(a, b)) + dot(a, b) * dot(a, b);
            const double right = lengthSquared(a) * lengthSquared(b);
            const double s = maxLength(a, b);
            EXPECT_NEAR(left, right, zeroTolerance(s * s * s * s));
        }
    }
}

TEST(MathVector, CrossSatisfiesTheBacCabIdentity) {
    // a x (b x c) == b (a . c) - c (a . b)
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            for (const Vec3& c : kSamples) {
                const Vec3 left = cross(a, cross(b, c));
                const Vec3 right = b * dot(a, c) - c * dot(a, b);
                const double s = maxLength(a, b, c);
                EXPECT_VEC_NEAR(left, right, zeroTolerance(s * s * s));
            }
        }
    }
}

TEST(MathVector, ScalarTripleProductIsCyclic) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            for (const Vec3& c : kSamples) {
                const double s = maxLength(a, b, c);
                const double tol = zeroTolerance(s * s * s);
                EXPECT_NEAR(scalarTriple(a, b, c), scalarTriple(b, c, a), tol);
                EXPECT_NEAR(scalarTriple(a, b, c), scalarTriple(c, a, b), tol);
                // Swapping any two arguments flips the sign.
                EXPECT_NEAR(scalarTriple(a, b, c), -scalarTriple(b, a, c), tol);
            }
        }
    }
}

// --- Length and normalization ----------------------------------------------

TEST(MathVector, NormalizedIsUnitLengthAndParallelToTheInput) {
    for (const Vec3& a : kSamples) {
        const Vec3 unit = normalized(a);
        EXPECT_APPROX(length(unit), 1.0);
        EXPECT_VEC_NEAR(cross(unit, a), Vec3::zero(), zeroTolerance(length(a)));
        EXPECT_GT(dot(unit, a), 0.0) << "normalization must not flip direction";
        EXPECT_VEC_APPROX(unit * length(a), a);
    }
}

TEST(MathVector, TryNormalizedRejectsZeroAndNonFinite) {
    EXPECT_FALSE(tryNormalized(Vec3::zero()).has_value());
    EXPECT_FALSE(tryNormalized(Vec2::zero()).has_value());
    EXPECT_FALSE(tryNormalized(Vec4::zero()).has_value());

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(tryNormalized(Vec3{nan, 0.0, 0.0}).has_value())
        << "a NaN length must fail, not compare its way through";

    const auto unit = tryNormalized(Vec3{0.0, 3.0, 4.0});
    ASSERT_TRUE(unit.has_value());
    EXPECT_VEC_APPROX(*unit, (Vec3{0.0, 0.6, 0.8}));
}

TEST(MathVector, NormalizedOfZeroYieldsNotANumber) {
    // Documented behaviour: the unchecked form propagates rather than
    // inventing a direction. A wrong unit vector would be far harder to trace.
    const Vec3 bad = normalized(Vec3::zero());
    EXPECT_TRUE(std::isnan(bad.x));
    EXPECT_TRUE(std::isnan(bad.y));
    EXPECT_TRUE(std::isnan(bad.z));
}

TEST(MathVector, DistanceIsSymmetricAndMatchesTheLengthOfTheDifference) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_EQ(distance(a, b), distance(b, a));
            EXPECT_EQ(distance(a, b), length(a - b));
            EXPECT_EQ(distanceSquared(a, b), lengthSquared(a - b));
        }
    }
}

TEST(MathVector, NormalisationRefusesWhatItCannotDoRatherThanReturningZero) {
    // The squared length overflows for a component beyond about 1.3e154 and
    // underflows below about 1.5e-162. Both vectors below have a perfectly
    // representable direction, and neither can be reached without rescaling
    // first, so the honest answer is nullopt. The division used to run anyway
    // and hand back a zero vector inside a successful optional, which is the
    // one thing this module is not supposed to do.
    constexpr double huge = 1e200;
    constexpr double tiny = 1e-200;

    EXPECT_FALSE(std::isfinite(length(Vec3{huge, huge, huge})));
    EXPECT_FALSE(tryNormalized(Vec3{huge, huge, huge}).has_value());
    EXPECT_FALSE(tryNormalized(Vec3{tiny, tiny, tiny}).has_value());
    EXPECT_FALSE(tryNormalized(Vec2{huge, huge}).has_value());
    EXPECT_FALSE(tryNormalized(Vec4{huge, huge, huge, huge}).has_value());

    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(tryNormalized(Vec3{inf, 0.0, 0.0}).has_value());

    // The limit is on the component, not on how many there are: one component
    // of 1e200 squares to 1e400 and overflows on its own. Just inside the
    // range, at 1e150, everything still works.
    EXPECT_FALSE(tryNormalized(Vec3{huge, 0.0, 0.0}).has_value());

    const auto inRange = tryNormalized(Vec3{1e150, 1e150, 0.0});
    ASSERT_TRUE(inRange.has_value());
    EXPECT_APPROX(length(*inRange), 1.0);
    EXPECT_VEC_NEAR(*inRange, (Vec3{std::sqrt(0.5), std::sqrt(0.5), 0.0}), 1e-15);
}

// --- Decomposition ---------------------------------------------------------

TEST(MathVector, ProjectAndRejectDecomposeTheVector) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& onto : kSamples) {
            const Vec3 along = project(a, onto);
            const Vec3 across = reject(a, onto);
            const double s = maxLength(a, onto);

            EXPECT_VEC_NEAR(along + across, a, zeroTolerance(s));
            EXPECT_NEAR(dot(across, onto), 0.0, zeroTolerance(s * s));
            EXPECT_VEC_NEAR(cross(along, onto), Vec3::zero(), zeroTolerance(s * s));
        }
    }
}

TEST(MathVector, ReflectPreservesLengthAndIsItsOwnInverse) {
    const Vec3 n = normalized(Vec3{1.0, 2.0, -0.5});

    for (const Vec3& a : kSamples) {
        const Vec3 mirrored = reflect(a, n);
        EXPECT_APPROX(length(mirrored), length(a));
        EXPECT_VEC_NEAR(reflect(mirrored, n), a, zeroTolerance(length(a)));
        // The component along the normal flips; the rest is untouched.
        EXPECT_NEAR(dot(mirrored, n), -dot(a, n), zeroTolerance(length(a)));
        EXPECT_VEC_NEAR(reject(mirrored, n), reject(a, n), zeroTolerance(length(a)));
    }
}

// --- Interpolation ---------------------------------------------------------

TEST(MathVector, LerpIsExactAtTheEndpoints) {
    for (const Vec3& a : kSamples) {
        for (const Vec3& b : kSamples) {
            EXPECT_EQ(lerp(a, b, 0.0), a);
            EXPECT_EQ(lerp(a, b, 1.0), b)
                << "the a + (b - a) * t form is not exact here; this one must be";
            EXPECT_VEC_APPROX(lerp(a, b, 0.5), (a + b) * 0.5);
        }
    }
}

TEST(MathVector, LerpExtrapolatesOutsideTheUnitInterval) {
    const Vec3 a{1.0, 0.0, 0.0};
    const Vec3 b{3.0, 0.0, 0.0};
    EXPECT_VEC_APPROX(lerp(a, b, 2.0), (Vec3{5.0, 0.0, 0.0}));
    EXPECT_VEC_APPROX(lerp(a, b, -1.0), (Vec3{-1.0, 0.0, 0.0}));
}

// --- Rotation invariance ---------------------------------------------------

TEST(MathVector, RotationPreservesLengthsAnglesAndHandedness) {
    // The strongest single check on the algebra: a rotation is an isometry, so
    // every length, dot product and cross product has to come through it
    // unchanged. Built here from cross and dot alone, before Matrix3 or
    // Quaternion exist to be trusted.
    const Vec3 axis = normalized(Vec3{1.0, -2.0, 0.5});

    for (const double angle : {0.0, 0.3, 1.0, 2.5, ysq::kPi<double>, 4.0}) {
        for (const Vec3& a : kSamples) {
            const Vec3 ra = rotateAbout(a, axis, angle);
            const double s = length(a);
            EXPECT_NEAR(length(ra), s, zeroTolerance(std::max(s, 1.0)));

            for (const Vec3& b : kSamples) {
                const Vec3 rb = rotateAbout(b, axis, angle);
                const double t = maxLength(a, b);
                EXPECT_NEAR(dot(ra, rb), dot(a, b), zeroTolerance(t * t));
                // Rotations preserve orientation, so the cross product rotates
                // with the operands rather than picking up a sign.
                EXPECT_VEC_NEAR(cross(ra, rb), rotateAbout(cross(a, b), axis, angle),
                                zeroTolerance(t * t));
            }
        }
    }
}

TEST(MathVector, RotationAboutAnAxisLeavesThatAxisAlone) {
    const Vec3 axis = normalized(Vec3{1.0, -2.0, 0.5});
    EXPECT_VEC_APPROX(rotateAbout(axis, axis, 1.234), axis);
}

// --- Angles ----------------------------------------------------------------

TEST(MathVector, AngleBetweenMatchesKnownAngles) {
    EXPECT_APPROX(angleBetween(Vec3::unitX(), Vec3::unitX()), 0.0);
    EXPECT_APPROX(angleBetween(Vec3::unitX(), Vec3::unitY()), ysq::kPi<double> / 2.0);
    EXPECT_APPROX(angleBetween(Vec3::unitX(), -Vec3::unitX()), ysq::kPi<double>);
    EXPECT_APPROX(angleBetween(Vec3{1.0, 1.0, 0.0}, Vec3::unitX()),
                  ysq::kPi<double> / 4.0);

    // Scale invariant.
    EXPECT_APPROX(angleBetween(Vec3::unitX() * 1e3, Vec3::unitY() * 1e-3),
                  ysq::kPi<double> / 2.0);
}

TEST(MathVector, AngleBetweenStaysAccurateForNearlyParallelVectors) {
    // This is why angleBetween is atan2(|a x b|, a . b) and not
    // acos(dot / (|a| |b|)). For a small angle the dot product rounds to
    // exactly 1 and acos returns exactly 0, losing the answer completely.
    const Vec3 a = normalized(Vec3{1.0, 2.0, 3.0});
    const Vec3 axis = normalized(cross(a, Vec3::unitZ()));

    for (const double angle : {1e-4, 1e-6, 1e-8}) {
        const Vec3 b = rotateAbout(a, axis, angle);
        EXPECT_NEAR(angleBetween(a, b), angle, angle * 1e-5) << "at angle " << angle;
    }

    // The contrast, stated as a floor rather than as an exact zero. The dot
    // product of two nearly parallel unit vectors is a sum of three products
    // whose last bits depend on whether the compiler contracted them into
    // fused multiply-adds, so it lands either at 1 or an ulp below it. Either
    // way acos has stopped resolving the angle, which is the point; asserting
    // on which of the two would be asserting on the code generator.
    const Vec3 b = rotateAbout(a, axis, 1e-8);
    const double viaAcos = std::acos(ysq::clamp(dot(a, b), -1.0, 1.0));
    EXPECT_TRUE(viaAcos == 0.0 || viaAcos > 1e-8)
        << "the acos form is pinned at its floor here, not tracking the angle: "
           "it returned "
        << viaAcos;
    EXPECT_NEAR(angleBetween(a, b), 1e-8, 1e-13) << "while this one still resolves it";
}

TEST(MathVector, TwoDimensionalAngleBetweenIsSigned) {
    EXPECT_APPROX(angleBetween(Vec2::unitX(), Vec2::unitY()), ysq::kPi<double> / 2.0);
    EXPECT_APPROX(angleBetween(Vec2::unitY(), Vec2::unitX()), -ysq::kPi<double> / 2.0);
}

// --- Two-dimensional specifics ---------------------------------------------

TEST(MathVector, TwoDimensionalCrossIsTheSignedParallelogramArea) {
    EXPECT_EQ(cross(Vec2::unitX(), Vec2::unitY()), 1.0);
    EXPECT_EQ(cross(Vec2::unitY(), Vec2::unitX()), -1.0);
    EXPECT_EQ(cross(Vec2{3.0, 0.0}, Vec2{0.0, 2.0}), 6.0);

    // It is the z component of the 3D cross of the lifted vectors.
    const Vec2 a{1.5, -2.0};
    const Vec2 b{0.25, 4.0};
    EXPECT_APPROX(cross(a, b), cross(Vec3{a.x, a.y, 0.0}, Vec3{b.x, b.y, 0.0}).z);
}

TEST(MathVector, PerpendicularIsAQuarterTurnCounterClockwise) {
    const Vec2 a{3.0, -1.0};
    const Vec2 p = perpendicular(a);
    EXPECT_EQ(p, (Vec2{1.0, 3.0}));
    EXPECT_APPROX(dot(a, p), 0.0);
    EXPECT_APPROX(length(p), length(a));
    EXPECT_GT(cross(a, p), 0.0) << "counter-clockwise, not clockwise";
    EXPECT_EQ(perpendicular(perpendicular(a)), -a);
}

// --- Four-dimensional specifics --------------------------------------------

TEST(MathVector, HomogeneousHelpersSetTheExpectedW) {
    const Vec3 v{1.0, 2.0, 3.0};
    EXPECT_EQ(Vec4::point(v).w, 1.0);
    EXPECT_EQ(Vec4::direction(v).w, 0.0);
    EXPECT_EQ(Vec4::point(v).xyz(), v);
    EXPECT_VEC_APPROX(perspectiveDivide(Vec4{2.0, 4.0, 6.0, 2.0}), v);
}

// --- Componentwise operations ----------------------------------------------

TEST(MathVector, ComponentwiseOperationsActPerComponent) {
    const Vec3 a{1.0, -5.0, 3.0};
    const Vec3 b{4.0, 2.0, -6.0};

    EXPECT_EQ(hadamard(a, b), (Vec3{4.0, -10.0, -18.0}));
    EXPECT_EQ(min(a, b), (Vec3{1.0, -5.0, -6.0}));
    EXPECT_EQ(max(a, b), (Vec3{4.0, 2.0, 3.0}));
    EXPECT_EQ(abs(a), (Vec3{1.0, 5.0, 3.0}));
    EXPECT_EQ(min(a, b) + max(a, b), a + b) << "min and max must partition";
}

// --- Scalar helpers --------------------------------------------------------

TEST(MathScalar, DegreesAndRadiansRoundTrip) {
    EXPECT_APPROX(ysq::radians(180.0), ysq::kPi<double>);
    EXPECT_APPROX(ysq::degrees(ysq::kPi<double>), 180.0);
    for (const double d : {0.0, 30.0, 90.0, -45.0, 360.0}) {
        EXPECT_APPROX(ysq::degrees(ysq::radians(d)), d);
    }
}

TEST(MathScalar, ClampAndSignBehaveAtTheBoundaries) {
    EXPECT_EQ(ysq::clamp(0.5, 0.0, 1.0), 0.5);
    EXPECT_EQ(ysq::clamp(-1.0, 0.0, 1.0), 0.0);
    EXPECT_EQ(ysq::clamp(2.0, 0.0, 1.0), 1.0);
    EXPECT_EQ(ysq::clamp(0.0, 0.0, 1.0), 0.0);
    EXPECT_EQ(ysq::clamp(1.0, 0.0, 1.0), 1.0);

    EXPECT_EQ(ysq::sign(3.0), 1.0);
    EXPECT_EQ(ysq::sign(-3.0), -1.0);
    EXPECT_EQ(ysq::sign(0.0), 0.0);
    EXPECT_EQ(ysq::sign(std::numeric_limits<double>::quiet_NaN()), 0.0)
        << "NaN is neither above nor below zero";
}

TEST(MathScalar, ApproxEqualMixesRelativeAndAbsoluteTolerance) {
    EXPECT_TRUE(ysq::approxEqual(1.0, 1.0 + kEps));
    EXPECT_FALSE(ysq::approxEqual(1.0, 1.1));

    // Relative: at magnitude 1e9 the tolerance is about 2.8e-5 absolute, so a
    // gap of 1e-6 passes and 1e-3 does not. An absolute-only rule would reject
    // both, which is what makes EXPECT_NEAR useless at this scale.
    EXPECT_TRUE(ysq::approxEqual(1e9, 1e9 + 1e-6));
    EXPECT_FALSE(ysq::approxEqual(1e9, 1e9 + 1e-3));

    // Absolute: near zero, where a relative tolerance collapses to nothing.
    EXPECT_TRUE(ysq::approxEqual(0.0, 1e-18));
    EXPECT_FALSE(ysq::approxEqual(0.0, 1e-6));

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(ysq::approxEqual(inf, inf));
    EXPECT_FALSE(ysq::approxEqual(inf, -inf));
    EXPECT_FALSE(ysq::approxEqual(nan, nan)) << "NaN is not equal to itself";

    EXPECT_TRUE(ysq::isNearZero(0.0));
    EXPECT_TRUE(ysq::isNearZero(-1e-18));
    EXPECT_FALSE(ysq::isNearZero(1e-6));
}

// --- Formatting ------------------------------------------------------------

TEST(MathVector, FormattingForwardsTheSpecToEachComponent) {
    EXPECT_EQ(std::format("{}", Vec2{1.0, 2.5}), "(1, 2.5)");
    EXPECT_EQ(std::format("{:.3f}", Vec3{1.0, 0.0, -9.81}), "(1.000, 0.000, -9.810)");
    EXPECT_EQ(std::format("{:.1f}", Vec4{1.0, 2.0, 3.0, 4.0}), "(1.0, 2.0, 3.0, 4.0)");
}

// --- The same identities at single precision -------------------------------

TEST(MathVector, IdentitiesHoldAtSinglePrecision) {
    // Not a formality: float is where -Wdouble-promotion has anything to say,
    // and where a stray double literal in a header would show up as a silent
    // widening rather than a compile error.
    using ysq::Vec3f;
    constexpr float tol = 1e-5f;

    const Vec3f a{1.0f, 2.0f, 3.0f};
    const Vec3f b{-4.0f, 0.5f, 7.25f};

    EXPECT_EQ(cross(a, b), -cross(b, a));
    EXPECT_NEAR(dot(cross(a, b), a), 0.0f, tol);
    EXPECT_NEAR(length(normalized(a)), 1.0f, tol);
    EXPECT_EQ(lerp(a, b, 1.0f), b);
    EXPECT_NEAR(lengthSquared(cross(a, b)) + dot(a, b) * dot(a, b),
                lengthSquared(a) * lengthSquared(b), 1e-2f);
}

}  // namespace
