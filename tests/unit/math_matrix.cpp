#include <Math/Matrix2.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Matrix4.hpp>

#include <Math/Format.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <limits>

namespace {

using ysq::Mat2;
using ysq::Mat3;
using ysq::Mat4;
using ysq::Vec2;
using ysq::Vec3;
using ysq::Vec4;

constexpr double kEps = std::numeric_limits<double>::epsilon();

double zeroTolerance(double scale) {
    return 64.0 * kEps * scale;
}

/// Invertible, not symmetric, not orthogonal, no zero entries to hide an index
/// mix-up behind. A matrix full of tidy zeros will pass a transposed
/// implementation without complaint.
const Mat3 kAwkward3 = Mat3::fromRows({2.0, -1.0, 3.0}, {0.5, 4.0, -2.0},
                                      {-1.5, 1.0, 5.0});

const Mat3 kAwkwardOther3 = Mat3::fromRows({1.0, 2.0, -1.0}, {3.0, -0.5, 2.0},
                                           {0.25, 1.0, 4.0});

const Mat4 kAwkward4 =
    Mat4::fromRows({2.0, -1.0, 3.0, 0.5}, {0.5, 4.0, -2.0, 1.0},
                   {-1.5, 1.0, 5.0, -3.0}, {1.0, 0.25, -0.5, 2.0});

const Mat4 kAwkwardOther4 =
    Mat4::fromRows({1.0, 2.0, -1.0, 3.0}, {3.0, -0.5, 2.0, 1.0},
                   {0.25, 1.0, 4.0, -2.0}, {-1.0, 0.5, 1.5, 1.0});

const Mat2 kAwkward2 = Mat2::fromRows({2.0, -1.0}, {0.5, 4.0});
const Mat2 kAwkwardOther2 = Mat2::fromRows({1.0, 3.0}, {-0.5, 2.0});

// --- Storage and access -----------------------------------------------------

TEST(MathMatrix, StorageIsColumnMajorSoItUploadsToOpenGlUntransposed) {
    // The whole convention rests on this: glUniformMatrix4fv with
    // transpose = GL_FALSE reads sixteen scalars as four columns. If the bytes
    // came out in row order every shader in the project would see a transposed
    // matrix, and nothing else in this file would catch it.
    const Mat4 m = Mat4::fromRows({1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0},
                                  {9.0, 10.0, 11.0, 12.0},
                                  {13.0, 14.0, 15.0, 16.0});

    std::array<double, 16> raw{};
    static_assert(sizeof(raw) == sizeof(m));
    std::memcpy(raw.data(), &m, sizeof(m));

    const std::array<double, 16> expected{1.0, 5.0, 9.0,  13.0, 2.0, 6.0,
                                          10.0, 14.0, 3.0, 7.0,  11.0, 15.0,
                                          4.0, 8.0, 12.0, 16.0};
    EXPECT_EQ(raw, expected);

    static_assert(std::is_standard_layout_v<Mat4>);
    static_assert(std::is_trivially_copyable_v<Mat4>);
    static_assert(sizeof(Mat4) == 16 * sizeof(double));
    static_assert(sizeof(Mat3) == 9 * sizeof(double));
    static_assert(sizeof(Mat2) == 4 * sizeof(double));
}

TEST(MathMatrix, SubscriptIndexesAColumnAndCallIndexesAnElement) {
    const Mat3 m = Mat3::fromRows({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0},
                                  {7.0, 8.0, 9.0});

    EXPECT_EQ(m[0], (Vec3{1.0, 4.0, 7.0})) << "operator[] must give a column";
    EXPECT_EQ(m[1], (Vec3{2.0, 5.0, 8.0}));
    EXPECT_EQ(m[2], (Vec3{3.0, 6.0, 9.0}));

    EXPECT_EQ(m.row(0), (Vec3{1.0, 2.0, 3.0}));
    EXPECT_EQ(m.row(2), (Vec3{7.0, 8.0, 9.0}));

    EXPECT_EQ(m(0, 1), 2.0) << "operator() is (row, col), reading order";
    EXPECT_EQ(m(1, 0), 4.0);
    EXPECT_EQ(m(2, 2), 9.0);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_EQ(m(r, c), m[c][r]) << "at (" << r << ", " << c << ")";
        }
    }
}

TEST(MathMatrix, FromRowsIsTheTransposeOfFromColumns) {
    const Vec3 a{1.0, 2.0, 3.0};
    const Vec3 b{4.0, 5.0, 6.0};
    const Vec3 c{7.0, 8.0, 9.0};

    EXPECT_EQ(Mat3::fromRows(a, b, c), transpose(Mat3::fromColumns(a, b, c)));
    EXPECT_EQ(Mat3::fromColumns(a, b, c)[0], a);
    EXPECT_EQ(Mat3::fromRows(a, b, c).row(0), a);
}

TEST(MathMatrix, DefaultConstructionIsZeroNotIdentity) {
    EXPECT_EQ(Mat2{}, Mat2::zero());
    EXPECT_EQ(Mat3{}, Mat3::zero());
    EXPECT_EQ(Mat4{}, Mat4::zero());
    EXPECT_NE(Mat4{}, Mat4::identity());
}

// --- Ring axioms ------------------------------------------------------------

TEST(MathMatrix, AdditionIsCommutativeAndAssociative) {
    EXPECT_EQ(kAwkward3 + kAwkwardOther3, kAwkwardOther3 + kAwkward3);
    EXPECT_MAT_APPROX((kAwkward3 + kAwkwardOther3) + Mat3::identity(),
                      kAwkward3 + (kAwkwardOther3 + Mat3::identity()));
    EXPECT_EQ(kAwkward3 + Mat3::zero(), kAwkward3);
    EXPECT_EQ(kAwkward3 - kAwkward3, Mat3::zero());
    EXPECT_EQ(-(-kAwkward3), kAwkward3);
}

TEST(MathMatrix, MultiplicationIsAssociativeAndDistributesOverAddition) {
    const Mat3& a = kAwkward3;
    const Mat3& b = kAwkwardOther3;
    const Mat3 c = Mat3::rotationZ(0.7);
    const double scale = 12.0 * 12.0 * 12.0;

    EXPECT_MAT_NEAR((a * b) * c, a * (b * c), zeroTolerance(scale));
    EXPECT_MAT_NEAR(a * (b + c), a * b + a * c, zeroTolerance(scale));
    EXPECT_MAT_NEAR((a + b) * c, a * c + b * c, zeroTolerance(scale));
}

TEST(MathMatrix, IdentityIsTheMultiplicativeIdentity) {
    EXPECT_EQ(kAwkward2 * Mat2::identity(), kAwkward2);
    EXPECT_EQ(Mat2::identity() * kAwkward2, kAwkward2);
    EXPECT_EQ(kAwkward3 * Mat3::identity(), kAwkward3);
    EXPECT_EQ(Mat3::identity() * kAwkward3, kAwkward3);
    EXPECT_EQ(kAwkward4 * Mat4::identity(), kAwkward4);
    EXPECT_EQ(Mat4::identity() * kAwkward4, kAwkward4);
}

TEST(MathMatrix, MultiplicationDoesNotCommute) {
    // Stated as a test because the whole composition convention depends on it:
    // a * b applies b first. A commutative implementation would be a
    // componentwise product wearing the wrong operator.
    EXPECT_NE(kAwkward3 * kAwkwardOther3, kAwkwardOther3 * kAwkward3);
}

TEST(MathMatrix, MatrixVectorProductIsTheColumnCombination) {
    const Mat3& m = kAwkward3;
    const Vec3 v{2.0, -3.0, 0.5};

    EXPECT_VEC_APPROX(m * v, m[0] * v.x + m[1] * v.y + m[2] * v.z);
    // And, equivalently, the row-by-row dot products.
    EXPECT_VEC_APPROX(m * v, (Vec3{dot(m.row(0), v), dot(m.row(1), v),
                                   dot(m.row(2), v)}));
    // Linear in the vector.
    const Vec3 w{1.0, 1.0, -2.0};
    EXPECT_VEC_APPROX(m * (v + w), m * v + m * w);
    EXPECT_VEC_APPROX(m * (v * 3.0), (m * v) * 3.0);
}

TEST(MathMatrix, CompositionAppliesTheRightHandFactorFirst) {
    const Mat3& a = kAwkward3;
    const Mat3& b = kAwkwardOther3;
    const Vec3 v{1.0, -2.0, 0.5};
    EXPECT_VEC_NEAR((a * b) * v, a * (b * v), zeroTolerance(400.0));
}

TEST(MathMatrix, CompoundAssignmentMatchesTheBinaryOperators) {
    Mat3 m = kAwkward3;
    EXPECT_EQ(&(m += kAwkwardOther3), &m);
    EXPECT_EQ(m, kAwkward3 + kAwkwardOther3);

    m = kAwkward3;
    m -= kAwkwardOther3;
    EXPECT_EQ(m, kAwkward3 - kAwkwardOther3);

    m = kAwkward3;
    m *= 2.0;
    EXPECT_EQ(m, kAwkward3 * 2.0);

    m = kAwkward3;
    m /= 2.0;
    EXPECT_EQ(m, kAwkward3 / 2.0);

    m = kAwkward3;
    m *= kAwkwardOther3;
    EXPECT_EQ(m, kAwkward3 * kAwkwardOther3) << "*= must right-multiply";
}

// --- Transpose --------------------------------------------------------------

TEST(MathMatrix, TransposeIsItsOwnInverseAndReversesProducts) {
    EXPECT_EQ(transpose(transpose(kAwkward3)), kAwkward3);
    EXPECT_EQ(transpose(transpose(kAwkward4)), kAwkward4);
    EXPECT_EQ(transpose(Mat3::identity()), Mat3::identity());

    EXPECT_MAT_NEAR(transpose(kAwkward3 * kAwkwardOther3),
                    transpose(kAwkwardOther3) * transpose(kAwkward3),
                    zeroTolerance(150.0));
    EXPECT_MAT_NEAR(transpose(kAwkward4 * kAwkwardOther4),
                    transpose(kAwkwardOther4) * transpose(kAwkward4),
                    zeroTolerance(200.0));

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_EQ(transpose(kAwkward3)(r, c), kAwkward3(c, r));
        }
    }
}

// --- Determinant ------------------------------------------------------------

TEST(MathMatrix, DeterminantIsMultiplicativeAndTransposeInvariant) {
    EXPECT_APPROX(determinant(Mat2::identity()), 1.0);
    EXPECT_APPROX(determinant(Mat3::identity()), 1.0);
    EXPECT_APPROX(determinant(Mat4::identity()), 1.0);

    EXPECT_NEAR(determinant(kAwkward2 * kAwkwardOther2),
                determinant(kAwkward2) * determinant(kAwkwardOther2),
                zeroTolerance(100.0));
    EXPECT_NEAR(determinant(kAwkward3 * kAwkwardOther3),
                determinant(kAwkward3) * determinant(kAwkwardOther3),
                zeroTolerance(5.0e3));
    EXPECT_NEAR(determinant(kAwkward4 * kAwkwardOther4),
                determinant(kAwkward4) * determinant(kAwkwardOther4),
                zeroTolerance(1.0e6));

    EXPECT_APPROX(determinant(transpose(kAwkward3)), determinant(kAwkward3));
    EXPECT_APPROX(determinant(transpose(kAwkward4)), determinant(kAwkward4));
}

TEST(MathMatrix, DeterminantOfAScaleIsTheProductOfItsFactors) {
    EXPECT_APPROX(determinant(Mat3::scale({2.0, 3.0, 4.0})), 24.0);
    EXPECT_APPROX(determinant(Mat2::scale({2.0, 3.0})), 6.0);
    // A Matrix4 scale leaves w alone, so the determinant is the 3D volume.
    EXPECT_APPROX(determinant(Mat4::scale({2.0, 3.0, 4.0})), 24.0);
}

TEST(MathMatrix, DeterminantVanishesForARepeatedColumn) {
    const Vec3 a{1.0, 2.0, 3.0};
    const Vec3 b{4.0, 5.0, 6.0};
    EXPECT_NEAR(determinant(Mat3::fromColumns(a, b, a)), 0.0, zeroTolerance(100.0));
    // A linear combination of the others is just as singular.
    EXPECT_NEAR(determinant(Mat3::fromColumns(a, b, a * 2.0 - b)), 0.0,
                zeroTolerance(100.0));
}

// --- Inverse ----------------------------------------------------------------

TEST(MathMatrix, InverseUndoesTheMatrixFromBothSides) {
    EXPECT_MAT_NEAR(kAwkward2 * inverse(kAwkward2), Mat2::identity(),
                    zeroTolerance(10.0));
    EXPECT_MAT_NEAR(inverse(kAwkward2) * kAwkward2, Mat2::identity(),
                    zeroTolerance(10.0));

    EXPECT_MAT_NEAR(kAwkward3 * inverse(kAwkward3), Mat3::identity(),
                    zeroTolerance(50.0));
    EXPECT_MAT_NEAR(inverse(kAwkward3) * kAwkward3, Mat3::identity(),
                    zeroTolerance(50.0));

    EXPECT_MAT_NEAR(kAwkward4 * inverse(kAwkward4), Mat4::identity(),
                    zeroTolerance(200.0));
    EXPECT_MAT_NEAR(inverse(kAwkward4) * kAwkward4, Mat4::identity(),
                    zeroTolerance(200.0));
}

TEST(MathMatrix, InverseIsItsOwnInverseAndReversesProducts) {
    EXPECT_MAT_NEAR(inverse(inverse(kAwkward3)), kAwkward3, zeroTolerance(50.0));
    EXPECT_MAT_NEAR(inverse(kAwkward3 * kAwkwardOther3),
                    inverse(kAwkwardOther3) * inverse(kAwkward3),
                    zeroTolerance(50.0));
    EXPECT_APPROX(determinant(inverse(kAwkward3)), 1.0 / determinant(kAwkward3));
}

TEST(MathMatrix, TryInverseRejectsSingularAndNonFiniteMatrices) {
    const Vec3 a{1.0, 2.0, 3.0};
    const Vec3 b{4.0, 5.0, 6.0};

    EXPECT_FALSE(tryInverse(Mat3::fromColumns(a, b, a)).has_value());
    EXPECT_FALSE(tryInverse(Mat3::zero()).has_value());
    EXPECT_FALSE(tryInverse(Mat2::zero()).has_value());
    EXPECT_FALSE(tryInverse(Mat4::zero()).has_value());

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(tryInverse(Mat3::diagonal({nan, 1.0, 1.0})).has_value());

    ASSERT_TRUE(tryInverse(kAwkward3).has_value());
    EXPECT_MAT_NEAR(*tryInverse(kAwkward3), inverse(kAwkward3),
                    zeroTolerance(50.0));
}

TEST(MathMatrix, InversionRefusesAnOverflowingDeterminantRatherThanReturningZero) {
    // A 4x4 determinant is a product of four entries, so entries around 1e100
    // overflow it. The old guard tested for zero and for NaN, and an infinity
    // is neither: it went on to divide the adjugate by infinity, producing a
    // zero matrix reported as a success.
    const Mat4 huge = Mat4::diagonal({1e100, 1e100, 1e100, 1e100});
    EXPECT_FALSE(std::isfinite(determinant(huge)));
    EXPECT_FALSE(tryInverse(huge).has_value());

    const Mat3 huge3 = Mat3::diagonal({1e150, 1e150, 1e150});
    EXPECT_FALSE(tryInverse(huge3).has_value());

    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(tryInverse(Mat3::diagonal({inf, 1.0, 1.0})).has_value());
    EXPECT_FALSE(solve(Mat3::diagonal({inf, 1.0, 1.0}), Vec3{1.0, 1.0, 1.0})
                     .has_value());

    // Well inside the range it still inverts, so the guard is not simply
    // refusing anything large.
    const Mat4 large = Mat4::diagonal({1e60, 1e60, 1e60, 1e60});
    ASSERT_TRUE(tryInverse(large).has_value());
    EXPECT_MAT_NEAR(large * *tryInverse(large), Mat4::identity(), 1e-12);
}

TEST(MathMatrix, UncheckedInverseOfASingularMatrixIsNotANumber) {
    // Documented behaviour: it propagates rather than returning a plausible
    // wrong matrix. tryInverse is the checked spelling.
    const Mat3 singular = Mat3::fromColumns({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0},
                                            {1.0, 2.0, 3.0});
    EXPECT_FALSE(std::isfinite(inverse(singular)(0, 0)));
}

TEST(MathMatrix, AdjugateTimesTheMatrixIsTheDeterminantEvenWhenSingular) {
    // The adjugate identity holds for every matrix, which is what makes it the
    // right thing to build the inverse out of.
    const Mat4 singular =
        Mat4::fromColumns(kAwkward4[0], kAwkward4[1], kAwkward4[2], kAwkward4[0]);
    EXPECT_NEAR(determinant(singular), 0.0, zeroTolerance(1.0e3));
    EXPECT_MAT_NEAR(adjugate(singular) * singular, Mat4::zero(),
                    zeroTolerance(1.0e4));

    EXPECT_MAT_NEAR(adjugate(kAwkward4) * kAwkward4,
                    Mat4::identity() * determinant(kAwkward4),
                    zeroTolerance(1.0e4));
}

TEST(MathMatrix, SolveAgreesWithTheCofactorInverse) {
    // Two entirely separate implementations: Gauss-Jordan with partial
    // pivoting against the adjugate. If a cofactor sign or index is wrong,
    // this is where it shows, because nothing is shared between them.
    const Vec3 b3{1.0, -2.0, 3.5};
    const auto x3 = solve(kAwkward3, b3);
    ASSERT_TRUE(x3.has_value());
    EXPECT_VEC_NEAR(*x3, inverse(kAwkward3) * b3, zeroTolerance(10.0));
    EXPECT_VEC_NEAR(kAwkward3 * *x3, b3, zeroTolerance(10.0));

    const Vec4 b4{1.0, -2.0, 3.5, 0.25};
    const auto x4 = solve(kAwkward4, b4);
    ASSERT_TRUE(x4.has_value());
    EXPECT_VEC_NEAR(*x4, inverse(kAwkward4) * b4, zeroTolerance(50.0));
    EXPECT_VEC_NEAR(kAwkward4 * *x4, b4, zeroTolerance(50.0));

    const Vec2 b2{1.0, -2.0};
    const auto x2 = solve(kAwkward2, b2);
    ASSERT_TRUE(x2.has_value());
    EXPECT_VEC_NEAR(*x2, inverse(kAwkward2) * b2, zeroTolerance(10.0));
}

TEST(MathMatrix, SolvePivotsPastAZeroLeadingEntry) {
    // Without partial pivoting this divides by zero on the first column.
    const Mat3 needsPivoting =
        Mat3::fromRows({0.0, 2.0, 1.0}, {1.0, 0.0, 3.0}, {4.0, 5.0, 0.0});
    const Vec3 b{3.0, 4.0, 9.0};

    const auto x = solve(needsPivoting, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_VEC_NEAR(needsPivoting * *x, b, zeroTolerance(20.0));
}

TEST(MathMatrix, SolveReportsSingularRatherThanReturningNonsense) {
    const Mat3 singular = Mat3::fromColumns({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0},
                                            {1.0, 2.0, 3.0});
    EXPECT_FALSE(solve(singular, Vec3{1.0, 1.0, 1.0}).has_value());
    EXPECT_FALSE(solve(Mat4::zero(), Vec4{1.0, 1.0, 1.0, 1.0}).has_value());
}

// --- Similarity invariants --------------------------------------------------

TEST(MathMatrix, TraceAndDeterminantSurviveAChangeOfBasis) {
    // Trace and determinant are properties of the linear map, not of the
    // basis it happens to be written in.
    const Mat3& a = kAwkward3;
    const Mat3& p = kAwkwardOther3;
    const Mat3 similar = inverse(p) * a * p;

    EXPECT_NEAR(trace(similar), trace(a), zeroTolerance(500.0));
    EXPECT_NEAR(determinant(similar), determinant(a), zeroTolerance(5.0e3));
}

TEST(MathMatrix, TraceIsLinearAndCyclic) {
    EXPECT_APPROX(trace(Mat3::identity()), 3.0);
    EXPECT_APPROX(trace(Mat4::identity()), 4.0);
    EXPECT_APPROX(trace(kAwkward3 + kAwkwardOther3),
                  trace(kAwkward3) + trace(kAwkwardOther3));
    EXPECT_NEAR(trace(kAwkward3 * kAwkwardOther3),
                trace(kAwkwardOther3 * kAwkward3), zeroTolerance(200.0));
}

// --- Rotations --------------------------------------------------------------

TEST(MathMatrix, RotationMatricesAreOrthogonalWithUnitDeterminant) {
    const Vec3 axis = normalized(Vec3{1.0, -2.0, 0.5});

    for (const double angle : {0.0, 0.3, 1.0, 2.5, ysq::kPi<double>, 4.0}) {
        const Mat3 r = Mat3::rotation(axis, angle);
        EXPECT_MAT_NEAR(transpose(r) * r, Mat3::identity(), zeroTolerance(4.0));
        EXPECT_NEAR(determinant(r), 1.0, zeroTolerance(4.0))
            << "a determinant of -1 would be a reflection, not a rotation";
        EXPECT_MAT_NEAR(inverseOrthogonal(r), inverse(r), zeroTolerance(4.0));
    }
}

TEST(MathMatrix, RotationMatchesRodriguesAppliedDirectly) {
    // Matrix3::rotation and Vector3::rotateAbout are separate derivations of
    // the same formula. Cross-checking them catches a transposed matrix, which
    // orthogonality alone would happily accept.
    const Vec3 axis = normalized(Vec3{1.0, -2.0, 0.5});
    const std::array<Vec3, 3> vectors{Vec3{1.0, 0.0, 0.0}, Vec3{2.0, -3.0, 0.5},
                                      Vec3{0.0, 0.0, 4.0}};

    for (const double angle : {0.3, 1.0, 2.5, -1.2}) {
        const Mat3 r = Mat3::rotation(axis, angle);
        for (const Vec3& v : vectors) {
            EXPECT_VEC_NEAR(r * v, rotateAbout(v, axis, angle), zeroTolerance(10.0))
                << "at angle " << angle;
        }
    }
}

TEST(MathMatrix, AxisRotationsAgreeWithTheGeneralForm) {
    for (const double angle : {0.3, 1.0, -2.5}) {
        EXPECT_MAT_NEAR(Mat3::rotationX(angle), Mat3::rotation(Vec3::unitX(), angle),
                        zeroTolerance(4.0));
        EXPECT_MAT_NEAR(Mat3::rotationY(angle), Mat3::rotation(Vec3::unitY(), angle),
                        zeroTolerance(4.0));
        EXPECT_MAT_NEAR(Mat3::rotationZ(angle), Mat3::rotation(Vec3::unitZ(), angle),
                        zeroTolerance(4.0));
    }

    // A quarter turn about Z takes +X to +Y, right-handed.
    EXPECT_VEC_NEAR(Mat3::rotationZ(ysq::kPi<double> / 2.0) * Vec3::unitX(),
                    Vec3::unitY(), zeroTolerance(4.0));
    EXPECT_VEC_NEAR(Mat3::rotationX(ysq::kPi<double> / 2.0) * Vec3::unitY(),
                    Vec3::unitZ(), zeroTolerance(4.0));
    EXPECT_VEC_NEAR(Mat3::rotationY(ysq::kPi<double> / 2.0) * Vec3::unitZ(),
                    Vec3::unitX(), zeroTolerance(4.0));
}

TEST(MathMatrix, CrossMatrixReproducesTheCrossProduct) {
    const Vec3 a{1.0, -2.0, 0.5};
    const std::array<Vec3, 3> vectors{Vec3{1.0, 0.0, 0.0}, Vec3{2.0, -3.0, 0.5},
                                      Vec3{-1.0, 1.0, 4.0}};

    for (const Vec3& v : vectors) {
        EXPECT_VEC_APPROX(Mat3::crossMatrix(a) * v, cross(a, v));
    }
    // Skew-symmetric, so its trace is zero and its transpose is its negation.
    EXPECT_EQ(transpose(Mat3::crossMatrix(a)), -Mat3::crossMatrix(a));
    EXPECT_APPROX(trace(Mat3::crossMatrix(a)), 0.0);
}

TEST(MathMatrix, OuterProductIsRankOne) {
    const Vec3 a{1.0, -2.0, 0.5};
    const Vec3 b{3.0, 1.0, -1.0};
    const Mat3 m = Mat3::outerProduct(a, b);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_APPROX(m(r, c), a[r] * b[c]);
        }
    }
    EXPECT_NEAR(determinant(m), 0.0, zeroTolerance(10.0)) << "rank one is singular";
    EXPECT_APPROX(trace(m), dot(a, b));
    // Applying it projects onto a, scaled by the overlap with b.
    EXPECT_VEC_APPROX(m * b, a * dot(b, b));
}

// --- Affine transforms ------------------------------------------------------

TEST(MathMatrix, TransformPointTranslatesAndTransformDirectionDoesNot) {
    const Vec3 offset{5.0, -2.0, 1.0};
    const Mat4 t = Mat4::translation(offset);
    const Vec3 v{1.0, 2.0, 3.0};

    EXPECT_VEC_APPROX(transformPoint(t, v), v + offset);
    EXPECT_VEC_APPROX(transformDirection(t, v), v)
        << "a direction has no position, so translation must not touch it";
}

TEST(MathMatrix, AffineTransformsComposeRightToLeft) {
    const Vec3 offset{5.0, -2.0, 1.0};
    const double angle = 0.7;
    const Mat4 t = Mat4::translation(offset);
    const Mat4 r = Mat4::rotationZ(angle);
    const Vec3 v{1.0, 2.0, 3.0};

    // T * R rotates first, then translates.
    EXPECT_VEC_NEAR(transformPoint(t * r, v),
                    rotateAbout(v, Vec3::unitZ(), angle) + offset,
                    zeroTolerance(10.0));
    // R * T translates first, then rotates the result.
    EXPECT_VEC_NEAR(transformPoint(r * t, v),
                    rotateAbout(v + offset, Vec3::unitZ(), angle),
                    zeroTolerance(10.0));
}

TEST(MathMatrix, InverseAffineMatchesTheGeneralInverse) {
    const Mat4 m = Mat4::translation({5.0, -2.0, 1.0}) *
                   Mat4::rotation(normalized(Vec3{1.0, 2.0, -1.0}), 0.9) *
                   Mat4::scale({2.0, 0.5, 3.0});

    EXPECT_MAT_NEAR(inverseAffine(m), inverse(m), zeroTolerance(50.0));
    EXPECT_MAT_NEAR(m * inverseAffine(m), Mat4::identity(), zeroTolerance(50.0));

    const Vec3 v{1.0, 2.0, 3.0};
    EXPECT_VEC_NEAR(transformPoint(inverseAffine(m), transformPoint(m, v)), v,
                    zeroTolerance(50.0));
}

TEST(MathMatrix, LinearPartAndTranslationRoundTrip) {
    const Mat3 linear = Mat3::rotation(Vec3::unitZ(), 0.4) *
                        Mat3::scale({2.0, 3.0, 0.5});
    const Vec3 offset{5.0, -2.0, 1.0};
    const Mat4 m = Mat4::fromLinearTranslation(linear, offset);

    EXPECT_MAT_APPROX(m.upperLeft3x3(), linear);
    EXPECT_VEC_APPROX(m.translationPart(), offset);
    EXPECT_EQ(m.row(3), (Vec4{0.0, 0.0, 0.0, 1.0}));
    EXPECT_MAT_APPROX(Mat4::fromLinear(linear).upperLeft3x3(), linear);
    EXPECT_VEC_APPROX(Mat4::fromLinear(linear).translationPart(), Vec3::zero());
}

// --- Projections ------------------------------------------------------------

TEST(MathMatrix, LookAtPutsTheEyeAtTheOriginLookingDownNegativeZ) {
    const Vec3 eye{3.0, 4.0, 5.0};
    const Vec3 center{0.0, 1.0, -2.0};
    const Mat4 view = Mat4::lookAt(eye, center, Vec3::unitY());

    EXPECT_VEC_NEAR(transformPoint(view, eye), Vec3::zero(), zeroTolerance(20.0));
    EXPECT_VEC_NEAR(transformDirection(view, normalized(center - eye)),
                    -Vec3::unitZ(), zeroTolerance(10.0));
    // The target ends up straight ahead: on the -Z axis.
    const Vec3 target = transformPoint(view, center);
    EXPECT_NEAR(target.x, 0.0, zeroTolerance(20.0));
    EXPECT_NEAR(target.y, 0.0, zeroTolerance(20.0));
    EXPECT_LT(target.z, 0.0);
    // A view matrix is rigid, so it preserves distances.
    EXPECT_APPROX(distance(transformPoint(view, eye), transformPoint(view, center)),
                  distance(eye, center));
}

TEST(MathMatrix, PerspectiveMapsTheFrustumOntoOpenGlClipSpace) {
    constexpr double nearPlane = 0.5;
    constexpr double farPlane = 100.0;
    constexpr double fovY = ysq::kPi<double> / 3.0;  // 60 degrees
    constexpr double aspect = 16.0 / 9.0;
    const Mat4 p = Mat4::perspective(fovY, aspect, nearPlane, farPlane);

    // OpenGL depth runs -1 at the near plane to +1 at the far plane.
    EXPECT_NEAR(projectPoint(p, Vec3{0.0, 0.0, -nearPlane}).z, -1.0,
                zeroTolerance(10.0));
    EXPECT_NEAR(projectPoint(p, Vec3{0.0, 0.0, -farPlane}).z, 1.0,
                zeroTolerance(10.0));

    // The top edge of the frustum at any depth lands on y = 1.
    const double depth = 4.0;
    const double topAt = depth * std::tan(fovY / 2.0);
    EXPECT_NEAR(projectPoint(p, Vec3{0.0, topAt, -depth}).y, 1.0,
                zeroTolerance(10.0));
    // And the right edge, which is the top scaled by the aspect ratio.
    EXPECT_NEAR(projectPoint(p, Vec3{topAt * aspect, 0.0, -depth}).x, 1.0,
                zeroTolerance(10.0));

    // Not affine: the last row is projective, so inverseAffine would be wrong.
    EXPECT_EQ(p.row(3), (Vec4{0.0, 0.0, -1.0, 0.0}));
}

TEST(MathMatrix, OrthographicMapsTheBoxOntoTheClipCube) {
    constexpr double left = -3.0;
    constexpr double right = 5.0;
    constexpr double bottom = -2.0;
    constexpr double top = 6.0;
    constexpr double nearPlane = 1.0;
    constexpr double farPlane = 20.0;
    const Mat4 p =
        Mat4::orthographic(left, right, bottom, top, nearPlane, farPlane);

    EXPECT_VEC_NEAR(projectPoint(p, Vec3{left, bottom, -nearPlane}),
                    (Vec3{-1.0, -1.0, -1.0}), zeroTolerance(20.0));
    EXPECT_VEC_NEAR(projectPoint(p, Vec3{right, top, -farPlane}),
                    (Vec3{1.0, 1.0, 1.0}), zeroTolerance(20.0));
    // Affine, unlike perspective.
    EXPECT_EQ(p.row(3), (Vec4{0.0, 0.0, 0.0, 1.0}));
}

// --- Formatting -------------------------------------------------------------

TEST(MathMatrix, FormattingPrintsByRowsNotByStorageOrder) {
    const Mat2 m = Mat2::fromRows({1.0, 2.0}, {3.0, 4.0});
    EXPECT_EQ(std::format("{}", m), "[[1, 2], [3, 4]]")
        << "storage is column-major, but a printed matrix has to read the way "
           "it is written";
    EXPECT_EQ(std::format("{:.1f}", Mat2::identity()),
              "[[1.0, 0.0], [0.0, 1.0]]");
}

// --- Single precision -------------------------------------------------------

TEST(MathMatrix, IdentitiesHoldAtSinglePrecision) {
    using ysq::Mat3f;
    using ysq::Vec3f;
    constexpr float tol = 1e-5f;

    const Mat3f a = Mat3f::fromRows({2.0f, -1.0f, 3.0f}, {0.5f, 4.0f, -2.0f},
                                    {-1.5f, 1.0f, 5.0f});
    EXPECT_MAT_NEAR(a * inverse(a), Mat3f::identity(), tol);
    EXPECT_MAT_NEAR(transpose(transpose(a)), a, tol);

    const Mat3f r = Mat3f::rotationZ(0.7f);
    EXPECT_MAT_NEAR(transpose(r) * r, Mat3f::identity(), tol);
    EXPECT_NEAR(determinant(r), 1.0f, tol);
}

}  // namespace
