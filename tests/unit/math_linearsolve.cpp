#include <Math/LinearSolve.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::MatrixN;
using ysq::VectorN;

}  // namespace

TEST(MathLinearSolve, VectorArithmeticMatchesComponentwiseExpectation) {
    const VectorN<double> a{1.0, 2.0, 3.0};
    const VectorN<double> b{4.0, 5.0, 6.0};

    const VectorN<double> sum = a + b;
    EXPECT_DOUBLE_EQ(sum[0], 5.0);
    EXPECT_DOUBLE_EQ(sum[1], 7.0);
    EXPECT_DOUBLE_EQ(sum[2], 9.0);

    EXPECT_DOUBLE_EQ(ysq::dot(a, b), 32.0);
    EXPECT_DOUBLE_EQ(ysq::norm(VectorN<double>{3.0, 4.0}), 5.0);
}

TEST(MathLinearSolve, MatrixVectorMultiplyMatchesHandComputedResult) {
    MatrixN<double> m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;

    const VectorN<double> v{5.0, 6.0};
    const VectorN<double> result = m * v;

    EXPECT_DOUBLE_EQ(result[0], 17.0);
    EXPECT_DOUBLE_EQ(result[1], 39.0);
}

TEST(MathLinearSolve, IdentityMatrixLeavesAVectorUnchanged) {
    const VectorN<double> v{1.0, 2.0, 3.0};
    const VectorN<double> result = MatrixN<double>::identity(3) * v;

    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST(MathLinearSolve, SolveRecoversTheKnownSolutionOfAWellConditionedSystem) {
    // 2x + y = 5, x + 3y = 10 -> x = 1, y = 3.
    MatrixN<double> a(2, 2);
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 3.0;
    const VectorN<double> b{5.0, 10.0};

    const std::optional<VectorN<double>> x = ysq::solve(a, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)[0], 1.0, 1e-12);
    EXPECT_NEAR((*x)[1], 3.0, 1e-12);
}

TEST(MathLinearSolve, SolveRequiresPivotingToHandleAZeroLeadingDiagonal) {
    // Without partial pivoting this would divide by the zero at (0, 0).
    MatrixN<double> a(2, 2);
    a(0, 0) = 0.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 1.0;
    const VectorN<double> b{2.0, 3.0};

    const std::optional<VectorN<double>> x = ysq::solve(a, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)[0], 1.0, 1e-12);
    EXPECT_NEAR((*x)[1], 2.0, 1e-12);
}

TEST(MathLinearSolve, SolveReturnsNulloptForASingularMatrix) {
    MatrixN<double> a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 4.0;
    const VectorN<double> b{1.0, 2.0};

    EXPECT_FALSE(ysq::solve(a, b).has_value());
}

TEST(MathLinearSolve, LuDecomposeThenLuSolveMatchesTheOneShotSolve) {
    MatrixN<double> a(3, 3);
    a(0, 0) = 4.0;
    a(0, 1) = 3.0;
    a(0, 2) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 6.0;
    a(1, 2) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 1.0;
    a(2, 2) = 5.0;
    const VectorN<double> b{1.0, 2.0, 3.0};

    const auto decomposition = ysq::luDecompose(a);
    ASSERT_TRUE(decomposition.has_value());
    const VectorN<double> fromLu = ysq::luSolve(*decomposition, b);

    const std::optional<VectorN<double>> fromSolve = ysq::solve(a, b);
    ASSERT_TRUE(fromSolve.has_value());

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(fromLu[i], (*fromSolve)[i], 1e-10);
    }

    // Residual check independent of both: a fromLu is actually b.
    const VectorN<double> residual = a * fromLu - b;
    EXPECT_NEAR(ysq::norm(residual), 0.0, 1e-9);
}

TEST(MathLinearSolve, CholeskySolveMatchesLuSolveForASymmetricPositiveDefiniteSystem) {
    MatrixN<double> a(3, 3);
    a(0, 0) = 4.0;
    a(0, 1) = 1.0;
    a(0, 2) = 0.0;
    a(1, 0) = 1.0;
    a(1, 1) = 3.0;
    a(1, 2) = 1.0;
    a(2, 0) = 0.0;
    a(2, 1) = 1.0;
    a(2, 2) = 2.0;
    const VectorN<double> b{1.0, 2.0, 3.0};

    const std::optional<VectorN<double>> viaCholesky = ysq::choleskySolve(a, b);
    const std::optional<VectorN<double>> viaLu = ysq::solve(a, b);
    ASSERT_TRUE(viaCholesky.has_value());
    ASSERT_TRUE(viaLu.has_value());

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((*viaCholesky)[i], (*viaLu)[i], 1e-10);
    }
}

TEST(MathLinearSolve, CholeskyDecomposeReturnsNulloptForANonPositiveDefiniteMatrix) {
    MatrixN<double> a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 2.0;
    a(1, 1) = 1.0;  // Symmetric, but eigenvalues are 3 and -1: not PD.

    EXPECT_FALSE(ysq::choleskyDecompose(a).has_value());
}
