#include <Math/LinearSolve.hpp>

#include <Compute/CPU/CpuBackend.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

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

TEST(MathLinearSolve, MatVecAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // Above LinearSolve.hpp's own GPU dispatch threshold and MatrixN<float>
    // (the only type that ever dispatches; see detail::matVecGpu), so
    // operator* exercises whichever compute backend is actually available.
    // ysq::CpuBackend is called directly as an independent reference (its
    // own agreement with every GPU backend is already covered by
    // tests/integration/compute_backends_agree.cpp; this test only checks
    // that operator*'s marshaling is wired correctly).
    // 2048*2048 = 4194304, exactly at kMatVecGpuDispatchThreshold (measured
    // by benchmarks/compute_thresholds.cpp; see LinearSolve.hpp's own
    // comment on that constant).
    constexpr std::size_t rows = 2048, cols = 2048;
    MatrixN<float> m(rows, cols);
    VectorN<float> v(cols);
    std::vector<float> matrixData(rows * cols);
    std::vector<float> vectorData(cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const float value =
                static_cast<float>(r) * 0.01f - static_cast<float>(c) * 0.02f;
            m(r, c) = value;
            matrixData[r * cols + c] = value;
        }
    }
    for (std::size_t c = 0; c < cols; ++c) {
        const float value = static_cast<float>(c) * 0.1f;
        v[c] = value;
        vectorData[c] = value;
    }

    const VectorN<float> result = m * v;

    const ysq::CpuBackend cpu;
    std::vector<float> resultReference(rows);
    cpu.matVec(matrixData, rows, cols, vectorData, resultReference);

    for (const std::size_t r : {std::size_t{0}, rows / 2, rows - 1}) {
        // A relative tolerance, not the fixed 1e-2f a smaller N could use:
        // at 2048 accumulated terms the result magnitude reaches the
        // millions, where float32's ~7 significant digits alone account for
        // more absolute error than a fixed tolerance sized for an O(1)
        // result would allow, the same reasoning
        // BatchGammaAgreesWithTheCpuReferenceOnEveryAvailableBackend (in
        // tests/integration/compute_backends_agree.cpp) uses.
        const float tolerance = std::max(1e-2f, std::abs(resultReference[r]) * 1e-4f);
        EXPECT_NEAR(result[r], resultReference[r], tolerance) << "row " << r;
    }
}

TEST(MathLinearSolve, MatMulAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // 129^3 = 2146689, above kMatMulGpuDispatchThreshold (2097152; measured
    // by benchmarks/compute_thresholds.cpp).
    constexpr std::size_t aRows = 129, aCols = 129, bCols = 129;
    MatrixN<float> a(aRows, aCols);
    MatrixN<float> b(aCols, bCols);
    std::vector<float> aData(aRows * aCols);
    std::vector<float> bData(aCols * bCols);
    for (std::size_t r = 0; r < aRows; ++r) {
        for (std::size_t c = 0; c < aCols; ++c) {
            const float value =
                static_cast<float>(r) * 0.03f - static_cast<float>(c) * 0.01f;
            a(r, c) = value;
            aData[r * aCols + c] = value;
        }
    }
    for (std::size_t r = 0; r < aCols; ++r) {
        for (std::size_t c = 0; c < bCols; ++c) {
            const float value =
                static_cast<float>(c) * 0.02f - static_cast<float>(r) * 0.01f;
            b(r, c) = value;
            bData[r * bCols + c] = value;
        }
    }

    const MatrixN<float> result = a * b;

    const ysq::CpuBackend cpu;
    std::vector<float> resultReference(aRows * bCols);
    cpu.matMul(aData, aRows, aCols, bData, bCols, resultReference);

    for (const std::size_t r : {std::size_t{0}, aRows / 2, aRows - 1}) {
        for (const std::size_t c : {std::size_t{0}, bCols / 2, bCols - 1}) {
            EXPECT_NEAR(result(r, c), resultReference[r * bCols + c], 1e-2f)
                << "entry " << r << "," << c;
        }
    }
}

TEST(MathLinearSolve, LuDecomposeAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // 725^2 = 525625, above kLuCholeskyGpuDispatchThreshold (524288; measured
    // by benchmarks/compute_thresholds.cpp).
    constexpr std::size_t n = 725;
    MatrixN<float> a(n, n);
    std::vector<float> aData(n * n);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            const float value =
                static_cast<float>(r) * 0.01f - static_cast<float>(c) * 0.02f + 0.001f;
            a(r, c) = value;
            aData[r * n + c] = value;
        }
        a(r, r) = static_cast<float>(n) * 10.0f;  // diagonally dominant: non-singular
        aData[r * n + r] = static_cast<float>(n) * 10.0f;
    }

    const std::optional<ysq::LuDecomposition<float>> result = ysq::luDecompose(a);
    ASSERT_TRUE(result.has_value());

    const ysq::CpuBackend cpu;
    std::vector<float> luReference(n * n);
    std::vector<std::uint32_t> pivotReference(n);
    ASSERT_TRUE(cpu.luDecomposeGpu(aData, n, luReference, pivotReference));

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(result->pivot[i], static_cast<std::size_t>(pivotReference[i]))
            << "pivot " << i;
    }
    for (const std::size_t r : {std::size_t{0}, n / 2, n - 1}) {
        for (const std::size_t c : {std::size_t{0}, n / 2, n - 1}) {
            EXPECT_NEAR(result->lu(r, c), luReference[r * n + c], 1e-1f)
                << "entry " << r << "," << c;
        }
    }
}

TEST(MathLinearSolve,
     CholeskyDecomposeAtLargeNAgreesWithTheComputeCpuReferenceOnTheGpuPath) {
    // Shares kLuCholeskyGpuDispatchThreshold with LU above.
    constexpr std::size_t n = 725;
    // A Gram matrix A^T A (symmetric positive-semidefinite) plus a scaled
    // identity (strictly positive-definite).
    std::vector<float> base(n * n);
    for (std::size_t i = 0; i < n * n; ++i) {
        base[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
    }
    MatrixN<float> a(n, n);
    std::vector<float> aData(n * n, 0.0f);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            float total = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                total += base[k * n + r] * base[k * n + c];
            }
            if (r == c) {
                total += static_cast<float>(n) * 1.0e3f;
            }
            a(r, c) = total;
            if (c <= r) {
                aData[r * n + c] = total;
            }
        }
    }

    const std::optional<MatrixN<float>> result = ysq::choleskyDecompose(a);
    ASSERT_TRUE(result.has_value());

    const ysq::CpuBackend cpu;
    std::vector<float> lReference(n * n);
    ASSERT_TRUE(cpu.choleskyDecomposeGpu(aData, n, lReference));

    for (const std::size_t r : {std::size_t{0}, n / 2, n - 1}) {
        for (const std::size_t c : {std::size_t{0}, n / 2, n - 1}) {
            if (c > r) {
                continue;
            }
            EXPECT_NEAR((*result)(r, c), lReference[r * n + c], 1.0f)
                << "entry " << r << "," << c;
        }
    }
}
