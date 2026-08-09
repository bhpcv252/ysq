#include <Math/Eigen.hpp>

#include <Math/Complex.hpp>
#include <Math/LinearSolve.hpp>
#include <Math/Scalar.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ysq::Complex;
using ysq::MatrixN;
using ysq::VectorN;

}  // namespace

TEST(MathEigen, DiagonalMatrixReturnsItsOwnEntriesAsEigenvaluesInAscendingOrder) {
    MatrixN<double> a(3, 3);
    a(0, 0) = 3.0;
    a(1, 1) = 1.0;
    a(2, 2) = 2.0;

    const auto decomposition = ysq::jacobiEigenSymmetric(a);

    EXPECT_NEAR(decomposition.eigenvalues[0], 1.0, 1e-10);
    EXPECT_NEAR(decomposition.eigenvalues[1], 2.0, 1e-10);
    EXPECT_NEAR(decomposition.eigenvalues[2], 3.0, 1e-10);
}

TEST(MathEigen, EigenvectorsAreOrthonormal) {
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

    const auto decomposition = ysq::jacobiEigenSymmetric(a);
    const MatrixN<double>& v = decomposition.eigenvectors;

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                dot += v(k, i) * v(k, j);
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-10);
        }
    }
}

TEST(MathEigen, ReconstructedMatrixMatchesTheOriginal) {
    // A = V diag(lambda) V^T must reproduce the original symmetric matrix,
    // the defining property of an eigendecomposition rather than merely a
    // property of this particular algorithm.
    MatrixN<double> a(3, 3);
    a(0, 0) = 2.0;
    a(0, 1) = -1.0;
    a(0, 2) = 0.0;
    a(1, 0) = -1.0;
    a(1, 1) = 2.0;
    a(1, 2) = -1.0;
    a(2, 0) = 0.0;
    a(2, 1) = -1.0;
    a(2, 2) = 2.0;

    const auto decomposition = ysq::jacobiEigenSymmetric(a);
    const MatrixN<double>& v = decomposition.eigenvectors;
    const VectorN<double>& lambda = decomposition.eigenvalues;

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double reconstructed = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                reconstructed += v(i, k) * lambda[k] * v(j, k);
            }
            EXPECT_NEAR(reconstructed, a(i, j), 1e-9);
        }
    }
}

TEST(MathEigen, KnownTridiagonalMatrixMatchesItsClosedFormEigenvalues) {
    // The standard [[2,-1],[-1,2]] tridiagonal-family matrix has eigenvalues
    // 2 - 2cos(k*pi/(n+1)) for k = 1..n; for n = 2 that is 1 and 3.
    MatrixN<double> a(2, 2);
    a(0, 0) = 2.0;
    a(0, 1) = -1.0;
    a(1, 0) = -1.0;
    a(1, 1) = 2.0;

    const auto decomposition = ysq::jacobiEigenSymmetric(a);

    EXPECT_NEAR(decomposition.eigenvalues[0], 1.0, 1e-10);
    EXPECT_NEAR(decomposition.eigenvalues[1], 3.0, 1e-10);
}

TEST(MathEigen, AlreadyDiagonalTwoByTwoConvergesImmediately) {
    MatrixN<double> a(2, 2);
    a(0, 0) = 5.0;
    a(1, 1) = -2.0;

    const auto decomposition = ysq::jacobiEigenSymmetric(a);

    EXPECT_NEAR(decomposition.eigenvalues[0], -2.0, 1e-12);
    EXPECT_NEAR(decomposition.eigenvalues[1], 5.0, 1e-12);
}

TEST(MathEigen, QrDecomposeReconstructsASquareMatrix) {
    // The worked example from Golub & Van Loan / the Wikipedia QR page.
    MatrixN<double> a(3, 3);
    a(0, 0) = 12.0;
    a(0, 1) = -51.0;
    a(0, 2) = 4.0;
    a(1, 0) = 6.0;
    a(1, 1) = 167.0;
    a(1, 2) = -68.0;
    a(2, 0) = -4.0;
    a(2, 1) = 24.0;
    a(2, 2) = -41.0;

    const auto qr = ysq::qrDecompose(a);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                dot += qr.q(k, i) * qr.q(k, j);
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-9);
        }
    }

    for (std::size_t i = 1; i < 3; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            EXPECT_NEAR(qr.r(i, j), 0.0, 1e-9);
        }
    }

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double reconstructed = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                reconstructed += qr.q(i, k) * qr.r(k, j);
            }
            EXPECT_NEAR(reconstructed, a(i, j), 1e-9);
        }
    }
}

TEST(MathEigen, QrDecomposeHandlesATallNonSquareMatrix) {
    MatrixN<double> a(4, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 0.0;
    a(2, 0) = 0.0;
    a(2, 1) = 1.0;
    a(3, 0) = 1.0;
    a(3, 1) = 1.0;

    const auto qr = ysq::qrDecompose(a);
    ASSERT_EQ(qr.q.rows(), 4u);
    ASSERT_EQ(qr.q.cols(), 4u);
    ASSERT_EQ(qr.r.rows(), 4u);
    ASSERT_EQ(qr.r.cols(), 2u);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                dot += qr.q(k, i) * qr.q(k, j);
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-9);
        }
    }

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            double reconstructed = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                reconstructed += qr.q(i, k) * qr.r(k, j);
            }
            EXPECT_NEAR(reconstructed, a(i, j), 1e-9);
        }
    }
}

TEST(MathEigen, GeneralEigenvaluesOfACompanionMatrixMatchesItsKnownRoots) {
    // The companion matrix of (x+1)(x+2)(x+3) = x^3 + 6x^2 + 11x + 6, whose
    // eigenvalues are exactly that polynomial's roots by construction.
    MatrixN<double> a(3, 3);
    a(0, 1) = 1.0;
    a(1, 2) = 1.0;
    a(2, 0) = -6.0;
    a(2, 1) = -11.0;
    a(2, 2) = -6.0;

    const std::vector<Complex<double>> eigenvalues = ysq::generalEigenvalues(a);
    ASSERT_EQ(eigenvalues.size(), 3u);

    std::vector<double> realParts;
    for (const Complex<double>& z : eigenvalues) {
        EXPECT_NEAR(z.im, 0.0, 1e-7);
        realParts.push_back(z.re);
    }
    std::sort(realParts.begin(), realParts.end());
    EXPECT_NEAR(realParts[0], -3.0, 1e-7);
    EXPECT_NEAR(realParts[1], -2.0, 1e-7);
    EXPECT_NEAR(realParts[2], -1.0, 1e-7);
}

TEST(MathEigen, GeneralEigenvaluesOfARotationLikeMatrixIsAComplexConjugatePair) {
    // [[cos, -sin], [sin, cos]] scaled by radius has eigenvalues
    // radius * (cos(angle) +/- i sin(angle)) -- a rotation's own spectrum.
    const double angle = ysq::kPi<double> / 3.0;
    const double radius = 2.0;
    MatrixN<double> a(2, 2);
    a(0, 0) = radius * std::cos(angle);
    a(0, 1) = -radius * std::sin(angle);
    a(1, 0) = radius * std::sin(angle);
    a(1, 1) = radius * std::cos(angle);

    const std::vector<Complex<double>> eigenvalues = ysq::generalEigenvalues(a);
    ASSERT_EQ(eigenvalues.size(), 2u);

    EXPECT_NEAR(eigenvalues[0].re, radius * std::cos(angle), 1e-9);
    EXPECT_NEAR(eigenvalues[1].re, radius * std::cos(angle), 1e-9);
    EXPECT_NEAR(std::abs(eigenvalues[0].im), radius * std::sin(angle), 1e-9);
    EXPECT_NEAR(eigenvalues[0].im, -eigenvalues[1].im, 1e-12);
}

TEST(MathEigen, GeneralEigenvaluesAgreesWithJacobiEigenSymmetricOnASymmetricMatrix) {
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

    const auto symmetric = ysq::jacobiEigenSymmetric(a);
    const std::vector<Complex<double>> general = ysq::generalEigenvalues(a);
    ASSERT_EQ(general.size(), 3u);

    std::vector<double> generalReal;
    for (const Complex<double>& z : general) {
        EXPECT_NEAR(z.im, 0.0, 1e-7);
        generalReal.push_back(z.re);
    }
    std::sort(generalReal.begin(), generalReal.end());
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(generalReal[i], symmetric.eigenvalues[i], 1e-7);
    }
}

TEST(MathEigen, SvdReconstructsAMatrix) {
    MatrixN<double> a(3, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 0.0;
    a(1, 0) = 0.0;
    a(1, 1) = 1.0;
    a(2, 0) = 1.0;
    a(2, 1) = 1.0;

    const auto result = ysq::svd(a);

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                dot += result.u(k, i) * result.u(k, j);
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-9);

            double vDot = 0.0;
            for (std::size_t k = 0; k < 2; ++k) {
                vDot += result.v(k, i) * result.v(k, j);
            }
            EXPECT_NEAR(vDot, (i == j) ? 1.0 : 0.0, 1e-9);
        }
    }

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            double reconstructed = 0.0;
            for (std::size_t k = 0; k < 2; ++k) {
                reconstructed +=
                    result.u(i, k) * result.singularValues[k] * result.v(j, k);
            }
            EXPECT_NEAR(reconstructed, a(i, j), 1e-9);
        }
    }

    EXPECT_GE(result.singularValues[0], result.singularValues[1]);
}

TEST(MathEigen, SvdSingularValuesMatchJacobiEigenSymmetricOfATransposeA) {
    // sigma_i = sqrt(lambda_i) is the defining relationship between the SVD
    // of a and the eigendecomposition of the Gram matrix a^T a.
    MatrixN<double> a(3, 2);
    a(0, 0) = 2.0;
    a(0, 1) = 0.0;
    a(1, 0) = 0.0;
    a(1, 1) = 3.0;
    a(2, 0) = 1.0;
    a(2, 1) = 1.0;

    const auto result = ysq::svd(a);

    const MatrixN<double> ata = ysq::transpose(a) * a;
    const auto symmetric = ysq::jacobiEigenSymmetric(ata);

    // singularValues is descending; jacobiEigenSymmetric's eigenvalues are
    // ascending, so they pair up in reverse order.
    EXPECT_NEAR(result.singularValues[0] * result.singularValues[0],
                symmetric.eigenvalues[1], 1e-8);
    EXPECT_NEAR(result.singularValues[1] * result.singularValues[1],
                symmetric.eigenvalues[0], 1e-8);
}
