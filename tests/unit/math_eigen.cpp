#include <Math/Eigen.hpp>

#include <Compute/CPU/CpuBackend.hpp>
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

TEST(MathEigen, QrDecomposeGpuDispatchMarshalsCorrectly) {
    // qrDecompose(a) only reaches detail::qrDecomposeGpuDispatch once
    // rows*cols crosses kEigenGpuDispatchThreshold (131072; measured by
    // benchmarks/compute_thresholds.cpp), and QR is a host loop of n
    // sequential GPU dispatches -- reaching that size through the public
    // API here would need a matrix north of 363x363, which measured in the
    // tens of seconds (per-dispatch CPU/GPU sync overhead across that many
    // round trips) for a check that is purely about marshaling, not
    // numerics. Calling detail::qrDecomposeGpuDispatch directly exercises
    // the identical marshaling code at a size actually worth running in a
    // unit test suite; tests/unit/multigrid.cpp's direct calls to
    // detail::restrictGrid/prolongateAndAdd are the same idea. Kernel
    // correctness itself is already covered, independently, by
    // tests/integration/compute_backends_agree.cpp.
    constexpr std::size_t rows = 12, cols = 8;
    MatrixN<float> a(rows, cols);
    std::vector<float> matrixData(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const float value =
                std::sin(0.7f * static_cast<float>(i)) - static_cast<float>(j) * 0.3f;
            a(i, j) = value;
            matrixData[i * cols + j] = value;
        }
    }

    const ysq::QrDecomposition<float> result = ysq::detail::qrDecomposeGpuDispatch(a);

    const ysq::CpuBackend cpu;
    std::vector<float> qReference(rows * rows);
    std::vector<float> rReference(rows * cols);
    cpu.qrDecomposeGpu(matrixData, rows, cols, qReference, rReference);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < rows; ++k) {
                total += result.q(i, k) * result.r(k, j);
            }
            EXPECT_NEAR(total, matrixData[i * cols + j], 1e-2f)
                << "QR entry " << i << "," << j;
        }
    }
}

TEST(MathEigen, JacobiEigenSymmetricGpuDispatchMarshalsCorrectly) {
    // Same reasoning as QrDecomposeGpuDispatchMarshalsCorrectly above:
    // calls detail::jacobiEigenSymmetricGpuDispatch directly rather than
    // needing a 363x363 matrix to cross the real threshold through the
    // public API.
    constexpr std::size_t n = 10;
    MatrixN<float> a(n, n);
    std::vector<float> matrixData(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const float value = std::sin(0.3f * static_cast<float>(i * n + j));
            a(i, j) = value;
            a(j, i) = value;
            matrixData[i * n + j] = value;
        }
    }

    const ysq::EigenDecomposition<float> result =
        ysq::detail::jacobiEigenSymmetricGpuDispatch(a, 100, 0.0f);

    const ysq::CpuBackend cpu;
    std::vector<float> diagonalReference(n * n);
    std::vector<float> eigenvectorsReference(n * n);
    cpu.jacobiEigenSymmetricGpu(matrixData, n, 100, 0.0f, diagonalReference,
                                eigenvectorsReference);
    std::vector<float> eigenvaluesReference(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvaluesReference[i] = diagonalReference[i * n + i];
    }
    std::sort(eigenvaluesReference.begin(), eigenvaluesReference.end());

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(result.eigenvalues[i], eigenvaluesReference[i], 1e-2f)
            << "eigenvalue " << i;
    }
}

TEST(MathEigen, SvdGpuDispatchMarshalsCorrectly) {
    // Same reasoning as QrDecomposeGpuDispatchMarshalsCorrectly above:
    // calls detail::svdGpuDispatch directly rather than needing a
    // 363x363 matrix to cross the real threshold through the public API.
    constexpr std::size_t rows = 12, cols = 8;
    MatrixN<float> a(rows, cols);
    std::vector<float> matrixData(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const float value = std::cos(0.4f * static_cast<float>(i + j));
            a(i, j) = value;
            matrixData[i * cols + j] = value;
        }
    }

    const ysq::SvdDecomposition<float> result = ysq::detail::svdGpuDispatch(a, 60, 0.0f);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < cols; ++k) {
                total += result.u(i, k) * result.singularValues[k] * result.v(j, k);
            }
            EXPECT_NEAR(total, matrixData[i * cols + j], 1e-2f)
                << "entry " << i << "," << j;
        }
    }
}

TEST(MathEigen, GeneralEigenvaluesAtLargeNStillWorksThroughTheTransitiveGpuPath) {
    // realSchur/generalEigenvalues have no bespoke GPU kernel of their own
    // (see src/Compute/README.md): their dominant cost is repeated internal
    // qrDecompose(block) calls, which get GPU-accelerated transitively once
    // qrDecompose itself does. This checks that composition still produces
    // correct results at a size where those internal calls actually cross
    // qrDecompose's own dispatch threshold at some point during shrinkage
    // (the leading block starts at n=363, above kEigenGpuDispatchThreshold's
    // 131072 elements, and shrinks every iteration, so later iterations
    // fall back to the CPU path within the very same generalEigenvalues
    // call).
    constexpr std::size_t n = 363;
    MatrixN<float> a(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const float value = std::sin(0.05f * static_cast<float>(i * n + j));
            a(i, j) = value;
            a(j, i) = value;
        }
    }

    const std::vector<Complex<float>> eigenvalues = ysq::generalEigenvalues(a, 500);
    const ysq::EigenDecomposition<float> symmetric =
        ysq::jacobiEigenSymmetric(a, 100, 0.0f);

    ASSERT_EQ(eigenvalues.size(), n);
    std::vector<float> realParts(n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(eigenvalues[i].im, 0.0f, 1e-1f)
            << "eigenvalue " << i << " imaginary part";
        realParts[i] = eigenvalues[i].re;
    }
    std::sort(realParts.begin(), realParts.end());
    for (const std::size_t i : {std::size_t{0}, n / 2, n - 1}) {
        EXPECT_NEAR(realParts[i], symmetric.eigenvalues[i], 1.0f) << "eigenvalue " << i;
    }
}
