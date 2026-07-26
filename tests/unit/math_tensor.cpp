#include <Math/Tensor.hpp>

#include <Math/Dual.hpp>
#include <Math/Format.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Matrix4.hpp>
#include <Math/Scalar.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <type_traits>

namespace {

using ysq::Mat3;
using ysq::Mat4;
using ysq::Vec3;
using ysq::Vec4;

using Rank0 = ysq::Tensor<double, 0, 4>;
using Rank1 = ysq::Tensor<double, 1, 4>;
using Rank2 = ysq::Tensor<double, 2, 4>;
using Rank3 = ysq::Tensor<double, 3, 4>;
using Rank4 = ysq::Tensor<double, 4, 4>;

constexpr double kEps = std::numeric_limits<double>::epsilon();

double zeroTolerance(double scale) {
    return 64.0 * kEps * scale;
}

/// Values that depend on every index differently, so a transposed or
/// mis-strided implementation cannot coincide with the right answer.
Rank2 distinctRank2() {
    Rank2 t{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            t(i, j) = 1.0 + static_cast<double>(i) + 10.0 * static_cast<double>(j);
        }
    }
    return t;
}

Rank4 distinctRank4() {
    Rank4 t{};
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            for (std::size_t c = 0; c < 4; ++c) {
                for (std::size_t d = 0; d < 4; ++d) {
                    t(a, b, c, d) =
                        static_cast<double>(a) + 2.0 * static_cast<double>(b) +
                        3.0 * static_cast<double>(c) + 4.0 * static_cast<double>(d);
                }
            }
        }
    }
    return t;
}

/// The flat Minkowski metric, signature (-, +, +, +). Its own inverse, which
/// is what makes it a good fixture for the index-raising identity.
Rank2 minkowski() {
    Rank2 t{};
    t(0, 0) = -1.0;
    t(1, 1) = 1.0;
    t(2, 2) = 1.0;
    t(3, 3) = 1.0;
    return t;
}

// --- Shape ------------------------------------------------------------------

TEST(MathTensor, ShapeIsFixedAtCompileTime) {
    static_assert(Rank0::rank() == 0);
    static_assert(Rank0::size() == 1);
    static_assert(Rank1::size() == 4);
    static_assert(Rank2::size() == 16);
    static_assert(Rank3::size() == 64);
    static_assert(Rank4::size() == 256);
    static_assert(Rank4::dimension() == 4);

    static_assert(sizeof(Rank2) == 16 * sizeof(double));
    static_assert(std::is_standard_layout_v<Rank2>);
    static_assert(std::is_trivially_copyable_v<Rank2>);

    EXPECT_EQ(Rank2{}, Rank2::zero());
}

TEST(MathTensor, StorageIsRowMajorSoTheLastIndexVariesFastest) {
    const Rank2 t = distinctRank2();
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(t(i, j), t[i * 4 + j]) << "at (" << i << ", " << j << ")";
        }
    }

    Rank3 u{};
    u(1, 2, 3) = 7.0;
    EXPECT_EQ(u[1 * 16 + 2 * 4 + 3], 7.0);

    // Rank zero is a scalar with one component and no indices.
    Rank0 scalar{};
    scalar() = 5.0;
    EXPECT_EQ(scalar(), 5.0);
    EXPECT_EQ(scalar[0], 5.0);
}

TEST(MathTensor, DeltaIsTheIdentityOnIndices) {
    const Rank2 d = Rank2::delta();
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(d(i, j), (i == j) ? 1.0 : 0.0);
        }
    }
    EXPECT_APPROX(trace(d), 4.0);
    EXPECT_EQ(Rank2::filled(2.0)(3, 1), 2.0);
}

// --- Arithmetic -------------------------------------------------------------

TEST(MathTensor, ArithmeticActsComponentwise) {
    const Rank2 a = distinctRank2();
    const Rank2 b = Rank2::filled(2.0);

    EXPECT_EQ((a + b)(1, 2), a(1, 2) + 2.0);
    EXPECT_EQ((a - b)(1, 2), a(1, 2) - 2.0);
    EXPECT_EQ((a * 3.0)(1, 2), a(1, 2) * 3.0);
    EXPECT_EQ((3.0 * a)(1, 2), a(1, 2) * 3.0);
    EXPECT_EQ((a / 2.0)(1, 2), a(1, 2) / 2.0);
    EXPECT_EQ((-a)(1, 2), -a(1, 2));
    EXPECT_EQ(+a, a);

    EXPECT_EQ(a + Rank2::zero(), a);
    EXPECT_EQ(a - a, Rank2::zero());
    EXPECT_EQ(a + b, b + a);

    Rank2 mutated = a;
    mutated += b;
    EXPECT_EQ(mutated, a + b);
    mutated = a;
    mutated *= 2.0;
    EXPECT_EQ(mutated, a * 2.0);
}

// --- Outer product ----------------------------------------------------------

TEST(MathTensor, OuterProductAddsRanksAndMultipliesComponents) {
    const Rank1 v = toTensor(Vec4{1.0, 2.0, 3.0, 4.0});
    const Rank1 w = toTensor(Vec4{5.0, -1.0, 0.5, 2.0});

    const auto product = outerProduct(v, w);
    static_assert(std::is_same_v<decltype(product), const Rank2>);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_APPROX(product(i, j), v(i) * w(j));
        }
    }

    // It agrees with the matrix outer product, which was tested separately.
    const Vec3 a{1.0, -2.0, 0.5};
    const Vec3 b{3.0, 1.0, -1.0};
    EXPECT_TENSOR_APPROX(outerProduct(toTensor(a), toTensor(b)),
                         toTensor(Mat3::outerProduct(a, b)));

    // Rank keeps adding.
    static_assert(std::is_same_v<decltype(outerProduct(distinctRank2(),
                                                       distinctRank2())),
                                 Rank4>);
}

// --- Contraction ------------------------------------------------------------

TEST(MathTensor, ContractingTwoVectorsIsTheDotProduct) {
    const Vec4 a{1.0, 2.0, 3.0, 4.0};
    const Vec4 b{5.0, -1.0, 0.5, 2.0};

    const auto result = contract<0, 0>(toTensor(a), toTensor(b));
    static_assert(std::is_same_v<decltype(result), const Rank0>);
    EXPECT_APPROX(result(), dot(a, b));
}

TEST(MathTensor, ContractingAMatrixWithAVectorIsTheMatrixVectorProduct) {
    const Mat4 m = Mat4::fromRows({2.0, -1.0, 3.0, 0.5}, {0.5, 4.0, -2.0, 1.0},
                                  {-1.5, 1.0, 5.0, -3.0}, {1.0, 0.25, -0.5, 2.0});
    const Vec4 v{1.0, -2.0, 3.5, 0.25};

    // Index 1 of the matrix is its column index, which is what a matrix-vector
    // product sums over.
    EXPECT_TENSOR_NEAR((contract<1, 0>(toTensor(m), toTensor(v))),
                       toTensor(m * v), zeroTolerance(50.0));
}

TEST(MathTensor, ContractingTwoMatricesIsTheMatrixProduct) {
    // The strongest check available on the index machinery: Matrix4's product
    // was tested on its own, and shares no code at all with the flat-index
    // stride arithmetic in contract().
    const Mat4 a = Mat4::fromRows({2.0, -1.0, 3.0, 0.5}, {0.5, 4.0, -2.0, 1.0},
                                  {-1.5, 1.0, 5.0, -3.0}, {1.0, 0.25, -0.5, 2.0});
    const Mat4 b = Mat4::fromRows({1.0, 2.0, -1.0, 3.0}, {3.0, -0.5, 2.0, 1.0},
                                  {0.25, 1.0, 4.0, -2.0}, {-1.0, 0.5, 1.5, 1.0});

    EXPECT_TENSOR_NEAR((contract<1, 0>(toTensor(a), toTensor(b))),
                       toTensor(a * b), zeroTolerance(200.0));

    // Contracting the other index of the left operand transposes it.
    EXPECT_TENSOR_NEAR((contract<0, 0>(toTensor(a), toTensor(b))),
                       toTensor(transpose(a) * b), zeroTolerance(200.0));
}

TEST(MathTensor, TheInverseMetricContractsWithTheMetricToTheKroneckerDelta) {
    // g^{mu nu} g_{nu rho} = delta^mu_rho. Minkowski is its own inverse, so
    // this needs no inversion machinery to state.
    const Rank2 eta = minkowski();
    EXPECT_TENSOR_APPROX((contract<1, 0>(eta, eta)), Rank2::delta());
    EXPECT_APPROX(trace(eta), 2.0);  // -1 + 1 + 1 + 1
}

TEST(MathTensor, ContractionIsAssociative) {
    const Rank2 a = distinctRank2();
    const Rank2 b = minkowski();
    const Rank1 v = toTensor(Vec4{1.0, -2.0, 3.5, 0.25});

    // (a b) v == a (b v)
    const auto left = contract<1, 0>(contract<1, 0>(a, b), v);
    const auto right = contract<1, 0>(a, contract<1, 0>(b, v));
    EXPECT_TENSOR_NEAR(left, right, zeroTolerance(500.0));
}

TEST(MathTensor, ContractionShrinksRankByTwo) {
    static_assert(std::is_same_v<decltype(contract<0, 0>(Rank1{}, Rank1{})), Rank0>);
    static_assert(std::is_same_v<decltype(contract<0, 0>(Rank2{}, Rank1{})), Rank1>);
    static_assert(std::is_same_v<decltype(contract<0, 0>(Rank2{}, Rank2{})), Rank2>);
    static_assert(std::is_same_v<decltype(contract<0, 0>(Rank4{}, Rank2{})), Rank4>);
}

// --- Self-contraction -------------------------------------------------------

TEST(MathTensor, TracingARankTwoTensorMatchesItsOrdinaryTrace) {
    const Rank2 t = distinctRank2();
    const auto traced = traceOver<0, 1>(t);
    static_assert(std::is_same_v<decltype(traced), const Rank0>);
    EXPECT_APPROX(traced(), trace(t));
    EXPECT_APPROX(trace(t), t(0, 0) + t(1, 1) + t(2, 2) + t(3, 3));

    // And the trace of an outer product is the dot product.
    const Vec4 a{1.0, 2.0, 3.0, 4.0};
    const Vec4 b{5.0, -1.0, 0.5, 2.0};
    EXPECT_APPROX((traceOver<0, 1>(outerProduct(toTensor(a), toTensor(b)))()),
                  dot(a, b));
}

TEST(MathTensor, TracingARankFourTensorContractsTheNamedIndices) {
    // The shape Ricci has: R_{bd} = R^a_{bad}, a trace over two indices of one
    // tensor rather than a contraction of two.
    const Rank4 r = distinctRank4();
    const auto traced = traceOver<0, 2>(r);
    static_assert(std::is_same_v<decltype(traced), const Rank2>);

    // With R(a,b,c,d) = a + 2b + 3c + 4d, summing a == c over four values
    // gives 24 + 8b + 16d.
    for (std::size_t b = 0; b < 4; ++b) {
        for (std::size_t d = 0; d < 4; ++d) {
            EXPECT_APPROX(traced(b, d), 24.0 + 8.0 * static_cast<double>(b) +
                                            16.0 * static_cast<double>(d));
        }
    }
}

// --- Index permutation and symmetry -----------------------------------------

TEST(MathTensor, TransposingIndicesIsAnInvolutionAndMatchesMatrixTranspose) {
    const Rank2 t = distinctRank2();

    EXPECT_EQ((transposeIndices<0, 1>(transposeIndices<0, 1>(t))), t);
    EXPECT_EQ((transposeIndices<0, 1>(t)), toTensor(transpose(toMatrix4(t))));
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_EQ((transposeIndices<0, 1>(t)(i, j)), t(j, i));
        }
    }

    // On a higher rank it moves only the two indices it names.
    const Rank4 r = distinctRank4();
    const Rank4 swapped = transposeIndices<1, 3>(r);
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            for (std::size_t c = 0; c < 4; ++c) {
                for (std::size_t d = 0; d < 4; ++d) {
                    EXPECT_EQ(swapped(a, b, c, d), r(a, d, c, b));
                }
            }
        }
    }
}

TEST(MathTensor, SymmetricAndAntisymmetricPartsDecomposeTheTensor) {
    const Rank2 t = distinctRank2();
    const Rank2 symmetric = symmetrize<0, 1>(t);
    const Rank2 antisymmetric = antisymmetrize<0, 1>(t);

    EXPECT_TENSOR_NEAR(symmetric + antisymmetric, t, zeroTolerance(100.0));

    EXPECT_TENSOR_NEAR((transposeIndices<0, 1>(symmetric)), symmetric,
                       zeroTolerance(100.0));
    EXPECT_TENSOR_NEAR((transposeIndices<0, 1>(antisymmetric)), -antisymmetric,
                       zeroTolerance(100.0));

    // The projectors are idempotent, and each kills the other's image.
    EXPECT_TENSOR_NEAR((symmetrize<0, 1>(symmetric)), symmetric,
                       zeroTolerance(100.0));
    EXPECT_TENSOR_NEAR((antisymmetrize<0, 1>(symmetric)), Rank2::zero(),
                       zeroTolerance(100.0));
    EXPECT_TENSOR_NEAR((symmetrize<0, 1>(antisymmetric)), Rank2::zero(),
                       zeroTolerance(100.0));

    // An antisymmetric tensor has zero trace.
    EXPECT_NEAR(trace(antisymmetric), 0.0, zeroTolerance(100.0));
}

TEST(MathTensor, SymmetryProjectorsWorkOnAnInnerPairOfIndices) {
    // Riemann's symmetries are on index pairs, not on the first two, so the
    // projectors have to work anywhere.
    const Rank4 r = distinctRank4();
    const Rank4 antisymmetric = antisymmetrize<2, 3>(r);

    EXPECT_TENSOR_NEAR((transposeIndices<2, 3>(antisymmetric)), -antisymmetric,
                       zeroTolerance(100.0));
    // Antisymmetric in a pair means the diagonal of that pair vanishes.
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            for (std::size_t c = 0; c < 4; ++c) {
                EXPECT_NEAR(antisymmetric(a, b, c, c), 0.0, zeroTolerance(10.0));
            }
        }
    }
}

// --- Conversions ------------------------------------------------------------

TEST(MathTensor, ConversionsWithTheFixedTypesRoundTrip) {
    const Vec3 v3{1.0, -2.0, 0.5};
    const Vec4 v4{1.0, -2.0, 0.5, 3.0};
    const Mat3 m3 = Mat3::fromRows({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0},
                                   {7.0, 8.0, 9.0});
    const Mat4 m4 = Mat4::fromRows({2.0, -1.0, 3.0, 0.5}, {0.5, 4.0, -2.0, 1.0},
                                   {-1.5, 1.0, 5.0, -3.0},
                                   {1.0, 0.25, -0.5, 2.0});

    EXPECT_EQ(toVector3(toTensor(v3)), v3);
    EXPECT_EQ(toVector4(toTensor(v4)), v4);
    EXPECT_EQ(toMatrix3(toTensor(m3)), m3);
    EXPECT_EQ(toMatrix4(toTensor(m4)), m4);

    // The tensor index order is (row, column), matching operator() on Matrix.
    EXPECT_EQ(toTensor(m3)(0, 2), m3(0, 2));
    EXPECT_EQ(toTensor(m3)(2, 0), m3(2, 0));
}

// --- Composition with Dual --------------------------------------------------

TEST(MathTensor, ComposesWithDualSoAMetricCanCarryItsOwnDerivatives) {
    // This is the shape Physics/Spacetime needs: a metric whose components
    // know how they vary, so the Christoffel symbols come out exact rather
    // than finite-differenced.
    using D = ysq::Dual<double>;
    using MetricDual = ysq::Tensor<D, 2, 4>;

    // g_tt = -(1 - 2/r), varying with r, in the Schwarzschild style.
    const double r = 10.0;
    MetricDual g = MetricDual::zero();
    g(0, 0) = D{-(1.0 - 2.0 / r), -2.0 / (r * r)};
    g(1, 1) = D{1.0 / (1.0 - 2.0 / r), 0.0};
    g(2, 2) = D{r * r, 2.0 * r};
    g(3, 3) = D{r * r, 2.0 * r};

    EXPECT_NEAR(trace(g).value, g(0, 0).value + g(1, 1).value + 2.0 * r * r, 1e-12);
    EXPECT_NEAR(trace(g).derivative, -2.0 / (r * r) + 4.0 * r, 1e-12);

    const MetricDual doubled = g + g;
    EXPECT_NEAR(doubled(2, 2).derivative, 4.0 * r, 1e-13);
    EXPECT_NEAR((contract<1, 0>(g, g)(2, 2).derivative), 2.0 * (r * r) * (2.0 * r),
                1e-9);
}

// --- Formatting -------------------------------------------------------------

TEST(MathTensor, FormattingPrintsTheFlatComponents) {
    ysq::Tensor<double, 2, 2> t{};
    t(0, 0) = 1.0;
    t(0, 1) = 2.0;
    t(1, 0) = 3.0;
    t(1, 1) = 4.0;
    EXPECT_EQ(std::format("{}", t), "(1, 2, 3, 4)");
}

}  // namespace
