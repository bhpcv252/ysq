#include <Math/Complex.hpp>

#include <Math/Dual.hpp>
#include <Math/Format.hpp>
#include <Math/Scalar.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <format>
#include <limits>

namespace {

using ysq::Cplx;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kPi = ysq::kPi<double>;

double zeroTolerance(double scale) {
    return 64.0 * kEps * scale;
}

/// Nothing purely real, purely imaginary, or on the unit circle, so an
/// implementation that happens to work for the easy cases has somewhere to
/// fail.
constexpr std::array<Cplx, 5> kSamples{
    Cplx{3.0, 4.0}, Cplx{-1.5, 2.25}, Cplx{0.5, -0.75}, Cplx{-2.0, -3.5}, Cplx{1.0, 0.0},
};

// --- Field axioms -----------------------------------------------------------

TEST(MathComplex, AdditionAndMultiplicationAreCommutativeAndAssociative) {
    for (const Cplx& a : kSamples) {
        for (const Cplx& b : kSamples) {
            EXPECT_EQ(a + b, b + a);
            EXPECT_EQ(a * b, b * a);
            for (const Cplx& c : kSamples) {
                EXPECT_VEC_NEAR((a + b) + c, a + (b + c), zeroTolerance(10.0));
                EXPECT_VEC_NEAR((a * b) * c, a * (b * c), zeroTolerance(200.0));
            }
        }
    }
}

TEST(MathComplex, MultiplicationDistributesOverAddition) {
    for (const Cplx& a : kSamples) {
        for (const Cplx& b : kSamples) {
            for (const Cplx& c : kSamples) {
                EXPECT_VEC_NEAR(a * (b + c), a * b + a * c, zeroTolerance(100.0));
            }
        }
    }
}

TEST(MathComplex, TheIdentitiesAndInversesBehave) {
    for (const Cplx& a : kSamples) {
        EXPECT_EQ(a + Cplx::zero(), a);
        EXPECT_EQ(a - a, Cplx::zero());
        EXPECT_EQ(a * Cplx::one(), a);
        EXPECT_EQ(-(-a), a);
        EXPECT_EQ(+a, a);

        EXPECT_VEC_NEAR(a * inverse(a), Cplx::one(), zeroTolerance(10.0));
        EXPECT_VEC_NEAR(a / a, Cplx::one(), zeroTolerance(10.0));
        EXPECT_VEC_NEAR(inverse(inverse(a)), a, zeroTolerance(10.0));
    }
}

TEST(MathComplex, TheImaginaryUnitSquaresToMinusOne) {
    EXPECT_EQ(Cplx::i() * Cplx::i(), -Cplx::one());
    EXPECT_EQ(Cplx::i() * Cplx::i() * Cplx::i() * Cplx::i(), Cplx::one());
    // Multiplying by i is a quarter turn.
    EXPECT_EQ((Cplx{3.0, 4.0}) * Cplx::i(), ((Cplx{-4.0, 3.0})));
}

TEST(MathComplex, ScalarOperationsAgreeWithTheComplexOnes) {
    const Cplx a{3.0, 4.0};
    EXPECT_EQ(a + 2.0, a + Cplx::real(2.0));
    EXPECT_EQ(2.0 + a, a + Cplx::real(2.0));
    EXPECT_EQ(a - 2.0, a - Cplx::real(2.0));
    EXPECT_EQ(2.0 - a, Cplx::real(2.0) - a);
    EXPECT_EQ(a * 2.0, a * Cplx::real(2.0));
    EXPECT_EQ(2.0 * a, a * Cplx::real(2.0));
    EXPECT_VEC_APPROX(a / 2.0, a / Cplx::real(2.0));
}

// --- Conjugation and modulus ------------------------------------------------

TEST(MathComplex, ConjugationIsAnInvolutionThatDistributes) {
    for (const Cplx& a : kSamples) {
        EXPECT_EQ(conj(conj(a)), a);
        EXPECT_APPROX(abs(conj(a)), abs(a));
        // z conj(z) is the squared modulus, and real.
        EXPECT_VEC_NEAR(a * conj(a), Cplx::real(lengthSquared(a)), zeroTolerance(50.0));

        for (const Cplx& b : kSamples) {
            EXPECT_EQ(conj(a + b), conj(a) + conj(b));
            EXPECT_VEC_NEAR(conj(a * b), conj(a) * conj(b), zeroTolerance(50.0));
        }
    }
}

TEST(MathComplex, ModulusIsMultiplicative) {
    for (const Cplx& a : kSamples) {
        EXPECT_APPROX(lengthSquared(a), a.re * a.re + a.im * a.im);
        EXPECT_APPROX(length(a), abs(a));
        for (const Cplx& b : kSamples) {
            EXPECT_NEAR(abs(a * b), abs(a) * abs(b), zeroTolerance(50.0));
            // And the triangle inequality.
            EXPECT_LE(abs(a + b), abs(a) + abs(b) + zeroTolerance(10.0));
        }
    }
}

TEST(MathComplex, ModulusDoesNotOverflowWhereTheAnswerFits) {
    // hypot rather than sqrt of the sum of squares. Squaring 1e200 is
    // infinity; the modulus it is asking about is perfectly representable.
    const Cplx huge{1e200, 1e200};
    EXPECT_TRUE(std::isfinite(abs(huge)));
    EXPECT_NEAR(abs(huge), std::sqrt(2.0) * 1e200, 1e188);

    const Cplx tiny{1e-200, 1e-200};
    EXPECT_GT(abs(tiny), 0.0) << "and squaring 1e-200 underflows to zero";
    EXPECT_NEAR(abs(tiny), std::sqrt(2.0) * 1e-200, 1e-212);
}

TEST(MathComplex, DivisionDoesNotOverflowWhereTheAnswerFits) {
    // Smith's formula. The textbook (a conj(b)) / |b|^2 computes |b|^2 = 2e400
    // and hands back NaN for a quotient that is close to 1.
    const Cplx numerator{1e200, 1e200};
    const Cplx denominator{1e200, 1e200};
    EXPECT_VEC_NEAR(numerator / denominator, Cplx::one(), 1e-14);

    // A denominator whose squared modulus overflows while the quotient itself
    // is perfectly ordinary. The ratio inside Smith's formula underflows to
    // zero here, which is harmless; squaring 1e200 is not.
    const Cplx lopsided = Cplx{1e200, 1e200} / Cplx{1e200, 1e-200};
    EXPECT_VEC_NEAR(lopsided, (Cplx{1.0, 1.0}), 1e-14);

    // Both branches of the formula, so neither is left untested. The values
    // are (1 + 2i) conj(b) / |b|^2 worked through by hand.
    EXPECT_VEC_NEAR((Cplx{1.0, 2.0}) / (Cplx{3.0, 0.5}), (Cplx{4.0 / 9.25, 5.5 / 9.25}),
                    zeroTolerance(10.0));
    EXPECT_VEC_NEAR((Cplx{1.0, 2.0}) / (Cplx{0.5, 3.0}), (Cplx{6.5 / 9.25, -2.0 / 9.25}),
                    zeroTolerance(10.0));
}

// --- Argument and polar form ------------------------------------------------

TEST(MathComplex, ArgumentAddsUnderMultiplication) {
    EXPECT_APPROX(arg(Cplx::one()), 0.0);
    EXPECT_APPROX(arg(Cplx::i()), kPi / 2.0);
    EXPECT_APPROX(arg((Cplx{-1.0, 0.0})), kPi);
    EXPECT_APPROX(arg((Cplx{0.0, -1.0})), -kPi / 2.0);

    for (const Cplx& a : kSamples) {
        for (const Cplx& b : kSamples) {
            double sum = arg(a) + arg(b);
            // Back into the principal range before comparing.
            if (sum > kPi) {
                sum -= 2.0 * kPi;
            }
            if (sum <= -kPi) {
                sum += 2.0 * kPi;
            }
            EXPECT_NEAR(arg(a * b), sum, zeroTolerance(10.0));
        }
    }
}

TEST(MathComplex, PolarFormRoundTrips) {
    for (const Cplx& a : kSamples) {
        EXPECT_VEC_NEAR(Cplx::polar(abs(a), arg(a)), a, zeroTolerance(20.0));
    }
    EXPECT_VEC_NEAR(Cplx::polar(2.0, kPi / 2.0), (Cplx{0.0, 2.0}), zeroTolerance(10.0));
}

// --- Transcendental functions -----------------------------------------------

TEST(MathComplex, ExponentialSatisfiesEulersIdentity) {
    for (const double angle : {0.0, 0.5, 1.0, kPi / 2.0, kPi, -2.3}) {
        EXPECT_VEC_NEAR(exp(Cplx::imaginary(angle)),
                        (Cplx{std::cos(angle), std::sin(angle)}), zeroTolerance(10.0))
            << "at angle " << angle;
    }
    // The famous one.
    EXPECT_VEC_NEAR(exp(Cplx::imaginary(kPi)) + Cplx::one(), Cplx::zero(),
                    zeroTolerance(10.0));
}

TEST(MathComplex, ExponentialTurnsAdditionIntoMultiplication) {
    for (const Cplx& a : kSamples) {
        for (const Cplx& b : kSamples) {
            EXPECT_VEC_NEAR(exp(a + b), exp(a) * exp(b),
                            zeroTolerance(abs(exp(a)) * abs(exp(b)) * 10.0));
        }
    }
}

TEST(MathComplex, LogarithmInvertsTheExponentialOnItsBranch) {
    // This direction always holds: log picks some logarithm, exp undoes it.
    for (const Cplx& a : kSamples) {
        EXPECT_VEC_NEAR(exp(log(a)), a, zeroTolerance(50.0));
    }

    // The other direction only holds where the imaginary part already sits
    // inside the principal branch.
    for (const Cplx& a : {Cplx{3.0, 1.0}, Cplx{-1.5, 2.25}, Cplx{0.5, -0.75}}) {
        ASSERT_LE(std::abs(a.im), kPi);
        EXPECT_VEC_NEAR(log(exp(a)), a, zeroTolerance(50.0));
    }

    // Outside it the imaginary part comes back folded by a multiple of 2 pi.
    // That is not a defect; it is what choosing a branch means.
    EXPECT_VEC_NEAR(log(exp(Cplx{3.0, 4.0})), (Cplx{3.0, 4.0 - 2.0 * kPi}),
                    zeroTolerance(50.0));
    EXPECT_VEC_NEAR(log(Cplx::one()), Cplx::zero(), zeroTolerance(10.0));
    EXPECT_VEC_NEAR(log((Cplx{-1.0, 0.0})), Cplx::imaginary(kPi), zeroTolerance(10.0));
}

TEST(MathComplex, SquareRootSquaresBackAndStaysOnThePrincipalBranch) {
    for (const Cplx& a : kSamples) {
        const Cplx root = sqrt(a);
        EXPECT_VEC_NEAR(root * root, a, zeroTolerance(50.0));
        EXPECT_GE(root.re, 0.0) << "the principal root has non-negative real part";
        EXPECT_NEAR(abs(root) * abs(root), abs(a), zeroTolerance(50.0));
    }

    // The negative reals, where the branch cut is and where a polar
    // implementation is at its worst.
    EXPECT_VEC_NEAR(sqrt((Cplx{-4.0, 0.0})), (Cplx{0.0, 2.0}), zeroTolerance(10.0));
    EXPECT_VEC_NEAR(sqrt((Cplx{-4.0, -0.0})), (Cplx{0.0, 2.0}), zeroTolerance(10.0));
    EXPECT_EQ(sqrt(Cplx::zero()), Cplx::zero());
    EXPECT_VEC_NEAR(sqrt((Cplx{4.0, 0.0})), (Cplx{2.0, 0.0}), zeroTolerance(10.0));

    // Just off the cut, where the answer is nearly purely imaginary.
    const Cplx nearCut = sqrt(Cplx{-1.0, 1e-14});
    EXPECT_VEC_NEAR(nearCut * nearCut, (Cplx{-1.0, 1e-14}), 1e-14);
}

TEST(MathComplex, PowerSatisfiesDeMoivre) {
    // (cos t + i sin t)^n == cos nt + i sin nt.
    const double angle = 0.7;
    const Cplx unit = Cplx::polar(1.0, angle);

    for (const double n : {2.0, 3.0, 5.0}) {
        EXPECT_VEC_NEAR(pow(unit, n), Cplx::polar(1.0, angle * n), zeroTolerance(20.0))
            << "at n = " << n;
    }

    // And the integer powers agree with repeated multiplication.
    const Cplx a{1.5, -0.5};
    EXPECT_VEC_NEAR(pow(a, 3.0), a * a * a, zeroTolerance(50.0));
    EXPECT_VEC_NEAR(pow(a, Cplx::real(2.0)), a * a, zeroTolerance(50.0));
}

// --- Composition with Dual --------------------------------------------------

TEST(MathComplex, ComposesWithDualForDerivativesThroughComplexArithmetic) {
    // The reason this is not std::complex: the standard only specifies that
    // template for float, double and long double, so a complex number whose
    // parts carry their own derivatives is not something it promises at all.
    using D = ysq::Dual<double>;
    using CplxDual = ysq::Complex<D>;

    // z(t) = (t^2) + i(3t), so dz/dt = 2t + 3i, and |z|^2 = t^4 + 9t^2 has
    // derivative 4t^3 + 18t.
    const double t = 1.7;
    const CplxDual z{D{t * t, 2.0 * t}, D{3.0 * t, 3.0}};

    EXPECT_NEAR(z.re.derivative, 2.0 * t, 1e-15);
    EXPECT_NEAR(lengthSquared(z).derivative, 4.0 * t * t * t + 18.0 * t, 1e-13);

    // And through a multiplication, where the product rule has to thread
    // through both the complex and the dual arithmetic at once.
    const CplxDual squared = z * z;
    // d(z^2)/dt = 2 z dz/dt
    const ysq::Cplx expected = 2.0 * ysq::Cplx{t * t, 3.0 * t} * ysq::Cplx{2.0 * t, 3.0};
    EXPECT_NEAR(squared.re.derivative, expected.re, 1e-13);
    EXPECT_NEAR(squared.im.derivative, expected.im, 1e-13);
}

// --- Formatting -------------------------------------------------------------

TEST(MathComplex, FormattingReadsAsASum) {
    EXPECT_EQ(std::format("{}", (Cplx{1.0, 2.0})), "1 + 2i");
    EXPECT_EQ(std::format("{}", (Cplx{1.0, -2.0})), "1 - 2i")
        << "the sign belongs to the operator, not to a printed -2";
    EXPECT_EQ(std::format("{:.2f}", (Cplx{1.5, -0.25})), "1.50 - 0.25i");
}

// --- Single precision -------------------------------------------------------

TEST(MathComplex, IdentitiesHoldAtSinglePrecision) {
    using ysq::Cplxf;
    constexpr float tol = 1e-5f;

    const Cplxf a{3.0f, 4.0f};
    const Cplxf b{-1.5f, 2.25f};

    EXPECT_NEAR(abs(a), 5.0f, tol);
    EXPECT_VEC_NEAR(a * inverse(a), Cplxf::one(), tol);
    EXPECT_NEAR(abs(a * b), abs(a) * abs(b), tol);
    EXPECT_VEC_NEAR(sqrt(a) * sqrt(a), a, 1e-4f);
}

}  // namespace
