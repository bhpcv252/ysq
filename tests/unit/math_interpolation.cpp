#include <Math/Interpolation.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

using ysq::Vec3;

// --- Linear -----------------------------------------------------------------

TEST(MathInterpolation, LerpIsExactAtTheEndpoints) {
    EXPECT_EQ(ysq::lerp(2.0, 5.0, 0.0), 2.0);
    EXPECT_EQ(ysq::lerp(2.0, 5.0, 1.0), 5.0)
        << "the a + (b - a) * t form is not exact here; this one must be";
    EXPECT_APPROX(ysq::lerp(2.0, 5.0, 0.5), 3.5);

    // Exact at t = 1 even where the two ends are wildly different in scale,
    // which is where the cheaper form goes wrong.
    EXPECT_EQ(ysq::lerp(1e16, 1.0, 1.0), 1.0);
    EXPECT_EQ(ysq::lerp(1e16, 1.0, 0.0), 1e16);

    EXPECT_APPROX(ysq::lerp(2.0, 5.0, 2.0), 8.0) << "extrapolates";
    EXPECT_APPROX(ysq::lerp(2.0, 5.0, -1.0), -1.0);
}

TEST(MathInterpolation, InverseLerpAndRemapUndoEachOther) {
    EXPECT_APPROX(ysq::inverseLerp(2.0, 5.0, 3.5), 0.5);
    EXPECT_APPROX(ysq::inverseLerp(2.0, 5.0, 2.0), 0.0);
    EXPECT_APPROX(ysq::inverseLerp(2.0, 5.0, 5.0), 1.0);

    for (const double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        EXPECT_APPROX(ysq::inverseLerp(2.0, 5.0, ysq::lerp(2.0, 5.0, t)), t);
    }

    EXPECT_APPROX(ysq::remap(5.0, 0.0, 10.0, 100.0, 200.0), 150.0);
    EXPECT_APPROX(ysq::remap(0.0, 0.0, 10.0, 100.0, 200.0), 100.0);
    // A reversed output range flips the direction.
    EXPECT_APPROX(ysq::remap(2.5, 0.0, 10.0, 1.0, 0.0), 0.75);
}

// --- Easing -----------------------------------------------------------------

TEST(MathInterpolation, SmoothstepIsClampedMonotonicAndFlatAtBothEnds) {
    EXPECT_APPROX(ysq::smoothstep(0.0, 1.0, 0.0), 0.0);
    EXPECT_APPROX(ysq::smoothstep(0.0, 1.0, 1.0), 1.0);
    EXPECT_APPROX(ysq::smoothstep(0.0, 1.0, 0.5), 0.5);

    EXPECT_APPROX(ysq::smoothstep(0.0, 1.0, -5.0), 0.0) << "clamps below";
    EXPECT_APPROX(ysq::smoothstep(0.0, 1.0, 5.0), 1.0) << "clamps above";

    double previous = -1.0;
    for (std::size_t i = 0; i <= 100; ++i) {
        const double value = ysq::smoothstep(0.0, 1.0, static_cast<double>(i) / 100.0);
        EXPECT_GE(value, previous) << "must not go backwards";
        previous = value;
    }

    // The slope vanishes at both ends, which is the whole point of it.
    constexpr double h = 1e-6;
    EXPECT_NEAR((ysq::smoothstep(0.0, 1.0, h) - ysq::smoothstep(0.0, 1.0, 0.0)) / h, 0.0,
                1e-5);
    EXPECT_NEAR((ysq::smoothstep(0.0, 1.0, 1.0) - ysq::smoothstep(0.0, 1.0, 1.0 - h)) / h,
                0.0, 1e-5);

    // It works on any interval, not only the unit one.
    EXPECT_APPROX(ysq::smoothstep(10.0, 20.0, 15.0), 0.5);
}

TEST(MathInterpolation, SmootherstepAlsoFlattensItsSecondDerivative) {
    EXPECT_APPROX(ysq::smootherstep(0.0, 1.0, 0.0), 0.0);
    EXPECT_APPROX(ysq::smootherstep(0.0, 1.0, 1.0), 1.0);
    EXPECT_APPROX(ysq::smootherstep(0.0, 1.0, 0.5), 0.5);

    // Second difference at the start, which smoothstep leaves non-zero.
    //
    // The step has to be small: a forward second difference reports f'' at the
    // middle of its own stencil, not at the left edge, and smootherstep's
    // second derivative climbs at 60 per unit from zero. At h = 1e-3 it would
    // already read 0.06 and the test would be measuring the stencil rather
    // than the function.
    constexpr double h = 1e-5;
    const auto secondDifference = [](auto easing) {
        return (easing(2.0 * h) - 2.0 * easing(h) + easing(0.0)) / (h * h);
    };
    const double smoother =
        secondDifference([](double x) { return ysq::smootherstep(0.0, 1.0, x); });
    const double smooth =
        secondDifference([](double x) { return ysq::smoothstep(0.0, 1.0, x); });

    EXPECT_NEAR(smoother, 0.0, 1e-2);
    EXPECT_NEAR(smooth, 6.0, 1e-2) << "smoothstep starts with a kick";
}

// --- Multilinear ------------------------------------------------------------

TEST(MathInterpolation, BilinearReducesToLerpOnTheEdges) {
    constexpr double v00 = 1.0;
    constexpr double v10 = 2.0;
    constexpr double v01 = 3.0;
    constexpr double v11 = 5.0;

    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 0.0, 0.0), v00);
    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 1.0, 0.0), v10);
    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 0.0, 1.0), v01);
    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 1.0, 1.0), v11);

    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 0.5, 0.0), ysq::lerp(v00, v10, 0.5));
    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 0.0, 0.5), ysq::lerp(v00, v01, 0.5));
    EXPECT_APPROX(ysq::bilinear(v00, v10, v01, v11, 0.5, 0.5),
                  (v00 + v10 + v01 + v11) / 4.0);
}

TEST(MathInterpolation, BilinearIsExactOnASeparableBilinearFunction) {
    // f(x, y) = 2 + 3x + 5y + 7xy is exactly what the rule reconstructs, so
    // any interior point has to come out right to the last bit.
    const auto f = [](double x, double y) {
        return 2.0 + 3.0 * x + 5.0 * y + 7.0 * x * y;
    };

    for (const double tx : {0.0, 0.3, 0.5, 0.9, 1.0}) {
        for (const double ty : {0.0, 0.2, 0.5, 0.8, 1.0}) {
            EXPECT_NEAR(
                ysq::bilinear(f(0.0, 0.0), f(1.0, 0.0), f(0.0, 1.0), f(1.0, 1.0), tx, ty),
                f(tx, ty), 1e-13);
        }
    }
}

TEST(MathInterpolation, TrilinearHitsEveryCorner) {
    const double c[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    EXPECT_APPROX(
        ysq::trilinear(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], 0.0, 0.0, 0.0),
        c[0]);
    EXPECT_APPROX(
        ysq::trilinear(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], 1.0, 1.0, 1.0),
        c[7]);
    EXPECT_APPROX(
        ysq::trilinear(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], 0.0, 0.0, 1.0),
        c[4]);
    // The centre is the average of all eight.
    EXPECT_APPROX(
        ysq::trilinear(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], 0.5, 0.5, 0.5),
        4.5);
}

// --- Cubics -----------------------------------------------------------------

TEST(MathInterpolation, CubicHermiteMatchesItsEndpointsAndTangents) {
    constexpr double p0 = 1.0;
    constexpr double m0 = 2.0;
    constexpr double p1 = 4.0;
    constexpr double m1 = -1.0;

    EXPECT_APPROX(ysq::cubicHermite(p0, m0, p1, m1, 0.0), p0);
    EXPECT_APPROX(ysq::cubicHermite(p0, m0, p1, m1, 1.0), p1);

    constexpr double h = 1e-6;
    EXPECT_NEAR(
        (ysq::cubicHermite(p0, m0, p1, m1, h) - ysq::cubicHermite(p0, m0, p1, m1, -h)) /
            (2.0 * h),
        m0, 1e-8);
    EXPECT_NEAR((ysq::cubicHermite(p0, m0, p1, m1, 1.0 + h) -
                 ysq::cubicHermite(p0, m0, p1, m1, 1.0 - h)) /
                    (2.0 * h),
                m1, 1e-8);
}

TEST(MathInterpolation, CatmullRomPassesThroughItsInnerControlPoints) {
    constexpr double p0 = 0.0;
    constexpr double p1 = 1.0;
    constexpr double p2 = 3.0;
    constexpr double p3 = 2.0;

    EXPECT_APPROX(ysq::catmullRom(p0, p1, p2, p3, 0.0), p1);
    EXPECT_APPROX(ysq::catmullRom(p0, p1, p2, p3, 1.0), p2);
}

TEST(MathInterpolation, CatmullRomReproducesAStraightLineExactly) {
    // Collinear control points have collinear tangents, so the cubic terms
    // must cancel completely.
    const auto line = [](double x) { return 2.5 * x - 1.0; };

    for (const double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        EXPECT_NEAR(ysq::catmullRom(line(0.0), line(1.0), line(2.0), line(3.0), t),
                    line(1.0 + t), 1e-14);
    }
}

TEST(MathInterpolation, CatmullRomIsContinuousInValueAndSlopeAcrossSegments) {
    const std::vector<double> points{0.0, 1.0, 3.0, 2.0, 5.0, 4.0};

    // The end of one segment is the start of the next, by construction.
    EXPECT_APPROX(ysq::catmullRom(points[0], points[1], points[2], points[3], 1.0),
                  ysq::catmullRom(points[1], points[2], points[3], points[4], 0.0));

    // And so is the slope, which is what C1 means and what makes the path
    // usable for a camera or a trajectory.
    constexpr double h = 1e-6;
    const double leaving =
        (ysq::catmullRom(points[0], points[1], points[2], points[3], 1.0) -
         ysq::catmullRom(points[0], points[1], points[2], points[3], 1.0 - h)) /
        h;
    const double arriving =
        (ysq::catmullRom(points[1], points[2], points[3], points[4], h) -
         ysq::catmullRom(points[1], points[2], points[3], points[4], 0.0)) /
        h;
    EXPECT_NEAR(leaving, arriving, 1e-4);
}

TEST(MathInterpolation, CubicBezierMatchesDeCasteljau) {
    constexpr double p0 = 0.0;
    constexpr double p1 = 1.0;
    constexpr double p2 = 4.0;
    constexpr double p3 = 3.0;

    EXPECT_APPROX(ysq::cubicBezier(p0, p1, p2, p3, 0.0), p0);
    EXPECT_APPROX(ysq::cubicBezier(p0, p1, p2, p3, 1.0), p3);

    // The repeated-lerp construction, which shares no code with the Bernstein
    // expansion the implementation uses.
    for (const double t : {0.0, 0.2, 0.5, 0.8, 1.0}) {
        const double a = ysq::lerp(p0, p1, t);
        const double b = ysq::lerp(p1, p2, t);
        const double c = ysq::lerp(p2, p3, t);
        const double d = ysq::lerp(a, b, t);
        const double e = ysq::lerp(b, c, t);
        EXPECT_NEAR(ysq::cubicBezier(p0, p1, p2, p3, t), ysq::lerp(d, e, t), 1e-14);
    }
}

TEST(MathInterpolation, TheCurvesWorkOnVectorsAsWellAsScalars) {
    // Generic in the value type, which is what lets the same function smooth a
    // scalar table and trace a path through positions.
    const Vec3 p0{0.0, 0.0, 0.0};
    const Vec3 p1{1.0, 2.0, 0.0};
    const Vec3 p2{3.0, 1.0, 1.0};
    const Vec3 p3{4.0, 4.0, 2.0};

    EXPECT_VEC_APPROX(ysq::catmullRom(p0, p1, p2, p3, 0.0), p1);
    EXPECT_VEC_APPROX(ysq::catmullRom(p0, p1, p2, p3, 1.0), p2);
    EXPECT_VEC_APPROX(ysq::cubicBezier(p0, p1, p2, p3, 0.0), p0);
    EXPECT_VEC_APPROX(ysq::cubicBezier(p0, p1, p2, p3, 1.0), p3);
    EXPECT_VEC_APPROX(ysq::bilinear(p0, p1, p2, p3, 0.0, 0.0), p0);

    // Componentwise, so each coordinate follows the scalar answer.
    const Vec3 midpoint = ysq::catmullRom(p0, p1, p2, p3, 0.4);
    EXPECT_APPROX(midpoint.x, ysq::catmullRom(p0.x, p1.x, p2.x, p3.x, 0.4));
    EXPECT_APPROX(midpoint.z, ysq::catmullRom(p0.z, p1.z, p2.z, p3.z, 0.4));
}

// --- Table lookup -----------------------------------------------------------

TEST(MathInterpolation, TableLookupIsExactAtKnotsAndLinearBetween) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 4.0};
    const std::vector<double> ys{0.0, 10.0, 30.0, 30.0};

    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_APPROX(*ysq::interpolateTable(xs, ys, xs[i]), ys[i]);
    }

    EXPECT_APPROX(*ysq::interpolateTable(xs, ys, 0.5), 5.0);
    EXPECT_APPROX(*ysq::interpolateTable(xs, ys, 1.5), 20.0);
    EXPECT_APPROX(*ysq::interpolateTable(xs, ys, 3.0), 30.0)
        << "the last interval is flat, and unevenly spaced";
}

TEST(MathInterpolation, TableLookupHoldsTheEndpointsRatherThanExtrapolating) {
    const std::vector<double> xs{1.0, 2.0, 3.0};
    const std::vector<double> ys{10.0, 20.0, 30.0};

    EXPECT_APPROX(*ysq::interpolateTable(xs, ys, -100.0), 10.0);
    EXPECT_APPROX(*ysq::interpolateTable(xs, ys, 100.0), 30.0);
}

TEST(MathInterpolation, TableLookupRejectsATableItCannotUse) {
    const std::vector<double> xs{1.0, 2.0, 3.0};
    const std::vector<double> shortY{10.0, 20.0};
    const std::vector<double> single{1.0};

    EXPECT_FALSE(ysq::interpolateTable(xs, shortY, 1.5).has_value());
    EXPECT_FALSE(ysq::interpolateTable(single, single, 1.0).has_value());
}

// --- Cubic spline -----------------------------------------------------------

TEST(MathSpline, PassesExactlyThroughEveryKnot) {
    // The defining property. A spline that misses its own data points is not
    // interpolating anything.
    const std::vector<double> xs{0.0, 1.0, 2.5, 4.0, 6.0, 7.0};
    const std::vector<double> ys{1.0, 3.0, 2.0, 5.0, 4.0, 6.0};

    const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());
    EXPECT_EQ(spline->size(), xs.size());
    EXPECT_APPROX(spline->lowerBound(), 0.0);
    EXPECT_APPROX(spline->upperBound(), 7.0);

    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR((*spline)(xs[i]), ys[i], 1e-12) << "at knot " << i;
    }
}

TEST(MathSpline, ReproducesAStraightLineExactly) {
    // A line has zero second derivative everywhere, which is exactly what the
    // natural end condition asks for, so this one is exact rather than merely
    // close. A quadratic or a cubic is not, and cannot be: their second
    // derivatives do not vanish at the boundary.
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> ys;
    for (const double x : xs) {
        ys.push_back(2.5 * x - 1.0);
    }

    const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());

    for (const double at : {0.3, 1.7, 2.0, 2.5, 3.9}) {
        EXPECT_NEAR((*spline)(at), 2.5 * at - 1.0, 1e-13) << "at " << at;
        EXPECT_NEAR(spline->derivative(at), 2.5, 1e-12);
    }
}

TEST(MathSpline, IsContinuousInValueAndSlopeAcrossEveryKnot) {
    const std::vector<double> xs{0.0, 1.0, 2.5, 4.0, 6.0};
    const std::vector<double> ys{1.0, 3.0, 2.0, 5.0, 4.0};

    const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());

    constexpr double h = 1e-6;
    for (std::size_t i = 1; i + 1 < xs.size(); ++i) {
        EXPECT_NEAR((*spline)(xs[i] - h), (*spline)(xs[i] + h), 1e-5)
            << "value jumps at knot " << i;

        const double before = ((*spline)(xs[i] - h) - (*spline)(xs[i] - 2.0 * h)) / h;
        const double after = ((*spline)(xs[i] + 2.0 * h) - (*spline)(xs[i] + h)) / h;
        EXPECT_NEAR(before, after, 1e-4) << "slope jumps at knot " << i;
    }
}

TEST(MathSpline, DerivativeAgreesWithADifferenceOfTheValues) {
    const std::vector<double> xs{0.0, 1.0, 2.5, 4.0, 6.0};
    const std::vector<double> ys{1.0, 3.0, 2.0, 5.0, 4.0};

    const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());

    constexpr double h = 1e-6;
    for (const double at : {0.4, 1.6, 3.0, 5.2}) {
        const double numerical = ((*spline)(at + h) - (*spline)(at - h)) / (2.0 * h);
        EXPECT_NEAR(spline->derivative(at), numerical, 1e-6) << "at " << at;
    }
}

TEST(MathSpline, SecondDerivativeVanishesAtBothEnds) {
    // What "natural" names, and the reason the spline is not exact on a
    // general cubic.
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> ys{0.0, 1.0, 8.0, 27.0, 64.0};

    const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());

    constexpr double h = 1e-4;
    const auto secondDifference = [&](double at) {
        return ((*spline)(at + h) - 2.0 * (*spline)(at) + (*spline)(at - h)) / (h * h);
    };

    // Stated as a comparison rather than against a fixed tolerance. The
    // spline's second derivative is linear across each interval, so a stencil
    // a finite distance inside the boundary necessarily reads a small non-zero
    // value; what the natural end condition promises is that it goes to zero
    // at the boundary, and the way to see that is that it is negligible there
    // and not in the middle.
    const double interior = std::abs(secondDifference(2.0));
    EXPECT_GT(interior, 1.0) << "the interior curvature is what it is compared to";

    EXPECT_LT(std::abs(secondDifference(0.0 + 2.0 * h)), interior * 0.05);
    EXPECT_LT(std::abs(secondDifference(4.0 - 2.0 * h)), interior * 0.05);
}

TEST(MathSpline, ConvergesOnASmoothFunctionAsKnotsAreAdded) {
    // Away from the boundary the error falls like the fourth power of the knot
    // spacing, so halving it should cut the error by roughly sixteen.
    const auto f = [](double x) { return std::sin(x); };

    const auto worstError = [&](std::size_t intervals) {
        std::vector<double> xs;
        std::vector<double> ys;
        for (std::size_t i = 0; i <= intervals; ++i) {
            const double x =
                6.0 * static_cast<double>(i) / static_cast<double>(intervals);
            xs.push_back(x);
            ys.push_back(f(x));
        }
        const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
        if (!spline) {
            return std::numeric_limits<double>::infinity();
        }

        double worst = 0.0;
        // Interior only: the natural end condition is wrong for a sine and its
        // error near the boundary converges more slowly.
        for (std::size_t i = 0; i <= 200; ++i) {
            const double at = 1.5 + 3.0 * static_cast<double>(i) / 200.0;
            worst = std::max(worst, std::abs((*spline)(at)-f(at)));
        }
        return worst;
    };

    const double coarse = worstError(10);
    const double fine = worstError(20);
    EXPECT_GT(coarse / fine, 8.0)
        << "coarse " << coarse << ", fine " << fine
        << ": expected roughly a factor of sixteen for fourth order";
}

TEST(MathSpline, RejectsInputItCannotBuildFrom) {
    const std::vector<double> good{0.0, 1.0, 2.0};
    const std::vector<double> unsorted{0.0, 2.0, 1.0};
    const std::vector<double> repeated{0.0, 1.0, 1.0};
    const std::vector<double> tooShort{0.0, 1.0};

    EXPECT_TRUE(ysq::CubicSpline<double>::natural(good, good).has_value());
    EXPECT_FALSE(ysq::CubicSpline<double>::natural(unsorted, good).has_value());
    EXPECT_FALSE(ysq::CubicSpline<double>::natural(repeated, good).has_value())
        << "a repeated knot would divide by a zero interval width";
    EXPECT_FALSE(ysq::CubicSpline<double>::natural(tooShort, tooShort).has_value());
    EXPECT_FALSE(ysq::CubicSpline<double>::natural(good, tooShort).has_value());
}

TEST(MathSpline, HoldsItsEndpointValuesOutsideTheTable) {
    const std::vector<double> xs{0.0, 1.0, 2.0};
    const std::vector<double> ys{5.0, 7.0, 11.0};

    const auto spline = ysq::CubicSpline<double>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());

    EXPECT_APPROX((*spline)(-10.0), 5.0);
    EXPECT_APPROX((*spline)(10.0), 11.0);
    EXPECT_APPROX(spline->derivative(-10.0), 0.0);
    EXPECT_APPROX(spline->derivative(10.0), 0.0);
}

// --- Non-finite input -------------------------------------------------------

TEST(MathInterpolation, NonFiniteInputPropagatesOrIsRefused) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> ys{0.0, 10.0, 20.0, 30.0};

    // A NaN lookup point is neither below the table nor above it, so it falls
    // through to the search and comes back as a NaN rather than as an endpoint.
    const auto atNan = ysq::interpolateTable(xs, ys, nan);
    ASSERT_TRUE(atNan.has_value());
    EXPECT_TRUE(std::isnan(*atNan));

    // A NaN knot fails the strictly-increasing test, so the spline refuses to
    // be built rather than being built wrong.
    EXPECT_FALSE(ysq::CubicSpline<double>::natural(std::vector<double>{0.0, nan, 2.0},
                                                   std::vector<double>{0.0, 1.0, 2.0})
                     .has_value());

    // The easing functions clamp, and a NaN passes through the clamp rather
    // than being pinned to an edge, which would be an invented answer.
    EXPECT_TRUE(std::isnan(ysq::smoothstep(0.0, 1.0, nan)));
    EXPECT_TRUE(std::isnan(ysq::lerp(0.0, 1.0, nan)));
}

// --- Single precision -------------------------------------------------------

TEST(MathInterpolation, WorksAtSinglePrecision) {
    EXPECT_EQ(ysq::lerp(2.0f, 5.0f, 1.0f), 5.0f);
    EXPECT_NEAR(ysq::smoothstep(0.0f, 1.0f, 0.5f), 0.5f, 1e-6f);
    EXPECT_NEAR(ysq::catmullRom(0.0f, 1.0f, 3.0f, 2.0f, 0.0f), 1.0f, 1e-6f);

    const std::vector<float> xs{0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> ys{0.0f, 2.5f, 5.0f, 7.5f};
    const auto spline = ysq::CubicSpline<float>::natural(xs, ys);
    ASSERT_TRUE(spline.has_value());
    EXPECT_NEAR((*spline)(1.5f), 3.75f, 1e-5f);
}

}  // namespace
