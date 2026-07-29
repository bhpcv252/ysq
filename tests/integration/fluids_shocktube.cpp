#include <Math/Statistics.hpp>
#include <Physics/Fluids/Eulerian.hpp>

#include <gtest/gtest.h>

#include <cstddef>

/// The Sod shock tube: the standard test problem for a compressible flow
/// solver, a single initial discontinuity between two states at rest that
/// separates into a rarefaction fan, a contact discontinuity and a shock.
/// src/Physics/README.md explains why this validates against the qualitative
/// structure of the exact Riemann solution rather than the solution
/// itself, which the first-order scheme used here is not built to resolve
/// exactly.
///
/// **The domain is periodic** (Physics/Fluids/Eulerian.hpp), so laying down
/// one half at the left state and one half at the right state does not
/// produce a single isolated discontinuity: it produces two, the intended
/// one at the midpoint and a second, identical one where the domain wraps
/// from index (count - 1) back to index 0. Both launch the same rarefaction
/// + contact + shock structure, so "undisturbed" has to be checked at the
/// points equidistant from both, the quarter and three-quarter marks, not
/// at the domain's own edges, which sit right next to the wrap-around
/// discontinuity. The shock outruns the rarefaction's leading edge (a
/// genuine physical asymmetry, not a bug: this Sod configuration's shock
/// speed is faster than its rarefaction head speed), so the
/// three-quarter mark, in the shock's path, needs a shorter safety margin
/// than the quarter mark, in the rarefaction's.

namespace {

constexpr std::size_t kCellCount = 400;
constexpr double kSpacing = 0.0025;  // domain length 1.0
constexpr double kGamma = 1.4;

constexpr double kLeftDensity = 1.0;
constexpr double kLeftPressure = 1.0;
constexpr double kRightDensity = 0.125;
constexpr double kRightPressure = 0.1;

TEST(FluidsShockTube, DevelopsAPressurePlateauBetweenTheInitialValues) {
    ysq::EulerianFluid1D fluid(kCellCount, kSpacing, kGamma);
    for (std::size_t i = 0; i < kCellCount; ++i) {
        if (i < kCellCount / 2) {
            fluid.setState(i, kLeftDensity, 0.0, kLeftPressure);
        } else {
            fluid.setState(i, kRightDensity, 0.0, kRightPressure);
        }
    }

    const double initialMass = fluid.totalMass();
    const double initialEnergy = fluid.totalEnergy();

    // Short enough that neither wave front, from either discontinuity, has
    // reached the quarter-mark checks below.
    const double dt = fluid.stableTimeStep(0.4);
    constexpr int steps = 150;
    for (int s = 0; s < steps; ++s) {
        fluid.step(dt);
    }

    // Conservation holds throughout a run of any length, discontinuity or
    // not: it is a property of the periodic finite-volume update, not of
    // how smooth the solution is.
    EXPECT_NEAR(fluid.totalMass(), initialMass, initialMass * 1e-9);
    EXPECT_NEAR(fluid.totalEnergy(), initialEnergy, initialEnergy * 1e-9);

    // Equidistant from the two discontinuities (the intended one and the
    // periodic wrap-around one), and not yet reached by either wave.
    EXPECT_NEAR(fluid.density(kCellCount / 4), kLeftDensity, kLeftDensity * 1e-3);
    EXPECT_NEAR(fluid.pressure(kCellCount / 4), kLeftPressure, kLeftPressure * 1e-3);
    EXPECT_NEAR(fluid.density(3 * kCellCount / 4), kRightDensity, kRightDensity * 1e-3);
    EXPECT_NEAR(fluid.pressure(3 * kCellCount / 4), kRightPressure,
                kRightPressure * 1e-3);

    // The defining property of a contact discontinuity: pressure and
    // velocity are continuous across it, so the star region between the
    // rarefaction's tail and the shock front is a plateau at one pressure,
    // strictly between the two initial pressures, even though density
    // jumps there. This window (indices 215 to 260, of 400) sits inside
    // that plateau for this configuration at this run length; the
    // preceding indices are still inside the smoothly-varying rarefaction
    // fan, and the following ones approach the shock.
    ysq::RunningStatistics<double> starPressure;
    ysq::RunningStatistics<double> starVelocity;
    for (std::size_t i = 215; i <= 260; ++i) {
        starPressure.add(fluid.pressure(i));
        starVelocity.add(fluid.velocity(i));
    }
    EXPECT_GT(starPressure.minimum(), kRightPressure);
    EXPECT_LT(starPressure.maximum(), kLeftPressure);
    EXPECT_LT(starPressure.range() / starPressure.mean(), 0.02)
        << "the star region should be close to a single plateau pressure";
    EXPECT_LT(starVelocity.range() / starVelocity.mean(), 0.02)
        << "and a single plateau velocity, the same continuity";

    // Gas flows from the high-pressure side into the low-pressure side.
    EXPECT_GT(fluid.velocity(kCellCount / 2), 0.0);
}

}  // namespace
