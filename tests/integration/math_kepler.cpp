#include <Math/CoordinateSystems.hpp>
#include <Math/Integrators/RK4.hpp>
#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Statistics.hpp>
#include <Math/Vector3.hpp>
#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <vector>

/// Six Math headers driving one physical problem.
///
/// Each of them is tested on its own elsewhere. What this file asks is whether
/// they add up: whether an integrator, a vector type, a coordinate conversion
/// and a running accumulator together reproduce a two-body orbit and its
/// conserved quantities. That is the shape every application in this project
/// will eventually have, and it is the first point at which a mistake in how
/// the pieces meet can show up at all.
///
/// The physics here is deliberately the simplest case with a closed form: one
/// body about a fixed centre, with the gravitational parameter set to one, so
/// the period of an orbit of semi-major axis a is exactly 2 pi a^(3/2). None
/// of it belongs in the engine; a Kepler orbit is a scenario, and scenarios
/// live in Applications.

namespace {

using ysq::PhaseState;
using ysq::Vec3;

using Phase = PhaseState<Vec3>;

constexpr double kPi = ysq::kPi<double>;

/// Inverse-square attraction toward the origin, with the gravitational
/// parameter set to one.
constexpr auto acceleration = [](double, const Vec3& position) {
    const double radius = length(position);
    return position * (-1.0 / (radius * radius * radius));
};

double orbitalEnergy(const Phase& state) {
    return 0.5 * lengthSquared(state.velocity) - 1.0 / length(state.position);
}

Vec3 angularMomentum(const Phase& state) {
    return cross(state.position, state.velocity);
}

double periodFor(double semiMajorAxis) {
    return 2.0 * kPi * std::pow(semiMajorAxis, 1.5);
}

/// Starts at apoapsis on the +x axis, moving toward +y, so the orbit runs
/// anticlockwise in the xy plane and periapsis sits at -x.
Phase startAtApoapsis(double semiMajorAxis, double eccentricity) {
    const double apoapsis = semiMajorAxis * (1.0 + eccentricity);
    const double speed =
        std::sqrt((1.0 - eccentricity) / (semiMajorAxis * (1.0 + eccentricity)));
    return {Vec3{apoapsis, 0.0, 0.0}, Vec3{0.0, speed, 0.0}};
}

/// Peak absolute deviation of a quantity from its starting value, over the
/// first and second halves of a run.
struct HalfByHalf {
    double first = 0.0;
    double second = 0.0;
};

// --- Kepler's third law -----------------------------------------------------

/// The time at which the orbit next crosses the +x axis going anticlockwise,
/// which for a run started there is one full period.
///
/// Found by watching for the sign change in y and interpolating, rather than
/// by trusting a sample to land on it. That is the same thing an event
/// detector in the geodesic solver will have to do.
template <class Stepper, class System>
double measurePeriod(Stepper& stepper, const System& system, const Phase& start,
                     double until, double step) {
    double previousTime = 0.0;
    double previousY = 0.0;
    bool haveCrossing = false;
    double crossing = 0.0;

    ysq::integrate(stepper, system, start, 0.0, until, step,
                   [&](double time, const Phase& state) {
                       const double y = state.position.y;
                       if (!haveCrossing && time > 0.0 && previousY < 0.0 && y >= 0.0 &&
                           state.position.x > 0.0) {
                           crossing = previousTime + (time - previousTime) *
                                                         (-previousY) / (y - previousY);
                           haveCrossing = true;
                       }
                       previousTime = time;
                       previousY = y;
                   });

    return haveCrossing ? crossing : -1.0;
}

TEST(MathKepler, ThePeriodFollowsTheThreeHalvesPowerOfTheSemiMajorAxis) {
    // Kepler's third law. The engine is never told it; it comes out of
    // integrating an inverse-square force with a symplectic method.
    std::vector<double> axes;
    std::vector<double> measured;

    for (const double semiMajorAxis : {0.5, 1.0, 2.0, 3.0}) {
        const double expected = periodFor(semiMajorAxis);
        ysq::VelocityVerletStepper<Vec3> stepper;
        const double period =
            measurePeriod(stepper, acceleration, startAtApoapsis(semiMajorAxis, 0.3),
                          expected * 1.5, expected / 20000.0);

        ASSERT_GT(period, 0.0) << "no crossing found at a = " << semiMajorAxis;
        EXPECT_NEAR(period, expected, expected * 1e-6) << std::format(
            "a = {}, measured {:.9f}, expected {:.9f}", semiMajorAxis, period, expected);

        axes.push_back(semiMajorAxis);
        measured.push_back(period);
    }

    // T^2 / a^3 is the same constant for every orbit, and that constant is
    // 4 pi^2 when the gravitational parameter is one.
    ysq::RunningStatistics<double> ratio;
    for (std::size_t i = 0; i < axes.size(); ++i) {
        ratio.add(measured[i] * measured[i] / (axes[i] * axes[i] * axes[i]));
    }
    EXPECT_NEAR(ratio.mean(), 4.0 * kPi * kPi, 1e-4);
    EXPECT_LT(ratio.range(), 1e-4) << "the ratio must not depend on the orbit";

    // And the same law seen as a straight line: log T against log a has slope
    // three halves.
    std::vector<double> logAxes;
    std::vector<double> logPeriods;
    for (std::size_t i = 0; i < axes.size(); ++i) {
        logAxes.push_back(std::log(axes[i]));
        logPeriods.push_back(std::log(measured[i]));
    }
    const auto fit = ysq::linearFit(logAxes, logPeriods);
    EXPECT_NEAR(fit.slope, 1.5, 1e-5);
    EXPECT_NEAR(fit.rSquared, 1.0, 1e-9);
}

TEST(MathKepler, ACircularOrbitStaysCircularAtASlightlyShiftedRadius) {
    // A symplectic method does not conserve the energy of the system it was
    // given; it conserves that of a nearby one. On a circular orbit that shows
    // up as a circle of very slightly the wrong radius, held forever, rather
    // than as a spiral. So the thing to assert is that the orbit does not
    // breathe, and that the offset in the radius shrinks like the square of
    // the step, which is what makes it the method's order and not a defect.
    constexpr double radius = 1.5;
    const Phase start{Vec3{radius, 0.0, 0.0}, Vec3{0.0, std::sqrt(1.0 / radius), 0.0}};
    const double period = periodFor(radius);

    const auto offsetAtStep = [&](double step) {
        ysq::VelocityVerletStepper<Vec3> stepper;
        ysq::RunningStatistics<double> distances;
        ysq::integrate(
            stepper, acceleration, start, 0.0, 20.0 * period, step,
            [&](double, const Phase& state) { distances.add(length(state.position)); });
        EXPECT_LT(distances.range(), 1e-5)
            << "a circular orbit that breathes is not circular";
        return std::abs(distances.mean() - radius);
    };

    const double coarse = offsetAtStep(period / 2000.0);
    const double fine = offsetAtStep(period / 4000.0);

    EXPECT_LT(coarse, 1e-5);
    EXPECT_NEAR(std::log2(coarse / fine), 2.0, 0.15) << std::format(
        "offset {:.3e} then {:.3e}: it has to fall as the square of the step", coarse,
        fine);

    // And the orbit stays in the plane it started in.
    EXPECT_APPROX(angularMomentum(start).x, 0.0);
    EXPECT_APPROX(angularMomentum(start).y, 0.0);
}

// --- Conserved quantities ---------------------------------------------------

/// Runs an orbit and reports the worst deviation of a quantity in each half of
/// the run. A bounded error gives two similar numbers; a secular drift gives a
/// second that is clearly larger.
template <class Stepper, class System>
HalfByHalf driftHalves(Stepper& stepper, const System& system, const Phase& start,
                       double until, double step,
                       const std::function<double(const Phase&)>& quantity) {
    const double initial = quantity(start);
    HalfByHalf worst;

    ysq::integrate(stepper, system, start, 0.0, until, step,
                   [&](double time, const Phase& state) {
                       const double deviation = std::abs(quantity(state) - initial);
                       if (time < until / 2.0) {
                           worst.first = std::max(worst.first, deviation);
                       } else {
                           worst.second = std::max(worst.second, deviation);
                       }
                   });
    return worst;
}

TEST(MathKepler, SymplecticEnergyErrorStaysBoundedOverManyOrbits) {
    constexpr double eccentricity = 0.5;
    const double period = periodFor(1.0);
    const Phase start = startAtApoapsis(1.0, eccentricity);

    ysq::VelocityVerletStepper<Vec3> stepper;
    const HalfByHalf drift = driftHalves(stepper, acceleration, start, 200.0 * period,
                                         period / 2000.0, orbitalEnergy);

    // The energy error of a symplectic method is set by the step size, not by
    // how long the run has been going. The second hundred orbits are no worse
    // than the first.
    EXPECT_GT(drift.first, 0.0);
    EXPECT_LT(drift.second / drift.first, 1.05) << std::format(
        "first half {:.3e}, second half {:.3e}", drift.first, drift.second);
    EXPECT_LT(drift.second, 1e-4) << "and small in absolute terms as well";
}

TEST(MathKepler, RungeKuttaEnergyDriftsOverTheSameRun) {
    // The contrast that decides which method a long integration should use.
    // RK4 is far more accurate per step here, and it is still the wrong
    // choice, because its energy error accumulates rather than oscillating.
    constexpr double eccentricity = 0.5;
    const double period = periodFor(1.0);
    const Phase start = startAtApoapsis(1.0, eccentricity);

    ysq::Rk4Stepper<Phase> stepper;
    const HalfByHalf drift = driftHalves(stepper, ysq::asPhaseSystem(acceleration), start,
                                         200.0 * period, period / 2000.0, orbitalEnergy);

    EXPECT_GT(drift.second / drift.first, 1.5) << std::format(
        "first half {:.3e}, second half {:.3e}: RK4 energy error has to grow",
        drift.first, drift.second);
}

TEST(MathKepler, VelocityVerletConservesAngularMomentumStructurally) {
    // Not approximately: exactly, up to rounding. A drift leaves r x v
    // unchanged because v is parallel to itself, and a kick leaves it
    // unchanged because the force is parallel to r. Neither half of the method
    // can touch it, whatever the step size.
    constexpr double eccentricity = 0.7;
    const double period = periodFor(1.0);
    const Phase start = startAtApoapsis(1.0, eccentricity);
    const Vec3 initial = angularMomentum(start);

    ysq::VelocityVerletStepper<Vec3> stepper;
    ysq::RunningStatistics<double> magnitudes;
    double worstDirection = 0.0;

    ysq::integrate(stepper, acceleration, start, 0.0, 50.0 * period, period / 500.0,
                   [&](double, const Phase& state) {
                       const Vec3 current = angularMomentum(state);
                       magnitudes.add(length(current));
                       worstDirection =
                           std::max(worstDirection, length(cross(current, initial)));
                   });

    EXPECT_NEAR(magnitudes.mean(), length(initial), 1e-12);
    EXPECT_LT(magnitudes.range() / length(initial), 1e-12)
        << "a deliberately coarse step, and it still costs nothing";
    EXPECT_LT(worstDirection, 1e-12) << "the orbital plane is fixed too";

    // Which is more than the step size buys for the energy at the same
    // setting: that error is bounded, but it is nowhere near this small.
    ysq::VelocityVerletStepper<Vec3> comparison;
    const HalfByHalf energyDrift = driftHalves(
        comparison, acceleration, start, 50.0 * period, period / 500.0, orbitalEnergy);
    EXPECT_GT(energyDrift.second, 1e-6)
        << "angular momentum is conserved for a structural reason that does "
           "not apply to the energy";
}

// --- The orbit itself -------------------------------------------------------

TEST(MathKepler, TheIntegratedPathSatisfiesTheConicEquation) {
    // The orbit is not merely closed and energy-conserving; it is the right
    // shape. Read through the polar conversion, since that is the form the
    // conic equation is written in.
    constexpr double semiMajorAxis = 1.0;
    constexpr double eccentricity = 0.6;
    const double period = periodFor(semiMajorAxis);
    const double semiLatusRectum = semiMajorAxis * (1.0 - eccentricity * eccentricity);

    const auto worstResidualAtStep = [&](double step) {
        ysq::VelocityVerletStepper<Vec3> stepper;
        ysq::RunningStatistics<double> residuals;

        ysq::integrate(
            stepper, acceleration, startAtApoapsis(semiMajorAxis, eccentricity), 0.0,
            period, step, [&](double, const Phase& state) {
                const ysq::Polar<double> polar = ysq::toPolar(state.position.xy());
                // Periapsis sits at -x, so the conic is written about that axis.
                const double predicted =
                    semiLatusRectum / (1.0 - eccentricity * std::cos(polar.angle));
                residuals.add(std::abs(polar.radius - predicted));

                EXPECT_APPROX(state.position.z, 0.0);
            });
        return residuals.maximum();
    };

    const double coarse = worstResidualAtStep(period / 5000.0);
    const double fine = worstResidualAtStep(period / 10000.0);

    EXPECT_LT(coarse, 1e-4);
    // Second order, again: the path is the right conic in the limit, and how
    // fast it gets there is the property worth pinning rather than whatever
    // residual one particular step happens to give.
    EXPECT_NEAR(std::log2(coarse / fine), 2.0, 0.15)
        << std::format("worst radial residual {:.3e} then {:.3e}", coarse, fine);
}

TEST(MathKepler, TheOrbitReachesTheApsidesItWasBuiltFrom) {
    constexpr double semiMajorAxis = 1.0;
    constexpr double eccentricity = 0.6;
    const double period = periodFor(semiMajorAxis);

    ysq::VelocityVerletStepper<Vec3> stepper;
    ysq::RunningStatistics<double> radii;
    ysq::RunningStatistics<double> speeds;

    ysq::integrate(stepper, acceleration, startAtApoapsis(semiMajorAxis, eccentricity),
                   0.0, period, period / 20000.0, [&](double, const Phase& state) {
                       radii.add(length(state.position));
                       speeds.add(length(state.velocity));
                   });

    EXPECT_NEAR(radii.maximum(), semiMajorAxis * (1.0 + eccentricity), 1e-6);
    EXPECT_NEAR(radii.minimum(), semiMajorAxis * (1.0 - eccentricity), 1e-5);

    // The vis-viva relation at each apsis, which the integrator was never told.
    const double apoapsisSpeed =
        std::sqrt((1.0 - eccentricity) / (semiMajorAxis * (1.0 + eccentricity)));
    const double periapsisSpeed =
        std::sqrt((1.0 + eccentricity) / (semiMajorAxis * (1.0 - eccentricity)));
    EXPECT_NEAR(speeds.minimum(), apoapsisSpeed, 1e-6);
    EXPECT_NEAR(speeds.maximum(), periapsisSpeed, 1e-4);
}

}  // namespace
