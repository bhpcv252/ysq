#include <Math/Integrators/Adaptive.hpp>
#include <Math/Integrators/Euler.hpp>
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
#include <limits>
#include <string>
#include <vector>

/// The test that decides whether this stage is finished.
///
/// A wrong Butcher tableau does not crash and does not obviously misbehave. It
/// produces a method that still converges, just more slowly than advertised,
/// and every result downstream is then quietly less accurate than the
/// simulation claims. Measuring the observed order is what catches that, and
/// it is the reason this file exists.

namespace {

using ysq::PhaseState;
using ysq::Vec3;

using ScalarPhase = PhaseState<double>;

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kPi = ysq::kPi<double>;

// --- Order measurement ------------------------------------------------------

/// Errors below this are rounding, not truncation, and the ratio between two
/// of them says nothing about the method. A hundred epsilons on a solution of
/// order one still leaves two clear digits of truncation signal.
constexpr double kRoundoffFloor = 100.0 * kEps;
/// Errors above this are outside the asymptotic regime, where the leading term
/// of the expansion does not yet dominate.
constexpr double kAsymptoticCeiling = 1e-3;

double medianOf(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

struct OrderResult {
    double order = 0.0;
    std::vector<double> errors;
    std::string report;
};

/// Doubles the step *count* `samples` times and reads the order off the error
/// ratios.
///
/// **By count, not by halving a step size.** The driver rounds its step down to
/// the nearest divisor of the interval so it can land exactly on the end, so
/// halving a step that does not already divide the span does not halve the step
/// actually taken: 0.4 and 0.3 over a unit interval both become 0.25. Feeding
/// such a sequence to a ratio test produces order estimates that jump around
/// between 1.7 and 7.5 with nothing wrong with the method at all. Asking for
/// N steps and then 2N makes the refinement exact.
///
/// The **median** of the consecutive ratios rather than a single pair or a
/// least-squares fit. One sample can still land badly, from FMA contraction on
/// one platform or from a step near a zero of the error term, and a median
/// absorbs that where a fit would drag the answer with it.
///
/// Every error is checked to be inside the window where truncation dominates.
/// Without that guard the measurement silently becomes a measurement of
/// rounding noise, which is the way an order test usually goes wrong.
OrderResult measureOrder(const std::function<double(double)>& errorAtStep,
                         double span, std::size_t initialSteps, int samples) {
    OrderResult result;
    std::size_t steps = initialSteps;

    for (int i = 0; i < samples; ++i) {
        result.errors.push_back(errorAtStep(span / static_cast<double>(steps)));
        steps *= 2;
    }

    std::vector<double> ratios;
    for (std::size_t i = 0; i + 1 < result.errors.size(); ++i) {
        ratios.push_back(std::log2(result.errors[i] / result.errors[i + 1]));
    }
    result.order = medianOf(ratios);

    result.report = std::format("from {} steps, doubling, errors:", initialSteps);
    for (const double error : result.errors) {
        result.report += std::format(" {:.3e}", error);
    }
    return result;
}

::testing::AssertionResult ordersMatch(const char* label, const OrderResult& measured,
                                       double expected) {
    for (const double error : measured.errors) {
        if (error < kRoundoffFloor) {
            return ::testing::AssertionFailure() << std::format(
                       "{}: error {:.3e} is below the rounding floor {:.3e}, so "
                       "the step range measures noise rather than truncation.\n{}",
                       label, error, kRoundoffFloor, measured.report);
        }
        if (error > kAsymptoticCeiling) {
            return ::testing::AssertionFailure() << std::format(
                       "{}: error {:.3e} is above {:.3e}, outside the asymptotic "
                       "regime.\n{}",
                       label, error, kAsymptoticCeiling, measured.report);
        }
    }

    if (std::abs(measured.order - expected) > 0.15) {
        return ::testing::AssertionFailure() << std::format(
                   "{}: observed order {:.3f}, expected {:.1f}.\n{}", label,
                   measured.order, expected, measured.report);
    }
    return ::testing::AssertionSuccess();
}

#define EXPECT_ORDER(label, measured, expected) \
    EXPECT_TRUE(ordersMatch(label, measured, expected))

// --- Test problems ----------------------------------------------------------

/// dy/dt = -y, y(0) = 1. Solution exp(-t).
constexpr auto decay = [](double, double y) { return -y; };
double decayExact(double t) {
    return std::exp(-t);
}

/// The logistic equation, dy/dt = y(1 - y), y(0) = 1/2. Nonlinear, smooth, and
/// with a closed form, so a method cannot pass by being exact on linear
/// problems alone.
constexpr auto logistic = [](double, double y) { return y * (1.0 - y); };
double logisticExact(double t) {
    return 1.0 / (1.0 + std::exp(-t));
}

/// q'' = -q with q(0) = 1, v(0) = 0. Solution (cos t, -sin t).
constexpr auto spring = [](double, double q) { return -q; };
ScalarPhase springExact(double t) {
    return {std::cos(t), -std::sin(t)};
}

double springEnergy(const ScalarPhase& state) {
    return 0.5 * (state.velocity * state.velocity + state.position * state.position);
}

template <class Stepper>
double scalarError(const std::function<double(double, double)>& system,
                   double start, double exactAtEnd, double until, double step) {
    Stepper stepper;
    const double computed = ysq::integrate(stepper, system, start, 0.0, until, step);
    return std::abs(computed - exactAtEnd);
}

template <class Stepper>
double springError(double until, double step) {
    Stepper stepper;
    const ScalarPhase computed =
        ysq::integrate(stepper, spring, ScalarPhase{1.0, 0.0}, 0.0, until, step);
    const ScalarPhase exact = springExact(until);
    return std::hypot(computed.position - exact.position,
                      computed.velocity - exact.velocity);
}

// --- Order, on a linear problem ---------------------------------------------

TEST(MathIntegrators, ExplicitMethodsHitTheirOrderOnALinearProblem) {
    const double exact = decayExact(1.0);

    EXPECT_ORDER("explicit Euler",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::ExplicitEulerStepper<double>>(
                             decay, 1.0, exact, 1.0, h);
                     },
                     1.0, 500, 5),
                 1.0);

    EXPECT_ORDER("midpoint",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::MidpointStepper<double>>(
                             decay, 1.0, exact, 1.0, h);
                     },
                     1.0, 20, 5),
                 2.0);

    EXPECT_ORDER("Heun",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::HeunStepper<double>>(decay, 1.0,
                                                                     exact, 1.0, h);
                     },
                     1.0, 20, 5),
                 2.0);

    EXPECT_ORDER("RK4",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::Rk4Stepper<double>>(decay, 1.0,
                                                                    exact, 1.0, h);
                     },
                     1.0, 8, 5),
                 4.0);

    EXPECT_ORDER("Dormand-Prince 5",
                 measureOrder(
                     [&](double h) {
                         return scalarError<
                             ysq::DormandPrince54Stepper<double>>(decay, 1.0, exact,
                                                                 1.0, h);
                     },
                     1.0, 8, 4),
                 5.0);
}

// --- Order, on a nonlinear problem ------------------------------------------

TEST(MathIntegrators, ExplicitMethodsHitTheirOrderOnANonlinearProblem) {
    // Exactness on dy/dt = -y is a weaker statement than it looks: several
    // wrong tableaux still integrate a pure exponential correctly. A nonlinear
    // right-hand side exercises the stage couplings.
    const double exact = logisticExact(4.0);

    EXPECT_ORDER("explicit Euler, logistic",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::ExplicitEulerStepper<double>>(
                             logistic, 0.5, exact, 4.0, h);
                     },
                     4.0, 512, 5),
                 1.0);

    EXPECT_ORDER("Heun, logistic",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::HeunStepper<double>>(
                             logistic, 0.5, exact, 4.0, h);
                     },
                     4.0, 64, 5),
                 2.0);

    EXPECT_ORDER("RK4, logistic",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::Rk4Stepper<double>>(logistic, 0.5,
                                                                    exact, 4.0, h);
                     },
                     4.0, 16, 5),
                 4.0);

    EXPECT_ORDER("Dormand-Prince 5, logistic",
                 measureOrder(
                     [&](double h) {
                         return scalarError<ysq::DormandPrince54Stepper<double>>(
                             logistic, 0.5, exact, 4.0, h);
                     },
                     4.0, 16, 4),
                 5.0);
}

// --- Order, of the symplectic methods ---------------------------------------

TEST(MathIntegrators, SymplecticMethodsHitTheirOrderOnTheOscillator) {
    // Measured on a separable second-order system, which is the only kind
    // these methods accept.
    EXPECT_ORDER("semi-implicit Euler",
                 measureOrder(
                     [](double h) {
                         return springError<ysq::SemiImplicitEulerStepper<double>>(
                             1.0, h);
                     },
                     1.0, 500, 5),
                 1.0);

    EXPECT_ORDER("velocity Verlet",
                 measureOrder(
                     [](double h) {
                         return springError<ysq::VelocityVerletStepper<double>>(1.0,
                                                                               h);
                     },
                     1.0, 20, 5),
                 2.0);

    EXPECT_ORDER("Forest-Ruth",
                 measureOrder(
                     [](double h) {
                         return springError<ysq::ForestRuthStepper<double>>(1.0, h);
                     },
                     1.0, 4, 5),
                 4.0);

    EXPECT_ORDER("PEFRL",
                 measureOrder(
                     [](double h) {
                         return springError<ysq::PefrlStepper<double>>(1.0, h);
                     },
                     1.0, 2, 5),
                 4.0);
}

TEST(MathIntegrators, PefrlIsMoreAccurateThanForestRuthAtTheSameOrder) {
    // Both are fourth order. PEFRL pays one more evaluation per step for a
    // considerably smaller error constant, which is why it exists.
    constexpr double step = 0.1;
    EXPECT_LT(springError<ysq::PefrlStepper<double>>(10.0, step),
              springError<ysq::ForestRuthStepper<double>>(10.0, step));
}

// --- What order does not capture --------------------------------------------

/// Peak absolute deviation of the energy from its initial value over a run.
template <class Stepper>
double worstEnergyDrift(std::size_t steps, double step) {
    Stepper stepper;
    const ScalarPhase start{1.0, 0.0};
    const double initial = springEnergy(start);

    double worst = 0.0;
    ysq::integrate(stepper, spring, start, 0.0, static_cast<double>(steps) * step,
                   step, [&](double, const ScalarPhase& state) {
                       worst = std::max(worst,
                                        std::abs(springEnergy(state) - initial));
                   });
    return worst;
}

/// The same, for an explicit method driven through the phase-space wrapper, so
/// the two are solving exactly the same problem.
template <class Stepper>
double worstEnergyDriftExplicit(std::size_t steps, double step) {
    Stepper stepper;
    const ScalarPhase start{1.0, 0.0};
    const double initial = springEnergy(start);
    const auto system = ysq::asPhaseSystem(spring);

    double worst = 0.0;
    ysq::integrate(stepper, system, start, 0.0, static_cast<double>(steps) * step,
                   step, [&](double, const ScalarPhase& state) {
                       worst = std::max(worst,
                                        std::abs(springEnergy(state) - initial));
                   });
    return worst;
}

TEST(MathIntegrators, SymplecticEnergyErrorIsBoundedWhereRungeKuttaDrifts) {
    // The property order says nothing about, and the reason a second-order
    // symplectic method beats a fourth-order explicit one over a long run.
    //
    // Stated as how the worst error grows when the run is made twice as long.
    // A bounded error does not care; a secular drift doubles. That is a
    // sharper statement than any single magnitude, and it does not depend on
    // the constants.
    constexpr double step = 0.1;
    constexpr std::size_t shortRun = 100'000;
    constexpr std::size_t longRun = 200'000;

    const double verletShort =
        worstEnergyDrift<ysq::VelocityVerletStepper<double>>(shortRun, step);
    const double verletLong =
        worstEnergyDrift<ysq::VelocityVerletStepper<double>>(longRun, step);

    const double rk4Short =
        worstEnergyDriftExplicit<ysq::Rk4Stepper<ScalarPhase>>(shortRun, step);
    const double rk4Long =
        worstEnergyDriftExplicit<ysq::Rk4Stepper<ScalarPhase>>(longRun, step);

    EXPECT_NEAR(verletLong / verletShort, 1.0, 0.01)
        << std::format("Verlet drift {:.3e} then {:.3e}: it must not grow",
                       verletShort, verletLong);

    EXPECT_GT(rk4Long / rk4Short, 1.8)
        << std::format("RK4 drift {:.3e} then {:.3e}: it must grow with time",
                       rk4Short, rk4Long);
    EXPECT_LT(rk4Long / rk4Short, 2.2) << "and roughly in proportion to it";
}

TEST(MathIntegrators, SymplecticEnergyErrorOscillatesRatherThanAccumulating) {
    // The same fact seen directly: over a long run the energy returns to where
    // it started, again and again, instead of settling away from it.
    ysq::VelocityVerletStepper<double> stepper;
    const ScalarPhase start{1.0, 0.0};
    const double initial = springEnergy(start);

    ysq::RunningStatistics<double> energy;
    ysq::integrate(stepper, spring, start, 0.0, 2000.0, 0.1,
                   [&](double, const ScalarPhase& state) {
                       energy.add(springEnergy(state) - initial);
                   });

    // Centred on zero, which a drifting series would not be.
    EXPECT_NEAR(energy.mean(), 0.0, 2e-3);
    EXPECT_LT(energy.range(), 1e-2);
    // And it genuinely varies rather than sitting still.
    EXPECT_GT(energy.range(), 1e-6);
}

TEST(MathIntegrators, ExplicitEulerGainsEnergyAndSemiImplicitEulerDoesNot) {
    // Two methods of the same order and the same cost. Explicit Euler is
    // unstable on an oscillator: its energy grows without bound however small
    // the step. Reordering two lines fixes that completely.
    constexpr double step = 0.01;
    constexpr double until = 200.0;
    const ScalarPhase start{1.0, 0.0};
    const double initial = springEnergy(start);

    ysq::ExplicitEulerStepper<ScalarPhase> explicitEuler;
    const ScalarPhase explicitEnd = ysq::integrate(
        explicitEuler, ysq::asPhaseSystem(spring), start, 0.0, until, step);

    ysq::SemiImplicitEulerStepper<double> semiImplicit;
    const ScalarPhase semiImplicitEnd =
        ysq::integrate(semiImplicit, spring, start, 0.0, until, step);

    EXPECT_GT(springEnergy(explicitEnd) / initial, 2.0)
        << "explicit Euler spirals outwards";
    EXPECT_NEAR(springEnergy(semiImplicitEnd) / initial, 1.0, 0.01)
        << "and the semi-implicit reordering does not";
}

// --- Adaptive stepping ------------------------------------------------------

TEST(MathIntegrators, TheAdaptiveMethodMeetsTheToleranceItIsGiven) {
    for (const double tolerance : {1e-6, 1e-8, 1e-10, 1e-12}) {
        ysq::DormandPrince54Stepper<double> stepper;
        ysq::AdaptiveSettings<double> settings;
        settings.absoluteTolerance = tolerance;
        settings.relativeTolerance = tolerance;

        const auto result =
            ysq::integrateAdaptive(stepper, logistic, 0.5, 0.0, 5.0, 0.1, settings);

        ASSERT_TRUE(result.succeeded) << "at tolerance " << tolerance;
        const double error = std::abs(result.state - logisticExact(5.0));
        // The controller works per step, so the accumulated error over the
        // whole run is allowed to be a modest multiple of the per-step ask.
        EXPECT_LT(error, tolerance * 100.0)
            << std::format("tolerance {:.0e}, error {:.3e}, {} steps", tolerance,
                           error, result.acceptedSteps);
    }
}

TEST(MathIntegrators, TighterTolerancesCostMoreStepsAndBuyMoreAccuracy) {
    double previousError = 1.0;
    std::size_t previousSteps = 0;

    for (const double tolerance : {1e-6, 1e-9, 1e-12}) {
        ysq::DormandPrince54Stepper<double> stepper;
        ysq::AdaptiveSettings<double> settings;
        settings.absoluteTolerance = tolerance;
        settings.relativeTolerance = tolerance;

        const auto result =
            ysq::integrateAdaptive(stepper, logistic, 0.5, 0.0, 5.0, 0.1, settings);
        ASSERT_TRUE(result.succeeded);

        const double error = std::abs(result.state - logisticExact(5.0));
        EXPECT_LT(error, previousError);
        EXPECT_GT(result.acceptedSteps, previousSteps);
        previousError = error;
        previousSteps = result.acceptedSteps;
    }
}

TEST(MathIntegrators, FirstSameAsLastSavesOneEvaluationPerStep) {
    // Dormand-Prince evaluates its seventh stage at the point the step lands
    // on, with exactly the weights of the propagated solution, so that stage
    // is the next step's first. Carrying it costs six evaluations per accepted
    // step instead of seven.
    ysq::DormandPrince54Stepper<double> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-8;
    settings.relativeTolerance = 1e-8;

    const auto result =
        ysq::integrateAdaptive(stepper, decay, 1.0, 0.0, 5.0, 0.1, settings);

    ASSERT_TRUE(result.succeeded);
    ASSERT_EQ(result.rejectedSteps, 0u) << "this problem should need no retries";
    ASSERT_GT(result.acceptedSteps, 5u);

    // Six per step, plus the very first stage, which has nothing to inherit.
    EXPECT_EQ(result.evaluations, 6 * result.acceptedSteps + 1)
        << std::format("{} evaluations over {} steps", result.evaluations,
                       result.acceptedSteps);
    EXPECT_LT(result.evaluations, 7 * result.acceptedSteps);
}

// --- Adaptive stepping on an eccentric orbit --------------------------------

namespace kepler {

/// Two-body acceleration about a fixed centre, with the gravitational
/// parameter set to one.
constexpr auto acceleration = [](double, const Vec3& position) {
    const double distance = length(position);
    return position * (-1.0 / (distance * distance * distance));
};

/// A highly eccentric orbit, started at apoapsis. Semi-major axis 1, so the
/// period is exactly 2 pi; eccentricity 0.9, so the speed at periapsis is
/// nineteen times the speed at apoapsis and a fixed step has to be sized for
/// the worst of it everywhere.
constexpr double kEccentricity = 0.9;
constexpr double kSemiMajorAxis = 1.0;

PhaseState<Vec3> start() {
    const double apoapsis = kSemiMajorAxis * (1.0 + kEccentricity);
    const double speed =
        std::sqrt((1.0 - kEccentricity) / (kSemiMajorAxis * (1.0 + kEccentricity)));
    return {Vec3{apoapsis, 0.0, 0.0}, Vec3{0.0, speed, 0.0}};
}

double period() {
    return 2.0 * kPi * std::sqrt(kSemiMajorAxis * kSemiMajorAxis * kSemiMajorAxis);
}

}  // namespace kepler

TEST(MathIntegrators, TheAdaptiveStepShrinksWhereTheOrbitIsFastest) {
    ysq::DormandPrince54Stepper<PhaseState<Vec3>> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-10;
    settings.relativeTolerance = 1e-10;

    std::vector<double> times;
    std::vector<double> radii;
    const auto result = ysq::integrateAdaptive(
        stepper, ysq::asPhaseSystem(kepler::acceleration), kepler::start(), 0.0,
        kepler::period(), 0.01, settings,
        [&](double t, const PhaseState<Vec3>& state) {
            times.push_back(t);
            radii.push_back(length(state.position));
        });

    ASSERT_TRUE(result.succeeded);
    ASSERT_GT(times.size(), 20u);

    // The step taken from each accepted point, paired with the radius there.
    double nearestStep = std::numeric_limits<double>::infinity();
    double farthestStep = 0.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        const double step = times[i] - times[i - 1];
        if (radii[i - 1] < kepler::kSemiMajorAxis * 0.3) {
            nearestStep = std::min(nearestStep, step);
        }
        if (radii[i - 1] > kepler::kSemiMajorAxis * 1.7) {
            farthestStep = std::max(farthestStep, step);
        }
    }

    EXPECT_LT(nearestStep * 10.0, farthestStep) << std::format(
        "step near periapsis {:.3e}, near apoapsis {:.3e}", nearestStep,
        farthestStep);
}

TEST(MathIntegrators, AdaptiveSteppingBeatsAFixedStepAtEqualCost) {
    // The comparison worth making is at equal work. An adaptive run spends its
    // evaluations where the orbit is changing fastest; a fixed step has to use
    // its finest spacing everywhere, and on an orbit with a nineteen-to-one
    // speed ratio most of that is wasted out near apoapsis.
    ysq::DormandPrince54Stepper<PhaseState<Vec3>> adaptiveStepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-9;
    settings.relativeTolerance = 1e-9;

    const auto system = ysq::asPhaseSystem(kepler::acceleration);
    const auto adaptive = ysq::integrateAdaptive(
        adaptiveStepper, system, kepler::start(), 0.0, kepler::period(), 0.01,
        settings);
    ASSERT_TRUE(adaptive.succeeded);

    // The orbit is closed, so after exactly one period the exact answer is the
    // state it started from.
    const double adaptiveError =
        distance(adaptive.state.position, kepler::start().position);

    // Fixed-step RK4 with the same number of right-hand side evaluations.
    const std::size_t budget = adaptive.evaluations / 4;
    ysq::Rk4Stepper<PhaseState<Vec3>> fixedStepper;
    const auto fixed =
        ysq::integrate(fixedStepper, system, kepler::start(), 0.0, kepler::period(),
                       kepler::period() / static_cast<double>(budget));
    const double fixedError = distance(fixed.position, kepler::start().position);

    EXPECT_LE(fixedStepper.evaluations(), adaptive.evaluations + 8u)
        << "the budgets have to match for the comparison to mean anything";
    EXPECT_LT(adaptiveError, fixedError) << std::format(
        "adaptive {:.3e} vs fixed {:.3e} over about {} evaluations", adaptiveError,
        fixedError, adaptive.evaluations);
}

TEST(MathIntegrators, TheControllerRecoversFromRejectedSteps) {
    // Started with a step far too large for periapsis, so the controller has
    // to reject and shrink before it can make progress.
    ysq::DormandPrince54Stepper<PhaseState<Vec3>> stepper;
    ysq::AdaptiveSettings<double> settings;
    settings.absoluteTolerance = 1e-11;
    settings.relativeTolerance = 1e-11;

    const auto result = ysq::integrateAdaptive(
        stepper, ysq::asPhaseSystem(kepler::acceleration), kepler::start(), 0.0,
        kepler::period(), 1.0, settings);

    EXPECT_TRUE(result.succeeded);
    EXPECT_GT(result.rejectedSteps, 0u) << "an over-large first step must be refused";
    EXPECT_LT(result.rejectedSteps, result.acceptedSteps)
        << "but rejections must not dominate the run";
    EXPECT_LT(distance(result.state.position, kepler::start().position), 1e-6);
}

// --- Single precision -------------------------------------------------------

TEST(MathIntegrators, TheMethodsRunAtSinglePrecision) {
    // Not a formality: a step size or a tableau coefficient that only works at
    // double precision would pass everything above.
    ysq::Rk4Stepper<float> rk4;
    const auto decayFloat = [](float, float y) { return -y; };
    EXPECT_NEAR(ysq::integrate(rk4, decayFloat, 1.0f, 0.0f, 1.0f, 0.01f),
                std::exp(-1.0f), 1e-6f);

    ysq::VelocityVerletStepper<float> verlet;
    const auto springFloat = [](float, float q) { return -q; };
    const auto end = ysq::integrate(verlet, springFloat,
                                    PhaseState<float>{1.0f, 0.0f}, 0.0f, 1.0f, 0.01f);
    EXPECT_NEAR(end.position, std::cos(1.0f), 1e-4f);

    ysq::DormandPrince54Stepper<float> dormandPrince;
    ysq::AdaptiveSettings<float> settings;
    const auto result = ysq::integrateAdaptive(dormandPrince, decayFloat, 1.0f, 0.0f,
                                               1.0f, 0.1f, settings);
    EXPECT_TRUE(result.succeeded);
    EXPECT_NEAR(result.state, std::exp(-1.0f), 1e-4f);
}

}  // namespace
