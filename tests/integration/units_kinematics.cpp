/// Units and the Math integrators on one problem.
///
/// Each is tested on its own elsewhere. What this file asks is whether they
/// meet properly, because they do not meet automatically: a dimensioned state
/// cannot be handed to a Runge-Kutta stepper, and the reason is structural
/// rather than a missing overload. Rk4Stepper declares `State m_k1{}`, so the
/// derivative is required to be the same type as the state; for a position
/// that derivative is a velocity, which is a different type by construction.
/// The step size has the same problem: `Scalar` is StateScalarT<State>, taken
/// from the state's own value type, so it is a bare double where the physics
/// says it is a time.
///
/// The shape that follows, and the one every Physics module will use: set up
/// and verify in quantities, cross to raw values at the integrator, and make
/// the crossing explicit rather than incidental.

#include <Math/Integrators/Symplectic.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Units/Acceleration.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <support/MathApprox.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

namespace {

using namespace ysq;
using namespace ysq::literals;

/// Specific orbital energy, v^2/2 - GM/r. An energy per unit mass, so the
/// mass of the orbiting body never enters, which is the whole point of the
/// quantity having its own name.
using SpecificEnergy = Quantity<dim::SpecificEnergy>;

constexpr GravitationalParameter kSunGM = constants::nominalSolarMassParameter;
constexpr Length kOrbitRadius = units::astronomicalUnit;

/// The circular orbit speed at a given radius, sqrt(GM/r).
///
/// Every step of this type checks, and the sqrt is only reachable because
/// GM/r has dimension L^2 T^-2, whose exponents are both even. That is not a
/// coincidence: a speed squared is what the expression means.
[[nodiscard]] Speed circularSpeed(GravitationalParameter gm, Length radius) {
    return sqrt(gm / radius);
}

/// Kepler's third law, T = 2 pi sqrt(a^3 / GM).
[[nodiscard]] Time orbitalPeriod(GravitationalParameter gm, Length semiMajorAxis) {
    return sqrt(raised<3>(semiMajorAxis) / gm) * 2.0 * kPi<double>;
}

}  // namespace

// ---------------------------------------------------------------------------
// The boundary
// ---------------------------------------------------------------------------

TEST(UnitsKinematics, CrossingToRawValuesAndBackIsExact) {
    const Length3 position{Vec3{1.5e11, -2.5e10, 7.0e9}};
    const Speed speed{29780.123456789};

    // Bit-identical, not approximately equal. The boundary is a change of
    // static type and nothing else; if it ever costs a rounding step, every
    // conservation test downstream inherits the error.
    EXPECT_EQ(Length3{position.value()}, position);
    EXPECT_EQ(Speed{speed.value()}, speed);
    EXPECT_EQ(position.value().x, 1.5e11);
    EXPECT_EQ(speed.in(units::metrePerSecond), speed.value());
}

TEST(UnitsKinematics, ADimensionedStateIsAVectorSpaceButNotADimensionedSystem) {
    // The vector-space half is satisfied: a position adds to a position and
    // scales by a number, which is all Runge-Kutta asks of a state.
    static_assert(OdeState<Length3>);
    static_assert(OdeState<Velocity3>);

    // The step size is where it starts going wrong. Math derives it from the
    // state's own value type, so it is a double, not a Time.
    static_assert(std::is_same_v<StateScalarT<Length3>, double>);
    static_assert(!std::is_same_v<StateScalarT<Length3>, Time>);

    // And this is where it stops. A dimensionally honest derivative of a
    // position is a velocity, and OdeSystem requires the derivative to be
    // convertible back to the state.
    const auto honestDerivative = [](double, const Length3&) {
        return Velocity3{Vec3::zero()};
    };
    static_assert(!OdeSystem<decltype(honestDerivative), Length3>);

    // Which leaves the raw form, and that is why Physics converts at the
    // boundary rather than integrating quantities directly.
    const auto rawDerivative = [](double, const Vec3&) { return Vec3::zero(); };
    static_assert(OdeSystem<decltype(rawDerivative), Vec3>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Constant acceleration, where the answer is known exactly
// ---------------------------------------------------------------------------

TEST(UnitsKinematics, ProjectileMatchesTheAnalyticSolution) {
    const Length3 start{Vec3{0.0, 0.0, 0.0}};
    const Velocity3 launch{Vec3{30.0, 40.0, 0.0}};
    const Acceleration3 gravity{Vec3{0.0, -units::standardGravity.value(), 0.0}};
    const Time flight = 4.0 * units::second;

    // Set up in quantities, cross once, integrate, cross back.
    VelocityVerletStepper<Vec3> stepper;
    const auto acceleration = [&](double, const Vec3&) { return gravity.value(); };

    const PhaseState<Vec3> initial{start.value(), launch.value()};
    const PhaseState<Vec3> final = integrate(stepper, acceleration, initial, 0.0,
                                             flight.value(), 1.0 / 4096.0);

    const Length3 position{final.position};
    const Velocity3 velocity{final.velocity};

    // q = q0 + v0 t + a t^2 / 2, and every term of that type checks.
    const Length3 expectedPosition =
        start + launch * flight + gravity * raised<2>(flight) * 0.5;
    const Velocity3 expectedVelocity = launch + gravity * flight;

    // Velocity Verlet is exact for a constant acceleration, so the tolerance
    // here is accumulated rounding over 16384 steps and nothing else.
    EXPECT_QUANTITY_VEC_NEAR(position, expectedPosition, 1.0e-9 * units::metre);
    EXPECT_QUANTITY_VEC_NEAR(velocity, expectedVelocity,
                             1.0e-9 * units::metrePerSecond);

    // Apex height, by a route that never touches a raw number: v0y^2 / 2g.
    const Length apex =
        raised<2>(Speed{launch.value().y}) / (2.0 * units::standardGravity);
    EXPECT_QUANTITY_NEAR(apex, 81.5773 * units::metre, 1.0e-3 * units::metre);
}

// ---------------------------------------------------------------------------
// A circular orbit, where the laws are written entirely in quantities
// ---------------------------------------------------------------------------

TEST(UnitsKinematics, KeplersThirdLawTypeChecksAndGivesTheYear) {
    const Time period = orbitalPeriod(kSunGM, kOrbitRadius);

    // One astronomical unit around the nominal Sun is a year, to within the
    // difference between the sidereal and Julian years.
    EXPECT_QUANTITY_NEAR(period, 1.0_yr, 0.01_yr);
    EXPECT_GT(period.in(units::day), 365.0);
    EXPECT_LT(period.in(units::day), 366.0);

    const Speed speed = circularSpeed(kSunGM, kOrbitRadius);
    EXPECT_QUANTITY_NEAR(speed, 29.78 * units::kilometrePerSecond,
                         0.01 * units::kilometrePerSecond);

    // The two are tied together by the geometry of a circle, and this is the
    // check that they were not each fitted to a remembered number.
    EXPECT_QUANTITY_NEAR(speed * period, 2.0 * kPi<double> * kOrbitRadius,
                         1.0 * units::metre);
}

TEST(UnitsKinematics, AnOrbitIntegratedForItsPeriodReturnsToItsStart) {
    const Time period = orbitalPeriod(kSunGM, kOrbitRadius);
    const Speed speed = circularSpeed(kSunGM, kOrbitRadius);

    const Length3 start = kOrbitRadius * Vec3::unitX();
    const Velocity3 launch = speed * Vec3::unitY();

    // The law, stated once in quantities, then evaluated on raw values inside
    // the integrator's inner loop where the dimensions are already checked.
    const double gm = kSunGM.value();
    const auto acceleration = [gm](double, const Vec3& q) {
        const double r = length(q);
        return q * (-gm / (r * r * r));
    };

    VelocityVerletStepper<Vec3> stepper;
    const PhaseState<Vec3> initial{start.value(), launch.value()};
    const PhaseState<Vec3> final = integrate(
        stepper, acceleration, initial, 0.0, period.value(), period.value() / 20000.0);

    const Length3 position{final.position};
    const Velocity3 velocity{final.velocity};

    // Back where it started, to a part in ten thousand of the orbit radius.
    EXPECT_QUANTITY_NEAR(distance(position, start), Length{0.0},
                         1.0e-4 * kOrbitRadius);
    EXPECT_QUANTITY_NEAR(length(position), kOrbitRadius, 1.0e-6 * kOrbitRadius);
    EXPECT_QUANTITY_NEAR(length(velocity), speed, 1.0e-6 * speed);
}

TEST(UnitsKinematics, SpecificOrbitalEnergyIsConservedAndHasTheRightValue) {
    const Time period = orbitalPeriod(kSunGM, kOrbitRadius);
    const Speed speed = circularSpeed(kSunGM, kOrbitRadius);

    const Length3 start = kOrbitRadius * Vec3::unitX();
    const Velocity3 launch = speed * Vec3::unitY();

    const auto specificEnergy = [](const Length3& q, const Velocity3& v) {
        return raised<2>(length(v)) * 0.5 - kSunGM / length(q);
    };

    // For a circular orbit the specific energy is -GM / 2a, exactly.
    const SpecificEnergy expected = -kSunGM / (2.0 * kOrbitRadius);
    static_assert(std::is_same_v<decltype(specificEnergy(start, launch)),
                                 SpecificEnergy>);
    EXPECT_QUANTITY_NEAR(specificEnergy(start, launch), expected,
                         abs(expected) * 1.0e-12);

    const double gm = kSunGM.value();
    const auto acceleration = [gm](double, const Vec3& q) {
        const double r = length(q);
        return q * (-gm / (r * r * r));
    };

    VelocityVerletStepper<Vec3> stepper;
    const PhaseState<Vec3> initial{start.value(), launch.value()};
    const PhaseState<Vec3> final = integrate(
        stepper, acceleration, initial, 0.0, period.value(), period.value() / 20000.0);

    // A symplectic method holds this bounded rather than letting it drift,
    // which is the property the whole integrator choice rests on.
    const SpecificEnergy after =
        specificEnergy(Length3{final.position}, Velocity3{final.velocity});
    EXPECT_QUANTITY_NEAR(after, expected, abs(expected) * 1.0e-9);
}
