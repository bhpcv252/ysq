#pragma once

#include <Math/Scalar.hpp>
#include <Math/SpecialFunctions.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

#include <cmath>

namespace ysq {

/// The Maxwell-Boltzmann speed distribution: the probability density over
/// molecular speed for particles of mass `particleMass` in thermal
/// equilibrium at `temperature`. General for any ideal gas -- nothing here
/// is specific to one gas or one scenario, the same standing
/// `Thermodynamics.hpp`'s ideal gas law and adiabatic relation already have.

/// The distribution's own scale parameter, `a = sqrt(kT/m)`: the speed at
/// which the exponential factor `exp(-v^2 / (2 a^2))` falls to `1/e`, and
/// the unit every closed-form speed statistic below is expressed as a
/// multiple of.
[[nodiscard]] inline Speed maxwellBoltzmannScale(Mass particleMass,
                                                 Temperature temperature) {
    return sqrt(constants::boltzmannConstant * temperature / particleMass);
}

/// The probability density `f(v)`: probability per unit speed that a
/// randomly chosen particle has speed in `[v, v + dv)`,
/// `f(v) = sqrt(2/pi) (v/a)^2 exp(-v^2 / (2a^2)) / a`.
[[nodiscard]] inline double maxwellBoltzmannSpeedDensity(Speed v, Mass particleMass,
                                                         Temperature temperature) {
    const Speed a = maxwellBoltzmannScale(particleMass, temperature);
    const double x = v.value() / a.value();
    return detail::sqrtOf(2.0 / kPi<double>) * x * x * std::exp(-x * x / 2.0) / a.value();
}

/// The cumulative distribution, `P(speed <= v)`, via the closed form
/// `erf(v / (sqrt(2) a)) - sqrt(2/pi) (v/a) exp(-v^2 / (2a^2))`
/// (`Math/SpecialFunctions.hpp`'s `erf`): the standard result for a
/// chi distribution with three degrees of freedom, which the Maxwell
/// speed distribution is (speed is the magnitude of a 3D
/// normally-distributed velocity).
[[nodiscard]] inline double maxwellBoltzmannSpeedCdf(Speed v, Mass particleMass,
                                                     Temperature temperature) {
    const Speed a = maxwellBoltzmannScale(particleMass, temperature);
    const double x = v.value() / a.value();
    return erf(x / std::sqrt(2.0)) -
           detail::sqrtOf(2.0 / kPi<double>) * x * std::exp(-x * x / 2.0);
}

/// The `n`-th raw moment `<v^n>`, via the gamma function
/// (`Math/SpecialFunctions.hpp`'s `gamma`):
///
/// ```
/// <v^n> = (2^(n/2 + 1) / sqrt(pi)) a^n Gamma((n + 3) / 2)
/// ```
///
/// the general form every closed-form speed statistic below is a special
/// case of (`n = 1` is the mean speed, `n = 2` the mean-square speed).
/// Returned as a raw `double` (in `(m/s)^n`) rather than a `Quantity`,
/// since a single dimension cannot parameterize an arbitrary power `n`
/// known only at run time.
[[nodiscard]] inline double maxwellBoltzmannMoment(unsigned n, Mass particleMass,
                                                   Temperature temperature) {
    const Speed a = maxwellBoltzmannScale(particleMass, temperature);
    const double nAsDouble = static_cast<double>(n);
    const double exponent = nAsDouble / 2.0 + 1.0;
    return (std::pow(2.0, exponent) / std::sqrt(kPi<double>)) *
           std::pow(a.value(), nAsDouble) * gamma((nAsDouble + 3.0) / 2.0);
}

/// The most probable speed, `sqrt(2 k T / m) = a sqrt(2)`: the density's
/// own peak, not a moment (the mode of a distribution and its mean
/// coincide only for a symmetric one, which this, skewed toward high
/// speed by the `v^2` factor, is not).
[[nodiscard]] inline Speed maxwellBoltzmannMostProbableSpeed(Mass particleMass,
                                                             Temperature temperature) {
    return maxwellBoltzmannScale(particleMass, temperature) * std::sqrt(2.0);
}

/// The mean speed, `<v> = sqrt(8 k T / (pi m))`: `maxwellBoltzmannMoment(1, ...)`.
[[nodiscard]] inline Speed maxwellBoltzmannMeanSpeed(Mass particleMass,
                                                     Temperature temperature) {
    return Speed{maxwellBoltzmannMoment(1, particleMass, temperature)};
}

/// The root-mean-square speed, `sqrt(<v^2>) = sqrt(3 k T / m)`:
/// `sqrt(maxwellBoltzmannMoment(2, ...))`, the speed whose corresponding
/// kinetic energy `(1/2) m v_rms^2 = (3/2) k T` is the equipartition
/// result for three translational degrees of freedom.
[[nodiscard]] inline Speed maxwellBoltzmannRmsSpeed(Mass particleMass,
                                                    Temperature temperature) {
    return Speed{std::sqrt(maxwellBoltzmannMoment(2, particleMass, temperature))};
}

}  // namespace ysq
