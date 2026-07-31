#pragma once

#include <Math/Scalar.hpp>
#include <Units/Acceleration.hpp>
#include <Units/Constants.hpp>
#include <Units/Energy.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Unit.hpp>

#include <cmath>

namespace ysq {

namespace dim {

/// R / M for a specific gas: energy per unit mass per unit temperature.
using SpecificGasConstant = Div<SpecificEnergy, Temperature>;

/// Power per unit area per unit temperature to the fourth: the Stefan-
/// Boltzmann constant's dimension.
using StefanBoltzmannConstant = Div<Power, Mul<Area, Raise<Temperature, 4>>>;

/// Length times temperature: Wien's displacement constant's dimension.
using WienConstant = Mul<Length, Temperature>;

}  // namespace dim

using SpecificGasConstant = Quantity<dim::SpecificGasConstant>;
using StefanBoltzmannConstant = Quantity<dim::StefanBoltzmannConstant>;
using WienConstant = Quantity<dim::WienConstant>;

namespace constants {

/// The Stefan-Boltzmann constant, sigma = 2 pi^5 k^4 / (15 h^3 c^2).
/// Computed from the SI-defining constants in Units/Constants.hpp rather
/// than typed independently, so it cannot drift from them: since the 2019
/// redefinition fixed h, k and c exactly, sigma is exact too, not measured.
inline constexpr StefanBoltzmannConstant stefanBoltzmann{
    2.0 * kPi<double> * kPi<double> * kPi<double> * kPi<double> * kPi<double> *
    boltzmannConstant.value() * boltzmannConstant.value() * boltzmannConstant.value() *
    boltzmannConstant.value() /
    (15.0 * planckConstant.value() * planckConstant.value() * planckConstant.value() *
     speedOfLight.value() * speedOfLight.value())};

/// Wien's displacement constant, b = (h c / k) / x, where x is the root of
/// x = 5 (1 - e^-x), the condition for Planck's law to peak. That equation
/// has no closed form, unlike sigma above, so unlike sigma this is
/// transcribed rather than computed here: x = 4.965114231744276..., and b
/// follows from it and the exact SI constants alone, so it is exact in the
/// same sense sigma is, not measured; the transcription is what cannot be
/// avoided, not an approximation.
inline constexpr WienConstant wienDisplacementConstant{2.897771955e-3};

}  // namespace constants

/// The ideal gas law, mass form: p = rho * R_specific * T, where
/// R_specific = R / M is particular to the gas (about 287 J/(kg K) for dry
/// air), the specific gas constant rather than the universal one, since
/// this module works in densities rather than moles.
[[nodiscard]] constexpr Pressure idealGasPressure(Density density,
                                                  SpecificGasConstant specificGasConstant,
                                                  Temperature temperature) noexcept {
    return density * specificGasConstant * temperature;
}

/// p V^gamma = const for a reversible adiabatic (isentropic) process: the
/// pressure after expanding or compressing an ideal gas from v1 to v2 with
/// no heat exchanged. gamma is the adiabatic index, the ratio of specific
/// heats (5/3 for a monatomic ideal gas, 7/5 for diatomic).
[[nodiscard]] inline Pressure adiabaticPressure(Pressure p1, Volume v1, Volume v2,
                                                double adiabaticIndex) {
    return p1 * std::pow(static_cast<double>(v1 / v2), adiabaticIndex);
}

/// The Stefan-Boltzmann law: the total power radiated by a sphere of
/// radius `radius` at uniform temperature `temperature`, treated as an
/// ideal black body.
[[nodiscard]] inline Power blackBodyLuminosity(Length radius, Temperature temperature) {
    const Area surfaceArea = 4.0 * kPi<double> * raised<2>(radius);
    return surfaceArea * constants::stefanBoltzmann * raised<4>(temperature);
}

/// Wien's displacement law: the wavelength at which a black body's
/// spectral radiance peaks, lambda_max = b / T.
[[nodiscard]] constexpr Length wienPeakWavelength(Temperature temperature) noexcept {
    return constants::wienDisplacementConstant / temperature;
}

/// The isothermal barometric scale height, H = R_specific T / g: how far an
/// atmosphere in hydrostatic equilibrium has to rise for its density to
/// fall by a factor of e. Follows directly from combining hydrostatic
/// equilibrium (dp/dh = -rho g) with idealGasPressure held at constant
/// T: R_specific T (drho/dh) = -rho g, i.e. drho/rho = -dh/H. General for
/// any gas and any surface gravity, not Earth's air specifically; Earth's
/// numbers (R_specific about 287 J/(kg K) for dry air, g about 9.8 m/s^2)
/// are scenario data, not part of this law.
[[nodiscard]] constexpr Length
isothermalScaleHeight(SpecificGasConstant specificGasConstant, Temperature temperature,
                      Acceleration surfaceGravity) noexcept {
    return specificGasConstant * temperature / surfaceGravity;
}

/// The isothermal barometric density profile, rho(h) = rho0 exp(-h /
/// scaleHeight), the exact consequence of isothermalScaleHeight's own
/// derivation: rho0 is the density at h = 0, and scaleHeight the same
/// quantity isothermalScaleHeight returns, though a caller is free to
/// supply either measured or derived directly, whichever a scenario
/// actually has.
[[nodiscard]] inline Density isothermalAtmosphereDensity(Density seaLevelDensity,
                                                         Length altitude,
                                                         Length scaleHeight) {
    return seaLevelDensity * std::exp(-(altitude / scaleHeight));
}

}  // namespace ysq
