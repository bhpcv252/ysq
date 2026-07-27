#pragma once

#include <Math/Scalar.hpp>
#include <Units/Energy.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Mass.hpp>
#include <Units/Temperature.hpp>
#include <Units/Time.hpp>
#include <Units/Unit.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

namespace dim {

/// Electric charge, current times time. The rest of the electromagnetic family
/// lands with Physics/Electromagnetism; this one is here because the SI cannot
/// be defined without it.
using ElectricCharge = Mul<Current, Time>;

/// A gravitational parameter, GM. Fixed exactly by convention for the Sun and
/// the Earth, unlike either mass on its own.
using GravitationalParameter = Div<Raise<Length, 3>, Raise<Time, 2>>;

/// The Avogadro constant's dimension, and luminous efficacy's.
using InverseAmount = Inverse<Amount>;
using LuminousEfficacy = Div<LuminousIntensity, Power>;

}  // namespace dim

using ElectricCharge = Quantity<dim::ElectricCharge>;
using GravitationalParameter = Quantity<dim::GravitationalParameter>;
using LuminousEfficacy = Quantity<dim::LuminousEfficacy>;

/// The constants that define the SI, and nothing else.
///
/// Since the 2019 redefinition the SI is not a set of artefacts and
/// measurements but a set of seven exactly fixed constants of nature; every
/// base unit falls out of them. That is why these live in Units rather than in
/// Physics: they are not facts the simulation discovers, they are the
/// definition of the vocabulary it speaks.
///
/// The line is drawn at "definitional", not at "fundamental". The Newtonian
/// constant of gravitation is as fundamental as anything here and is not in
/// this file: it is measured, it has a relative uncertainty around 2e-5, and
/// it parameterizes a specific interaction. It belongs with the gravity that
/// uses it, in Physics.
namespace constants {

/// The unperturbed ground-state hyperfine transition frequency of caesium-133.
/// Fixes the second.
inline constexpr Frequency caesiumHyperfineFrequency{9192631770.0};

/// The speed of light in vacuum. Fixes the metre, given the second.
inline constexpr Speed speedOfLight{299792458.0};

/// The Planck constant. Fixes the kilogram, given the metre and the second.
inline constexpr Action planckConstant{6.62607015e-34};

/// The elementary charge. Fixes the ampere.
inline constexpr ElectricCharge elementaryCharge{1.602176634e-19};

/// The Boltzmann constant. Fixes the kelvin.
inline constexpr HeatCapacity boltzmannConstant{1.380649e-23};

/// The Avogadro constant. Fixes the mole.
inline constexpr Quantity<dim::InverseAmount> avogadroConstant{6.02214076e23};

/// The luminous efficacy of monochromatic radiation of frequency 540 THz.
/// Fixes the candela.
inline constexpr LuminousEfficacy luminousEfficacy{683.0};

/// The reduced Planck constant. Exact, being the Planck constant over a
/// mathematical constant, and the form quantum mechanics is actually written
/// in.
///
/// Derived from planckConstant rather than typed out, so the two cannot drift
/// apart, and through Math's kPi rather than a pi literal of its own, so there
/// is one pi in the project.
inline constexpr Action reducedPlanckConstant{planckConstant.value() /
                                              (2.0 * kPi<double>)};

/// IAU 2015 Resolution B3 nominal mass parameters, exact by convention.
///
/// These, not the masses in kilograms, are what an orbit should be integrated
/// with. GM is measured directly and to high precision from orbits; splitting
/// it into a mass requires dividing by a measured G and throws away four
/// significant figures for nothing. units::solarMass exists because
/// applications ask for kilograms, and it is the lossy form.
inline constexpr GravitationalParameter nominalSolarMassParameter{1.32712440e20};
inline constexpr GravitationalParameter nominalEarthMassParameter{3.986004e14};

}  // namespace constants

}  // namespace ysq
