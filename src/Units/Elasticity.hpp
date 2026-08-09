#pragma once

#include <Units/Force.hpp>
#include <Units/Unit.hpp>

namespace ysq {

/// Stress and pressure share a dimension, both being a force per unit
/// area, so these are two names for one type -- the same limitation
/// `Force.hpp`'s `Torque`/`Energy` and `Temperature.hpp`'s
/// `Entropy`/`HeatCapacity` already have; see `src/Units/README.md`.
using Stress = Quantity<dim::Pressure>;

/// Strain -- a change in length over a length -- is a ratio of two lengths
/// and so is dimensionless, the same standing every other dimensionless
/// ratio in this module has (a Lorentz factor, a refractive index): named
/// here only so code that means a strain can say so.
using Strain = Dimensionless;

}  // namespace ysq
