#pragma once

#include <Units/Length.hpp>
#include <Units/Mass.hpp>
#include <Units/Unit.hpp>

namespace ysq {

namespace dim {

/// Amount of substance per unit volume: how much of something is dissolved
/// or mixed into a given space, independent of what that space's total
/// volume happens to be.
using Concentration = Div<Amount, Volume>;

/// Mass per unit amount: the conversion factor between "how many" (moles)
/// and "how much" (kilograms) for a specific substance, since the two are
/// not interchangeable without knowing which substance is meant.
using MolarMass = Div<Mass, Amount>;

}  // namespace dim

/// The seventh SI base quantity, alongside `Length`, `Mass`, `Time`,
/// `Current`, `Temperature` and `LuminousIntensity` in `Unit.hpp`. Not
/// declared there directly with the other six only because nothing in this
/// engine needed it as its own named type until a chemistry-adjacent
/// quantity (`Concentration`, `MolarMass`) did; `Units/Constants.hpp`'s
/// `avogadroConstant` has used the bare `dim::Amount`/`dim::InverseAmount`
/// dimensions since before this alias existed, and continues to.
using AmountOfSubstance = Quantity<dim::Amount>;

using Concentration = Quantity<dim::Concentration>;
using MolarMass = Quantity<dim::MolarMass>;

namespace units {

inline constexpr AmountOfSubstance mole{1.0};
inline constexpr Concentration molePerCubicMetre{1.0};
inline constexpr MolarMass kilogramPerMole{1.0};

}  // namespace units

}  // namespace ysq
