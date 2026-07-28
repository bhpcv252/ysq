#pragma once

#include <Physics/Body.hpp>
#include <Units/Electromagnetism.hpp>
#include <Units/Force.hpp>

namespace ysq {

/// The force on a charged body from electric and magnetic fields:
/// F = q (E + v x B).
[[nodiscard]] Force3 lorentzForce(const Body& body, const ElectricField3& electric,
                                  const MagneticFluxDensity3& magnetic);

}  // namespace ysq
