#pragma once

#include <Physics/Mechanics/Spring.hpp>
#include <Units/Elasticity.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>

namespace ysq {

/// Linear elasticity: Hooke's law for a continuum, the generalization of
/// `Mechanics/Spring.hpp`'s point-mass version to a distributed elastic
/// medium (a rod, a beam, anything that stretches proportionally to load
/// rather than concentrating all its compliance at one point).

/// Hooke's law for a continuum: `stress = E * strain`, `E` the material's
/// own Young's modulus.
[[nodiscard]] constexpr Stress hookeStress(Stress youngsModulus, Strain strain) noexcept {
    return youngsModulus * strain;
}

/// The axial extension of a rod of natural length `naturalLength` and
/// cross-sectional area `crossSection`, made of a material with Young's
/// modulus `youngsModulus`, under an axial `load`: the direct consequence
/// of Hooke's law, `stress = F / A`, `strain = extension / L0`,
/// `sigma = E epsilon`, solved for `extension`.
[[nodiscard]] constexpr Length axialExtension(Force load, Length naturalLength,
                                              Area crossSection,
                                              Stress youngsModulus) noexcept {
    return load * naturalLength / (crossSection * youngsModulus);
}

/// The spring constant of a discretized rod segment: `k = E A / L0`, the
/// standard finite-element "bar element" stiffness. This is the bridge
/// between this header and `Mechanics/Spring.hpp`'s `SpringConstant` --
/// why that type lives where it does, generalized here from one spring to
/// a whole elastic continuum divided into segments, each behaving as a
/// spring of this stiffness.
[[nodiscard]] constexpr SpringConstant
equivalentSpringConstant(Stress youngsModulus, Area crossSection,
                         Length naturalLength) noexcept {
    return youngsModulus * crossSection / naturalLength;
}

}  // namespace ysq
