#pragma once

#include <Physics/Thermodynamics/Thermodynamics.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Temperature.hpp>
#include <Units/Unit.hpp>

#include <cmath>

namespace ysq {

/// Radiative heat exchange between two finite surfaces: the generalization
/// of `Thermodynamics.hpp`'s Stefan-Boltzmann law (radiation into free
/// space -- a view factor of 1, a sink at absolute zero) to exchange
/// between two bodies each at their own finite temperature, only some of
/// each other's radiation reaching the other.

/// The net radiative power surface 1 (area `area1`, temperature
/// `temperature1`) exchanges with surface 2 (`temperature2`), given the
/// fraction `viewFactor` (`F_12`, `0` to `1`) of surface 1's own radiation
/// that reaches surface 2: `Q = sigma area1 F_12 (T1^4 - T2^4)`. Positive
/// means net power flows from surface 1 to surface 2. Reduces to
/// `blackBodyLuminosity`'s own formula exactly at `viewFactor = 1`,
/// `temperature2 = 0`: radiating into free space is the special case of
/// exchanging with a surface that intercepts everything and returns
/// nothing.
[[nodiscard]] constexpr Power netRadiativeExchange(Area area1, double viewFactor,
                                                   Temperature temperature1,
                                                   Temperature temperature2) noexcept {
    return constants::stefanBoltzmann * area1 * viewFactor *
           (raised<4>(temperature1) - raised<4>(temperature2));
}

/// The view factor between two coaxial, parallel circular disks of radius
/// `radius1`/`radius2`, separated by `distance`: one of the few surface
/// pairs with an exact closed-form view factor (Incropera et al.,
/// *Fundamentals of Heat and Mass Transfer*) rather than needing a
/// numerical double-surface integral, general for any two such disks
/// (a radiator and a target, a star and a nearby disk-shaped collector),
/// not a specific scenario's geometry.
///
/// ```
/// R_i = radius1 / distance,  R_j = radius2 / distance
/// S = 1 + (1 + R_j^2) / R_i^2
/// F_ij = (1/2) [S - sqrt(S^2 - 4 (R_j / R_i)^2)]
/// ```
///
/// Evaluated as `2x / (S + sqrt(S^2 - 4x))`, `x = (R_j / R_i)^2`, the
/// algebraically equivalent form reached by multiplying the textbook
/// expression above by its own conjugate: the textbook form subtracts two
/// nearly equal large numbers whenever the disks are far apart relative to
/// their radii (`S` large), losing most of the result's own precision to
/// cancellation exactly where the view factor itself is smallest and that
/// precision matters most; this form only ever divides, so it stays
/// accurate at any separation.
[[nodiscard]] inline double coaxialDiskViewFactor(Length radius1, Length radius2,
                                                  Length distance) {
    const double ri = radius1.value() / distance.value();
    const double rj = radius2.value() / distance.value();
    const double x = (rj / ri) * (rj / ri);

    const double s = 1.0 + (1.0 + rj * rj) / (ri * ri);
    return 2.0 * x / (s + std::sqrt(s * s - 4.0 * x));
}

}  // namespace ysq
