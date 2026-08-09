#pragma once

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Units/Constants.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Unit.hpp>

namespace ysq {

/// The force a body feels from the momentum light itself carries: the
/// piece that turns `Optics/Illumination.hpp`'s visibility and irradiance
/// computation into an actual acceleration, rather than only a color and a
/// geometric fact about what is lit. Nothing here decides how much of a
/// source is visible or what color it is -- that is `Illumination.hpp`'s
/// job -- only what a known irradiance, once it reaches a surface, does to
/// it.

/// The irradiance at `distance` from an isotropic point source of total
/// radiant power `luminosity`: the inverse-square law, `luminosity` spread
/// evenly over the sphere of that radius.
[[nodiscard]] inline Irradiance irradianceFromPointSource(RadiantPower luminosity,
                                                          Length distance) {
    return luminosity / (Dimensionless{4.0 * kPi<double>} * raised<2>(distance));
}

/// The force of radiation pressure on a surface of `crossSectionalArea`
/// facing the source, under irradiance `irradiance`: momentum flux
/// (irradiance divided by `c`) times area times a radiation-pressure
/// coefficient `Cr` -- 1 for a perfect absorber (every photon's momentum
/// transferred once), up to 2 for a perfect reflector (reversing each
/// photon's momentum transfers it twice) -- pushing the surface directly
/// away from the source along `directionFromSource` (a unit vector,
/// pointing from the source toward the body).
[[nodiscard]] inline Force3 radiationPressureForce(Irradiance irradiance,
                                                   Area crossSectionalArea,
                                                   double radiationPressureCoefficient,
                                                   const Vec3& directionFromSource) {
    const Force magnitude =
        (irradiance * crossSectionalArea * radiationPressureCoefficient) /
        constants::speedOfLight;
    return magnitude * directionFromSource;
}

/// The common case assembled from both pieces above: a spherical body of
/// `bodyRadius`, facing an isotropic point source of `luminosity` at
/// `sourcePosition` -- the "cannonball" model standard in astrodynamics,
/// where the body's real cross-section facing the source is approximated
/// by its own silhouette, `pi bodyRadius^2`, regardless of attitude. Zero
/// if `bodyPosition` coincides with `sourcePosition`, where the direction
/// to push along is not defined.
[[nodiscard]] inline Force3 radiationPressureForce(RadiantPower luminosity,
                                                   const Length3& sourcePosition,
                                                   const Length3& bodyPosition,
                                                   Length bodyRadius,
                                                   double radiationPressureCoefficient) {
    const Length3 separation = bodyPosition - sourcePosition;
    const Length distance = length(separation);
    if (isNearZero(distance)) {
        return Force3::zero();
    }

    const Vec3 direction = normalized(separation.value());
    const Irradiance irradiance = irradianceFromPointSource(luminosity, distance);
    const Area crossSection = Dimensionless{kPi<double>} * raised<2>(bodyRadius);

    return radiationPressureForce(irradiance, crossSection, radiationPressureCoefficient,
                                  direction);
}

}  // namespace ysq
