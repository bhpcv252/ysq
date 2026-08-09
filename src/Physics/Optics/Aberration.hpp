#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Mechanics/Kinematics.hpp>
#include <Units/Constants.hpp>
#include <Units/Velocity.hpp>

namespace ysq {

/// Relativistic aberration: how a light ray's own direction changes
/// between two frames in relative motion -- the direction counterpart to
/// `FrequencyShift.hpp`'s Doppler shift, the same relative motion changing
/// both a photon's frequency and, for anything not exactly along the boost
/// axis, its apparent direction too.
///
/// Aberration is exactly relativistic velocity addition
/// (`Mechanics/Kinematics.hpp`'s `relativisticVelocityAdd`) applied to a
/// velocity of magnitude `c`: a photon moving in
/// `directionInOriginalFrame` has velocity `c * that direction` in the
/// original frame, and transforming that velocity into a frame moving at
/// `frameVelocity` relative to the original one gives the same photon's
/// velocity there, whose direction -- still magnitude `c`, since the speed
/// of light is frame-independent -- is the aberrated direction.
///
/// This tracks the photon's own direction of *travel*, not the apparent
/// direction to whatever emitted it (its source lies the other way,
/// `-directionInOriginalFrame`): the relativistic beaming that concentrates
/// an isotropic field's apparent *sources* toward a fast frame's forward
/// direction is this same shift on the travel direction, seen through that
/// negation.
[[nodiscard]] inline Vec3 aberratedDirection(const Vec3& directionInOriginalFrame,
                                             const Velocity3& frameVelocity) {
    const Velocity3 photonVelocity{normalized(directionInOriginalFrame) *
                                   constants::speedOfLight.value()};
    return normalized(relativisticVelocityAdd(photonVelocity, frameVelocity).value());
}

/// The classic 1D aberration formula: `cos(theta') = (cos(theta) - beta) /
/// (1 - beta cos(theta))`, `beta = boostSpeed / c`, for a photon whose
/// direction makes angle `theta` (given as its cosine) with the boost
/// axis in the original frame. The special case of `aberratedDirection`
/// above where the photon's direction and the boost already share an
/// axis, useful directly for the standard textbook setup without
/// assembling a `Vec3` and a `Velocity3` for it.
[[nodiscard]] inline double aberratedCosine(double cosAngleInOriginalFrame,
                                            Speed boostSpeed) {
    const double beta = (boostSpeed / constants::speedOfLight).value();
    return (cosAngleInOriginalFrame - beta) / (1.0 - beta * cosAngleInOriginalFrame);
}

}  // namespace ysq
