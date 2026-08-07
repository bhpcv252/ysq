#pragma once

#include <Math/Quaternion.hpp>

namespace ysq::applications {

/// A rotation that carries the local +Z axis to point at a given
/// right ascension/declination direction: the standard construction for
/// turning a published pole direction (a moon's Laplace-plane pole, a
/// planet's own rotational pole) into a frame usable in a single, shared
/// global reference frame.
///
/// The published `(rightAscension, declination)` is the pole's direction in
/// whatever frame it was measured in (JPL's satellite pole directions are
/// ICRF/J2000 mean equatorial); this returns the rotation from a *local*
/// frame -- +Z the pole, `stateVectorFromElements`'s reference plane the
/// local xy-plane -- into that same measured frame. A caller composes it
/// with whatever fixed rotation relates the measured frame to the
/// simulation's own global one (see `src/Applications/Helper/README.md`'s
/// note on the ecliptic/equatorial mismatch between planetary and
/// satellite data).
///
/// `Rz(rightAscension + pi/2) . Rx(pi/2 - declination)`: rotate +Z toward
/// the pole's declination first (about the frame's own x-axis, the
/// standard co-declination amount), then swing the whole thing around to
/// the pole's right ascension -- the same two-angle construction the IAU
/// uses to define any body's rotational pole from its own announced
/// (RA, Dec).
[[nodiscard]] Quat poleRotation(double rightAscension, double declination);

}  // namespace ysq::applications
