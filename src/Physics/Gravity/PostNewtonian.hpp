#pragma once

#include <Physics/Body.hpp>
#include <Units/Force.hpp>

namespace ysq {

/// The 1PN (first post-Newtonian) correction to the gravitational
/// acceleration on a test particle orbiting a dominant source, in the
/// standard PPN form with gamma = beta = 1 (general relativity):
///
///     a_1PN = (GM / (c^2 r^2)) * [ (4 GM/r - v^2) n + 4 (v . n) v ]
///
/// where r and v are the position and velocity of `testParticle` relative to
/// `source`, n is the unit vector from source to testParticle, and GM uses
/// only source's mass. This is the correction that produces perihelion
/// precession; src/Physics/README.md has the derivation and the analytic
/// precession-per-orbit formula it is validated against.
///
/// **Scope: two bodies, one of them a test particle.** This is exact in the
/// limit source's mass dominates, which is the regime the analytic
/// precession formula itself assumes (Mercury around the Sun, not two
/// comparable masses). A full N-body correction is the Einstein-Infeld-
/// Hoffmann equations, which add cross terms between every pair and are not
/// implemented here; extending to that is future work, not a limitation of
/// this formula being wrong for its stated regime.
///
/// The caller adds this to newtonianAcceleration(testParticle.position, {source})
/// to get the full post-Newtonian acceleration; the two rungs of the ladder
/// compose rather than replacing one another.
[[nodiscard]] Acceleration3 postNewtonianCorrection(const Body& testParticle,
                                                    const Body& source);

}  // namespace ysq
