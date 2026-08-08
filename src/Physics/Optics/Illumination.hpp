#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>

#include <array>
#include <span>

namespace ysq {

/// A purely opaque sphere: blocks straight-line light, nothing bends around
/// it.
struct OpaqueOccluder {
    Vec3 center{};
    double radius{};
};

/// An occluder with both an opaque core and a refracting, scattering
/// atmosphere: everything Optics/RefractiveMedium.hpp and
/// Optics/RayleighScattering.hpp need, gathered so illuminate() below has
/// one thing to carry per such body. General for any body shaped this way,
/// not Earth-specific; a scenario supplies Earth's real numbers.
struct RefractingOccluder {
    Vec3 center{};
    double opaqueRadius{};
    RefractiveMedium medium;
    double surfaceNumberDensity{};
    double scatteringScaleHeight{};
};

/// What reached the target: `transmission` is the fraction of the source's
/// intensity that arrived, per wavelength in whatever order the caller's
/// `wavelengths` array was in (an RGB triple, ordinarily); `geometricVisibility`
/// is the fraction of sampled source points with a clear, unbent line of
/// sight, independent of any medium, useful for a caller that wants to
/// separate "how much of the disk is blocked" from "what color got
/// through".
struct IlluminationResult {
    Vec3 transmission{1.0, 1.0, 1.0};
    double geometricVisibility = 1.0;
};

/// General extended-source illumination: how much light from a spherical
/// source, of `sourceRadius` centered at `sourceCenter`, reaches `target`,
/// through a scene of `opaqueOccluders` (block outright) and at most one
/// `refractingOccluder` (its own opaque core blocks unless a bent path
/// through its atmosphere reaches `target` instead, attenuated per
/// Optics/RayleighScattering.hpp at each of `wavelengths`).
///
/// Samples `sourceSamples` points around the source's limb as seen from
/// `target`; each sample that is not blocked contributes full transmission,
/// each blocked only by `refractingOccluder`'s core attempts a bent path
/// (a bisection over impact parameter against `stepBudget` propagation
/// steps per trial, the same shooting technique
/// Optics/Lensing.hpp::deflectionAngle uses, here searching for the impact
/// parameter whose bent path reaches `target` rather than for a deflection
/// angle), and each blocked by anything else contributes nothing. The
/// result is the average over all samples.
///
/// Nothing here knows what an eclipse is, or which occluder is a planet:
/// this is "how much of an extended light source is visible from a point,
/// accounting for straight occlusion and one graded-index medium", general
/// for any scene shaped this way. `sourceSamples` and `stepBudget` trade
/// fidelity for speed; a caller driving this every rendered frame picks
/// small values, a caller validating a specific number picks large ones.
[[nodiscard]] IlluminationResult
illuminate(const Vec3& sourceCenter, double sourceRadius,
           std::span<const OpaqueOccluder> opaqueOccluders,
           const RefractingOccluder* refractingOccluder, const Vec3& target,
           const std::array<double, 3>& wavelengths, int sourceSamples, int stepBudget);

/// The fraction of a circular light source's own disc, by area, that
/// remains unobstructed as seen from `point`, for a single opaque
/// spherical `occluder` -- real eclipse/transit geometry (each disc's own
/// apparent angular radius as seen from `point`, and the angular
/// separation between their centers), not a sampled or ray-marched
/// approximation like `illuminate()` above: closed-form, O(1) regardless
/// of how fine an answer is wanted, and with no atmosphere or color of
/// its own to trace through, which is what makes this cheap enough to
/// call once per particle for a large population every rendered frame,
/// unlike `illuminate()`.
///
/// 1.0 means fully lit (the occluder's own disc, as seen from `point`,
/// does not overlap the source's at all); 0.0 means the occluder fully
/// covers the source (a total eclipse, or an occluder whose own apparent
/// size at least matches the source's); a value in between is the real,
/// geometrically exact partial fraction -- an annular eclipse (the
/// occluder's own disc entirely inside the source's, but too small to
/// cover it) included. Returns 1.0 outright, with no further geometry, if
/// `occluder` is not even nearer to `point` than `sourceCenter` is, since
/// it cannot be sitting between them.
[[nodiscard]] double discOcclusionFraction(const Vec3& point, const Vec3& sourceCenter,
                                           double sourceRadius, const Vec3& occluderCenter,
                                           double occluderRadius);

}  // namespace ysq
