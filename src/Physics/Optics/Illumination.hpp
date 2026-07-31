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

}  // namespace ysq
