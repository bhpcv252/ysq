#include <Physics/Optics/Illumination.hpp>

#include <Math/Intersection.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector4.hpp>
#include <Physics/Optics/Propagation.hpp>
#include <Physics/Optics/RayleighScattering.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

namespace ysq {

namespace {

/// The result of tracing one candidate grazing path: the excess bending
/// angle beyond the flat-space sweep (Optics/Lensing.hpp::deflectionAngle's
/// own quantity, recomputed here rather than called out to, so the same
/// propagation pass also accumulates the Rayleigh path integral without a
/// second trip through the solver), and the integral of number density
/// along that path -- the whole optical depth, at any wavelength, is just
/// this integral times that wavelength's cross-section, so one path trace
/// serves every wavelength.
struct GrazingTrace {
    bool valid = false;
    double deflection = 0.0;
    double numberDensityIntegral = 0.0;
};

[[nodiscard]] GrazingTrace traceGrazingPath(const RefractiveMedium& medium,
                                            double impactParameter, double startRadius,
                                            double step, int maxSteps,
                                            double surfaceNumberDensity,
                                            double scatteringScaleHeight) {
    if (impactParameter >= startRadius) {
        // The null condition has no real solution launching inward from
        // startRadius with this large an impact parameter: skip the
        // propagation entirely rather than run it to no purpose.
        return {};
    }
    const PhaseState<Vector4<double>> ray =
        refractiveMediumRayFromImpactParameter(medium, impactParameter, startRadius);

    double previousRadius = startRadius;
    double previousAzimuth = 0.0;
    double densityIntegral = 0.0;
    bool wasInside = false;
    bool crossed = false;
    double crossingAzimuth = 0.0;

    (void)propagate(
        medium, ray, step * static_cast<double>(maxSteps),
        static_cast<std::size_t>(maxSteps),
        [&](double, const PhaseState<Vector4<double>>& state) {
            if (crossed) {
                return;
            }
            const double r = state.position.y;
            const double n = medium.refractiveIndex(r);
            const double dl = (state.velocity.x / n) * step;
            densityIntegral +=
                exponentialNumberDensity(r, surfaceNumberDensity, medium.radius(),
                                         scatteringScaleHeight) *
                dl;

            if (r < startRadius) {
                wasInside = true;
            }
            if (wasInside && previousRadius < startRadius && r >= startRadius) {
                const double t = (startRadius - previousRadius) / (r - previousRadius);
                crossingAzimuth =
                    previousAzimuth + t * (state.position.w - previousAzimuth);
                crossed = true;
            }
            previousRadius = r;
            previousAzimuth = state.position.w;
        });

    if (!crossed) {
        return {};
    }
    const double flatSweep = kPi<double> - 2.0 * std::asin(impactParameter / startRadius);
    const double deflection = std::abs(crossingAzimuth) - flatSweep;
    return {true, deflection, densityIntegral};
}

/// Rotates `direction` toward `-towardAxis` by `angle`: the sense every
/// graded-index or gravitational bending curves a ray in, toward the denser
/// medium or the mass. `direction` and `towardAxis` must be orthonormal.
[[nodiscard]] Vec3 bendToward(const Vec3& direction, const Vec3& towardAxis,
                              double angle) {
    return direction * std::cos(angle) - towardAxis * std::sin(angle);
}

/// Whether a candidate impact parameter `b`'s bent path (traced only for
/// its deflection angle, the cheaper of GrazingTrace's two results) ends up
/// heading toward `target`, as a signed perpendicular miss distance in the
/// `towardAxis` direction: the root findGrazingImpactParameter below
/// bisects on `b` to drive this to zero.
[[nodiscard]] std::optional<double>
missDistance(const RefractiveMedium& medium, const Vec3& bendPoint, const Vec3& incoming,
             const Vec3& towardAxis, const Vec3& target, double impactParameter,
             double startRadius, double step, int maxSteps) {
    const GrazingTrace trace =
        traceGrazingPath(medium, impactParameter, startRadius, step, maxSteps, 0.0, 1.0);
    if (!trace.valid) {
        return std::nullopt;
    }
    const Vec3 outgoing = bendToward(incoming, towardAxis, trace.deflection);
    const double s = dot(target - bendPoint, outgoing);
    const Vec3 closestPoint = bendPoint + outgoing * s;
    return dot(target - closestPoint, towardAxis);
}

/// Brackets and bisects for the impact parameter whose bent path reaches
/// `target`, searching outward from the occluder's opaque radius. Returns
/// nullopt if no sign change is found in range, meaning nothing in this
/// medium bends light from this source point to this target.
[[nodiscard]] std::optional<double>
findGrazingImpactParameter(const RefractiveMedium& medium, const Vec3& bendAxisCenter,
                           const Vec3& incoming, const Vec3& towardAxis,
                           const Vec3& target, double opaqueRadius, double startRadius,
                           double step, int maxSteps) {
    constexpr int kBracketSamples = 4;
    constexpr int kBisectionIterations = 7;
    // Scale heights past the opaque radius to search: must stay below the
    // 15-scale-height margin illuminate() built startRadius from, or the
    // impact parameter exceeds startRadius and every trace beyond this
    // point is skipped outright (see traceGrazingPath's own guard).
    constexpr double kSearchSpan = 12.0;

    const double scaleHeight = medium.scaleHeight();
    const auto evaluate = [&](double b) {
        const Vec3 bendPoint = bendAxisCenter + towardAxis * b;
        return missDistance(medium, bendPoint, incoming, towardAxis, target, b,
                            startRadius, step, maxSteps);
    };

    double lowB = opaqueRadius;
    std::optional<double> lowMiss = evaluate(lowB);
    for (int i = 1; i <= kBracketSamples; ++i) {
        const double highB =
            opaqueRadius +
            (kSearchSpan * static_cast<double>(i) / kBracketSamples) * scaleHeight;
        const std::optional<double> highMiss = evaluate(highB);
        if (lowMiss && highMiss && (*lowMiss < 0.0) != (*highMiss < 0.0)) {
            double lo = lowB;
            double hi = highB;
            double loMiss = *lowMiss;
            for (int iter = 0; iter < kBisectionIterations; ++iter) {
                const double mid = 0.5 * (lo + hi);
                const std::optional<double> midMiss = evaluate(mid);
                if (!midMiss) {
                    break;
                }
                if ((*midMiss < 0.0) == (loMiss < 0.0)) {
                    lo = mid;
                    loMiss = *midMiss;
                } else {
                    hi = mid;
                }
            }
            return 0.5 * (lo + hi);
        }
        lowB = highB;
        lowMiss = highMiss;
    }
    return std::nullopt;
}

}  // namespace

IlluminationResult illuminate(const Vec3& sourceCenter, double sourceRadius,
                              std::span<const OpaqueOccluder> opaqueOccluders,
                              const RefractingOccluder* refractingOccluder,
                              const Vec3& target,
                              const std::array<double, 3>& wavelengths, int sourceSamples,
                              int stepBudget) {
    const Vec3 viewDir = normalized(sourceCenter - target);
    const Vec3 arbitrary = (std::abs(viewDir.x) < 0.9) ? Vec3::unitX() : Vec3::unitY();
    const Vec3 u = normalized(cross(arbitrary, viewDir));
    const Vec3 v = cross(viewDir, u);

    Vec3 transmissionSum{};
    double geometricVisible = 0.0;

    for (int i = 0; i < sourceSamples; ++i) {
        const double angle =
            kTau<double> * static_cast<double>(i) / static_cast<double>(sourceSamples);
        const Vec3 sample =
            sourceCenter + (u * std::cos(angle) + v * std::sin(angle)) * sourceRadius;

        bool blockedByOther = false;
        for (const OpaqueOccluder& occluder : opaqueOccluders) {
            if (segmentIntersectsSphere(
                    sample, target, Sphere3<double>{occluder.center, occluder.radius})) {
                blockedByOther = true;
                break;
            }
        }
        if (blockedByOther) {
            continue;
        }

        bool blockedByCore = false;
        if (refractingOccluder) {
            blockedByCore = segmentIntersectsSphere(
                sample, target,
                Sphere3<double>{refractingOccluder->center,
                                refractingOccluder->opaqueRadius});
        }

        if (!blockedByCore) {
            transmissionSum += Vec3{1.0, 1.0, 1.0};
            geometricVisible += 1.0;
            continue;
        }

        const Vec3 incoming = normalized(refractingOccluder->center - sample);
        const Vec3 toTargetPerp =
            (target - refractingOccluder->center) -
            incoming * dot(target - refractingOccluder->center, incoming);
        const std::optional<Vec3> towardAxis = tryNormalized(toTargetPerp);
        if (!towardAxis) {
            continue;
        }

        const double startRadius = refractingOccluder->opaqueRadius +
                                   15.0 * refractingOccluder->medium.scaleHeight();
        // A grazing ray spends most of its affine parameter moving
        // tangentially, not radially: near periapsis dr/dlambda approaches
        // zero, so the affine range a round trip needs is not the radial
        // distance (startRadius - opaqueRadius) but the flat-space chord
        // length 2 sqrt(startRadius^2 - opaqueRadius^2) (opaqueRadius, the
        // most grazing case this ever asks for, is the worst one), tripled
        // for margin against the slower near-periapsis convergence real
        // bending adds. Deriving the step from stepBudget against that
        // range, rather than fixing the step size outright, keeps
        // stepBudget purely a resolution knob: any budget covers the whole
        // round trip, just more or less finely.
        const double chordLength =
            std::sqrt(startRadius * startRadius - refractingOccluder->opaqueRadius *
                                                      refractingOccluder->opaqueRadius);
        const double step = 6.0 * chordLength / stepBudget;

        const std::optional<double> impactParameter = findGrazingImpactParameter(
            refractingOccluder->medium, refractingOccluder->center, incoming, *towardAxis,
            target, refractingOccluder->opaqueRadius, startRadius, step, stepBudget);
        if (!impactParameter) {
            continue;
        }

        const GrazingTrace trace =
            traceGrazingPath(refractingOccluder->medium, *impactParameter, startRadius,
                             step, stepBudget, refractingOccluder->surfaceNumberDensity,
                             refractingOccluder->scatteringScaleHeight);
        if (!trace.valid) {
            continue;
        }

        Vec3 sampleTransmission{};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const double crossSection = rayleighCrossSection(
                wavelengths[channel],
                1.0 + refractingOccluder->medium.surfaceRefractivity(),
                refractingOccluder->surfaceNumberDensity);
            sampleTransmission[channel] =
                transmission(crossSection * trace.numberDensityIntegral);
        }
        transmissionSum += sampleTransmission;
    }

    IlluminationResult result;
    result.transmission = transmissionSum / static_cast<double>(sourceSamples);
    result.geometricVisibility = geometricVisible / static_cast<double>(sourceSamples);
    return result;
}

double discOcclusionFraction(const Vec3& point, const Vec3& sourceCenter, double sourceRadius,
                             const Vec3& occluderCenter, double occluderRadius) {
    const Vec3 toSource = sourceCenter - point;
    const Vec3 toOccluder = occluderCenter - point;
    const double distanceToSource = length(toSource);
    const double distanceToOccluder = length(toOccluder);

    if (distanceToSource <= 0.0 || distanceToOccluder <= 0.0 ||
        distanceToOccluder >= distanceToSource) {
        // Coincident with a center (degenerate), or the occluder is not
        // even nearer than the source -- it cannot be sitting between
        // point and source either way.
        return 1.0;
    }

    // Apparent angular radius of each disc as seen from point: the real
    // quantity an eclipse's own geometry is about, not either body's
    // linear size directly.
    const double sourceAngularRadius = std::atan(sourceRadius / distanceToSource);
    const double occluderAngularRadius = std::atan(occluderRadius / distanceToOccluder);

    const double cosSeparation =
        std::clamp(dot(toSource, toOccluder) / (distanceToSource * distanceToOccluder), -1.0, 1.0);
    const double separation = std::acos(cosSeparation);

    if (separation >= sourceAngularRadius + occluderAngularRadius) {
        return 1.0;  // the two discs do not overlap at all
    }
    if (separation <= std::abs(sourceAngularRadius - occluderAngularRadius)) {
        // One disc entirely inside the other: a total eclipse (occluder at
        // least as big, angularly) or an annular one (occluder smaller,
        // covering only the fraction of the source's own area its disc's
        // area is -- pi cancels in the ratio).
        return occluderAngularRadius >= sourceAngularRadius
                  ? 0.0
                  : 1.0 - (occluderAngularRadius * occluderAngularRadius) /
                              (sourceAngularRadius * sourceAngularRadius);
    }

    // Partial eclipse: the standard closed-form area of intersection of
    // two circles (radii r1, r2, centers separated by d), as a fraction of
    // the source disc's own area.
    const double d = separation;
    const double r1 = sourceAngularRadius;
    const double r2 = occluderAngularRadius;
    const double part1 = r1 * r1 * std::acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1));
    const double part2 = r2 * r2 * std::acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2));
    const double part3 = 0.5 * std::sqrt(std::max(
                                   0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)));
    const double overlapArea = part1 + part2 - part3;
    const double sourceArea = kPi<double> * r1 * r1;
    return std::clamp(1.0 - overlapArea / sourceArea, 0.0, 1.0);
}

}  // namespace ysq
