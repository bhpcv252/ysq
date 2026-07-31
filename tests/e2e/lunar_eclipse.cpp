#include <Applications/LunarEclipse/Scenario.hpp>

#include <Math/Vector3.hpp>

#include <Physics/Optics/Illumination.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>

#include <gtest/gtest.h>

#include <array>
#include <string>

/// LunarEclipse's own scenario (real Sun/Earth/Moon radii, Earth's real
/// atmosphere), asserting the phase sequence a real eclipse actually has:
/// none, at the Moon well outside Earth's penumbra; penumbral, inside the
/// penumbra but outside the umbra, some but not all of the Sun's disk
/// blocked; total, deep in the umbra, no direct sunlight at all, only
/// refracted, reddened light. The real orbital integration this would take
/// to reach an actual eclipse season is a job for the running application's
/// own time-scale control, minutes of real time even fast-forwarded, not a
/// test; this instead places the Moon at explicit offsets from the shadow
/// axis, computed from Scenario's own real Sun/Earth distance and radii,
/// the same geometry Optics/Illumination.hpp resolves either way, and
/// checks the sequence a crossing would actually show.

namespace {

using namespace ysq::lunar_eclipse;

std::string phaseLabelFor(const ysq::IlluminationResult& result) {
    if (0.99 <= result.geometricVisibility) {
        return "None";
    }
    if (0.6 <= result.geometricVisibility) {
        return "Penumbral";
    }
    if (0.01 <= result.geometricVisibility) {
        return "Partial";
    }
    return "Total";
}

}  // namespace

TEST(LunarEclipseE2E, PhaseSequenceAcrossTheShadowMatchesRealEclipseGeometry) {
    const Scenario scenario = makeScenario();
    const std::vector<ysq::Body> bodies = scenario.allBodies();
    const ysq::Body& sun = bodies[0];
    const ysq::Body& earth = bodies[1];

    // A syzygy snapshot: Sun and Earth at the scenario's own real
    // separation, Moon at Earth's real orbital distance directly along the
    // anti-solar axis, offset perpendicular to it to sample across the
    // shadow. Real Sun/Earth radii at this real separation give a real
    // umbra radius of about 4600 km and penumbra radius of about 8200 km
    // at the Moon's distance, the actual reason total lunar eclipses
    // happen at all: the umbra is wider than the Moon.
    const ysq::Vec3 sunCenter = sun.position.value();
    const ysq::Vec3 earthCenter = earth.position.value();
    const ysq::Vec3 antiSolar = normalized(earthCenter - sunCenter);
    const double moonDistance = 3.844e8;

    ysq::RefractingOccluder earthOccluder;
    earthOccluder.center = earthCenter;
    earthOccluder.opaqueRadius = earth.radius.value();
    earthOccluder.medium = scenario.earthAtmosphere;
    earthOccluder.surfaceNumberDensity = scenario.earthSurfaceNumberDensity;
    earthOccluder.scatteringScaleHeight = scenario.earthScatteringScaleHeight;

    const ysq::Vec3 axisPoint = earthCenter + antiSolar * moonDistance;
    ysq::Vec3 perpendicular = ysq::Vec3::unitZ();
    if (1.0 - std::abs(dot(perpendicular, antiSolar)) < 1.0e-6) {
        perpendicular = ysq::Vec3::unitY();
    }
    perpendicular = normalized(perpendicular - antiSolar * dot(perpendicular, antiSolar));

    const std::array<double, 3> wavelengths{630.0e-9, 532.0e-9, 465.0e-9};
    constexpr int kSourceSamples = 4;
    constexpr int kStepBudget = 500;

    const auto illuminateAtOffset = [&](double offset) {
        const ysq::Vec3 target = axisPoint + perpendicular * offset;
        return ysq::illuminate(sunCenter, sun.radius.value(), {}, &earthOccluder, target,
                               wavelengths, kSourceSamples, kStepBudget);
    };

    const ysq::IlluminationResult outsidePenumbra = illuminateAtOffset(9.0e6);
    const ysq::IlluminationResult inPenumbraOnly = illuminateAtOffset(7.0e6);
    const ysq::IlluminationResult inUmbra = illuminateAtOffset(1.0e6);
    const ysq::IlluminationResult umbraCenter = illuminateAtOffset(0.0);

    EXPECT_EQ(phaseLabelFor(outsidePenumbra), "None");
    EXPECT_EQ(phaseLabelFor(inPenumbraOnly), "Penumbral");
    EXPECT_EQ(phaseLabelFor(inUmbra), "Total");
    EXPECT_EQ(phaseLabelFor(umbraCenter), "Total");

    // Monotonic: moving from clear sky toward the umbra's center can only
    // reduce how much of the Sun's disk has a clear line of sight, never
    // increase it.
    EXPECT_GE(outsidePenumbra.geometricVisibility, inPenumbraOnly.geometricVisibility);
    EXPECT_GE(inPenumbraOnly.geometricVisibility, inUmbra.geometricVisibility);

    // Still not literally dark at the very center of a real total eclipse:
    // some refracted, reddened light always gets through.
    const double centerTransmission = umbraCenter.transmission.x +
                                      umbraCenter.transmission.y +
                                      umbraCenter.transmission.z;
    EXPECT_GT(centerTransmission, 0.0);
    EXPECT_GT(umbraCenter.transmission.x, umbraCenter.transmission.z);
}
