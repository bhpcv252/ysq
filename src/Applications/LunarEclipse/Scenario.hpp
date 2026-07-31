#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Optics/RefractiveMedium.hpp>

#include <vector>

namespace ysq::lunar_eclipse {

/// Sun, Earth, Moon and Jupiter, real masses, radii and orbital elements
/// (real eccentricities and the Moon's real 5.145 degree inclination, not
/// the circular, coplanar simplification SolarSystem uses), integrated by
/// the same n-body Newtonian stepper. Jupiter is a gravitational perturber
/// only: it is in `allBodies()` so its gravity (and, unlike SolarSystem,
/// nothing else about it) is felt, but nothing about this scenario renders
/// it. Earth alone carries J2, real moments of inertia, real axial tilt and
/// rotation, and a real atmosphere; nothing else needs any of those.
///
/// Orbital elements are real, published, approximate values at an
/// arbitrary epoch: this does not reproduce any specific real eclipse's
/// calendar date, only the real shapes and real physics an eclipse season
/// emerges from. See src/Applications/README.md.
struct Scenario {
    Body sun;
    Body earth;
    Body moon;
    Body jupiter;

    Vec3f sunColor{1.0f, 0.9f, 0.6f};
    Vec3f earthColor{0.25f, 0.45f, 0.85f};
    Vec3f moonColor{0.7f, 0.7f, 0.68f};

    /// Earth's real atmosphere, general parameters
    /// Optics/RefractiveMedium.hpp needs, built once here since they never
    /// change; only Earth's *position* (needed to place the medium each
    /// frame) does.
    RefractiveMedium earthAtmosphere;
    double earthSurfaceNumberDensity = 0.0;
    double earthScatteringScaleHeight = 0.0;

    /// Sun, Earth, Moon, Jupiter, in that order: gravity sums them in this
    /// order, and index 1/2 are Earth/Moon wherever a caller needs them
    /// specifically (main.cpp's render and Illumination calls).
    [[nodiscard]] std::vector<Body> allBodies() const;
};

[[nodiscard]] Scenario makeScenario();

/// One astronomical unit of separation maps to this many render units.
/// `toRenderRadius` uses the exact same factor: true to scale, position
/// and size both, deliberately unlike SolarSystem's cosmetic
/// `renderRadius`. Real sizes span an enormous range at any single linear
/// scale (the Moon's true radius is smaller than a camera near-clip plane
/// sized for viewing the whole Sun-Earth distance), so main.cpp's camera
/// has to handle that dynamically (near/far planes tied to current zoom
/// distance) rather than this constant trying to compromise between them.
inline constexpr float kRenderUnitsPerAu = 5.0f;

[[nodiscard]] Vec3f toRenderPosition(const Length3& position);

/// The same `kRenderUnitsPerAu` factor `toRenderPosition` uses, applied to
/// a body's actual physical radius rather than its position: one scale,
/// used consistently, is what makes this true to scale rather than
/// artistic.
[[nodiscard]] float toRenderRadius(Length radius);

}  // namespace ysq::lunar_eclipse
