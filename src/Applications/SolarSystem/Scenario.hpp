#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>

#include <string>
#include <vector>

namespace ysq::solar_system {

/// One planet: its physical Body plus render-only display metadata. Color and
/// renderRadius are chosen for legibility, not measured; see kRenderUnitsPerAu
/// below for why size is not to scale.
struct Planet {
    std::string name;
    Body body;
    Vec3f color;
    float renderRadius = 0.2f;
};

/// Sun + five planets (Mercury through Jupiter), circular orbits at each
/// planet's real semi-major axis, real (textbook-approximate) masses. Total
/// system momentum is exactly zero by construction: the Sun's initial
/// momentum is set to minus the sum of the planets', the same center-of-mass
/// setup tests/integration/orbit_stability.cpp uses for two bodies.
///
/// Masses and orbital radii are standard published approximate values, and
/// eccentricity is taken as zero. This is a recognizable, physically valid
/// solar system for a demo, not a precision ephemeris.
struct Scenario {
    Body sun;
    Vec3f sunColor;
    float sunRenderRadius = 1.5f;
    std::vector<Planet> planets;

    /// Every body, Sun first, in the order gravity should sum them.
    [[nodiscard]] std::vector<Body> allBodies() const;
};

[[nodiscard]] Scenario makeScenario();

/// Physics runs in real SI meters; rendering does not use that scale
/// directly, or bodies would be either invisible or off in the distance.
/// One astronomical unit of separation maps to this many render units.
inline constexpr float kRenderUnitsPerAu = 5.0f;

/// Body *sizes* are never to this or any consistent scale: the Sun's true
/// radius is about 109 Earth radii, which at a zoom that shows the outer
/// planets' orbits would make every planet a sub-pixel dot.
[[nodiscard]] Vec3f toRenderPosition(const Length3& position);

}  // namespace ysq::solar_system
