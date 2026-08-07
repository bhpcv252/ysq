#pragma once

#include <Applications/Helper/BodyCatalog.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ysq::solar_system {

/// The real Sun, all 8 planets, and every moon JPL SSD publishes orbital
/// elements for -- loaded from `data/solar_system_bodies.csv` via
/// `Applications/Helper/BodyCatalog.hpp`, not hardcoded. Real masses,
/// radii, and orbital elements throughout; see that file's own header
/// comment for sourcing, and the note there on the moons whose mass/radius
/// is estimated (not individually measured, and gravitationally negligible
/// either way) rather than the roughly forty JPL has precise values for.
///
/// Total system momentum is exactly zero by construction: the Sun's own
/// momentum is set to minus the sum of every other body's, the same
/// center-of-mass convention `LunarEclipse`'s scenario already uses.
struct Scenario {
    /// Sun first (index 0, `applications::CatalogBody::parent` empty),
    /// then every planet and moon, `parent` naming the body it orbits by
    /// name (a planet's own name for a moon, "Sun" for a planet).
    std::vector<applications::CatalogBody> bodies;

    /// Every body's `Body`, in `bodies`' own order, for the physics
    /// integrator.
    [[nodiscard]] std::vector<Body> allBodies() const;
};

/// `std::nullopt` if the data file could not be loaded or parsed; `error`
/// (when given) names why, the caller decides how to report it.
[[nodiscard]] std::optional<Scenario> makeScenario(std::string* error = nullptr);

/// Physics runs in real SI meters; rendering does not use that scale
/// directly, or bodies would be either invisible or off in the distance.
/// One astronomical unit of separation maps to this many render units.
inline constexpr float kRenderUnitsPerAu = 5.0f;

/// `toRenderPosition` and `toRenderRadius` use the exact same factor: true
/// to scale, position and size both, the same convention
/// `LunarEclipse::toRenderRadius` already uses (unlike this file's own
/// previous cosmetic, not-to-scale `Planet::renderRadius`).
[[nodiscard]] Vec3f toRenderPosition(const Length3& position);
[[nodiscard]] float toRenderRadius(Length radius);

}  // namespace ysq::solar_system
