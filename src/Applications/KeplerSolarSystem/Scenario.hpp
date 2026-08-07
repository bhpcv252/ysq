#pragma once

#include <Applications/Helper/BodyCatalog.hpp>
#include <Applications/Helper/KeplerPopulation.hpp>
#include <Math/Vector3.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ysq::kepler_solar_system {

/// A real planetary ring, as what it actually is: a swarm of independently
/// orbiting particles, each on its own real Kepler ellipse around `parent`
/// (never the Sun), sampled within the real inner/outer edge that planet's
/// own ring system actually spans (see Scenario.cpp's own sourcing
/// comment). Every particle genuinely orbits -- an inner one completes a
/// revolution faster than an outer one, Kepler's third law, the same real
/// differential rotation a real ring shows -- rather than being a static
/// piece of geometry parented to the planet's own transform.
///
/// **Known simplification**: sampled in the shared frame `bodies`' own
/// elements are expressed in, not tilted to the real planet's own
/// equatorial/spin-axis plane (Uranus's rings, famously, are tilted
/// almost edge-on to its orbit) -- this catalog's schema carries a pole
/// only for moons, not for a planet itself, so a real per-planet tilt
/// is future work, not modeled here.
///
/// `particles` is a real, but not literally million-particle, count at
/// real individual size (see Scenario.cpp's own comment on why this is
/// tens of thousands, not the literal millions a real ring's own optical
/// density would take to look continuous -- a real, current limitation of
/// evaluating every particle's own orbit on the CPU each frame, not a
/// choice). A caller drawing this floors the on-screen size at roughly a
/// pixel the same way `KeplerParticle::realRadiusMeters`'s own doc
/// comment describes.
struct RingPopulation {
    std::string parent;
    std::vector<applications::KeplerParticle> particles;
};

/// The real Sun, all 8 planets, every moon JPL SSD publishes orbital
/// elements for, and the 5 IAU-recognized dwarf planets -- loaded from
/// data/solar_system_bodies.csv via `Applications/Helper/BodyCatalog.hpp`'s
/// `loadKeplerBodyCatalog`, which keeps every non-root body's live orbital
/// elements rather than resolving once to a fixed initial state:
/// `ysq::stateVectorAtTime` is what turns those into a position at
/// whatever simulation time the app asks for, on demand, every frame,
/// costing the same regardless of how large a jump that time is. See
/// src/Applications/KeplerSolarSystem/README.md for what this trades away
/// against `Applications::SolarSystem`'s real N-body integration.
///
/// `asteroidBelt` and `kuiperBelt` are procedural populations
/// (`Applications::Helper::generateKeplerPopulation`), Sun-parented,
/// sampled within the real semi-major-axis range each belt actually
/// occupies; see Scenario.cpp for the exact ranges and their sourcing.
struct Scenario {
    /// Sun first (index 0, `applications::KeplerCatalogBody::parent`
    /// empty), then every planet, moon, and dwarf planet, `parent` naming
    /// the body it orbits by name.
    std::vector<applications::KeplerCatalogBody> bodies;
    std::vector<applications::KeplerParticle> asteroidBelt;
    std::vector<applications::KeplerParticle> kuiperBelt;
    std::vector<RingPopulation> rings;
};

/// `std::nullopt` if the data file could not be loaded or parsed; `error`
/// (when given) names why, the caller decides how to report it.
[[nodiscard]] std::optional<Scenario> makeScenario(std::string* error = nullptr);

/// Physics runs in real SI meters; rendering does not use that scale
/// directly, or bodies would be either invisible or off in the distance.
/// One astronomical unit of separation maps to this many render units --
/// the same convention and the same value `Applications::SolarSystem`'s own
/// `kRenderUnitsPerAu` uses, kept as this app's own separate constant
/// rather than a shared dependency between the two apps.
inline constexpr float kRenderUnitsPerAu = 5.0f;

/// `toRenderPosition` and `toRenderRadius` use the exact same factor: true
/// to scale, position and size both.
[[nodiscard]] Vec3f toRenderPosition(const Vec3& metersPosition);
[[nodiscard]] float toRenderRadius(double metersRadius);

}  // namespace ysq::kepler_solar_system
