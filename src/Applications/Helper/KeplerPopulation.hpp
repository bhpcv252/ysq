#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Gravity/Kepler.hpp>

#include <cstdint>
#include <vector>

namespace ysq::applications {

/// A single procedurally-generated body on its own Kepler orbit: no mass of
/// its own that matters to anything (an asteroid, a ring grain), just
/// enough to evaluate and render it -- which already-loaded body it orbits,
/// that body's own gravitational parameter (`Physics/Gravity/Kepler.hpp`'s
/// `stateVectorAtTime` needs it directly, the same role `KeplerCatalogBody`'s
/// own `parentGm` plays), its own elements, and how to draw it. Used for
/// populations too numerous to name individually in a catalog (an asteroid
/// belt, a planet's ring), unlike `KeplerCatalogBody`'s one row per real,
/// named body.
struct KeplerParticle {
    int parentIndex = -1;
    double parentGm = 0.0;
    OrbitalElementsAtEpoch elements;
    /// True to scale, converted the same way a real body's own radius is.
    /// A caller drawing this will usually have to floor the on-screen
    /// size at roughly a pixel (a real individual particle genuinely is
    /// smaller than that from any orbital distance, and a rasterizer
    /// cannot draw less than a pixel regardless) -- a rendering
    /// necessity, not a second, artistic size to choose instead of this
    /// one; see `KeplerSolarSystem/main.cpp`'s own `worldSizeForPixels`
    /// floor for the worked example.
    double realRadiusMeters = 0.0;
    Vec3f color;
};

/// A procedural population of `count` particles orbiting `parentIndex`
/// (`parentGm` its gravitational parameter): semi-major axis sampled
/// uniformly in `[minSemiMajorAxis, maxSemiMajorAxis]`, eccentricity
/// uniform in `[0, maxEccentricity]`, inclination uniform in
/// `[0, maxInclination]` (radians, already in the same shared frame the
/// rest of the scenario's elements are expressed in -- unlike
/// `KeplerCatalogBody`, a synthetic population has no real published pole
/// of its own to rotate out of), and longitude of ascending
/// node/argument of periapsis/mean anomaly each uniform in `[0, tau)`. A
/// real belt or ring is not one ellipse but a swarm of them; this is the
/// swarm.
///
/// `seed` makes the result deterministic: the same seed always produces the
/// same particles, for a stable picture across runs and a testable
/// generator.
///
/// General procedural-population generation, not specific to any one belt
/// or ring; a caller supplies the real astronomical ranges (the real
/// asteroid belt's semi-major axis span, Saturn's real ring radii, whatever
/// the scenario needs).
///
/// `realRadiusMeters` is one representative real size applied to every
/// particle in the population, not an individually measured one -- there
/// is no published radius for an unnamed, procedurally-placed body, the
/// same honest-estimate convention `SolarSystem/data/solar_system_bodies.csv`'s
/// own header comment already documents for its small, unmeasured moons.
[[nodiscard]] std::vector<KeplerParticle> generateKeplerPopulation(
    int parentIndex, double parentGm, double minSemiMajorAxis, double maxSemiMajorAxis,
    double maxEccentricity, double maxInclination, int count, std::uint64_t seed,
    double realRadiusMeters, const Vec3f& color);

}  // namespace ysq::applications
