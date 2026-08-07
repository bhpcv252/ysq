#include <Applications/KeplerSolarSystem/Scenario.hpp>

#include <Applications/Helper/KeplerPopulation.hpp>
#include <Core/Csv.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Gravity/Kepler.hpp>
#include <Physics/Gravity/PostNewtonian.hpp>
#include <Units/Constants.hpp>
#include <Units/Length.hpp>
#include <Units/Unit.hpp>

#include <array>
#include <cstdint>
#include <format>
#include <unordered_map>
#include <string>

namespace ysq::kepler_solar_system {

namespace {

/// Baked in at build time (see CMakeLists.txt): an absolute path is robust
/// to whatever directory the built executable is actually run from, unlike
/// a path relative to the current working directory.
constexpr const char* kDataFilePath =
    YSQ_KEPLER_SOLAR_SYSTEM_DATA_DIR "/solar_system_bodies.csv";

/// The J2000 mean obliquity, 23.4392911 degrees: the fixed rotation about
/// the shared vernal-equinox axis that carries the planets' own
/// ecliptic-referenced elements into the equatorial frame every moon's
/// pole_ra/pole_dec already targets. See
/// src/Applications/Helper/README.md's Pole section for the derivation and
/// why the two kinds of real data do not share a frame to begin with.
[[nodiscard]] Quat eclipticToEquatorialRotation() {
    return Quat::fromAxisAngle(Vec3::unitX(), radians(23.4392911));
}

/// Every real ring this app draws, inner/outer edge in kilometers from the
/// planet's own center. Sourced from published values (NASA/JPL mission
/// summaries and standard references for each system):
///
/// - Jupiter: the halo + main ring, 92,000-128,940 km, plus one combined
///   band for the two faint outer "gossamer" rings, 129,000-226,000 km
///   (individually, dust along Amalthea's own orbit out to 182,000 km and
///   along Thebe's out to 226,000 km; combined here rather than as two
///   heavily overlapping bands, since both are real but far fainter than
///   the main ring either way).
/// - Saturn, five bands, most of any planet here, matching how much ring
///   structure Saturn actually has: D (1.11-1.23 Saturn radii, the
///   faintest, innermost), C+B together (1.23-1.95), A (2.02-2.27) --
///   the real Cassini Division, 1.95-2.02 Saturn radii, sits empty
///   between the last two, not a simplification, it is genuinely far
///   emptier than the rings on either side of it -- then F (140,180-
///   140,680 km, a real, narrow ring just outside A) and G
///   (166,000-175,000 km, faint). Saturn's own equatorial radius,
///   60,268 km, is what the Saturn-radii bands convert by. The very
///   wide, extremely diffuse E ring (real span roughly 180,000-480,000
///   km, peak density near Enceladus's own 238,000 km orbit) is not
///   included -- at this app's particle counts it would be indistinguishable
///   from empty space, which would be more misleading than omitting it.
/// - Uranus: the dense classical nine rings plus the epsilon ring,
///   38,000-51,500 km, plus the two real, faint outer rings Hubble found,
///   mu (86,000-103,000 km) and nu (66,100-69,900 km).
/// - Neptune: 41,000-63,930 km, the Galle ring through the Adams ring
///   (whose own famous arcs are not individually modeled -- they are a
///   real azimuthal clumping within the ring, not a difference in radius,
///   which this app's radius-only ring model has no way to represent).
struct RingKm {
    const char* parent;
    double innerKm;
    double outerKm;
};

constexpr double kSaturnRadiusKm = 60268.0;

constexpr std::array<RingKm, 11> kRingsKm{{
    {"Jupiter", 92000.0, 128940.0},
    {"Jupiter", 129000.0, 226000.0},
    {"Saturn", 1.11 * kSaturnRadiusKm, 1.23 * kSaturnRadiusKm},
    {"Saturn", 1.23 * kSaturnRadiusKm, 1.95 * kSaturnRadiusKm},
    {"Saturn", 2.02 * kSaturnRadiusKm, 2.27 * kSaturnRadiusKm},
    {"Saturn", 140180.0, 140680.0},
    {"Saturn", 166000.0, 175000.0},
    {"Uranus", 38000.0, 51500.0},
    {"Uranus", 66100.0, 69900.0},
    {"Uranus", 86000.0, 103000.0},
    {"Neptune", 41000.0, 63930.0},
}};

constexpr double kRingMaxEccentricity = 0.005;
constexpr double kRingMaxInclinationDeg = 0.5;
constexpr Vec3f kRingParticleColor{0.72f, 0.68f, 0.6f};

/// Real individual size, a large but not literally-millions count, per
/// ring band -- 11 bands' worth of these adds up (evaluating every one's
/// own Kepler orbit on the CPU, every frame, is the real cost; a million
/// particles per band across 11 bands would be hundreds of millions of
/// Newton-Raphson solves a frame). This is chosen to keep the total
/// real-time with every ring band visible at once, not a rounded-down
/// "good enough" guess -- honestly sparse at this scale is the correct
/// result of that constraint, not a bug to hide (main.cpp's own
/// pixel-size floor is what keeps it visible at all despite that).
constexpr int kRingParticleCount = 8000;
/// A real ring's own particles range from dust grains to, rarely, a
/// house-sized moonlet; there is no one real size, so this is a
/// representative "typical visible chunk" rather than a measurement, the
/// same honest-estimate status the asteroid/Kuiper belt's own
/// `kAsteroidRealRadiusMeters`/`kKuiperRealRadiusMeters` below have.
constexpr double kRingParticleRealRadiusMeters = 3.0;

/// The real main asteroid belt's own semi-major-axis range, 2.1-3.3 AU
/// (between Mars and Jupiter, where Kirkwood-gap-clearing resonances with
/// Jupiter bound it on either side) and the real, if rougher, main
/// eccentricity/inclination spread real belt asteroids occupy. This
/// populates the *shape* of the belt; it says nothing about individual
/// belt resonance structure (Kirkwood gaps within the belt itself), which
/// only emerges from actual gravitational interaction with Jupiter -- not
/// modeled here, since every particle is an independent, non-interacting
/// Kepler orbit. `kAsteroidBeltSeed` is fixed so the belt looks the same
/// from one run to the next.
constexpr double kAsteroidBeltMinAu = 2.1;
constexpr double kAsteroidBeltMaxAu = 3.3;
constexpr double kAsteroidBeltMaxEccentricity = 0.3;
constexpr double kAsteroidBeltMaxInclinationDeg = 20.0;
constexpr int kAsteroidBeltCount = 4000;
constexpr std::uint64_t kAsteroidBeltSeed = 1;
/// A representative real belt asteroid: most of the belt by count is small
/// bodies well under this, a few (Vesta, Pallas) are far larger -- there is
/// no one real size for an unnamed, procedurally-placed asteroid, so this
/// is a typical one, not a measurement.
constexpr double kAsteroidRealRadiusMeters = 2000.0;

/// The real classical Kuiper belt's own rough semi-major-axis range,
/// 30-50 AU (Neptune's own orbit out to the 1:2 mean-motion resonance),
/// where Pluto itself (a resonant, not classical, Kuiper belt object) sits.
/// Same non-interacting-particle caveat as the asteroid belt above: real
/// Kuiper belt structure (resonant clumping at Pluto's own 3:2 resonance,
/// for instance) is a real gravitational effect this population does not
/// reproduce.
constexpr double kKuiperBeltMinAu = 30.0;
constexpr double kKuiperBeltMaxAu = 50.0;
constexpr double kKuiperBeltMaxEccentricity = 0.25;
constexpr double kKuiperBeltMaxInclinationDeg = 30.0;
constexpr int kKuiperBeltCount = 3000;
constexpr std::uint64_t kKuiperBeltSeed = 2;
/// A representative real Kuiper belt object among the ones actually
/// catalogued (most of the real population, uncatalogued and far smaller,
/// is well under this) -- a typical size, not a measurement, the same
/// status `kAsteroidRealRadiusMeters` above has.
constexpr double kKuiperRealRadiusMeters = 50000.0;

}  // namespace

std::optional<Scenario> makeScenario(std::string* error) {
    CsvError csvError;
    const std::optional<Csv> table = Csv::load(kDataFilePath, &csvError);
    if (!table) {
        if (error != nullptr) {
            *error = std::format("{}: line {}: {}", kDataFilePath, csvError.line,
                                 csvError.message);
        }
        return std::nullopt;
    }

    std::optional<std::vector<applications::KeplerCatalogBody>> bodies =
        applications::loadKeplerBodyCatalog(*table, eclipticToEquatorialRotation(),
                                            applications::kJ2000JulianDate, error);
    if (!bodies) {
        return std::nullopt;
    }

    Scenario scenario;
    scenario.bodies = std::move(*bodies);

    // bodies[0] is the Sun (the one root row; see loadKeplerBodyCatalog's
    // own doc comment), the same convention Applications::SolarSystem's own
    // Scenario relies on.
    const double gmSun = constants::G.value() * scenario.bodies[0].massKg;

    // Every Sun-parented body (planet or dwarf planet, never a moon: a
    // moon's own dominant nearby source is its planet, not the Sun) gets
    // the real relativistic perihelion precession rate implied by its own
    // (a, e) around the Sun -- Physics/Gravity/PostNewtonian.hpp's own
    // closed form, converted from radians-per-orbit to radians-per-second
    // by its own mean motion. This is what a fixed, non-precessing Kepler
    // ellipse cannot show on its own; see
    // src/Applications/KeplerSolarSystem/README.md.
    for (applications::KeplerCatalogBody& body : scenario.bodies) {
        if (body.parent != "Sun" || !body.elements.has_value()) {
            continue;
        }
        const double a = body.elements->semiMajorAxis;
        const double e = body.elements->eccentricity;
        const double meanMotion = keplerMeanMotion(gmSun, a);
        const double precessionPerOrbit = perihelionPrecessionPerOrbit(gmSun, a, e);
        body.elements->precessionRatePerSecond = precessionPerOrbit * meanMotion / kTau<double>;
    }

    const double auMeters = units::astronomicalUnit.value();

    scenario.asteroidBelt = applications::generateKeplerPopulation(
        /*parentIndex=*/0, gmSun, kAsteroidBeltMinAu * auMeters, kAsteroidBeltMaxAu * auMeters,
        kAsteroidBeltMaxEccentricity, radians(kAsteroidBeltMaxInclinationDeg),
        kAsteroidBeltCount, kAsteroidBeltSeed, kAsteroidRealRadiusMeters,
        Vec3f{0.55f, 0.5f, 0.42f});

    scenario.kuiperBelt = applications::generateKeplerPopulation(
        /*parentIndex=*/0, gmSun, kKuiperBeltMinAu * auMeters, kKuiperBeltMaxAu * auMeters,
        kKuiperBeltMaxEccentricity, radians(kKuiperBeltMaxInclinationDeg), kKuiperBeltCount,
        kKuiperBeltSeed, kKuiperRealRadiusMeters, Vec3f{0.6f, 0.65f, 0.72f});

    // Each ring orbits its own planet, not the Sun: a real particle's own
    // gravitational parameter is that planet's, so its own period (and
    // real differential rotation against every other particle in the same
    // ring) comes out right.
    std::unordered_map<std::string, std::size_t> indexByName;
    for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
        indexByName[scenario.bodies[i].name] = i;
    }

    scenario.rings.reserve(kRingsKm.size());
    for (std::size_t ringIndex = 0; ringIndex < kRingsKm.size(); ++ringIndex) {
        const RingKm& ring = kRingsKm[ringIndex];
        const std::size_t parentIndex = indexByName.at(ring.parent);
        const double parentGm = constants::G.value() * scenario.bodies[parentIndex].massKg;

        const double innerMeters = ring.innerKm * 1000.0;
        const double outerMeters = ring.outerKm * 1000.0;

        RingPopulation population;
        population.parent = ring.parent;
        population.particles = applications::generateKeplerPopulation(
            static_cast<int>(parentIndex), parentGm, innerMeters, outerMeters,
            kRingMaxEccentricity, radians(kRingMaxInclinationDeg), kRingParticleCount,
            /*seed=*/2000 + ringIndex, kRingParticleRealRadiusMeters, kRingParticleColor);

        scenario.rings.push_back(std::move(population));
    }

    return scenario;
}

Vec3f toRenderPosition(const Vec3& metersPosition) {
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return Vec3f{static_cast<float>(metersPosition.x * scale),
                static_cast<float>(metersPosition.y * scale),
                static_cast<float>(metersPosition.z * scale)};
}

float toRenderRadius(double metersRadius) {
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return static_cast<float>(metersRadius * scale);
}

}  // namespace ysq::kepler_solar_system
