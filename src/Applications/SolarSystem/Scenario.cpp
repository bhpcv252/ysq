#include <Applications/SolarSystem/Scenario.hpp>

#include <Core/Csv.hpp>
#include <Core/ExecutablePath.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Units/Length.hpp>
#include <Units/Unit.hpp>

#include <filesystem>
#include <format>
#include <utility>

namespace ysq::solar_system {

namespace {

/// Only reached on a local, uninstalled build (see CMakeLists.txt): a
/// packaged release ships its own copy right next to the executable
/// instead, which resolveDataFilePath() below finds first.
constexpr const char* kFallbackDataFilePath =
    YSQ_SOLAR_SYSTEM_DATA_DIR "/solar_system_bodies.csv";

/// Looks for the catalog next to whichever executable is actually running
/// (the app itself in a packaged release; a test binary in the build tree)
/// via Core::executableDirectory, so a relocated copy of the app finds its
/// own data regardless of what directory it was unzipped into. Falls back
/// to the compile-time source path only when no such shipped copy exists --
/// the ordinary case for a local build or a test linking this library
/// directly, neither of which is ever installed.
[[nodiscard]] std::filesystem::path resolveDataFilePath() {
    if (const std::optional<std::filesystem::path> exeDir = executableDirectory()) {
        std::filesystem::path shipped = *exeDir / "data" / "solar_system_bodies.csv";
        if (std::filesystem::exists(shipped)) {
            return shipped;
        }
    }
    return kFallbackDataFilePath;
}

/// The J2000 mean obliquity, 23.4392911 degrees: the fixed rotation about
/// the shared vernal-equinox axis that carries the planets' own
/// ecliptic-referenced elements into the equatorial frame every moon's
/// pole_ra/pole_dec already targets. See
/// src/Applications/Helper/README.md's Pole section for the derivation and
/// why the two kinds of real data do not share a frame to begin with.
[[nodiscard]] Quat eclipticToEquatorialRotation() {
    return Quat::fromAxisAngle(Vec3::unitX(), radians(23.4392911));
}

}  // namespace

std::optional<Scenario> makeScenario(std::string* error) {
    const std::filesystem::path dataFilePath = resolveDataFilePath();
    CsvError csvError;
    const std::optional<Csv> table = Csv::load(dataFilePath, &csvError);
    if (!table) {
        if (error != nullptr) {
            *error = std::format("{}: line {}: {}", dataFilePath.string(), csvError.line,
                                 csvError.message);
        }
        return std::nullopt;
    }

    std::optional<std::vector<applications::CatalogBody>> bodies =
        applications::loadBodyCatalog(*table, eclipticToEquatorialRotation(),
                                      applications::kJ2000JulianDate, error);
    if (!bodies) {
        return std::nullopt;
    }

    // Center-of-mass convention: the Sun (bodies[0], the one root) balances
    // every other body's momentum, so the system's total is exactly zero,
    // the same setup LunarEclipse's own scenario uses.
    Vec3 sunMomentum = Vec3::zero();
    for (std::size_t i = 1; i < bodies->size(); ++i) {
        sunMomentum -= (*bodies)[i].body.momentum.value();
    }
    (*bodies)[0].body.momentum = Momentum3{sunMomentum};

    Scenario scenario;
    scenario.bodies = std::move(*bodies);
    return scenario;
}

std::vector<Body> Scenario::allBodies() const {
    std::vector<Body> result;
    result.reserve(bodies.size());
    for (const applications::CatalogBody& catalogBody : bodies) {
        result.push_back(catalogBody.body);
    }
    return result;
}

Vec3f toRenderPosition(const Length3& position) {
    const Vec3 meters = position.value();
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return Vec3f{static_cast<float>(meters.x * scale),
                 static_cast<float>(meters.y * scale),
                 static_cast<float>(meters.z * scale)};
}

float toRenderRadius(Length radius) {
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return static_cast<float>(radius.value() * scale);
}

}  // namespace ysq::solar_system
