#include <Applications/SolarSystem/Scenario.hpp>

#include <Core/Csv.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Units/Length.hpp>
#include <Units/Unit.hpp>

#include <format>
#include <utility>

namespace ysq::solar_system {

namespace {

/// Baked in at build time (see CMakeLists.txt): an absolute path is robust
/// to whatever directory the built executable is actually run from, unlike
/// a path relative to the current working directory.
constexpr const char* kDataFilePath = YSQ_SOLAR_SYSTEM_DATA_DIR "/solar_system_bodies.csv";

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
    CsvError csvError;
    const std::optional<Csv> table = Csv::load(kDataFilePath, &csvError);
    if (!table) {
        if (error != nullptr) {
            *error = std::format("{}: line {}: {}", kDataFilePath, csvError.line,
                                 csvError.message);
        }
        return std::nullopt;
    }

    std::optional<std::vector<applications::CatalogBody>> bodies = applications::loadBodyCatalog(
        *table, eclipticToEquatorialRotation(), applications::kJ2000JulianDate, error);
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
    return Vec3f{static_cast<float>(meters.x * scale), static_cast<float>(meters.y * scale),
                static_cast<float>(meters.z * scale)};
}

float toRenderRadius(Length radius) {
    const double scale =
        static_cast<double>(kRenderUnitsPerAu) / units::astronomicalUnit.value();
    return static_cast<float>(radius.value() * scale);
}

}  // namespace ysq::solar_system
