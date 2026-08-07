#include <Applications/Helper/BodyCatalog.hpp>

#include <Applications/Helper/Pole.hpp>
#include <Math/Scalar.hpp>
#include <Physics/Gravity/Kepler.hpp>
#include <Physics/Gravity/Newtonian.hpp>

#include <cmath>
#include <format>
#include <functional>
#include <unordered_map>

namespace ysq::applications {

namespace {

struct RawRow {
    std::string name;
    std::string parent;
    double massKg = 0.0;
    double radiusKm = 0.0;
    Vec3f color{};
    bool hasElements = false;
    // The orbital shape, unlike trueAnomaly, does not depend on epoch:
    // these five are filled in directly. The sixth, meanAnomalyAtRowEpoch,
    // needs propagating to the target epoch before it becomes a
    // trueAnomaly; that needs the parent's mass, not known until resolve().
    double semiMajorAxisMeters = 0.0;
    double eccentricity = 0.0;
    double inclination = 0.0;
    double longitudeOfAscendingNode = 0.0;
    double argumentOfPeriapsis = 0.0;
    double meanAnomalyAtRowEpoch = 0.0;
    double rowEpochJulianDate = 0.0;
    bool hasPole = false;
    Quat poleFrame{};
};

bool fail(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

/// Parses one row into a RawRow, or reports why it could not and returns
/// false. Every failure names the row's own source line and, once known,
/// its name -- the same "tell the caller exactly where" `Csv`'s own parse
/// errors already do.
bool parseRow(const Csv::Row& row, double targetEpochJulianDate, RawRow& out,
             std::string* error) {
    if (!row.has("name")) {
        return fail(error, std::format("row {}: missing column 'name'", row.lineNumber()));
    }
    out.name = row.get<std::string>("name", "");
    if (out.name.empty()) {
        return fail(error, std::format("row {}: empty name", row.lineNumber()));
    }

    out.parent = row.get<std::string>("parent", "");

    const std::optional<double> mass = row.tryGet<double>("mass_kg");
    if (!mass) {
        return fail(error, std::format("row {} ('{}'): missing or unparsable mass_kg",
                                       row.lineNumber(), out.name));
    }
    out.massKg = *mass;

    const std::optional<double> radius = row.tryGet<double>("radius_km");
    if (!radius) {
        return fail(error, std::format("row {} ('{}'): missing or unparsable radius_km",
                                       row.lineNumber(), out.name));
    }
    out.radiusKm = *radius;

    const std::optional<double> colorR = row.tryGet<double>("color_r");
    const std::optional<double> colorG = row.tryGet<double>("color_g");
    const std::optional<double> colorB = row.tryGet<double>("color_b");
    if (!colorR || !colorG || !colorB) {
        return fail(error, std::format("row {} ('{}'): missing or unparsable color",
                                       row.lineNumber(), out.name));
    }
    out.color = Vec3f{static_cast<float>(*colorR), static_cast<float>(*colorG),
                      static_cast<float>(*colorB)};

    const bool isRoot = out.parent.empty();
    const std::optional<double> semiMajorAxisKm = row.tryGet<double>("semi_major_axis_km");

    if (isRoot && semiMajorAxisKm.has_value()) {
        return fail(error, std::format(
                               "row {} ('{}'): a root body (empty parent) must not have "
                               "orbital elements",
                               row.lineNumber(), out.name));
    }
    if (!isRoot && !semiMajorAxisKm.has_value()) {
        return fail(error,
                    std::format("row {} ('{}'): a non-root body must have orbital elements",
                               row.lineNumber(), out.name));
    }

    if (semiMajorAxisKm) {
        const std::optional<double> eccentricity = row.tryGet<double>("eccentricity");
        const std::optional<double> inclinationDeg = row.tryGet<double>("inclination_deg");
        const std::optional<double> nodeDeg =
            row.tryGet<double>("longitude_of_ascending_node_deg");
        const std::optional<double> argPeriapsisDeg =
            row.tryGet<double>("argument_of_periapsis_deg");
        const std::optional<double> meanAnomalyDeg = row.tryGet<double>("mean_anomaly_deg");
        if (!eccentricity || !inclinationDeg || !nodeDeg || !argPeriapsisDeg ||
            !meanAnomalyDeg) {
            return fail(error, std::format("row {} ('{}'): incomplete orbital elements",
                                           row.lineNumber(), out.name));
        }

        out.hasElements = true;
        out.semiMajorAxisMeters = *semiMajorAxisKm * 1000.0;  // km -> m
        out.eccentricity = *eccentricity;
        out.inclination = radians(*inclinationDeg);
        out.longitudeOfAscendingNode = radians(*nodeDeg);
        out.argumentOfPeriapsis = radians(*argPeriapsisDeg);
        out.meanAnomalyAtRowEpoch = radians(*meanAnomalyDeg);
        out.rowEpochJulianDate =
            row.get<double>("epoch_jd", targetEpochJulianDate);

        const std::optional<double> poleRaDeg = row.tryGet<double>("pole_ra_deg");
        const std::optional<double> poleDecDeg = row.tryGet<double>("pole_dec_deg");
        if (poleRaDeg.has_value() != poleDecDeg.has_value()) {
            return fail(error, std::format("row {} ('{}'): pole_ra_deg and pole_dec_deg must "
                                           "be given together",
                                           row.lineNumber(), out.name));
        }
        if (poleRaDeg) {
            out.hasPole = true;
            out.poleFrame = poleRotation(radians(*poleRaDeg), radians(*poleDecDeg));
        }
    }

    return true;
}

}  // namespace

std::optional<std::vector<CatalogBody>> loadBodyCatalog(const Csv& table,
                                                         const Quat& referenceFrameRotation,
                                                         double targetEpochJulianDate,
                                                         std::string* error) {
    std::vector<RawRow> rows(table.rowCount());
    for (std::size_t i = 0; i < table.rowCount(); ++i) {
        if (!parseRow(table.row(i), targetEpochJulianDate, rows[i], error)) {
            return std::nullopt;
        }
    }

    std::unordered_map<std::string, std::size_t> indexByName;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!indexByName.insert({rows[i].name, i}).second) {
            fail(error, std::format("duplicate body name '{}'", rows[i].name));
            return std::nullopt;
        }
    }

    std::size_t rootCount = 0;
    for (const RawRow& raw : rows) {
        if (raw.parent.empty()) {
            ++rootCount;
        }
    }
    if (rootCount != 1) {
        fail(error, std::format("expected exactly one root body (empty parent), found {}",
                                rootCount));
        return std::nullopt;
    }

    for (const RawRow& raw : rows) {
        if (!raw.parent.empty() && !indexByName.contains(raw.parent)) {
            fail(error,
                std::format("body '{}' has unknown parent '{}'", raw.name, raw.parent));
            return std::nullopt;
        }
    }

    // Resolve every body's absolute state, memoized and cycle-checked: a
    // body's position depends on its parent's already-resolved one, and
    // the file need not list parents before their own children for this to
    // work.
    std::vector<std::optional<Body>> resolved(rows.size());
    enum class VisitState { Unvisited, InProgress, Done };
    std::vector<VisitState> visit(rows.size(), VisitState::Unvisited);
    bool ok = true;

    std::function<void(std::size_t)> resolve = [&](std::size_t i) {
        if (!ok || visit[i] == VisitState::Done) {
            return;
        }
        if (visit[i] == VisitState::InProgress) {
            fail(error, std::format("body '{}' is part of a parent cycle", rows[i].name));
            ok = false;
            return;
        }
        visit[i] = VisitState::InProgress;

        const RawRow& raw = rows[i];
        if (raw.parent.empty()) {
            Body body;
            body.mass = Mass{raw.massKg};
            body.radius = Length{raw.radiusKm * 1000.0};
            resolved[i] = body;
            visit[i] = VisitState::Done;
            return;
        }

        const std::size_t parentIndex = indexByName.at(raw.parent);
        resolve(parentIndex);
        if (!ok) {
            return;
        }
        const Body& parentBody = *resolved[parentIndex];

        const double gm = constants::G.value() * parentBody.mass.value();

        // Mean motion from the two-body relation n = sqrt(gm / a^3), used
        // only to carry a row's own mean anomaly forward (or back) from
        // whatever epoch it was published at to the target epoch; the
        // orbit's shape itself (a, e, i, node, argument of periapsis) does
        // not depend on epoch at all.
        const double meanMotion = keplerMeanMotion(gm, raw.semiMajorAxisMeters);
        const double elapsedSeconds =
            (targetEpochJulianDate - raw.rowEpochJulianDate) * 86400.0;
        const double meanAnomalyAtTarget =
            raw.meanAnomalyAtRowEpoch + meanMotion * elapsedSeconds;

        OrbitalElements elements{};
        elements.semiMajorAxis = raw.semiMajorAxisMeters;
        elements.eccentricity = raw.eccentricity;
        elements.inclination = raw.inclination;
        elements.longitudeOfAscendingNode = raw.longitudeOfAscendingNode;
        elements.argumentOfPeriapsis = raw.argumentOfPeriapsis;
        elements.trueAnomaly =
            trueAnomalyFromMeanAnomaly(meanAnomalyAtTarget, raw.eccentricity);

        const KeplerStateVector local = stateVectorFromElements(elements, gm);

        const Quat& frame = raw.hasPole ? raw.poleFrame : referenceFrameRotation;
        const Vec3 relativePosition = rotate(frame, local.position);
        const Vec3 relativeVelocity = rotate(frame, local.velocity);

        Body body;
        body.mass = Mass{raw.massKg};
        body.radius = Length{raw.radiusKm * 1000.0};
        body.position = Length3{parentBody.position.value() + relativePosition};
        body.momentum =
            Momentum3{(parentBody.velocity().value() + relativeVelocity) * raw.massKg};

        resolved[i] = body;
        visit[i] = VisitState::Done;
    };

    for (std::size_t i = 0; i < rows.size() && ok; ++i) {
        resolve(i);
    }
    if (!ok) {
        return std::nullopt;
    }

    std::vector<CatalogBody> result;
    result.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        result.push_back(
            CatalogBody{rows[i].name, rows[i].parent, rows[i].color, *resolved[i]});
    }
    return result;
}

std::optional<std::vector<KeplerCatalogBody>>
loadKeplerBodyCatalog(const Csv& table, const Quat& referenceFrameRotation,
                      double targetEpochJulianDate, std::string* error) {
    std::vector<RawRow> rows(table.rowCount());
    for (std::size_t i = 0; i < table.rowCount(); ++i) {
        if (!parseRow(table.row(i), targetEpochJulianDate, rows[i], error)) {
            return std::nullopt;
        }
    }

    std::unordered_map<std::string, std::size_t> indexByName;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!indexByName.insert({rows[i].name, i}).second) {
            fail(error, std::format("duplicate body name '{}'", rows[i].name));
            return std::nullopt;
        }
    }

    std::size_t rootCount = 0;
    for (const RawRow& raw : rows) {
        if (raw.parent.empty()) {
            ++rootCount;
        }
    }
    if (rootCount != 1) {
        fail(error, std::format("expected exactly one root body (empty parent), found {}",
                                rootCount));
        return std::nullopt;
    }

    for (const RawRow& raw : rows) {
        if (!raw.parent.empty() && !indexByName.contains(raw.parent)) {
            fail(error,
                std::format("body '{}' has unknown parent '{}'", raw.name, raw.parent));
            return std::nullopt;
        }
    }

    // No recursive position to resolve here (unlike loadBodyCatalog above):
    // every field a KeplerCatalogBody needs comes from its own row and its
    // immediate parent's, not the whole chain up to the root. A parent
    // cycle still makes the catalog malformed, though, so it is still
    // rejected here, just by walking each row's own parent chain up to
    // rows.size() steps rather than loadBodyCatalog's memoized DFS: this
    // graph is a tree, not a general one, so a cycle is exactly "the root
    // was not reached in time".
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::size_t current = i;
        std::size_t steps = 0;
        while (!rows[current].parent.empty()) {
            current = indexByName.at(rows[current].parent);
            ++steps;
            if (steps > rows.size()) {
                fail(error, std::format("body '{}' is part of a parent cycle", rows[i].name));
                return std::nullopt;
            }
        }
    }

    std::vector<KeplerCatalogBody> result;
    result.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const RawRow& raw = rows[i];

        KeplerCatalogBody body;
        body.name = raw.name;
        body.parent = raw.parent;
        body.color = raw.color;
        body.massKg = raw.massKg;
        body.radiusMeters = raw.radiusKm * 1000.0;

        if (raw.parent.empty()) {
            result.push_back(std::move(body));
            continue;
        }

        body.parentIndex = static_cast<int>(indexByName.at(raw.parent));
        const RawRow& parentRaw = rows[static_cast<std::size_t>(body.parentIndex)];
        const double gm = constants::G.value() * parentRaw.massKg;
        body.parentGm = gm;

        // Same epoch-propagation relation as loadBodyCatalog's own
        // resolve(): mean motion n = sqrt(gm / a^3), carrying this row's
        // own mean anomaly forward (or back) from whatever epoch it was
        // published at to targetEpochJulianDate -- meanAnomalyAtEpoch below
        // is exactly the mean anomaly at elapsedSeconds = 0 from that
        // target epoch, matching stateVectorAtTime's own convention.
        const double meanMotion = keplerMeanMotion(gm, raw.semiMajorAxisMeters);
        const double elapsedSeconds =
            (targetEpochJulianDate - raw.rowEpochJulianDate) * 86400.0;

        OrbitalElementsAtEpoch elements{};
        elements.semiMajorAxis = raw.semiMajorAxisMeters;
        elements.eccentricity = raw.eccentricity;
        elements.inclination = raw.inclination;
        elements.longitudeOfAscendingNode = raw.longitudeOfAscendingNode;
        elements.argumentOfPeriapsis = raw.argumentOfPeriapsis;
        elements.meanAnomalyAtEpoch = raw.meanAnomalyAtRowEpoch + meanMotion * elapsedSeconds;
        body.elements = elements;

        body.frameRotation = raw.hasPole ? raw.poleFrame : referenceFrameRotation;

        result.push_back(std::move(body));
    }
    return result;
}

}  // namespace ysq::applications
