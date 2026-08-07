#pragma once

#include <Core/Csv.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>
#include <Physics/Gravity/Kepler.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ysq::applications {

/// One body loaded from a catalog: its physical `Body` plus the bookkeeping
/// no physics primitive carries -- what to call it, what to render it in,
/// and what it orbits.
struct CatalogBody {
    std::string name;
    std::string parent;  // empty for a root body (nothing orbited)
    Vec3f color;
    Body body;
};

/// Loads a hierarchy of real bodies -- planets around a star, moons around
/// their planet, any real orbital system given as classical elements --
/// from a `Csv` table, resolving each body's absolute position and
/// velocity from its parent's (already resolved) state. General
/// scenario-setup infrastructure, not specific to any one solar system: the
/// column schema is the only thing tying this to a particular dataset, and
/// nothing about it assumes a Sun or eight planets.
///
/// **Required columns**: `name`, `parent` (empty for a root body -- exactly
/// one row must have an empty parent), `mass_kg`, `radius_km`, `color_r`,
/// `color_g`, `color_b`.
///
/// **Orbital columns, all empty together for a root body, all present
/// otherwise**: `semi_major_axis_km`, `eccentricity`, `inclination_deg`,
/// `longitude_of_ascending_node_deg`, `argument_of_periapsis_deg`,
/// `mean_anomaly_deg`. Angles are the mean anomaly convention real
/// published data uses (`Physics/Gravity/Kepler.hpp`'s
/// `trueAnomalyFromMeanAnomaly` handles the conversion internally); elements
/// are relative to the *parent's* own local reference plane, not a shared
/// global one.
///
/// **Optional columns**: `pole_ra_deg`, `pole_dec_deg`. Present together,
/// they give that local reference plane's own pole (a moon's Laplace
/// plane), in whatever frame the caller's `referenceFrameRotation` result
/// itself is expressed in -- see `Pole.hpp`. Absent, the elements are
/// assumed already expressed in the simulation's own shared frame and only
/// `referenceFrameRotation` is applied (the ecliptic-to-equatorial
/// obliquity tilt, for planetary data against an equatorial simulation
/// frame; identity if the caller's data and simulation frame already
/// agree).
///
/// **Optional column**: `epoch_jd`, a Julian Date. Real data is not all
/// published at the same reference epoch -- JPL's own satellite tables mix
/// J2000 with several later ones -- so a row whose `epoch_jd` differs from
/// `targetEpochJulianDate` has its mean anomaly propagated forward first,
/// at the mean motion `n = sqrt(gm / a^3)` its own (already-resolved)
/// parent and semi-major axis imply. Absent, the row's data is assumed
/// already at `targetEpochJulianDate`, no propagation applied -- the
/// common case for data that does share the target epoch.
///
/// Returns `std::nullopt` on a malformed table (a missing required column,
/// an unparsable value, a parent that never resolves to a row, a cycle, or
/// anything other than exactly one root), with `error` set to a message
/// naming the offending row's line number where the table has one.
[[nodiscard]] std::optional<std::vector<CatalogBody>>
loadBodyCatalog(const Csv& table, const Quat& referenceFrameRotation,
                double targetEpochJulianDate, std::string* error = nullptr);

/// The J2000 epoch, 2000-01-01 12:00 TT/TDB, as a Julian Date: the standard
/// reference epoch real orbital element data is normally quoted at, and
/// this engine's own convenient default for `targetEpochJulianDate`.
inline constexpr double kJ2000JulianDate = 2451545.0;

/// One body loaded from a catalog for repeated, live re-evaluation, rather
/// than the one fixed initial Cartesian state `CatalogBody` above resolves
/// once for a real integrator to take over from. Everything needed to call
/// `Physics/Gravity/Kepler.hpp`'s `stateVectorAtTime` for this body at any
/// later time is kept: its own elements anchored at `targetEpochJulianDate`, which row
/// (by index into the same result vector) is its parent, that parent's own
/// gravitational parameter, and the local-to-shared frame rotation already
/// resolved from the row's pole (or `referenceFrameRotation`, if it had
/// none) -- the same composition `loadBodyCatalog` performs, just kept
/// instead of collapsed into one snapshot.
///
/// `elements`/`parentGm`/`frameRotation` are only meaningful when `parent`
/// is non-empty; the one root body (the Sun, in every real consumer) has
/// none of these, since nothing is orbited to propagate an ellipse around.
struct KeplerCatalogBody {
    std::string name;
    std::string parent;  // empty for the one root body
    Vec3f color;
    double massKg = 0.0;
    double radiusMeters = 0.0;
    int parentIndex = -1;  // -1 for the root
    std::optional<OrbitalElementsAtEpoch> elements;
    double parentGm = 0.0;
    Quat frameRotation = Quat::identity();
};

/// `loadBodyCatalog`'s own row parsing, parent resolution, and epoch
/// propagation, kept as live elements instead of collapsed to one fixed
/// state: for a caller that re-evaluates every body's position at
/// simulation time directly (a closed-form Kepler propagator) rather than
/// handing one initial condition to a real n-body integrator. Same column
/// schema, same validation, same errors; see `loadBodyCatalog`'s own doc
/// comment for both.
[[nodiscard]] std::optional<std::vector<KeplerCatalogBody>>
loadKeplerBodyCatalog(const Csv& table, const Quat& referenceFrameRotation,
                      double targetEpochJulianDate, std::string* error = nullptr);

}  // namespace ysq::applications
