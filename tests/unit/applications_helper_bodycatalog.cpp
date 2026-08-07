#include <Applications/Helper/BodyCatalog.hpp>

#include <Core/Csv.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Physics/Gravity/Newtonian.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <format>
#include <string>

namespace {

using ysq::applications::CatalogBody;
using ysq::applications::KeplerCatalogBody;
using ysq::applications::loadBodyCatalog;
using ysq::applications::loadKeplerBodyCatalog;
using ysq::radians;
using ysq::stateVectorAtTime;

const ysq::Quat kIdentity = ysq::Quat::identity();

std::optional<ysq::Csv> parseOrDie(const std::string& text) {
    ysq::CsvError error;
    std::optional<ysq::Csv> table = ysq::Csv::parse(text, &error);
    EXPECT_TRUE(table.has_value()) << "line " << error.line << ": " << error.message;
    return table;
}

const CatalogBody& find(const std::vector<CatalogBody>& bodies, const std::string& name) {
    for (const CatalogBody& body : bodies) {
        if (body.name == name) {
            return body;
        }
    }
    ADD_FAILURE() << "no body named '" << name << "'";
    return bodies.front();
}

const KeplerCatalogBody& findKepler(const std::vector<KeplerCatalogBody>& bodies,
                                    const std::string& name) {
    for (const KeplerCatalogBody& body : bodies) {
        if (body.name == name) {
            return body;
        }
    }
    ADD_FAILURE() << "no body named '" << name << "'";
    return bodies.front();
}

}  // namespace

TEST(ApplicationsHelperBodyCatalog, RootBodySitsAtTheOriginWithZeroVelocity) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;
    ASSERT_EQ(bodies->size(), 1u);

    const CatalogBody& star = find(*bodies, "Star");
    EXPECT_DOUBLE_EQ(length(star.body.position.value()), 0.0);
    EXPECT_DOUBLE_EQ(length(star.body.velocity().value()), 0.0);
    EXPECT_DOUBLE_EQ(star.body.mass.value(), 2e30);
}

TEST(ApplicationsHelperBodyCatalog, ACircularChildOrbitsAtExactlyItsSemiMajorAxis) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n"
        "Planet,Star,6e24,6000,0,0,1,150000000,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;

    const CatalogBody& planet = find(*bodies, "Planet");
    const double distanceMeters = length(planet.body.position.value());
    EXPECT_NEAR(distanceMeters, 150000000.0 * 1000.0, 1e-3);

    // Circular orbit: speed matches sqrt(GM/r) exactly.
    const double gm = ysq::constants::G.value() * 2e30;
    const double expectedSpeed = std::sqrt(gm / distanceMeters);
    EXPECT_NEAR(length(planet.body.velocity().value()), expectedSpeed, expectedSpeed * 1e-9);
}

TEST(ApplicationsHelperBodyCatalog, AMoonsAbsolutePositionAddsItsPlanetsOwnOffset) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n"
        "Planet,Star,6e24,6000,0,0,1,150000000,0,0,0,0,0,,\n"
        "Moon,Planet,7e22,1700,1,1,1,384400,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;

    const CatalogBody& planet = find(*bodies, "Planet");
    const CatalogBody& moon = find(*bodies, "Moon");

    // Both orbits are circular and coplanar with these elements (identity
    // frame, zero inclination/node/periapsis/anomaly), so the moon sits at
    // exactly planet-distance + moon-distance along the shared axis.
    const double expected = 150000000.0e3 + 384400.0e3;
    EXPECT_NEAR(moon.body.position.value().x, expected, 1e-3);

    EXPECT_EQ(moon.parent, "Planet");
    EXPECT_EQ(planet.parent, "Star");
}

TEST(ApplicationsHelperBodyCatalog, ReferenceFrameRotationAppliesOnlyWherePoleIsAbsent) {
    // Two children of the same star: one with no pole (gets the caller's
    // referenceFrameRotation), one with an explicit pole pointing along
    // +x (gets that instead, ignoring referenceFrameRotation entirely).
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n"
        "NoPole,Star,6e24,6000,0,0,1,1000,0,0,0,0,0,,\n"
        "WithPole,Star,6e24,6000,0,0,1,1000,0,0,0,0,0,90,0\n");
    ASSERT_TRUE(table.has_value());

    // A 90-degree rotation about x: chosen specifically because it produces
    // a result along a *different* axis than WithPole's own pole rotation
    // would, so using the wrong one for either body is unmissable rather
    // than an accidental match.
    const ysq::Quat ninetyAboutX =
        ysq::Quat::fromAxisAngle(ysq::Vec3::unitX(), ysq::kPi<double> / 2.0);

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, ninetyAboutX, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;

    const CatalogBody& noPole = find(*bodies, "NoPole");
    const CatalogBody& withPole = find(*bodies, "WithPole");

    // NoPole: circular, coplanar orbit at trueAnomaly 0 sits at
    // (1000 km, 0, 0) in the perifocal frame; referenceFrameRotation (90
    // degrees about x) leaves x untouched and sends +y to +z.
    const double distanceMeters = 1000.0 * 1000.0;
    EXPECT_NEAR(noPole.body.position.value().x, distanceMeters, 1e-3);
    EXPECT_NEAR(noPole.body.position.value().y, 0.0, 1e-6);
    EXPECT_NEAR(noPole.body.position.value().z, 0.0, 1e-6);

    // WithPole: poleRotation(90deg, 0) = Rz(180deg) . Rx(90deg) (Pole.hpp's
    // own construction). Applied to this body's own local +x position: a
    // rotation about x leaves a vector already on the x-axis unchanged, so
    // Rx(90deg) does nothing to it; Rz(180deg) then negates x -- landing on
    // -x, not on referenceFrameRotation's (unchanged +x) outcome at all.
    EXPECT_NEAR(withPole.body.position.value().x, -distanceMeters, 1e-3);
    EXPECT_NEAR(withPole.body.position.value().y, 0.0, 1e-3);
    EXPECT_NEAR(withPole.body.position.value().z, 0.0, 1e-3);
}

TEST(ApplicationsHelperBodyCatalog, MissingRootIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "A,B,1,1,1,1,1,1,0,0,0,0,0,,\n"
        "B,A,1,1,1,1,1,1,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
    EXPECT_FALSE(error.empty());
}

TEST(ApplicationsHelperBodyCatalog, MultipleRootsIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "A,,1,1,1,1,1,,,,,,,,\n"
        "B,,1,1,1,1,1,,,,,,,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalog, UnknownParentIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "Orphan,Nobody,1,1,1,1,1,1,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalog, ACycleAmongNonRootBodiesIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "A,B,1,1,1,1,1,1,0,0,0,0,0,,\n"
        "B,A,1,1,1,1,1,1,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalog, RootWithOrbitalElementsIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,1,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalog, NonRootWithoutOrbitalElementsIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "Orphan,Star,1,1,1,1,1,,,,,,,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalog, PoleRaWithoutPoleDecIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "Moon,Star,1,1,1,1,1,1,0,0,0,0,0,10,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalog, MissingRequiredColumnIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie("name,parent\nStar,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
    EXPECT_FALSE(error.empty());
}

TEST(ApplicationsHelperBodyCatalog, RowsAtTheTargetEpochAlreadyNeedNoPropagation) {
    // No epoch_jd column at all: rowEpochJulianDate defaults to
    // targetEpochJulianDate, so elapsed time is exactly zero and the row's
    // own mean anomaly is used unchanged. This is the common case for data
    // already published at the target epoch.
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n"
        "Planet,Star,6e24,6000,0,0,1,1000,0,0,0,0,90,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;

    // e = 0, so mean anomaly = true anomaly = 90 degrees exactly: position
    // lands on +y (see the perifocal formula's own reference case).
    const CatalogBody& planet = find(*bodies, "Planet");
    EXPECT_NEAR(planet.body.position.value().x, 0.0, 1e-3);
    EXPECT_NEAR(planet.body.position.value().y, 1000.0 * 1000.0, 1e-3);
}

TEST(ApplicationsHelperBodyCatalog, EpochJdPropagatesMeanAnomalyByExactlyOneQuarterPeriod) {
    // Independent derivation: pick rowEpoch = targetEpoch - period/4 for a
    // circular orbit starting at mean anomaly 0 (periapsis) at its own
    // epoch. By the time it reaches the target epoch it must have swept
    // exactly a quarter turn, landing on true anomaly 90 degrees -- the
    // same physical position RowsAtTheTargetEpochAlreadyNeedNoPropagation
    // checks directly, but reached here via propagation instead of being
    // given outright.
    //
    // Solar-mass/AU scale deliberately, not a toy: a period of a fraction
    // of a second (an arbitrarily small a and huge gm) would put rowEpoch
    // within roundoff of targetEpoch (both ~2.45e6 in magnitude), losing
    // the elapsed time itself to cancellation before propagation ever
    // runs. A roughly year-long period keeps the subtraction well
    // conditioned.
    const double starMassKg = 1.989e30;
    const double semiMajorAxisMeters = 1.496e11;
    const double gm = ysq::constants::G.value() * starMassKg;
    const double period = ysq::kTau<double> *
                          std::sqrt(semiMajorAxisMeters * semiMajorAxisMeters *
                                    semiMajorAxisMeters / gm);
    const double targetEpoch = ysq::applications::kJ2000JulianDate;
    const double rowEpoch = targetEpoch - (period / 4.0) / 86400.0;

    const std::string csvText = std::format(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg,epoch_jd\n"
        "Star,,{},700000,1,1,1,,,,,,,,,\n"
        "Planet,Star,6e24,6000,0,0,1,{},0,0,0,0,0,,,{}\n",
        starMassKg, semiMajorAxisMeters / 1000.0, rowEpoch);
    const std::optional<ysq::Csv> table = parseOrDie(csvText);
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, targetEpoch, &error);
    ASSERT_TRUE(bodies.has_value()) << error;

    const CatalogBody& planet = find(*bodies, "Planet");
    EXPECT_NEAR(planet.body.position.value().x, 0.0, semiMajorAxisMeters * 1e-9);
    EXPECT_NEAR(planet.body.position.value().y, semiMajorAxisMeters,
               semiMajorAxisMeters * 1e-9);
}

TEST(ApplicationsHelperBodyCatalog, TheRealSolarSystemDataFileLoadsToRealDistancesAndOrder) {
    // Sanity check on the actual curated data file, not synthetic input:
    // catches a transcription mistake (a wrong column, a stray AU-vs-km
    // slip) that a synthetic-CSV test can never exercise.
    ysq::CsvError csvError;
    const std::optional<ysq::Csv> table =
        ysq::Csv::load(YSQ_SOLAR_SYSTEM_DATA_DIR "/solar_system_bodies.csv", &csvError);
    ASSERT_TRUE(table.has_value()) << "line " << csvError.line << ": " << csvError.message;

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;
    // Sun + 8 planets + every moon; see
    // TheRealSolarSystemDataFileFullyResolvesEveryMoon for the per-body
    // validation across all of them. This test's own focus is narrower:
    // the planets specifically, and their ordering from the Sun.
    ASSERT_EQ(bodies->size(), table->rowCount());

    const CatalogBody& sun = find(*bodies, "Sun");
    EXPECT_TRUE(sun.parent.empty());
    EXPECT_NEAR(sun.body.mass.value(), 1.988409870698051e30, 1e20);
    EXPECT_DOUBLE_EQ(length(sun.body.position.value()), 0.0);

    const double auMeters = 1.495978707e11;
    const std::vector<std::pair<std::string, double>> expectedAu{
        {"Mercury", 0.387}, {"Venus", 0.723},  {"Earth", 1.000},  {"Mars", 1.524},
        {"Jupiter", 5.203}, {"Saturn", 9.537}, {"Uranus", 19.189}, {"Neptune", 30.070}};

    double previousDistance = 0.0;
    for (const auto& [name, semiMajorAxisAu] : expectedAu) {
        const CatalogBody& planet = find(*bodies, name);
        EXPECT_EQ(planet.parent, "Sun") << name;

        const double distanceMeters = length(planet.body.position.value());
        // Not exactly semiMajorAxisAu * auMeters: these are real, eccentric
        // orbits evaluated at a real mean anomaly, not periapsis, so
        // instantaneous distance ranges over exactly a(1 +/- e). Mercury's
        // real e = 0.2056 is the largest of the eight, so its distance can
        // legitimately swing +/-20.6% from the semi-major axis; 25% is a
        // real margin covering every planet's own eccentricity, not a
        // threshold picked to make the test pass.
        EXPECT_NEAR(distanceMeters, semiMajorAxisAu * auMeters,
                   semiMajorAxisAu * auMeters * 0.25)
            << name;
        EXPECT_GT(distanceMeters, previousDistance)
            << name << " should orbit farther out than the previous planet";
        previousDistance = distanceMeters;
    }
}

TEST(ApplicationsHelperBodyCatalog, TheRealSolarSystemDataFileFullyResolvesEveryMoon) {
    // The whole file, not just the planets: every body's distance from its
    // *own parent* (not the Sun) must fall within [a(1-e), a(1+e)] of that
    // row's own semi-major axis and eccentricity -- an exact, per-row
    // physical bound, read back from the same table rather than
    // hardcoded, so this catches a real transcription slip (a wrong
    // parent, a unit mismatch, a row that landed at the Sun's position
    // instead of its actual parent's) across all ~165 moons at once.
    ysq::CsvError csvError;
    const std::optional<ysq::Csv> table =
        ysq::Csv::load(YSQ_SOLAR_SYSTEM_DATA_DIR "/solar_system_bodies.csv", &csvError);
    ASSERT_TRUE(table.has_value()) << "line " << csvError.line << ": " << csvError.message;

    // J2000 mean obliquity, the fixed rotation that carries the planets'
    // (ecliptic-referenced) elements into the same equatorial frame the
    // satellite data's own pole_ra/pole_dec already targets; see
    // src/Applications/Helper/README.md's Pole section.
    const ysq::Quat eclipticToEquatorial =
        ysq::Quat::fromAxisAngle(ysq::Vec3::unitX(), radians(23.4392911));

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies = loadBodyCatalog(
        *table, eclipticToEquatorial, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;
    ASSERT_EQ(bodies->size(), table->rowCount());

    std::size_t nonRootBodiesChecked = 0;
    for (std::size_t i = 0; i < table->rowCount(); ++i) {
        const ysq::Csv::Row row = table->row(i);
        const std::string name = row.get<std::string>("name", "");
        const std::string parent = row.get<std::string>("parent", "");

        const CatalogBody& body = find(*bodies, name);

        ASSERT_TRUE(std::isfinite(body.body.position.value().x)) << name;
        ASSERT_TRUE(std::isfinite(body.body.position.value().y)) << name;
        ASSERT_TRUE(std::isfinite(body.body.position.value().z)) << name;
        ASSERT_TRUE(std::isfinite(length(body.body.velocity().value()))) << name;

        if (parent.empty()) {
            continue;
        }
        const std::optional<double> semiMajorAxisKm = row.tryGet<double>("semi_major_axis_km");
        const std::optional<double> eccentricity = row.tryGet<double>("eccentricity");
        ASSERT_TRUE(semiMajorAxisKm.has_value()) << name;
        ASSERT_TRUE(eccentricity.has_value()) << name;

        const CatalogBody& parentBody = find(*bodies, parent);
        const double distanceFromParent =
            length(body.body.position.value() - parentBody.body.position.value());
        const double aMeters = *semiMajorAxisKm * 1000.0;
        const double lo = aMeters * (1.0 - *eccentricity) * 0.999;
        const double hi = aMeters * (1.0 + *eccentricity) * 1.001;

        EXPECT_GE(distanceFromParent, lo) << name << " (parent " << parent << ")";
        EXPECT_LE(distanceFromParent, hi) << name << " (parent " << parent << ")";
        ++nonRootBodiesChecked;
    }

    EXPECT_EQ(nonRootBodiesChecked, table->rowCount() - 1);
}

TEST(ApplicationsHelperBodyCatalog, DuplicateNameIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "Star,,1,1,1,1,1,,,,,,,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> bodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalogKepler, RootBodyHasNoElementsAndNegativeParentIndex) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<KeplerCatalogBody>> bodies =
        loadKeplerBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(bodies.has_value()) << error;

    const KeplerCatalogBody& star = findKepler(*bodies, "Star");
    EXPECT_EQ(star.parentIndex, -1);
    EXPECT_FALSE(star.elements.has_value());
    EXPECT_DOUBLE_EQ(star.massKg, 2e30);
}

TEST(ApplicationsHelperBodyCatalogKepler,
    AtElapsedZeroStateVectorAtTimeAgreesWithLoadBodyCatalogsOwnResolvedPosition) {
    // The two loaders parse the same rows and the same epoch propagation;
    // they must describe the exact same physical state at elapsedSeconds =
    // 0 (which is targetEpochJulianDate), just one collapsed to a fixed
    // Body and the other kept live. This is the one test that would catch
    // loadKeplerBodyCatalog quietly disagreeing with the already-trusted
    // loadBodyCatalog about what a row's data actually means.
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,2e30,700000,1,1,1,,,,,,,,\n"
        "Planet,Star,6e24,6000,0,0,1,150000000,0.2,10,20,30,40,,\n"
        "Moon,Planet,7e22,1700,1,1,1,384400,0.05,5,15,25,35,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<CatalogBody>> catalogBodies =
        loadBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(catalogBodies.has_value()) << error;

    std::string keplerError;
    const std::optional<std::vector<KeplerCatalogBody>> keplerBodies = loadKeplerBodyCatalog(
        *table, kIdentity, ysq::applications::kJ2000JulianDate, &keplerError);
    ASSERT_TRUE(keplerBodies.has_value()) << keplerError;

    for (const char* name : {"Planet", "Moon"}) {
        const CatalogBody& expected = find(*catalogBodies, name);
        const KeplerCatalogBody& actual = findKepler(*keplerBodies, name);

        ASSERT_TRUE(actual.elements.has_value()) << name;
        const CatalogBody& parent = find(*catalogBodies, expected.parent);
        const auto local = stateVectorAtTime(*actual.elements, actual.parentGm, 0.0);
        const ysq::Vec3 rotatedPosition = rotate(actual.frameRotation, local.position);
        const ysq::Vec3 absolutePosition = parent.body.position.value() + rotatedPosition;

        EXPECT_NEAR(absolutePosition.x, expected.body.position.value().x, 1e-3) << name;
        EXPECT_NEAR(absolutePosition.y, expected.body.position.value().y, 1e-3) << name;
        EXPECT_NEAR(absolutePosition.z, expected.body.position.value().z, 1e-3) << name;
    }
}

TEST(ApplicationsHelperBodyCatalogKepler, MultipleRootsIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "A,,1,1,1,1,1,,,,,,,,\n"
        "B,,1,1,1,1,1,,,,,,,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<KeplerCatalogBody>> bodies =
        loadKeplerBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalogKepler, UnknownParentIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "Orphan,Nobody,1,1,1,1,1,1,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<KeplerCatalogBody>> bodies =
        loadKeplerBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalogKepler, ACycleAmongNonRootBodiesIsAnError) {
    const std::optional<ysq::Csv> table = parseOrDie(
        "name,parent,mass_kg,radius_km,color_r,color_g,color_b,semi_major_axis_km,"
        "eccentricity,inclination_deg,longitude_of_ascending_node_deg,"
        "argument_of_periapsis_deg,mean_anomaly_deg,pole_ra_deg,pole_dec_deg\n"
        "Star,,1,1,1,1,1,,,,,,,,\n"
        "A,B,1,1,1,1,1,1,0,0,0,0,0,,\n"
        "B,A,1,1,1,1,1,1,0,0,0,0,0,,\n");
    ASSERT_TRUE(table.has_value());

    std::string error;
    const std::optional<std::vector<KeplerCatalogBody>> bodies =
        loadKeplerBodyCatalog(*table, kIdentity, ysq::applications::kJ2000JulianDate, &error);
    EXPECT_FALSE(bodies.has_value());
}

TEST(ApplicationsHelperBodyCatalogKepler,
    TheRealSolarSystemDataFileCrossChecksAgainstLoadBodyCatalog) {
    // Same cross-check as
    // AtElapsedZeroStateVectorAtTimeAgreesWithLoadBodyCatalogsOwnResolvedPosition,
    // over the real, curated data file rather than a synthetic table: every
    // one of the ~175 real bodies, not just two.
    ysq::CsvError csvError;
    const std::optional<ysq::Csv> table =
        ysq::Csv::load(YSQ_SOLAR_SYSTEM_DATA_DIR "/solar_system_bodies.csv", &csvError);
    ASSERT_TRUE(table.has_value()) << "line " << csvError.line << ": " << csvError.message;

    const ysq::Quat eclipticToEquatorial =
        ysq::Quat::fromAxisAngle(ysq::Vec3::unitX(), radians(23.4392911));

    std::string error;
    const std::optional<std::vector<CatalogBody>> catalogBodies = loadBodyCatalog(
        *table, eclipticToEquatorial, ysq::applications::kJ2000JulianDate, &error);
    ASSERT_TRUE(catalogBodies.has_value()) << error;

    std::string keplerError;
    const std::optional<std::vector<KeplerCatalogBody>> keplerBodies = loadKeplerBodyCatalog(
        *table, eclipticToEquatorial, ysq::applications::kJ2000JulianDate, &keplerError);
    ASSERT_TRUE(keplerBodies.has_value()) << keplerError;
    ASSERT_EQ(keplerBodies->size(), catalogBodies->size());

    for (const CatalogBody& expected : *catalogBodies) {
        if (expected.parent.empty()) {
            continue;
        }
        const KeplerCatalogBody& actual = findKepler(*keplerBodies, expected.name);
        ASSERT_TRUE(actual.elements.has_value()) << expected.name;

        const CatalogBody& parent = find(*catalogBodies, expected.parent);
        const auto local = stateVectorAtTime(*actual.elements, actual.parentGm, 0.0);
        const ysq::Vec3 rotatedPosition = rotate(actual.frameRotation, local.position);
        const ysq::Vec3 absolutePosition = parent.body.position.value() + rotatedPosition;
        const double toleranceMeters =
            std::max(1e-3, length(expected.body.position.value()) * 1e-9);

        EXPECT_NEAR(absolutePosition.x, expected.body.position.value().x, toleranceMeters)
            << expected.name;
        EXPECT_NEAR(absolutePosition.y, expected.body.position.value().y, toleranceMeters)
            << expected.name;
        EXPECT_NEAR(absolutePosition.z, expected.body.position.value().z, toleranceMeters)
            << expected.name;
    }
}
