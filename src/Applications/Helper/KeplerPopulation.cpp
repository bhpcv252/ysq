#include <Applications/Helper/KeplerPopulation.hpp>

#include <Math/Scalar.hpp>

#include <random>

namespace ysq::applications {

std::vector<KeplerParticle> generateKeplerPopulation(
    int parentIndex, double parentGm, double minSemiMajorAxis, double maxSemiMajorAxis,
    double maxEccentricity, double maxInclination, int count, std::uint64_t seed,
    double realRadiusMeters, const Vec3f& color) {
    std::vector<KeplerParticle> result;
    if (count <= 0) {
        return result;
    }
    result.reserve(static_cast<std::size_t>(count));

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> semiMajorAxisDist(minSemiMajorAxis,
                                                              maxSemiMajorAxis);
    std::uniform_real_distribution<double> eccentricityDist(0.0, maxEccentricity);
    std::uniform_real_distribution<double> inclinationDist(0.0, maxInclination);
    std::uniform_real_distribution<double> angleDist(0.0, kTau<double>);

    for (int i = 0; i < count; ++i) {
        OrbitalElementsAtEpoch elements{};
        elements.semiMajorAxis = semiMajorAxisDist(rng);
        elements.eccentricity = eccentricityDist(rng);
        elements.inclination = inclinationDist(rng);
        elements.longitudeOfAscendingNode = angleDist(rng);
        elements.argumentOfPeriapsis = angleDist(rng);
        elements.meanAnomalyAtEpoch = angleDist(rng);
        // precessionRatePerSecond stays 0: real GR precession is
        // negligible at belt/ring scale, and a synthetic swarm has no
        // per-particle need for it the way a named body might.

        result.push_back(KeplerParticle{parentIndex, parentGm, elements, realRadiusMeters,
                                        color});
    }

    return result;
}

}  // namespace ysq::applications
