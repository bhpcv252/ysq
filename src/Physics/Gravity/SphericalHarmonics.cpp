#include <Physics/Gravity/SphericalHarmonics.hpp>

#include <Math/Calculus.hpp>
#include <Math/Scalar.hpp>
#include <Math/SpecialFunctions.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Units/Unit.hpp>

#include <cmath>

namespace ysq {

double sphericalHarmonicsPotential(const SphericalHarmonicsField& field,
                                   const Vec3& position) {
    const double r = length(position);
    if (isNearZero(r)) {
        return 0.0;
    }
    const double gm = constants::G.value() * field.mass.value();

    // sin(latitude) = z / r directly, from the definition of geocentric
    // latitude in spherical coordinates; no need to round-trip through
    // asin then sin.
    const double sinLatitude = position.z / r;
    const double longitude = std::atan2(position.y, position.x);
    const double reOverR = field.referenceRadius.value() / r;

    double sum = 1.0;
    double reOverRPowN = reOverR * reOverR;  // (Re/r)^2, for degree n = 2 first
    for (std::size_t i = 0; i < field.cosine.size(); ++i) {
        const unsigned n = static_cast<unsigned>(i + 2);
        const std::vector<double>& cosineRow = field.cosine[i];
        const std::vector<double>& sineRow = field.sine[i];

        double innerSum = 0.0;
        for (unsigned m = 0; m <= n; ++m) {
            const double cnm = cosineRow[m];
            const double snm = sineRow[m];
            if (cnm == 0.0 && snm == 0.0) {
                continue;
            }
            const double associatedLegendre = legendreP(n, m, sinLatitude);
            const double mLongitude = static_cast<double>(m) * longitude;
            innerSum += associatedLegendre *
                        (cnm * std::cos(mLongitude) + snm * std::sin(mLongitude));
        }

        sum += reOverRPowN * innerSum;
        reOverRPowN *= reOverR;
    }

    return (gm / r) * sum;
}

Acceleration3 sphericalHarmonicsAcceleration(const SphericalHarmonicsField& field,
                                             const Length3& position) {
    const auto potential = [&field](const Vec3& p) {
        return sphericalHarmonicsPotential(field, p);
    };
    return Acceleration3{numericalGradient(potential, position.value())};
}

}  // namespace ysq
