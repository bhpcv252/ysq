#include <Physics/Gravity/PostNewtonian.hpp>

#include <Physics/Gravity/Newtonian.hpp>
#include <Units/Constants.hpp>

namespace ysq {

Acceleration3 postNewtonianCorrection(const Body& testParticle, const Body& source) {
    const Vec3 r = (testParticle.position - source.position).value();
    const Vec3 v = (testParticle.velocity() - source.velocity()).value();

    const double rMag = length(r);
    const Vec3 n = r / rMag;
    const double vMagSquared = lengthSquared(v);
    const double radialSpeed = dot(v, n);

    const double gm = constants::G.value() * source.mass.value();
    const double c = constants::speedOfLight.value();

    const Vec3 accel = (n * (4.0 * gm / rMag - vMagSquared) + v * (4.0 * radialSpeed)) *
                       (gm / (c * c * rMag * rMag));
    return Acceleration3{accel};
}

}  // namespace ysq
