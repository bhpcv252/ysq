#include <Applications/Helper/Pole.hpp>

#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

namespace ysq::applications {

Quat poleRotation(double rightAscension, double declination) {
    const double halfPi = kPi<double> / 2.0;
    const Quat aboutX = Quat::fromAxisAngle(Vec3::unitX(), halfPi - declination);
    const Quat aboutZ = Quat::fromAxisAngle(Vec3::unitZ(), rightAscension + halfPi);
    return aboutZ * aboutX;
}

}  // namespace ysq::applications
