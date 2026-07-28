#include <Physics/Electromagnetism/Lorentz.hpp>

#include <Math/Vector3.hpp>

namespace ysq {

Force3 lorentzForce(const Body& body, const ElectricField3& electric,
                    const MagneticFluxDensity3& magnetic) {
    const Vec3 magneticTerm = cross(body.velocity().value(), magnetic.value());
    const Vec3 total = electric.value() + magneticTerm;
    return Force3{total * body.charge.value()};
}

}  // namespace ysq
