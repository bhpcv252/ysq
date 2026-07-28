#include <Physics/Electromagnetism/Field.hpp>

#include <Math/Vector3.hpp>

#include <cmath>

namespace ysq {

ElectricField3 electricField(const Length3& at, std::span<const Body> sources) {
    const Vec3 rawAt = at.value();
    Vec3 total{};

    for (const Body& source : sources) {
        const Vec3 delta = rawAt - source.position.value();
        const double r2 = lengthSquared(delta);
        if (r2 <= 0.0) {
            continue;
        }
        const double r = std::sqrt(r2);
        total += delta * (source.charge.value() / (r2 * r));
    }
    return ElectricField3{total * constants::coulombConstant.value()};
}

MagneticFluxDensity3 magneticField(const Length3& at, std::span<const Body> sources) {
    const Vec3 rawAt = at.value();
    Vec3 total{};

    for (const Body& source : sources) {
        const Vec3 delta = rawAt - source.position.value();
        const double r2 = lengthSquared(delta);
        if (r2 <= 0.0) {
            continue;
        }
        const double r = std::sqrt(r2);
        const Vec3 velocity = source.velocity().value();
        total += cross(velocity, delta) * (source.charge.value() / (r2 * r));
    }
    return MagneticFluxDensity3{
        total * (constants::vacuumPermeability.value() / (4.0 * kPi<double>))};
}

}  // namespace ysq
