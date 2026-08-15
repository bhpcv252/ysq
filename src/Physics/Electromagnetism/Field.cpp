#include <Physics/Electromagnetism/Field.hpp>

#include <Compute/ComputeBackend.hpp>
#include <Math/Vector3.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ysq {

namespace {

// Measured on the development machine (Apple Silicon, Metal backend) by
// benchmarks/compute_thresholds.cpp: the below-threshold CPU path here
// (electricFields()/magneticFields()'s own full O(n^2) loop, each body
// against every other) matches Compute::CpuBackend's reference kernel
// shape exactly, unlike Gravity/Newtonian.cpp's own copy of this constant,
// so this is a direct, unadjusted crossover measurement. Only
// electricFieldNBody was measured; magneticFieldNBody's extra per-pair
// velocity/cross-product work only makes it more GPU-favorable, never
// less, so electric field's crossover is the safe (if marginally
// conservative for magnetic field) choice for both. Re-run the benchmark
// and update this if the reference machine or backend ever changes.
constexpr std::size_t kGpuDispatchThreshold = 512;

void extractPositions(std::span<const Body> bodies, std::vector<float>& x,
                      std::vector<float>& y, std::vector<float>& z) {
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Vec3 p = bodies[i].position.value();
        x[i] = static_cast<float>(p.x);
        y[i] = static_cast<float>(p.y);
        z[i] = static_cast<float>(p.z);
    }
}

}  // namespace

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

std::vector<ElectricField3> electricFields(std::span<const Body> bodies) {
    const std::size_t n = bodies.size();

    if (n >= kGpuDispatchThreshold) {
        std::vector<float> posX(n);
        std::vector<float> posY(n);
        std::vector<float> posZ(n);
        extractPositions(bodies, posX, posY, posZ);
        std::vector<float> charge(n);
        for (std::size_t i = 0; i < n; ++i) {
            charge[i] = static_cast<float>(bodies[i].charge.value());
        }

        std::vector<float> fieldX(n);
        std::vector<float> fieldY(n);
        std::vector<float> fieldZ(n);
        defaultBackend().electricFieldNBody(
            posX, posY, posZ, charge,
            static_cast<float>(constants::coulombConstant.value()), fieldX, fieldY,
            fieldZ);

        std::vector<ElectricField3> result(n);
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = ElectricField3{Vec3{static_cast<double>(fieldX[i]),
                                            static_cast<double>(fieldY[i]),
                                            static_cast<double>(fieldZ[i])}};
        }
        return result;
    }

    std::vector<Vec3> totals(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 at = bodies[i].position.value();
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const Vec3 delta = at - bodies[j].position.value();
            const double r2 = lengthSquared(delta);
            if (r2 <= 0.0) {
                continue;
            }
            const double r = std::sqrt(r2);
            totals[i] += delta * (bodies[j].charge.value() / (r2 * r));
        }
    }

    std::vector<ElectricField3> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = ElectricField3{totals[i] * constants::coulombConstant.value()};
    }
    return result;
}

std::vector<MagneticFluxDensity3> magneticFields(std::span<const Body> bodies) {
    const std::size_t n = bodies.size();

    if (n >= kGpuDispatchThreshold) {
        std::vector<float> posX(n);
        std::vector<float> posY(n);
        std::vector<float> posZ(n);
        extractPositions(bodies, posX, posY, posZ);
        std::vector<float> velX(n);
        std::vector<float> velY(n);
        std::vector<float> velZ(n);
        std::vector<float> charge(n);
        for (std::size_t i = 0; i < n; ++i) {
            const Vec3 v = bodies[i].velocity().value();
            velX[i] = static_cast<float>(v.x);
            velY[i] = static_cast<float>(v.y);
            velZ[i] = static_cast<float>(v.z);
            charge[i] = static_cast<float>(bodies[i].charge.value());
        }
        const float permeabilityOver4Pi = static_cast<float>(
            constants::vacuumPermeability.value() / (4.0 * kPi<double>));

        std::vector<float> fieldX(n);
        std::vector<float> fieldY(n);
        std::vector<float> fieldZ(n);
        defaultBackend().magneticFieldNBody(posX, posY, posZ, velX, velY, velZ, charge,
                                            permeabilityOver4Pi, fieldX, fieldY, fieldZ);

        std::vector<MagneticFluxDensity3> result(n);
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = MagneticFluxDensity3{Vec3{static_cast<double>(fieldX[i]),
                                                  static_cast<double>(fieldY[i]),
                                                  static_cast<double>(fieldZ[i])}};
        }
        return result;
    }

    std::vector<Vec3> totals(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 at = bodies[i].position.value();
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const Vec3 delta = at - bodies[j].position.value();
            const double r2 = lengthSquared(delta);
            if (r2 <= 0.0) {
                continue;
            }
            const double r = std::sqrt(r2);
            const Vec3 velocity = bodies[j].velocity().value();
            totals[i] += cross(velocity, delta) * (bodies[j].charge.value() / (r2 * r));
        }
    }

    std::vector<MagneticFluxDensity3> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = MagneticFluxDensity3{
            totals[i] * (constants::vacuumPermeability.value() / (4.0 * kPi<double>))};
    }
    return result;
}

}  // namespace ysq
