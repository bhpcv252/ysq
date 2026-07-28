#include <Physics/Fluids/SPH.hpp>

#include <Math/Scalar.hpp>

#include <cmath>
#include <cstddef>

namespace ysq {

namespace {

constexpr double kNormalization3D = 1.0 / kPi<double>;

}  // namespace

double cubicSplineKernel(double r, double smoothingLength) {
    const double q = r / smoothingLength;
    const double sigma =
        kNormalization3D / (smoothingLength * smoothingLength * smoothingLength);

    if (q < 1.0) {
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    }
    if (q < 2.0) {
        const double t = 2.0 - q;
        return sigma * 0.25 * t * t * t;
    }
    return 0.0;
}

Vec3 cubicSplineKernelGradient(const Vec3& separation, double smoothingLength) {
    const double r = length(separation);
    if (r <= 0.0) {
        return Vec3{};
    }

    const double q = r / smoothingLength;
    const double sigma =
        kNormalization3D / (smoothingLength * smoothingLength * smoothingLength);

    double dWdq;
    if (q < 1.0) {
        dWdq = sigma * (-3.0 * q + 2.25 * q * q);
    } else if (q < 2.0) {
        const double t = 2.0 - q;
        dWdq = -0.75 * sigma * t * t;
    } else {
        dWdq = 0.0;
    }

    const double dWdr = dWdq / smoothingLength;
    return (separation / r) * dWdr;
}

void computeDensityAndPressure(std::span<SPHParticle> particles, double smoothingLength,
                               double equationOfStateK, double polytropicIndex) {
    for (SPHParticle& target : particles) {
        double density = 0.0;
        for (const SPHParticle& source : particles) {
            const double r = length(target.position - source.position);
            density += source.mass * cubicSplineKernel(r, smoothingLength);
        }
        target.density = density;
        target.pressure = equationOfStateK * std::pow(density, polytropicIndex);
    }
}

std::vector<Vec3> pressureAccelerations(std::span<const SPHParticle> particles,
                                        double smoothingLength) {
    std::vector<Vec3> result(particles.size());

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const SPHParticle& target = particles[i];
        const double targetTerm = target.pressure / (target.density * target.density);

        Vec3 total{};
        for (std::size_t j = 0; j < particles.size(); ++j) {
            if (i == j) {
                continue;
            }
            const SPHParticle& source = particles[j];
            const double sourceTerm = source.pressure / (source.density * source.density);
            const Vec3 gradient = cubicSplineKernelGradient(
                target.position - source.position, smoothingLength);
            total += gradient * (source.mass * (targetTerm + sourceTerm));
        }
        result[i] = -total;
    }
    return result;
}

}  // namespace ysq
