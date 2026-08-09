#include <Physics/Fluids/SPH.hpp>

#include <Math/Scalar.hpp>
#include <Math/SpatialPartition/KdTree.hpp>

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

namespace {

/// Every particle position, in a `KdTree3` built fresh from `particles`'
/// current state: cheap next to the O(n^2) all-pairs sum it replaces, and a
/// tree built from stale positions would silently miss neighbours that have
/// since moved into range, so it is never cached across calls.
[[nodiscard]] KdTree3<double> buildPositionTree(std::span<const SPHParticle> particles) {
    std::vector<Vec3> positions;
    positions.reserve(particles.size());
    for (const SPHParticle& particle : particles) {
        positions.push_back(particle.position);
    }
    return KdTree3<double>(std::move(positions));
}

}  // namespace

void computeDensityAndPressure(std::span<SPHParticle> particles, double smoothingLength,
                               double equationOfStateK, double polytropicIndex) {
    const KdTree3<double> tree = buildPositionTree(particles);
    // The kernel is exactly zero past q = 2 (r = 2h), so restricting the sum
    // to this radius is not an approximation: it skips exactly the pairs
    // that would have contributed zero anyway.
    const double supportRadius = 2.0 * smoothingLength;

    for (SPHParticle& target : particles) {
        double density = 0.0;
        for (std::size_t j : tree.radiusQuery(target.position, supportRadius)) {
            const SPHParticle& source = particles[j];
            const double r = length(target.position - source.position);
            density += source.mass * cubicSplineKernel(r, smoothingLength);
        }
        target.density = density;
        target.pressure = equationOfStateK * std::pow(density, polytropicIndex);
    }
}

std::vector<Vec3> pressureAccelerations(std::span<const SPHParticle> particles,
                                        double smoothingLength) {
    const KdTree3<double> tree = buildPositionTree(particles);
    const double supportRadius = 2.0 * smoothingLength;

    std::vector<Vec3> result(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const SPHParticle& target = particles[i];
        const double targetTerm = target.pressure / (target.density * target.density);

        Vec3 total{};
        for (std::size_t j : tree.radiusQuery(target.position, supportRadius)) {
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
