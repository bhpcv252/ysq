#include <Physics/Gravity/Newtonian.hpp>

#include <Math/Vector3.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace ysq {

namespace {

/// The shared kernel: softened Newtonian acceleration contributed by one
/// source of gravitational parameter `gm` (already G * mass), evaluated at a
/// displacement `delta` = source position - query position.
[[nodiscard]] Vec3 softenedTerm(const Vec3& delta, double gm, double softeningSquared) {
    const double r2 = lengthSquared(delta) + softeningSquared;
    const double r = std::sqrt(r2);
    return delta * (gm / (r2 * r));
}

}  // namespace

Force3 newtonianForce(const Body& on, const Body& from) {
    const Length3 delta = from.position - on.position;
    const Length r = length(delta);
    const Force magnitude = constants::G * on.mass * from.mass / raised<2>(r);
    return magnitude * normalized(delta);
}

Acceleration3 newtonianAcceleration(const Length3& at, std::span<const Body> sources,
                                    Length softening) {
    const Vec3 rawAt = at.value();
    const double softeningSquared = softening.value() * softening.value();

    Vec3 total{};
    for (const Body& source : sources) {
        const double gm = constants::G.value() * source.mass.value();
        total += softenedTerm(source.position.value() - rawAt, gm, softeningSquared);
    }
    return Acceleration3{total};
}

std::vector<Acceleration3> newtonianAccelerations(std::span<const Body> bodies,
                                                  Length softening) {
    const double softeningSquared = softening.value() * softening.value();
    std::vector<Acceleration3> result(bodies.size());

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Vec3 at = bodies[i].position.value();
        Vec3 total{};
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            if (i == j) {
                continue;
            }
            const double gm = constants::G.value() * bodies[j].mass.value();
            total += softenedTerm(bodies[j].position.value() - at, gm, softeningSquared);
        }
        result[i] = Acceleration3{total};
    }
    return result;
}

Energy newtonianPotentialEnergy(std::span<const Body> bodies, Length softening) {
    const double softeningSquared = softening.value() * softening.value();
    double total = 0.0;

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const Vec3 delta = bodies[j].position.value() - bodies[i].position.value();
            const double r = std::sqrt(lengthSquared(delta) + softeningSquared);
            total -= constants::G.value() * bodies[i].mass.value() *
                     bodies[j].mass.value() / r;
        }
    }
    return Energy{total};
}

NewtonianField::NewtonianField(std::span<const Body> bodies, Length softening)
    : m_softeningSquared(softening.value() * softening.value()) {
    m_gravitationalParameters.reserve(bodies.size());
    for (const Body& body : bodies) {
        m_gravitationalParameters.push_back(constants::G.value() * body.mass.value());
    }
}

NBodyState NewtonianField::operator()(double, const NBodyState& positions) const {
    assert(positions.size() == m_gravitationalParameters.size());
    NBodyState result(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        Vec3 total{};
        for (std::size_t j = 0; j < positions.size(); ++j) {
            if (i == j) {
                continue;
            }
            total += softenedTerm(positions[j] - positions[i],
                                  m_gravitationalParameters[j], m_softeningSquared);
        }
        result[i] = total;
    }
    return result;
}

}  // namespace ysq
