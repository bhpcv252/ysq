#include <Physics/Gravity/PostNewtonian.hpp>

#include <Math/Scalar.hpp>
#include <Units/Constants.hpp>

#include <cassert>

namespace ysq {

namespace {

/// The shared kernel `postNewtonianCorrection` and `RelativisticNBodySystem`
/// both reduce to: the 1PN acceleration term for a body at relative
/// position `r` and relative velocity `v` from a source of gravitational
/// parameter `gm` (already G * mass).
[[nodiscard]] Vec3 relativisticAccelerationTerm(const Vec3& r, const Vec3& v, double gm) {
    const double rMag = length(r);
    const Vec3 n = r / rMag;
    const double vMagSquared = lengthSquared(v);
    const double radialSpeed = dot(v, n);
    const double c = constants::speedOfLight.value();

    return (n * (4.0 * gm / rMag - vMagSquared) + v * (4.0 * radialSpeed)) *
           (gm / (c * c * rMag * rMag));
}

}  // namespace

Acceleration3 postNewtonianCorrection(const Body& testParticle, const Body& source) {
    const Vec3 r = (testParticle.position - source.position).value();
    const Vec3 v = (testParticle.velocity() - source.velocity()).value();
    const double gm = constants::G.value() * source.mass.value();
    return Acceleration3{relativisticAccelerationTerm(r, v, gm)};
}

double perihelionPrecessionPerOrbit(double gm, double semiMajorAxis, double eccentricity) {
    const double c = constants::speedOfLight.value();
    return 6.0 * kPi<double> * gm /
           (c * c * semiMajorAxis * (1.0 - eccentricity * eccentricity));
}

Vec3 relativisticJerkTerm(const Vec3& r, const Vec3& v, double gm) {
    const double rMag = length(r);
    const Vec3 n = r / rMag;
    const double vMagSquared = lengthSquared(v);
    const double s = dot(v, n);  // radial speed; also d(rMag)/dt
    const double c = constants::speedOfLight.value();

    // Two-body Newtonian relative acceleration -- see the header comment
    // on why no more of the full N-body picture belongs here.
    const Vec3 a = n * (-gm / (rMag * rMag));

    const double k = gm / (c * c * rMag * rMag);
    const double kDot = -2.0 * k * s / rMag;

    const double bracket = 4.0 * gm / rMag - vMagSquared;
    const Vec3 b = n * bracket + v * (4.0 * s);

    const double aDotN = dot(a, n);
    const double sDot = aDotN + (vMagSquared - s * s) / rMag;
    const double bracketDot = -4.0 * gm * s / (rMag * rMag) - 2.0 * dot(v, a);
    const Vec3 nDot = (v - n * s) / rMag;

    const Vec3 bDot = nDot * bracket + n * bracketDot + v * (4.0 * sDot) + a * (4.0 * s);

    return b * kDot + bDot * k;
}

RelativisticNBodyJerkSystem::RelativisticNBodyJerkSystem(std::span<const Body> bodies,
                                                         std::vector<int> primaryIndex,
                                                         Length softening)
    : m_newtonian(bodies, softening), m_primaryIndex(std::move(primaryIndex)) {
    assert(bodies.size() == m_primaryIndex.size());
    m_primaryGravitationalParameters.resize(bodies.size(), 0.0);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const int primary = m_primaryIndex[i];
        if (primary < 0 || static_cast<std::size_t>(primary) == i) {
            continue;
        }
        m_primaryGravitationalParameters[i] =
            constants::G.value() * bodies[static_cast<std::size_t>(primary)].mass.value();
    }
}

std::pair<Vec3, Vec3> RelativisticNBodyJerkSystem::operator()(
    std::size_t bodyIndex, const NBodyState& positions, const NBodyState& velocities) const {
    auto [acceleration, jerk] = m_newtonian(bodyIndex, positions, velocities);

    const int primary = m_primaryIndex[bodyIndex];
    if (primary < 0 || static_cast<std::size_t>(primary) == bodyIndex) {
        return {acceleration, jerk};
    }
    const std::size_t p = static_cast<std::size_t>(primary);
    const Vec3 r = positions[bodyIndex] - positions[p];
    const Vec3 v = velocities[bodyIndex] - velocities[p];
    const double gm = m_primaryGravitationalParameters[bodyIndex];

    acceleration += relativisticAccelerationTerm(r, v, gm);
    jerk += relativisticJerkTerm(r, v, gm);
    return {acceleration, jerk};
}

RelativisticNBodySystem::RelativisticNBodySystem(std::span<const Body> bodies,
                                                 std::vector<int> primaryIndex,
                                                 Length softening)
    : m_newtonian(bodies, softening), m_primaryIndex(std::move(primaryIndex)) {
    assert(bodies.size() == m_primaryIndex.size());
    m_primaryGravitationalParameters.resize(bodies.size(), 0.0);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const int primary = m_primaryIndex[i];
        if (primary < 0 || static_cast<std::size_t>(primary) == i) {
            continue;
        }
        m_primaryGravitationalParameters[i] =
            constants::G.value() * bodies[static_cast<std::size_t>(primary)].mass.value();
    }
}

PhaseState<NBodyState> RelativisticNBodySystem::operator()(
    double time, const PhaseState<NBodyState>& state) const {
    NBodyState acceleration = m_newtonian(time, state.position);

    for (std::size_t i = 0; i < m_primaryIndex.size(); ++i) {
        const int primary = m_primaryIndex[i];
        if (primary < 0 || static_cast<std::size_t>(primary) == i) {
            continue;
        }
        const std::size_t p = static_cast<std::size_t>(primary);
        const Vec3 r = state.position[i] - state.position[p];
        const Vec3 v = state.velocity[i] - state.velocity[p];
        acceleration[i] +=
            relativisticAccelerationTerm(r, v, m_primaryGravitationalParameters[i]);
    }

    return PhaseState<NBodyState>{state.velocity, acceleration};
}

}  // namespace ysq
