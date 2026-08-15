#include <Physics/Gravity/Newtonian.hpp>

#include <Compute/ComputeBackend.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Vector3.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ysq {

namespace {

// Below this body count, the CPU pairwise loop (which also does half the
// arithmetic GPU dispatch would, since it visits each pair once rather than
// each body against every other) is faster than paying for a dispatch at
// all: uploading positions, launching a kernel, and reading accelerations
// back.
//
// benchmarks/compute_thresholds.cpp measured a raw crossover of 512
// against Compute::CpuBackend's own (simpler, full double-loop) reference
// kernel, but that reference does roughly twice the arithmetic this file's
// actual below-threshold pairwise loop does (see the comment above), so
// this doubles the raw measurement rather than using it directly: at the
// raw crossover, GPU only barely won (a 2.14x ratio), which halving to
// account for the real loop's own ~2x advantage would erase or reverse.
// The next size step up (1024) still won by a comfortable, real-loop-
// adjusted margin (an unadjusted 4.37x ratio, roughly 2.2x once halved),
// so that is the safe, conservative choice.
constexpr std::size_t kGpuDispatchThreshold = 1024;

[[nodiscard]] bool anyBodyIsOblate(std::span<const Body> bodies) {
    for (const Body& body : bodies) {
        if (body.j2 != 0.0) {
            return true;
        }
    }
    return false;
}

/// Direct-sum Newtonian gravity through Compute::defaultBackend(), point
/// masses only. Never called with an oblate body in `positions`: GPU
/// dispatch is a one-sided per-target sum, which cannot carry the
/// momentum-conserving reaction terms J2 needs between an oblate source and
/// the body it perturbs (see ComputeBackend.hpp's gravitationalNBody), so
/// every caller here checks anyBodyIsOblate() (or its own per-body
/// equivalent) first and falls back to the CPU pairwise loop otherwise.
[[nodiscard]] NBodyState gravitationalNBodyGpu(std::span<const Vec3> positions,
                                               std::span<const double> gm,
                                               double softeningSquared) {
    const std::size_t n = positions.size();
    std::vector<float> posX(n);
    std::vector<float> posY(n);
    std::vector<float> posZ(n);
    std::vector<float> gmFloat(n);
    for (std::size_t i = 0; i < n; ++i) {
        posX[i] = static_cast<float>(positions[i].x);
        posY[i] = static_cast<float>(positions[i].y);
        posZ[i] = static_cast<float>(positions[i].z);
        gmFloat[i] = static_cast<float>(gm[i]);
    }

    std::vector<float> accX(n);
    std::vector<float> accY(n);
    std::vector<float> accZ(n);
    defaultBackend().gravitationalNBody(posX, posY, posZ, gmFloat,
                                        static_cast<float>(softeningSquared), accX, accY,
                                        accZ);

    NBodyState result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = Vec3{static_cast<double>(accX[i]), static_cast<double>(accY[i]),
                         static_cast<double>(accZ[i])};
    }
    return result;
}

/// The shared kernel: softened Newtonian acceleration contributed by one
/// source of gravitational parameter `gm` (already G * mass), evaluated at a
/// displacement `delta` = source position - query position.
[[nodiscard]] Vec3 softenedTerm(const Vec3& delta, double gm, double softeningSquared) {
    const double r2 = lengthSquared(delta) + softeningSquared;
    const double r = std::sqrt(r2);
    return delta * (gm / (r2 * r));
}

/// A source's own polar axis in the inertial frame: its body-frame +Z,
/// rotated by its orientation. Meaningless when the source's j2 is zero, so
/// callers only ever weight this by a coefficient that is already zero then.
[[nodiscard]] Vec3 spinAxisOf(const Body& body) {
    return rotate(body.orientation, Vec3::unitZ());
}

/// The time-derivative of `softenedTerm`, at the same `delta` = source
/// position - query position and its own time-derivative `deltaVelocity` =
/// source velocity - query velocity: `d/dt(gm * delta * R^-3)`, R^2 =
/// |delta|^2 + softeningSquared not depending on time. Same convention as
/// `softenedTerm`: points however the acceleration contribution it is the
/// derivative of already does.
[[nodiscard]] Vec3 softenedJerkTerm(const Vec3& delta, const Vec3& deltaVelocity,
                                    double gm, double softeningSquared) {
    const double r2 = lengthSquared(delta) + softeningSquared;
    const double r = std::sqrt(r2);
    const double r3 = r2 * r;
    const double r5 = r3 * r2;
    const double radialTerm = dot(delta, deltaVelocity);
    return deltaVelocity * (gm / r3) - delta * (3.0 * gm * radialTerm / r5);
}

/// The extra acceleration a source's J2 oblateness adds on a point whose
/// separation from the source (query position minus source position) is
/// `fromSource`, standard result (e.g. Vallado, "Fundamentals of
/// Astrodynamics and Applications"): a = -(3/2) J2 mu Req^2 / r^4 *
/// [(1 - 5 s^2) rHat + 2 s spinAxis], s = rHat . spinAxis, the sine of the
/// point's latitude above the source's own equatorial plane.
///
/// `j2Coefficient` is (3/2) * j2 * G * sourceMass * equatorialRadius^2,
/// precomputed once per source since it does not depend on the query point;
/// it is exactly zero whenever the source's j2 is zero, so this term
/// vanishes on its own, without a branch, exactly reproducing the point-mass
/// case.
[[nodiscard]] Vec3 oblatenessTerm(const Vec3& fromSource, double j2Coefficient,
                                  const Vec3& spinAxis, double softeningSquared) {
    const double r2 = lengthSquared(fromSource) + softeningSquared;
    const Vec3 rHat = fromSource / std::sqrt(r2);
    const double s = dot(rHat, spinAxis);
    const double perR4 = j2Coefficient / (r2 * r2);
    return rHat * (-perR4 * (1.0 - 5.0 * s * s)) + spinAxis * (-2.0 * perR4 * s);
}

}  // namespace

Force3 newtonianForce(const Body& on, const Body& from) {
    const Length3 delta = from.position - on.position;
    const Length r = length(delta);
    const Force magnitude = constants::G * on.mass * from.mass / raised<2>(r);
    Force3 result = magnitude * normalized(delta);

    // J2 is not symmetric in on/from the way the monopole term is: it comes
    // from one specific body's shape, so it has to be handled in both
    // directions separately, each guarded by that body's own j2, for
    // Newton's third law to hold when only one of the two is oblate.
    if (from.j2 != 0.0) {
        const double j2Coefficient = 1.5 * from.j2 * constants::G.value() *
                                     from.mass.value() * from.radius.value() *
                                     from.radius.value();
        const Vec3 onFromFrom = -delta.value();
        result += Force3{on.mass.value() * oblatenessTerm(onFromFrom, j2Coefficient,
                                                          spinAxisOf(from), 0.0)};
    }
    if (on.j2 != 0.0) {
        // The reaction to the force `from` would feel from `on`'s own
        // bulge: exactly that force, computed the same way with the roles
        // swapped, negated.
        const double j2CoefficientOn = 1.5 * on.j2 * constants::G.value() *
                                       on.mass.value() * on.radius.value() *
                                       on.radius.value();
        const Vec3 fromFromOn = delta.value();
        result -= Force3{from.mass.value() * oblatenessTerm(fromFromOn, j2CoefficientOn,
                                                            spinAxisOf(on), 0.0)};
    }
    return result;
}

Acceleration3 newtonianAcceleration(const Length3& at, std::span<const Body> sources,
                                    Length softening) {
    const Vec3 rawAt = at.value();
    const double softeningSquared = softening.value() * softening.value();

    Vec3 total{};
    for (const Body& source : sources) {
        const double gm = constants::G.value() * source.mass.value();
        const Vec3 sourceToAt = rawAt - source.position.value();
        total += softenedTerm(-sourceToAt, gm, softeningSquared);
        if (source.j2 != 0.0) {
            const double j2Coefficient =
                1.5 * source.j2 * gm * source.radius.value() * source.radius.value();
            total += oblatenessTerm(sourceToAt, j2Coefficient, spinAxisOf(source),
                                    softeningSquared);
        }
    }
    return Acceleration3{total};
}

std::vector<Acceleration3> newtonianAccelerations(std::span<const Body> bodies,
                                                  Length softening) {
    const double softeningSquared = softening.value() * softening.value();

    if (bodies.size() >= kGpuDispatchThreshold && !anyBodyIsOblate(bodies)) {
        std::vector<Vec3> positions(bodies.size());
        std::vector<double> gm(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            positions[i] = bodies[i].position.value();
            gm[i] = constants::G.value() * bodies[i].mass.value();
        }
        const NBodyState accelerations =
            gravitationalNBodyGpu(positions, gm, softeningSquared);
        std::vector<Acceleration3> result(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            result[i] = Acceleration3{accelerations[i]};
        }
        return result;
    }

    std::vector<Vec3> totals(bodies.size());

    // Pairwise (i < j), each pair visited once, is what makes it possible
    // to apply J2's Newton's-third-law reaction explicitly below: iterating
    // source by source would compute each body's acceleration
    // independently, with no place to put a reaction force back onto an
    // oblate source.
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const Body& bi = bodies[i];
            const Body& bj = bodies[j];
            const double gmI = constants::G.value() * bi.mass.value();
            const double gmJ = constants::G.value() * bj.mass.value();
            const Vec3 jFromI = bj.position.value() - bi.position.value();

            totals[i] += softenedTerm(jFromI, gmJ, softeningSquared);
            totals[j] += softenedTerm(-jFromI, gmI, softeningSquared);

            if (bj.j2 != 0.0) {
                const double j2CoefficientJ =
                    1.5 * bj.j2 * gmJ * bj.radius.value() * bj.radius.value();
                const Vec3 termOnI = oblatenessTerm(-jFromI, j2CoefficientJ,
                                                    spinAxisOf(bj), softeningSquared);
                totals[i] += termOnI;
                totals[j] -= termOnI * (bi.mass.value() / bj.mass.value());
            }
            if (bi.j2 != 0.0) {
                const double j2CoefficientI =
                    1.5 * bi.j2 * gmI * bi.radius.value() * bi.radius.value();
                const Vec3 termOnJ = oblatenessTerm(jFromI, j2CoefficientI,
                                                    spinAxisOf(bi), softeningSquared);
                totals[j] += termOnJ;
                totals[i] -= termOnJ * (bj.mass.value() / bi.mass.value());
            }
        }
    }

    std::vector<Acceleration3> result(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        result[i] = Acceleration3{totals[i]};
    }
    return result;
}

namespace {

/// The J2 potential energy the pair (querySide oblate) contributes:
/// G Mquery Msource Req^2 / r^3 * j2 (3 s^2 - 1) / 2, s = sin(latitude) of
/// `queryPosition` above the oblate body's own equatorial plane. Positive,
/// not negative: oblatenessTerm() above is exactly -1/Mquery times this
/// potential's gradient with respect to queryPosition (verified directly by
/// differentiating this and comparing every component against
/// oblatenessTerm's own formula, both of the closed-form Physics.Gravity
/// unit tests already check the latter against), not a second,
/// independently derived formula.
[[nodiscard]] double oblatenessPotentialEnergy(const Vec3& queryPosition,
                                               double queryMass, double sourceMass,
                                               double j2, double sourceRadius,
                                               const Vec3& sourcePosition,
                                               const Vec3& sourceSpinAxis) {
    if (j2 == 0.0) {
        return 0.0;
    }
    const Vec3 fromSource = queryPosition - sourcePosition;
    const double r = length(fromSource);
    const double s = dot(fromSource, sourceSpinAxis) / r;
    const double reqOverR = sourceRadius / r;
    return constants::G.value() * queryMass * sourceMass / r * j2 * reqOverR * reqOverR *
           (3.0 * s * s - 1.0) * 0.5;
}

}  // namespace

Energy newtonianPotentialEnergy(std::span<const Body> bodies, Length softening) {
    const double softeningSquared = softening.value() * softening.value();
    double total = 0.0;

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const Vec3 delta = bodies[j].position.value() - bodies[i].position.value();
            const double r = std::sqrt(lengthSquared(delta) + softeningSquared);
            total -= constants::G.value() * bodies[i].mass.value() *
                     bodies[j].mass.value() / r;

            total += oblatenessPotentialEnergy(
                bodies[i].position.value(), bodies[i].mass.value(),
                bodies[j].mass.value(), bodies[j].j2, bodies[j].radius.value(),
                bodies[j].position.value(), spinAxisOf(bodies[j]));
            total += oblatenessPotentialEnergy(
                bodies[j].position.value(), bodies[j].mass.value(),
                bodies[i].mass.value(), bodies[i].j2, bodies[i].radius.value(),
                bodies[i].position.value(), spinAxisOf(bodies[i]));
        }
    }
    return Energy{total};
}

NewtonianField::NewtonianField(std::span<const Body> bodies, Length softening)
    : m_softeningSquared(softening.value() * softening.value()),
      m_hasOblateBody(anyBodyIsOblate(bodies)) {
    m_gravitationalParameters.reserve(bodies.size());
    m_j2Coefficients.reserve(bodies.size());
    m_spinAxes.reserve(bodies.size());
    for (const Body& body : bodies) {
        const double gm = constants::G.value() * body.mass.value();
        m_gravitationalParameters.push_back(gm);
        m_j2Coefficients.push_back(1.5 * body.j2 * gm * body.radius.value() *
                                   body.radius.value());
        m_spinAxes.push_back(body.j2 != 0.0 ? spinAxisOf(body) : Vec3::zero());
    }
}

NBodyState NewtonianField::operator()(double, const NBodyState& positions) const {
    assert(positions.size() == m_gravitationalParameters.size());

    // The path this functor is actually called through every integration
    // step (VelocityVerletStepper and friends), so this is where GPU
    // dispatch matters most in practice, more than the one-shot
    // newtonianAccelerations() above.
    if (positions.size() >= kGpuDispatchThreshold && !m_hasOblateBody) {
        return gravitationalNBodyGpu(
            std::span<const Vec3>{positions.begin(), positions.end()},
            m_gravitationalParameters, m_softeningSquared);
    }

    NBodyState result(positions.size());
    // Pairwise, for the same Newton's-third-law reason
    // newtonianAccelerations() is; see its own comment.
    for (std::size_t i = 0; i < positions.size(); ++i) {
        for (std::size_t j = i + 1; j < positions.size(); ++j) {
            const double gmI = m_gravitationalParameters[i];
            const double gmJ = m_gravitationalParameters[j];
            const Vec3 jFromI = positions[j] - positions[i];

            result[i] += softenedTerm(jFromI, gmJ, m_softeningSquared);
            result[j] += softenedTerm(-jFromI, gmI, m_softeningSquared);

            if (m_j2Coefficients[j] != 0.0) {
                const Vec3 termOnI = oblatenessTerm(-jFromI, m_j2Coefficients[j],
                                                    m_spinAxes[j], m_softeningSquared);
                result[i] += termOnI;
                result[j] -= termOnI * (gmI / gmJ);
            }
            if (m_j2Coefficients[i] != 0.0) {
                const Vec3 termOnJ = oblatenessTerm(jFromI, m_j2Coefficients[i],
                                                    m_spinAxes[i], m_softeningSquared);
                result[j] += termOnJ;
                result[i] -= termOnJ * (gmJ / gmI);
            }
        }
    }
    return result;
}

NewtonianJerkField::NewtonianJerkField(std::span<const Body> bodies, Length softening)
    : m_softeningSquared(softening.value() * softening.value()) {
    m_gravitationalParameters.reserve(bodies.size());
    for (const Body& body : bodies) {
        m_gravitationalParameters.push_back(constants::G.value() * body.mass.value());
    }
}

std::pair<Vec3, Vec3> NewtonianJerkField::operator()(std::size_t bodyIndex,
                                                     const NBodyState& positions,
                                                     const NBodyState& velocities) const {
    assert(positions.size() == m_gravitationalParameters.size());
    assert(velocities.size() == positions.size());

    Vec3 acceleration{};
    Vec3 jerk{};
    for (std::size_t j = 0; j < positions.size(); ++j) {
        if (j == bodyIndex) {
            continue;
        }
        const Vec3 delta = positions[j] - positions[bodyIndex];
        const Vec3 deltaVelocity = velocities[j] - velocities[bodyIndex];
        const double gm = m_gravitationalParameters[j];
        acceleration += softenedTerm(delta, gm, m_softeningSquared);
        jerk += softenedJerkTerm(delta, deltaVelocity, gm, m_softeningSquared);
    }
    return {acceleration, jerk};
}

}  // namespace ysq
