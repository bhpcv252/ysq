#include <Physics/Mechanics/RigidBody.hpp>

#include <Math/Integrators/RK4.hpp>
#include <Math/ODE.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Gravity/Newtonian.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

namespace {

/// tau = (3 GM / r^3) rHat x (I . rHat), evaluated entirely in the body
/// frame: `orientationNow` is what turns a body-frame vector into the
/// inertial frame the rest of `perturberPosition`/`bodyPosition` are
/// expressed in, so rHat is rotated into the body frame first (its inverse
/// applied), and the result handed back is a body-frame torque, left to the
/// caller to rotate back out. Kept in the body frame, rather than rotating
/// in and out here, because rigidBodySystem below needs to sum several of
/// these before doing that rotation once, not once per perturber.
[[nodiscard]] Vec3 gravityGradientTorqueBodyFrame(const Quat& orientationNow,
                                                  const Vec3& principalMoments,
                                                  const Vec3& bodyPosition,
                                                  const Vec3& perturberPosition,
                                                  double perturberGM) {
    const Vec3 separation = perturberPosition - bodyPosition;
    const double r = length(separation);
    const Vec3 nHat = separation / r;
    const Vec3 nHatBody = rotate(conjugate(orientationNow), nHat);
    const Vec3 iDotN{principalMoments.x * nHatBody.x, principalMoments.y * nHatBody.y,
                     principalMoments.z * nHatBody.z};
    return cross(nHatBody, iDotN) * (3.0 * perturberGM / (r * r * r));
}

/// The state Mechanics/RigidBody's integrator runs on: orientation plus
/// angular momentum, both raw and unit-stripped (angular momentum in
/// kg m^2/s), the same units-cross-the-boundary-once convention
/// Mechanics/Dynamics.hpp's NBodyState draws for translation.
struct RotationalState {
    using value_type = double;

    Quat orientation = Quat::identity();
    Vec3 angularMomentum{};

    constexpr RotationalState& operator+=(const RotationalState& other) noexcept {
        orientation += other.orientation;
        angularMomentum += other.angularMomentum;
        return *this;
    }

    constexpr RotationalState& operator-=(const RotationalState& other) noexcept {
        orientation -= other.orientation;
        angularMomentum -= other.angularMomentum;
        return *this;
    }

    constexpr RotationalState& operator*=(double scalar) noexcept {
        orientation *= scalar;
        angularMomentum *= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr RotationalState
    operator+(RotationalState a, const RotationalState& b) noexcept {
        a += b;
        return a;
    }

    /// OdeState requires `a - b` and `s * a` as well as the forms actually
    /// used by Rk4Stepper below; [[maybe_unused]] because this translation
    /// unit's own stepper never happens to call these two directions, not
    /// because they are dead code the concept could do without.
    [[nodiscard]] [[maybe_unused]] friend constexpr RotationalState
    operator-(RotationalState a, const RotationalState& b) noexcept {
        a -= b;
        return a;
    }

    [[nodiscard]] friend constexpr RotationalState operator*(RotationalState a,
                                                             double scalar) noexcept {
        a *= scalar;
        return a;
    }

    [[nodiscard]] [[maybe_unused]] friend constexpr RotationalState
    operator*(double scalar, RotationalState a) noexcept {
        a *= scalar;
        return a;
    }
};

/// dq/dt = 0.5 q (0, omega_body); dL/dt = torque, torque re-evaluated at
/// every RK4 stage from that stage's own orientation, not frozen at the
/// step's start, since it costs nothing more than the same sum already
/// being done. omega_body comes from the state's (inertial-frame) angular
/// momentum rotated into the body frame and divided through
/// `principalMoments`, the diagonal inertia tensor in that frame.
[[nodiscard]] auto rigidBodySystem(const Vec3& principalMoments, const Vec3& bodyPosition,
                                   std::span<const Vec3> perturberPositions,
                                   std::span<const double> perturberGMs) {
    return [principalMoments, bodyPosition, perturberPositions,
            perturberGMs](double, const RotationalState& state) {
        const Quat q = state.orientation;
        const Vec3 angularMomentumBody = rotate(conjugate(q), state.angularMomentum);
        const Vec3 angularVelocityBody{angularMomentumBody.x / principalMoments.x,
                                       angularMomentumBody.y / principalMoments.y,
                                       angularMomentumBody.z / principalMoments.z};

        const Quat omegaQuat = Quat::fromScalarVector(0.0, angularVelocityBody);
        const Quat orientationRate = (q * omegaQuat) * 0.5;

        Vec3 torqueBody{};
        for (std::size_t i = 0; i < perturberPositions.size(); ++i) {
            torqueBody +=
                gravityGradientTorqueBodyFrame(q, principalMoments, bodyPosition,
                                               perturberPositions[i], perturberGMs[i]);
        }
        const Vec3 torqueInertial = rotate(q, torqueBody);

        return RotationalState{orientationRate, torqueInertial};
    };
}

}  // namespace

Torque3 gravityGradientTorque(const Body& oblateBody, const Length3& perturberPosition,
                              Mass perturberMass) {
    const double gm = constants::G.value() * perturberMass.value();
    const Vec3 torqueBody = gravityGradientTorqueBodyFrame(
        oblateBody.orientation, oblateBody.principalMomentsOfInertia.value(),
        oblateBody.position.value(), perturberPosition.value(), gm);
    return Torque3{rotate(oblateBody.orientation, torqueBody)};
}

Torque3 gravityGradientTorque(const Body& oblateBody, std::span<const Body> perturbers) {
    Torque3 total{};
    for (const Body& perturber : perturbers) {
        total += gravityGradientTorque(oblateBody, perturber.position, perturber.mass);
    }
    return total;
}

void stepRigidBody(Body& body, std::span<const Body> perturbers, double dt) {
    if (body.principalMomentsOfInertia == MomentOfInertia3{}) {
        return;
    }

    std::vector<Vec3> perturberPositions;
    std::vector<double> perturberGMs;
    perturberPositions.reserve(perturbers.size());
    perturberGMs.reserve(perturbers.size());
    for (const Body& perturber : perturbers) {
        perturberPositions.push_back(perturber.position.value());
        perturberGMs.push_back(constants::G.value() * perturber.mass.value());
    }

    const auto system =
        rigidBodySystem(body.principalMomentsOfInertia.value(), body.position.value(),
                        perturberPositions, perturberGMs);

    RotationalState state{body.orientation, body.angularMomentum.value()};
    RotationalState next{};

    Rk4Stepper<RotationalState> stepper;
    stepper.step(system, 0.0, state, dt, next);

    body.orientation = normalized(next.orientation);
    body.angularMomentum = AngularMomentum3{next.angularMomentum};
}

}  // namespace ysq
