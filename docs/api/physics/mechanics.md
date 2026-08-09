# Physics/Mechanics API reference

`Body`, reference frames, relativistic kinematics, `NBodyState`, rigid-body
rotation, and a set of general-purpose forces (springs, drag, friction,
collisions, constraints) any scenario can assemble a mechanism from. Start
with [docs/physics/mechanics.md](../../physics/mechanics.md) for the ideas;
[src/Physics/README.md](../../../src/Physics/README.md) for the full design
notes shared across every theory.

## `Physics/Body.hpp`

The primitive every other theory in `Physics` works on.

```cpp
struct Body {
    Mass mass{};
    ElectricCharge charge{};
    Length3 position{};
    Momentum3 momentum{};

    constexpr Velocity3 velocity() const noexcept;   // momentum / mass; non-relativistic, exact only at v << c
};
```

`Body` stores **momentum, not velocity**. In the Newtonian limit velocity is
just momentum over mass, but momentum stays correct once
`Mechanics/Kinematics.hpp`'s relativistic relation `p = gamma * m * v`
applies: velocity alone can't be turned back into momentum once `gamma`
matters, while momentum can always be turned into velocity.

## `Physics/Mechanics/Frame.hpp`

Inertial reference frames, Galilean transform only (`v << c`; for the
relativistic velocity transform between frames, see `Kinematics.hpp` below).

```cpp
struct Frame {
    Length3 origin{};
    Velocity3 velocity{};
    static constexpr Frame lab() noexcept;   // the zero frame
};

constexpr Body transformTo(const Frame& frame, const Body& body) noexcept;
constexpr Body transformFrom(const Frame& frame, const Body& body) noexcept;  // the inverse
```

`transformTo` shifts position by the frame's origin and momentum by the
body's mass times the frame's velocity (a Galilean boost changes velocity by
a constant offset, and momentum follows). `transformFrom` is the exact
inverse.

## `Physics/Mechanics/Kinematics.hpp`

Special-relativistic kinematics: worldlines, proper time, four-velocity.
Everything here reduces to the familiar Newtonian relation at `v << c`: one
set of relations, not two.

```cpp
Dimensionless lorentzFactor(Speed v);            // 1 / sqrt(1 - (v/c)^2)
Dimensionless lorentzFactor(const Velocity3& v);

Velocity4 fourVelocity(const Velocity3& v);
// u^mu = (gamma c, gamma v); normalized so u.u = -c^2 in the (-,+,+,+) signature

Dimensionless properTimeRate(Speed v);           // dTau/dt = 1/gamma; 1 at rest, 0 as v -> c

template <class SpeedAt>
Time properTimeElapsed(SpeedAt&& speedAt, Time from, Time to, std::size_t intervals = 1000);
// integral of dt/gamma over [from, to] via Simpson's rule; speedAt(Time) -> Speed

Velocity3 relativisticVelocityAdd(const Velocity3& u, const Velocity3& frameVelocity);
```

| Function | Notes |
| --- | --- |
| `lorentzFactor` | **Undefined (NaN) at or above the speed of light**, and nothing clamps that away: a clamped answer would be silently wrong; NaN at least propagates and is loud. |
| `properTimeElapsed` | Quadrature over a callable rather than a closed-form step, because a worldline's speed history is data, not a formula, in general. Units cross the boundary once (at the call into `simpson`) and back on the way out. |
| `relativisticVelocityAdd` | The general (non-collinear) formula, splitting `u` into components parallel/perpendicular to the boost direction. This is what `Frame`'s Galilean transform approximates at `v << c`. |

```cpp
const ysq::Dimensionless gamma = ysq::lorentzFactor(speed);
const ysq::Time tau = ysq::properTimeElapsed(
    [&](ysq::Time t) { return speedAtTime(t); }, t0, t1);
```

## `Physics/Mechanics/Dynamics.hpp`

`NBodyState`: the boundary between a span of `Body` and what `Math`'s
integrators actually run on. `Math`'s integrators need only a plain vector
space (see `OdeState` in [docs/api/math/integrators.md](../math/integrators.md));
a dimensioned `Quantity` from `Units` deliberately isn't that, so units
cross the boundary exactly once here, not once per force evaluation.

```cpp
class NBodyState {
public:
    using value_type = Vec3;

    NBodyState() = default;
    explicit NBodyState(std::size_t count);
    NBodyState(std::initializer_list<Vec3> values);

    std::size_t size() const noexcept;
    Vec3& operator[](std::size_t index) noexcept;   // asserted
    // begin/end, += -= *= /=, + - * /, ==; satisfies OdeState
};

NBodyState positionsOf(std::span<const Body> bodies);    // metres, in order
NBodyState velocitiesOf(std::span<const Body> bodies);    // m/s, Body::velocity()

void applyState(std::span<Body> bodies, const NBodyState& positions,
                const NBodyState& velocities);
// writes an integrated state back; momentum recovered as mass * velocity
```

Every gravity model in [docs/api/physics/gravity.md](gravity.md) reads and
writes `NBodyState` in its inner loop, never `Body` directly:

```cpp
std::vector<ysq::Body> bodies = /* masses, positions, momenta */;
ysq::NewtonianField field(bodies);

ysq::VelocityVerletStepper<ysq::NBodyState> stepper;
ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies), ysq::velocitiesOf(bodies)};
ysq::PhaseState<ysq::NBodyState> next;
stepper.step(field, 0.0, state, stepSize, next);

ysq::applyState(bodies, next.position, next.velocity);
```

`positionsOf`/`velocitiesOf`/`applyState` require `bodies`, `positions`, and
`velocities` to all be the same size (asserted, not checked, in release).

## `Physics/Mechanics/RigidBody.hpp`

General rigid-body rotation: gravity-gradient torque from a body's own
inertia asymmetry, and Euler's rotation equation, carried as an
inertial-frame angular momentum. General for any body with a nonzero
`principalMomentsOfInertia`, drawing on the identical inertia asymmetry
`Gravity/Newtonian.hpp`'s J2 term already reads.

```cpp
Torque3 gravityGradientTorque(const Body& oblateBody, const Length3& perturberPosition,
                              Mass perturberMass);
Torque3 gravityGradientTorque(const Body& oblateBody, std::span<const Body> perturbers);

void stepRigidBody(Body& body, std::span<const Body> perturbers, double dt);

struct PrincipalInertia {
    MomentOfInertia3 moments;   // ascending, matching jacobiEigenSymmetric's convention
    Quat orientation;           // principal-axis frame -> rawInertiaTensor's own frame
};

PrincipalInertia diagonalizeInertia(const Matrix3<double>& rawInertiaTensor);
```

| Function | Description |
| --- | --- |
| `gravityGradientTorque` | `tau = (3GM/r^3) rHat x (I . rHat)`, evaluated in `oblateBody`'s own body frame and rotated back by its current orientation. Zero whenever `principalMomentsOfInertia` is zero, without a branch. The span overload sums over every perturber. |
| `stepRigidBody(body, perturbers, dt)` | Advances `orientation` and `angularMomentum` by one fixed step under the gravity-gradient torque, then renormalizes the orientation (RK4 does not preserve the unit-quaternion constraint exactly). A no-op when `principalMomentsOfInertia` is zero. |
| `diagonalizeInertia(rawInertiaTensor)` | Eigendecomposes a symmetric inertia tensor (`Math/Eigen.hpp`'s `jacobiEigenSymmetric`) into `PrincipalInertia`: the moments are the eigenvalues, the orientation the eigenvectors as rotation columns. Only the lower triangle of `rawInertiaTensor` is read. The last axis is negated whenever the raw eigenvector triad comes out as a reflection (determinant -1), so `orientation` is always a proper rotation. |

```cpp
ysq::stepRigidBody(bodies[earthIndex], otherBodies, stepSize);

const ysq::PrincipalInertia principal = ysq::diagonalizeInertia(rawInertiaTensor);
spacecraft.principalMomentsOfInertia = principal.moments;
spacecraft.orientation = principal.orientation;
```

## `Physics/Mechanics/Spring.hpp`

Hooke's law with optional linear damping along the spring's own axis: a
general elastic-force model, not tied to any one scenario.

```cpp
using SpringConstant = Quantity<dim::SpringConstant>;         // force/length
using DampingCoefficient = Quantity<dim::DampingCoefficient>; // force/speed

Force3 springForce(const Body& on, const Length3& anchor, SpringConstant stiffness,
                   Length restLength, DampingCoefficient damping = DampingCoefficient{0.0});

Force3 springForce(const Body& a, const Body& b, SpringConstant stiffness, Length restLength,
                   DampingCoefficient damping = DampingCoefficient{0.0});
```

| Function | Description |
| --- | --- |
| `springForce(on, anchor, ...)` | `F = -k(\|r\|-restLength) rHat - c(v.rHat) rHat` against a fixed anchor point, which contributes no velocity of its own to the damping term. |
| `springForce(a, b, ...)` | The same law between two bodies; `b`'s own force is the exact negation (Newton's third law). |

```cpp
const ysq::Force3 tether = ysq::springForce(body, anchorPoint, stiffness, restLength, damping);
```

## `Physics/Mechanics/Drag.hpp`

Two drag regimes: linear (Stokes, low Reynolds number) and quadratic (high
Reynolds number, most everyday solid objects in air or water).

```cpp
using LinearDragCoefficient = Quantity<dim::LinearDragCoefficient>;  // force/speed

Force3 linearDragForce(const Body& on, LinearDragCoefficient coefficient,
                       const Velocity3& mediumVelocity = Velocity3::zero());

Force3 quadraticDragForce(const Body& on, Density mediumDensity, double dragCoefficient,
                          Area crossSectionalArea,
                          const Velocity3& mediumVelocity = Velocity3::zero());
```

| Function | Description |
| --- | --- |
| `linearDragForce` | `F = -b v_rel`: drag scaling with speed, the regime a small, slow object in a viscous medium is actually in. |
| `quadraticDragForce` | `F = -(1/2) rho Cd A \|v_rel\| v_rel`: drag scaling with speed squared. `dragCoefficient` is the dimensionless shape factor `Cd` (about 0.47 for a sphere). |

```cpp
const ysq::Force3 drag = ysq::quadraticDragForce(body, airDensity, 0.47, crossSection);
```

## `Physics/Mechanics/Friction.hpp`

Coulomb friction, split into its two regimes, both taking a contact normal
and a normal-force magnitude the caller already knows (from a collision
impulse, a resting contact, or `m g cos(theta)` on an incline).

```cpp
Force3 kineticFrictionForce(const Body& on, const Vec3& contactNormal,
                            Force normalForceMagnitude, double kineticFrictionCoefficient,
                            const Velocity3& surfaceVelocity = Velocity3::zero());

Force3 staticFrictionForce(const Vec3& contactNormal, Force normalForceMagnitude,
                           double staticFrictionCoefficient, const Force3& drivingForce);
```

| Function | Description |
| --- | --- |
| `kineticFrictionForce` | `F = -mu_k N tHat`, opposing the tangential sliding velocity. Zero if there is no tangential relative motion. |
| `staticFrictionForce` | Cancels `drivingForce`'s tangential component exactly while under the Coulomb limit `mu_s N` (equilibrium); returns the capped value `mu_s N` once the limit is exceeded, the point past which `kineticFrictionForce` becomes the correct law. |

```cpp
const ysq::Force3 friction =
    ysq::kineticFrictionForce(body, contactNormal, normalForce, muK);
```

## `Physics/Mechanics/Collision.hpp`

Contact detection and impulse-based resolution for spheres.

```cpp
struct Contact {
    Length3 point;
    Vec3 normal;             // first shape toward second
    Length penetrationDepth;
};

std::optional<Contact> detectCollision(const Body& a, const Body& b);              // sphere-sphere
std::optional<Contact> detectCollision(const Body& sphere, const AABB3<double>& box);

void resolveCollision(Body& a, Body& b, const Contact& contact, double restitution = 1.0);
void resolveCollision(Body& body, const Contact& contact, double restitution = 1.0);  // vs. immovable
```

| Function | Description |
| --- | --- |
| `detectCollision` | `nullopt` if the shapes do not overlap; each body's own `radius` is the sphere. |
| `resolveCollision` | The standard point-mass impulse formula along `contact.normal`. `restitution` of 0 is perfectly inelastic, 1 perfectly elastic. A no-op if the bodies are already separating along the normal. Does not itself correct the penetration; `contact.penetrationDepth` is there for a caller's own positional-correction scheme. The single-`Body` overload is the infinite-mass limit against an immovable object. |

```cpp
if (const auto contact = ysq::detectCollision(a, b)) {
    ysq::resolveCollision(a, b, *contact, /*restitution=*/0.8);
}
```

## `Physics/Mechanics/Constraints.hpp`

Sequential-impulse constraint solving with Baumgarte stabilization: one
Gauss-Seidel sweep per call, converging over a handful of sweeps for a
chain of constraints rather than an exact simultaneous solve.

```cpp
void solveDistanceConstraint(Body& a, Body& b, Length targetDistance, Time dt,
                             double baumgarteFactor = 0.2);
void solveDistanceConstraint(Body& body, const Length3& anchor, Length targetDistance, Time dt,
                             double baumgarteFactor = 0.2);

void solvePointConstraint(Body& body, const Length3& anchor, Time dt,
                          double baumgarteFactor = 0.2);
```

| Function | Description |
| --- | --- |
| `solveDistanceConstraint` | Keeps two bodies (or a body and a fixed anchor) `targetDistance` apart: a rigid rod, a pendulum, a chain link. |
| `solvePointConstraint` | Pins a body's position to `anchor` exactly, all three axes at once: a hinge with zero length, a weld. |
| `baumgarteFactor` | Trades correction speed against stability: 0 never corrects accumulated positional drift, 1 corrects it all in one (overshooting) step; 0.1-0.3 is conventional. |

```cpp
ysq::solveDistanceConstraint(bob, pivot, ropeLength, dt);
```

## `Physics/Mechanics/Hermite.hpp`

A 4th-order predictor-corrector (Makino & Aarseth 1992) and the per-body
scheduler built on it, giving each body its own step size instead of one
shared global step. See
[src/Physics/README.md](../../../src/Physics/README.md)'s "Individual
timesteps" section for the full derivation and the measured performance
result against the real Solar System catalog. Force-law-agnostic, the
same sense `NBodyState` above already is:
[docs/api/physics/gravity.md](gravity.md) supplies the concrete gravity
(and 1PN) jerk this is evaluated against.

```cpp
std::pair<Vec3, Vec3> hermitePredict(const Vec3& position, const Vec3& velocity,
                                     const Vec3& acceleration, const Vec3& jerk, double dt);

std::pair<Vec3, Vec3> hermiteCorrect(const Vec3& oldAcceleration, const Vec3& oldJerk,
                                     const Vec3& newAcceleration, const Vec3& newJerk,
                                     double dt, const Vec3& predictedPosition,
                                     const Vec3& predictedVelocity);

double hermiteTimestep(const Vec3& acceleration, const Vec3& jerk, double eta,
                       double baseInterval);
// dt = sqrt(eta * |a| / |jerk|), rounded down to a power-of-two fraction of baseInterval

template <class F>
concept IndividualJerkField =
    /* (bodyIndex, predictedPositions, predictedVelocities) -> pair<Vec3, Vec3> */;

class IndividualTimestepScheduler {
public:
    IndividualTimestepScheduler(NBodyState positions, NBodyState velocities,
                                NBodyState accelerations, NBodyState jerks,
                                double initialTime, double eta, double baseInterval);

    template <IndividualJerkField JerkField>
    void advanceTo(const JerkField& jerkField, double targetTime, int maxUpdates);

    std::pair<Vec3, Vec3> predictedState(std::size_t bodyIndex, double atTime) const;
    double currentTime() const noexcept;
    std::size_t bodyCount() const noexcept;
};
```

| Member | Description |
| --- | --- |
| `advanceTo` | Whichever body's own (last update + its own step) is soonest goes next: every other body predicted to that instant, the caller's jerk field asked for the mover's own new (acceleration, jerk), corrected, requeued with a freshly chosen step. Stops at `targetTime` or after `maxUpdates` single-body updates -- the individual-update analogue of a uniform stepper's `maxStepsPerAdvance`. |
| `predictedState` | Every body's position/velocity predicted to `atTime`, regardless of whether its own next scheduled update lands exactly there. Rendering, trails, and any energy/momentum diagnostic should all read through this, not a body's own last true update: total system energy is only meaningful when every body is read at the same instant, and bodies advancing at their own rate are essentially never all exactly synchronized except incidentally. |

**Not symplectic**: 4th-order accurate, the same category RK4 is in, not
`VelocityVerletStepper`'s -- no guarantee against slow energy drift over
an arbitrarily long run.

```cpp
ysq::RelativisticNBodyJerkSystem jerkField(bodies, primaryIndex, softening);
// initialAccelerations/initialJerks: jerkField(i, positions, velocities) for every i
ysq::IndividualTimestepScheduler scheduler(positions, velocities, initialAccelerations,
                                           initialJerks, 0.0, eta, baseInterval);
scheduler.advanceTo(jerkField, targetTime, maxUpdates);
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/mechanics)
and let us know.
