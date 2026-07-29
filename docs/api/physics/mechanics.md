# Physics/Mechanics API reference

`Body`, reference frames, relativistic kinematics, and `NBodyState`: the
boundary where a span of `Body` crosses into what `Math`'s integrators
actually run on. Start with
[docs/physics/mechanics.md](../../physics/mechanics.md) for the ideas;
[src/Physics/README.md](../../../src/Physics/README.md) for the full design
notes shared across all seven theories.

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

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/mechanics)
and let us know.
