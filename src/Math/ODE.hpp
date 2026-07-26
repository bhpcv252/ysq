#pragma once

#include <Math/Scalar.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

namespace ysq {

/// The interface an ODE integrator works against, and the drivers that run
/// one. The methods themselves live in Math/Integrators/.
///
/// **Steppers are objects, not free functions.** Two independent reasons.
/// Dormand-Prince is FSAL, so it carries the last stage derivative into the
/// next step and costs six evaluations per accepted step instead of seven;
/// that is state across calls. And a heap-allocated state needs four
/// temporaries per RK4 step, which as free functions would mean four
/// allocations per step in an N-body inner loop. A stepper owns its scratch
/// and sizes it once.
///
/// Every stepper exposes `State`, `Scalar` and `order`, and a step() taking
/// (system, time, state, stepSize, out). What `system` means is the stepper's
/// business: an explicit method wants dy/dt = f(t, y), a symplectic one wants
/// an acceleration a(t, q). That is why the same drivers run both.

/// The scalar a state is built out of. A bare double is its own scalar; a
/// container recurses into its element type.
///
/// Recursion is what makes a nested state work: a PhaseState of Vector3 is
/// built out of doubles, not out of Vector3s, and the step size that
/// multiplies it has to be the innermost scalar.
template <class S>
struct StateScalar {
    using type = typename StateScalar<typename S::value_type>::type;
};

template <std::floating_point S>
struct StateScalar<S> {
    using type = S;
};

template <class S>
using StateScalarT = typename StateScalar<S>::type;

/// A state has to be a vector space over its scalar: that, and nothing more,
/// is what a Runge-Kutta method needs of it. The in-place forms are required
/// as well so a stepper can work in its own scratch rather than allocating.
template <class S>
concept OdeState = requires(S a, const S& b, StateScalarT<S> s) {
    { a + b } -> std::convertible_to<S>;
    { a - b } -> std::convertible_to<S>;
    { a * s } -> std::convertible_to<S>;
    { s * a } -> std::convertible_to<S>;
    a += b;
    a *= s;
};

/// dy/dt = f(t, y).
template <class F, class S>
concept OdeSystem = OdeState<S> && requires(const F& f, StateScalarT<S> t,
                                            const S& y) {
    { f(t, y) } -> std::convertible_to<S>;
};

/// a = a(t, q), the acceleration of a separable system H = T(p) + V(q).
///
/// Structurally identical to OdeSystem, and deliberately named differently
/// anyway: the two are not interchangeable, and handing an acceleration to
/// RK4 or a full derivative to Verlet compiles cleanly and integrates the
/// wrong problem. Wrap an acceleration with asPhaseSystem() to give it to an
/// explicit method.
///
/// Both concepts constrain the step() of every stepper. They cannot tell the
/// two apart, since nothing in the signature distinguishes them, but they do
/// turn a genuinely wrong callable into a constraint failure naming the call
/// site rather than an error from ten frames inside a Runge-Kutta stage.
template <class F, class S>
concept AccelerationField = OdeSystem<F, S>;

/// Position and velocity together, which is the state a second-order system
/// actually has.
///
/// It is itself an OdeState, so an explicit method can integrate one directly
/// through asPhaseSystem(). That is what lets RK4 and velocity Verlet be run
/// on the same problem and compared, which is the only way to show that
/// bounded energy error is a property of the method rather than of the setup.
template <OdeState S>
struct PhaseState {
    using value_type = S;

    S position{};
    S velocity{};

    /// Present so the componentwise error norm can walk into it. Index 0 is
    /// the position.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return 2; }

    [[nodiscard]] constexpr S& operator[](std::size_t index) noexcept {
        return (index == 0) ? position : velocity;
    }

    [[nodiscard]] constexpr const S& operator[](std::size_t index) const noexcept {
        return (index == 0) ? position : velocity;
    }

    constexpr PhaseState& operator+=(const PhaseState& other) {
        position += other.position;
        velocity += other.velocity;
        return *this;
    }

    constexpr PhaseState& operator-=(const PhaseState& other) {
        position -= other.position;
        velocity -= other.velocity;
        return *this;
    }

    constexpr PhaseState& operator*=(StateScalarT<S> scalar) {
        position *= scalar;
        velocity *= scalar;
        return *this;
    }

    constexpr PhaseState& operator/=(StateScalarT<S> scalar) {
        position /= scalar;
        velocity /= scalar;
        return *this;
    }

    [[nodiscard]] friend constexpr PhaseState operator+(
        const PhaseState& a, const PhaseState& b) {
        return {a.position + b.position, a.velocity + b.velocity};
    }

    [[nodiscard]] friend constexpr PhaseState operator-(
        const PhaseState& a, const PhaseState& b) {
        return {a.position - b.position, a.velocity - b.velocity};
    }

    [[nodiscard]] friend constexpr PhaseState operator-(const PhaseState& a) {
        return {-a.position, -a.velocity};
    }

    [[nodiscard]] friend constexpr PhaseState operator*(const PhaseState& a,
                                                        StateScalarT<S> scalar) {
        return {a.position * scalar, a.velocity * scalar};
    }

    [[nodiscard]] friend constexpr PhaseState operator*(StateScalarT<S> scalar,
                                                        const PhaseState& a) {
        return a * scalar;
    }

    [[nodiscard]] friend constexpr PhaseState operator/(const PhaseState& a,
                                                        StateScalarT<S> scalar) {
        return {a.position / scalar, a.velocity / scalar};
    }

    [[nodiscard]] friend constexpr bool operator==(const PhaseState&,
                                                   const PhaseState&) = default;
};

/// Turns an acceleration a(t, q) into the first-order system
/// d(q, v)/dt = (v, a(t, q)), so an explicit method can integrate it.
template <class Acceleration>
[[nodiscard]] constexpr auto asPhaseSystem(Acceleration acceleration) {
    return [acceleration](auto time, const auto& phase) {
        using Phase = std::remove_cvref_t<decltype(phase)>;
        return Phase{phase.velocity, acceleration(time, phase.position)};
    };
}

/// A state whose size is known only at run time, for an N-body problem or
/// anything else that is sized by its input.
///
/// Arithmetic on mismatched sizes is a precondition, asserted in debug builds
/// and unchecked in release. An integrator that changes the size of its own
/// state mid-run has gone wrong somewhere findable, and an assertion names
/// that place; a check on every element of every operation would not, and
/// would be paid for on every step.
template <std::floating_point T>
class StateVector {
public:
    using value_type = T;

    StateVector() = default;

    explicit StateVector(std::size_t count, T fill = T{}) : m_values(count, fill) {}

    StateVector(std::initializer_list<T> values) : m_values(values) {}

    [[nodiscard]] std::size_t size() const noexcept { return m_values.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_values.empty(); }
    void resize(std::size_t count) { m_values.resize(count); }

    [[nodiscard]] T& operator[](std::size_t index) noexcept {
        assert(index < m_values.size());
        return m_values[index];
    }

    [[nodiscard]] const T& operator[](std::size_t index) const noexcept {
        assert(index < m_values.size());
        return m_values[index];
    }

    [[nodiscard]] auto begin() noexcept { return m_values.begin(); }
    [[nodiscard]] auto end() noexcept { return m_values.end(); }
    [[nodiscard]] auto begin() const noexcept { return m_values.begin(); }
    [[nodiscard]] auto end() const noexcept { return m_values.end(); }
    [[nodiscard]] const T* data() const noexcept { return m_values.data(); }

    StateVector& operator+=(const StateVector& other) {
        assert(m_values.size() == other.m_values.size());
        for (std::size_t i = 0; i < m_values.size(); ++i) {
            m_values[i] += other.m_values[i];
        }
        return *this;
    }

    StateVector& operator-=(const StateVector& other) {
        assert(m_values.size() == other.m_values.size());
        for (std::size_t i = 0; i < m_values.size(); ++i) {
            m_values[i] -= other.m_values[i];
        }
        return *this;
    }

    StateVector& operator*=(T scalar) {
        for (T& value : m_values) {
            value *= scalar;
        }
        return *this;
    }

    StateVector& operator/=(T scalar) {
        for (T& value : m_values) {
            value /= scalar;
        }
        return *this;
    }

    [[nodiscard]] friend StateVector operator+(StateVector a,
                                               const StateVector& b) {
        a += b;
        return a;
    }

    [[nodiscard]] friend StateVector operator-(StateVector a,
                                               const StateVector& b) {
        a -= b;
        return a;
    }

    [[nodiscard]] friend StateVector operator-(StateVector a) {
        a *= T{-1};
        return a;
    }

    [[nodiscard]] friend StateVector operator*(StateVector a, T scalar) {
        a *= scalar;
        return a;
    }

    [[nodiscard]] friend StateVector operator*(T scalar, StateVector a) {
        a *= scalar;
        return a;
    }

    [[nodiscard]] friend StateVector operator/(StateVector a, T scalar) {
        a /= scalar;
        return a;
    }

    [[nodiscard]] friend bool operator==(const StateVector&,
                                         const StateVector&) = default;

private:
    std::vector<T> m_values;
};

namespace detail {

/// Walks a state down to its scalar leaves, accumulating the weighted error
/// ratio. Recursive, so it handles a PhaseState of Vector3 the same way it
/// handles a bare double, and it is one pass rather than one pass per
/// component.
template <class S, class T>
void accumulateErrorRatio(const S& error, const S& current, const S& next, T absTol,
                          T relTol, T& sumOfSquares, std::size_t& count) {
    if constexpr (std::floating_point<S>) {
        const T scale =
            absTol + relTol * std::max(std::abs(current), std::abs(next));
        const T ratio = error / scale;
        sumOfSquares += ratio * ratio;
        ++count;
    } else {
        for (std::size_t i = 0; i < current.size(); ++i) {
            accumulateErrorRatio(error[i], current[i], next[i], absTol, relTol,
                                 sumOfSquares, count);
        }
    }
}

}  // namespace detail

/// Root-mean-square of the error in each component, measured against that
/// component's own tolerance.
///
/// Mixed absolute and relative, per component, rather than a single norm of
/// the whole state. A position of 1e11 metres and a velocity of 1e4 metres per
/// second have nothing to say to each other on one scale, and a plain norm
/// would let the large component set the step for all of them. An answer at or
/// below 1 means the step is acceptable.
template <class S, class T = StateScalarT<S>>
[[nodiscard]] T errorNorm(const S& error, const S& current, const S& next,
                          T absTol, T relTol) {
    T sumOfSquares{};
    std::size_t count = 0;
    detail::accumulateErrorRatio(error, current, next, absTol, relTol, sumOfSquares,
                                 count);
    if (count == 0) {
        return T{0};
    }
    return std::sqrt(sumOfSquares / static_cast<T>(count));
}

template <std::floating_point T>
struct AdaptiveSettings {
    /// Defaults scale with the precision of T. A fixed 1e-9 is a reasonable
    /// ask of a double and simply unreachable for a float, whose epsilon is
    /// already 1.2e-7.
    T absoluteTolerance = std::sqrt(std::numeric_limits<T>::epsilon());
    T relativeTolerance = std::sqrt(std::numeric_limits<T>::epsilon());

    T minimumStep = T{0};
    T maximumStep = std::numeric_limits<T>::infinity();

    /// Aim slightly under the predicted step, so a step is usually accepted
    /// rather than usually rejected.
    T safety = static_cast<T>(0.9);
    /// Bounds on how far one step may change, which keeps the controller from
    /// chasing a single unlucky estimate.
    T minimumScale = static_cast<T>(0.2);
    T maximumScale = static_cast<T>(5);

    int maximumRejections = 20;

    /// A ceiling on accepted steps, so a run that is merely very slow is
    /// reported rather than continuing indefinitely. maximumRejections bounds
    /// how long the controller may struggle at one point; this bounds the whole
    /// integration, which is what an unattended run needs.
    std::size_t maximumSteps = 10'000'000;
};

template <class S, std::floating_point T>
struct AdaptiveResult {
    S state{};
    T time{};
    std::size_t acceptedSteps = 0;
    std::size_t rejectedSteps = 0;
    std::size_t evaluations = 0;
    /// False if the controller hit maximumRejections or drove the step below
    /// minimumStep before reaching the end.
    bool succeeded = false;
};

/// Fixed-step integration from `from` to `to`.
///
/// The step is adjusted down to the nearest divisor of the interval, so the
/// run lands exactly on `to` and every step is the same size. A loop that
/// instead adds h until it passes the end accumulates a rounding error in the
/// final time and takes one short step, and both of those corrupt an order-of-
/// accuracy measurement, which is what this is mostly used for.
template <class Stepper, class System, class Observer>
auto integrate(Stepper& stepper, const System& system,
               typename Stepper::State state, typename Stepper::Scalar from,
               typename Stepper::Scalar to, typename Stepper::Scalar step,
               Observer&& observe) -> typename Stepper::State {
    using T = typename Stepper::Scalar;

    const T span = to - from;
    if (!(T{0} < step) || span == T{0}) {
        observe(from, state);
        return state;
    }

    const auto count = static_cast<std::size_t>(
        std::max(T{1}, std::ceil(std::abs(span) / step)));
    const T actualStep = span / static_cast<T>(count);

    observe(from, state);

    typename Stepper::State next = state;
    for (std::size_t i = 0; i < count; ++i) {
        const T time = from + static_cast<T>(i) * actualStep;
        stepper.step(system, time, state, actualStep, next);
        state = next;
        // Recomputed from the index rather than accumulated, so the reported
        // time does not drift over a long run.
        observe(from + static_cast<T>(i + 1) * actualStep, state);
    }
    return state;
}

template <class Stepper, class System>
auto integrate(Stepper& stepper, const System& system,
               typename Stepper::State state, typename Stepper::Scalar from,
               typename Stepper::Scalar to, typename Stepper::Scalar step) ->
    typename Stepper::State {
    return integrate(stepper, system, state, from, to, step,
                     [](auto, const auto&) {});
}

/// The number of fixed steps integrate() will actually take.
template <std::floating_point T>
[[nodiscard]] std::size_t stepCount(T from, T to, T step) {
    const T span = to - from;
    if (!(T{0} < step) || span == T{0}) {
        return 0;
    }
    return static_cast<std::size_t>(std::max(T{1}, std::ceil(std::abs(span) / step)));
}

/// Adaptive integration with a PI step controller.
///
/// The step for the next attempt comes from the error estimate of this one and
/// of the previous one. A plain proportional controller reacts to each
/// estimate alone and oscillates: it overshoots, gets rejected, overcorrects,
/// and wastes evaluations. The integral term damps that, at the cost of two
/// exponents that have to be matched to the method's order.
///
/// Runs in either direction, like the fixed-step driver. `step` below is
/// always a magnitude and the direction is applied where it is used, which is
/// what keeps the controller and the bounds in `settings` reading the same way
/// for a backward run as for a forward one.
///
/// `initialStep` is a starting guess, not a promise; the controller replaces
/// it after one step. A zero or non-finite guess is replaced with a hundredth
/// of the interval rather than being taken literally, since taken literally it
/// would make no progress at all.
template <class Stepper, class System, class Observer>
auto integrateAdaptive(Stepper& stepper, const System& system,
                       typename Stepper::State state,
                       typename Stepper::Scalar from, typename Stepper::Scalar to,
                       typename Stepper::Scalar initialStep,
                       const AdaptiveSettings<typename Stepper::Scalar>& settings,
                       Observer&& observe)
    -> AdaptiveResult<typename Stepper::State, typename Stepper::Scalar> {
    using S = typename Stepper::State;
    using T = typename Stepper::Scalar;

    const std::size_t evaluationsAtStart = stepper.evaluations();

    AdaptiveResult<S, T> result;
    result.state = state;
    result.time = from;

    observe(from, state);

    if (from == to) {
        result.succeeded = true;
        return result;
    }

    const T span = to - from;
    const T direction = (span < T{0}) ? T{-1} : T{1};

    // Hairer's exponents for a method whose error estimate is of order p: the
    // integral term takes a small share of the proportional one.
    const T beta = static_cast<T>(0.04);
    const T alpha = T{1} / static_cast<T>(Stepper::order) - static_cast<T>(0.75) * beta;

    const T guess = std::abs(initialStep);
    T step = std::min((guess > T{0}) ? guess : std::abs(span) / T{100},
                      settings.maximumStep);
    T previousError = T{1};
    S next = state;
    // Sized from the state rather than default-constructed, so a state whose
    // extent is only known at run time has somewhere to be written.
    S error = state;
    int consecutiveRejections = 0;

    const auto recordCost = [&] {
        result.evaluations = stepper.evaluations() - evaluationsAtStart;
    };

    while (direction * (to - result.time) > T{0}) {
        step = std::min(step, std::abs(to - result.time));

        // Catches a zero step, a NaN one, and a floor set above what is left to
        // cover. Without it the loop makes no progress and never ends.
        if (!(step > T{0}) || step < settings.minimumStep) {
            recordCost();
            return result;
        }

        stepper.step(system, result.time, result.state, direction * step, next,
                     error);

        const T measured = errorNorm(error, result.state, next,
                                     settings.absoluteTolerance,
                                     settings.relativeTolerance);

        // An exactly zero error would divide by zero in the controller; treat
        // it as the smallest error worth reacting to.
        const T clamped = std::max(measured, std::numeric_limits<T>::epsilon());

        if (result.acceptedSteps >= settings.maximumSteps) {
            recordCost();
            return result;
        }

        if (measured <= T{1}) {
            const T advanced = result.time + direction * step;
            // A step too small to change the time at all would otherwise spin
            // here forever, accepting and advancing by nothing.
            if (advanced == result.time) {
                recordCost();
                return result;
            }

            result.time = advanced;
            result.state = next;
            ++result.acceptedSteps;
            consecutiveRejections = 0;
            observe(result.time, result.state);

            const T factor = settings.safety * std::pow(clamped, -alpha) *
                             std::pow(previousError, beta);
            step *= clamp(factor, settings.minimumScale, settings.maximumScale);
            step = std::min(step, settings.maximumStep);
            previousError = std::max(clamped, static_cast<T>(1e-4));
        } else {
            ++result.rejectedSteps;
            if (++consecutiveRejections > settings.maximumRejections) {
                recordCost();
                return result;
            }
            // No integral term on a rejection: the previous error describes a
            // step that was not taken.
            const T factor = settings.safety * std::pow(clamped, -alpha);
            step *= std::max(factor, settings.minimumScale);
        }
    }

    recordCost();
    result.succeeded = true;
    return result;
}

template <class Stepper, class System>
auto integrateAdaptive(Stepper& stepper, const System& system,
                       typename Stepper::State state,
                       typename Stepper::Scalar from, typename Stepper::Scalar to,
                       typename Stepper::Scalar initialStep,
                       const AdaptiveSettings<typename Stepper::Scalar>& settings)
    -> AdaptiveResult<typename Stepper::State, typename Stepper::Scalar> {
    return integrateAdaptive(stepper, system, state, from, to, initialStep, settings,
                             [](auto, const auto&) {});
}

}  // namespace ysq
