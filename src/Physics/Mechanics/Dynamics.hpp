#pragma once

#include <Math/Vector3.hpp>
#include <Physics/Body.hpp>

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <vector>

namespace ysq {

/// The state N-body integration actually runs on: one raw, unitless Vec3 per
/// body, either every position or every velocity.
///
/// Math's integrators need a vector space over a scalar and nothing more (see
/// OdeState in Math/ODE.hpp): addition, subtraction, and scaling. Units cross
/// the boundary once, converting a span of Body into this and back, the same
/// boundary units_kinematics.cpp draws for a single body; see
/// src/Physics/README.md's "Units cross the boundary once" section. Every
/// force law under Physics/Gravity that plugs
/// into a symplectic or Runge-Kutta stepper produces and consumes this type
/// rather than a span of Body directly.
class NBodyState {
public:
    using value_type = Vec3;

    NBodyState() = default;
    explicit NBodyState(std::size_t count) : m_values(count) {}
    NBodyState(std::initializer_list<Vec3> values) : m_values(values) {}

    [[nodiscard]] std::size_t size() const noexcept { return m_values.size(); }

    [[nodiscard]] Vec3& operator[](std::size_t index) noexcept {
        assert(index < m_values.size());
        return m_values[index];
    }

    [[nodiscard]] const Vec3& operator[](std::size_t index) const noexcept {
        assert(index < m_values.size());
        return m_values[index];
    }

    [[nodiscard]] auto begin() noexcept { return m_values.begin(); }
    [[nodiscard]] auto end() noexcept { return m_values.end(); }
    [[nodiscard]] auto begin() const noexcept { return m_values.begin(); }
    [[nodiscard]] auto end() const noexcept { return m_values.end(); }

    NBodyState& operator+=(const NBodyState& other) noexcept {
        assert(m_values.size() == other.m_values.size());
        for (std::size_t i = 0; i < m_values.size(); ++i) {
            m_values[i] += other.m_values[i];
        }
        return *this;
    }

    NBodyState& operator-=(const NBodyState& other) noexcept {
        assert(m_values.size() == other.m_values.size());
        for (std::size_t i = 0; i < m_values.size(); ++i) {
            m_values[i] -= other.m_values[i];
        }
        return *this;
    }

    NBodyState& operator*=(double scalar) noexcept {
        for (Vec3& value : m_values) {
            value *= scalar;
        }
        return *this;
    }

    NBodyState& operator/=(double scalar) noexcept {
        for (Vec3& value : m_values) {
            value /= scalar;
        }
        return *this;
    }

    [[nodiscard]] friend NBodyState operator+(NBodyState a,
                                              const NBodyState& b) noexcept {
        a += b;
        return a;
    }

    [[nodiscard]] friend NBodyState operator-(NBodyState a,
                                              const NBodyState& b) noexcept {
        a -= b;
        return a;
    }

    [[nodiscard]] friend NBodyState operator-(NBodyState a) noexcept {
        a *= -1.0;
        return a;
    }

    [[nodiscard]] friend NBodyState operator*(NBodyState a, double scalar) noexcept {
        a *= scalar;
        return a;
    }

    [[nodiscard]] friend NBodyState operator*(double scalar, NBodyState a) noexcept {
        a *= scalar;
        return a;
    }

    [[nodiscard]] friend NBodyState operator/(NBodyState a, double scalar) noexcept {
        a /= scalar;
        return a;
    }

    [[nodiscard]] friend bool operator==(const NBodyState&, const NBodyState&) = default;

private:
    std::vector<Vec3> m_values;
};

/// The position of every body, in metres, in the order given.
[[nodiscard]] NBodyState positionsOf(std::span<const Body> bodies);

/// The velocity of every body, in metres per second: Body::velocity(), which
/// is the non-relativistic p / m.
[[nodiscard]] NBodyState velocitiesOf(std::span<const Body> bodies);

/// Writes an integrated state back into a span of Body. Momentum is
/// recovered as mass times velocity, the same non-relativistic relation
/// velocitiesOf inverts; bodies, positions and velocities must all be the
/// same size.
void applyState(std::span<Body> bodies, const NBodyState& positions,
                const NBodyState& velocities);

}  // namespace ysq
