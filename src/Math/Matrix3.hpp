#pragma once

#include <Math/Matrix2.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

/// Three by three matrix: rotations and the general linear part of a 3D
/// transform, plus symmetric things like an inertia tensor.
///
/// Column-major storage, column-vector convention, exactly as Matrix2. See
/// that header for the reasoning.
template <Numeric T>
struct Matrix3 {
    using value_type = T;
    using Column = Vector3<T>;

    std::array<Column, 3> columns{};

    [[nodiscard]] static constexpr std::size_t rows() noexcept { return 3; }
    [[nodiscard]] static constexpr std::size_t cols() noexcept { return 3; }

    [[nodiscard]] constexpr Column& operator[](std::size_t col) noexcept {
        assert(col < cols());
        return columns[col];
    }

    [[nodiscard]] constexpr const Column& operator[](std::size_t col) const noexcept {
        assert(col < cols());
        return columns[col];
    }

    [[nodiscard]] constexpr T& operator()(std::size_t row, std::size_t col) noexcept {
        assert(row < rows() && col < cols());
        return columns[col][row];
    }

    [[nodiscard]] constexpr const T& operator()(std::size_t row,
                                                std::size_t col) const noexcept {
        assert(row < rows() && col < cols());
        return columns[col][row];
    }

    [[nodiscard]] constexpr Column row(std::size_t index) const noexcept {
        return {columns[0][index], columns[1][index], columns[2][index]};
    }

    [[nodiscard]] constexpr Matrix2<T> upperLeft2x2() const noexcept {
        return Matrix2<T>::fromColumns(columns[0].xy(), columns[1].xy());
    }

    [[nodiscard]] static constexpr Matrix3 zero() noexcept { return {}; }

    [[nodiscard]] static constexpr Matrix3 fromColumns(const Column& c0,
                                                       const Column& c1,
                                                       const Column& c2) noexcept {
        return Matrix3{std::array<Column, 3>{{c0, c1, c2}}};
    }

    /// Takes its arguments the way a matrix is written on paper.
    [[nodiscard]] static constexpr Matrix3 fromRows(const Column& r0,
                                                    const Column& r1,
                                                    const Column& r2) noexcept {
        return fromColumns({r0.x, r1.x, r2.x}, {r0.y, r1.y, r2.y},
                           {r0.z, r1.z, r2.z});
    }

    [[nodiscard]] static constexpr Matrix3 identity() noexcept {
        return fromColumns({T{1}, T{0}, T{0}}, {T{0}, T{1}, T{0}},
                           {T{0}, T{0}, T{1}});
    }

    [[nodiscard]] static constexpr Matrix3 diagonal(const Column& d) noexcept {
        return fromColumns({d.x, T{0}, T{0}}, {T{0}, d.y, T{0}},
                           {T{0}, T{0}, d.z});
    }

    [[nodiscard]] static constexpr Matrix3 scale(const Column& s) noexcept {
        return diagonal(s);
    }

    /// The outer product a b^T. Rank one, and the building block of a
    /// projection matrix.
    [[nodiscard]] static constexpr Matrix3 outerProduct(const Column& a,
                                                        const Column& b) noexcept {
        return fromColumns(a * b.x, a * b.y, a * b.z);
    }

    /// The matrix that reproduces cross(a, v) when applied to v: the
    /// skew-symmetric [a]_x. Angular velocity acts on a body this way.
    [[nodiscard]] static constexpr Matrix3 crossMatrix(const Column& a) noexcept {
        return fromRows({T{0}, -a.z, a.y}, {a.z, T{0}, -a.x}, {-a.y, a.x, T{0}});
    }

    /// Right-handed, counter-clockwise looking down the axis toward the
    /// origin.
    [[nodiscard]] static Matrix3 rotationX(T angle) {
        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        return fromRows({T{1}, T{0}, T{0}}, {T{0}, c, -s}, {T{0}, s, c});
    }

    [[nodiscard]] static Matrix3 rotationY(T angle) {
        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        return fromRows({c, T{0}, s}, {T{0}, T{1}, T{0}}, {-s, T{0}, c});
    }

    [[nodiscard]] static Matrix3 rotationZ(T angle) {
        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        return fromRows({c, -s, T{0}}, {s, c, T{0}}, {T{0}, T{0}, T{1}});
    }

    /// Rodrigues' rotation about a unit axis, right-handed.
    [[nodiscard]] static Matrix3 rotation(const Column& axis, T angle) {
        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        return identity() * c + crossMatrix(axis) * s +
               outerProduct(axis, axis) * (T{1} - c);
    }

    constexpr Matrix3& operator+=(const Matrix3& other) noexcept {
        for (std::size_t i = 0; i < 3; ++i) {
            columns[i] += other.columns[i];
        }
        return *this;
    }

    constexpr Matrix3& operator-=(const Matrix3& other) noexcept {
        for (std::size_t i = 0; i < 3; ++i) {
            columns[i] -= other.columns[i];
        }
        return *this;
    }

    constexpr Matrix3& operator*=(T scalar) noexcept {
        for (std::size_t i = 0; i < 3; ++i) {
            columns[i] *= scalar;
        }
        return *this;
    }

    constexpr Matrix3& operator/=(T scalar) noexcept {
        for (std::size_t i = 0; i < 3; ++i) {
            columns[i] /= scalar;
        }
        return *this;
    }

    constexpr Matrix3& operator*=(const Matrix3& other) noexcept {
        *this = *this * other;
        return *this;
    }

    [[nodiscard]] friend constexpr Matrix3 operator+(const Matrix3& m) noexcept {
        return m;
    }

    [[nodiscard]] friend constexpr Matrix3 operator-(const Matrix3& m) noexcept {
        return fromColumns(-m.columns[0], -m.columns[1], -m.columns[2]);
    }

    [[nodiscard]] friend constexpr Matrix3 operator+(const Matrix3& a,
                                                     const Matrix3& b) noexcept {
        return fromColumns(a.columns[0] + b.columns[0], a.columns[1] + b.columns[1],
                           a.columns[2] + b.columns[2]);
    }

    [[nodiscard]] friend constexpr Matrix3 operator-(const Matrix3& a,
                                                     const Matrix3& b) noexcept {
        return fromColumns(a.columns[0] - b.columns[0], a.columns[1] - b.columns[1],
                           a.columns[2] - b.columns[2]);
    }

    [[nodiscard]] friend constexpr Matrix3 operator*(const Matrix3& m,
                                                     T scalar) noexcept {
        return fromColumns(m.columns[0] * scalar, m.columns[1] * scalar,
                           m.columns[2] * scalar);
    }

    [[nodiscard]] friend constexpr Matrix3 operator*(T scalar,
                                                     const Matrix3& m) noexcept {
        return m * scalar;
    }

    [[nodiscard]] friend constexpr Matrix3 operator/(const Matrix3& m,
                                                     T scalar) noexcept {
        return fromColumns(m.columns[0] / scalar, m.columns[1] / scalar,
                           m.columns[2] / scalar);
    }

    [[nodiscard]] friend constexpr Column operator*(const Matrix3& m,
                                                    const Column& v) noexcept {
        return m.columns[0] * v.x + m.columns[1] * v.y + m.columns[2] * v.z;
    }

    [[nodiscard]] friend constexpr Matrix3 operator*(const Matrix3& a,
                                                     const Matrix3& b) noexcept {
        return fromColumns(a * b.columns[0], a * b.columns[1], a * b.columns[2]);
    }

    [[nodiscard]] friend constexpr bool operator==(const Matrix3&,
                                                   const Matrix3&) = default;
};

template <Numeric T>
[[nodiscard]] constexpr Matrix3<T> transpose(const Matrix3<T>& m) noexcept {
    return Matrix3<T>::fromRows(m.columns[0], m.columns[1], m.columns[2]);
}

/// The scalar triple product of the columns, which is the same number.
template <Numeric T>
[[nodiscard]] constexpr T determinant(const Matrix3<T>& m) noexcept {
    return scalarTriple(m.columns[0], m.columns[1], m.columns[2]);
}

template <Numeric T>
[[nodiscard]] constexpr T trace(const Matrix3<T>& m) noexcept {
    return m(0, 0) + m(1, 1) + m(2, 2);
}

namespace detail {

/// Rows of the inverse are the cross products of the other two columns over
/// the determinant. That is the adjugate, written in the form column-major
/// storage already has to hand.
template <Numeric T>
[[nodiscard]] constexpr Matrix3<T> adjugateRows(const Matrix3<T>& m) noexcept {
    return Matrix3<T>::fromRows(cross(m.columns[1], m.columns[2]),
                                cross(m.columns[2], m.columns[0]),
                                cross(m.columns[0], m.columns[1]));
}

}  // namespace detail

/// nullopt when the matrix is singular or holds a NaN. Near-singular is not
/// detected; use solve() for that.
template <Numeric T>
[[nodiscard]] constexpr std::optional<Matrix3<T>> tryInverse(
    const Matrix3<T>& m) noexcept {
    const T det = determinant(m);
    // Finiteness as well as non-zero. A determinant that overflowed used to
    // pass both of the old checks and then divide the adjugate down to a zero
    // matrix, reported as a success.
    if (det == T{0} || !detail::isFiniteValue(det)) {
        return std::nullopt;
    }
    return detail::adjugateRows(m) / det;
}

/// Unchecked: a singular matrix yields infinities or NaN rather than an error.
template <Numeric T>
[[nodiscard]] constexpr Matrix3<T> inverse(const Matrix3<T>& m) noexcept {
    return detail::adjugateRows(m) / determinant(m);
}

/// The inverse of a rotation is its transpose, at a fraction of the cost.
/// Correct only for an orthogonal matrix; wrong, silently, for anything with
/// scale or shear in it.
template <Numeric T>
[[nodiscard]] constexpr Matrix3<T> inverseOrthogonal(const Matrix3<T>& m) noexcept {
    return transpose(m);
}

/// Solves M x = b by elimination with partial pivoting. nullopt if singular.
template <Numeric T>
[[nodiscard]] std::optional<Vector3<T>> solve(const Matrix3<T>& m,
                                              const Vector3<T>& b) {
    return detail::solveByElimination(m, b);
}

using Mat3 = Matrix3<double>;
using Mat3f = Matrix3<float>;

}  // namespace ysq
