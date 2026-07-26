#pragma once

#include <Math/Scalar.hpp>
#include <Math/Vector2.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

namespace detail {

/// Gauss-Jordan elimination with partial pivoting, generic over the fixed-size
/// matrix types: anything with operator()(row, col) and a Column exposing
/// size(). Lives in the smallest matrix header so Matrix3 and Matrix4 can use
/// it through the include chain.
///
/// This is the conditioning-sensitive path. The cofactor inverses in these
/// headers are exact expressions evaluated in floating point, which is fine
/// for a well-conditioned transform and progressively worse as a matrix
/// approaches singular. Pivoting is what keeps the error bounded there, at the
/// cost of branches and no constexpr. Reach for it when the matrix comes from
/// measurement or from a metric rather than from a transform you built.
///
/// Takes both arguments by value: elimination destroys them.
template <class M, class V>
[[nodiscard]] std::optional<V> solveByElimination(M a, V b) {
    using T = typename V::value_type;
    constexpr std::size_t n = V::size();

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        T best = absOf(a(col, col));
        for (std::size_t row = col + 1; row < n; ++row) {
            const T candidate = absOf(a(row, col));
            if (best < candidate) {
                best = candidate;
                pivot = row;
            }
        }

        // Catches an exact zero and a NaN alike, neither being greater than
        // zero, and an infinity, which would divide the row down to zeros.
        if (!(T{0} < best) || !isFiniteValue(best)) {
            return std::nullopt;
        }

        if (pivot != col) {
            for (std::size_t k = 0; k < n; ++k) {
                const T swapped = a(col, k);
                a(col, k) = a(pivot, k);
                a(pivot, k) = swapped;
            }
            const T swapped = b[col];
            b[col] = b[pivot];
            b[pivot] = swapped;
        }

        const T diagonal = a(col, col);
        for (std::size_t k = col; k < n; ++k) {
            a(col, k) /= diagonal;
        }
        b[col] /= diagonal;

        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const T factor = a(row, col);
            if (factor == T{0}) {
                continue;
            }
            for (std::size_t k = col; k < n; ++k) {
                a(row, k) -= factor * a(col, k);
            }
            b[row] -= factor * b[col];
        }
    }

    return b;
}

}  // namespace detail

/// Two by two matrix.
///
/// **Column-major storage, column-vector convention.** `columns[j]` is the
/// j-th column, transforms compose right to left (`M = T * R * S`) and apply
/// as `v' = M * v`. This is GLSL's convention, so a matrix uploads to a
/// uniform with `transpose = GL_FALSE` and the shader code reads the same as
/// the C++.
///
/// operator[] indexes a column; operator()(row, col) indexes an element in the
/// order matrices are written and read. Mixing them up is the classic bug
/// here, so the two spellings are deliberately different shapes.
template <Numeric T>
struct Matrix2 {
    using value_type = T;
    using Column = Vector2<T>;

    std::array<Column, 2> columns{};

    [[nodiscard]] static constexpr std::size_t rows() noexcept { return 2; }
    [[nodiscard]] static constexpr std::size_t cols() noexcept { return 2; }

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
        return {columns[0][index], columns[1][index]};
    }

    [[nodiscard]] static constexpr Matrix2 zero() noexcept { return {}; }

    [[nodiscard]] static constexpr Matrix2 fromColumns(const Column& c0,
                                                       const Column& c1) noexcept {
        return Matrix2{std::array<Column, 2>{{c0, c1}}};
    }

    /// Takes its arguments the way a matrix is written on paper, which is what
    /// makes a literal in a test readable.
    [[nodiscard]] static constexpr Matrix2 fromRows(const Column& r0,
                                                    const Column& r1) noexcept {
        return fromColumns({r0.x, r1.x}, {r0.y, r1.y});
    }

    [[nodiscard]] static constexpr Matrix2 identity() noexcept {
        return fromColumns({T{1}, T{0}}, {T{0}, T{1}});
    }

    [[nodiscard]] static constexpr Matrix2 diagonal(const Column& d) noexcept {
        return fromColumns({d.x, T{0}}, {T{0}, d.y});
    }

    [[nodiscard]] static constexpr Matrix2 scale(const Column& s) noexcept {
        return diagonal(s);
    }

    /// Counter-clockwise by `angle` radians.
    [[nodiscard]] static Matrix2 rotation(T angle) {
        using std::cos;
        using std::sin;
        const T c = cos(angle);
        const T s = sin(angle);
        return fromRows({c, -s}, {s, c});
    }

    constexpr Matrix2& operator+=(const Matrix2& other) noexcept {
        columns[0] += other.columns[0];
        columns[1] += other.columns[1];
        return *this;
    }

    constexpr Matrix2& operator-=(const Matrix2& other) noexcept {
        columns[0] -= other.columns[0];
        columns[1] -= other.columns[1];
        return *this;
    }

    constexpr Matrix2& operator*=(T scalar) noexcept {
        columns[0] *= scalar;
        columns[1] *= scalar;
        return *this;
    }

    constexpr Matrix2& operator/=(T scalar) noexcept {
        columns[0] /= scalar;
        columns[1] /= scalar;
        return *this;
    }

    /// Right-multiplies: `a *= b` is `a = a * b`, matching how transforms
    /// compose.
    constexpr Matrix2& operator*=(const Matrix2& other) noexcept {
        *this = *this * other;
        return *this;
    }

    [[nodiscard]] friend constexpr Matrix2 operator+(const Matrix2& m) noexcept {
        return m;
    }

    [[nodiscard]] friend constexpr Matrix2 operator-(const Matrix2& m) noexcept {
        return fromColumns(-m.columns[0], -m.columns[1]);
    }

    [[nodiscard]] friend constexpr Matrix2 operator+(const Matrix2& a,
                                                     const Matrix2& b) noexcept {
        return fromColumns(a.columns[0] + b.columns[0], a.columns[1] + b.columns[1]);
    }

    [[nodiscard]] friend constexpr Matrix2 operator-(const Matrix2& a,
                                                     const Matrix2& b) noexcept {
        return fromColumns(a.columns[0] - b.columns[0], a.columns[1] - b.columns[1]);
    }

    [[nodiscard]] friend constexpr Matrix2 operator*(const Matrix2& m,
                                                     T scalar) noexcept {
        return fromColumns(m.columns[0] * scalar, m.columns[1] * scalar);
    }

    [[nodiscard]] friend constexpr Matrix2 operator*(T scalar,
                                                     const Matrix2& m) noexcept {
        return m * scalar;
    }

    [[nodiscard]] friend constexpr Matrix2 operator/(const Matrix2& m,
                                                     T scalar) noexcept {
        return fromColumns(m.columns[0] / scalar, m.columns[1] / scalar);
    }

    /// Linear combination of the columns, which is what column-major storage
    /// makes the natural order.
    [[nodiscard]] friend constexpr Column operator*(const Matrix2& m,
                                                    const Column& v) noexcept {
        return m.columns[0] * v.x + m.columns[1] * v.y;
    }

    [[nodiscard]] friend constexpr Matrix2 operator*(const Matrix2& a,
                                                     const Matrix2& b) noexcept {
        return fromColumns(a * b.columns[0], a * b.columns[1]);
    }

    [[nodiscard]] friend constexpr bool operator==(const Matrix2&,
                                                   const Matrix2&) = default;
};

template <Numeric T>
[[nodiscard]] constexpr Matrix2<T> transpose(const Matrix2<T>& m) noexcept {
    return Matrix2<T>::fromRows(m.columns[0], m.columns[1]);
}

template <Numeric T>
[[nodiscard]] constexpr T determinant(const Matrix2<T>& m) noexcept {
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}

template <Numeric T>
[[nodiscard]] constexpr T trace(const Matrix2<T>& m) noexcept {
    return m(0, 0) + m(1, 1);
}

/// nullopt when the matrix is singular or holds a NaN. Near-singular is not
/// detected: the result is finite but inaccurate, which is what
/// detail::solveByElimination exists for.
template <Numeric T>
[[nodiscard]] constexpr std::optional<Matrix2<T>> tryInverse(
    const Matrix2<T>& m) noexcept {
    const T det = determinant(m);
    // Finiteness as well as non-zero. A determinant that overflowed used to
    // pass both of the old checks and then divide the adjugate down to a zero
    // matrix, reported as a success.
    if (det == T{0} || !detail::isFiniteValue(det)) {
        return std::nullopt;
    }
    return Matrix2<T>::fromRows({m(1, 1), -m(0, 1)}, {-m(1, 0), m(0, 0)}) / det;
}

/// Unchecked: a singular matrix yields infinities or NaN rather than an error.
/// Use tryInverse where the input can legitimately be singular.
template <Numeric T>
[[nodiscard]] constexpr Matrix2<T> inverse(const Matrix2<T>& m) noexcept {
    const T det = determinant(m);
    return Matrix2<T>::fromRows({m(1, 1), -m(0, 1)}, {-m(1, 0), m(0, 0)}) / det;
}

/// Solves M x = b by elimination with partial pivoting. nullopt if singular.
template <Numeric T>
[[nodiscard]] std::optional<Vector2<T>> solve(const Matrix2<T>& m,
                                              const Vector2<T>& b) {
    return detail::solveByElimination(m, b);
}

using Mat2 = Matrix2<double>;
using Mat2f = Matrix2<float>;

}  // namespace ysq
