#pragma once

#include <Math/Matrix3.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>

namespace ysq {

/// Four by four matrix: the render transform, and the rank-2 objects of
/// spacetime once a metric is contracted down to components.
///
/// Column-major storage, column-vector convention, exactly as Matrix2. See
/// that header for the reasoning.
///
/// The projection factories target OpenGL's clip space, with z in [-1, 1] and
/// a right-handed eye space looking down -Z. Vulkan's [0, 1] depth range is a
/// different matrix; if a Vulkan path ever needs one it gets its own named
/// factory rather than a flag on this one.
template <Numeric T>
struct Matrix4 {
    using value_type = T;
    using Column = Vector4<T>;

    std::array<Column, 4> columns{};

    [[nodiscard]] static constexpr std::size_t rows() noexcept { return 4; }
    [[nodiscard]] static constexpr std::size_t cols() noexcept { return 4; }

    [[nodiscard]] constexpr Column& operator[](std::size_t col) noexcept {
        assert(col < cols());
        return columns[col];
    }

    [[nodiscard]] constexpr const Column& operator[](std::size_t col) const noexcept {
        assert(col < cols());
        return columns[col];
    }

    [[nodiscard]] constexpr T& operator()(std::size_t rowIndex,
                                          std::size_t col) noexcept {
        assert(rowIndex < rows() && col < cols());
        return columns[col][rowIndex];
    }

    [[nodiscard]] constexpr const T& operator()(std::size_t rowIndex,
                                                std::size_t col) const noexcept {
        assert(rowIndex < rows() && col < cols());
        return columns[col][rowIndex];
    }

    [[nodiscard]] constexpr Column row(std::size_t index) const noexcept {
        return {columns[0][index], columns[1][index], columns[2][index],
                columns[3][index]};
    }

    /// The linear part of an affine transform: rotation, scale and shear
    /// without the translation.
    [[nodiscard]] constexpr Matrix3<T> upperLeft3x3() const noexcept {
        return Matrix3<T>::fromColumns(columns[0].xyz(), columns[1].xyz(),
                                       columns[2].xyz());
    }

    [[nodiscard]] constexpr Vector3<T> translationPart() const noexcept {
        return columns[3].xyz();
    }

    [[nodiscard]] static constexpr Matrix4 zero() noexcept { return {}; }

    [[nodiscard]] static constexpr Matrix4 fromColumns(const Column& c0, const Column& c1,
                                                       const Column& c2,
                                                       const Column& c3) noexcept {
        return Matrix4{std::array<Column, 4>{{c0, c1, c2, c3}}};
    }

    /// Takes its arguments the way a matrix is written on paper.
    [[nodiscard]] static constexpr Matrix4 fromRows(const Column& r0, const Column& r1,
                                                    const Column& r2,
                                                    const Column& r3) noexcept {
        return fromColumns({r0.x, r1.x, r2.x, r3.x}, {r0.y, r1.y, r2.y, r3.y},
                           {r0.z, r1.z, r2.z, r3.z}, {r0.w, r1.w, r2.w, r3.w});
    }

    [[nodiscard]] static constexpr Matrix4 identity() noexcept {
        return fromColumns({T{1}, T{0}, T{0}, T{0}}, {T{0}, T{1}, T{0}, T{0}},
                           {T{0}, T{0}, T{1}, T{0}}, {T{0}, T{0}, T{0}, T{1}});
    }

    [[nodiscard]] static constexpr Matrix4 diagonal(const Column& d) noexcept {
        return fromColumns({d.x, T{0}, T{0}, T{0}}, {T{0}, d.y, T{0}, T{0}},
                           {T{0}, T{0}, d.z, T{0}}, {T{0}, T{0}, T{0}, d.w});
    }

    [[nodiscard]] static constexpr Matrix4
    translation(const Vector3<T>& offset) noexcept {
        Matrix4 result = identity();
        result.columns[3] = Column::point(offset);
        return result;
    }

    [[nodiscard]] static constexpr Matrix4 scale(const Vector3<T>& s) noexcept {
        return diagonal({s.x, s.y, s.z, T{1}});
    }

    /// Embeds a 3x3 linear transform, leaving the translation at zero.
    [[nodiscard]] static constexpr Matrix4 fromLinear(const Matrix3<T>& linear) noexcept {
        return fromColumns(
            Column::direction(linear.columns[0]), Column::direction(linear.columns[1]),
            Column::direction(linear.columns[2]), {T{0}, T{0}, T{0}, T{1}});
    }

    [[nodiscard]] static constexpr Matrix4
    fromLinearTranslation(const Matrix3<T>& linear, const Vector3<T>& offset) noexcept {
        return fromColumns(Column::direction(linear.columns[0]),
                           Column::direction(linear.columns[1]),
                           Column::direction(linear.columns[2]), Column::point(offset));
    }

    [[nodiscard]] static Matrix4 rotationX(T angle) {
        return fromLinear(Matrix3<T>::rotationX(angle));
    }

    [[nodiscard]] static Matrix4 rotationY(T angle) {
        return fromLinear(Matrix3<T>::rotationY(angle));
    }

    [[nodiscard]] static Matrix4 rotationZ(T angle) {
        return fromLinear(Matrix3<T>::rotationZ(angle));
    }

    /// Rodrigues' rotation about a unit axis, right-handed.
    [[nodiscard]] static Matrix4 rotation(const Vector3<T>& axis, T angle) {
        return fromLinear(Matrix3<T>::rotation(axis, angle));
    }

    /// Right-handed view matrix: maps `eye` to the origin and the direction
    /// toward `center` onto -Z, which is where OpenGL's eye space looks.
    [[nodiscard]] static Matrix4 lookAt(const Vector3<T>& eye, const Vector3<T>& center,
                                        const Vector3<T>& up) {
        const Vector3<T> forward = normalized(center - eye);
        const Vector3<T> side = normalized(cross(forward, up));
        const Vector3<T> trueUp = cross(side, forward);
        return fromRows({side.x, side.y, side.z, -dot(side, eye)},
                        {trueUp.x, trueUp.y, trueUp.z, -dot(trueUp, eye)},
                        {-forward.x, -forward.y, -forward.z, dot(forward, eye)},
                        {T{0}, T{0}, T{0}, T{1}});
    }

    /// Symmetric perspective projection. fovY is the full vertical field of
    /// view in radians; both planes are positive distances in front of the eye.
    ///
    /// The plane arguments are nearPlane and farPlane rather than near and far
    /// because the Windows headers define those two as macros.
    [[nodiscard]] static Matrix4 perspective(T fovY, T aspect, T nearPlane, T farPlane) {
        using std::tan;
        const T focal = T{1} / tan(fovY / T{2});
        const T range = nearPlane - farPlane;
        return fromRows({focal / aspect, T{0}, T{0}, T{0}}, {T{0}, focal, T{0}, T{0}},
                        {T{0}, T{0}, (farPlane + nearPlane) / range,
                         (T{2} * farPlane * nearPlane) / range},
                        {T{0}, T{0}, T{-1}, T{0}});
    }

    [[nodiscard]] static constexpr Matrix4
    orthographic(T left, T right, T bottom, T top, T nearPlane, T farPlane) noexcept {
        const T width = right - left;
        const T height = top - bottom;
        const T depth = farPlane - nearPlane;
        return fromRows({T{2} / width, T{0}, T{0}, -(right + left) / width},
                        {T{0}, T{2} / height, T{0}, -(top + bottom) / height},
                        {T{0}, T{0}, T{-2} / depth, -(farPlane + nearPlane) / depth},
                        {T{0}, T{0}, T{0}, T{1}});
    }

    constexpr Matrix4& operator+=(const Matrix4& other) noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            columns[i] += other.columns[i];
        }
        return *this;
    }

    constexpr Matrix4& operator-=(const Matrix4& other) noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            columns[i] -= other.columns[i];
        }
        return *this;
    }

    constexpr Matrix4& operator*=(T scalar) noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            columns[i] *= scalar;
        }
        return *this;
    }

    constexpr Matrix4& operator/=(T scalar) noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            columns[i] /= scalar;
        }
        return *this;
    }

    constexpr Matrix4& operator*=(const Matrix4& other) noexcept {
        *this = *this * other;
        return *this;
    }

    [[nodiscard]] friend constexpr Matrix4 operator+(const Matrix4& m) noexcept {
        return m;
    }

    [[nodiscard]] friend constexpr Matrix4 operator-(const Matrix4& m) noexcept {
        return fromColumns(-m.columns[0], -m.columns[1], -m.columns[2], -m.columns[3]);
    }

    [[nodiscard]] friend constexpr Matrix4 operator+(const Matrix4& a,
                                                     const Matrix4& b) noexcept {
        return fromColumns(a.columns[0] + b.columns[0], a.columns[1] + b.columns[1],
                           a.columns[2] + b.columns[2], a.columns[3] + b.columns[3]);
    }

    [[nodiscard]] friend constexpr Matrix4 operator-(const Matrix4& a,
                                                     const Matrix4& b) noexcept {
        return fromColumns(a.columns[0] - b.columns[0], a.columns[1] - b.columns[1],
                           a.columns[2] - b.columns[2], a.columns[3] - b.columns[3]);
    }

    [[nodiscard]] friend constexpr Matrix4 operator*(const Matrix4& m,
                                                     T scalar) noexcept {
        return fromColumns(m.columns[0] * scalar, m.columns[1] * scalar,
                           m.columns[2] * scalar, m.columns[3] * scalar);
    }

    [[nodiscard]] friend constexpr Matrix4 operator*(T scalar,
                                                     const Matrix4& m) noexcept {
        return m * scalar;
    }

    [[nodiscard]] friend constexpr Matrix4 operator/(const Matrix4& m,
                                                     T scalar) noexcept {
        return fromColumns(m.columns[0] / scalar, m.columns[1] / scalar,
                           m.columns[2] / scalar, m.columns[3] / scalar);
    }

    [[nodiscard]] friend constexpr Column operator*(const Matrix4& m,
                                                    const Column& v) noexcept {
        return m.columns[0] * v.x + m.columns[1] * v.y + m.columns[2] * v.z +
               m.columns[3] * v.w;
    }

    [[nodiscard]] friend constexpr Matrix4 operator*(const Matrix4& a,
                                                     const Matrix4& b) noexcept {
        return fromColumns(a * b.columns[0], a * b.columns[1], a * b.columns[2],
                           a * b.columns[3]);
    }

    [[nodiscard]] friend constexpr bool operator==(const Matrix4&,
                                                   const Matrix4&) = default;
};

template <Numeric T>
[[nodiscard]] constexpr Matrix4<T> transpose(const Matrix4<T>& m) noexcept {
    return Matrix4<T>::fromRows(m.columns[0], m.columns[1], m.columns[2], m.columns[3]);
}

template <Numeric T>
[[nodiscard]] constexpr T trace(const Matrix4<T>& m) noexcept {
    return m(0, 0) + m(1, 1) + m(2, 2) + m(3, 3);
}

namespace detail {

/// The 3x3 left after deleting one row and one column.
template <Numeric T>
[[nodiscard]] constexpr Matrix3<T> minorOf(const Matrix4<T>& m, std::size_t skipRow,
                                           std::size_t skipCol) noexcept {
    Matrix3<T> result{};
    std::size_t outRow = 0;
    for (std::size_t r = 0; r < 4; ++r) {
        if (r == skipRow) {
            continue;
        }
        std::size_t outCol = 0;
        for (std::size_t c = 0; c < 4; ++c) {
            if (c == skipCol) {
                continue;
            }
            result(outRow, outCol) = m(r, c);
            ++outCol;
        }
        ++outRow;
    }
    return result;
}

template <Numeric T>
[[nodiscard]] constexpr T cofactorOf(const Matrix4<T>& m, std::size_t atRow,
                                     std::size_t atCol) noexcept {
    const T minorDet = determinant(minorOf(m, atRow, atCol));
    return ((atRow + atCol) % 2 == 0) ? minorDet : -minorDet;
}

}  // namespace detail

/// Laplace expansion along the first row, with each 3x3 minor evaluated by
/// Matrix3's determinant.
///
/// This is the readable formulation, not the fastest one: sixteen 3x3
/// determinants where a shared-subexpression version needs about half the
/// multiplies. A 4x4 inverse is not an inner loop here, and the thing this
/// eventually inverts is a metric, where being able to read the code and see
/// that it is the adjugate is worth more than the multiplies.
template <Numeric T>
[[nodiscard]] constexpr T determinant(const Matrix4<T>& m) noexcept {
    return m(0, 0) * detail::cofactorOf(m, 0, 0) + m(0, 1) * detail::cofactorOf(m, 0, 1) +
           m(0, 2) * detail::cofactorOf(m, 0, 2) + m(0, 3) * detail::cofactorOf(m, 0, 3);
}

/// The transpose of the cofactor matrix. adjugate(m) * m == determinant(m) * I
/// for every matrix, singular ones included.
template <Numeric T>
[[nodiscard]] constexpr Matrix4<T> adjugate(const Matrix4<T>& m) noexcept {
    Matrix4<T> result{};
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            result(r, c) = detail::cofactorOf(m, c, r);
        }
    }
    return result;
}

/// nullopt when the matrix is singular or holds a NaN. Near-singular is not
/// detected: the result is finite but inaccurate. For a matrix that came from
/// measurement or from a metric rather than from a transform you built, prefer
/// solve(), which pivots.
template <Numeric T>
[[nodiscard]] constexpr std::optional<Matrix4<T>>
tryInverse(const Matrix4<T>& m) noexcept {
    const T det = determinant(m);
    // Finiteness as well as non-zero: an overflowed determinant is nonzero
    // but not finite, and dividing the adjugate by it would silently produce
    // a zero matrix that reads as a successful result rather than an error.
    if (det == T{0} || !detail::isFiniteValue(det)) {
        return std::nullopt;
    }
    return adjugate(m) / det;
}

/// Unchecked: a singular matrix yields infinities or NaN rather than an error.
template <Numeric T>
[[nodiscard]] constexpr Matrix4<T> inverse(const Matrix4<T>& m) noexcept {
    return adjugate(m) / determinant(m);
}

/// Inverse of an affine transform, whose last row is (0, 0, 0, 1). Inverts the
/// 3x3 linear part and rotates the translation back through it, which is a
/// good deal cheaper than the general adjugate.
///
/// Wrong, silently, for a matrix with a projective last row. The general
/// inverse is the one to use on a projection matrix.
template <Numeric T>
[[nodiscard]] constexpr Matrix4<T> inverseAffine(const Matrix4<T>& m) noexcept {
    const Matrix3<T> linear = inverse(m.upperLeft3x3());
    return Matrix4<T>::fromLinearTranslation(linear, -(linear * m.translationPart()));
}

/// Solves M x = b by elimination with partial pivoting. nullopt if singular.
template <Numeric T>
[[nodiscard]] std::optional<Vector4<T>> solve(const Matrix4<T>& m, const Vector4<T>& b) {
    return detail::solveByElimination(m, b);
}

/// Applies an affine transform to a position: translation included, no
/// perspective divide. Use projectPoint if the matrix is projective.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> transformPoint(const Matrix4<T>& m,
                                                  const Vector3<T>& v) noexcept {
    return (m * Vector4<T>::point(v)).xyz();
}

/// Applies a transform to a direction: translation ignored.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> transformDirection(const Matrix4<T>& m,
                                                      const Vector3<T>& v) noexcept {
    return (m * Vector4<T>::direction(v)).xyz();
}

/// Applies a transform to a position and divides through by w. This is what a
/// projection matrix needs.
template <Numeric T>
[[nodiscard]] constexpr Vector3<T> projectPoint(const Matrix4<T>& m,
                                                const Vector3<T>& v) noexcept {
    return perspectiveDivide(m * Vector4<T>::point(v));
}

using Mat4 = Matrix4<double>;
using Mat4f = Matrix4<float>;

}  // namespace ysq
