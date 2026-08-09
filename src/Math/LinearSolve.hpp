#pragma once

#include <Math/Scalar.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <vector>

namespace ysq {

/// General dense linear algebra: systems bigger than `Matrix2`/`Matrix3`/
/// `Matrix4` were ever meant to hold.
///
/// Those three are fixed-size on purpose, sized to the geometry they carry
/// (a transform, a metric). `MatrixN`/`VectorN` here are the deliberately
/// dynamic counterpart, for a system whose size is only known at run time: a
/// normal-equations fit, a constraint system with one row per contact, a
/// discretized boundary-value problem. `solve` (general, via LU with partial
/// pivoting) and `choleskySolve` (symmetric positive-definite only, about
/// half the work) are the two entry points; everything else here is what
/// they are built from.

/// A dynamically sized column vector, `std::vector`-backed.
template <std::floating_point T>
class VectorN {
public:
    explicit VectorN(std::size_t size, T fill = T{0}) : m_values(size, fill) {}
    VectorN(std::initializer_list<T> values) : m_values(values) {}

    [[nodiscard]] std::size_t size() const noexcept { return m_values.size(); }

    [[nodiscard]] T& operator[](std::size_t i) {
        assert(i < size());
        return m_values[i];
    }
    [[nodiscard]] const T& operator[](std::size_t i) const {
        assert(i < size());
        return m_values[i];
    }

    VectorN& operator+=(const VectorN& other) {
        assert(size() == other.size());
        for (std::size_t i = 0; i < size(); ++i) {
            m_values[i] += other.m_values[i];
        }
        return *this;
    }
    VectorN& operator-=(const VectorN& other) {
        assert(size() == other.size());
        for (std::size_t i = 0; i < size(); ++i) {
            m_values[i] -= other.m_values[i];
        }
        return *this;
    }
    VectorN& operator*=(T scalar) {
        for (T& value : m_values) {
            value *= scalar;
        }
        return *this;
    }

    [[nodiscard]] friend VectorN operator+(VectorN a, const VectorN& b) {
        a += b;
        return a;
    }
    [[nodiscard]] friend VectorN operator-(VectorN a, const VectorN& b) {
        a -= b;
        return a;
    }
    [[nodiscard]] friend VectorN operator*(VectorN v, T scalar) {
        v *= scalar;
        return v;
    }
    [[nodiscard]] friend VectorN operator*(T scalar, VectorN v) {
        v *= scalar;
        return v;
    }

private:
    std::vector<T> m_values;
};

template <std::floating_point T>
[[nodiscard]] T dot(const VectorN<T>& a, const VectorN<T>& b) {
    assert(a.size() == b.size());
    T total{};
    for (std::size_t i = 0; i < a.size(); ++i) {
        total += a[i] * b[i];
    }
    return total;
}

template <std::floating_point T>
[[nodiscard]] T norm(const VectorN<T>& v) {
    using std::sqrt;
    return sqrt(dot(v, v));
}

/// A dynamically sized, row-major dense matrix.
template <std::floating_point T>
class MatrixN {
public:
    MatrixN(std::size_t rows, std::size_t cols, T fill = T{0})
        : m_rows(rows), m_cols(cols), m_data(rows * cols, fill) {}

    [[nodiscard]] std::size_t rows() const noexcept { return m_rows; }
    [[nodiscard]] std::size_t cols() const noexcept { return m_cols; }

    [[nodiscard]] T& operator()(std::size_t row, std::size_t col) {
        assert(row < m_rows && col < m_cols);
        return m_data[row * m_cols + col];
    }
    [[nodiscard]] const T& operator()(std::size_t row, std::size_t col) const {
        assert(row < m_rows && col < m_cols);
        return m_data[row * m_cols + col];
    }

    [[nodiscard]] static MatrixN identity(std::size_t n) {
        MatrixN result(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

private:
    std::size_t m_rows;
    std::size_t m_cols;
    std::vector<T> m_data;
};

template <std::floating_point T>
[[nodiscard]] MatrixN<T> transpose(const MatrixN<T>& m) {
    MatrixN<T> result(m.cols(), m.rows());
    for (std::size_t r = 0; r < m.rows(); ++r) {
        for (std::size_t c = 0; c < m.cols(); ++c) {
            result(c, r) = m(r, c);
        }
    }
    return result;
}

template <std::floating_point T>
[[nodiscard]] VectorN<T> operator*(const MatrixN<T>& m, const VectorN<T>& v) {
    assert(m.cols() == v.size());
    VectorN<T> result(m.rows());
    for (std::size_t r = 0; r < m.rows(); ++r) {
        T total{};
        for (std::size_t c = 0; c < m.cols(); ++c) {
            total += m(r, c) * v[c];
        }
        result[r] = total;
    }
    return result;
}

template <std::floating_point T>
[[nodiscard]] MatrixN<T> operator*(const MatrixN<T>& a, const MatrixN<T>& b) {
    assert(a.cols() == b.rows());
    MatrixN<T> result(a.rows(), b.cols());
    for (std::size_t r = 0; r < a.rows(); ++r) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const T aRK = a(r, k);
            for (std::size_t c = 0; c < b.cols(); ++c) {
                result(r, c) += aRK * b(k, c);
            }
        }
    }
    return result;
}

/// An LU decomposition with partial pivoting, in Doolittle form: `lu` packs
/// both factors into one matrix (L's implicit unit diagonal is never
/// stored), and `pivot[i]` is the original row now sitting in row `i` after
/// the swaps elimination made. Kept as its own type, rather than folded
/// straight into `solve`, so a caller solving the same system against many
/// right-hand sides pays the O(n^3) factorization once.
template <std::floating_point T>
struct LuDecomposition {
    MatrixN<T> lu;
    std::vector<std::size_t> pivot;
};

/// Gaussian elimination with partial pivoting, same conditioning rationale
/// as `Matrix2.hpp`'s `detail::solveByElimination`: without pivoting, a
/// small (or zero) diagonal entry either divides by nearly nothing or halts
/// outright, and the row with the largest remaining entry in this column is
/// never that fragile because every other row's elimination factor against
/// it is at most 1 in magnitude. `nullopt` for a singular or non-finite
/// matrix; near-singular is not detected, matching `Matrix4.hpp`'s
/// `tryInverse`.
template <std::floating_point T>
[[nodiscard]] std::optional<LuDecomposition<T>> luDecompose(MatrixN<T> a) {
    assert(a.rows() == a.cols());
    const std::size_t n = a.rows();
    std::vector<std::size_t> pivot(n);
    for (std::size_t i = 0; i < n; ++i) {
        pivot[i] = i;
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivotRow = k;
        T best = detail::absOf(a(k, k));
        for (std::size_t row = k + 1; row < n; ++row) {
            const T candidate = detail::absOf(a(row, k));
            if (best < candidate) {
                best = candidate;
                pivotRow = row;
            }
        }

        if (!(T{0} < best) || !detail::isFiniteValue(best)) {
            return std::nullopt;
        }

        if (pivotRow != k) {
            for (std::size_t col = 0; col < n; ++col) {
                const T swapped = a(k, col);
                a(k, col) = a(pivotRow, col);
                a(pivotRow, col) = swapped;
            }
            std::swap(pivot[k], pivot[pivotRow]);
        }

        for (std::size_t row = k + 1; row < n; ++row) {
            const T factor = a(row, k) / a(k, k);
            a(row, k) = factor;
            for (std::size_t col = k + 1; col < n; ++col) {
                a(row, col) -= factor * a(k, col);
            }
        }
    }

    return LuDecomposition<T>{a, pivot};
}

/// Forward substitution against `L` (unit diagonal, so no division there),
/// then back substitution against `U`, with `b` permuted by the pivot
/// `luDecompose` recorded.
template <std::floating_point T>
[[nodiscard]] VectorN<T> luSolve(const LuDecomposition<T>& decomposition,
                                 const VectorN<T>& b) {
    const std::size_t n = decomposition.lu.rows();
    VectorN<T> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        T total = b[decomposition.pivot[i]];
        for (std::size_t j = 0; j < i; ++j) {
            total -= decomposition.lu(i, j) * y[j];
        }
        y[i] = total;
    }

    VectorN<T> x(n);
    for (std::size_t i = n; i-- > 0;) {
        T total = y[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            total -= decomposition.lu(i, j) * x[j];
        }
        x[i] = total / decomposition.lu(i, i);
    }
    return x;
}

/// Solves `a x = b` for a general square matrix. `nullopt` if `a` is singular
/// (or the wrong shape). One-shot convenience over `luDecompose` + `luSolve`
/// for a caller with a single right-hand side.
template <std::floating_point T>
[[nodiscard]] std::optional<VectorN<T>> solve(const MatrixN<T>& a, const VectorN<T>& b) {
    if (a.rows() != a.cols() || a.rows() != b.size()) {
        return std::nullopt;
    }
    const std::optional<LuDecomposition<T>> decomposition = luDecompose(a);
    if (!decomposition) {
        return std::nullopt;
    }
    return luSolve(*decomposition, b);
}

/// The Cholesky factorization `a = L L^T` of a symmetric positive-definite
/// `a`: about half the arithmetic of LU, and no pivoting is needed because
/// positive-definiteness alone already keeps every pivot both positive and
/// the largest available in its column. `nullopt` the moment a diagonal
/// entry under the square root would be non-positive, which is exactly the
/// condition "not actually positive-definite" (a matrix that is symmetric
/// but not positive-definite has no real Cholesky factor at all, so this
/// doubles as the positive-definiteness check, not a separate one run
/// first). Only the lower triangle of `a` is read, so an asymmetric input is
/// silently treated as if its lower triangle described the whole matrix.
template <std::floating_point T>
[[nodiscard]] std::optional<MatrixN<T>> choleskyDecompose(const MatrixN<T>& a) {
    assert(a.rows() == a.cols());
    using std::sqrt;
    const std::size_t n = a.rows();
    MatrixN<T> l(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            T total = a(i, j);
            for (std::size_t k = 0; k < j; ++k) {
                total -= l(i, k) * l(j, k);
            }

            if (i == j) {
                if (!(T{0} < total) || !detail::isFiniteValue(total)) {
                    return std::nullopt;
                }
                l(i, j) = sqrt(total);
            } else {
                l(i, j) = total / l(j, j);
            }
        }
    }
    return l;
}

/// Solves `a x = b` for a symmetric positive-definite `a` via Cholesky.
/// `nullopt` if `a` is not positive-definite (or the wrong shape) — this is
/// the natural solver for a normal-equations system `A^T A x = A^T b` or any
/// other Gram-matrix-shaped problem, both guaranteed positive-definite
/// whenever `A` has full column rank.
template <std::floating_point T>
[[nodiscard]] std::optional<VectorN<T>> choleskySolve(const MatrixN<T>& a,
                                                      const VectorN<T>& b) {
    if (a.rows() != a.cols() || a.rows() != b.size()) {
        return std::nullopt;
    }
    const std::optional<MatrixN<T>> l = choleskyDecompose(a);
    if (!l) {
        return std::nullopt;
    }

    const std::size_t n = a.rows();
    VectorN<T> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        T total = b[i];
        for (std::size_t j = 0; j < i; ++j) {
            total -= (*l)(i, j) * y[j];
        }
        y[i] = total / (*l)(i, i);
    }

    VectorN<T> x(n);
    for (std::size_t i = n; i-- > 0;) {
        T total = y[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            total -= (*l)(j, i) * x[j];
        }
        x[i] = total / (*l)(i, i);
    }
    return x;
}

}  // namespace ysq
