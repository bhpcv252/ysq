#pragma once

#include <Math/Complex.hpp>
#include <Math/LinearSolve.hpp>
#include <Math/Scalar.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace ysq {

/// Eigenvalues and eigenvectors of a symmetric matrix (`jacobiEigenSymmetric`
/// below), plus the general (non-symmetric) case and two related dense
/// factorizations: `qrDecompose`, `realSchur`/`generalEigenvalues`, and
/// `svd`.
///
/// The symmetric case gets its own dedicated method rather than routing
/// through the general one: it is the one case where the eigenvalues are
/// always real and the eigenvectors always form an orthonormal basis,
/// which is what makes the cyclic Jacobi method both simple and
/// numerically robust, and every consumer this engine has so far
/// (an inertia tensor, a covariance matrix, a Gram matrix) is symmetric by
/// construction. The general case below exists for the caller building
/// something this engine's own consumers do not need yet -- a state-space
/// stability analysis, a general linear map's spectrum -- where the
/// eigenvalues are not guaranteed real and the eigenvectors are not
/// guaranteed orthogonal.

template <std::floating_point T>
struct EigenDecomposition {
    /// Ascending order, paired index-for-index with `eigenvectors`' columns.
    VectorN<T> eigenvalues;
    /// Column `i` is the unit eigenvector for `eigenvalues[i]`, and every
    /// column is orthogonal to every other: together they are an orthonormal
    /// basis that diagonalizes the original matrix.
    MatrixN<T> eigenvectors;
};

/// The cyclic Jacobi eigenvalue algorithm: repeatedly zero one off-diagonal
/// entry with a plane rotation chosen to do exactly that, cycling through
/// every entry above the diagonal in a fixed order rather than always
/// picking the largest remaining one (the "classical" variant). A rotation
/// that zeros `(p, q)` generally un-zeros entries the algorithm already
/// cleared, but each full sweep shrinks the total off-diagonal energy, and
/// enough sweeps converge to a diagonal matrix whose diagonal is the
/// eigenvalues, with the accumulated rotations as the eigenvectors.
///
/// The rotation angle at each step uses the numerically stable form from
/// Golub & Van Loan (the tangent of half the angle, computed without ever
/// dividing by a near-zero quantity when `a(p, q)` is already small) rather
/// than the textbook `theta = 0.5 atan2(2 a_pq, a_qq - a_pp)`, which loses
/// precision exactly when it matters least (a(p,q) already tiny) and exactly
/// when it matters most (a(p,p) == a(q,q)) alike.
///
/// Only the lower triangle of `a` is read, matching `choleskyDecompose`'s
/// convention: an asymmetric input is silently treated as if its lower
/// triangle described the whole (implicitly symmetric) matrix.
template <std::floating_point T>
[[nodiscard]] EigenDecomposition<T>
jacobiEigenSymmetric(MatrixN<T> a, int maxSweeps = 100, T tolerance = T{0}) {
    assert(a.rows() == a.cols());
    const std::size_t n = a.rows();

    // Fill in the upper triangle from the lower one so the rotation below,
    // which reads and writes both, sees a genuinely symmetric matrix.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            a(j, i) = a(i, j);
        }
    }

    MatrixN<T> v = MatrixN<T>::identity(n);

    const T effectiveTolerance =
        (tolerance > T{0}) ? tolerance : std::numeric_limits<T>::epsilon() * T{100};

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        T offDiagonalSquared{};
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                offDiagonalSquared += a(p, q) * a(p, q);
            }
        }
        if (offDiagonalSquared < effectiveTolerance * effectiveTolerance) {
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const T apq = a(p, q);
                if (detail::absOf(apq) <= std::numeric_limits<T>::epsilon()) {
                    continue;
                }

                const T app = a(p, p);
                const T aqq = a(q, q);
                const T theta = (aqq - app) / (T{2} * apq);
                const T sign = (theta < T{0}) ? T{-1} : T{1};
                const T t =
                    sign / (detail::absOf(theta) + detail::sqrtOf(theta * theta + T{1}));
                const T c = T{1} / detail::sqrtOf(t * t + T{1});
                const T s = t * c;

                a(p, p) = app - t * apq;
                a(q, q) = aqq + t * apq;
                a(p, q) = T{0};
                a(q, p) = T{0};

                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) {
                        continue;
                    }
                    const T aip = a(i, p);
                    const T aiq = a(i, q);
                    a(i, p) = c * aip - s * aiq;
                    a(p, i) = a(i, p);
                    a(i, q) = s * aip + c * aiq;
                    a(q, i) = a(i, q);
                }

                for (std::size_t i = 0; i < n; ++i) {
                    const T vip = v(i, p);
                    const T viq = v(i, q);
                    v(i, p) = c * vip - s * viq;
                    v(i, q) = s * vip + c * viq;
                }
            }
        }
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&a](std::size_t lhs, std::size_t rhs) {
        return a(lhs, lhs) < a(rhs, rhs);
    });

    VectorN<T> eigenvalues(n);
    MatrixN<T> eigenvectors(n, n);
    for (std::size_t col = 0; col < n; ++col) {
        eigenvalues[col] = a(order[col], order[col]);
        for (std::size_t row = 0; row < n; ++row) {
            eigenvectors(row, col) = v(row, order[col]);
        }
    }

    return EigenDecomposition<T>{eigenvalues, eigenvectors};
}

/// A QR decomposition `a = q * r`: `q` orthogonal (`m x m`), `r` upper
/// triangular in its top `n x n` block and zero below (`m x n`), for any
/// `a` with at least as many rows as columns. Computed via Householder
/// reflections: for each column in turn, a reflection is chosen that
/// zeros everything below the diagonal in that column without disturbing
/// the columns already zeroed, the same "reduce one column, preserve the
/// rest" structure `luDecompose` uses for elimination, but by an
/// orthogonal (and so unconditionally stable, no pivoting needed) rather
/// than a merely invertible transform.
template <std::floating_point T>
struct QrDecomposition {
    MatrixN<T> q;
    MatrixN<T> r;
};

template <std::floating_point T>
[[nodiscard]] QrDecomposition<T> qrDecompose(MatrixN<T> a) {
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    assert(m >= n);

    MatrixN<T> q = MatrixN<T>::identity(m);
    const std::size_t steps = std::min(m, n);

    for (std::size_t k = 0; k < steps; ++k) {
        T normX{};
        for (std::size_t i = k; i < m; ++i) {
            normX += a(i, k) * a(i, k);
        }
        normX = detail::sqrtOf(normX);
        if (normX == T{0}) {
            continue;
        }

        // Reflecting toward -sign(a(k,k)) * normX, rather than +normX,
        // avoids subtracting two near-equal numbers when a(k,k) is already
        // close to normX -- the same cancellation `quadraticRealRoots`
        // avoids with its own sign choice.
        const T alpha = (a(k, k) < T{0}) ? normX : -normX;
        std::vector<T> v(m - k);
        for (std::size_t i = k; i < m; ++i) {
            v[i - k] = a(i, k);
        }
        v[0] -= alpha;

        T vNormSquared{};
        for (const T& value : v) {
            vNormSquared += value * value;
        }
        if (vNormSquared == T{0}) {
            continue;
        }

        // Apply the Householder reflection H = I - 2 v v^T / (v^T v) to a
        // from the left (rows k..m-1 of every remaining column) ...
        for (std::size_t col = k; col < n; ++col) {
            T dotProduct{};
            for (std::size_t i = k; i < m; ++i) {
                dotProduct += v[i - k] * a(i, col);
            }
            const T factor = T{2} * dotProduct / vNormSquared;
            for (std::size_t i = k; i < m; ++i) {
                a(i, col) -= factor * v[i - k];
            }
        }

        // ... and accumulate it into q from the right (q <- q H), so q
        // ends up as the product of every reflection applied so far.
        for (std::size_t row = 0; row < m; ++row) {
            T dotProduct{};
            for (std::size_t i = k; i < m; ++i) {
                dotProduct += q(row, i) * v[i - k];
            }
            const T factor = T{2} * dotProduct / vNormSquared;
            for (std::size_t i = k; i < m; ++i) {
                q(row, i) -= factor * v[i - k];
            }
        }
    }

    return QrDecomposition<T>{q, a};
}

namespace detail {

/// Reduces `a` to upper Hessenberg form (zero below the first subdiagonal)
/// by similarity, `a <- q^T a q`, via the same Householder-reflection idea
/// as `qrDecompose` but applied on both sides at once (a similarity
/// transform, so the eigenvalues are unchanged) rather than only the left.
/// This is the standard first stage of a general eigenvalue solve: the
/// shifted QR algorithm below converges on a Hessenberg matrix in O(n^2)
/// per iteration instead of O(n^3) on a dense one, and reducing to
/// Hessenberg form first is itself a one-time O(n^3) cost.
template <std::floating_point T>
void hessenbergReduce(MatrixN<T>& a, MatrixN<T>& q) {
    const std::size_t n = a.rows();
    for (std::size_t k = 0; k + 2 < n; ++k) {
        T normX{};
        for (std::size_t i = k + 1; i < n; ++i) {
            normX += a(i, k) * a(i, k);
        }
        normX = sqrtOf(normX);
        if (normX == T{0}) {
            continue;
        }

        const T alpha = (a(k + 1, k) < T{0}) ? normX : -normX;
        std::vector<T> v(n - k - 1);
        for (std::size_t i = k + 1; i < n; ++i) {
            v[i - k - 1] = a(i, k);
        }
        v[0] -= alpha;

        T vNormSquared{};
        for (const T& value : v) {
            vNormSquared += value * value;
        }
        if (vNormSquared == T{0}) {
            continue;
        }

        for (std::size_t col = 0; col < n; ++col) {
            T dotProduct{};
            for (std::size_t i = k + 1; i < n; ++i) {
                dotProduct += v[i - k - 1] * a(i, col);
            }
            const T factor = T{2} * dotProduct / vNormSquared;
            for (std::size_t i = k + 1; i < n; ++i) {
                a(i, col) -= factor * v[i - k - 1];
            }
        }
        for (std::size_t row = 0; row < n; ++row) {
            T dotProduct{};
            for (std::size_t i = k + 1; i < n; ++i) {
                dotProduct += a(row, i) * v[i - k - 1];
            }
            const T factor = T{2} * dotProduct / vNormSquared;
            for (std::size_t i = k + 1; i < n; ++i) {
                a(row, i) -= factor * v[i - k - 1];
            }
        }
        for (std::size_t row = 0; row < n; ++row) {
            T dotProduct{};
            for (std::size_t i = k + 1; i < n; ++i) {
                dotProduct += q(row, i) * v[i - k - 1];
            }
            const T factor = T{2} * dotProduct / vNormSquared;
            for (std::size_t i = k + 1; i < n; ++i) {
                q(row, i) -= factor * v[i - k - 1];
            }
        }
    }
}

}  // namespace detail

/// The real Schur decomposition `a = q * t * q^T`: `q` orthogonal, `t`
/// quasi-upper-triangular (upper triangular except possibly for isolated
/// 2x2 blocks straddling the diagonal). A real matrix cannot always be
/// brought to fully triangular form by a real orthogonal transform -- a
/// complex-conjugate pair of eigenvalues has no real eigenvector to
/// triangularize around -- so a 2x2 block is where that pair's own
/// two-dimensional invariant subspace lives instead; `eigenvaluesFromSchur`
/// reads real eigenvalues directly off a 1x1 block's own diagonal entry
/// and complex-conjugate pairs off a 2x2 block via the quadratic formula.
template <std::floating_point T>
struct SchurDecomposition {
    MatrixN<T> q;
    MatrixN<T> t;
};

/// Hessenberg reduction followed by the shifted QR algorithm with
/// deflation (single-shift, Wilkinson's choice of shift): each iteration
/// factors the still-active leading block as `(a - shift*I) = qr` and
/// recombines it as `r*q + shift*I`, a similarity transform that drives
/// subdiagonal entries toward zero from the bottom right upward. Once a
/// subdiagonal entry is negligible relative to its neighboring diagonal
/// entries, it is deflated to exactly zero and the active block shrinks;
/// a trailing block that shrinks to size 2 without deflating further is
/// left as-is rather than iterated on, since a real 2x2 with genuinely
/// complex eigenvalues can never converge to a smaller block by further
/// real orthogonal similarity -- `eigenvaluesFromSchur` extracts its
/// eigenvalues directly instead.
template <std::floating_point T>
[[nodiscard]] SchurDecomposition<T> realSchur(MatrixN<T> a, int maxIterations = 500) {
    assert(a.rows() == a.cols());
    const std::size_t n = a.rows();
    MatrixN<T> q = MatrixN<T>::identity(n);
    detail::hessenbergReduce(a, q);

    const T tolerance = std::numeric_limits<T>::epsilon() * T{100};
    std::size_t p = n;
    int iterations = 0;

    while (p > 0 && iterations < maxIterations) {
        if (p == 1) {
            break;
        }

        const T sub = detail::absOf(a(p - 1, p - 2));
        const T scale = detail::absOf(a(p - 2, p - 2)) + detail::absOf(a(p - 1, p - 1));
        if (sub <= tolerance * (scale == T{0} ? T{1} : scale)) {
            a(p - 1, p - 2) = T{0};
            p -= 1;
            continue;
        }

        if (p == 2) {
            break;
        }

        const T sub2 = detail::absOf(a(p - 2, p - 3));
        const T scale2 = detail::absOf(a(p - 3, p - 3)) + detail::absOf(a(p - 2, p - 2));
        if (sub2 <= tolerance * (scale2 == T{0} ? T{1} : scale2)) {
            a(p - 2, p - 3) = T{0};
            p -= 2;
            continue;
        }

        const T trace = a(p - 2, p - 2) + a(p - 1, p - 1);
        const T det =
            a(p - 2, p - 2) * a(p - 1, p - 1) - a(p - 2, p - 1) * a(p - 1, p - 2);
        const T discriminant = trace * trace - T{4} * det;
        T shift{};
        if (discriminant >= T{0}) {
            const T sqrtDiscriminant = detail::sqrtOf(discriminant);
            const T lambda1 = (trace + sqrtDiscriminant) / T{2};
            const T lambda2 = (trace - sqrtDiscriminant) / T{2};
            shift = (detail::absOf(lambda1 - a(p - 1, p - 1)) <
                     detail::absOf(lambda2 - a(p - 1, p - 1)))
                        ? lambda1
                        : lambda2;
        } else {
            shift = a(p - 1, p - 1);
        }

        MatrixN<T> block(p, p);
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                block(i, j) = a(i, j);
            }
            block(i, i) -= shift;
        }
        const QrDecomposition<T> qr = qrDecompose(block);

        MatrixN<T> qStep = MatrixN<T>::identity(n);
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                qStep(i, j) = qr.q(i, j);
            }
        }

        a = transpose(qStep) * a * qStep;
        q = q * qStep;
        ++iterations;
    }

    return SchurDecomposition<T>{q, a};
}

/// Reads eigenvalues off a real Schur form's quasi-triangular diagonal: a
/// 1x1 block is a real eigenvalue directly, a 2x2 block is solved via the
/// quadratic formula on its own trace and determinant (real roots if its
/// own discriminant is non-negative, a complex-conjugate pair otherwise).
template <std::floating_point T>
[[nodiscard]] std::vector<Complex<T>> eigenvaluesFromSchur(const MatrixN<T>& t) {
    const std::size_t n = t.rows();
    const T tolerance = std::numeric_limits<T>::epsilon() * T{100};
    std::vector<Complex<T>> result;

    std::size_t i = 0;
    while (i < n) {
        const bool isLast = (i + 1 == n);
        const T sub = isLast ? T{0} : detail::absOf(t(i + 1, i));
        const T scale =
            isLast ? T{0} : detail::absOf(t(i, i)) + detail::absOf(t(i + 1, i + 1));

        if (isLast || sub <= tolerance * (scale == T{0} ? T{1} : scale)) {
            result.push_back(Complex<T>{t(i, i), T{0}});
            i += 1;
            continue;
        }

        const T trace = t(i, i) + t(i + 1, i + 1);
        const T det = t(i, i) * t(i + 1, i + 1) - t(i, i + 1) * t(i + 1, i);
        const T discriminant = trace * trace - T{4} * det;
        if (discriminant >= T{0}) {
            const T sqrtDiscriminant = detail::sqrtOf(discriminant);
            result.push_back(Complex<T>{(trace + sqrtDiscriminant) / T{2}, T{0}});
            result.push_back(Complex<T>{(trace - sqrtDiscriminant) / T{2}, T{0}});
        } else {
            const T sqrtNegDiscriminant = detail::sqrtOf(-discriminant);
            result.push_back(Complex<T>{trace / T{2}, sqrtNegDiscriminant / T{2}});
            result.push_back(Complex<T>{trace / T{2}, -sqrtNegDiscriminant / T{2}});
        }
        i += 2;
    }

    return result;
}

/// The eigenvalues of a general (not-necessarily-symmetric) square matrix,
/// real or complex-conjugate-pair as the matrix itself determines.
/// Eigenvectors are deliberately not returned here: a numerically solid
/// eigenvector for a complex eigenvalue needs a complex linear solve
/// (inverse iteration against a complex-shifted system), which would need
/// a complex counterpart to `Math/LinearSolve.hpp` that does not exist yet
/// and is new scope beyond eigenvalues themselves.
template <std::floating_point T>
[[nodiscard]] std::vector<Complex<T>> generalEigenvalues(const MatrixN<T>& a,
                                                         int maxIterations = 500) {
    return eigenvaluesFromSchur(realSchur(a, maxIterations).t);
}

/// A singular value decomposition `a = u * diag(singularValues) * v^T`:
/// `u` (`m x n`, orthonormal columns), `singularValues` (descending), `v`
/// (`n x n`, orthogonal). Requires `a.rows() >= a.cols()`; a caller with
/// fewer rows than columns transposes `a` first and swaps `u` and `v` back
/// afterward (`svd(transpose(a))` gives `a^T = u' diag(s) v'^T`, so
/// `a = v' diag(s) u'^T`).
///
/// Computed by one-sided Jacobi (Hestenes 1958): repeatedly rotates pairs
/// of columns of a working copy of `a` toward orthogonality, the same
/// plane-rotation idea `jacobiEigenSymmetric` already uses to zero
/// off-diagonal entries, aimed at a different target (two columns' inner
/// product, not a symmetric matrix's off-diagonal entry). At convergence
/// the working copy's columns are exactly `u` scaled by the singular
/// values, so the singular values fall out as the converged column norms
/// and `u` as those columns renormalized; `v` accumulates the same
/// rotations column-for-column, exactly mirroring how
/// `jacobiEigenSymmetric` accumulates its own rotations into eigenvectors.
template <std::floating_point T>
struct SvdDecomposition {
    MatrixN<T> u;
    VectorN<T> singularValues;
    MatrixN<T> v;
};

template <std::floating_point T>
[[nodiscard]] SvdDecomposition<T> svd(MatrixN<T> a, int maxSweeps = 60) {
    assert(a.rows() >= a.cols());
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    MatrixN<T> v = MatrixN<T>::identity(n);

    const T tolerance = std::numeric_limits<T>::epsilon() * T{100};

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        T offDiagonalSquared{};
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                T alpha{};
                T beta{};
                T gamma{};
                for (std::size_t i = 0; i < m; ++i) {
                    alpha += a(i, p) * a(i, p);
                    beta += a(i, q) * a(i, q);
                    gamma += a(i, p) * a(i, q);
                }
                offDiagonalSquared += gamma * gamma;

                const T threshold = tolerance * detail::sqrtOf(alpha * beta);
                if (detail::absOf(gamma) <= threshold) {
                    continue;
                }

                const T zeta = (beta - alpha) / (T{2} * gamma);
                const T sign = (zeta < T{0}) ? T{-1} : T{1};
                const T t =
                    sign / (detail::absOf(zeta) + detail::sqrtOf(T{1} + zeta * zeta));
                const T c = T{1} / detail::sqrtOf(T{1} + t * t);
                const T s = c * t;

                for (std::size_t i = 0; i < m; ++i) {
                    const T ap = a(i, p);
                    const T aq = a(i, q);
                    a(i, p) = c * ap - s * aq;
                    a(i, q) = s * ap + c * aq;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const T vp = v(i, p);
                    const T vq = v(i, q);
                    v(i, p) = c * vp - s * vq;
                    v(i, q) = s * vp + c * vq;
                }
            }
        }
        if (offDiagonalSquared < tolerance * tolerance) {
            break;
        }
    }

    VectorN<T> singularValues(n);
    for (std::size_t col = 0; col < n; ++col) {
        T normSquared{};
        for (std::size_t i = 0; i < m; ++i) {
            normSquared += a(i, col) * a(i, col);
        }
        singularValues[col] = detail::sqrtOf(normSquared);
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&singularValues](std::size_t lhs, std::size_t rhs) {
                  return singularValues[lhs] > singularValues[rhs];
              });

    MatrixN<T> u(m, n);
    MatrixN<T> vSorted(n, n);
    VectorN<T> sortedValues(n);
    for (std::size_t col = 0; col < n; ++col) {
        const std::size_t source = order[col];
        sortedValues[col] = singularValues[source];
        const T sigma = singularValues[source];
        for (std::size_t row = 0; row < m; ++row) {
            u(row, col) = (sigma > tolerance) ? a(row, source) / sigma : T{0};
        }
        for (std::size_t row = 0; row < n; ++row) {
            vSorted(row, col) = v(row, source);
        }
    }

    return SvdDecomposition<T>{u, sortedValues, vSorted};
}

}  // namespace ysq
