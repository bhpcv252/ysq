#pragma once

#include <Math/Matrix3.hpp>
#include <Math/Matrix4.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <array>
#include <cassert>
#include <cstddef>

namespace ysq {

namespace detail {

constexpr std::size_t integerPow(std::size_t base, std::size_t exponent) noexcept {
    std::size_t result = 1;
    for (std::size_t i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

/// Stride of the index at `position`, for row-major storage: the last index
/// varies fastest.
constexpr std::size_t tensorStride(std::size_t rank, std::size_t dim,
                                   std::size_t position) noexcept {
    return integerPow(dim, rank - 1 - position);
}

constexpr std::size_t tensorIndexAt(std::size_t flat, std::size_t rank,
                                    std::size_t dim,
                                    std::size_t position) noexcept {
    return (flat / tensorStride(rank, dim, position)) % dim;
}

/// The same flat index with the index at `position` deleted, renumbered for a
/// tensor of one lower rank.
constexpr std::size_t tensorWithoutIndex(std::size_t flat, std::size_t rank,
                                         std::size_t dim,
                                         std::size_t position) noexcept {
    const std::size_t stride = tensorStride(rank, dim, position);
    return (flat / (stride * dim)) * stride + (flat % stride);
}

constexpr std::size_t tensorSwapIndices(std::size_t flat, std::size_t rank,
                                        std::size_t dim, std::size_t first,
                                        std::size_t second) noexcept {
    const std::size_t strideFirst = tensorStride(rank, dim, first);
    const std::size_t strideSecond = tensorStride(rank, dim, second);
    const std::size_t indexFirst = (flat / strideFirst) % dim;
    const std::size_t indexSecond = (flat / strideSecond) % dim;
    return flat - indexFirst * strideFirst - indexSecond * strideSecond +
           indexSecond * strideFirst + indexFirst * strideSecond;
}

}  // namespace detail

/// A dense tensor of fixed rank, with every index running over the same
/// dimension.
///
/// That last restriction is deliberate rather than lazy. In relativity every
/// index is a spacetime index, so Dim is 4 throughout: the metric is
/// Tensor<T, 2, 4>, the Christoffel symbols Tensor<T, 3, 4>, the Riemann
/// tensor Tensor<T, 4, 4>. Fixing rank and dimension at compile time makes
/// every one of those a stack object with no allocation in a geodesic inner
/// loop, and makes a contraction over the wrong index a compile error rather
/// than a wrong number.
///
/// Storage is row-major, so the last index varies fastest. operator() takes
/// exactly Rank indices; operator[] reaches the flat storage directly and is
/// there for whole-tensor loops, not for indexing.
///
/// Raising and lowering indices needs a metric, so it lives in
/// Physics/Spacetime rather than here. What is here is the index algebra that
/// does not care what the numbers mean.
template <Numeric T, std::size_t Rank, std::size_t Dim>
struct Tensor {
    static_assert(Dim > 0, "a tensor needs at least one dimension");

    using value_type = T;

    static constexpr std::size_t kComponents = detail::integerPow(Dim, Rank);

    std::array<T, kComponents> components{};

    [[nodiscard]] static constexpr std::size_t rank() noexcept { return Rank; }
    [[nodiscard]] static constexpr std::size_t dimension() noexcept { return Dim; }
    [[nodiscard]] static constexpr std::size_t size() noexcept {
        return kComponents;
    }

    template <class... Indices>
    [[nodiscard]] static constexpr std::size_t flatten(Indices... indices) noexcept {
        // Rank is checked at compile time by operator(); the values are not,
        // and an out-of-range one indexes past the storage rather than
        // wrapping. Debug builds say so at the point of the mistake.
        assert(((static_cast<std::size_t>(indices) < Dim) && ...));

        std::size_t flat = 0;
        ((flat = flat * Dim + static_cast<std::size_t>(indices)), ...);
        return flat;
    }

    template <class... Indices>
    [[nodiscard]] constexpr T& operator()(Indices... indices) noexcept {
        static_assert(sizeof...(Indices) == Rank,
                      "a tensor takes exactly Rank indices");
        return components[flatten(indices...)];
    }

    template <class... Indices>
    [[nodiscard]] constexpr const T& operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == Rank,
                      "a tensor takes exactly Rank indices");
        return components[flatten(indices...)];
    }

    /// Flat storage access, in row-major order.
    [[nodiscard]] constexpr T& operator[](std::size_t flat) noexcept {
        assert(flat < kComponents);
        return components[flat];
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t flat) const noexcept {
        assert(flat < kComponents);
        return components[flat];
    }

    [[nodiscard]] static constexpr Tensor zero() noexcept { return {}; }

    [[nodiscard]] static constexpr Tensor filled(T value) noexcept {
        Tensor result{};
        for (std::size_t i = 0; i < kComponents; ++i) {
            result.components[i] = value;
        }
        return result;
    }

    /// The Kronecker delta.
    [[nodiscard]] static constexpr Tensor delta() noexcept
        requires(Rank == 2)
    {
        Tensor result{};
        for (std::size_t i = 0; i < Dim; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

    constexpr Tensor& operator+=(const Tensor& other) noexcept {
        for (std::size_t i = 0; i < kComponents; ++i) {
            components[i] += other.components[i];
        }
        return *this;
    }

    constexpr Tensor& operator-=(const Tensor& other) noexcept {
        for (std::size_t i = 0; i < kComponents; ++i) {
            components[i] -= other.components[i];
        }
        return *this;
    }

    constexpr Tensor& operator*=(T scalar) noexcept {
        for (std::size_t i = 0; i < kComponents; ++i) {
            components[i] *= scalar;
        }
        return *this;
    }

    constexpr Tensor& operator/=(T scalar) noexcept {
        for (std::size_t i = 0; i < kComponents; ++i) {
            components[i] /= scalar;
        }
        return *this;
    }

    [[nodiscard]] friend constexpr Tensor operator+(const Tensor& t) noexcept {
        return t;
    }

    [[nodiscard]] friend constexpr Tensor operator-(const Tensor& t) noexcept {
        Tensor result{};
        for (std::size_t i = 0; i < kComponents; ++i) {
            result.components[i] = -t.components[i];
        }
        return result;
    }

    [[nodiscard]] friend constexpr Tensor operator+(const Tensor& a,
                                                    const Tensor& b) noexcept {
        Tensor result = a;
        result += b;
        return result;
    }

    [[nodiscard]] friend constexpr Tensor operator-(const Tensor& a,
                                                    const Tensor& b) noexcept {
        Tensor result = a;
        result -= b;
        return result;
    }

    [[nodiscard]] friend constexpr Tensor operator*(const Tensor& t,
                                                    T scalar) noexcept {
        Tensor result = t;
        result *= scalar;
        return result;
    }

    [[nodiscard]] friend constexpr Tensor operator*(T scalar,
                                                    const Tensor& t) noexcept {
        return t * scalar;
    }

    [[nodiscard]] friend constexpr Tensor operator/(const Tensor& t,
                                                    T scalar) noexcept {
        Tensor result = t;
        result /= scalar;
        return result;
    }

    [[nodiscard]] friend constexpr bool operator==(const Tensor&,
                                                   const Tensor&) = default;
};

/// The tensor product. Rank adds; every component of a meets every component
/// of b.
template <Numeric T, std::size_t RankA, std::size_t RankB, std::size_t Dim>
[[nodiscard]] constexpr Tensor<T, RankA + RankB, Dim> outerProduct(
    const Tensor<T, RankA, Dim>& a, const Tensor<T, RankB, Dim>& b) noexcept {
    Tensor<T, RankA + RankB, Dim> result{};
    constexpr std::size_t sizeB = Tensor<T, RankB, Dim>::size();

    for (std::size_t i = 0; i < Tensor<T, RankA, Dim>::size(); ++i) {
        for (std::size_t j = 0; j < sizeB; ++j) {
            result[i * sizeB + j] = a[i] * b[j];
        }
    }
    return result;
}

/// Contracts index I of a against index J of b, summing over the shared value.
/// The surviving indices of a come first, then those of b.
///
/// The index positions are template parameters, so contracting a rank-3 tensor
/// on an index it does not have fails to compile rather than reading past the
/// end of something.
template <std::size_t I, std::size_t J, Numeric T, std::size_t RankA,
          std::size_t RankB, std::size_t Dim>
[[nodiscard]] constexpr Tensor<T, RankA + RankB - 2, Dim> contract(
    const Tensor<T, RankA, Dim>& a, const Tensor<T, RankB, Dim>& b) noexcept {
    static_assert(I < RankA, "contracted index is out of range for the left tensor");
    static_assert(J < RankB,
                  "contracted index is out of range for the right tensor");

    Tensor<T, RankA + RankB - 2, Dim> result{};
    constexpr std::size_t restB = detail::integerPow(Dim, RankB - 1);

    for (std::size_t aFlat = 0; aFlat < Tensor<T, RankA, Dim>::size(); ++aFlat) {
        const std::size_t shared = detail::tensorIndexAt(aFlat, RankA, Dim, I);
        const std::size_t aRest = detail::tensorWithoutIndex(aFlat, RankA, Dim, I);

        for (std::size_t bFlat = 0; bFlat < Tensor<T, RankB, Dim>::size(); ++bFlat) {
            if (detail::tensorIndexAt(bFlat, RankB, Dim, J) != shared) {
                continue;
            }
            const std::size_t bRest =
                detail::tensorWithoutIndex(bFlat, RankB, Dim, J);
            result[aRest * restB + bRest] += a[aFlat] * b[bFlat];
        }
    }
    return result;
}

/// Contracts two indices of the same tensor against each other, dropping the
/// rank by two. This is how Ricci comes out of Riemann.
template <std::size_t I, std::size_t J, Numeric T, std::size_t Rank,
          std::size_t Dim>
[[nodiscard]] constexpr Tensor<T, Rank - 2, Dim> traceOver(
    const Tensor<T, Rank, Dim>& t) noexcept {
    static_assert(I < Rank && J < Rank, "traced index is out of range");
    static_assert(I != J, "a trace needs two distinct indices");

    constexpr std::size_t lower = (I < J) ? I : J;
    constexpr std::size_t upper = (I < J) ? J : I;

    Tensor<T, Rank - 2, Dim> result{};
    for (std::size_t flat = 0; flat < Tensor<T, Rank, Dim>::size(); ++flat) {
        if (detail::tensorIndexAt(flat, Rank, Dim, I) !=
            detail::tensorIndexAt(flat, Rank, Dim, J)) {
            continue;
        }
        // Remove the later index first, so the earlier position is still valid.
        const std::size_t once = detail::tensorWithoutIndex(flat, Rank, Dim, upper);
        result[detail::tensorWithoutIndex(once, Rank - 1, Dim, lower)] += t[flat];
    }
    return result;
}

/// The ordinary trace of a rank-2 tensor, as a scalar rather than a rank-0
/// tensor.
template <Numeric T, std::size_t Dim>
[[nodiscard]] constexpr T trace(const Tensor<T, 2, Dim>& t) noexcept {
    T sum{};
    for (std::size_t i = 0; i < Dim; ++i) {
        sum += t(i, i);
    }
    return sum;
}

template <std::size_t I, std::size_t J, Numeric T, std::size_t Rank,
          std::size_t Dim>
[[nodiscard]] constexpr Tensor<T, Rank, Dim> transposeIndices(
    const Tensor<T, Rank, Dim>& t) noexcept {
    static_assert(I < Rank && J < Rank, "swapped index is out of range");

    Tensor<T, Rank, Dim> result{};
    for (std::size_t flat = 0; flat < Tensor<T, Rank, Dim>::size(); ++flat) {
        result[detail::tensorSwapIndices(flat, Rank, Dim, I, J)] = t[flat];
    }
    return result;
}

/// The part of t unchanged by swapping indices I and J.
template <std::size_t I, std::size_t J, Numeric T, std::size_t Rank,
          std::size_t Dim>
[[nodiscard]] constexpr Tensor<T, Rank, Dim> symmetrize(
    const Tensor<T, Rank, Dim>& t) noexcept {
    return (t + transposeIndices<I, J>(t)) / T{2};
}

/// The part of t that changes sign under swapping indices I and J.
template <std::size_t I, std::size_t J, Numeric T, std::size_t Rank,
          std::size_t Dim>
[[nodiscard]] constexpr Tensor<T, Rank, Dim> antisymmetrize(
    const Tensor<T, Rank, Dim>& t) noexcept {
    return (t - transposeIndices<I, J>(t)) / T{2};
}

// Conversions with the small fixed types. A rank-2 tensor at Dim 4 and a
// Matrix4 hold the same numbers; which one to reach for is about whether the
// operation is index algebra or linear algebra.

template <Numeric T>
[[nodiscard]] constexpr Tensor<T, 1, 3> toTensor(const Vector3<T>& v) noexcept {
    Tensor<T, 1, 3> result{};
    for (std::size_t i = 0; i < 3; ++i) {
        result(i) = v[i];
    }
    return result;
}

template <Numeric T>
[[nodiscard]] constexpr Tensor<T, 1, 4> toTensor(const Vector4<T>& v) noexcept {
    Tensor<T, 1, 4> result{};
    for (std::size_t i = 0; i < 4; ++i) {
        result(i) = v[i];
    }
    return result;
}

template <Numeric T>
[[nodiscard]] constexpr Tensor<T, 2, 3> toTensor(const Matrix3<T>& m) noexcept {
    Tensor<T, 2, 3> result{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            result(r, c) = m(r, c);
        }
    }
    return result;
}

template <Numeric T>
[[nodiscard]] constexpr Tensor<T, 2, 4> toTensor(const Matrix4<T>& m) noexcept {
    Tensor<T, 2, 4> result{};
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            result(r, c) = m(r, c);
        }
    }
    return result;
}

template <Numeric T>
[[nodiscard]] constexpr Vector3<T> toVector3(const Tensor<T, 1, 3>& t) noexcept {
    return {t(0), t(1), t(2)};
}

template <Numeric T>
[[nodiscard]] constexpr Vector4<T> toVector4(const Tensor<T, 1, 4>& t) noexcept {
    return {t(0), t(1), t(2), t(3)};
}

template <Numeric T>
[[nodiscard]] constexpr Matrix3<T> toMatrix3(const Tensor<T, 2, 3>& t) noexcept {
    Matrix3<T> result{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            result(r, c) = t(r, c);
        }
    }
    return result;
}

template <Numeric T>
[[nodiscard]] constexpr Matrix4<T> toMatrix4(const Tensor<T, 2, 4>& t) noexcept {
    Matrix4<T> result{};
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            result(r, c) = t(r, c);
        }
    }
    return result;
}

/// The shapes the spacetime code will be built out of. Named here so those
/// modules can say what they mean rather than repeating the parameters.
template <Numeric T>
using MetricTensor = Tensor<T, 2, 4>;

template <Numeric T>
using ChristoffelSymbols = Tensor<T, 3, 4>;

template <Numeric T>
using RiemannTensor = Tensor<T, 4, 4>;

}  // namespace ysq
