#pragma once

#include <Math/Complex.hpp>
#include <Math/CoordinateSystems.hpp>
#include <Math/Dual.hpp>
#include <Math/Matrix2.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Matrix4.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector2.hpp>
#include <Math/Vector3.hpp>
#include <Math/Vector4.hpp>

#include <cstddef>
#include <format>

/// std::formatter specializations for the Math types, so they drop straight
/// into Core/Logger and std::format.
///
///     logging::info("v = {:.3f}", velocity);   // v = (1.000, 0.000, -9.810)
///
/// Separate from the type headers on purpose. <format> is a heavy include and
/// the vector headers sit on the integration inner path, where every
/// translation unit that touches them would otherwise pay for it. Include this
/// only where something is actually being printed.
///
/// The format spec is forwarded to the component type, so any spec valid for a
/// double is valid here and applies to every component.
///
/// This covers the value types and nothing else. It deliberately does not
/// reach into ODE.hpp for PhaseState and StateVector: a formatting header
/// pulling in the integrator interface, and <vector> with it, is the wrong way
/// round, and nothing in the engine formats an integrator state. Printing one
/// is a debugging concern, and the formatters for those two live with the test
/// support that wants them. The generic templates below are public so they
/// can.

namespace ysq::detail {

/// One implementation for every vector-like value: anything with value_type,
/// size() and operator[]. size() is called on the instance, so a state whose
/// size is only known at run time works as well as a fixed one.
template <class V, class CharT>
struct VectorFormatter : std::formatter<typename V::value_type, CharT> {
    template <class Context>
    auto format(const V& v, Context& ctx) const {
        using Base = std::formatter<typename V::value_type, CharT>;
        auto out = ctx.out();
        *out++ = CharT('(');
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i != 0) {
                *out++ = CharT(',');
                *out++ = CharT(' ');
            }
            ctx.advance_to(out);
            out = Base::format(v[i], ctx);
        }
        *out++ = CharT(')');
        return out;
    }
};

/// One implementation for every fixed-size matrix: anything with value_type,
/// static rows() and cols(), and operator()(row, col).
///
/// Prints by rows, `[[m00, m01], [m10, m11]]`, even though the storage is
/// column-major. Display order follows how a matrix is read and written, not
/// how it is laid out; a log line in storage order would be a transposed
/// matrix on the screen and would cost someone an afternoon.
template <class M, class CharT>
struct MatrixFormatter : std::formatter<typename M::value_type, CharT> {
    template <class Context>
    auto format(const M& m, Context& ctx) const {
        using Base = std::formatter<typename M::value_type, CharT>;
        auto out = ctx.out();
        *out++ = CharT('[');
        for (std::size_t r = 0; r < M::rows(); ++r) {
            if (r != 0) {
                *out++ = CharT(',');
                *out++ = CharT(' ');
            }
            *out++ = CharT('[');
            for (std::size_t c = 0; c < M::cols(); ++c) {
                if (c != 0) {
                    *out++ = CharT(',');
                    *out++ = CharT(' ');
                }
                ctx.advance_to(out);
                out = Base::format(m(r, c), ctx);
            }
            *out++ = CharT(']');
        }
        *out++ = CharT(']');
        return out;
    }
};

/// Two-part numbers that read as a sum: a complex number as `1 - 2i`, a dual
/// as `1 - 2eps`. The sign is decided here and the magnitude formatted
/// unsigned, so the user's spec applies to both parts without a `+` in it
/// producing `1 + -2i`.
template <class V, class CharT, char... Suffix>
struct BinomialFormatter : std::formatter<typename V::value_type, CharT> {
    template <class Context>
    auto format(const V& v, Context& ctx) const {
        using T = typename V::value_type;
        using Base = std::formatter<T, CharT>;

        const bool negative = v[1] < T{0};
        const T magnitude = negative ? -v[1] : v[1];

        auto out = ctx.out();
        ctx.advance_to(out);
        out = Base::format(v[0], ctx);
        *out++ = CharT(' ');
        *out++ = negative ? CharT('-') : CharT('+');
        *out++ = CharT(' ');
        ctx.advance_to(out);
        out = Base::format(magnitude, ctx);
        ((*out++ = CharT(Suffix)), ...);
        return out;
    }
};

}  // namespace ysq::detail

template <class T, class CharT>
struct std::formatter<ysq::Vector2<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Vector2<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Vector3<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Vector3<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Vector4<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Vector4<T>, CharT> {};

/// Prints as (w, x, y, z), scalar part first, matching the storage order.
template <class T, class CharT>
struct std::formatter<ysq::Quaternion<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Quaternion<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Matrix2<T>, CharT>
    : ysq::detail::MatrixFormatter<ysq::Matrix2<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Matrix3<T>, CharT>
    : ysq::detail::MatrixFormatter<ysq::Matrix3<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Matrix4<T>, CharT>
    : ysq::detail::MatrixFormatter<ysq::Matrix4<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Complex<T>, CharT>
    : ysq::detail::BinomialFormatter<ysq::Complex<T>, CharT, 'i'> {};

template <class T, class CharT>
struct std::formatter<ysq::Dual<T>, CharT>
    : ysq::detail::BinomialFormatter<ysq::Dual<T>, CharT, 'e', 'p', 's'> {};

/// Coordinate triples print in declaration order: (radius, polar, azimuth) for
/// spherical, (radius, azimuth, height) for cylindrical.
template <class T, class CharT>
struct std::formatter<ysq::Spherical<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Spherical<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Cylindrical<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Cylindrical<T>, CharT> {};

template <class T, class CharT>
struct std::formatter<ysq::Polar<T>, CharT>
    : ysq::detail::VectorFormatter<ysq::Polar<T>, CharT> {};

/// Flat, in row-major order, which is the order the components are stored in.
/// A rank-4 tensor at Dim 4 prints 256 numbers; this is a debugging aid, not a
/// presentation format.
template <class T, std::size_t Rank, std::size_t Dim, class CharT>
struct std::formatter<ysq::Tensor<T, Rank, Dim>, CharT>
    : ysq::detail::VectorFormatter<ysq::Tensor<T, Rank, Dim>, CharT> {};
