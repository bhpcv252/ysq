#pragma once

#include <Math/Format.hpp>
#include <Units/Unit.hpp>

#include <cstddef>
#include <format>

/// std::formatter for every Quantity, so a dimensioned value drops straight
/// into Core/Logger and std::format.
///
///     log::info("v = {:.3f}", speed);        // v = 7800.000 m s^-1
///     log::info("r = {:.2e}", position);     // r = (1.50e+11, 0.00e+00, ...) m
///
/// Separate from Unit.hpp for the same reason Math/Format.hpp is separate from
/// the vector headers: <format> is a heavy include and quantities sit on the
/// integration inner path. Include this only where something is printed.
///
/// The format spec is forwarded to the value, so any spec valid for a double
/// is valid here, and for a vector-valued quantity it applies to every
/// component exactly as it does in Math.
///
/// **Units print as base-unit powers, never as named derived symbols.** An
/// energy prints `m^2 kg s^-2` rather than `J`, and a torque prints the same
/// thing, because that is the truth: the two share a dimension and this module
/// cannot tell them apart. A formatter that guessed `J` would be inventing a
/// distinction the type system does not carry, and would be wrong half the
/// time it mattered. Base-unit powers are mechanical, unambiguous, and never
/// claim more than is known.
///
/// Factors appear in SI order, m kg s A K mol cd, with signed exponents.

namespace ysq::detail {

inline constexpr const char* kBaseUnitSymbols[7] = {"m", "kg", "s",  "A",
                                                    "K", "mol", "cd"};

/// Writes a signed integer without going through std::format, so the whole
/// thing stays character-type generic.
template <class CharT, class Out>
Out writeInteger(Out out, int value) {
    if (value < 0) {
        *out++ = CharT('-');
    }

    // Built up backwards, then reversed. An exponent is at most a couple of
    // digits, so the buffer is generous.
    char digits[12] = {};
    std::size_t count = 0;
    unsigned magnitude =
        (value < 0) ? (0u - static_cast<unsigned>(value)) : static_cast<unsigned>(value);
    do {
        digits[count++] = static_cast<char>('0' + (magnitude % 10u));
        magnitude /= 10u;
    } while (magnitude != 0u);

    while (count != 0) {
        *out++ = CharT(digits[--count]);
    }
    return out;
}

template <class D, class CharT, class Out>
Out writeUnitSymbol(Out out) {
    for (std::size_t i = 0; i < 7; ++i) {
        const int exponent = D::exponents[i];
        if (exponent == 0) {
            continue;
        }

        *out++ = CharT(' ');
        for (const char* symbol = kBaseUnitSymbols[i]; *symbol != '\0'; ++symbol) {
            *out++ = CharT(*symbol);
        }

        if (exponent != 1) {
            *out++ = CharT('^');
            out = writeInteger<CharT>(out, exponent);
        }
    }
    return out;
}

}  // namespace ysq::detail

template <class D, class V, class CharT>
struct std::formatter<ysq::Quantity<D, V>, CharT> : std::formatter<V, CharT> {
    template <class Context>
    auto format(const ysq::Quantity<D, V>& q, Context& ctx) const {
        using Base = std::formatter<V, CharT>;
        auto out = Base::format(q.value(), ctx);
        return ysq::detail::writeUnitSymbol<D, CharT>(out);
    }
};
