#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <string_view>

namespace ysq {

/// RFC 4122 version 4 (random) identifier.
class UUID {
public:
    using Bytes = std::array<std::uint8_t, 16>;

    /// The nil UUID.
    constexpr UUID() noexcept = default;
    explicit constexpr UUID(const Bytes& bytes) noexcept : m_bytes(bytes) {}

    [[nodiscard]] static UUID generate();

    /// Accepts exactly 8-4-4-4-12 hex, either case. Nothing else.
    [[nodiscard]] static std::optional<UUID> parse(std::string_view text);

    /// Lowercase 8-4-4-4-12.
    [[nodiscard]] std::string toString() const;

    [[nodiscard]] constexpr const Bytes& bytes() const noexcept { return m_bytes; }
    [[nodiscard]] constexpr bool isNil() const noexcept { return m_bytes == Bytes{}; }

    [[nodiscard]] friend constexpr bool operator==(const UUID&, const UUID&) = default;
    [[nodiscard]] friend constexpr auto operator<=>(const UUID&, const UUID&) = default;

private:
    Bytes m_bytes{};
};

/// Explicitly seeded generator, for runs that have to be reproducible. The
/// sequence depends only on the seed, so a scenario replays identically.
class UuidGenerator {
public:
    explicit UuidGenerator(std::uint64_t seed) : m_engine(seed) {}

    [[nodiscard]] UUID operator()();

private:
    std::mt19937_64 m_engine;
};

}  // namespace ysq

template <>
struct std::hash<ysq::UUID> {
    [[nodiscard]] std::size_t operator()(const ysq::UUID& id) const noexcept;
};
