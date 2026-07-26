#include <Core/UUID.hpp>

#include <atomic>
#include <chrono>

namespace ysq {

namespace {

constexpr char kHexDigits[] = "0123456789abcdef";
constexpr std::size_t kTextLength = 36;
constexpr std::size_t kDashPositions[] = {8, 13, 18, 23};

// std::random_device is deterministic on some MinGW builds, so it is mixed with
// a clock reading and a process-wide counter rather than trusted alone.
std::uint64_t mixedSeed() {
    static std::atomic<std::uint64_t> counter{0};

    std::random_device device;
    const std::uint64_t entropy = (static_cast<std::uint64_t>(device()) << 32) ^
                                  static_cast<std::uint64_t>(device());
    const auto now = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    return entropy ^ (now * 0x9e3779b97f4a7c15ULL) ^
           counter.fetch_add(1, std::memory_order_relaxed);
}

UUID fromDraws(std::uint64_t high, std::uint64_t low) {
    UUID::Bytes bytes{};
    for (std::size_t i = 0; i < 8; ++i) {
        const unsigned shift = static_cast<unsigned>(8 * (7 - i));
        bytes[i] = static_cast<std::uint8_t>((high >> shift) & 0xFFu);
        bytes[8 + i] = static_cast<std::uint8_t>((low >> shift) & 0xFFu);
    }
    bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0Fu) | 0x40u);  // version 4
    bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3Fu) | 0x80u);  // variant 10xx
    return UUID{bytes};
}

std::optional<std::uint8_t> hexValue(char c) {
    if (c >= '0' && c <= '9') {
        return static_cast<std::uint8_t>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<std::uint8_t>(c - 'a' + 10);
    }
    if (c >= 'A' && c <= 'F') {
        return static_cast<std::uint8_t>(c - 'A' + 10);
    }
    return std::nullopt;
}

}  // namespace

UUID UUID::generate() {
    thread_local std::mt19937_64 engine{mixedSeed()};
    return fromDraws(engine(), engine());
}

std::optional<UUID> UUID::parse(std::string_view text) {
    if (text.size() != kTextLength) {
        return std::nullopt;
    }
    for (const std::size_t position : kDashPositions) {
        if (text[position] != '-') {
            return std::nullopt;
        }
    }

    Bytes bytes{};
    std::size_t nibble = 0;
    for (const char c : text) {
        if (c == '-') {
            continue;  // positions were validated above
        }
        const std::optional<std::uint8_t> value = hexValue(c);
        if (!value) {
            return std::nullopt;
        }
        std::uint8_t& byte = bytes[nibble / 2];
        byte = (nibble % 2 == 0) ? static_cast<std::uint8_t>(*value << 4)
                                 : static_cast<std::uint8_t>(byte | *value);
        ++nibble;
    }

    // A dash anywhere other than the four validated positions leaves this short.
    if (nibble != 2 * bytes.size()) {
        return std::nullopt;
    }
    return UUID{bytes};
}

std::string UUID::toString() const {
    std::string out(kTextLength, '-');
    std::size_t pos = 0;
    for (std::size_t i = 0; i < m_bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            ++pos;  // step over the dash already in place
        }
        out[pos++] = kHexDigits[m_bytes[i] >> 4];
        out[pos++] = kHexDigits[m_bytes[i] & 0x0F];
    }
    return out;
}

UUID UuidGenerator::operator()() {
    return fromDraws(m_engine(), m_engine());
}

}  // namespace ysq

std::size_t std::hash<ysq::UUID>::operator()(const ysq::UUID& id) const noexcept {
    // FNV-1a over the 16 bytes. Accumulated in 64 bits regardless of size_t.
    std::uint64_t result = 1469598103934665603ULL;
    for (const std::uint8_t byte : id.bytes()) {
        result ^= byte;
        result *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(result);
}
