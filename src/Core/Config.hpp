#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ysq {

struct ConfigError {
    /// 1-based. Zero when the failure was not tied to a line, such as a file
    /// that could not be opened.
    std::size_t line = 0;
    std::string message;
};

namespace detail {

std::optional<bool> parseBool(std::string_view text);
std::optional<long long> parseSigned(std::string_view text);
std::optional<unsigned long long> parseUnsigned(std::string_view text);
std::optional<double> parseDouble(std::string_view text);

std::string formatSigned(long long value);
std::string formatUnsigned(unsigned long long value);
std::string formatDouble(double value);

/// The integer types std::in_range accepts. char and the character types are
/// deliberately not config integers; store them as strings.
template <class T>
inline constexpr bool isConfigInteger =
    std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char> &&
    !std::is_same_v<T, wchar_t> && !std::is_same_v<T, char8_t> &&
    !std::is_same_v<T, char16_t> && !std::is_same_v<T, char32_t>;

template <class>
inline constexpr bool alwaysFalse = false;

}  // namespace detail

/// Flat key/value configuration with typed access.
///
/// Keys are dotted paths. The text form is INI, and a `[section]` header is
/// simply a prefix: `[physics] timestep = 0.001` is the key `physics.timestep`.
///
///     # comment
///     timeScale = 1.0
///
///     [physics]
///     integrator = rk4
///     timestep   = 0.001
///
/// Round-tripping is total rather than nearly total, which costs a few rules:
///
/// - Valid key characters are alphanumerics, `_`, `-` and `.`. Segments must be
///   non-empty, so no leading, trailing or doubled dots.
/// - Values are trimmed of surrounding whitespace when stored, so leading and
///   trailing spaces are not preserved, and a value cannot contain a newline.
/// - Comments start at the beginning of a line. A `#` inside a value is data.
/// - Booleans are written `true` or `false`, and read from `true`, `false`,
///   `1` or `0`, in any case.
/// - Doubles are written in the shortest form that reads back bit-identical.
class Config {
public:
    /// Supported types: bool, integral, floating-point and std::string. A
    /// missing key, or a value that will not parse as T, yields nullopt.
    ///
    /// Every read is a map lookup and a fresh parse of the stored text. Read
    /// what a run needs once, into whatever owns it; this is not a hot-path
    /// container.
    template <class T>
    [[nodiscard]] std::optional<T> tryGet(std::string_view key) const;

    template <class T>
    [[nodiscard]] T get(std::string_view key, const T& fallback) const;
    /// So get(key, "default") works without spelling out get<std::string>.
    [[nodiscard]] std::string get(std::string_view key, const char* fallback) const;

    /// False if the key is malformed or the value cannot survive a round trip.
    template <class T>
    bool set(std::string_view key, const T& value);
    bool set(std::string_view key, std::string_view value);
    bool set(std::string_view key, const char* value);

    [[nodiscard]] bool has(std::string_view key) const;
    bool erase(std::string_view key);
    void clear();
    [[nodiscard]] std::size_t size() const noexcept { return m_values.size(); }
    /// Sorted.
    [[nodiscard]] std::vector<std::string> keys() const;

    /// Values in `overrides` win.
    void merge(const Config& overrides);

    [[nodiscard]] std::string toString() const;

    /// Refuses a file larger than maxBytes rather than allocating whatever it
    /// was pointed at. Config files are kilobytes; the default is generous.
    static constexpr std::uintmax_t kDefaultMaxFileBytes = 16u * 1024u * 1024u;

    [[nodiscard]] static std::optional<Config> parse(std::string_view text,
                                                     ConfigError* error = nullptr);
    [[nodiscard]] static std::optional<Config>
    load(const std::filesystem::path& path, ConfigError* error = nullptr,
         std::uintmax_t maxBytes = kDefaultMaxFileBytes);
    [[nodiscard]] bool save(const std::filesystem::path& path) const;

    [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;

private:
    [[nodiscard]] const std::string* find(std::string_view key) const;
    /// Validates and trims. Every write goes through here.
    bool store(std::string_view key, std::string value);

    // std::less<> so lookups take a string_view without building a std::string;
    // sorted so toString() is deterministic.
    std::map<std::string, std::string, std::less<>> m_values;
};

template <class T>
std::optional<T> Config::tryGet(std::string_view key) const {
    const std::string* text = find(key);
    if (text == nullptr) {
        return std::nullopt;
    }

    if constexpr (std::is_same_v<T, bool>) {
        return detail::parseBool(*text);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return *text;
    } else if constexpr (detail::isConfigInteger<T>) {
        if constexpr (std::is_signed_v<T>) {
            const std::optional<long long> value = detail::parseSigned(*text);
            if (!value || !std::in_range<T>(*value)) {
                return std::nullopt;
            }
            return static_cast<T>(*value);
        } else {
            const std::optional<unsigned long long> value = detail::parseUnsigned(*text);
            if (!value || !std::in_range<T>(*value)) {
                return std::nullopt;
            }
            return static_cast<T>(*value);
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        const std::optional<double> value = detail::parseDouble(*text);
        if (!value) {
            return std::nullopt;
        }
        // Range-checked like the integer path, so reading a double-sized value
        // as float reports "no" rather than quietly returning inf. Infinities
        // and NaN are values here, not overflow, so they pass through.
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, long double>) {
            if (std::isfinite(*value) &&
                std::abs(*value) > static_cast<double>(std::numeric_limits<T>::max())) {
                return std::nullopt;
            }
        }
        return static_cast<T>(*value);
    } else {
        static_assert(detail::alwaysFalse<T>,
                      "Config supports bool, integral, floating-point and std::string");
    }
}

template <class T>
T Config::get(std::string_view key, const T& fallback) const {
    return tryGet<T>(key).value_or(fallback);
}

template <class T>
bool Config::set(std::string_view key, const T& value) {
    if constexpr (std::is_same_v<T, bool>) {
        return store(key, value ? "true" : "false");
    } else if constexpr (std::is_same_v<T, std::string>) {
        return store(key, value);
    } else if constexpr (detail::isConfigInteger<T>) {
        if constexpr (std::is_signed_v<T>) {
            return store(key, detail::formatSigned(static_cast<long long>(value)));
        } else {
            return store(key,
                         detail::formatUnsigned(static_cast<unsigned long long>(value)));
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        return store(key, detail::formatDouble(static_cast<double>(value)));
    } else {
        static_assert(detail::alwaysFalse<T>,
                      "Config supports bool, integral, floating-point and std::string");
    }
}

}  // namespace ysq
