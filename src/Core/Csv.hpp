#pragma once

#include <Core/Config.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ysq {

struct CsvError {
    /// 1-based, the first line of the record the failure was found in. Zero
    /// when the failure was not tied to a line, such as a file that could
    /// not be opened.
    std::size_t line = 0;
    std::string message;
};

/// A CSV table, header row plus typed data rows.
///
/// The text form is RFC 4180 with two small, explicitly documented
/// extensions: comment lines and field trimming. Round-tripping is not a
/// goal the way it is for `Config` -- this reads data a consumer downloaded
/// or hand-curated, it does not own writing it back out.
///
///     # comment, skipped like a blank line
///     name, mass_kg, semi_major_axis_km
///     Io, 8.9319e22, 421800
///     "Europa, Jupiter II", 4.7998e22, 671100
///
/// - The first non-comment, non-blank line is the header: column names,
///   comma-separated. Header names must be non-empty and unique.
/// - Every data row must have exactly as many fields as the header has
///   columns; a short or long row is a parse error naming its line, not a
///   silently padded or truncated one.
/// - A field is quoted with `"`, doubled (`""`) to embed a literal quote,
///   and a quoted field may contain a literal comma or newline. A `"`
///   appearing outside a quoted field is a parse error.
/// - An unquoted field is trimmed of leading and trailing whitespace before
///   being stored; a quoted field is not, since the point of quoting is to
///   preserve exactly what is inside it.
/// - A line whose first non-whitespace character is `#` is a comment and is
///   skipped entirely, the same convention `Config`'s text form uses. A
///   real field beginning with `#` must be quoted.
/// - A line with no characters at all is skipped. A line that parses to
///   fields that happen to be empty (`,,`) is a real, present row.
class Csv {
public:
    /// One data row, typed access by column name. A `Row` is a view into
    /// the `Csv` that produced it and does not outlive it, the same
    /// convention `Config::find`'s returned pointer already uses.
    class Row {
    public:
        /// Supported types: bool, integral, floating-point and
        /// std::string. A missing column, or a field that will not parse
        /// as T, yields nullopt.
        template <class T>
        [[nodiscard]] std::optional<T> tryGet(std::string_view column) const;

        template <class T>
        [[nodiscard]] T get(std::string_view column, const T& fallback) const;
        /// So get(column, "default") works without spelling out
        /// get<std::string>.
        [[nodiscard]] std::string get(std::string_view column, const char* fallback) const;

        [[nodiscard]] bool has(std::string_view column) const;

        /// The source line this row started on, 1-based: useful in an error
        /// message a caller raises about one row's own data (a mass that
        /// parsed but is physically nonsensical, for instance), the same
        /// role `ConfigError::line` plays for `Config`.
        [[nodiscard]] std::size_t lineNumber() const noexcept;

    private:
        friend class Csv;
        Row(const Csv& table, std::size_t rowIndex) noexcept
            : m_table(&table), m_rowIndex(rowIndex) {}

        const Csv* m_table;
        std::size_t m_rowIndex;
    };

    class RowIterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Row;

        RowIterator() = default;
        RowIterator(const Csv& table, std::size_t rowIndex) noexcept
            : m_table(&table), m_rowIndex(rowIndex) {}

        [[nodiscard]] Row operator*() const { return m_table->row(m_rowIndex); }
        RowIterator& operator++() noexcept {
            ++m_rowIndex;
            return *this;
        }
        RowIterator operator++(int) noexcept {
            RowIterator copy = *this;
            ++*this;
            return copy;
        }
        [[nodiscard]] friend bool operator==(const RowIterator&,
                                             const RowIterator&) noexcept = default;

    private:
        const Csv* m_table = nullptr;
        std::size_t m_rowIndex = 0;
    };

    [[nodiscard]] const std::vector<std::string>& columns() const noexcept {
        return m_header;
    }
    [[nodiscard]] std::size_t columnCount() const noexcept { return m_header.size(); }
    [[nodiscard]] std::size_t rowCount() const noexcept { return m_fields.size(); }
    [[nodiscard]] bool hasColumn(std::string_view column) const;

    [[nodiscard]] Row row(std::size_t index) const;
    [[nodiscard]] RowIterator begin() const noexcept { return RowIterator{*this, 0}; }
    [[nodiscard]] RowIterator end() const noexcept { return RowIterator{*this, rowCount()}; }

    /// Refuses a file larger than maxBytes rather than allocating whatever
    /// it was pointed at. The default is generous for a data table
    /// downloaded from somewhere else, the one place `Core` reads a file it
    /// did not write.
    static constexpr std::uintmax_t kDefaultMaxFileBytes = 64u * 1024u * 1024u;

    [[nodiscard]] static std::optional<Csv> parse(std::string_view text,
                                                   CsvError* error = nullptr);
    [[nodiscard]] static std::optional<Csv>
    load(const std::filesystem::path& path, CsvError* error = nullptr,
         std::uintmax_t maxBytes = kDefaultMaxFileBytes);

private:
    [[nodiscard]] std::size_t columnIndex(std::string_view column) const;
    [[nodiscard]] const std::string* field(std::size_t rowIndex,
                                           std::string_view column) const;

    std::vector<std::string> m_header;
    std::vector<std::vector<std::string>> m_fields;  // one row per entry, header-aligned
    std::vector<std::size_t> m_lineNumbers;          // 1-based, parallel to m_fields
};

// Reuses Config.hpp's own detail::parseBool/parseSigned/parseUnsigned/
// parseDouble, detail::isConfigInteger and detail::alwaysFalse rather than a
// second copy: parsing one text field as bool/integer/double is exactly the
// same problem Config already solved, and this is the one other place in
// Core that needs it.
template <class T>
std::optional<T> Csv::Row::tryGet(std::string_view column) const {
    const std::string* text = m_table->field(m_rowIndex, column);
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
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, long double>) {
            if (std::isfinite(*value) &&
                std::abs(*value) > static_cast<double>(std::numeric_limits<T>::max())) {
                return std::nullopt;
            }
        }
        return static_cast<T>(*value);
    } else {
        static_assert(detail::alwaysFalse<T>,
                      "Csv::Row supports bool, integral, floating-point and std::string");
    }
}

template <class T>
T Csv::Row::get(std::string_view column, const T& fallback) const {
    return tryGet<T>(column).value_or(fallback);
}

}  // namespace ysq
