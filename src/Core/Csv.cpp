#include <Core/Csv.hpp>

#include <cassert>
#include <format>
#include <fstream>
#include <ios>
#include <iterator>
#include <system_error>
#include <unordered_map>

namespace ysq {

namespace {

bool isSpace(char c) {
    return c == ' ' || c == '\t' || c == '\v' || c == '\f';
}

std::string_view trim(std::string_view text) {
    std::size_t first = 0;
    while (first < text.size() && isSpace(text[first])) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first && isSpace(text[last - 1])) {
        --last;
    }
    return text.substr(first, last - first);
}

void setError(CsvError* error, std::size_t line, std::string message) {
    if (error != nullptr) {
        *error = CsvError{line, std::move(message)};
    }
}

/// One raw record: its fields exactly as tokenized (not yet trimmed, since
/// a quoted field must not be), and the 1-based line its first character
/// appeared on.
struct RawRecord {
    std::vector<std::string> fields;
    std::size_t line = 0;
};

/// The whole tokenizing pass: quote-aware, so a quoted field may contain a
/// literal comma or newline, and a comment or blank line is dropped before
/// it ever becomes a record. This is the one place line-counting happens;
/// everything downstream just trusts each RawRecord::line.
[[nodiscard]] std::optional<std::vector<RawRecord>> tokenize(std::string_view text,
                                                              CsvError* error) {
    std::vector<RawRecord> records;
    std::size_t pos = 0;
    std::size_t lineNumber = 1;

    while (pos < text.size()) {
        const std::size_t recordLine = lineNumber;

        // A comment line: first non-whitespace character on this line is
        // '#'. Detected before tokenizing so a quoted field is never
        // mistaken for one.
        std::size_t peek = pos;
        while (peek < text.size() && isSpace(text[peek])) {
            ++peek;
        }
        if (peek < text.size() && text[peek] == '#') {
            while (pos < text.size() && text[pos] != '\n') {
                ++pos;
            }
            if (pos < text.size()) {
                ++pos;  // consume the newline itself
                ++lineNumber;
            }
            continue;
        }

        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        // A quoted field is never trimmed (see the class doc comment); a
        // plain one always is. Tracked per field, reset at every comma and
        // at the start of a record.
        bool fieldWasQuoted = false;
        // True from the moment a quote closes until the next delimiter:
        // only whitespace is legal there (`"x" ,` is common and fine), any
        // other character means the field is malformed rather than
        // something to silently glue onto the end of the quoted content.
        bool afterClosingQuote = false;
        bool sawAnyCharacter = false;
        bool recordComplete = false;

        const auto pushField = [&fields, &field, &fieldWasQuoted] {
            fields.push_back(fieldWasQuoted ? field : std::string{trim(field)});
            field.clear();
            fieldWasQuoted = false;
        };

        while (pos < text.size() && !recordComplete) {
            const char c = text[pos];

            if (inQuotes) {
                if (c == '"') {
                    if (pos + 1 < text.size() && text[pos + 1] == '"') {
                        field += '"';
                        pos += 2;
                    } else {
                        inQuotes = false;
                        afterClosingQuote = true;
                        ++pos;
                    }
                } else {
                    if (c == '\n') {
                        ++lineNumber;
                    }
                    field += c;
                    ++pos;
                }
                sawAnyCharacter = true;
                continue;
            }

            if (afterClosingQuote && c != ',' && c != '\n' && c != '\r') {
                if (isSpace(c)) {
                    ++pos;
                    continue;
                }
                setError(error, lineNumber,
                         "unexpected content after a closed quoted field");
                return std::nullopt;
            }

            switch (c) {
                case '"':
                    // Only whitespace may precede an opening quote (` "x"`
                    // is a quoted field with incidental leading space, same
                    // as any other field); real content before it means
                    // this `"` is not opening anything.
                    if (!trim(field).empty()) {
                        setError(error, lineNumber,
                                 "unexpected '\"' inside an unquoted field");
                        return std::nullopt;
                    }
                    field.clear();  // discard whitespace seen before the quote
                    inQuotes = true;
                    fieldWasQuoted = true;
                    sawAnyCharacter = true;
                    ++pos;
                    break;
                case ',':
                    pushField();
                    afterClosingQuote = false;
                    sawAnyCharacter = true;
                    ++pos;
                    break;
                case '\r':
                    ++pos;
                    break;
                case '\n':
                    // Only push a trailing field if this line had real
                    // content before the newline (a comma, a quote, or a
                    // non-whitespace character): a bare "\n", or one with
                    // only whitespace before it, must stay a genuinely
                    // empty record so the blank-line check below drops it,
                    // rather than becoming a one-empty-field row.
                    if (sawAnyCharacter || !fields.empty()) {
                        pushField();
                    }
                    afterClosingQuote = false;
                    ++pos;
                    ++lineNumber;
                    recordComplete = true;
                    break;
                default:
                    field += c;
                    if (!isSpace(c)) {
                        sawAnyCharacter = true;
                    }
                    ++pos;
                    break;
            }
        }

        if (inQuotes) {
            setError(error, recordLine, "unterminated quoted field");
            return std::nullopt;
        }

        if (!recordComplete) {
            // End of text without a trailing newline: whatever was
            // accumulated is the final record, provided it is not nothing
            // at all.
            if (sawAnyCharacter || !fields.empty()) {
                pushField();
            }
        }

        if (sawAnyCharacter || !fields.empty()) {
            records.push_back(RawRecord{std::move(fields), recordLine});
        }
        // A record with no fields and no characters at all is a blank
        // line: skipped, matching Config's own convention.
    }

    return records;
}

}  // namespace

std::size_t Csv::Row::lineNumber() const noexcept {
    return m_table->m_lineNumbers[m_rowIndex];
}

bool Csv::Row::has(std::string_view column) const {
    return m_table->hasColumn(column);
}

std::string Csv::Row::get(std::string_view column, const char* fallback) const {
    const std::string* text = m_table->field(m_rowIndex, column);
    if (text != nullptr) {
        return *text;
    }
    return fallback == nullptr ? std::string{} : std::string{fallback};
}

bool Csv::hasColumn(std::string_view column) const {
    for (const std::string& name : m_header) {
        if (name == column) {
            return true;
        }
    }
    return false;
}

Csv::Row Csv::row(std::size_t index) const {
    assert(index < m_fields.size());
    return Row{*this, index};
}

std::size_t Csv::columnIndex(std::string_view column) const {
    for (std::size_t i = 0; i < m_header.size(); ++i) {
        if (m_header[i] == column) {
            return i;
        }
    }
    return m_header.size();
}

const std::string* Csv::field(std::size_t rowIndex, std::string_view column) const {
    const std::size_t index = columnIndex(column);
    if (index >= m_header.size()) {
        return nullptr;
    }
    return &m_fields[rowIndex][index];
}

std::optional<Csv> Csv::parse(std::string_view text, CsvError* error) {
    const std::optional<std::vector<RawRecord>> records = tokenize(text, error);
    if (!records) {
        return std::nullopt;
    }

    if (records->empty()) {
        setError(error, 0, "no header row");
        return std::nullopt;
    }

    Csv table;
    table.m_header = records->front().fields;

    if (table.m_header.empty()) {
        setError(error, records->front().line, "header row is empty");
        return std::nullopt;
    }

    std::unordered_map<std::string_view, std::size_t> seen;
    for (std::size_t i = 0; i < table.m_header.size(); ++i) {
        const std::string& name = table.m_header[i];
        if (name.empty()) {
            setError(error, records->front().line,
                     std::format("header column {} is empty", i + 1));
            return std::nullopt;
        }
        if (!seen.insert({name, i}).second) {
            setError(error, records->front().line,
                     std::format("duplicate header column '{}'", name));
            return std::nullopt;
        }
    }

    for (std::size_t r = 1; r < records->size(); ++r) {
        const RawRecord& record = (*records)[r];
        if (record.fields.size() != table.m_header.size()) {
            setError(error, record.line,
                     std::format("row has {} field(s), expected {} (matching the header)",
                                 record.fields.size(), table.m_header.size()));
            return std::nullopt;
        }
        table.m_fields.push_back(record.fields);
        table.m_lineNumbers.push_back(record.line);
    }

    return table;
}

std::optional<Csv> Csv::load(const std::filesystem::path& path, CsvError* error,
                             std::uintmax_t maxBytes) {
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
        setError(error, 0, std::format("cannot stat '{}': {}", path.string(), ec.message()));
        return std::nullopt;
    }
    if (size > maxBytes) {
        setError(error, 0,
                 std::format("'{}' is {} bytes, over the {} byte limit", path.string(), size,
                             maxBytes));
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        setError(error, 0, std::format("cannot open '{}'", path.string()));
        return std::nullopt;
    }
    const std::string text{std::istreambuf_iterator<char>{file},
                           std::istreambuf_iterator<char>{}};
    return parse(text, error);
}

}  // namespace ysq
