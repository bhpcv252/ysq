#include <Core/Config.hpp>

#include <charconv>
#include <format>
#include <fstream>
#include <ios>
#include <iterator>
#include <system_error>

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

char lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool equalsIgnoringCase(std::string_view text, std::string_view expected) {
    if (text.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (lower(text[i]) != expected[i]) {
            return false;
        }
    }
    return true;
}

bool isKeyChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '_' || c == '-' || c == '.';
}

bool isValidKey(std::string_view key) {
    if (key.empty() || key.front() == '.' || key.back() == '.') {
        return false;
    }
    bool previousWasDot = false;
    for (const char c : key) {
        if (!isKeyChar(c)) {
            return false;
        }
        if (c == '.' && previousWasDot) {
            return false;  // an empty segment would not survive a round trip
        }
        previousWasDot = (c == '.');
    }
    return true;
}

bool isValidValue(std::string_view value) {
    return value.find('\n') == std::string_view::npos &&
           value.find('\r') == std::string_view::npos;
}

void setError(ConfigError* error, std::size_t line, std::string message) {
    if (error != nullptr) {
        *error = ConfigError{line, std::move(message)};
    }
}

}  // namespace

namespace detail {

std::optional<bool> parseBool(std::string_view text) {
    if (equalsIgnoringCase(text, "true") || text == "1") {
        return true;
    }
    if (equalsIgnoringCase(text, "false") || text == "0") {
        return false;
    }
    return std::nullopt;
}

std::optional<long long> parseSigned(std::string_view text) {
    long long value = 0;
    const char* const end = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

std::optional<unsigned long long> parseUnsigned(std::string_view text) {
    unsigned long long value = 0;
    const char* const end = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

// std::from_chars rather than a stream or strtod: it is locale-independent, so
// the file format cannot start depending on the host's LC_NUMERIC, and it round
// trips what std::format writes exactly, including denormals and infinities.
std::optional<double> parseDouble(std::string_view text) {
    double value = 0.0;
    const char* const end = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

std::string formatSigned(long long value) {
    return std::format("{}", value);
}

std::string formatUnsigned(unsigned long long value) {
    return std::format("{}", value);
}

// std::format's default for a double is the shortest representation that reads
// back bit-identical, which is exactly the round-trip guarantee this needs.
std::string formatDouble(double value) {
    return std::format("{}", value);
}

}  // namespace detail

bool Config::set(std::string_view key, std::string_view value) {
    return store(key, std::string{value});
}

bool Config::set(std::string_view key, const char* value) {
    return store(key, value == nullptr ? std::string{} : std::string{value});
}

std::string Config::get(std::string_view key, const char* fallback) const {
    const std::string* text = find(key);
    if (text != nullptr) {
        return *text;
    }
    return fallback == nullptr ? std::string{} : std::string{fallback};
}

bool Config::has(std::string_view key) const {
    return find(key) != nullptr;
}

bool Config::erase(std::string_view key) {
    const auto entry = m_values.find(key);
    if (entry == m_values.end()) {
        return false;
    }
    m_values.erase(entry);
    return true;
}

void Config::clear() {
    m_values.clear();
}

std::vector<std::string> Config::keys() const {
    std::vector<std::string> result;
    result.reserve(m_values.size());
    for (const auto& [key, value] : m_values) {
        result.push_back(key);
    }
    return result;
}

void Config::merge(const Config& overrides) {
    for (const auto& [key, value] : overrides.m_values) {
        m_values.insert_or_assign(key, value);
    }
}

const std::string* Config::find(std::string_view key) const {
    const auto entry = m_values.find(key);
    return entry == m_values.end() ? nullptr : &entry->second;
}

bool Config::store(std::string_view key, std::string value) {
    if (!isValidKey(key)) {
        return false;
    }
    const std::string_view trimmed = trim(value);
    if (!isValidValue(trimmed)) {
        return false;
    }
    m_values.insert_or_assign(std::string{key}, std::string{trimmed});
    return true;
}

std::string Config::toString() const {
    std::string out;

    // Un-sectioned keys first: the map interleaves them with dotted keys, and a
    // top-level key emitted after a section header would reparse into it.
    for (const auto& [key, value] : m_values) {
        if (key.find('.') == std::string::npos) {
            out += std::format("{} = {}\n", key, value);
        }
    }

    std::string_view section;
    for (const auto& [key, value] : m_values) {
        const std::size_t dot = key.find('.');
        if (dot == std::string::npos) {
            continue;
        }
        const std::string_view head = std::string_view{key}.substr(0, dot);
        if (head != section) {
            if (!out.empty()) {
                out += '\n';
            }
            out += std::format("[{}]\n", head);
            section = head;
        }
        out += std::format("{} = {}\n", std::string_view{key}.substr(dot + 1), value);
    }
    return out;
}

std::optional<Config> Config::parse(std::string_view text, ConfigError* error) {
    Config config;
    std::string section;
    std::size_t lineNumber = 0;

    for (std::size_t pos = 0; pos < text.size();) {
        std::size_t end = text.find('\n', pos);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        std::string_view line = text.substr(pos, end - pos);
        pos = end + 1;
        ++lineNumber;

        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        line = trim(line);
        if (line.empty() || line.front() == '#' || line.front() == ';') {
            continue;
        }

        if (line.front() == '[') {
            if (line.back() != ']') {
                setError(error, lineNumber, "unterminated section header");
                return std::nullopt;
            }
            const std::string_view name = trim(line.substr(1, line.size() - 2));
            if (!isValidKey(name)) {
                setError(error, lineNumber, std::format("invalid section '{}'", name));
                return std::nullopt;
            }
            section = name;
            continue;
        }

        const std::size_t separator = line.find('=');
        if (separator == std::string_view::npos) {
            setError(error, lineNumber, "expected 'key = value'");
            return std::nullopt;
        }

        const std::string_view name = trim(line.substr(0, separator));
        const std::string_view value = trim(line.substr(separator + 1));
        const std::string key =
            section.empty() ? std::string{name} : std::format("{}.{}", section, name);

        if (!config.store(key, std::string{value})) {
            setError(error, lineNumber, std::format("invalid key '{}'", key));
            return std::nullopt;
        }
    }

    return config;
}

std::optional<Config> Config::load(const std::filesystem::path& path,
                                   ConfigError* error, std::uintmax_t maxBytes) {
    // Size first: this is the only place Core reads a file it did not write, and
    // reading it whole is the simple implementation. The bound keeps a wrong
    // path, a device node or a truncated download from becoming an allocation
    // the size of whatever was pointed at.
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
        setError(error, 0, std::format("cannot stat '{}': {}", path.string(),
                                       ec.message()));
        return std::nullopt;
    }
    if (size > maxBytes) {
        setError(error, 0,
                 std::format("'{}' is {} bytes, over the {} byte limit",
                             path.string(), size, maxBytes));
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

bool Config::save(const std::filesystem::path& path) const {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return false;
    }
    file << toString();

    // Closing explicitly rather than letting the destructor do it: the
    // destructor's flush can fail on a full disk, and by then good() has already
    // been read and the failure reported as success.
    file.close();
    return file.good();
}

}  // namespace ysq
