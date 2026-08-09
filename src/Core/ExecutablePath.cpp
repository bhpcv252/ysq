#include <Core/ExecutablePath.hpp>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <cstdint>
#include <string>
#include <system_error>

namespace ysq {

namespace {

#if defined(_WIN32)

// GetModuleFileNameW truncates silently rather than reporting how much
// space it actually needed, so the only way to know the result fit is to
// see it come back shorter than the buffer offered.
std::optional<std::filesystem::path> rawExecutablePath() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length =
            GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return std::nullopt;
        }
        if (length < buffer.size()) {
            buffer.resize(length);
            return std::filesystem::path{buffer};
        }
        buffer.resize(buffer.size() * 2);
    }
}

#elif defined(__APPLE__)

// The standard two-call idiom: the first call (nullptr buffer, size 0)
// always fails but writes the required buffer size back through size.
std::optional<std::filesystem::path> rawExecutablePath() {
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);

    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return std::nullopt;
    }
    buffer.resize(std::char_traits<char>::length(buffer.c_str()));

    // Not necessarily fully resolved or absolute (Apple's own documented
    // caveat); canonical() both resolves it and follows any symlink in the
    // path, the same way the Linux branch below resolves /proc/self/exe.
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::canonical(buffer, error);
    return error ? std::nullopt : std::optional{resolved};
}

#else

std::optional<std::filesystem::path> rawExecutablePath() {
    std::error_code error;
    const std::filesystem::path resolved =
        std::filesystem::canonical("/proc/self/exe", error);
    return error ? std::nullopt : std::optional{resolved};
}

#endif

}  // namespace

std::optional<std::filesystem::path> executablePath() {
    return rawExecutablePath();
}

std::optional<std::filesystem::path> executableDirectory() {
    const std::optional<std::filesystem::path> path = executablePath();
    if (!path) {
        return std::nullopt;
    }
    return path->parent_path();
}

}  // namespace ysq
