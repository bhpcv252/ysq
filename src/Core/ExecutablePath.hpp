#pragma once

#include <filesystem>
#include <optional>

namespace ysq {

/// The currently running process's own executable, and where it lives on
/// disk. A general OS-process fact, not a graphics one, so it belongs here
/// rather than in `Platform`: the primitive an application uses to find
/// data shipped alongside it (see src/Applications/README.md's convention)
/// regardless of what directory it was launched from or unzipped into.

/// The absolute, symlink-resolved path to the running executable.
/// `std::nullopt` only if the underlying OS call itself fails, which does
/// not happen in ordinary use -- every supported platform can always
/// report its own running executable's path.
[[nodiscard]] std::optional<std::filesystem::path> executablePath();

/// `executablePath()`'s own parent directory. `std::nullopt` under the
/// same condition `executablePath()` is.
[[nodiscard]] std::optional<std::filesystem::path> executableDirectory();

}  // namespace ysq
