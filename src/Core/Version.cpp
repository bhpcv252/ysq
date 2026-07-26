#include <Core/Version.hpp>

#include <format>

namespace ysq {

std::string versionString() {
    const Version v = version();
    return std::format("{}.{}.{}", v.major, v.minor, v.patch);
}

}  // namespace ysq
