#include <Core/Version.hpp>

#include <gtest/gtest.h>

#include <string>

// version() has to be usable at compile time; a runtime-only constant would be
// no use for static configuration checks later.
static_assert(ysq::version() == ysq::version());

TEST(CoreVersion, StringMatchesCMakeProjectVersion) {
    // YSQ_EXPECTED_VERSION_STRING comes straight from project(YSQ VERSION ...).
    // versionString() comes from the configured header. They agreeing is what
    // proves configure_file substituted the right numbers.
    EXPECT_EQ(ysq::versionString(), std::string{YSQ_EXPECTED_VERSION_STRING});
}

TEST(CoreVersion, StringIsMajorMinorPatch) {
    const ysq::Version v = ysq::version();
    const std::string expected = std::to_string(v.major) + "." + std::to_string(v.minor) +
                                 "." + std::to_string(v.patch);
    EXPECT_EQ(ysq::versionString(), expected);
}
