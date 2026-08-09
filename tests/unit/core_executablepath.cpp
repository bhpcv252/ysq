#include <Core/ExecutablePath.hpp>

#include <gtest/gtest.h>

#include <filesystem>

namespace {

TEST(CoreExecutablePath, ExecutablePathResolvesToARealFile) {
    const std::optional<std::filesystem::path> path = ysq::executablePath();

    ASSERT_TRUE(path.has_value());
    EXPECT_TRUE(path->is_absolute());
    EXPECT_TRUE(std::filesystem::is_regular_file(*path)) << path->string();
}

TEST(CoreExecutablePath, ExecutableDirectoryIsARealDirectoryContainingThePath) {
    const std::optional<std::filesystem::path> directory = ysq::executableDirectory();
    const std::optional<std::filesystem::path> path = ysq::executablePath();

    ASSERT_TRUE(directory.has_value());
    ASSERT_TRUE(path.has_value());
    EXPECT_TRUE(std::filesystem::is_directory(*directory)) << directory->string();
    EXPECT_EQ(directory->string(), path->parent_path().string());
}

}  // namespace
