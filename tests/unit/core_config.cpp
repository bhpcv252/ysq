#include <Core/Config.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace {

ysq::Config populated() {
    ysq::Config config;
    EXPECT_TRUE(config.set("timeScale", 2.5));
    EXPECT_TRUE(config.set("physics.integrator", std::string{"rk4"}));
    EXPECT_TRUE(config.set("physics.timestep", std::numbers::pi));
    EXPECT_TRUE(config.set("physics.substeps", 4));
    EXPECT_TRUE(config.set("physics.adaptive", true));
    EXPECT_TRUE(config.set("render.vsync", false));
    EXPECT_TRUE(config.set("render.title", "YSQ"));
    EXPECT_TRUE(config.set("render.samples", 8u));
    return config;
}

TEST(CoreConfig, RoundTripsEverySupportedType) {
    const ysq::Config original = populated();

    ysq::ConfigError error;
    const std::optional<ysq::Config> reparsed =
        ysq::Config::parse(original.toString(), &error);

    ASSERT_TRUE(reparsed.has_value()) << "line " << error.line << ": " << error.message;
    EXPECT_EQ(*reparsed, original);

    EXPECT_DOUBLE_EQ(reparsed->get<double>("timeScale", 0.0), 2.5);
    EXPECT_EQ(reparsed->get<std::string>("physics.integrator", ""), "rk4");
    EXPECT_DOUBLE_EQ(reparsed->get<double>("physics.timestep", 0.0), std::numbers::pi);
    EXPECT_EQ(reparsed->get<int>("physics.substeps", 0), 4);
    EXPECT_TRUE(reparsed->get<bool>("physics.adaptive", false));
    EXPECT_FALSE(reparsed->get<bool>("render.vsync", true));
    EXPECT_EQ(reparsed->get<std::string>("render.title", ""), "YSQ");
    EXPECT_EQ(reparsed->get<unsigned>("render.samples", 0u), 8u);
}

// Not just "the values match": the text itself has to be a fixed point, or the
// file would churn every time something rewrites it.
TEST(CoreConfig, SerialisationIsAFixedPoint) {
    const std::string first = populated().toString();
    const std::optional<ysq::Config> reparsed = ysq::Config::parse(first);

    ASSERT_TRUE(reparsed.has_value());
    EXPECT_EQ(reparsed->toString(), first);
}

TEST(CoreConfig, WritesSectionsAndReadsThemBackAsDottedKeys) {
    const std::optional<ysq::Config> config =
        ysq::Config::parse("# a comment\n"
                           "\n"
                           "timeScale = 1.0\n"
                           "\n"
                           "[physics]\n"
                           "integrator = rk4\n"
                           "timestep   = 0.001\n"
                           "\n"
                           "; another comment style\n"
                           "[render]\n"
                           "vsync = true\n");

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->keys(),
              (std::vector<std::string>{"physics.integrator", "physics.timestep",
                                        "render.vsync", "timeScale"}));
    EXPECT_EQ(config->get<std::string>("physics.integrator", ""), "rk4");
    EXPECT_DOUBLE_EQ(config->get<double>("physics.timestep", 0.0), 0.001);
    EXPECT_TRUE(config->get<bool>("render.vsync", false));
}

TEST(CoreConfig, DoublesSurviveExactly) {
    const std::vector<double> values{std::numbers::pi,
                                     1.0 / 3.0,
                                     std::numeric_limits<double>::min(),
                                     std::numeric_limits<double>::denorm_min(),
                                     std::numeric_limits<double>::max(),
                                     -0.0,
                                     1e-300,
                                     6.02214076e23};

    ysq::Config config;
    for (std::size_t i = 0; i < values.size(); ++i) {
        ASSERT_TRUE(config.set("v" + std::to_string(i), values[i]));
    }

    const std::optional<ysq::Config> reparsed = ysq::Config::parse(config.toString());
    ASSERT_TRUE(reparsed.has_value());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::string key = "v" + std::to_string(i);
        EXPECT_EQ(reparsed->get<double>(key, 0.0), values[i]) << "at " << key;
    }
}

TEST(CoreConfig, NonFiniteDoublesRoundTrip) {
    ysq::Config config;
    ASSERT_TRUE(config.set("inf", std::numeric_limits<double>::infinity()));
    ASSERT_TRUE(config.set("negativeInf", -std::numeric_limits<double>::infinity()));
    ASSERT_TRUE(config.set("nan", std::numeric_limits<double>::quiet_NaN()));

    const std::optional<ysq::Config> reparsed = ysq::Config::parse(config.toString());
    ASSERT_TRUE(reparsed.has_value());

    EXPECT_EQ(reparsed->get<double>("inf", 0.0), std::numeric_limits<double>::infinity());
    EXPECT_EQ(reparsed->get<double>("negativeInf", 0.0),
              -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(reparsed->get<double>("nan", 0.0)));
}

TEST(CoreConfig, MissingKeyYieldsTheFallback) {
    const ysq::Config config;

    EXPECT_FALSE(config.has("nothing"));
    EXPECT_FALSE(config.tryGet<int>("nothing").has_value());
    EXPECT_EQ(config.get<int>("nothing", 42), 42);
    EXPECT_EQ(config.get<std::string>("nothing", "fallback"), "fallback");
}

TEST(CoreConfig, UnparsableValueYieldsTheFallbackRatherThanThrowing) {
    ysq::Config config;
    ASSERT_TRUE(config.set("integrator", "rk4"));
    ASSERT_TRUE(config.set("timestep", 0.5));

    EXPECT_EQ(config.get<int>("integrator", -1), -1);
    EXPECT_EQ(config.get<double>("integrator", -1.0), -1.0);
    EXPECT_FALSE(config.get<bool>("integrator", false));
    EXPECT_EQ(config.get<int>("timestep", -1), -1) << "0.5 is not an integer";
    EXPECT_EQ(config.get<std::string>("timestep", ""), "0.5");
}

TEST(CoreConfig, IntegerRangeIsChecked) {
    ysq::Config config;
    ASSERT_TRUE(config.set("big", std::numeric_limits<std::int64_t>::max()));
    ASSERT_TRUE(config.set("negative", -5));

    EXPECT_EQ(config.get<std::int64_t>("big", 0),
              std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(config.get<std::int16_t>("big", 7), 7) << "silently truncated";
    EXPECT_EQ(config.get<unsigned>("negative", 9u), 9u) << "negative read as unsigned";
    EXPECT_EQ(config.get<int>("negative", 0), -5);
}

TEST(CoreConfig, BoolsAcceptTheDocumentedSpellings) {
    const std::optional<ysq::Config> config = ysq::Config::parse(
        "a = true\nb = TRUE\nc = 1\nd = false\ne = False\nf = 0\ng = yes\n");

    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(config->get<bool>("a", false));
    EXPECT_TRUE(config->get<bool>("b", false));
    EXPECT_TRUE(config->get<bool>("c", false));
    EXPECT_FALSE(config->get<bool>("d", true));
    EXPECT_FALSE(config->get<bool>("e", true));
    EXPECT_FALSE(config->get<bool>("f", true));
    EXPECT_TRUE(config->get<bool>("g", true)) << "'yes' is not a documented spelling";
    EXPECT_FALSE(config->tryGet<bool>("g").has_value());
}

TEST(CoreConfig, RejectsKeysAndValuesThatWouldNotRoundTrip) {
    ysq::Config config;

    EXPECT_FALSE(config.set("", 1));
    EXPECT_FALSE(config.set(".leading", 1));
    EXPECT_FALSE(config.set("trailing.", 1));
    EXPECT_FALSE(config.set("double..dot", 1));
    EXPECT_FALSE(config.set("has space", 1));
    EXPECT_FALSE(config.set("has=equals", 1));
    EXPECT_FALSE(config.set("has[bracket]", 1));
    EXPECT_FALSE(config.set("multi", "line\nvalue"));
    EXPECT_EQ(config.size(), 0u);

    EXPECT_TRUE(config.set("valid.key-with_parts2", 1));
}

TEST(CoreConfig, TrimsSurroundingWhitespace) {
    ysq::Config config;
    ASSERT_TRUE(config.set("padded", "  spaced out  "));
    EXPECT_EQ(config.get<std::string>("padded", ""), "spaced out");

    const std::optional<ysq::Config> parsed =
        ysq::Config::parse("   padded   =    spaced out   \n");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->get<std::string>("padded", ""), "spaced out");
}

TEST(CoreConfig, ReadingAsAFloatIsRangeCheckedLikeAnInteger) {
    ysq::Config config;
    ASSERT_TRUE(config.set("huge", 1e300));
    ASSERT_TRUE(config.set("ordinary", 1.5));
    ASSERT_TRUE(config.set("boundless", std::numeric_limits<double>::infinity()));

    EXPECT_FALSE(config.tryGet<float>("huge").has_value())
        << "a double-sized value read as float must not become inf";
    EXPECT_FLOAT_EQ(config.get<float>("huge", 7.5f), 7.5f);
    EXPECT_FLOAT_EQ(config.get<float>("ordinary", 0.0f), 1.5f);
    EXPECT_DOUBLE_EQ(config.get<double>("huge", 0.0), 1e300);

    // An infinity is a value here, not an overflow, so it passes through.
    EXPECT_TRUE(std::isinf(config.get<float>("boundless", 0.0f)));
}

TEST(CoreConfig, StringFallbackNeedsNoExplicitType) {
    const ysq::Config config = populated();

    EXPECT_EQ(config.get("physics.integrator", "none"), "rk4");
    EXPECT_EQ(config.get("nothing", "fallback"), "fallback");

    const char* const absent = nullptr;
    EXPECT_EQ(config.get("nothing", absent), "");
}

TEST(CoreConfig, KeepsAnEqualsSignInsideAValue) {
    ysq::Config config;
    ASSERT_TRUE(config.set("expression", "a=b=c"));
    EXPECT_EQ(config.toString(), "expression = a=b=c\n");

    // Splitting on the first '=' is what makes this round trip.
    const std::optional<ysq::Config> reparsed = ysq::Config::parse(config.toString());
    ASSERT_TRUE(reparsed.has_value());
    EXPECT_EQ(reparsed->get<std::string>("expression", ""), "a=b=c");
}

TEST(CoreConfig, KeepsAHashInsideAValue) {
    const std::optional<ysq::Config> config = ysq::Config::parse("colour = #ff8800\n");

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->get<std::string>("colour", ""), "#ff8800");
    EXPECT_EQ(config->toString(), "colour = #ff8800\n");
}

TEST(CoreConfig, EmptyValueIsAnEmptyString) {
    const std::optional<ysq::Config> config = ysq::Config::parse("empty =\n");

    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(config->has("empty"));
    EXPECT_EQ(config->get<std::string>("empty", "fallback"), "");
    EXPECT_EQ(config->toString(), "empty = \n");
}

TEST(CoreConfig, ReportsTheOffendingLine) {
    ysq::ConfigError error;

    EXPECT_FALSE(ysq::Config::parse("a = 1\n\nb = 2\nthis line has no equals\n", &error)
                     .has_value());
    EXPECT_EQ(error.line, 4u);
    EXPECT_FALSE(error.message.empty());

    EXPECT_FALSE(ysq::Config::parse("[unterminated\n", &error).has_value());
    EXPECT_EQ(error.line, 1u);

    EXPECT_FALSE(ysq::Config::parse("[physics]\nhas space = 1\n", &error).has_value());
    EXPECT_EQ(error.line, 2u);
}

TEST(CoreConfig, ParsesAnEmptyDocument) {
    const std::optional<ysq::Config> config = ysq::Config::parse("");

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->size(), 0u);
    EXPECT_EQ(config->toString(), "");
}

TEST(CoreConfig, HandlesCarriageReturns) {
    const std::optional<ysq::Config> config =
        ysq::Config::parse("[physics]\r\ntimestep = 0.001\r\n");

    ASSERT_TRUE(config.has_value());
    EXPECT_DOUBLE_EQ(config->get<double>("physics.timestep", 0.0), 0.001);
}

TEST(CoreConfig, MergeLetsOverridesWin) {
    ysq::Config base;
    ASSERT_TRUE(base.set("a", 1));
    ASSERT_TRUE(base.set("b", 2));

    ysq::Config overrides;
    ASSERT_TRUE(overrides.set("b", 20));
    ASSERT_TRUE(overrides.set("c", 30));

    base.merge(overrides);

    EXPECT_EQ(base.get<int>("a", 0), 1);
    EXPECT_EQ(base.get<int>("b", 0), 20);
    EXPECT_EQ(base.get<int>("c", 0), 30);
}

TEST(CoreConfig, EraseAndClear) {
    ysq::Config config = populated();

    EXPECT_TRUE(config.erase("timeScale"));
    EXPECT_FALSE(config.erase("timeScale"));
    EXPECT_FALSE(config.has("timeScale"));

    config.clear();
    EXPECT_EQ(config.size(), 0u);
}

TEST(CoreConfig, SavesAndLoadsAFile) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ysq_core_config_test.ini";
    std::filesystem::remove(path);

    const ysq::Config original = populated();
    ASSERT_TRUE(original.save(path));

    ysq::ConfigError error;
    const std::optional<ysq::Config> loaded = ysq::Config::load(path, &error);
    ASSERT_TRUE(loaded.has_value()) << "line " << error.line << ": " << error.message;
    EXPECT_EQ(*loaded, original);

    std::filesystem::remove(path);
    EXPECT_FALSE(ysq::Config::load(path, &error).has_value());
    EXPECT_EQ(error.line, 0u) << "a missing file is not a line-level error";
}

TEST(CoreConfig, SaveReportsAFailedWrite) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                       "ysq_no_such_directory_zz" / "config.ini";

    EXPECT_FALSE(populated().save(path)) << "saving into a missing directory";
}

TEST(CoreConfig, RefusesAFileOverTheSizeLimit) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ysq_core_config_large.ini";
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(file.is_open());
        file << "key = " << std::string(512, 'x') << "\n";
    }

    ysq::ConfigError error;
    EXPECT_FALSE(ysq::Config::load(path, &error, 64).has_value());
    EXPECT_EQ(error.line, 0u);
    EXPECT_NE(error.message.find("limit"), std::string::npos) << error.message;

    // The same file is fine under the default limit.
    EXPECT_TRUE(ysq::Config::load(path, &error).has_value());

    std::filesystem::remove(path);
}

}  // namespace
