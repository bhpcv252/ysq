#include <Core/UUID.hpp>

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace {

TEST(CoreUuid, DefaultIsNil) {
    constexpr ysq::UUID nil;

    static_assert(nil.isNil());
    EXPECT_EQ(nil.toString(), "00000000-0000-0000-0000-000000000000");
    EXPECT_FALSE(ysq::UUID::generate().isNil());
}

TEST(CoreUuid, GeneratedIdsCarryVersion4AndTheRfcVariant) {
    for (int i = 0; i < 256; ++i) {
        const ysq::UUID id = ysq::UUID::generate();
        const ysq::UUID::Bytes& bytes = id.bytes();

        EXPECT_EQ(bytes[6] & 0xF0, 0x40) << "version nibble is not 4: " << id.toString();
        EXPECT_EQ(bytes[8] & 0xC0, 0x80)
            << "variant bits are not 10xx: " << id.toString();
    }
}

TEST(CoreUuid, StringFormIsLowercaseAndDashed) {
    const std::string text = ysq::UUID::generate().toString();

    ASSERT_EQ(text.size(), 36u);
    EXPECT_EQ(text[8], '-');
    EXPECT_EQ(text[13], '-');
    EXPECT_EQ(text[18], '-');
    EXPECT_EQ(text[23], '-');
    EXPECT_EQ(text[14], '4') << "the version digit belongs at index 14";
    for (const char c : text) {
        EXPECT_TRUE(c == '-' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "unexpected character '" << c << "' in " << text;
    }
}

TEST(CoreUuid, RoundTripsThroughItsStringForm) {
    for (int i = 0; i < 64; ++i) {
        const ysq::UUID original = ysq::UUID::generate();
        const std::optional<ysq::UUID> parsed = ysq::UUID::parse(original.toString());

        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(*parsed, original);
        EXPECT_EQ(parsed->toString(), original.toString());
    }
}

TEST(CoreUuid, ParsesUppercaseAndEmitsLowercase) {
    const std::optional<ysq::UUID> parsed =
        ysq::UUID::parse("A1B2C3D4-E5F6-4789-8ABC-DEF012345678");

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->toString(), "a1b2c3d4-e5f6-4789-8abc-def012345678");
}

TEST(CoreUuid, RejectsMalformedText) {
    EXPECT_FALSE(ysq::UUID::parse("").has_value());
    EXPECT_FALSE(ysq::UUID::parse("not-a-uuid").has_value());
    // Right length, no dashes.
    EXPECT_FALSE(ysq::UUID::parse("a1b2c3d4e5f647898abcdef0123456789abc").has_value());
    // Dashes in the wrong places.
    EXPECT_FALSE(ysq::UUID::parse("a1b2c3d4e-5f6-4789-8abc-def012345678").has_value());
    // Non-hex digit.
    EXPECT_FALSE(ysq::UUID::parse("g1b2c3d4-e5f6-4789-8abc-def012345678").has_value());
    // One character too long, and one too short.
    EXPECT_FALSE(ysq::UUID::parse("a1b2c3d4-e5f6-4789-8abc-def0123456789").has_value());
    EXPECT_FALSE(ysq::UUID::parse("a1b2c3d4-e5f6-4789-8abc-def01234567").has_value());
    // Braced form is not accepted.
    EXPECT_FALSE(ysq::UUID::parse("{a1b2c3d4-e5f6-4789-8abc-def012345678}").has_value());
}

TEST(CoreUuid, GeneratedIdsAreDistinct) {
    constexpr int kCount = 10000;
    std::unordered_set<ysq::UUID> seen;
    seen.reserve(kCount);

    for (int i = 0; i < kCount; ++i) {
        EXPECT_TRUE(seen.insert(ysq::UUID::generate()).second) << "collision at " << i;
    }
    EXPECT_EQ(seen.size(), static_cast<std::size_t>(kCount));
}

TEST(CoreUuid, SeededGeneratorReproducesItsSequence) {
    ysq::UuidGenerator first(12345);
    ysq::UuidGenerator second(12345);
    ysq::UuidGenerator other(54321);

    std::vector<ysq::UUID> firstRun;
    for (int i = 0; i < 8; ++i) {
        firstRun.push_back(first());
    }

    for (const ysq::UUID& expected : firstRun) {
        EXPECT_EQ(second(), expected);
    }
    EXPECT_NE(other(), firstRun.front());
    EXPECT_EQ(firstRun.front().bytes()[6] & 0xF0, 0x40) << "seeded ids are still v4";
}

TEST(CoreUuid, OrdersAndHashes) {
    const ysq::UUID low = *ysq::UUID::parse("00000000-0000-4000-8000-000000000001");
    const ysq::UUID high = *ysq::UUID::parse("00000000-0000-4000-8000-000000000002");

    EXPECT_LT(low, high);
    EXPECT_NE(low, high);
    EXPECT_EQ(low, *ysq::UUID::parse(low.toString()));

    const std::hash<ysq::UUID> hasher;
    EXPECT_EQ(hasher(low), hasher(*ysq::UUID::parse(low.toString())));
    EXPECT_NE(hasher(low), hasher(high));
}

}  // namespace
