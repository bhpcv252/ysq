#include <Core/Csv.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

TEST(CoreCsv, ParsesAHeaderAndTypedRows) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse(
        "name,mass_kg,semi_major_axis_km,is_regular\n"
        "Io,8.9319e22,421800,true\n"
        "Europa,4.7998e22,671100,true\n");

    ASSERT_TRUE(csv.has_value());
    EXPECT_EQ(csv->columnCount(), 4u);
    EXPECT_EQ(csv->rowCount(), 2u);
    EXPECT_EQ(csv->columns(), (std::vector<std::string>{"name", "mass_kg",
                                                         "semi_major_axis_km", "is_regular"}));

    const ysq::Csv::Row io = csv->row(0);
    EXPECT_EQ(io.get<std::string>("name", ""), "Io");
    EXPECT_DOUBLE_EQ(io.get<double>("mass_kg", 0.0), 8.9319e22);
    EXPECT_EQ(io.get<int>("semi_major_axis_km", 0), 421800);
    EXPECT_TRUE(io.get<bool>("is_regular", false));

    const ysq::Csv::Row europa = csv->row(1);
    EXPECT_EQ(europa.get<std::string>("name", ""), "Europa");
}

TEST(CoreCsv, RangeForWalksEveryRowInOrder) {
    const std::optional<ysq::Csv> csv =
        ysq::Csv::parse("name,a\nMercury,0.387\nVenus,0.723\nEarth,1.000\n");
    ASSERT_TRUE(csv.has_value());

    std::vector<std::string> names;
    for (const ysq::Csv::Row& row : *csv) {
        names.push_back(row.get<std::string>("name", ""));
    }
    EXPECT_EQ(names, (std::vector<std::string>{"Mercury", "Venus", "Earth"}));
}

TEST(CoreCsv, CommentAndBlankLinesAreSkippedNotTreatedAsRows) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse(
        "# source: JPL, epoch 2000-01-01.5 TDB\n"
        "\n"
        "name,a\n"
        "   \n"
        "Mercury,0.387\n"
        "# a mid-file comment\n"
        "Venus,0.723\n");

    ASSERT_TRUE(csv.has_value());
    EXPECT_EQ(csv->rowCount(), 2u);
    EXPECT_EQ(csv->row(0).get<std::string>("name", ""), "Mercury");
    EXPECT_EQ(csv->row(1).get<std::string>("name", ""), "Venus");
}

TEST(CoreCsv, EmptyFieldsAreARealRowNotSkipped) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("a,b,c\n,,\n1,,3\n");
    ASSERT_TRUE(csv.has_value());
    ASSERT_EQ(csv->rowCount(), 2u);
    EXPECT_EQ(csv->row(0).get<std::string>("a", "fallback"), "");
    EXPECT_EQ(csv->row(1).get<std::string>("a", "fallback"), "1");
    EXPECT_EQ(csv->row(1).get<std::string>("b", "fallback"), "");
}

TEST(CoreCsv, QuotedFieldsPreserveCommasNewlinesAndEscapedQuotes) {
    const std::optional<ysq::Csv> csv =
        ysq::Csv::parse("name,note\n"
                        "\"Europa, Jupiter II\",\"line one\nline two\"\n"
                        "Callisto,\"she said \"\"hello\"\"\"\n");

    ASSERT_TRUE(csv.has_value());
    ASSERT_EQ(csv->rowCount(), 2u);
    EXPECT_EQ(csv->row(0).get<std::string>("name", ""), "Europa, Jupiter II");
    EXPECT_EQ(csv->row(0).get<std::string>("note", ""), "line one\nline two");
    EXPECT_EQ(csv->row(1).get<std::string>("note", ""), "she said \"hello\"");
}

TEST(CoreCsv, UnquotedFieldsAreTrimmedQuotedFieldsAreNot) {
    const std::optional<ysq::Csv> csv =
        ysq::Csv::parse("a, b ,c\n 1 , \" 2 \" , 3\n");
    ASSERT_TRUE(csv.has_value());
    // Header names are trimmed too, so " b " reads back as "b".
    EXPECT_TRUE(csv->hasColumn("b"));
    EXPECT_EQ(csv->row(0).get<std::string>("a", ""), "1");
    EXPECT_EQ(csv->row(0).get<std::string>("b", ""), " 2 ");
    EXPECT_EQ(csv->row(0).get<std::string>("c", ""), "3");
}

TEST(CoreCsv, MissingColumnYieldsTheFallbackRatherThanThrowing) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("a\n1\n");
    ASSERT_TRUE(csv.has_value());
    EXPECT_FALSE(csv->row(0).has("nothere"));
    EXPECT_FALSE(csv->row(0).tryGet<int>("nothere").has_value());
    EXPECT_EQ(csv->row(0).get<int>("nothere", 42), 42);
}

TEST(CoreCsv, UnparsableValueYieldsTheFallbackRatherThanThrowing) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("name,a\nMercury,not-a-number\n");
    ASSERT_TRUE(csv.has_value());
    EXPECT_EQ(csv->row(0).get<double>("a", -1.0), -1.0);
    EXPECT_EQ(csv->row(0).get<int>("a", -1), -1);
}

TEST(CoreCsv, RowLineNumberPointsAtTheSourceLineIncludingSkippedComments) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse(
        "# comment\n"
        "name,a\n"
        "\n"
        "Mercury,0.387\n"
        "Venus,0.723\n");
    ASSERT_TRUE(csv.has_value());
    EXPECT_EQ(csv->row(0).lineNumber(), 4u);
    EXPECT_EQ(csv->row(1).lineNumber(), 5u);
}

TEST(CoreCsv, RaggedRowIsAParseErrorNamingItsLine) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv =
        ysq::Csv::parse("name,a,b\nMercury,0.387\n", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.line, 2u);
}

TEST(CoreCsv, DuplicateHeaderColumnIsAParseError) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("name,name\na,b\n", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.line, 1u);
}

TEST(CoreCsv, UnterminatedQuoteIsAParseError) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("name\n\"unterminated\n", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.line, 2u);
}

TEST(CoreCsv, ContentAfterAClosedQuoteIsAParseError) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("name\n\"quoted\"stray\n", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.line, 2u);
}

TEST(CoreCsv, QuoteInsideAnUnquotedFieldIsAParseError) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("name\nab\"cd\n", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.line, 2u);
}

TEST(CoreCsv, EmptyTextIsAMissingHeaderError) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.message, "no header row");
}

TEST(CoreCsv, LastRowWithoutATrailingNewlineIsStillRead) {
    const std::optional<ysq::Csv> csv = ysq::Csv::parse("name,a\nMercury,0.387");
    ASSERT_TRUE(csv.has_value());
    ASSERT_EQ(csv->rowCount(), 1u);
    EXPECT_EQ(csv->row(0).get<double>("a", 0.0), 0.387);
}

TEST(CoreCsv, LoadsFromAFileOnDisk) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ysq_core_csv_test.csv";
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << "name,a\nMercury,0.387\nVenus,0.723\n";
    }

    const std::optional<ysq::Csv> csv = ysq::Csv::load(path);
    std::filesystem::remove(path);

    ASSERT_TRUE(csv.has_value());
    EXPECT_EQ(csv->rowCount(), 2u);
    EXPECT_EQ(csv->row(0).get<std::string>("name", ""), "Mercury");
}

TEST(CoreCsv, LoadRefusesAFileOverTheByteLimit) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ysq_core_csv_test_big.csv";
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << "name,a\nMercury,0.387\n";
    }

    ysq::CsvError error;
    const std::optional<ysq::Csv> csv = ysq::Csv::load(path, &error, /*maxBytes=*/4);
    std::filesystem::remove(path);

    EXPECT_FALSE(csv.has_value());
    EXPECT_EQ(error.line, 0u);
}

TEST(CoreCsv, LoadReportsAMissingFileRatherThanThrowing) {
    ysq::CsvError error;
    const std::optional<ysq::Csv> csv =
        ysq::Csv::load("/nonexistent/path/does_not_exist.csv", &error);
    EXPECT_FALSE(csv.has_value());
    EXPECT_FALSE(error.message.empty());
}

}  // namespace
