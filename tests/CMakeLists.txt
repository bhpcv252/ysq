include(GoogleTest)

# One executable per test file. gtest_discover_tests then registers each TEST()
# case with CTest individually, so a failure names the case, not the binary.
function(ysq_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE
        ${ARG_LIBS}
        GTest::gtest_main
        ysq::warnings)
    gtest_discover_tests(${name})
endfunction()

add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(smoke)
