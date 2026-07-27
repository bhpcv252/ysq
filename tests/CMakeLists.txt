include(GoogleTest)

# Test-only helpers, header-only. The include directory is tests/ itself, so a
# test writes #include <support/MathApprox.hpp> and the path reads the same way
# as a module-qualified engine include.
add_library(ysq_test_support INTERFACE)
add_library(ysq::TestSupport ALIAS ysq_test_support)
target_include_directories(ysq_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ysq_test_support INTERFACE ysq::Math GTest::gtest)

# One executable per test file. gtest_discover_tests then registers each TEST()
# case with CTest individually, so a failure names the case, not the binary.
#
# STRICT swaps in ysq::warnings_strict. Only the targets that exist to compile
# the engine core's templates under the conversion warnings should ask for it;
# see src/Math/CMakeLists.txt for why the module cannot carry them itself.
function(ysq_add_test name)
    cmake_parse_arguments(ARG "STRICT" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    if(ARG_STRICT)
        set(_warnings ysq::warnings_strict)
    else()
        set(_warnings ysq::warnings)
    endif()
    target_link_libraries(${name} PRIVATE
        ${ARG_LIBS}
        GTest::gtest_main
        ${_warnings})
    gtest_discover_tests(${name})
endfunction()

add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(smoke)

# Targets that are expected to fail to compile. Off by default only if asked:
# each one costs a nested compiler invocation, which is noticeable on Windows.
if(YSQ_BUILD_COMPILE_FAIL_TESTS)
    add_subdirectory(compile_fail)
endif()
