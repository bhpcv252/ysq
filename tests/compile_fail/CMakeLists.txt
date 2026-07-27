# Tests that assert something does NOT compile.
#
# The primary check for this lives in tests/unit/units_dimensions.cpp, as
# concepts evaluated with static_assert. It is free, portable, and runs inside
# an ordinary test. These targets are the second line: they prove the real
# compiler rejects the real construct, not just that a concept says it would.
#
# Two properties of the mechanism are worth knowing before adding a case.
#
# WILL_FAIL only checks for a nonzero exit code, so a source with a typo in it
# also "passes". Nothing here is trustworthy on its own. Every case below is
# paired with its positive form in units_dimensions.cpp, and that pairing is
# the actual guarantee: the positive proves the construct is otherwise valid,
# the negative proves the dimension is what rejects it.
#
# Each test shells out to a nested `cmake --build`, so without RESOURCE_LOCK a
# `ctest -j` run would start several builds in the same tree at once.
#
# Matching the diagnostic text with PASS_REGULAR_EXPRESSION was considered and
# rejected: GCC, Clang and MSVC word these differently, and the expression
# would either be so loose it proves nothing or so tight it breaks on a
# compiler upgrade.

if(CMAKE_CONFIGURATION_TYPES)
    set(_ysq_config_arg --config $<CONFIG>)
else()
    set(_ysq_config_arg)
endif()

function(ysq_add_compile_fail_test name)
    add_executable(${name} EXCLUDE_FROM_ALL ${name}.cpp)
    target_link_libraries(${name} PRIVATE ysq::Units)

    add_test(NAME compile_fail.${name}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                --target ${name} ${_ysq_config_arg})

    set_tests_properties(compile_fail.${name} PROPERTIES
        WILL_FAIL TRUE
        RESOURCE_LOCK ysq_build_tree)
endfunction()

ysq_add_compile_fail_test(units_add_length_to_mass)
ysq_add_compile_fail_test(units_assign_time_to_length)
ysq_add_compile_fail_test(units_construct_length_from_number)
ysq_add_compile_fail_test(units_compare_length_to_mass)
ysq_add_compile_fail_test(units_sqrt_of_odd_dimension)
ysq_add_compile_fail_test(units_multiply_two_vectors)
