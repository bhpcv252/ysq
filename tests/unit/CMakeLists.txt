ysq_add_test(test_core_version SOURCES core_version.cpp LIBS ysq::Core)

# The version as CMake renders it, so the test compares two independent paths
# out of project(YSQ VERSION ...) rather than one value against itself.
target_compile_definitions(test_core_version PRIVATE
    YSQ_EXPECTED_VERSION_STRING="${PROJECT_VERSION}")
