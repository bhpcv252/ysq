ysq_add_test(test_core_clock SOURCES core_clock.cpp LIBS ysq::Core)
ysq_add_test(test_core_config SOURCES core_config.cpp LIBS ysq::Core)
ysq_add_test(test_core_event SOURCES core_event.cpp LIBS ysq::Core)
ysq_add_test(test_core_logger SOURCES core_logger.cpp LIBS ysq::Core)
ysq_add_test(test_core_timer SOURCES core_timer.cpp LIBS ysq::Core)
ysq_add_test(test_core_uuid SOURCES core_uuid.cpp LIBS ysq::Core)
ysq_add_test(test_core_version SOURCES core_version.cpp LIBS ysq::Core)

# The version as CMake renders it, so the test compares two independent paths
# out of project(YSQ VERSION ...) rather than one value against itself.
target_compile_definitions(test_core_version PRIVATE
    YSQ_EXPECTED_VERSION_STRING="${PROJECT_VERSION}")
