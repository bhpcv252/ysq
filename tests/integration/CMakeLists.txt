# Cross-module behaviour. Core is alone in the engine for now, so this covers
# its parts working as one runtime rather than one module against another.

ysq_add_test(test_core_runtime SOURCES core_runtime.cpp LIBS ysq::Core)
