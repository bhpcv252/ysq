# A developer tool for measuring real CPU-vs-GPU crossover points, not part of
# the test suite: see compute_thresholds.cpp's own comment for why it lives
# here rather than under tests/.
add_executable(benchmark_compute_thresholds compute_thresholds.cpp)
target_link_libraries(benchmark_compute_thresholds PRIVATE ysq::Compute ysq::Physics)
