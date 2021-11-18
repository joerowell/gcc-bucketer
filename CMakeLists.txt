cmake_minimum_required(VERSION 3.10)
project(
  GCCBucketer
  VERSION 0.5
  LANGUAGES CXX
  DESCRIPTION "BDGL bucketing with GCC intrinsics")

# ##############################################################################
# GTest
# ##############################################################################
add_subdirectory(googletest)
enable_testing()
include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# ##############################################################################
# Find benchmark
# ##############################################################################
# We need GoogleBenchmark.
find_package(benchmark REQUIRED)

# ##############################################################################
# Unit Tests
# ##############################################################################

# This is the list of sanitisers we use Each option will have its own
# corresponding binaries produced
# NB Asan also includes Tsan etc. 
set(Sanitisers Asan)
#Tsan)

# Reason for placing these here: easier removal from the flags if you can just
# delete a line here
set(WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wsign-conversion
    -Wmisleading-indentation
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wcast-qual
    -Wmissing-declarations
    -Wsign-promo
    -Wdisabled-optimization
    # -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast
    # Include -Wmismatched-tags if using GCC 10 or later (see e.g
    # https://stackoverflow.com/questions/66550681/gcc-mismatched-tags-option-giving-unrecognized-command-line-option)
    # -Wmismatched-tags
)

set(COVERAGE -fprofile-arcs -ftest-coverage)

#[[
  Because we've listed these arguments as separate elements in a list, we need to concatenate
  them into a nice string that'll essentially print them as if the arguments were concatenated into
  a single string. This requires quoting them, but we only need to do this in the definition of
  G_C_FLAGS
  ]]
#

list(JOIN WARNINGS " " WARNINGS)
list(JOIN COVERAGE " " COVERAGE)

# Global settings for great good The definition for G_C_FLAGS expands the
# warnings and coverage arguments from above.

set(G_C_FLAGS "-g3 -march=native -flax-vector-conversions " "${G_C_FLAGS}" "${WARNINGS}" "${COVERAGE}")
set(G_L_FLAGS "gtest gtest_main gcov")
set(G_C_OPT_FLAGS "-Ofast -march=native -mtune=native -flto -ftree-vectorize")

# This splits them up into a form that CMAKE can deal with when passing to
# executables
separate_arguments(C_FLAGS UNIX_COMMAND "${G_C_FLAGS}")
separate_arguments(L_FLAGS UNIX_COMMAND "${G_L_FLAGS}")

add_executable(SimdGCCTests src/simd_gcc.t.cpp src/simd.cpp src/fht_lsh_old.cpp)
target_compile_options(SimdGCCTests PRIVATE ${C_FLAGS})
target_link_libraries(SimdGCCTests PRIVATE ${L_FLAGS})
set_target_properties(SimdGCCTests PROPERTIES CXX_STANDARD 17)
add_test(SimdGCCTests SimdGCCTests)

add_executable(SimdIntelTests src/simd_intel.t.cpp src/simd.cpp src/fht_lsh_old.cpp)
target_compile_options(SimdIntelTests PRIVATE ${C_FLAGS})
target_link_libraries(SimdIntelTests PRIVATE ${L_FLAGS})
set_target_properties(SimdIntelTests PROPERTIES CXX_STANDARD 17)
add_test(SimdIntelTests SimdIntelTests)

# ##############################################################################
# Benchmarking
# ##############################################################################

# To begin, we use different compile flags. You won't want to run these
# benchmarks with assertions: what you *really* want to do is write benchmarks
# that show how fast the code will be in release mode.
# WARNING: funroll-loops seems to screw with Google benchmark. Leave it out for better results.

# NOTE: This is commented out because installing Google benchmark is confusing for GH actions.
# To enable these, remove the # [[ blocks 

# [[
set(G_C_FLAGS_BENCH
    "-Ofast -flax-vector-conversions -march=native -mtune=native -ftree-vectorize")
  
set(G_L_FLAGS_BENCH "benchmark::benchmark")


separate_arguments(C_FLAGS_BENCH UNIX_COMMAND "${G_C_FLAGS_BENCH}")
separate_arguments(L_FLAGS_BENCH UNIX_COMMAND "${G_L_FLAGS_BENCH}")

add_executable(SimdBench src/simd.b.cpp src/fht_lsh_old.cpp)
target_compile_options(SimdBench PRIVATE ${C_FLAGS_BENCH})
target_link_libraries(SimdBench PRIVATE ${L_FLAGS_BENCH})
set_target_properties(SimdBench PROPERTIES CXX_STANDARD 17)

# Turns out we should benchmark the bucketer too..
add_executable(BucketerBenchOld src/test_lsh_old.cpp src/fht_lsh_old.cpp)
target_compile_options(BucketerBenchOld PRIVATE ${C_FLAGS_BENCH})
target_link_libraries(BucketerBenchOld PRIVATE ${L_FLAGS_BENCH})
set_target_properties(BucketerBenchOld PROPERTIES CXX_STANDARD 17)

add_executable(BucketerBenchIntel src/test_lsh_intel.cpp src/fht_lsh.cpp)
target_compile_options(BucketerBenchIntel PRIVATE ${C_FLAGS_BENCH})
target_link_libraries(BucketerBenchIntel PRIVATE ${L_FLAGS_BENCH})
set_target_properties(BucketerBenchIntel PROPERTIES CXX_STANDARD 17)

add_executable(BucketerBenchGCC src/test_lsh_gcc.cpp src/fht_lsh.cpp)
target_compile_options(BucketerBenchGCC PRIVATE ${C_FLAGS_BENCH})
target_link_libraries(BucketerBenchGCC PRIVATE ${L_FLAGS_BENCH})
set_target_properties(BucketerBenchGCC PROPERTIES CXX_STANDARD 17)
#]]
