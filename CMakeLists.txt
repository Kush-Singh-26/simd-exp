cmake_minimum_required(VERSION 4.0)
project(simd-exp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)
find_package(benchmark REQUIRED)
include(CTest)

add_library(simd INTERFACE)
target_include_directories(simd
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(simd INTERFACE cxx_std_23)

find_package(OpenMP REQUIRED)
target_link_libraries(simd INTERFACE OpenMP::OpenMP_CXX)

set(ARCH_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  list(APPEND ARCH_FLAGS -mavx2 -mfma -mf16c)
endif()

function(add_simd_bench name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE simd benchmark::benchmark)
  target_compile_options(${name} PRIVATE -O3 ${ARCH_FLAGS})
endfunction()

function(add_simd_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE simd GTest::gtest_main)
  target_compile_options(${name} PRIVATE -O3 ${ARCH_FLAGS})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

file(GLOB BENCH_SOURCES "bench/*_bench.cpp")
file(GLOB TEST_SOURCES  "test/*_test.cpp")

set(BENCH_TARGETS "")
foreach(src ${BENCH_SOURCES})
  cmake_path(GET src STEM name)
  list(APPEND BENCH_TARGETS ${name})
  add_simd_bench(${name} ${src})
endforeach()

if(BUILD_TESTING)
  foreach(src ${TEST_SOURCES})
    cmake_path(GET src STEM name)
    add_simd_test(${name} ${src})
  endforeach()
endif()

add_custom_target(run_all
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/results
)
add_dependencies(run_all ${BENCH_TARGETS})

foreach(name ${BENCH_TARGETS})
  add_custom_command(TARGET run_all POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "--- ${name} ---"
    COMMAND $<TARGET_FILE:${name}>
      --benchmark_out=${CMAKE_CURRENT_SOURCE_DIR}/bench/results/${name}.json
      --benchmark_format=json
      --benchmark_counters_tabular=true
  )
endforeach()
