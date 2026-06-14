cmake_minimum_required(VERSION 3.16)
project(simd-exp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(benchmark REQUIRED)

# Expose header-only library directory
add_library(simd INTERFACE)
target_include_directories(simd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS bench/*.cpp)
set(BENCH_TARGETS "")
foreach(src ${BENCH_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE simd benchmark::benchmark)
  target_compile_options(${name} PRIVATE -mavx2 -mfma -O3)
  list(APPEND BENCH_TARGETS ${name})
endforeach()

add_custom_target(run_all
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/bench/results
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