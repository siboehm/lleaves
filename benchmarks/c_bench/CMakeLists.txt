cmake_minimum_required(VERSION 3.19)
project(c_bench)

set(CMAKE_CXX_STANDARD 11)
set(MODEL $ENV{LLEAVES_BENCHMARK_MODEL}) # NYC_taxi / airline / mtpl2

add_executable(benchmark c_bench.cpp)

file(REMOVE ${MODEL}.o)
add_custom_target(
        run ALL
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen_binary.py ${MODEL}
        BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL}.o
        COMMENT "Compiling model"
)
add_dependencies(benchmark run)

target_link_libraries(benchmark cnpy ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL}.o)
