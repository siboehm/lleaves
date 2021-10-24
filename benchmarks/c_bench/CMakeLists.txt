cmake_minimum_required(VERSION 3.19)
project(c_bench)

set(CMAKE_CXX_STANDARD 11)
set(MODEL $ENV{LLEAVES_BENCHMARK_MODEL}) # NYC_taxi / airline / mtpl2

add_executable(c_bench c_bench.cpp)

# remove the cached model file
file(REMOVE ${MODEL}.o)
# generate new model file
find_package(Python COMPONENTS Interpreter)
add_custom_target(
        run ALL
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_binary.py ${MODEL}
        BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL}.o
        COMMENT "Compiling model"
)
add_dependencies(c_bench run)
find_package(benchmark REQUIRED)

target_link_libraries(c_bench cnpy ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL}.o benchmark::benchmark)
