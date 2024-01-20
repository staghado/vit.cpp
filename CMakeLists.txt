cmake_minimum_required(VERSION 3.12)
project(vit
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags, modify here if needed!
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


# Additional flags for C++ (the ffast-math is to be used with caution!)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# deps
add_subdirectory(ggml)

# vit executable
add_executable(${PROJECT_NAME} main.cpp vit.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ggml)
target_include_directories(${PROJECT_NAME} PUBLIC .)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# quantize executable
option(BUILD_QUANTIZE "Build the quantization executable" ON)

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
    target_compile_features(quantize PUBLIC cxx_std_11)
endif()

# benchmark executable (optional)
option(BUILD_BENCHMARK "Build the benchmark executable" OFF)

if(BUILD_BENCHMARK)
    find_package(nlohmann_json 3.2.0 REQUIRED)  # JSON package needed for benchmark

    # JSON package needed for benchmark
    add_executable(benchmark tests/benchmark.cpp vit.cpp)
    target_link_libraries(benchmark PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(benchmark PUBLIC ggml)
    target_include_directories(benchmark PUBLIC .)
    target_compile_features(benchmark PUBLIC cxx_std_17)
endif()