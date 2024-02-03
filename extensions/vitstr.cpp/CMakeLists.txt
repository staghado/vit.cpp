cmake_minimum_required(VERSION 3.12)
project(vitstr
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Additional flags for C++ (the ffast-math is to be used with caution!)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# deps
add_subdirectory(/home/said/projects/vit.cpp/ggml /home/said/projects/vit.cpp/extensions/vitstr.cpp/build/ggml)
# add_subdirectory(${CMAKE_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

# vit executable, adjust paths for source files
add_executable(${PROJECT_NAME} main.cpp vitstr.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ggml)
target_include_directories(${PROJECT_NAME} PUBLIC extensions/vitstr)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# # quantize executable
# add_executable(quantize quantize.cpp)
# target_link_libraries(quantize PUBLIC ggml)
# target_include_directories(quantize PUBLIC .)
# target_compile_features(quantize PUBLIC cxx_std_11)

# benchmark executable
# add_executable(benchmark tests/benchmark.cpp vit.cpp)
# target_link_libraries(benchmark PUBLIC ggml)
# target_include_directories(benchmark PUBLIC .)
# target_compile_features(benchmark PUBLIC cxx_std_17)