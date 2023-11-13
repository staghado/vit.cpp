cmake_minimum_required(VERSION 3.12)
project("vit.cpp")

add_subdirectory(ggml)

add_executable("vit.cpp" vit.cpp)

target_link_libraries("vit.cpp" PRIVATE ggml)