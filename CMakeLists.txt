cmake_minimum_required(VERSION 3.5)
project(vitcpp)

add_executable(vitcpp vit.cpp)
target_include_directories(vitcpp PRIVATE ggml)
include_directories(${CMAKE_SOURCE_DIR})