cmake_minimum_required(VERSION 3.5)
project(vitcpp)

# Include the ggml subdirectory
add_subdirectory(ggml)

# Add your executable
add_executable(vitcpp vit.cpp)

# Link your executable with the ggml library
target_link_libraries(vitcpp PRIVATE ggml)