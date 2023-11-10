cmake_minimum_required(VERSION 3.6)
project(vitcpp)

# Include the ggml subdirectory
add_subdirectory(ggml)

# Add executable
add_executable(vitcpp vit.cpp)

# Link the executable with the ggml library
target_link_libraries(vitcpp PRIVATE ggml)