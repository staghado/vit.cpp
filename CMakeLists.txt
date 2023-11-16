cmake_minimum_required(VERSION 3.5)
project(vit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for Release mode
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Assuming the ggml.h is in an 'include' directory at your project root
include_directories(include)

# If ggml is a subdirectory of your project with its own CMakeLists.txt
# add_subdirectory(ggml)

# Define the executable
add_executable(${PROJECT_NAME} vit.cpp)

# Link libraries (if ggml and common are separate libraries you are using)
# target_link_libraries(${PROJECT_NAME} PRIVATE ggml common)