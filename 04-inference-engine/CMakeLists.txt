cmake_minimum_required(VERSION 3.10) 
project(04-inference-engine)

# Set C++ Standard (C++17 recommended for modern syntax)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Create Executable
add_executable(main main.cpp)

# Link Libraries (OpenCV DNN module)
target_link_libraries(main ${OpenCV_LIBS})