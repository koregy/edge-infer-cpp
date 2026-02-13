cmake_minimum_required(VERSION 3.10)
project(01-pixel-control)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Create the executable target 'main' from 'main.cpp'
add_executable(main main.cpp)

# Link the OpenCV libraries to the executable target
target_link_libraries(main ${OpenCV_LIBS})