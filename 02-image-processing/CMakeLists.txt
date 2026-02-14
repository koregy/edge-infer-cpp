cmake_minimum_required(VERSION 3.10)
project(02-image-processing)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})