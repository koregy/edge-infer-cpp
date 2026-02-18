cmake_minimum_required(VERSION 3.10) 
project(05-optimization)

# Set C++ Standard (C++17 recommended for modern syntax)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-linux-x64-1.19.2")

# Include directories
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Create Executable
add_executable(benchmark_ort benchmark_ort.cpp)

# Link Libraries
target_link_libraries(benchmark_ort
    ${OpenCV_LIBS}
    onnxruntime     # Links libonnxruntime.so
    # onnxruntime_providers_cuda
)

set_target_properties(benchmark_ort PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)