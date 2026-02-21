cmake_minimum_required(VERSION 3.10)
project(EdgeInferenceEngine)

# Use C++ 17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------
# 1. OpenCV Configuration
# -------------------------------------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------------------------------------------------
# 2. ONNX Runtime Configuration (Based on v1.19.2)
# -------------------------------------------------------
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-linux-x64-1.19.2")
include_directories("${ONNXRUNTIME_ROOT}/include")
link_directories("${ONNXRUNTIME_ROOT}/lib")

# -------------------------------------------------------
# 3. Include Custom Header Directory
# -------------------------------------------------------
# Allows the compiler to find .hpp files inside the 'include' folder.
include_directories(include)

# -------------------------------------------------------
# 4. Create Executable and Link Libraries
# -------------------------------------------------------
# Compile main.cpp and inference_engine.cpp together.
add_executable(inference_app src/main.cpp src/inference_engine.cpp)

# Link OpenCV and ONNX Runtime libraries to the executable.
target_link_libraries(inference_app ${OpenCV_LIBS} onnxruntime)

# Set RPATH (Runtime Search Path)
# Ensures the executable can automatically find the ONNX Runtime library at runtime.
set_target_properties(inference_app PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)