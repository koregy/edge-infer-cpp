# Edge AI Inference Engine: C++ Optimization Log

**Project Goal:** To develop a high-performance, lightweight computer vision & AI inference system optimized for **Edge Devices** (Linux/Embedded). This project focuses on minimizing memory overhead, maximizing CPU cache locality, and implementing core algorithms from scratch using C++.

---

## Development Roadmap & Status

| Day | Module | Topic | Status |
| :---: | :--- | :--- | :---: |
| **01** | **Pixel Control** | Direct Memory Access & Row-Major Order Analysis | Completed |
| **02** | **Image Processing** | Sobel Edge Detection (Convolution) & Benchmarking | Completed |
| 03 | Model Conversion | PyTorch to ONNX Pipeline | Pending |
| 04 | Inference Engine | ONNX Runtime C++ Integration | Pending |
| 05 | Optimization | Latency & Memory Profiling | Pending |
| 06 | Documentation | Portfolio & Technical Report | Pending |

---

## Day 1: Direct Memory Access (Foundation)

**Objective**
To understand `cv::Mat` memory layout (Row-Major Order) and implement pixel manipulation using raw pointers (`uchar*`) to avoid function call overhead.

### Key Implementation
- **Memory Analysis:** Analyzed how 2D images are flattened into 1D memory blocks.
- **Pointer Arithmetic:** Replaced high-level API (`at<Vec3b>`) with direct memory address calculations.

### Code Snippet
```cpp
// Direct memory access for optimization
// Calculating offset manually: index = (row * width + col) * channels
int index = (y * width + x) * 3; 
data[index] = 0; // Access Blue channel directly

// [Correction] Use 'step' to handle memory padding alignment
// index = y * image.step + x * image.elemSize();
int index = y * src.step + x * 3;
```
### Result Visualization
<img src="assets/01-pixel-control.png" width="400" alt="Pixel Control Result" />

## Day 2: High-Performance Convolution (Optimization)
**Objective**
Implement Sobel Edge Detection from scratch and conduct a performance benchmark between "Naive Implementation" and "Pointer Optimization".

### Theory & Strategy
- **Convolution:** Implemented a 3x3 sliding window algorithm to calculate spatial gradients ($G_x, G_y$).
- **Memory Optimization:**
    - **Naive Method:** Uses `image.at<uchar>(y, x)` which includes boundary checks for safety but incurs high overhead.
    - **Pointer Method:** Uses `uchar* ptr` to access memory addresses directly. This minimizes instruction cycles and maximizes CPU Cache Hits (Spatial Locality).

### Implementation Details (Code Snippet)
```cpp
// Optimized: Pointer Arithmetic (No function call overhead)
const uchar* ptr_prev = src.ptr<uchar>(y - 1);
const uchar* ptr_curr = src.ptr<uchar>(y);
const uchar* ptr_next = src.ptr<uchar>(y + 1);

// Accessing pixels directly via pointers
int gx = -ptr_prev[x-1] + ptr_prev[x+1] - 2*ptr_curr[x-1] + ...
```

### **Performance Benchmark & Analysis**

#### **Evaluation Environment**
To ensure the reliability of the results, all tests were conducted in a controlled environment:
* **CPU:** 13th Gen Intel(R) Core(TM) i7-1370P (20 vCPUs, 24MB Cache)
* **OS:** Linux (Ubuntu 22.04.5 LTS)
* **Compiler:** `g++` (Ubuntu 11.4.0-1ubuntu1~22.04.2)
* **Optimization Flag:** `-O3` (Highest Optimization Level)
* **Input:** 512x512 Grayscale Image / **Iterations:** 1,000 loops

#### **Benchmark Results**
| Implementation Method |Total Time (1000 runs) | Avg Time per Frame | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Naive Implementation** | 671 ms | 0.671 ms | 1.0x |
| **Optimized Pointer** | **118 ms** | **0.118 ms** | **5.69x Faster** |

*(Note: The pointer optimization significantly reduces CPU cycles by maximizing cache locality.)*

### Result Visualization
<img src="assets/02-edge-detection.png" width="400" alt="Sobel Edge Result" />
<img src="assets/02-speedup-log.png" width="400" alt="Sobel Edge Result" />

### **Key Insight & Future Implications**
The **5.69x speedup** achieved through manual pointer optimization and memory locality enhancement provides critical advantages for Edge AI systems:

1.  **Computational Budget Realignment:** By drastically reducing the preprocessing (Sobel) overhead, we can allocate more CPU/GPU cycles to the actual Inference Stage, allowing for more complex model architectures.
2.  **Edge-Ready Efficiency:** This optimization is essential for achieving high FPS in **resource-constrained environments** (e.g., Jetson Nano, Raspberry Pi) where every millisecond of latency determines the system's real-time reliability.
3.  **Scalability:** The same memory access pattern can be extended to other convolution-based filters or custom tensor operations in the inference engine.