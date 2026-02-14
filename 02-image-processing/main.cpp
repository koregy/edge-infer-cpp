#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>       // For benchmarking
#include <cmath>        // For abs()
#include <algorithm>    // For std::min

using namespace cv;
using namespace std::chrono;

// ---------------------------------------------------------
// [Method 1] Naive Implementation (Safe but Slow)
// ---------------------------------------------------------
// Description:
// Uses 'at<uchar>()' which performs boundary checks for every pixel access
// This safety mechanism causes significant overhead in real-time processing
void sobelNaive(const Mat& src, Mat& dst) {
    int width = src.cols;
    int height = src.rows;

    // Iterate through the image (excluding 1-pixel border to avoid boundary checks)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            // 1. Calculate Gx (Vertical Edge Detection)
            // Kernel:
            // -1   0   1
            // -2   0   2
            // -1   0   1
            int gx = -src.at<uchar>(y - 1, x - 1) + src.at<uchar>(y - 1, x + 1)
                    - 2 * src.at<uchar>(y, x - 1) + 2 * src.at<uchar>(y, x + 1)
                    - src.at<uchar>(y + 1, x - 1) + src.at<uchar>(y + 1, x + 1);

            // 2. Calculate Gy (Horizontal Edge Detection)
            // Kernel:
            // -1   -2  -1
            // 0    0   0
            // 1    2   1
            int gy = -src.at<uchar>(y - 1, x - 1) - 2 * src.at<uchar>(y - 1, x) - src.at<uchar>(y - 1, x + 1)
                    + src.at<uchar>(y + 1, x - 1) + 2 * src.at<uchar>(y + 1, x) + src.at<uchar>(y + 1, x + 1);

            // 3. Calculate Gradient Magnitude
            // Approximation: |Gx| + |Gy| is computationally cheaper than sqrt(Gx^2 + Gy^2)
            // and sufficient for edge detection on embedded systems
            int sum = abs(gx) + abs(gy);

            // [Saturation] Clamp values > 255 to 255 to prevent overflow
            dst.at<uchar>(y, x) = (uchar)std::min(sum, 255);
        }
    }
}

// ---------------------------------------------------------
// [Method 2] Optimized Implementation (Pointer Access)
// ---------------------------------------------------------
// Description:
// Directly accesses memory addresses using pointers
// This eliminates functiion call overhead and maximizes CPU cache hits by following Row-Major Order
void sobelPointer(const Mat& src, Mat& dst) {
    int width = src.cols;
    int height = src.rows;

    // Process each row
    for (int y = 1; y < height - 1; y++) {

        // [Optimization] Pre-fetch row pointers
        // Accessing memory sequentially (ptr[x], ptr[x+1]) improves cache locality
        const uchar* ptr_prev = src.ptr<uchar>(y - 1);  // Row y-1
        const uchar* ptr_curr = src.ptr<uchar>(y);      // Row y
        const uchar* ptr_next = src.ptr<uchar>(y + 1);  // Row y+1

        uchar* ptr_dst = dst.ptr<uchar>(y); // Destination Row

        for (int x = 1; x < width - 1; x++) {
            // 1. Calculate Gx using pointer arithmetic
            int gx = -ptr_prev[x - 1] + ptr_prev[x + 1]
                    - 2 * ptr_curr[x - 1] + 2 * ptr_curr[x + 1]
                    - ptr_next[x - 1] + ptr_next[x + 1];
            
            // 2. Calculate Gy using pointer arithmetic
            int gy = -ptr_prev[x - 1] - 2 * ptr_prev[x] - ptr_prev[x + 1]
                    + ptr_next[x - 1] + 2 * ptr_next[x] + ptr_next[x + 1];
            
            // 3. Magnitude & Saturation
            int sum = abs(gx) + abs(gy);
            ptr_dst[x] = (uchar)std::min(sum, 255);
        }      
    }
}

int main() {
    // ---------------------------------------------------------
    // 1. Load Image
    // ---------------------------------------------------------
    // Note: 'test.jpg' must be present in the build directory
    std::cout << ">> [Step 1] Loading Image... " << std::flush;
    Mat img = imread("test.jpg", IMREAD_GRAYSCALE);
    if (img.empty()){
        std::cerr << "[Error] Image not found! Please check 'test.jpg'." << std::endl;
        return -1;
    }

    std::cout << ">> [Info] Image Size: " << img.cols << "x" << img.rows << std::endl;

    // Initialize destination matrices with zeros
    Mat dstNaive = Mat::zeros(img.size(), CV_8UC1);
    Mat dstPointer = Mat::zeros(img.size(), CV_8UC1);

    // ---------------------------------------------------------
    // 2. Benchmarking (Performance Test)
    // ---------------------------------------------------------
    int iterations = 1000; // Number of loops for average time

    // (1) Measure Naive Method
    std::cout << ">> [Step 2] Running Naive Method... \n" << std::flush;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        sobelNaive(img, dstNaive);
    }
    auto end = high_resolution_clock::now();
    auto durationNaive = duration_cast<milliseconds>(end - start).count();

    // (2) Measure Optimized (Pointer) Method
    std::cout << ">> [Step 3] Running Pointer Method... \n" << std::flush;
    start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        sobelPointer(img, dstPointer);
    }
    end = high_resolution_clock::now();
    auto durationPointer = duration_cast<milliseconds>(end - start).count();

    // ---------------------------------------------------------
    // 3. Report Results
    // ---------------------------------------------------------
    std::cout << "===== [Benchmark Result (" << iterations << " iterations)] =====" << std::endl;
    std::cout << "1. Naive Method (.at)\t: " << durationNaive << " ms" << std::endl;
    std::cout << "2. Pointer Method (Opt)\t: " << durationPointer << " ms" << std::endl;

    if (durationPointer > 0) {
        double speedup = (double)durationNaive / durationPointer;
        std::cout << ">> Speedup\t\t: " << speedup << "x faster!" << std::endl;
    } else {
        std::cout << ">> Speedup\t\t: Infinite (Too fast to measure!)" << std::endl;
    }

    // Display Results
    imshow("Original", img);
    imshow("Sobel Edge (Pointer)", dstPointer);
    // imwrite("result_sobel.jpg", dstPointer);
    // std::cout << ">> [System] Result image saved to 'result_sobel.jpg" << std::endl;

    waitKey(0);
    return 0;
}
