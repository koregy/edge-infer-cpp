/*
Understanding Memory Layout & Pointer Arithmetic in C++
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    // -------------------------------------------------
    // 1. Memory Allocation
    // -------------------------------------------------
    // Create an image with width 640 and height 480
    // CV_8UC3: 8-bit unsigned char x 3 Channels (Color)
    // Initial Color: Blue (255, 0, 0) - Note: OpeCV uses BGR order
    int width = 640;
    int height = 480;
    Mat img(height, width, CV_8UC3, Scalar(255, 0, 0));
 
    // -------------------------------------------------
    // 2. Get Raw Pointer (Direct Memory Access)
    // -------------------------------------------------
    // 'img.data' points to the starting memory address of the image data
    // [Key for On-Device Optimization]
    // Access memory directly to avoid the overhead of function calls (like at<Vec3b)
    uchar* data = img.data;

    // -------------------------------------------------
    // 3. Rendering Logic - Draw a Red Cross
    // -------------------------------------------------
    int centerX = width / 2;
    int centerY = height / 2;
    int crossSize = 100; // Length of the cross arms

    // (1) Draw Horizontal Line
    for (int x = centerX - crossSize; x < centerX + crossSize; x++) {
        // [Address Calculation Formula]
        // Index = (Row * Width + Column) * Channels
        // Multiplied by 3 because one pixel occupies 3 bytes (BGR)
        int index = (centerY * width + x) * 3;

        // Direct Memory Write (BGR Order)
        data[index] = 0;        // Blue: 0
        data[index + 1] = 0;    // Green: 0
        data[index + 2] = 255;  // Red: 255
    }

    // (2) Draw Vertical Line
    for (int y = centerY - crossSize; y < centerY + crossSize; y++) {
        // Calculate the memory index (x is fixed, y varies)
        int index = (y * width + centerX) * 3;

        data[index] = 0;
        data[index + 1] = 0;
        data[index + 2] = 255;
    }
    
    // -------------------------------------------------
    // 4. Output Result
    // -------------------------------------------------
    std::cout << "[System] Rendering Complete via Pointer Access." << std::endl;
    imshow("Day 1: Direct Memory Access", img);

    // Wait for a key press (Prevents the window from closing immediately)
    waitKey(0);

    return 0;
}
