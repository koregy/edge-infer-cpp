#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/resource.h> // For memory profiling

using namespace cv;
using namespace cv::dnn;

// -----------------------------------------------------------
// [Helper Function] Get Current RAM Usage (RSS) in MB
// -----------------------------------------------------------
double get_memory_usage() {
    struct rusage usage;
    // getrusage returns resource usage statistics for the calling process
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in kilobytes on Linux
        return usage.ru_maxrss / 1024.0;
    }
    return -1.0;
}

// [Helper Function] Load class names
std::vector<std::string> load_class_names(const std::string& filename) {
    std::vector<std::string> classes;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "[Error] Could not open class file!" << std::endl;
        exit(1);
    }
    while (std::getline(file, line)) {
        classes.push_back(line);
    }
    return classes;
}

int main() {
    // -----------------------------------------------------------
    // 1. Configuration
    // -----------------------------------------------------------
    std::string model_path = "../../models/resnet18.onnx";
    std::string image_path = "../../assets/test_image.jpg";
    std::string class_file = "../imagenet_classes.txt";

    // Benchmark Settings
    const int WARMUP_ROUNDS = 5;    // Ignore these first runs
    const int TEST_ROUNDS = 100;    // Measure these runs

    // -----------------------------------------------------------
    // 2. Load Network & Input
    // -----------------------------------------------------------
    std::cout << "[System] Initial Memory Usage: " << get_memory_usage() << " MB" << std::endl;
    std::cout << "[System] Loading ONNX model..." << std::endl;

    Net net = readNetFromONNX(model_path);
    if (net.empty()) return -1;

    // Backend Settings (CPU)
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    std::cout << "[System] Model Loaded. Memory Usage: " << get_memory_usage() << " MB" << std::endl;
    
    Mat image = imread(image_path);
    if (image.empty()) {
        std::cerr << "[Error] Image not found" << std::endl;
        return -1;
    }

    // Preprocessingg (Blob)
    Mat blob;
    blobFromImage(image, blob, 1.0 / 255.0, Size(224, 224), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // -----------------------------------------------------------
    // 3. Warm-up Phase
    // -----------------------------------------------------------
    std::cout << "\n[Benchmark] Starting Warm-up (" << WARMUP_ROUNDS << " rounds)..." << std::endl;
    for (int i = 0; i < WARMUP_ROUNDS; i++) {
        net.forward(); // Run without measuring time
    }
    std::cout << "[Benchmark] Warm-up complete" << std::endl;

    // -----------------------------------------------------------
    // 4. Benchmarking Phase (100 Loops)
    // -----------------------------------------------------------
    std::cout << "[Benchmark] Running " << TEST_ROUNDS << " iterations..." << std::endl;

    TickMeter tm;
    double total_time = 0.0;
    double min_time = 10000.0;
    double max_time = 0.0;

    for (int i = 0; i < TEST_ROUNDS; i++) {
        tm.reset();
        tm.start();

        // --- Inference ---
        Mat output = net.forward();
        // -----------------

        tm.stop();
        double current_time = tm.getTimeMilli();

        total_time += current_time;
        if (current_time < min_time) min_time = current_time;
        if (current_time > max_time) max_time = current_time;

        // Progress check
        if ((i + 1) % 20 == 0) std::cout << "." << std::flush;
    }
    std::cout << std::endl;

    // -----------------------------------------------------------
    // 5. Report Results
    // -----------------------------------------------------------
    double avg_time = total_time / TEST_ROUNDS;
    double fps = 1000.0 / avg_time;
    double final_memory = get_memory_usage();

    std::cout << "\n=============================================" << std::endl;
    std::cout << "          Edge AI Performance Report           " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << " Model       : ResNet18 (ONNX)" << std::endl;
    std::cout << " Device      : CPU (OpenCV DNN)" << std::endl;
    std::cout << " Loops       : " << TEST_ROUNDS << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << " Avg Latency : " << avg_time << " ms" << std::endl;
    std::cout << " Min Latency : " << min_time << " ms" << std::endl;
    std::cout << " Max Latency : " << max_time << " ms" << std::endl;
    std::cout << " Throughput  : " << fps << " FPS" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << " RAM Usage   : " << final_memory << " MB" << std::endl;
    std::cout << "=============================================" << std::endl; 

    return 0;
}