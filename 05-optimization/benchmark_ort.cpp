#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>   // For high-resolution timing
#include <sys/resource.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // Standard ONNX Runtime Header

// ImageNet Normalization Constants
const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

// [Helper Function] Get Peak RAM Usage using getrusage
// Returns the maximum resident set size used (in MB)
double get_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in kilobytes on Linux
        return usage.ru_maxrss / 1024.0;
    }
    return -1.0;
}

int main() {
    // -------------------------------------------------------
    // 0. Configuration
    // -------------------------------------------------------
    std::string model_path = "../../models/resnet18.onnx";
    std::string image_path = "../../assets/test_image.jpg";
    const int NUM_WARMUP = 10;  // Number of warm-up runs (not timed)
    const int NUM_LOOPS = 100;  // Number of benchmark runs

    double mem_baseline = get_memory_usage();

    // -------------------------------------------------------
    // 1. Initialize ONNX Runtime
    // -------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNet18_Benchmark");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(0); // Set number of threads
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "[Info] Loading model form " << model_path << "..." << std::endl;
    Ort::Session session(env, model_path.c_str(), session_options);

    double mem_after_load = get_memory_usage();

    // -------------------------------------------------------
    // 2. Prepare Input (Pre-allocate memory)
    // -------------------------------------------------------
    Ort::AllocatorWithDefaultOptions allocator;

    // Get Input/Output Names
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    // Load and Preprocess Image ONCE
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[Error] Could not read image: " << image_path << std::endl;
        // Create a dummy black image if file not found
        img = cv::Mat::zeros(224, 224, CV_8UC3);
    }

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);

    // Prepare Tensor Data (HWC -> CHW)
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);
    float* img_data = (float*)resized_img.data;

    // Loop to reorder data from HWC to CHW and apply Normalization
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                float pixel_value = img_data[h * 224 * 3 + w * 3 + c];
                pixel_value = ((pixel_value - MEAN[c]) / STD[c]);
                input_tensor_values[c * 224 * 224 + h * 224 + w] = pixel_value;
            }
        }
    }

    // Create ONNX Runtime Tensor
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    // -------------------------------------------------------
    // 3. Warm-up Phase
    // -------------------------------------------------------
    std::cout << "[Info] Starting Warm-up (" << NUM_WARMUP << " runs)..." << std::endl;
    for (int i = 0; i < NUM_WARMUP; i++) {
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }
    
    // -------------------------------------------------------
    // 4. Benchmark Loop
    // -------------------------------------------------------
    std::cout << "[Info] Starting Benchmark (" << NUM_LOOPS << " runs)..." << std::endl;
    std::vector<double> inference_times;
    inference_times.reserve(NUM_LOOPS);

    for (int i = 0; i < NUM_LOOPS; i++) {
        // Start Timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run Inference
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Stop Timer
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate duration in milliseconds
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        inference_times.push_back(duration.count());
    }

    double mem_final_peak = get_memory_usage();

    // -------------------------------------------------------
    // 5. Calculate Statistics
    // -------------------------------------------------------
    double sum = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
    double avg = sum / NUM_LOOPS;
    double min_val = *std::min_element(inference_times.begin(), inference_times.end());
    double max_val = *std::max_element(inference_times.begin(), inference_times.end());
    double fps = 1000.0 / avg;

    // -------------------------------------------------------
    // 6. Print Report
    // -------------------------------------------------------
    std::cout << "\n=============================================" << std::endl;
    std::cout << "         BENCHMARK REPORT (ResNet18)         " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << " Model       : " << model_path << std::endl;
    std::cout << " Device      : CPU" << std::endl;
    std::cout << " Loops       : " << NUM_LOOPS << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << " Avg Latency : " << avg << " ms" << std::endl;
    std::cout << " Min Latency : " << min_val << " ms" << std::endl;
    std::cout << " Max Latency : " << max_val << " ms" << std::endl;
    std::cout << " Throughput  : " << fps << " FPS" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << " [Peak Memory Usage]" << std::endl;
    std::cout << " Start Peak: : " << mem_baseline << " MB" << std::endl;
    std::cout << " Load Peak   : " << mem_after_load << " MB" << std::endl;
    std::cout << " Final Peak: : " << mem_final_peak << " MB" << std::endl;
    std::cout << "=============================================" << std::endl; 

    return 0;
}
