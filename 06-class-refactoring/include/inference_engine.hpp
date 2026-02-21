#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Struct to hold the final prediction result
struct InferenceResult {
    int class_id;
    std::string class_name;
    float confidence;
};

class InferenceEngine {
public:
    // Constructor: Initializes the ONNX Runtime environment and loads the model
    InferenceEngine(const std::string& model_path, const std::string& classes_path);

    // Main method to run inference on a single image
    InferenceResult predict(const cv::Mat& input_image);

private:
    // ONNX Runtime Core Objects (Kept alive during the entire object lifecycle)
    Ort::Env env_{nullptr};
    Ort::SessionOptions session_options_{nullptr};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    // I/O Tensor Information
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_{1, 3, 224, 224};

    // Class Names Mapping
    std::vector<std::string> class_names_;

    // Internal helper methods
    void load_class_names(const std::string& path);
    std::vector<float> preprocess(const cv::Mat& input_image);
};