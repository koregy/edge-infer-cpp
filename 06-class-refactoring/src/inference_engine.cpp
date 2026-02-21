#include "inference_engine.hpp"
#include <fstream>
#include <iostream>

// ImageNet Normalization Constants
const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& classes_path) {
    // 1. Initialize ORT Environment
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ResNet18_Inference");
    
    // 2. Configure Session Options
    session_options_ = Ort::SessionOptions();
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3. Load Model (Executed only ONCE)
    std::cout << "[Info] Loading ONNX model from: " << model_path << std::endl;
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    // 4. Extract Input and Output Node Names safely
    Ort::AllocatorWithDefaultOptions allocator;
    
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    input_name_ = input_name_ptr.get(); // Deep copy to std::string

    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    output_name_ = output_name_ptr.get();

    // 5. Create CPU Memory Info
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 6. Load Class Labels
    load_class_names(classes_path);
}

void InferenceEngine::load_class_names(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "[Error] Could not open class file: " << path << std::endl;
        exit(1);
    }
    while (std::getline(file, line)) {
        class_names_.push_back(line);
    }
    std::cout << "[Info] Loaded " << class_names_.size() << " class names." << std::endl;
}

std::vector<float> InferenceEngine::preprocess(const cv::Mat& input_image) {
    cv::Mat resized_img;
    cv::resize(input_image, resized_img, cv::Size(224, 224));
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);

    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);
    float* img_data = (float*)resized_img.data;

    // Convert HWC to CHW & Apply Normalization
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                float pixel_value = img_data[h * 224 * 3 + w * 3 + c];
                pixel_value = (pixel_value - MEAN[c]) / STD[c];
                input_tensor_values[c * 224 * 224 + h * 224 + w] = pixel_value;
            }
        }
    }
    return input_tensor_values;
}

InferenceResult InferenceEngine::predict(const cv::Mat& input_image) {
    // 1. Preprocess Image
    std::vector<float> input_tensor_values = preprocess(input_image);

    // 2. Create ONNX Tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape_.data(),
        input_shape_.size()
    );

    const char* input_names_arr[] = {input_name_.c_str()};
    const char* output_names_arr[] = {output_name_.c_str()};

    // 3. Run Inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_arr,
        &input_tensor,
        1,
        output_names_arr,
        1
    );

    // 4. Postprocess (ArgMax)
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();
    
    int pred_id = 0;
    float max_score = -10000.0f;

    for (int i = 0; i < 1000; i++) {
        if (output_arr[i] > max_score) {
            max_score = output_arr[i];
            pred_id = i;
        }
    }

    // 5. Wrap up results
    InferenceResult result;
    result.class_id = pred_id;
    result.confidence = max_score;
    if (pred_id < class_names_.size()) {
        result.class_name = class_names_[pred_id];
    } else {
        result.class_name = "Unknown";
    }

    return result;
}