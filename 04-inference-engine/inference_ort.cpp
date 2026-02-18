#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // Standard ONNX Runtime Header
#include <fstream>
#include <string>

// [Helper Function] Load class names from text file
std::vector<std::string> load_class_names(const std::string& filename) {
    std::vector<std::string> classes;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "[Error] Could not open class file: " << filename << std::endl;
        exit(1);
    }
    while (std::getline(file,line)) {
        classes.push_back(line);
    }
    return classes;
}

// ImageNet Normalization Constants
// Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

int main() {
    // Paths
    std::string model_path = "../../models/resnet18.onnx";
    std::string image_path = "../../assets/test_image.jpg";

    // -------------------------------------------------------
    // 1. Initialize ONNX Runtime Environment
    // -------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNet18_Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1); // Set number of threads
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "[Info] Loading model form " << model_path << "..." << std::endl;
    Ort::Session session(env, model_path.c_str(), session_options);

    // -------------------------------------------------------
    // 2. Prepare Input/Output Info
    // -------------------------------------------------------
    Ort::AllocatorWithDefaultOptions allocator;

    // Get Input Name (Note: The API might differ slightly based on ORT version)
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();

    // Get Output Name
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    // -------------------------------------------------------
    // 3. Preprocessing (OpenCV)
    // -------------------------------------------------------
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[Error] Could not read image: " << image_path << std::endl;
        return -1;
    }

    // Resize to model input size (224x224)
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));

    // Convert BGR (OpenCV default) to RGB
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    // Convert to Float32 and Scale to [0, 1]
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);

    // -------------------------------------------------------
    // 4. Create Input Tensor (HWC -> CHW Transformation)
    // -------------------------------------------------------
    // PyTorch models expect input in (Batch, Channel, Height, Width) format
    // OpenCV images are stored in (Height, Width, Channel) format

    size_t input_tensor_size = 1 * 3 * 224 * 224; // Batch * C * H * W
    std::vector<float> input_tensor_values(input_tensor_size);

    float* img_data = (float*)resized_img.data;

    // Loop to reorder data from HWC to CHW and apply Normalization
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                // Original Index (OpenCV): Row-Major (h, w, c)
                float pixel_value = img_data[h * 224 * 3 + w * 3 + c];

                // Apply Normalization: (value - mean) / std
                pixel_value = ((pixel_value - MEAN[c]) / STD[c]);

                // Target Index (ONNX): Planar (c, h, w)
                input_tensor_values[c * 224 * 224 + h * 224 + w] = pixel_value;
            }
        }
    }

    // Define Tensor Shape
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    // Create Memory Info for CPU
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create the ONNX Runtime Tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // -------------------------------------------------------
    // 5. Run Inference
    // -------------------------------------------------------
    std::cout << "[Info] Running inference..." << std::endl;

    const char* input_names_arr[] = {input_name};
    const char* output_names_arr[] = {output_name};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names_arr,
        &input_tensor,
        1,  // Number of inputs
        output_names_arr,
        1   // Number of outputs
    );

    // -------------------------------------------------------
    // 6. Post-processing
    // -------------------------------------------------------
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();

    // Find the class with the highest score (ArgMax)
    int pred_id = 0;
    float max_score = -10000.0f;    // Initialize with a very low value

    // Iterate through all 1000 ImageNet classes
    for (int i = 0; i < 1000; i++) {
        if (output_arr[i] > max_score) {
            max_score = output_arr[i];
            pred_id = i;
        }
    }

    std::vector<std::string> classes = load_class_names("../imagenet_classes.txt");
    
    std::cout << "[Result] Predicted Class ID: " << pred_id << std::endl;
    if (pred_id < classes.size()) {
        std::cout << "[Result] Class Name: " << classes[pred_id] << std::endl;
    }
    
    std::cout << "[Result] Confidence Score (Logit): " << max_score << std::endl;

    return 0;
}
