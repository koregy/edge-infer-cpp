#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace cv::dnn;

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

int main() {
    // -------------------------------------------------------
    // 1. Configuration & Paths
    // -------------------------------------------------------
    std::string model_path = "../../models/resnet18.onnx"; // Path to the ONNX model
    std::string image_path = "../../assets/test_image.jpg";   // Input image
    std::string class_file = "../imagenet_classes.txt";    // Class name file

    // -------------------------------------------------------
    // 2. Load the Network
    // -------------------------------------------------------
    std::cout << "[System] Loading ONNX model from: " << model_path << std::endl;

    // readNetFromONNX automatically parses the ONNX graph structure
    Net net = readNetFromONNX(model_path);

    if (net.empty()) {
        std::cerr << "[Error] Failed to load the network! Check the file path." << std::endl;
        return -1;
    }

    // [Optimization] Set backend to OpenCV (CPU default)
    net.setPreferableBackend(DNN_BACKEND_OPENCV);   // for GPU: DNN_BACKEND_CUDA
    net.setPreferableTarget(DNN_TARGET_CPU);

    // -------------------------------------------------------
    // 3. Prepare Input Data (Preprocessing)
    // -------------------------------------------------------
    Mat image = imread(image_path);
    if (image.empty()) {
        std::cerr << "[Error] Could not read the image: " << image_path << std::endl;
        return -1;
    }

    // Create a 4D Blob from the image (Bath, Channel, Height, Width)
    // ResNet18 expects: 224x224 input size
    // Mean subtraction: (0.485, 0.456, 0.406) * 255 -> commonly approximated as (123.675, 116.28, 103.53) for OpenCV
    // SwapRB: OpenCV uses BGR, but PyTorch models are trained on RGB. So set swapRB=true
    Mat blob;
    blobFromImage(image, blob, 1.0 / 255.0, Size(224, 224), Scalar(0, 0, 0), true, false);

    // Set the blob as input to the network
    net.setInput(blob);

    // -------------------------------------------------------
    // 4. Inference (Forward Pass) & Latency Measurement
    // -------------------------------------------------------
    std::cout << "[System] Running Inference..." << std::endl;

    // TickMeter is useful for measuring execution time accurately
    TickMeter tm;
    tm.start();

    // The actual "Forward Pass"
    Mat output = net.forward();

    tm.stop();
    std::cout << "[Result] Inference Time: " << tm.getTimeMilli() << " ms" << std::endl;

    // -------------------------------------------------------
    // 5. Post-processing (Interpret the Result)
    // -------------------------------------------------------
    // The output is a 1x1000 matrix (probabilities/logits for each class)
    Point classIdPoint;
    double confidence;

    // Find the index with the maximum value (Best match)
    minMaxLoc(output.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    // Load class names to print the human-readable label
    std::vector<std::string> classes = load_class_names(class_file);
    std::string label = format("%s (Confidence: %.2f)", classes[classId].c_str(), confidence);

    std::cout << "[Result] Prediction: " << label << std::endl;

    // Visualization
    putText(image, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    // Save the result
    std::string output_file = "../../assets/result_inference.jpg";
    imwrite(output_file, image);
    std::cout << "[System] Output saved to " << output_file << std::endl;

    return 0;
}