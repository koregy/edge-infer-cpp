import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os

def export_resnet18():
    """
    Export PyTorch ResNet18 model to ONNX format
    This script prepares the model for C++ inference engines (e.g., OpenCV DNN, TensorRT)
    """

    # 1. Setup Output Directory
    # Define where to save the converted model
    output_dir = "../models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[Info] Created output directory: {output_dir}")

    onnx_file_path = os.path.join(output_dir, "resnet18.onnx")

    # 2. Load Pre-trained Model
    print("[Info] Loading pre-trained ResNet18 model...")
    # Using ResNet18 as a baseline due to its standard architecture (Residual Blocks)
    # 'pretrained=True' is deprecated. Using 'weights' parameter instead
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(pretrained=True)

    # Switch to Inference Mode
    # This disables training-specific layers like Dropout and fixes Batch Normalization statistics
    # Without this, the exported model will produce incorrect results
    model.eval()

    # 3. Create Dummy Input
    # Generate a random input tensor to trace the model's execution graph
    # Format: (Batch_Size, Channels, Height, Width) -> Standard ImageNet shape
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. Export to ONNX
    print(f"[Info] Exporting model to {onnx_file_path}...")

    torch.onnx.export(
        model,                          # Model to be exported
        dummy_input,                    # Input tensor for tracing
        onnx_file_path,                 # Output file path
        export_params=True,             # Store the trained parameter weights inside the model file
        opset_version=11,               # ONNX Opset Version 11 (Highly compatible with OpenCV DNN / TensorRT)
        do_constant_folding=True,       # Optimization: Execute constant nodes ahead of time to simplify the graph
        input_names=['input'],          # Name of the input node (Used in C++ for binding)
        output_names=['output'],        # Name of the output node (Used in C++ for retrieval)
        dynamic_axes={                  # Allow variable batch sizes (e.g., processing 1 or 8 images at once)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"[Success] Model saved to {onnx_file_path}")

if __name__ == "__main__":
    export_resnet18()