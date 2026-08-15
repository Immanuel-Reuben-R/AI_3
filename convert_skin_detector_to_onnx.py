import torch
import torchvision.models as models
import torch.nn as nn
import os

def convert_to_onnx():
    print("Converting Skin Detector Model to ONNX...")
    model_path = r'F:\UI\skin_detector_model.pth'
    if not os.path.exists(model_path):
        print(f"File {model_path} not found.")
        return

    # Must match the architecture from training
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = r'F:\UI\skin_detector_model.onnx'
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Saved: {onnx_path}")

if __name__ == '__main__':
    convert_to_onnx()
