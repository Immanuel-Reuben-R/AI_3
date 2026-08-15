import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(16, 1, 2, stride=2))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        x4 = self.dec2(x3)
        return x4

_model = None

def is_skin_present(image_path, model_path=r'F:\UI\skin_segmentation_model.pth', threshold=0.1):
    global _model
    try:
        if _model is None:
            _model = SimpleUNet()
            _model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            _model.eval()

        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))
        tensor_img = transforms.ToTensor()(image).unsqueeze(0)

        with torch.no_grad():
            output = _model(tensor_img)
            preds = (torch.sigmoid(output) > 0.5).float()
            
        skin_ratio = preds.mean().item()
        
        # If less than 'threshold' (e.g., 10%) of the image is skin, reject it
        return skin_ratio >= threshold, skin_ratio
        
    except Exception as e:
        print(f"Skin filter error: {e}")
        return True, 1.0 # Default to true if model is missing or fails
