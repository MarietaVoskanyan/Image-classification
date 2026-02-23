# Image-classification
# Bird Species Classification

This project implements a deep learning pipeline for classifying bird species using images. It leverages **pre-trained VGG16 and GoogLeNet models** with custom fully connected layers for transfer learning.

## Features
- Fine-tuning of pre-trained CNNs (VGG16, GoogLeNet) on a **25-class bird dataset**.
- Training with **frozen convolutional layers** and custom classifier heads.
- Use of **AdamW optimizer**, **label smoothing**, and **learning rate scheduling**.
- Model saving/loading with **class-to-index mapping** for single-image inference.
- Visualization of **training/validation accuracy and loss**.

## Usage
```python
from PIL import Image
from torchvision import transforms
import torch

# Load model
checkpoint = torch.load("vgg16_birds_updated.pth", map_location="cpu")
model = ...  # rebuild architecture
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict a new image
img = Image.open("sample_img.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
x = transform(img).unsqueeze(0)
with torch.no_grad():
    outputs = model(x)
    pred_idx = outputs.argmax(1).item()

print("Predicted bird:", {v: k for k, v in checkpoint["class_to_idx"].items()}[pred_idx])
