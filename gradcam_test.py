import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------- Load your trained model ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))
model.eval()

# ---------- Target layer: the last conv block ----------
target_layers = [model.layer4[-1]]

# ---------- Load and preprocess an image ----------
image_path = "ima.jpg"   # <-- change this to any image you want to test
img = Image.open(image_path).convert("RGB")
img_resized = img.resize((224, 224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# ---------- Get prediction ----------
with torch.no_grad():
    prob_fake = torch.sigmoid(model(input_tensor)).item()
label = "FAKE" if prob_fake > 0.5 else "REAL"
print(f"Prediction: {label} (fake probability: {prob_fake:.4f})")

# ---------- Generate Grad-CAM heatmap ----------
cam = GradCAM(model=model, target_layers=target_layers)
# grayscale_cam = cam(input_tensor=input_tensor)[0]  # shape: (224, 224)
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

targets = [RawScoresOutputTarget()]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# ---------- Overlay heatmap on original image ----------
rgb_img = np.array(img_resized).astype(np.float32) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# ---------- Show side by side ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(img_resized)
ax1.set_title(f"Original\nPrediction: {label} ({prob_fake*100:.1f}%)")
ax1.axis("off")

ax2.imshow(visualization)
ax2.set_title("Grad-CAM Heatmap")
ax2.axis("off")

plt.tight_layout()
plt.savefig("gradcam_output.png", dpi=150)
plt.show()
print("Saved as gradcam_output.png")
