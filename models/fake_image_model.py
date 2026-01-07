import torch
import timm  # EfficientNet library
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os

# ===== CONFIG =====
MODEL_PATH = "fake_real_efficientnet_best.pth"  # trained model file
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']   # allowed image types

# ===== DEVICE =====
device = torch.device("cpu")  # CPU inference
print("Using device:", device)

# ===== LOAD MODEL SAFELY =====
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print("✅ Model loaded successfully")

except (RuntimeError, FileNotFoundError) as e:
    print(f"⚠️ Error loading model: {e}")
    print("🚫 Please re-download or regenerate 'fake_real_efficientnet_best.pth' before running the app.")
    model = None  # prevent crash if model is unavailable

# ===== IMAGE TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize image
    transforms.ToTensor(),          # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # normalize
])

# ===== SINGLE IMAGE PREDICTION FROM PATH =====
def predict_image(image_path):
    """Predict single image from file path"""
    if model is None:
        return {"error": "Model not loaded. Please provide a valid model file."}

    image = Image.open(image_path).convert("RGB")
    inp = transform(image).unsqueeze(0).to(device)
    return predict_image_from_tensor(inp, os.path.basename(image_path))

# ===== SINGLE IMAGE PREDICTION FROM NUMPY ARRAY =====
def predict_image_from_array(np_array):
    """Predict single image from numpy array"""
    if model is None:
        return {"error": "Model not loaded. Please provide a valid model file."}

    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
    inp = transform(image).unsqueeze(0).to(device)
    return predict_image_from_tensor(inp, "from_array")

# ===== HELPER FUNCTION =====
def predict_image_from_tensor(inp, filename="image"):
    """Get prediction from input tensor"""
    if model is None:
        return {"error": "Model not loaded. Please provide a valid model file."}

    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][pred])
        label = "Real" if pred == 1 else "Fake"

    return {
        "filename": filename,
        "fake": True if label == "Fake" else False,
        "label": label,
        "confidence": confidence,
        "reason": f"Model classified this image as {label}"
    }

# ===== MULTIPLE IMAGE PREDICTION =====
def predict_folder(folder_path):
    """Predict all images in a folder"""
    results = []
    if model is None:
        print("🚫 Model not loaded. Please fix the model file before prediction.")
        return results

    image_files = [f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print(f"No images found in {folder_path}")
        return results

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        result = predict_image(img_path)
        results.append(result)
    return results

# ===== TEST BLOCK =====
if __name__ == "__main__":
    test_folder = "test_images"  # folder with test images
    results = predict_folder(test_folder)
    for r in results:
        print(r)
