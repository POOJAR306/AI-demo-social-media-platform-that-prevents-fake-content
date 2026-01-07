import torch, timm
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path

# -------------------------
# Device setup → choose GPU if available else CPU
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load EfficientNet model for video frame classification
# -------------------------
MODEL_PATH = Path("models/video_fake_real_final.pth")  # trained model path
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)  # EfficientNet B0
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # load saved weights
model.to(device)  # move to device
model.eval()  # set to evaluation mode

# -------------------------
# Transform for frames → resize, tensor, normalize
# -------------------------
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Extract frames from video
# -------------------------
def extract_frames(video_path, fps=1):
    """Extract frames at a specific fps"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # get video FPS
    frame_interval = max(int(video_fps / fps), 1)  # select frames based on desired FPS
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR→RGB
            frames.append(Image.fromarray(frame_rgb))  # store as PIL Image
        count += 1
    cap.release()
    return frames  # return list of frames

# -------------------------
# Predict if video is Real or Fake
# -------------------------
def detect_video(video_path, fps=1):
    frames = extract_frames(video_path, fps=fps)  # extract frames
    if len(frames) == 0:
        return "Error: No frames extracted"
    
    preds = []
    for img in frames:
        x = val_tf(img).unsqueeze(0).to(device)  # transform + add batch dim
        with torch.no_grad():  # inference
            output = model(x)
            label = torch.argmax(output, 1).item()  # 0=fake, 1=real
        preds.append(label)
    
    # Majority voting → most frequent label = video label
    video_label = max(set(preds), key=preds.count)
    return "Real" if video_label == 1 else "Fake"
