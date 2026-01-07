import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path

# Path to saved model and device setup
save_path = Path(r"C:\Users\pooja\Downloads\demo_social_media_demo_fixed\models\wav2vec2_fake_real_model").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Resolved local model path: {save_path}")

# Load processor from local path
processor = Wav2Vec2Processor.from_pretrained(str(save_path), local_files_only=True)

# Try loading model normally, fix classifier if size mismatch occurs
try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        str(save_path),
        local_files_only=True,
        use_safetensors=True
    )
except RuntimeError as e:
    print("⚠️ Size mismatch detected in classifier. Reloading with manual patch...")
    
    # Load base Wav2Vec2 model without classifier
    from transformers import Wav2Vec2Model
    base_model = Wav2Vec2Model.from_pretrained(str(save_path), local_files_only=True, use_safetensors=True)
    
    # Custom classifier to match trained model
    import torch.nn as nn
    class CustomWav2Vec2Classifier(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.classifier = nn.Linear(768, 2)  # 768 = hidden size, 2 classes

        def forward(self, input_values):
            outputs = self.base(input_values).last_hidden_state
            pooled = outputs.mean(dim=1)  # average pooling
            logits = self.classifier(pooled)
            return type("Obj", (object,), {"logits": logits})

    model = CustomWav2Vec2Classifier(base_model)

# Move model to device and set to evaluation
model.to(device)
model.eval()
print("✅ Model loaded successfully and classifier fixed")

# Function to classify audio file
def classify_audio(file_path, max_duration=20):
    # Load audio and fix length
    y, sr = librosa.load(file_path, sr=16000)
    max_len = int(16000 * max_duration)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    # Process audio to input tensors
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    # Make prediction without computing gradients
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_label = "Real" if np.argmax(probs) == 0 else "Fake"
        confidence = float(probs[np.argmax(probs)] * 100)

    # Return result as dictionary
    return {
        "label": pred_label,
        "confidence": confidence
    }
