import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the same CNN model architecture used during training
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Emotion labels 
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model
model = EmotionCNN(num_classes=len(class_labels)).to(DEVICE)
model.load_state_dict(torch.load('emotion_cnn_model.pth', map_location=DEVICE))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit App
st.set_page_config(page_title="Emotion Detection App")
st.title("Emotion Detection from Image")

st.markdown("""
Upload a **face image** to detect the **emotion**. Supported formats: JPG, PNG. Max size: 5MB.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check size limit (e.g., 5MB)
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large. Please upload an image smaller than 5MB.")
    else:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            emotion = class_labels[predicted.item()]

        st.success(f"Predicted Emotion: **{emotion}**")
