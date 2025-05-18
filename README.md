# Emotion Detection from Images

## Overview

This project is an **Emotion Detection System** built using **Python, PyTorch, OpenCV, and Streamlit**. It enables users to upload an image through a web interface and accurately detect the facial emotion present using a Convolutional Neural Network (CNN). The project combines computer vision, deep learning, and web development to deliver an intuitive, real-time emotion classification tool.

## Features

* **Image Upload Interface**: Streamlit-based UI that restricts uploads to images only (JPG/PNG, <5MB).
* **Real-Time Emotion Prediction**: Predicts emotion from the uploaded facial image using a trained CNN model.
* **Face Detection Support**: Uses pretrained models (OpenCV/Dlib/Mediapipe) to detect and crop faces.
* **Facial Landmark Extraction**: Extracts facial features using Dlib or Mediapipe.
* **Model Evaluation**: Evaluates model performance with accuracy, precision, recall, and F1 score.
* **Preprocessing Pipeline**: All images are resized, normalized, and converted to grayscale for training/testing.

## Technologies Used

* Python
* PyTorch
* torchvision
* Streamlit
* OpenCV
* PIL
* Dlib / Mediapipe (for facial landmarks)

## Dataset

* **FER-2013**: Facial Expression Recognition dataset from Kaggle
* Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* Dataset Structure:

  * `train/`: Training images categorized in emotion folders
  * `test/`: Testing images categorized similarly

## Prerequisites

Ensure you have the following installed:

* Python (>=3.9)
* pip
* Required Python libraries:

```bash
pip install torch torchvision opencv-python mediapipe dlib streamlit
```

## Usage

1. Train the model:

   * Run the `cnn_emotion_training.py` script to train and save your CNN model.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Upload a face image.
4. View the predicted emotion.

## CNN Model Architecture

* 3 Convolutional layers with ReLU + MaxPooling
* 2 Fully connected layers with Dropout
* Input size: 48x48 grayscale image
* Output: 7 emotion classes

## Directory Structure

```
├── train/                 # Training data organized by emotion
├── test/                  # Test data organized by emotion
├── cnn_emotion_training.py
├── app.py                 # Streamlit app
├── emotion_cnn_model.pth  # Trained model
```

## Contribution

Feel free to contribute by forking the repository and submitting pull requests.

## License

This project is licensed under the MIT License.

## Author

Ramadevi N
