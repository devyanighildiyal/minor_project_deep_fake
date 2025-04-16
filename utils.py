import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load your hybrid model
model = load_model("hybrid_deepfake.h5")

# Modify this according to your hybrid model’s input requirements
def preprocess_image(image):
    img = image.resize((224, 224))  # Change based on model
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_deepfake(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    return label, round(confidence * 100, 2)
