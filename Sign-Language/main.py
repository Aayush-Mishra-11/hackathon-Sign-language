from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from translator import process_frame
from camera_handler import capture_video

app = Flask(__name__)

model_path = "sign_language_model.h5"
sign_language_model = load_model(model_path)

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    img = cv2.resize(img, (28, 28)) 
    img = img.reshape(28, 28, 1).astype('float32') / 255.0
    return img

def predict_gesture(frame):
    preprocessed = preprocess_frame(frame)
    input_tensor = np.expand_dims(preprocessed, axis=0) 
    predictions = sign_language_model.predict(input_tensor)
    predicted_label = np.argmax(predictions, axis=1)[0]
    return predicted_label

def start_system():
    print("Starting the Sign Language Translator System...")
    capture_video(process_frame)

if __name__ == "__main__":
    start_system()