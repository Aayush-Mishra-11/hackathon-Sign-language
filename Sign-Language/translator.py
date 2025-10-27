import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from dataset_handler import load_dataset, load_dataset_with_folders
from camera_handler import capture_video
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

X_train, Y_train, X_test, Y_test = load_dataset()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

class_labels = [chr(i) for i in range(65, 91)]

def process_frame(frame):
    """Process a video frame and display the prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28)).flatten() / 255.0
    prediction = model.predict([resized])[0]
    predicted_character = class_labels[prediction]

    cv2.putText(frame, f"Prediction: {predicted_character}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Translator', frame)

def train_model():
    """Train the model using the enhanced dataset."""
    X_train, Y_train, X_test, Y_test = load_dataset_with_folders()

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    Y_train = to_categorical(Y_train, num_classes=26)
    Y_test = to_categorical(Y_test, num_classes=26)

    #cnn
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')
    ])

    #compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

    #save
    model.save('sign_language_model.h5')

    print("Model training complete and saved as 'sign_language_model.h5'.")

if __name__ == "__main__":
    capture_video(process_frame)