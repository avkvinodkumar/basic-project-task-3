# gender_age_prediction.py

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pandas as pd

# Load the dataset
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data, test_data = data.split(test_size=0.2, random_state=42)

# Define the face detection function
def detect_faces(img):
    faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Define the face alignment function
def align_face(img, face):
    x, y, w, h = face
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (256, 256))
    return face_img

# Define the feature extraction function
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# Define the model training function
def train_model(train_data):
    X_train = []
    y_train = []
    for img in train_data['image']:
        faces = detect_faces(img)
        for face in faces:
            face_img = align_face(img, face)
            features = extract_features(face_img)
            X_train.append(features.flatten())
            y_train.append(train_data['label'][train_data['image'] == img].values[0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Define the model evaluation function
def evaluate_model(model, test_data):
    X_test = []
    y_test = []
    for img in test_data['image']:
        faces = detect_faces(img)
        for face in faces:
            face_img = align_face(img, face)
            features = extract_features(face_img)
            X_test.append(features.flatten())
            y_test.append(test_data['label'][test_data['image'] == img].values[0])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test))
    matrix = confusion_matrix(y_test, model.predict(X_test))
    return accuracy, report, matrix

# Train the model
model = train_model(train_data)

# Evaluate the model
accuracy, report, matrix = evaluate_model(model, test_data)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the saved model and make predictions on new data
with open('model.pkl', 'rb') as f:
    model.load(f)

new_img = cv2.imread('new_image.jpg')
faces = detect_faces(new_img)
for face in faces:
    face_img = align_face(new_img, face)
    features = extract_features(face_img)
    prediction = model.predict(features.flatten())
    print("Prediction:", prediction)

cv2.destroyAllWindows()
