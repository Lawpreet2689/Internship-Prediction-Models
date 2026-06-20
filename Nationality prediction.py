import cv2
import numpy as np
import gradio as gr
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def detect_nationality(img_path):
    """Predicts nationality from image."""
    result = DeepFace.analyze(img_path, actions=["race"], enforce_detection=False)
    return result[0]["dominant_race"].capitalize()

def detect_emotion(img_path):
    """Predicts emotion from image."""
    result = DeepFace.analyze(img_path, actions=["emotion"], enforce_detection=False)
    return result[0]["dominant_emotion"].capitalize()

def predict_age(img_path):
    """Predicts age if nationality is Indian or US."""
    result = DeepFace.analyze(img_path, actions=["age"], enforce_detection=False)
    return result[0]["age"]

def extract_dress_color(img_path):
    """Extracts the most dominant dress color from the image."""
    img = cv2.imread(img_path)
    if img is None:
        return "Error: Could not read image"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(img)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return f"RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"

def process_image(img_path):
    """Main function to process image and return predictions."""
    try:
        nationality = detect_nationality(img_path)
        emotion = detect_emotion(img_path)
        age = None
        dress_color = None

        if nationality in ["Indian", "US"]:
            age = predict_age(img_path)

        if nationality in ["Indian", "African"]:
            dress_color = extract_dress_color(img_path)

        return nationality, emotion, age, dress_color

    except Exception as e:
        return f"Error: {str(e)}", None, None, None

def gui_interface(image):
    """Gradio interface for image upload and results display."""
    nationality, emotion, age, dress_color = process_image(image)
    output_text = f"Nationality: {nationality}\nEmotion: {emotion}"
    if age:
        output_text += f"\nAge: {age}"
    if dress_color:
        output_text += f"\nDress Color: {dress_color}"
    return output_text

demo = gr.Interface(fn=gui_interface, inputs=gr.Image(type="filepath"), outputs="text")
demo.launch()
