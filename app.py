import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = 'C:/Users/nabajeeban panda/Downloads/plant_disease_model_optimized.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Example: Replace this with the actual class names used in your model
class_names = ["Apple___Apple_scab",
  "Apple___Black_rot",
 "Apple___Cedar_apple_rust",
  "Apple___healthy",
 "Blueberry___healthy",
  "Cherry_(including_sour)___Powdery_mildew",
 "Cherry_(including_sour)___healthy",
 "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
 "Corn_(maize)___Common_rust_",
 "Corn_(maize)___Northern_Leaf_Blight",
 "Corn_(maize)___healthy",
 "Grape___Black_rot",
 "Grape___Esca_(Black_Measles)",
 "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
 "Grape___healthy",
 "Orange___Haunglongbing_(Citrus_greening)",
 "Peach___Bacterial_spot",
 "Peach___healthy",
 "Pepper,_bell___Bacterial_spot",
 "Pepper,_bell___healthy",
 "Potato___Early_blight",
 "Potato___Late_blight",
 "Potato___healthy",
 "Raspberry___healthy",
 "Soybean___healthy",
 "Squash___Powdery_mildew",
 "Strawberry___Leaf_scorch",
 "Strawberry___healthy",
 "Tomato___Bacterial_spot",
 "Tomato___Early_blight",
 "Tomato___Late_blight",
 "Tomato___Leaf_Mold",
 "Tomato___Septoria_leaf_spot",
 "Tomato___Spider_mites Two-spotted_spider_mite",
 "Tomato___Target_Spot",
 "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
 "Tomato___Tomato_mosaic_virus",
 "Tomato___healthy"]

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size expected by your model
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to make predictions
def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit web page setup
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to predict the disease.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction
    prediction = predict(image, model)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Print class name
    st.write(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
