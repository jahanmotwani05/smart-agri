import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import tensorflow as tf  # Ensure TensorFlow is imported

# Load the trained model using caching
@st.cache_resource
def load_model_cache():
    # Load the model without specifying custom_objects
    return load_model('plant_disease_model.h5')

# Use the cached model
model = load_model_cache()

# Define the class names
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato_Early_blight',
    'Tomato_Late_blight', 'Tomato_healthy'
]

st.title('Plant Disease Detection')

# Upload the file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_img(io.BytesIO(uploaded_file.read()), target_size=(256, 256))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to array and preprocess
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    # Show the prediction result
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")

st.write("Upload an image to get started!")
