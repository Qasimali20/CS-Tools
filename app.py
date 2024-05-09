import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

# Load the pre-trained model
model = tf.keras.models.load_model('malaria_cell.h5')

# Set the app's title and description
st.title("Malaria Cell Classification")
st.write("Upload an image of a cell to determine if it is infected with malaria.")

# File uploader to upload a cell image
uploaded_file = st.file_uploader("Upload a cell image", type=["png", "jpg", "jpeg"])

# Create a section for displaying results and progress
if uploaded_file:
    # Show a loading spinner while processing the image
    with st.spinner("Processing the image..."):
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=32)

        # Preprocess the image to match the model's input
        image_resized = image.resize((64, 64))  # Resize to model input size
        image_array = np.array(image_resized)  # Convert to numpy array
        image_array = image_array.reshape((1, 64, 64, 3))  # Reshape for model

        # Predict whether the cell is infected or not
        prediction = model.predict(image_array)
        predicted_class = "Infected" if prediction[0][0] > 0.5 else "Uninfected"

    # Display the prediction result with a custom message
    st.success(f"The cell is: {predicted_class}")

    # Include additional details for context
    if predicted_class == "Infected":
        st.warning("This cell is predicted to be infected with malaria.")
    else:
        st.info("This cell is predicted to be uninfected.")

else:
    st.info("Please upload an image to get started.")
