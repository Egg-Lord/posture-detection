import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load model
loaded = tf.saved_model.load("saved_model_posture")
infer = loaded.signatures["serving_default"]

class_names = ["Leaning_Back", "Proper", "slouch"]

st.title("Posture Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img = np.array(img)

    h, w, _ = img.shape

    # Crop
    crop = img[
        int(h * 0.10):int(h * 0.95),
        int(w * 0.15):int(w * 0.85)
    ]

    # Preprocess
    resized = cv2.resize(crop, (224, 224))
    img_array = np.expand_dims(resized, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    prediction = infer(tf.convert_to_tensor(img_array))["output_0"].numpy()

    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(crop, caption="Cropped Input", use_column_width=True)
    st.write(f"Prediction: **{pred_class} ({confidence*100:.1f}%)**")