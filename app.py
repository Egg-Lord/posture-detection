import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

loaded = tf.saved_model.load("saved_model_posture")
infer = loaded.signatures["serving_default"]

class_names = ["Leaning_Back", "Proper", "slouch"]

st. ("Posture Detection")
st.write("Upload a posture image to classify it.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img)

    h, w, _ = img.shape

    crop = img[
        int(h * 0.10):int(h * 0.95),
        int(w * 0.15):int(w * 0.85)
    ]

    resized = cv2.resize(crop, (224, 224))
    img_array = np.expand_dims(resized, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = infer(tf.convert_to_tensor(img_array))["output_0"].numpy()

    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(crop, caption="Cropped Input", use_container_width=True)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.write("Class Probabilities:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"{cls}: {prob * 100:.2f}%")