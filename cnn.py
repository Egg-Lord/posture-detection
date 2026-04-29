import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("final_custom_cnn_posture.keras")

class_names = ["Leaning_back", "Proper", "slouch"]  # match your class_indices

st.title("Custom CNN Posture Detection")
st.write("Upload an image to classify posture using the custom CNN model.")

uploaded_file = st.file_uploader(
    "Upload posture image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img)

    h, w, _ = img.shape

    crop = img[
        int(h * 0.10):int(h * 0.95),
        int(w * 0.15):int(w * 0.85)
    ]

    resized = cv2.resize(crop, (224, 224))
    img_array = resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(crop, caption="Cropped Input", use_container_width=True)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.write("Class Probabilities:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"{cls}: {prob * 100:.2f}%")