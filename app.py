import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.title("Live Posture Detection")

loaded = tf.saved_model.load("saved_model_posture")
infer = loaded.signatures["serving_default"]

class_names = ["Leaning_Back", "Proper", "slouch"]

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class PostureProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w, _ = img.shape

        crop = img[
            int(h * 0.10):int(h * 0.95),
            int(w * 0.15):int(w * 0.85)
        ]

        resized = cv2.resize(crop, (224, 224))
        input_arr = np.expand_dims(resized, axis=0)
        input_arr = tf.keras.applications.mobilenet_v2.preprocess_input(input_arr)

        prediction = infer(tf.convert_to_tensor(input_arr))["output_0"].numpy()

        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        text = f"{pred_class} ({confidence * 100:.1f}%)"

        cv2.putText(
            crop,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(crop, format="bgr24")

webrtc_streamer(
    key="posture-detection",
    video_processor_factory=PostureProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False}
)