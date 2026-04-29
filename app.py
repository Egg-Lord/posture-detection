import cv2
import numpy as np
import tensorflow as tf

# Load SavedModel
loaded = tf.saved_model.load("saved_model_posture")
infer = loaded.signatures["serving_default"]

# Make sure this matches your class_indices from training
class_names = ["Leaning_Back", "Proper", "slouch"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Bigger crop (matches dataset better)
    crop = frame[
        int(h * 0.10):int(h * 0.95),
        int(w * 0.15):int(w * 0.85)
    ]

    # Preprocess
    img = cv2.resize(crop, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    input_tensor = tf.convert_to_tensor(img_array)
    prediction = infer(input_tensor)["output_0"].numpy()

    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    text = f"{pred_class} ({confidence * 100:.1f}%)"

    # Put text on cropped image
    cv2.putText(crop, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show ONLY cropped view
    cv2.imshow("Posture Detection", crop)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()