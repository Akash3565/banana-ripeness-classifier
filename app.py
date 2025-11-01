import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("banana_ripeness_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

st.title("🍌 Banana Ripeness Classifier")
uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

IMG_SIZE = (224, 224)
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)
    x = preprocess(image)
    preds = model.predict(x)
    pred_idx = int(np.argmax(preds[0]))
    pred_label = class_names[pred_idx]
    confidence = float(np.max(preds[0]))
    st.subheader(f"Prediction: {pred_label}")
    st.write(f"Confidence: {confidence:.2f}")
    st.write("Class probabilities:")
    for cls, prob in zip(class_names, preds[0]):
        st.write(f"- {cls}: {prob:.2f}")
