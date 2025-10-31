# banana-ripeness-classifier
Banana ripeness classification project using TensorFlow/Keras with MobileNetV2. Trained on Roboflow dataset (6 classes: freshripe, freshunripe, ripe, unripe, overripe, rotten). Includes end‑to‑end pipeline and Streamlit app for live predictions and demo deployment.

# 🍌 Banana Ripeness Classifier

A deep learning project to classify banana ripeness stages using **MobileNetV2** and **TensorFlow/Keras**, deployed as an interactive **Streamlit web app**.

## 📌 Project Overview
This project demonstrates an end-to-end machine learning pipeline:
- Dataset preparation (ripe, unripe, freshripe, freshunripe, overripe, rotten).
- Transfer learning with MobileNetV2 for image classification.
- Model training, evaluation, and testing on unseen images.
- Deployment as a Streamlit app for live predictions.

## 🗂️ Dataset
- Source: Roboflow dataset (`banana-ripeness-1`).
- Structure: `train/`, `valid/`, `test/` folders with 6 classes:
  - `freshripe`, `freshunripe`, `ripe`, `unripe`, `overripe`, `rotten`.

## ⚙️ Model
- **Base model**: MobileNetV2 (pretrained on ImageNet).
- **Custom layers**: GlobalAveragePooling → Dense(128, relu) → Dropout → Dense(6, softmax).
- **Loss**: Categorical Crossentropy.
- **Optimizer**: Adam.

## 📊 Results
- Training accuracy: ~XX%  
- Validation accuracy: ~XX%  
- Test accuracy: ~XX%  
*(Fill these in after running your final training)*

## 🚀 Streamlit App
### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
