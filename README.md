# Image-Sentiment-Analysis | Classify Happy 😊 or Sad 😢

🔗 **Try the live app:** [image-sentiment-analysis.streamlit.app](https://image-sentiment-analysis.streamlit.app/)

A real-time image sentiment analyzer built with **Streamlit**, **TensorFlow/Keras**, **OpenCV**, and **streamlit-webrtc**.  
Upload an image or use your webcam — the model classifies the expression as **Happy 😊** or **Sad 😢** and shows confidence.

---

## 🚀 Project Overview

This app demonstrates a **browser-friendly** emotion classification pipeline:

- Image upload or live webcam capture  
- Preprocessing & resizing to model input  
- Deep-learning prediction (binary: Happy / Sad)  
- Clean, responsive Streamlit UI with visual feedback

---

## 🎯 Key Features

✅ **Two input modes:** Upload Image or Use Webcam  
✅ **Real-time inference** with on-device processing  
✅ **Confidence visualization** (progress bar + overlay text)  
✅ **Session history** (timestamp, expression, confidence)  
✅ **Friendly UI feedback** — balloons for Happy, snow for Sad  
✅ **WebRTC support** for live camera streaming

---

## 🛠️ Tech Stack

- **Python 3.10+**
- Streamlit `1.47.0`
- TensorFlow `2.12.0` + Keras
- OpenCV (headless) `4.7.0.72`
- NumPy `1.23.5`, Pillow `11.3.0`
- pandas `2.3.1`, matplotlib `3.10.3`
- streamlit-webrtc

---

## 📂 Project Structure

```plaintext
📂 Image-Sentiment-Analysis/
 ├── app.py                # Streamlit app (upload + webcam + inference)
 ├── requirements.txt      # Python dependencies
 ├── README.md             # Project documentation
 └── models/
     └── saved_model/      # TensorFlow SavedModel (loaded by the app)
