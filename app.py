import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Streamlit page setup
st.set_page_config(
    page_title="Image Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model (cache to avoid reloading)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/saved_model')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to classify image
def predict_emotion(img, model):
    try:
        # Ensure image has 3 channels (RGB)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize and preprocess image
        img = cv2.resize(img, (256, 256))
        img_array = np.array(img)
        resize = tf.image.resize(img_array, (256, 256))
        
        # Prediction
        yhat = model.predict(np.expand_dims(resize/255, 0))
        prediction = yhat[0][0]
        
        # Determine emotion and confidence
        emotion = "Happy 😊" if prediction <= 0.5 else "Sad 😢"
        confidence = (1 - prediction) if emotion == "Happy 😊" else prediction
        
        return emotion, confidence, resize.numpy().astype(int)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None

# Function to display results
def display_results(original_img, processed_img, emotion, confidence):
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_img, caption="Original Image", use_container_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        st.image(processed_img, caption="Processed Image (256x256)", use_container_width=True)
        
        # Progress bar for confidence
        st.progress(float(confidence))
        
        # Show result with custom styling
        if emotion == "Happy 😊":
            st.markdown(
                f'<div style="font-size: 20px; font-weight: bold; padding: 10px; '
                f'border-radius: 5px; margin-top: 20px; background-color: #D5F5E3; '
                f'color: #27AE60;">Predicted Emotion: {emotion} (Confidence: {confidence*100:.2f}%)</div>',
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                f'<div style="font-size: 20px; font-weight: bold; padding: 10px; '
                f'border-radius: 5px; margin-top: 20px; background-color: #FADBD8; '
                f'color: #E74C3C;">Predicted Emotion: {emotion} (Confidence: {confidence*100:.2f}%)</div>',
                unsafe_allow_html=True
            )
            st.snow()

# Custom VideoTransformer for webcam processing
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.snapshot = None
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store the latest frame for snapshot
        self.snapshot = img
        
        # Display the live video with emotion detection
        emotion, confidence, _ = predict_emotion(img, self.model)
        
        # Add emotion text to the frame
        text = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 24px !important;
        color: #2E86C1;
    }
    </style>
    """, unsafe_allow_html=True)

# Main app function
def main():
    # Title and description
    st.title("😊 Image Sentiment Analysis 😢")
    st.markdown("""
    Upload an image or use your webcam to analyze facial expressions.
    The model will classify the expression as **Happy 😊** or **Sad 😢**.
    """)
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=["Timestamp", "Expression", "Confidence"])
    
    # Analysis section
    st.header("Analyze Image")
    option = st.radio("Select input method:", ("Upload Image", "Use Webcam"))
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image directly without tempfile
            img = Image.open(uploaded_file)
            img = np.array(img)
            
            if img is not None:
                with st.spinner("Analyzing image..."):
                    emotion, confidence, processed_img = predict_emotion(img, model)
                
                if emotion and confidence is not None:
                    display_results(img, processed_img, emotion, confidence)
                    
                    # Add to history
                    new_entry = pd.DataFrame({
                        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        "Expression": [emotion],
                        "Confidence": [f"{confidence*100:.2f}%"]
                    })
                    st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
    
    else:  # Webcam option
        st.warning("Note: Webcam feature works best in local environment. On Streamlit Cloud, it may have limited functionality.")
        
        # Create a context for the webcam
        ctx = webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=EmotionVideoTransformer,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            async_transform=True
        )
        
        if ctx.video_transformer:
            if st.button("Capture Snapshot"):
                snapshot = ctx.video_transformer.snapshot
                if snapshot is not None:
                    with st.spinner("Analyzing snapshot..."):
                        emotion, confidence, processed_img = predict_emotion(snapshot, model)
                    
                    if emotion and confidence is not None:
                        display_results(snapshot, processed_img, emotion, confidence)
                        
                        # Add to history
                        new_entry = pd.DataFrame({
                            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                            "Expression": [emotion],
                            "Confidence": [f"{confidence*100:.2f}%"]
                        })
                        st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
    
    # Info section
    st.sidebar.title("About This App")
    st.sidebar.markdown("""
    This app uses deep learning to analyze emotions from images.
    
    **How it works?**
    1. Upload an image or use webcam
    2. Model analyzes facial expressions
    3. App displays results with confidence level
    
    **Technologies used:**
    - TensorFlow/Keras
    - OpenCV for image processing
    - Streamlit for UI
    """)

if __name__ == "__main__":
    if model is not None:
        main()