import streamlit as st
import torch
from transformers import pipeline
from PIL import Image
import io
from gtts import gTTS
from huggingface_hub import login
import os

# Configuration
st.set_page_config(
    page_title="AI Image Analyzer",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = None

def initialize_huggingface():
    """Initialize Hugging Face authentication"""
    try:
        hf_token = st.secrets["hf_token"]  # Use Streamlit secrets
    except:
        hf_token = os.getenv("HF_TOKEN", "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")
    
    try:
        login(token=hf_token)
        return True
    except Exception as e:
        st.error(f"Error authenticating with Hugging Face: {str(e)}")
        return False

@st.cache_resource
def load_model(model_name):
    """Load model with caching and error handling"""
    try:
        if model_name == "deepseek":
            return pipeline("image-text-to-text", 
                          model="deepseek-ai/deepseek-vl-1.3b-base",
                          device="cpu")  # Force CPU to avoid CUDA memory issues
        else:
            st.error(f"Unknown model: {model_name}")
            return None
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.warning(f"Text-to-speech conversion failed: {str(e)}")
        return None

def process_image(image, model):
    """Process image with error handling"""
    try:
        result = model(image, "Describe this image in detail")
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("🖼️ AI Image Analyzer")
    
    # Initialize Hugging Face
    if not initialize_huggingface():
        st.stop()
    
    # Model selection
    model_options = {
        "DeepSeek VL": "deepseek"
    }
    
    selected_model = st.selectbox(
        "Select AI Model",
        list(model_options.keys())
    )
    
    # Load selected model
    with st.spinner("Loading model..."):
        model = load_model(model_options[selected_model])
        if model is None:
            st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            with st.spinner("Analyzing image..."):
                response = process_image(image, model)
                
                if response:
                    st.subheader("Analysis")
                    st.write(response)
                    
                    # Text-to-speech
                    st.subheader("Audio Description")
                    audio_file = text_to_speech(response)
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                    
                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "user",
                        "content": "Image uploaded"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
        
        except Exception as e:
            st.error(f"Error processing upload: {str(e)}")
    
    # Display chat history
    if st.session_state.messages:
        st.subheader("Chat History")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

if __name__ == "__main__":
    main()
