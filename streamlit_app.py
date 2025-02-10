import streamlit as st
from litellm import completion
import os
from PIL import Image
import io
from gtts import gTTS
import time

# Page configuration
st.set_page_config(
    page_title="AI Image & Text Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = None

def initialize_api():
    """Initialize Hugging Face API"""
    try:
        # Try to get API key from Streamlit secrets
        hf_token = st.secrets["hf_token"]
    except:
        # Fallback to environment variable
        hf_token = os.getenv("HUGGINGFACE_API_KEY", "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")
    
    os.environ["HUGGINGFACE_API_KEY"] = hf_token
    return True

def process_image_with_model(image, model_name):
    """Process image using Hugging Face model via LiteLLM"""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create prompt for image analysis
        messages = [
            {
                "content": "Please analyze this image and provide a detailed description.",
                "role": "user"
            }
        ]
        
        # Make API call with image
        response = completion(
            model=f"huggingface/{model_name}",
            messages=messages,
            api_base="https://api-inference.huggingface.co/models",
            image=img_byte_arr
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
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

def main():
    st.title("🤖 AI Image & Text Analyzer")
    
    # Initialize API
    if not initialize_api():
        st.stop()
    
    # Model selection
    models = {
        "DeepSeek VL": "deepseek-ai/deepseek-vl-1.3b-base",
        "PaLI-GEMMA": "google/paligemma-3b-pt-224"
    }
    
    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )
    
    # Upload section
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            with st.spinner("Analyzing image..."):
                response = process_image_with_model(image, models[selected_model])
                
                if response:
                    # Display analysis
                    st.subheader("Analysis")
                    
                    # Create columns for text and audio
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(response)
                        
                    with col2:
                        # Add audio description
                        st.subheader("Listen")
                        audio_file = text_to_speech(response)
                        if audio_file:
                            st.audio(audio_file, format='audio/mp3')
                    
                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Uploaded image for analysis using {selected_model}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
        
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
    
    # Chat history
    if st.session_state.messages:
        st.subheader("Analysis History")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

if __name__ == "__main__":
    main()
