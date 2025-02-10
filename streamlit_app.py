import streamlit as st
from litellm import completion
import os
from PIL import Image
import io
from gtts import gTTS
from transformers import pipeline
import time

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

def initialize_models():
    """Initialize models with caching"""
    @st.cache_resource
    def load_image_model():
        return pipeline("image-to-text", model="deepseek-ai/deepseek-vl-1.3b-base")
    
    return load_image_model()

def setup_api():
    """Setup API keys and configurations"""
    try:
        hf_token = st.secrets.get("hf_token", os.getenv("HUGGINGFACE_API_KEY"))
        os.environ["HUGGINGFACE_API_KEY"] = hf_token
        return True
    except Exception as e:
        st.error(f"Error setting up API: {str(e)}")
        return False

def process_image(image, model):
    """Process image and generate description"""
    try:
        result = model(image)
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_chatbot_response(prompt, context=""):
    """Get response from chatbot"""
    try:
        messages = [{"content": f"{context}\n\nUser: {prompt}", "role": "user"}]
        response = completion(
            model="huggingface/facebook/blenderbot-400M-distill",
            messages=messages,
            api_base="https://api-inference.huggingface.co/models",
            stream=True
        )
        return response
    except Exception as e:
        st.error(f"Error getting chatbot response: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech"""
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
    st.title("🤖 AI Chatbot with Image Understanding")
    
    # Setup API
    if not setup_api():
        st.stop()
    
    # Load models
    with st.spinner("Loading models..."):
        image_model = initialize_models()
    
    # Sidebar for image upload
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.current_image = image
            
            # Process image
            with st.spinner("Analyzing image..."):
                image_description = process_image(image, image_model)
                if image_description:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I see: {image_description}"
                    })
    
    # Main chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                # Add audio button for assistant responses
                audio = text_to_speech(message["content"])
                if audio:
                    st.audio(audio, format="audio/mp3")
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get context from current image if available
        context = ""
        if st.session_state.current_image:
            image_description = process_image(st.session_state.current_image, image_model)
            if image_description:
                context = f"The current image shows: {image_description}"
        
        # Get chatbot response
        response = get_chatbot_response(user_input, context)
        
        if response:
            # Create placeholder for streaming response
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in response:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Add audio for response
            audio = text_to_speech(full_response)
            if audio:
                st.audio(audio, format="audio/mp3")

if __name__ == "__main__":
    main()
