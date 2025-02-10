import streamlit as st
import requests
import json
from PIL import Image
import io
from gtts import gTTS
import base64

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Constants
HF_TOKEN = "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"
VISION_API_URL = "https://api-inference.huggingface.co/models/microsoft/git-base-coco"
CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

def process_image(image):
    """Process image using Hugging Face Vision API"""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create headers with token
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Encode image to base64
        encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Prepare payload
        payload = {
            "inputs": {
                "image": encoded_image
            }
        }

        # Make API request
        response = requests.post(
            VISION_API_URL,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            description = response.json()[0]['generated_text']
            return description
        else:
            st.error(f"API Error: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_chatbot_response(prompt):
    """Get response from chatbot API"""
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt
        }

        response = requests.post(
            CHAT_API_URL,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            st.error(f"API Error: {response.status_code}")
            return None

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
    
    # Sidebar for image upload
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Image")
            st.session_state.current_image = image
            
            # Process image
            with st.spinner("Analyzing image..."):
                image_description = process_image(image)
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
                audio = text_to_speech(message["content"])
                if audio:
                    st.audio(audio, format="audio/mp3")
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        
        # Get response
        with st.spinner("Thinking..."):
            # Include image context if available
            context = ""
            if st.session_state.current_image:
                image_desc = process_image(st.session_state.current_image)
                if image_desc:
                    context = f"Context: {image_desc}. "
            
            full_prompt = f"{context}User: {user_input}"
            response = get_chatbot_response(full_prompt)
            
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Text-to-speech for response
                audio = text_to_speech(response)
                if audio:
                    st.audio(audio, format="audio/mp3")

if __name__ == "__main__":
    main()
