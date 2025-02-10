import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import io
from gtts import gTTS
import os
from huggingface_hub import login

# Hugging Face login
HF_TOKEN = "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"  # Better to use st.secrets in production
login(token=HF_TOKEN)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def load_models():
    """Load both models and processors"""
    # Load PaLI-GEMMA model
    pali_processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    pali_model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224")
    
    # Load DeepSeek model
    deepseek_pipeline = pipeline("image-text-to-text", model="deepseek-ai/deepseek-vl-1.3b-base")
    
    return pali_processor, pali_model, deepseek_pipeline

def process_image_pali(image, processor, model):
    """Process image using PaLI-GEMMA model"""
    inputs = processor(images=image, text="Describe this image in detail", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return processor.decode(outputs[0], skip_special_tokens=True)

def process_image_deepseek(image, pipeline):
    """Process image using DeepSeek model"""
    result = pipeline(image, "Describe this image in detail")
    return result[0]['generated_text']

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def main():
    st.title("Multimodal AI Chatbot")
    
    # Load models
    with st.spinner("Loading models..."):
        pali_processor, pali_model, deepseek_pipeline = load_models()
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process with both models
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PaLI-GEMMA Analysis")
            pali_response = process_image_pali(image, pali_processor, pali_model)
            st.write(pali_response)
            
            # Add text-to-speech for PaLI response
            audio_file = text_to_speech(pali_response)
            st.audio(audio_file, format='audio/mp3')
        
        with col2:
            st.subheader("DeepSeek Analysis")
            deepseek_response = process_image_deepseek(image, deepseek_pipeline)
            st.write(deepseek_response)
            
            # Add text-to-speech for DeepSeek response
            audio_file = text_to_speech(deepseek_response)
            st.audio(audio_file, format='audio/mp3')
        
        # Combined analysis
        st.subheader("Combined Analysis")
        combined_response = f"Combined insights:\n{pali_response}\n\nAdditional context:\n{deepseek_response}"
        st.write(combined_response)
        
        # Store in chat history
        st.session_state.messages.append({"role": "user", "content": "Image uploaded"})
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
    
    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()
