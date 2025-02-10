import streamlit as st
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image
import io
from gtts import gTTS
import wget
import os
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"
MODEL_PATH = "llama-2-7b-chat.Q2_K.gguf"
HF_TOKEN = "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"
VISION_MODEL = "microsoft/cogvlm-grounding-generalist"

def download_model():
    """Download the Llama model if not present"""
    if not os.path.exists(MODEL_PATH):
        def bar_custom(current, total, width=80):
            print(f"Downloading {current / total * 100}% [{current} / {total}] bytes")
        wget.download(MODEL_URL, bar=bar_custom)

def init_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        page_title="Multimodal AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    st.header("Multimodal AI Assistant")
    st.sidebar.title("Options")

def select_llm() -> LlamaCPP:
    """Initialize Llama model"""
    return LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_new_tokens=500,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

@st.cache_resource
def load_vision_model():
    """Load vision model and processor"""
    try:
        processor = AutoProcessor.from_pretrained(
            VISION_MODEL,
            use_auth_token=HF_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            VISION_MODEL,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading vision model: {str(e)}")
        return None, None

def process_image(image: Image.Image, prompt: str, model, processor) -> str:
    """Process image using vision model"""
    try:
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def text_to_speech(text: str) -> Optional[io.BytesIO]:
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        logger.warning(f"Text-to-speech conversion failed: {str(e)}")
        return None

def init_messages():
    """Initialize or clear chat messages"""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant that can understand both text and images. Reply in markdown format."
            )
        ]

def get_text_answer(llm, messages) -> str:
    """Get answer from Llama model"""
    response = llm.complete(messages)
    return response.text

def main():
    init_page()
    
    # Download and load models
    download_model()
    llm = select_llm()
    vision_model, vision_processor = load_vision_model()
    
    init_messages()
    
    # Sidebar for image upload
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Initial image analysis
            if vision_model and vision_processor:
                with st.spinner("Analyzing image..."):
                    description = process_image(
                        image,
                        "Describe this image in detail.",
                        vision_model,
                        vision_processor
                    )
                    if description:
                        st.session_state.messages.append(AIMessage(content=description))
                        audio = text_to_speech(description)
                        if audio:
                            st.audio(audio, format="audio/mp3")
    
    # Chat interface
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        with st.spinner("Bot is typing ..."):
            if uploaded_file and vision_model and vision_processor and "image" in user_input.lower():
                # Process image-related query
                answer = process_image(image, user_input, vision_model, vision_processor)
            else:
                # Process text-only query
                answer = get_text_answer(llm, user_input)
            
            if answer:
                st.session_state.messages.append(AIMessage(content=answer))
                audio = text_to_speech(answer)
    
    # Display message history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                if hasattr(message, 'audio'):
                    st.audio(message.audio, format="audio/mp3")
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

if __name__ == "__main__":
    main()
