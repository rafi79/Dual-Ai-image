import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import io
from gtts import gTTS
import time
import logging
from typing import Optional, Tuple, Dict, Any
import gc
import os
from litellm import completion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY_LENGTH = 20
MAX_IMAGE_SIZE = (1024, 1024)
SUPPORTED_FORMATS = {'png', 'jpg', 'jpeg'}
MAX_TOKENS = 512
VISION_MODEL = "microsoft/cogvlm-grounding-generalist"
TEXT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = "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"

# Set Hugging Face token
os.environ["HUGGINGFACE_API_KEY"] = HF_TOKEN

# Page configuration
st.set_page_config(
    page_title="AI Visual Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f0fe;
    }
    </style>
    """, unsafe_allow_html=True)

class ImageProcessor:
    @staticmethod
    def validate_image(image: Image.Image) -> Tuple[bool, str]:
        """Validate image size and format"""
        if image.size[0] * image.size[1] > MAX_IMAGE_SIZE[0] * MAX_IMAGE_SIZE[1]:
            return False, "Image too large. Please upload a smaller image."
        return True, "Image valid"

    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """Preprocess image for model input"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE)
        return image

class ChatHistory:
    def __init__(self, max_length: int = MAX_HISTORY_LENGTH):
        self.max_length = max_length
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def add_message(self, role: str, content: str, image: Optional[Image.Image] = None):
        """Add message to history with cleanup if needed"""
        self.cleanup_old_messages()
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "image": image,
            "timestamp": time.time()
        })

    def cleanup_old_messages(self):
        """Remove old messages if history is too long"""
        if len(st.session_state.messages) >= self.max_length:
            st.session_state.messages = st.session_state.messages[-self.max_length:]

    def get_last_image(self) -> Optional[Image.Image]:
        """Get the last used image from history"""
        for message in reversed(st.session_state.messages):
            if "image" in message and message["image"] is not None:
                return message["image"]
        return None

class ModelManager:
    def __init__(self):
        self.initialize_session_state()

    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if 'processor' not in st.session_state:
            st.session_state.processor = None
        if 'tokenizer' not in st.session_state:
            st.session_state.tokenizer = None

    @st.cache_resource
    def load_processor():
        """Load vision model processor"""
        try:
            processor = AutoProcessor.from_pretrained(
                VISION_MODEL,
                use_auth_token=HF_TOKEN
            )
            return processor
        except Exception as e:
            logger.error(f"Error loading processor: {str(e)}")
            raise RuntimeError(f"Failed to load processor: {str(e)}")

    @st.cache_resource
    def load_tokenizer():
        """Load text model tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                TEXT_MODEL,
                use_auth_token=HF_TOKEN
            )
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

    @staticmethod
    async def process_text(prompt: str) -> str:
        """Process text using LiteLLM"""
        try:
            response = completion(
                model=TEXT_MODEL,
                messages=[{"content": prompt, "role": "user"}],
                api_base="https://api-inference.huggingface.co/models",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                stream=True
            )
            
            complete_response = ""
            for chunk in response:
                if chunk and hasattr(chunk, 'choices') and chunk.choices:
                    complete_response += chunk.choices[0].delta.content or ""
            
            return complete_response
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return None

    @staticmethod
    async def process_image(image: Image.Image, prompt: str) -> str:
        """Process image using vision model"""
        try:
            # Prepare image inputs
            inputs = st.session_state.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )

            # Make API call to Hugging Face for image processing
            response = completion(
                model=VISION_MODEL,
                messages=[{
                    "content": prompt,
                    "role": "user",
                    "image": inputs["pixel_values"].tolist()
                }],
                api_base="https://api-inference.huggingface.co/models",
                headers={"Authorization": f"Bearer {HF_TOKEN}"}
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

class TTSManager:
    @staticmethod
    def text_to_speech(text: str) -> Optional[io.BytesIO]:
        """Convert text to speech with error handling"""
        try:
            tts = gTTS(text=text, lang='en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp
        except Exception as e:
            logger.warning(f"Text-to-speech conversion failed: {str(e)}")
            return None

async def main():
    st.title("🤖 AI Visual Chat")
    
    # Initialize components
    model_manager = ModelManager()
    chat_history = ChatHistory()
    
    # Load models on first run
    if st.session_state.processor is None or st.session_state.tokenizer is None:
        try:
            with st.spinner("Loading models..."):
                st.session_state.processor = ModelManager.load_processor()
                st.session_state.tokenizer = ModelManager.load_tokenizer()
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()
    
    # Sidebar for image upload
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image... (optional)",
            type=list(SUPPORTED_FORMATS)
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                is_valid, message = ImageProcessor.validate_image(image)
                if not is_valid:
                    st.error(message)
                    st.stop()
                
                image = ImageProcessor.preprocess_image(image)
                st.image(image, use_column_width=True, caption="Uploaded Image")
                
                # Initial image analysis
                with st.spinner("Analyzing image..."):
                    response = await ModelManager.process_image(
                        image,
                        "Describe this image in detail."
                    )
                    if response:
                        chat_history.add_message("assistant", response, image)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Main chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                audio = TTSManager.text_to_speech(message["content"])
                if audio:
                    st.audio(audio, format="audio/mp3")
    
    # Chat input
    user_input = st.chat_input("Ask a question...")
    
    if user_input:
        # Add user message to chat
        chat_history.add_message("user", user_input)
        
        # Get last used image
        last_image = chat_history.get_last_image()
        
        # Process query
        with st.spinner("Thinking..."):
            if last_image:
                response = await ModelManager.process_image(last_image, user_input)
            else:
                response = await ModelManager.process_text(user_input)
            
            if response:
                chat_history.add_message("assistant", response, last_image)
                
                # Add text-to-speech
                audio = TTSManager.text_to_speech(response)
                if audio:
                    st.audio(audio, format="audio/mp3")
            else:
                st.error("Failed to generate response. Please try again.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
