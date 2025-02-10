import streamlit as st
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
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
MODEL_PATH = "deepseek-ai/deepseek-vl-1.3b-base"
HF_TOKEN = "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"  # Securely store this!

# Set Hugging Face token
os.environ["HUGGINGFACE_API_KEY"] = HF_TOKEN

# Page configuration
st.set_page_config(
    page_title="Enhanced DeepSeek-VL Chatbot",
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

class LiteLLMManager:
    @staticmethod
    async def get_completion(prompt: str) -> str:
        """Get completion from Hugging Face model using LiteLLM"""
        try:
            response = completion(
                model="huggingface/facebook/blenderbot-400M-distill",
                messages=[{"content": prompt, "role": "user"}],
                api_base="https://api-inference.huggingface.co/models",
                stream=True
            )
            
            # Collect streamed response
            complete_response = ""
            for chunk in response:
                if chunk and hasattr(chunk, 'choices') and chunk.choices:
                    complete_response += chunk.choices[0].delta.content or ""
            
            return complete_response
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            return None

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
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'processor' not in st.session_state:
            st.session_state.processor = None

    @st.cache_resource
    def load_model() -> Tuple[Any, Any]:
        """Load DeepSeek-VL model and processor with error handling"""
        try:
            logger.info("Loading model and processor...")
            
            # Load processor and model
            vl_chat_processor = VLChatProcessor.from_pretrained(
                MODEL_PATH,
                use_auth_token=HF_TOKEN
            )
            
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                use_auth_token=HF_TOKEN,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Move model to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                vl_gpt = vl_gpt.to(device)
            vl_gpt = vl_gpt.eval()
            
            logger.info(f"Model loaded successfully on {device}")
            return vl_chat_processor, vl_gpt

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    @staticmethod
    async def process_input(prompt: str, image: Optional[Image.Image] = None) -> str:
        """Process input using either image+text or text-only mode"""
        if image:
            # Use DeepSeek-VL for image+text processing
            return ModelManager.process_image_and_text(
                image,
                prompt,
                st.session_state.processor,
                st.session_state.model
            )
        else:
            # Use LiteLLM for text-only processing
            return await LiteLLMManager.get_completion(prompt)

    @staticmethod
    def process_image_and_text(
        image: Image.Image,
        prompt: str,
        processor: Any,
        model: Any
    ) -> Optional[str]:
        """Process image and text using DeepSeek-VL"""
        try:
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{prompt}",
                    "images": [image]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            prepare_inputs = processor(
                conversations=conversation,
                images=[image],
                force_batchify=True
            ).to(model.device)
            
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            
            with torch.no_grad():
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    bos_token_id=processor.tokenizer.bos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True
                )
            
            answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return answer

        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
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
    st.title("🤖 Enhanced DeepSeek-VL Visual Chat")
    
    # Initialize components
    model_manager = ModelManager()
    chat_history = ChatHistory()
    
    # Load model on first run
    if st.session_state.model is None or st.session_state.processor is None:
        try:
            with st.spinner("Loading DeepSeek-VL model..."):
                processor, model = ModelManager.load_model()
                st.session_state.processor = processor
                st.session_state.model = model
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()
    
    # Sidebar for image upload and settings
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
                    response = await ModelManager.process_input(
                        "Describe this image in detail.",
                        image
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
            response = await ModelManager.process_input(user_input, last_image)
            
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
