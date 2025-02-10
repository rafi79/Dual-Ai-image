import streamlit as st
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from PIL import Image
import io
from gtts import gTTS

# Page configuration
st.set_page_config(
    page_title="DeepSeek-VL Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

@st.cache_resource
def load_model():
    """Load DeepSeek-VL model and processor"""
    try:
        model_path = "deepseek-ai/deepseek-vl-1.3b-base"
        
        # Load processor and tokenizer
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        
        # Load model
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
        vl_gpt = vl_gpt.eval()
        
        return vl_chat_processor, vl_gpt
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_image_and_text(image, prompt, processor, model):
    """Process image and text using DeepSeek-VL"""
    try:
        # Prepare conversation
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
        
        # Prepare inputs
        prepare_inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        ).to(model.device)
        
        # Get image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        # Decode response
        answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
        
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
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
    st.title("🤖 DeepSeek-VL Visual Chat")
    
    # Load model on first run
    if st.session_state.model is None or st.session_state.processor is None:
        with st.spinner("Loading DeepSeek-VL model..."):
            processor, model = load_model()
            if model is not None and processor is not None:
                st.session_state.processor = processor
                st.session_state.model = model
            else:
                st.error("Failed to load model")
                st.stop()
    
    # Sidebar for image upload
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Image")
            
            # Initial image analysis
            with st.spinner("Analyzing image..."):
                response = process_image_and_text(
                    image,
                    "Describe this image in detail.",
                    st.session_state.processor,
                    st.session_state.model
                )
                if response:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "image": image
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
    user_input = st.chat_input("Ask about the image...")
    
    if user_input and "image" in st.session_state.messages[-1]:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get last used image
        last_image = st.session_state.messages[-2]["image"]
        
        # Process query
        with st.spinner("Thinking..."):
            response = process_image_and_text(
                last_image,
                user_input,
                st.session_state.processor,
                st.session_state.model
            )
            
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "image": last_image
                })
                
                # Add text-to-speech
                audio = text_to_speech(response)
                if audio:
                    st.audio(audio, format="audio/mp3")

if __name__ == "__main__":
    main()
