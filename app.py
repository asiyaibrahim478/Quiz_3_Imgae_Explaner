"""
Image Captioner and Explainer
A Streamlit app that generates captions and detailed explanations for images
using Hugging Face's BLIP model.
"""

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Image Captioner and Explainer",
    page_icon="🖼️",
    layout="wide"
)


@st.cache_resource
def load_model():
    """
    Load the BLIP model and processor with caching.
    Returns processor and model objects.
    """
    try:
        # Import here to avoid module-level import errors
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        with st.spinner("Loading AI model... (First time may take a few minutes)"):
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        return processor, model, torch
    except ImportError as ie:
        st.error(f"Import Error: {str(ie)}")
        st.error("Please ensure all packages in requirements.txt are installed.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def validate_and_resize_image(image, max_size=512):
    """
    Validate and resize image if needed.
    Args:
        image: PIL Image object
        max_size: Maximum dimension size
    Returns:
        Resized PIL Image or None if invalid
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        width, height = image.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"Invalid image: {str(e)}")
        return None


def get_image_from_url(url):
    """
    Fetch image from URL.
    Args:
        url: Image URL string
    Returns:
        PIL Image or None if failed
    """
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content length (5MB limit)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 5 * 1024 * 1024:
            st.error("Image too large. Please use an image under 5MB.")
            return None
        
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException:
        st.error("Invalid URL or unable to fetch image.")
        return None
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None


def generate_caption(image, processor, model, torch_module, max_length=20):
    """
    Generate short caption for image.
    Args:
        image: PIL Image
        processor: BLIP processor
        model: BLIP model
        torch_module: torch module
        max_length: Maximum caption length
    Returns:
        Caption string
    """
    try:
        inputs = processor(image, return_tensors="pt")
        
        # Move to GPU if available
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch_module.no_grad():
            out = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None


def generate_explanation(image, processor, model, torch_module):
    """
    Generate detailed explanation for image.
    Args:
        image: PIL Image
        processor: BLIP processor
        model: BLIP model
        torch_module: torch module
    Returns:
        Explanation string
    """
    try:
        # Use conditional generation with prompt for detailed description
        text_prompt = "a detailed description of"
        inputs = processor(image, text_prompt, return_tensors="pt")
        
        # Move to GPU if available
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate longer description
        with torch_module.no_grad():
            out = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                temperature=0.7
            )
        explanation = processor.decode(out[0], skip_special_tokens=True)
        
        # If explanation is too short, generate another variant
        if len(explanation.split()) < 15:
            inputs2 = processor(image, return_tensors="pt")
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            with torch_module.no_grad():
                out2 = model.generate(
                    **inputs2,
                    max_length=80,
                    num_beams=5,
                    do_sample=True,
                    temperature=0.9
                )
            explanation2 = processor.decode(out2[0], skip_special_tokens=True)
            explanation = f"{explanation}. {explanation2}"
        
        return explanation
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return None


def main():
    """Main application function."""
    
    # Title and description
    st.title("🖼️ Image Captioner and Explainer")
    st.markdown(
        """
        Upload an image, capture from camera, or provide a URL to get an 
        automatic caption and detailed explanation using a free AI model.
        
        **Instructions:**
        1. Select an input method from the sidebar
        2. Provide your image
        3. Click 'Generate Caption and Explanation'
        4. View the results!
        
        *Note: First load may take a few minutes while the AI model downloads.*
        """
    )
    
    # Load model
    processor, model, torch_module = load_model()
    if processor is None or model is None or torch_module is None:
        st.error("❌ Failed to load the AI model.")
        st.info("💡 **Troubleshooting steps:**")
        st.markdown("""
        1. Make sure you're deploying on Streamlit Cloud (not running locally without dependencies)
        2. Check that your `requirements.txt` file is in the root of your repository
        3. Wait a few minutes for Streamlit Cloud to install all packages
        4. Click 'Manage app' → 'Reboot app' if the error persists
        5. Check the logs in 'Manage app' for detailed error messages
        """)
        return
    
    # Sidebar for input method selection
    st.sidebar.title("Input Method")
    input_method = st.sidebar.radio(
        "Choose how to provide an image:",
        ["Upload Image", "Capture from Camera", "Enter URL"]
    )
    
    # Initialize session state for image
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    # Initialize image variable
    image = None
    
    # Main area - conditional rendering based on input method
    if input_method == "Upload Image":
        st.subheader("📁 Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, PNG, JPEG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            # Check file size (5MB limit)
            if uploaded_file.size > 5 * 1024 * 1024:
                st.error("File too large. Please upload an image under 5MB.")
            else:
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
    
    elif input_method == "Capture from Camera":
        st.subheader("📷 Capture from Camera")
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            st.session_state.current_image = image
    
    else:  # Enter URL
        st.subheader("🔗 Enter Image URL")
        url = st.text_input(
            "Paste a public image URL",
            placeholder="https://images.unsplash.com/photo-..."
        )
        
        if url and st.button("Fetch Image"):
            with st.spinner("Fetching image..."):
                image = get_image_from_url(url)
                if image is not None:
                    st.session_state.current_image = image
    
    # Use image from session state
    if st.session_state.current_image is not None:
        image = st.session_state.current_image
    
    # Process and display results
    st.divider()
    
    if image is not None:
        # Validate and resize
        image = validate_and_resize_image(image)
        
        if image is not None:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            
            with col2:
                # Generate button
                if st.button("✨ Generate Caption and Explanation", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your image..."):
                        # Generate caption
                        caption = generate_caption(image, processor, model, torch_module)
                        
                        # Generate explanation
                        explanation = generate_explanation(image, processor, model, torch_module)
                        
                        if caption and explanation:
                            # Display results
                            st.success("✅ Generation complete!")
                            
                            st.subheader("📝 Generated Caption:")
                            st.info(caption)
                            
                            st.subheader("📄 Detailed Explanation:")
                            st.write(explanation)
                        else:
                            st.error(
                                "Error generating description. "
                                "Please try another image."
                            )
    else:
        st.info("👆 Please provide an image using one of the methods above.")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
        Powered by Hugging Face's BLIP model and Streamlit<br>
        <small>Free and open-source AI image captioning</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
