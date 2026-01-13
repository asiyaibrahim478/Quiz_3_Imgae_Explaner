import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Image Captioner & Explainer",
    page_icon="🖼️",
    layout="wide"
)

# Load model with caching to avoid reloading
@st.cache_resource
def load_model():
    """Load the BLIP model and processor once and cache it"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(device)
    return processor, model, device

# Initialize model
processor, model, device = load_model()

def resize_image(image, max_size=800):
    """Resize image to max dimension while maintaining aspect ratio"""
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def generate_caption(image, detail_level="Short"):
    """Generate caption and explanation for the image"""
    # Resize image for processing
    processed_image = resize_image(image)
    
    # Generate short caption
    inputs = processor(processed_image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = model.generate(**inputs, max_length=50)
    short_caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    
    # Generate detailed explanation based on detail level
    if detail_level == "Short":
        prompt = "a detailed description:"
        max_length = 75
    elif detail_level == "Medium":
        prompt = "a comprehensive description including colors, actions, and context:"
        max_length = 100
    else:  # Detailed
        prompt = "an in-depth analysis of this image, including all visible elements, colors, actions, setting, context, and possible interpretations:"
        max_length = 150
    
    inputs = processor(processed_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        explanation_ids = model.generate(**inputs, max_length=max_length)
    detailed_explanation = processor.decode(explanation_ids[0], skip_special_tokens=True)
    
    return short_caption, detailed_explanation

def add_caption_to_image(image, caption):
    """Add caption text overlay to image"""
    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    
    # Calculate font size based on image dimensions
    img_width, img_height = img_copy.size
    font_size = max(20, min(40, img_width // 25))
    
    # Try to use a better font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position (bottom center)
    x = (img_width - text_width) // 2
    y = img_height - text_height - 20
    
    # Draw semi-transparent background
    padding = 10
    background_coords = [
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding
    ]
    draw.rectangle(background_coords, fill=(0, 0, 0, 180))
    
    # Draw text
    draw.text((x, y), caption, font=font, fill=(255, 255, 255, 255))
    
    return img_copy

def fetch_image_from_url(url):
    """Fetch image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Failed to fetch image from URL: {str(e)}")

# Sidebar
st.sidebar.title("📝 Instructions")
st.sidebar.markdown("""
**How to use this app:**

1. **Select input method** from the dropdown below
2. **Provide an image** using one of these methods:
   - Upload an image file (JPG, PNG, JPEG)
   - Capture from your device camera
   - Enter an image URL
3. **Adjust detail level** for the explanation
4. **View results:**
   - Original image with caption overlay
   - Short caption text
   - Detailed explanation

**Note:** First run may take time to load the model.
""")

st.sidebar.markdown("---")

# Input method selector
input_method = st.sidebar.selectbox(
    "Select Input Method",
    ["Upload Image", "Camera Capture", "Image URL"]
)

# Detail level selector
detail_level = st.sidebar.selectbox(
    "Detail Level for Explanation",
    ["Short", "Medium", "Detailed"],
    index=1
)

# Main app
st.title("🖼️ Image Captioner & Explainer")
st.markdown("*Generate captions and detailed explanations for your images using AI*")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

image = None

# Handle different input methods
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

elif input_method == "Camera Capture":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        try:
            image = Image.open(camera_image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            st.error(f"Error loading camera image: {str(e)}")

else:  # Image URL
    url = st.text_input("Enter image URL", placeholder="https://example.com/image.jpg")
    if url:
        try:
            with st.spinner("Fetching image from URL..."):
                image = fetch_image_from_url(url)
        except Exception as e:
            st.error(str(e))

# Process image if available
if image is not None:
    # Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Generate captions
    with st.spinner("🤖 AI is analyzing your image..."):
        try:
            short_caption, detailed_explanation = generate_caption(image, detail_level)
            captioned_image = add_caption_to_image(image, short_caption)
            
            with col2:
                st.subheader("Image with Caption Overlay")
                st.image(captioned_image, use_container_width=True)
            
            # Display results
            st.markdown("---")
            
            st.subheader("📝 Short Caption")
            st.info(short_caption)
            
            st.subheader("📖 Detailed Explanation")
            with st.expander("Click to view detailed explanation", expanded=True):
                st.markdown(f"*{detailed_explanation}*")
            
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

else:
    # Placeholder message
    st.info("👆 Please provide an image using one of the methods in the sidebar to get started.")
    
    # Show example
    st.markdown("---")
    st.markdown("### 💡 Example Use Cases")
    st.markdown("""
    - **Photo organization**: Generate descriptions for your photo library
    - **Accessibility**: Create alt text for images
    - **Content creation**: Get inspiration for image descriptions
    - **Education**: Analyze and understand image content
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Salesforce BLIP Model | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
