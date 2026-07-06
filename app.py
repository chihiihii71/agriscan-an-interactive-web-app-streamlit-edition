import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import plotly.express as px
from huggingface_hub import hf_hub_download

# -------------------------------
# 1. Config & Text Dictionary
# -------------------------------
NUM_CLASSES = 9
REPO_ID = "Jaoooooo9/agriscan-streamlit"

CLASS_NAMES = {
    "en": [
        "DeepWeeds (Chinee Apple)", "DeepWeeds (Lantana)", "DeepWeeds (Parkinsonia)",
        "DeepWeeds (Parthenium)", "DeepWeeds (Prickly Acacia)", "DeepWeeds (Rubber Vine)",
        "DeepWeeds (Siam Weed)", "DeepWeeds (Snake Weed)", "Not DeepWeeds (Negative)"
    ],
    "bn": [
        "আগাছা (চিনি অ্যাপল)", "আগাছা (ল্যান্টানা)", "আগাছা (পার্কিনসোনিয়া)",
        "আগাছা (পার্থেনিয়াম)", "আগাছা (প্রিকলি অ্যাকাসিয়া)", "আগাছা (রাবার ভাইন)",
        "আগাছা (সিয়াম উইড)", "আগাছা (স্নেক উইড)", "আগাছা নয় (নেগেটিভ)"
    ]
}

# ... (Keep your existing TEXT dictionary here) ...
# (Assuming your TEXT dictionary is unchanged from your previous code)

# -------------------------------
# 2. Page Config
# -------------------------------
st.set_page_config(page_title="AgriScan", layout="wide")

# -------------------------------
# 3. Model Loading (Fixed to be more robust)
# -------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="resnest50d_model.pth")
    # Load model architecture
    model = timm.create_model("resnest50d", pretrained=False, num_classes=NUM_CLASSES)
    # Load weights with strict=False to avoid minor key mismatch errors
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# -------------------------------
# 4. Prediction Logic
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, class_list):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return class_list[pred_idx], probs.cpu().numpy()

# -------------------------------
# 5. UI Layout
# -------------------------------
# Sidebar
lang = st.sidebar.radio("Language", ["English", "বাংলা (Banglish)"])
lang_code = "bn" if lang == "বাংলা (Banglish)" else "en"
T = TEXT[lang_code]
CLASSES = CLASS_NAMES[lang_code]

# Main Area
st.title(T["title"])
uploaded_file = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    label, probs = predict(image, CLASSES)
    
    # Top Banner
    st.markdown(f"### {T['predicted']}: {label}")
    
    # Two Columns with fixed ratio
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Use native image sizing instead of CSS hacks
        st.image(image, use_container_width=True)
        
    with col2:
        df = pd.DataFrame({"Class": CLASSES, "Probability": probs * 100})
        fig = px.bar(df, x="Probability", y="Class", orientation='h', color="Probability", color_continuous_scale="Greens")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
