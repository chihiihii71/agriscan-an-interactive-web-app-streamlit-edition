import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

# -------------------------------
# 1. Page Config & CSS
# -------------------------------
st.set_page_config(
    page_title="AgriScan : An Interactive Web App", 
    layout="wide", 
    page_icon="https://unpkg.com/lucide-static@latest/icons/sprout.svg"
)

# Helper for Lucide Icons (Fetching from CDN)
def get_lucide(name, size=20, color="#1e4620"):
    url = f"https://unpkg.com/lucide-static@latest/icons/{name}.svg"
    # CSS Filter trick to change the icon color dynamically
    return f'<img src="{url}" width="{size}" height="{size}" style="filter: invert(18%) sepia(29%) saturate(1637%) hue-rotate(85deg) brightness(90%) contrast(92%); vertical-align: middle; margin-right: 8px;">'

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .pred-banner {
            background: linear-gradient(135deg, #1e4620 0%, #2ecc71 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 2. Config & Text
# -------------------------------
NUM_CLASSES = 9
REPO_ID = "Jaoooooo9/agriscan-streamlit"

CLASS_NAMES = {
    "en": ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negative"],
    "bn": ["চিনি অ্যাপল", "ল্যান্টানা", "পার্কিনসোনিয়া", "পার্থেনিয়াম", "প্রিকলি একাশিয়া", "রাবার ভাইন", "সিয়াম উইড", "স্নেক উইড", "নেগেটিভ"]
}

TEXT = {
    "en": {
        "title": "AgriScan : An Interactive Web App",
        "subtitle": "Smart Weed Detection. Choose an input method below.",
        "tab_upload": "Upload Image",
        "tab_camera": "Live Camera",
        "upload": "Drag and drop or click to upload",
        "predicted": "Detected Species",
        "confidence": "AI Confidence",
        "prob_chart": "Confidence Breakdown",
        "analyzing": "Analyzing image with AI...",
        "success": "Analysis complete!"
    },
    "bn": {
        "title": "অ্যাগ্রিস্ক্যান: ইন্টারঅ্যাকটিভ ওয়েব অ্যাপ",
        "subtitle": "স্মার্ট আগাছা শনাক্তকরণ। নিচে একটি পদ্ধতি বেছে নিন।",
        "tab_upload": "ছবি আপলোড",
        "tab_camera": "লাইভ ক্যামেরা",
        "upload": "ছবি আপলোড করতে এখানে ক্লিক করুন",
        "predicted": "শনাক্তকৃত প্রজাতি",
        "confidence": "নিশ্চয়তার মাত্রা",
        "prob_chart": "সম্ভাব্যতার তালিকা",
        "analyzing": "এআই দ্বারা ছবি বিশ্লেষণ করা হচ্ছে...",
        "success": "বিশ্লেষণ সম্পন্ন হয়েছে!"
    }
}

# -------------------------------
# 3. Sidebar & Model Loading
# -------------------------------
lang = st.sidebar.radio("Language", options=["English", "বাংলা (Banglish)"], index=0)
lang_code = "bn" if lang == "বাংলা (Banglish)" else "en"
T = TEXT[lang_code]
CLASSES = CLASS_NAMES[lang_code]

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="resnest50d_model.pth")
    model = timm.create_model("resnest50d", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return CLASSES[pred_idx], probs.cpu().numpy()

# -------------------------------
# 4. Main Interface
# -------------------------------
st.title(T["title"])
st.markdown(f"### {get_lucide('activity')} {T['subtitle']}", unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    f"{get_lucide('upload')} {T['tab_upload']}", 
    f"{get_lucide('camera')} {T['tab_camera']}"
])

img_to_process = None

with tab1:
    file = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file: img_to_process = Image.open(file).convert("RGB")

with tab2:
    cam = st.camera_input(T["tab_camera"], label_visibility="collapsed")
    if cam: img_to_process = Image.open(cam).convert("RGB")

if img_to_process:
    with st.spinner(T["analyzing"]):
        label, probs = predict(img_to_process)
        pred_prob = max(probs) * 100
        st.toast(T["success"])

    st.markdown(f"""
        <div class="pred-banner">
            <h1 style="color:white; margin:0; font-size: 2.5em;">{get_lucide('leaf', 30, '#ffffff')} {label}</h1>
            <p style="color:#e8f5e9; font-size: 1.2em; margin: 5px 0 0 0;">{T["confidence"]}: <b>{pred_prob:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])
    with c1: st.image(img_to_process, use_container_width=True)
    with c2:
        fig = go.Figure(data=[go.Pie(labels=CLASSES, values=probs*100, hole=0.55)])
        fig.update_layout(title_text=T["prob_chart"], showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
