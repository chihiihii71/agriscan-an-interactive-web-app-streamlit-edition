import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import plotly.express as px
from huggingface_hub import hf_hub_download

# -------------------------------
# 1. Config
# -------------------------------
NUM_CLASSES = 9
REPO_ID = "Jaoooooo9/agriscan-streamlit"

CLASS_NAMES = {
    "en": [
        "Chinee Apple", "Lantana", "Parkinsonia",
        "Parthenium", "Prickly Acacia", "Rubber Vine",
        "Siam Weed", "Snake Weed", "Negative"
    ],
    "bn": [
        "চিনি অ্যাপল", "ল্যান্টানা", "পার্কিনসোনিয়া",
        "পার্থেনিয়াম", "প্রিকলি একাশিয়া", "রাবার ভাইন",
        "সিয়াম উইড", "স্নেক উইড", "নেগেটিভ"
    ]
}

TEXT = {
    "en": {
        "title": "AgriScan : Interactive Web App",
        "subtitle": "Upload a weed image to identify its species and class probabilities.",
        "upload": "Upload an image",
        "predicted": "Predicted Class",
        "confidence": "Confidence",
        "prob_chart": "Class Probabilities",
        "prob_table": "Detailed Probabilities",
    },
    "bn": {
        "title": "AgriScan : Interactive Web App",
        "subtitle": "আগাছা শনাক্ত করতে একটি ছবি আপলোড করুন এবং এর প্রজাতি ও নিশ্চিততার মাত্রা দেখুন।",
        "upload": "ছবি আপলোড করুন",
        "predicted": "শনাক্তকৃত প্রজাতি",
        "confidence": "নিশ্চয়তার মাত্রা",
        "prob_chart": "বিভিন্ন শ্রেণির সম্ভাবনা",
        "prob_table": "বিস্তারিত সম্ভাবনার তালিকা",
    }
}

# -------------------------------
# 2. Page Config
# -------------------------------
st.set_page_config(page_title="AgriScan — Interactive Web App", layout="wide")

# -------------------------------
# 3. Language Toggle
# -------------------------------
lang = st.sidebar.radio(
    "🌐 ভাষা / Language",
    options=["বাংলা (Banglish)", "English"],
    index=0,
    horizontal=True
)

lang_code = "bn" if lang == "বাংলা (Banglish)" else "en"
T = TEXT[lang_code]
CLASSES = CLASS_NAMES[lang_code]

# -------------------------------
# 4. Model Loading (ResNeSt50d Only)
# -------------------------------
@st.cache_resource
def get_model_path():
    return hf_hub_download(repo_id=REPO_ID, filename="resnest50d_model.pth")

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("resnest50d", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

MODEL_PATH = get_model_path()
model, device = load_model(MODEL_PATH)

st.title(T["title"])
st.write(T["subtitle"])

# -------------------------------
# 5. Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return CLASSES[pred_idx], probs.cpu().numpy()

# -------------------------------
# 6. UI — Results
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=T["upload"], width=400)

with col2:
    if uploaded_file is not None:
        label, probs = predict(image)
        pred_prob = max(probs) * 100

        st.markdown(f"""
            <div style="background-color:#1e4620;padding:14px;border-radius:10px;margin-bottom:12px">
                <h3 style="color:white;margin:0">🌿 {T["predicted"]}: {label}</h3>
                <p style="color:#a7f3d0;margin:4px 0 0 0">{T["confidence"]}: {pred_prob:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.subheader(T["prob_chart"])
        df = pd.DataFrame({"Class": CLASSES, "Probability": probs * 100})
        fig = px.bar(df, x="Class", y="Probability", color="Probability", color_continuous_scale="Greens")
        fig.update_layout(xaxis_tickangle=-30, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(T["prob_table"])
        df_sorted = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        st.dataframe(df_sorted.style.format({"Probability": "{:.2f}%"}), use_container_width=True)
