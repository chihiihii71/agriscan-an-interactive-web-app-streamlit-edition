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
        "Chinee Apple", "Lantana", "Parkinsonia",
        "Parthenium", "Prickly Acacia", "Rubber Vine",
        "Siam Weed", "Snake Weed", "Negative"
    ],
    "bn": [
        "চিনি অ্যাপল", "ল্যান্টানা", "পার্কিনসোনিয়া",
        "পার্থেনিয়াম", "প্রিকলি একাশিয়া", "রাবার ভাইন",
        "সিয়াম উইড", "স্নেক উইড", "নেগেটিভ"
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
        "prob_table": "Detailed Probabilities & Report",
        "analyzing": "Analyzing image...",
        "download": "Download CSV Report",
        "about": "About AgriScan",
        "about_desc": "This AI tool uses a ResNeSt50d model to classify invasive deepweed species with high accuracy."
    },
    "bn": {
        "title": "অ্যাগ্রিস্ক্যান: ইন্টারঅ্যাকটিভ ওয়েব অ্যাপ",
        "subtitle": "আগাছা শনাক্ত করতে একটি ছবি আপলোড করুন এবং এর প্রজাতি ও নিশ্চয়তার মাত্রা দেখুন।",
        "upload": "ছবি আপলোড করুন",
        "predicted": "শনাক্তকৃত প্রজাতি",
        "confidence": "নিশ্চয়তার মাত্রা",
        "prob_chart": "বিভিন্ন শ্রেণির সম্ভাবনা",
        "prob_table": "বিস্তারিত সম্ভাবনার তালিকা ও রিপোর্ট",
        "analyzing": "ছবি বিশ্লেষণ করা হচ্ছে...",
        "download": "রিপোর্ট ডাউনলোড করুন (CSV)",
        "about": "অ্যাগ্রিস্ক্যান সম্পর্কে",
        "about_desc": "এই এআই টুলটি একটি ResNeSt50d মডেল ব্যবহার করে অত্যন্ত নিখুঁতভাবে ক্ষতিকর আগাছা শনাক্ত করতে পারে।"
    }
}

# -------------------------------
# 2. Page Config
# -------------------------------
st.set_page_config(page_title="AgriScan — Interactive Web App", layout="wide")

# -------------------------------
# 3. Sidebar (Controls & Info)
# -------------------------------
lang = st.sidebar.radio(
    "ভাষা / Language",
    options=["বাংলা (Banglish)", "English"],
    index=0
)

lang_code = "bn" if lang == "বাংলা (Banglish)" else "en"
T = TEXT[lang_code]
CLASSES = CLASS_NAMES[lang_code]

st.sidebar.markdown("---")
st.sidebar.subheader(T["about"])
st.sidebar.info(T["about_desc"])

# -------------------------------
# 4. Model Loading
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

# -------------------------------
# 5. Main UI Header
# -------------------------------
st.title(T["title"])
st.write(T["subtitle"])

# -------------------------------
# 6. Image Transform & Prediction
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
# 7. UI — Interactive Dashboard Layout
# -------------------------------
uploaded_file = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Loading State ---
    with st.spinner(T["analyzing"]):
        image = Image.open(uploaded_file).convert("RGB")
        label, probs = predict(image)
        pred_prob = max(probs) * 100
        df = pd.DataFrame({"Class": CLASSES, "Probability": probs * 100})
        df_sorted = df.sort_values("Probability", ascending=False).reset_index(drop=True)

    # --- 1. Top Result Banner ---
    st.markdown(f"""
        <div style="background-color:#1e4620;padding:20px;border-radius:10px;margin-bottom:20px;text-align:center;">
            <h2 style="color:white;margin:0">{T["predicted"]}: {label}</h2>
            <h4 style="color:#a7f3d0;margin:5px 0 0 0">{T["confidence"]}: {pred_prob:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)

    # --- 2. Side-by-Side Image & Chart (Forces single-screen view) ---
    col_img, col_chart = st.columns([1, 1.2]) # Adjusts width ratio

    with col_img:
        st.image(image, use_container_width=True, caption=label)

    with col_chart:
        st.markdown(f"**{T['prob_chart']}**")
        # Polished Plotly Chart
        fig = px.bar(
            df, x="Class", y="Probability", 
            text_auto='.1f', # Shows numbers directly on the bars
            color="Probability", 
            color_continuous_scale="Greens"
        )
        fig.update_layout(
            xaxis_tickangle=-45, 
            showlegend=False, 
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), # Removes empty space around chart
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 3. Expandable Details & Download Feature ---
    with st.expander(T["prob_table"], expanded=False):
        col_table, col_btn = st.columns([3, 1])
        
        with col_table:
            st.dataframe(df_sorted.style.format({"Probability": "{:.2f}%"}), use_container_width=True)
            
        with col_btn:
            st.write("") # Spacing
            st.write("") # Spacing
            # Create downloadable CSV
            csv = df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=T["download"],
                data=csv,
                file_name=f"agriscan_report_{label}.csv",
                mime="text/csv",
                use_container_width=True
            )
