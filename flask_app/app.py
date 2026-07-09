from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import timm
from huggingface_hub import hf_hub_download
import base64
from io import BytesIO

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Updated Class Names to reflect the DeepWeeds taxonomy
CLASS_NAMES = {
    "en": [
        "DeepWeeds (Chinee Apple)", 
        "DeepWeeds (Lantana)", 
        "DeepWeeds (Parkinsonia)", 
        "DeepWeeds (Parthenium)", 
        "DeepWeeds (Prickly Acacia)", 
        "DeepWeeds (Rubber Vine)", 
        "DeepWeeds (Siam Weed)", 
        "DeepWeeds (Snake Weed)", 
        "Not DeepWeeds (Negative)"
    ],
    "bn": [
        "আগাছা (চিনি অ্যাপল)", 
        "আগাছা (ল্যান্টানা)", 
        "আগাছা (পার্কিনসোনিয়া)", 
        "আগাছা (পার্থেনিয়াম)", 
        "আগাছা (প্রিকলি অ্যাকাসিয়া)", 
        "আগাছা (রাবার ভাইন)", 
        "আগাছা (সিয়াম উইড)", 
        "আগাছা (স্নেক উইড)", 
        "আগাছা নয় (নেগেটিভ)"
    ]
}

TEXT = {
    "en": {
        "title": "AgriScan : An Interactive Web App",
        "subtitle": "Upload a DeepWeeds image to identify its species and class probabilities.",
        "upload": "Analyze Image",
        "predicted": "Detected Species",
        "confidence": "AI Confidence"
    },
    "bn": {
        "title": "অ্যাগ্রিস্ক্যান: ‌এন ইন্টারঅ্যাকটিভ ওয়েব অ্যাপ",
        "subtitle": "আগাছা শনাক্ত করতে একটি ছবি আপলোড করুন এবং প্রজাতি ও নিশ্চিততার মাত্রা দেখুন।",
        "upload": "বিশ্লেষণ করুন",
        "predicted": "শনাক্তকৃত প্রজাতি",
        "confidence": "নিশ্চয়তার মাত্রা"
    }
}

# Load Model
model_path = hf_hub_download(repo_id="Jaoooooo9/agriscan-streamlit", filename="resnest50d_model.pth")
model = timm.create_model("resnest50d", pretrained=False, num_classes=9)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# WEB INTERFACE ROUTE
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get("lang", "en")
    if lang not in CLASS_NAMES: lang = "en"
        
    if request.method == "POST":
        lang = request.form.get("lang", lang)
        img = None
                
        if "file" in request.files and request.files["file"].filename != "":
            img = Image.open(request.files["file"].stream).convert("RGB")
                    
        elif "camera_image" in request.form and request.form["camera_image"] != "":
            img_data = request.form["camera_image"].split(",")[1]
            img = Image.open(BytesIO(base64.b64decode(img_data))).convert("RGB")
            
        if img is None:
            return "No image provided", 400
            
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # AI Prediction
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(input_tensor)
            probs = torch.softmax(preds, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100
            all_probs = (probs * 100).cpu().numpy().tolist()
        
        # UI visualization now uses the unique labels directly
        return render_template(
            "result.html",
            prediction=CLASS_NAMES[lang][pred_idx],
            confidence=f"{confidence:.2f}",
            all_probs=all_probs, 
            class_names=CLASS_NAMES[lang], 
            img_data=img_base64, 
            text=TEXT[lang], 
            lang=lang
        )
    return render_template("index.html", text=TEXT[lang], lang=lang)

# ---------------------------------------------------------
# REST API ROUTE
# ---------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    lang = request.args.get("lang", "en")
    if lang not in CLASS_NAMES: lang = "en"
    
    img = None
    try:
        if "file" in request.files and request.files["file"].filename != "":
            img = Image.open(request.files["file"].stream).convert("RGB")
                    
        elif request.is_json and "image" in request.json:
            img_data = request.json["image"]
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img = Image.open(BytesIO(base64.b64decode(img_data))).convert("RGB")
                    
        else:
            return jsonify({"success": False, "error": "No image provided."}), 400
        
        # AI Prediction
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(input_tensor)
            probs = torch.softmax(preds, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100
            all_probs = (probs * 100).cpu().numpy().tolist()
        
        # Direct dictionary creation using updated DeepWeeds labels
        prob_dict = {CLASS_NAMES[lang][i]: round(all_probs[i], 2) for i in range(9)}
        
        return jsonify({
            "success": True,
            "language": lang,
            "prediction": CLASS_NAMES[lang][pred_idx],
            "confidence_percent": round(confidence, 2),
            "probabilities": prob_dict
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)