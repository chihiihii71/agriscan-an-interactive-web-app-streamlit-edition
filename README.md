<div align="center">

#  AgriScan — An Interactive Web App

### Vision Transformers with Self-Supervised Pretraining vs. CNN Architectures for Precision Agriculture

[![IEEE Published](https://img.shields.io/badge/IEEE-QPAIN%202026-00629B?style=for-the-badge&logo=ieee&logoColor=white)]([https://doi.org/10.1109/QPAIN69676.2026.11545838](https://ieeexplore.ieee.org/abstract/document/11545838))
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://agriscan-deepweeds-deploy.streamlit.app/)
[![Flask App](https://img.shields.io/badge/Flask-Live%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://huggingface.co/spaces/Jaoooooo9/agriscan-interactive-web-app-flask-edition)

**Published at the 2026 IEEE 2nd International Conference on Quantum Photonics,
Artificial Intelligence, and Networking (QPAIN)** · 16–18 April 2026 · Chittagong, Bangladesh

</div>

---

## 🔗 Live Demo

| Application | Purpose | Link |
|---|---|---|
| 🎛️ Streamlit Dashboard | Research-style comparison interface, bilingual, CSV export | [Open Streamlit App](https://agriscan-deepweeds-deploy.streamlit.app/) |
| 📱 Flask Field App | Consumer-facing mobile app with live camera capture | [Open Flask App](https://huggingface.co/spaces/Jaoooooo9/agriscan-interactive-web-app-flask-edition) |
| 📄 Published Paper | IEEE QPAIN 2026 | [DOI: 10.1109/QPAIN69676.2026.11545838](https://doi.org/10.1109/QPAIN69676.2026.11545838) |

<div align="center">

[![Try AgriScan Now](https://img.shields.io/badge/AgriScan-Try%20the%20Live%20Classifier-16a34a?style=for-the-badge&logo=streamlit&logoColor=white)](https://agriscan-deepweeds-deploy.streamlit.app/)

</div>

---

## 1. Why This Project

Invasive weed species cause direct crop loss when misidentified or removed incorrectly.
A farmer who cannot distinguish an invasive weed from a young crop shoot risks cutting the
wrong plant — damaging yield instead of protecting it. Manual identification is slow, requires
expert botanical knowledge most smallholder farmers do not have access to, and does not scale
across large fields.

**The goal of this project was twofold:**

1. Build an accurate, automated weed classifier that any farmer — regardless of technical or
   botanical expertise — can use by simply pointing a camera at a plant.
2. Investigate whether a newer, less commonly applied architecture (Vision Transformers with
   Self-Supervised Pretraining) could offer advantages over the CNN architectures that dominate
   existing weed-detection literature, rather than defaulting to the standard approach without
   question.

This is not just a model-training exercise — it is a full pipeline from research question,
through rigorous comparison, to two deployed applications a real farmer can open on a phone.

---

## 2. Why CNN Architectures Were Used as the Comparison Baseline

Existing weed and crop classification literature is overwhelmingly CNN-based. Prior work
reviewed for this study includes ResNet-50 and Inception-v3 (Olsen et al., 95.7% accuracy),
VGG16 and YOLOv3 (Ahmad et al., up to 98.90%), GoogleNet (Syechu et al., 92.38%), and
ensemble CNN approaches across Xception, DenseNet, MobileNetV3, and EfficientNet (Sunil et al.).

CNNs were selected as the comparison baseline because:

- They represent the **established, validated standard** in the weed-detection literature
- Their inductive biases — locality and translation invariance, built directly into the
  convolution operation — make them naturally data-efficient on moderate-sized datasets
- Without a CNN baseline, there is no way to know whether a newer architecture is actually
  an improvement or simply different

Four CNN-family architectures were trained for this comparison: **ResNeSt50d** (split-attention
residual network), **ECAResNet50d** (efficient channel attention), **Custom ResNet50 with CBAM**
(channel + spatial attention module), and **ResNet50ViT** (a CNN-Transformer hybrid).

---

## 3. Why Vision Transformers with Self-Supervised Pretraining (ViT-SSL)

Most weed-detection systems default to CNNs without testing whether newer architectures offer
a real advantage. This project deliberately introduces a **ViT Base Patch16 224 DINO** model —
a Vision Transformer pretrained using DINO self-supervised learning — as a genuine alternative
approach, not a token comparison.

**Why ViT-SSL specifically, rather than a standard supervised ViT:**

- Standard ViTs typically require large *labeled* datasets to perform well — a constraint
  agricultural datasets rarely meet
- **DINO self-supervised pretraining** allows the transformer to learn general visual
  representations from *unlabeled* image patches first, building rich 768-dimensional feature
  vectors per patch, before fine-tuning on the labeled DeepWeeds classes
- This two-stage approach (self-supervised pretraining → supervised fine-tuning) is
  specifically suited to domains like agriculture where labeled data is expensive to collect
  but unlabeled field imagery is comparatively abundant
- Recent literature (Rozendo et al., 2024) shows ViT and hybrid CNN-ViT-PVT ensembles achieving
  strong results on weed classification when attention mechanisms are properly leveraged —
  motivating a direct empirical test on this specific dataset

---

## 4. How the Comparison Was Conducted

Every model — CNN and ViT-SSL alike — was evaluated under **identical experimental conditions**
to ensure the comparison is scientifically valid, not just a side-by-side accuracy number:

| Comparison Control | Applied To All 5 Models |
|---|---|
| Dataset | Same DeepWeeds dataset — 17,509 images, 9 classes |
| Data Splits | Independent train / validation / test sets, zero image overlap (no data leakage) |
| Preprocessing | Same augmentation pipeline (crop, resize, color jitter, flip) |
| Metrics | Precision, Recall, F1-score — per class, macro average, and weighted average |
| Visual Diagnostics | ROC curves, confusion matrices, accuracy-loss-epoch curves |
| Interpretability | Grad-CAM activation mapping for every model |
| Class Fairness | Explicit per-class analysis to isolate the effect of the "Negative" (background) class on multi-class accuracy |

This controlled methodology is what makes the comparison meaningful — the same evaluation
lens was applied to a pure CNN, an attention-augmented CNN, a CNN-ViT hybrid, and a pure
ViT-SSL model.

---

## 5. Methodology Pipeline

The full research-to-deployment pipeline is visualised in the system architecture diagram in
Section 14. In summary: the DeepWeeds dataset is augmented once (cropping, resizing, color
jitter, flipping), then split across all five architectures. The four CNN variants train
directly on the augmented images, while the ViT-SSL pipeline first splits each image into 196
patches, processes them through the DINO-pretrained transformer to generate 768-dimensional
feature vectors per patch, and fine-tunes on the labeled DeepWeeds classes. Every model is then
evaluated under identical conditions — F1 score, confusion matrix, accuracy-loss curves, ROC
curve, and Grad-CAM — before the best performer is selected for deployment.

---

## 6. Full Model Comparison Results

| Model | Architecture Type | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| 🏆 **ResNeSt50d** | CNN — Split Attention | **98.23%** | **98.0%** | **98.0%** | **98.0%** |
| ECAResNet50d | CNN — Efficient Channel Attention | 96.38% | 96.0% | 94.0% | 95.0% |
| **ViT Base Patch16 224 DINO** | **Vision Transformer — SSL Pretrained** | **94.00%** | 91.0% | 91.0% | 91.0% |
| ResNet50ViT | Hybrid — CNN + ViT (no SSL) | 93.00% | 90.0% | 91.0% | 90.0% |
| ResNet50 + CBAM | CNN — Channel + Spatial Attention | 87.00% | 80.0% | 90.0% | 84.0% |

---

## 7. Why ViT-SSL Did Not Surpass the Top CNN Accuracy — What the Comparison Revealed

This is the most important analytical finding of the study, and it deserves a precise
explanation rather than a simple "CNN won" conclusion.

**The result that matters most is not that ViT-SSL finished second — it is *where* it finished
second.** ViT Base Patch16 224 DINO (94.00%) **outperformed the ResNet50ViT hybrid (93.00%)** —
a model that also uses transformer attention, but without self-supervised pretraining. This
is a controlled, direct comparison: same transformer mechanism, the only difference is DINO
pretraining. The result isolates and confirms that **self-supervised pretraining provided a
measurable, real improvement** to transformer performance on this exact dataset.

**Why it still did not surpass ResNeSt50d's 98.23%:**

Vision Transformers lack the convolutional inductive biases — locality and translation
invariance — that are architecturally built into CNNs. This means ViTs must *learn* spatial
relationships from data rather than having them assumed by the architecture. This is well
documented in transformer literature: ViTs typically require pretraining or fine-tuning at a
scale of hundreds of thousands to millions of images to outperform CNNs; below that scale,
CNNs generally retain the advantage.

The DeepWeeds dataset used here — 17,509 images across 9 classes — sits well below that
threshold. ResNeSt50d's split-attention mechanism, layered on top of convolutional locality
bias, is architecturally better matched to a dataset of this size for capturing the
fine-grained visual patterns that distinguish visually similar weed species.

**The honest conclusion:** this is not evidence that ViT-SSL is the wrong approach — it is
evidence of *exactly where* the data-scale threshold sits for this architecture on this task.
Fine-tuning improved the model's performance to the point of beating a comparable
non-SSL transformer hybrid; it did not, and was not expected to, eliminate a fundamental
data-scale requirement that is a known property of the architecture family, not a flaw in
this implementation.

---

## 8. Interpretability — Grad-CAM Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was generated for every model to
visualise which image regions drove each classification decision. This step exists because
accuracy numbers alone do not build trust with end users — a farmer needs to know the model
is looking at the actual plant, not background soil or lighting artifacts.

Grad-CAM analysis across all 9 weed classes confirmed ResNeSt50d produced the sharpest and
most consistent activation regions, reinforcing the quantitative result and providing visual
evidence that the model's decisions are grounded in genuine plant features.

---

## 9. What Product Was Built From This Research

While five architectures were evaluated during the research phase, only the empirically
best-performing model — **ResNeSt50d at 98.23% accuracy** — was selected for production
deployment. This is a deliberate engineering decision: farmers interact with the single most
reliable classifier available, not a research comparison tool.

That one model was then deployed as **two distinct applications**, each built for a different
purpose and a different user.

---

## 10. Why Two Separate Applications — Streamlit *and* Flask

Building both was not redundant — each serves a different real-world use case:

| | 🎛️ Streamlit App | 📱 Flask App |
|---|---|---|
| **Primary user** | Researchers, agritech evaluators | Farmers, field workers |
| **Input method** | File upload only | File upload **or** live camera capture |
| **Design style** | Clean data-dashboard layout | Glassmorphism, mobile-first, polished consumer UI |
| **Visualization** | Server-rendered bar chart (Plotly Python) | Client-rendered donut chart (Plotly.js) with confidence at center |
| **Data export** | CSV report download | Not included — optimized for quick field use, not analysis |
| **Language support** | English / বাংলা toggle | English / বাংলা toggle |
| **Best suited for** | Comparing results, exporting data for further analysis | Standing in a field, photographing a plant directly, getting an instant answer |

The Flask app's live camera capture is the critical field-usability feature — a farmer does
not need to have a saved photo; they open the page on their phone, tap "Open Camera," and
photograph the plant directly. This mirrors how the app would realistically be used in
actual agricultural conditions, over a mobile connection, with no app installation required.

---

## 11. How the Streamlit Application Works

- Downloads the winning ResNeSt50d model weights from Hugging Face Hub at startup and caches
  it in memory for the session, so inference stays fast without reloading on every interaction
- A single sidebar toggle switches every visible string — titles, labels, buttons — between
  English and বাংলা by swapping which language dictionary the interface reads from
- Class names are stored bilingually, so a farmer sees the plant's name in their own language
  rather than an unfamiliar English botanical term
- An uploaded image is resized and normalized, then passed through the cached model to produce
  class probabilities across all 9 species
- Results are displayed as a top prediction banner, a Plotly bar chart of all class
  probabilities, and an expandable detailed table
- Every prediction can be exported as a CSV file, useful for research logging or extension-office
  record keeping

---

## 12. How the Flask Application Works

- Requests access to the device's rear-facing camera directly in the browser, streaming live
  video into the page with no app installation required
- On capture, the current video frame is drawn to a hidden canvas and converted into a
  base64-encoded image string
- The backend accepts either a standard file upload or the captured camera image through the
  same form submission — both paths converge into one inference function
- Base64 camera captures are decoded server-side into a standard image object and processed
  identically to an uploaded file, so no logic is duplicated between input methods
- Language selection is carried as a URL parameter that persists through the upload, the
  results page, and the "Scan Another Image" action
- The results page renders a donut chart client-side, with the confidence percentage annotated
  directly in the center
- Custom glassmorphism styling gives the app a distinct, polished, consumer-facing identity,
  separate from the Streamlit dashboard's data-focused look

---

## 13. Tech Stack

### Research & Model Training

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning Framework | PyTorch |
| Model Library | TIMM (PyTorch Image Models) |
| CNN Architectures | ResNeSt50d, ECAResNet50d, ResNet50 + CBAM |
| Transformer Architecture | ViT Base Patch16 224 (DINO self-supervised pretraining) |
| Hybrid Architecture | ResNet50ViT |
| Interpretability | Grad-CAM |
| Evaluation | scikit-learn (Precision, Recall, F1, ROC, Confusion Matrix) |
| Experiment Environment | Jupyter Notebook |

### Streamlit Application

| Layer | Technology |
|---|---|
| Framework | Streamlit |
| Inference | PyTorch, TIMM |
| Visualization | Plotly Express |
| Data Handling | Pandas |
| Model Hosting | Hugging Face Hub (`hf_hub_download`) |
| Deployment | Streamlit Community Cloud |
| Localization | Bilingual English / Bangla (custom dictionary-based) |

### Flask Application

| Layer | Technology |
|---|---|
| Framework | Flask |
| Inference | PyTorch, TIMM |
| Visualization | Plotly.js (client-side) |
| Image Handling | Pillow, Base64 encoding |
| Camera Capture | Browser MediaDevices API (`getUserMedia`) |
| Styling | Custom CSS — glassmorphism design |
| Model Hosting | Hugging Face Hub (`hf_hub_download`) |
| Deployment | Hugging Face Spaces |
| Localization | Bilingual English / Bangla (query-parameter based) |

---

## 14. System Architecture

<div align="center">

![AgriScan System Architecture](docs/agriscan_architecture.png)

</div>

The diagram traces the full path from raw dataset to field-ready application: the DeepWeeds
dataset is augmented once, then split across all five architectures for training under
identical conditions. Every model passes through the same evaluation stage, from which the
best performer — ResNeSt50d — is selected and pushed to Hugging Face Hub. From there, the same
model weights serve two independent, purpose-built applications: a research-oriented Streamlit
dashboard and a farmer-facing Flask app with live camera capture.

---

## 15. Features

- **5-architecture comparative study**: ResNeSt50d, ECAResNet50d, ResNet50 + CBAM, ResNet50ViT, ViT Base Patch16 224 DINO, evaluated under identical experimental conditions
- **Grad-CAM interpretability** across all 9 weed classes for every model
- **Bilingual interface** (English / বাংলা) on both deployed applications — no farmer is excluded by language
- **Two deployment targets**: a research-oriented Streamlit dashboard and a mobile-first Flask field app
- **Live camera capture** directly in the browser — no app installation, works on any smartphone
- **CSV export** of classification results for research or extension-office record keeping
- **9-class detection**: Chinee Apple, Lantana, Parkinsonia, Parthenium, Prickly Acacia, Rubber Vine, Siam Weed, Snake Weed, and a Negative (background) class to prevent false positives on non-plant imagery
- **Data-leakage-free evaluation**: independent train/validation/test splits verified with zero image overlap

---

## 16. Real-World Industry Impact

**For individual farmers:** A smartphone camera becomes an instant weed-identification tool —
no botanical training required, no internet dependency beyond the initial page load, and no
language barrier, directly reducing the risk of removing the wrong plant and protecting crop
yield.

**For agricultural extension services:** The Streamlit dashboard's CSV export enables
extension officers to log classification results across many field visits, building a dataset
of regional weed prevalence over time — useful for targeted intervention planning.

**For agritech and precision-agriculture companies:** The 5-model comparative study provides
a decision-making reference for backbone selection — companies building drone-based or
edge-device weed-detection systems can weigh ResNeSt50d's accuracy against ECAResNet50d's
lighter computational footprint depending on their hardware constraints.

**For future automation:** Because inference in both applications is fully decoupled from the
user interface (a single PyTorch model, loaded once, called identically for both file uploads
and camera captures), the same classification core could be integrated into drone imaging
pipelines or IoT field-sensor systems with the interface layer swapped out entirely.

---

## 17. Setup & Installation

### Streamlit App

```bash
git clone https://github.com/chihiihii71/agriscan-deepweeds.git
cd agriscan-deepweeds

pip install streamlit torch torchvision timm Pillow pandas plotly huggingface_hub

streamlit run app.py
```

### Flask App

```bash
pip install flask torch torchvision timm Pillow huggingface_hub

python flask_app.py
```

Both applications download the ResNeSt50d model automatically from Hugging Face Hub at
startup — no manual model download required.

---

## 18. Project Structure

```
agriscan-deepweeds/
├── streamlit_app/
│   └── app.py                    ← Bilingual Streamlit dashboard
├── flask_app/
│   ├── app.py                    ← Flask backend with camera capture
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       └── style.css             ← Glassmorphism design
├── research/
│   └── grad-cam.ipynb            ← Full training, evaluation, comparison notebook
└── README.md
```

---

## 19. Limitations

- ViT-SSL performance is fundamentally constrained by dataset scale — 17,509 images sits
  below the threshold where transformer architectures typically overtake CNNs
- Model accuracy may degrade under heavy background noise, unusual lighting, or occlusion
  not represented in the training distribution
- The Flask app's camera capture requires browser permission and a secure (HTTPS) context —
  will not function over an insecure connection

---

## 20. Future Work

- Expand the training dataset toward the scale where ViT-SSL's transformer architecture can
  fully demonstrate its potential advantage over CNN inductive biases
- Explore hybrid ensemble strategies combining ResNeSt50d's spatial precision with ViT-SSL's
  global attention, rather than treating them as competing rather than complementary
- Extend the Flask application with offline-first capability (service workers) for use in
  areas with unreliable mobile connectivity
- Integrate the classification core directly into drone-based or edge-device field imaging
  systems

---

## 21. Publication

```
K. Nahar, S. I. S. Abir, M. M. Safa, K. M. S. Kamal and A. W. Reza, "DeepWeeds: Vision Transformers for Image Classification with Self-Supervised Pretraining," 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence & Networking (QPAIN), Chittagong, Bangladesh, 2026, pp. 1-6, doi: 10.1109/QPAIN69676.2026.11545838. keywords: {Modeling;Accuracy;Deep learning;Industrial plants;Plants (biology);Printing;Labeling;Measurement;Convolutional neural networks;Training;DeepWeeds;Vision Transformers;ResNeSt50d;ECAResNet50d;ViT Base Patch16 224 Dino Model;Accuracy;F1-Score;ROC curve;Grad-CAM},

```

```bibtex
@INPROCEEDINGS{11545838,
  author={Nahar, Kamrun and Abir, Shakib Ibna Sorowar and Safa, Mahia Mehrun and Kamal, K. M. Safin and Reza, Ahmed Wasif},
  booktitle={2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence & Networking (QPAIN)}, 
  title={DeepWeeds: Vision Transformers for Image Classification with Self-Supervised Pretraining}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Modeling;Accuracy;Deep learning;Industrial plants;Plants (biology);Printing;Labeling;Measurement;Convolutional neural networks;Training;DeepWeeds;Vision Transformers;ResNeSt50d;ECAResNet50d;ViT Base Patch16 224 Dino Model;Accuracy;F1-Score;ROC curve;Grad-CAM},
  doi={10.1109/QPAIN69676.2026.11545838}}

}
```

---

<div align="center">
<sub>East West University · Department of Computer Science and Engineering · Dhaka, Bangladesh</sub>
</div>
