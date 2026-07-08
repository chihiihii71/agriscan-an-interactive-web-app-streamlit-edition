<div align="center">

# 🌿 AgriScan: An Interactive Web Application

### Vision Transformers with Self-Supervised Pretraining vs. CNN Architectures for Precision Agriculture

*A research-driven deep learning system for automated invasive weed classification using comparative CNN and Vision Transformer architectures.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://agriscan-deepweeds-deploy.streamlit.app/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Application-000000?style=for-the-badge&logo=flask&logoColor=white)](https://huggingface.co/spaces/Jaoooooo9/agriscan-interactive-web-app-flask-edition)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Hosting-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](Jaoooooo9/agriscan-interactive-web-app-flask-edition)
[![IEEE](https://img.shields.io/badge/Flask-Web%20Application-000000?style=for-the-badge&logo=flask&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11545838))

</div>

---

# 🔗 Live Demo

AgriScan is available through two independent deployment platforms designed for different user groups. The Streamlit application provides a research-oriented dashboard for visualizing model predictions and exporting results, while the Flask application offers a lightweight mobile interface with browser-based camera capture for field use.

| Application | Purpose | Link |
|-------------|---------|------|
| 🎛️ **Streamlit Dashboard** | Research dashboard with bilingual interface, probability visualization, and CSV export | **Open Streamlit App** |
| 📱 **Flask Web Application** | Mobile-friendly field application with live camera capture | **Open Flask App (https://huggingface.co/spaces/Jaoooooo9/agriscan-interactive-web-app-flask-edition)** |
| 📄 **IEEE Publication** | Published conference paper describing the comparative study | **DOI: 10.1109/QPAIN69676.2026.11545838 ([https://ieeexplore.ieee.org/abstract/document/11545838))](https://doi.org/10.1109/QPAIN69676.2026.11545838)** |

<div align="center">

### 🚀 Try the Live Classifier

*(Insert your existing "Try AgriScan Now" badge here.)*

</div>

---

# 🌱 Project Overview

Accurate weed identification is a critical component of precision agriculture. Misidentifying invasive weeds can lead to crop damage, unnecessary herbicide application, increased production costs, and reduced agricultural yield. Traditional identification methods rely heavily on botanical expertise and manual field inspection, making them impractical for many farmers and agricultural practitioners.

**AgriScan** was developed to investigate whether modern deep learning architectures can improve automated weed classification while remaining practical for real-world deployment. Rather than evaluating a single model, this project presents a controlled comparative study between four convolutional neural network (CNN) architectures and a Vision Transformer (ViT) pretrained using DINO self-supervised learning.

All models were trained and evaluated using identical experimental conditions on the **DeepWeeds** dataset to ensure a fair comparison. Performance was assessed using multiple evaluation metrics, including accuracy, precision, recall, F1-score, ROC analysis, confusion matrices, and Grad-CAM visualizations.

The highest-performing architecture, **ResNeSt50d**, achieved **98.23% classification accuracy** and was subsequently deployed through two complementary web applications:

- **Streamlit Dashboard** — designed for researchers, students, and agricultural analysts.
- **Flask Web Application** — designed for farmers and field workers with support for live smartphone camera capture.

Together, these components demonstrate how a research-oriented machine learning pipeline can be transformed into an accessible, production-ready decision support system for precision agriculture.

---

# ✨ Key Features

### 🤖 Deep Learning Research

- Comparative evaluation of **five deep learning architectures** under identical experimental conditions.
- Evaluation of a **DINO self-supervised Vision Transformer** alongside established CNN architectures.
- Comprehensive performance assessment using multiple quantitative and qualitative evaluation metrics.
- Grad-CAM–based interpretability analysis to visualize model decision regions.
- Selection of the best-performing model for deployment based on experimental evidence.

---

### 🌾 Weed Classification

- Classification of **nine DeepWeeds categories**.
- Detection of **eight invasive weed species** and one **Negative (background)** class.
- Designed for realistic field environments with varying lighting and background conditions.
- Supports automated weed identification for precision agriculture applications.

---

### 💻 Interactive Applications

- Research-oriented **Streamlit dashboard**.
- Mobile-friendly **Flask web application**.
- Browser-based **live camera capture**.
- Standard image upload.
- Interactive probability visualization.
- CSV report generation.
- Responsive user interface.

---

### 🌍 Accessibility

- Full bilingual interface (**English / বাংলা**).
- Shared AI inference engine across both applications.
- Model hosted through Hugging Face Hub.
- Cross-platform deployment accessible from desktop and mobile devices.

---

### 📊 Research Contributions

- Controlled comparison between CNN and Vision Transformer architectures.
- Investigation of self-supervised pretraining for agricultural image classification.
- Reproducible evaluation methodology.
- Deployment of the best-performing model into real-world applications.
- Published IEEE conference paper describing the complete research workflow.

---

# 🏗️ Research Pipeline

<div align="center">

![AgriScan Research Pipeline](docs/agriscan_research_pipeline.png)

**Figure 1.** End-to-end research workflow for training, evaluating, and selecting the best-performing deep learning model.

</div>

---

## Overview

Figure 1 illustrates the complete research workflow followed throughout this study. The pipeline begins with the **DeepWeeds** dataset, where all images undergo a consistent augmentation process before being used to train five deep learning architectures under identical experimental settings. Each model is evaluated using the same training, validation, and testing protocol to ensure a fair comparison. Following quantitative evaluation and qualitative interpretability analysis, the highest-performing model (**ResNeSt50d**) is selected and prepared for deployment in both the Streamlit dashboard and the Flask web application.

The standardized workflow ensures that performance differences arise from the model architectures themselves rather than inconsistencies in data preparation, evaluation methodology, or deployment strategy.

---

# 🌾 Dataset

The models developed in this project were trained and evaluated using the **DeepWeeds** dataset, a widely recognized benchmark for invasive weed classification. Unlike laboratory-controlled datasets, DeepWeeds contains real field photographs captured under varying environmental conditions, allowing the models to learn robust visual representations that generalize more effectively to practical agricultural scenarios.

---

## Dataset Overview

| Property | Description |
|-----------|-------------|
| **Dataset** | DeepWeeds |
| **Total Images** | 17,509 |
| **Number of Classes** | 9 |
| **Target Classes** | 8 invasive weed species + 1 Negative class |
| **Image Type** | RGB field photographs |
| **Environment** | Natural agricultural and pastoral environments |
| **Primary Task** | Multi-class image classification |

---

## Weed Categories

The dataset consists of eight invasive weed species together with one **Negative** class. Including a background class enables the model to distinguish target weeds from ordinary vegetation, soil, and other non-target objects, reducing false-positive predictions during real-world deployment.

| Class | Botanical Name | Description |
|--------|----------------|-------------|
| **Chinee Apple** | *Ziziphus mauritiana* | Invasive shrub/tree species |
| **Lantana** | *Lantana camara* | Toxic invasive shrub |
| **Parkinsonia** | *Parkinsonia aculeata* | Thorny invasive tree |
| **Parthenium** | *Parthenium hysterophorus* | Highly invasive agricultural weed |
| **Prickly Acacia** | *Vachellia nilotica* | Thorn-forming invasive tree |
| **Rubber Vine** | *Cryptostegia grandiflora* | Smothering climbing vine |
| **Siam Weed** | *Chromolaena odorata* | Fast-growing perennial shrub |
| **Snake Weed** | *Stachytarpheta jamaicensis* | Herbaceous invasive weed |
| **Negative** | N/A | Background vegetation, soil, and non-target plants |

---

## Data Preparation

Before model training, all images were processed using a consistent preprocessing and augmentation pipeline to improve model robustness and reduce overfitting. Every architecture received the same processed dataset to ensure experimental fairness.

### Image Preprocessing

- Image resizing
- Center cropping
- Tensor conversion
- Image normalization

### Data Augmentation

- Random cropping
- Random horizontal flipping
- Color jitter
- Image resizing

---

## Experimental Protocol

To ensure a scientifically valid comparison between all five architectures, every model followed the same experimental procedure.

| Component | Configuration |
|-----------|---------------|
| **Dataset** | DeepWeeds |
| **Training Images** | Same dataset for every model |
| **Validation Strategy** | Independent validation split |
| **Testing Strategy** | Independent unseen test split |
| **Data Leakage** | Zero image overlap between splits |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score |
| **Visual Evaluation** | Confusion Matrix, ROC Curve, Accuracy/Loss Curves, Grad-CAM |

This controlled evaluation framework ensures that differences in performance are attributable to the learning capability of each architecture rather than inconsistencies in data preparation or evaluation methodology.

---

## Dataset Citation

If you use the DeepWeeds dataset in your own work, please cite the original publication.

### IEEE Citation

> A. Olsen, D. A. Konovalov, B. Philippa, P. Ridd, J. C. Wood, J. Johns, W. Banks, B. Girgenti, O. Kenny, J. Whinney, B. Calvert, M. Rahimi Azghadi, and R. D. White, "DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning," *Scientific Reports*, vol. 9, no. 2058, 2019.

### BibTeX

```bibtex
@article{DeepWeeds2019,
  author = {Alex Olsen and Dmitry A. Konovalov and Bronson Philippa and Peter Ridd and Jake C. Wood and Jamie Johns and Wesley Banks and Benjamin Girgenti and Owen Kenny and James Whinney and Brendan Calvert and Mostafa {Rahimi Azghadi} and Ronald D. White},
  title = {DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning},
  journal = {Scientific Reports},
  volume = {9},
  number = {2058},
  year = {2019},
  doi = {10.1038/s41598-018-38343-3}
}
```

---

# 📈 Experimental Evaluation

Following the research pipeline described in Figure 1, all five architectures were trained and evaluated using the same dataset, preprocessing pipeline, train/validation/test splits, and evaluation metrics. This standardized experimental design ensures that the reported performance differences are attributable to the model architectures rather than inconsistencies in data preparation or evaluation methodology.

Model performance was assessed using both **quantitative metrics** and **qualitative visual analysis**, providing a comprehensive comparison of classification accuracy, generalization capability, and interpretability.

---

## 📊 Evaluation Metrics

The following evaluation criteria were used throughout the comparative study.

| Category | Metrics |
|-----------|---------|
| **Classification Performance** | Accuracy, Precision, Recall, F1-score |
| **Class-wise Performance** | Per-class Precision, Recall, F1-score |
| **Visual Evaluation** | Confusion Matrix, ROC Curve |
| **Training Behaviour** | Accuracy & Loss Curves |
| **Model Interpretability** | Grad-CAM |

---

# 🏆 Model Comparison Results

The table below summarizes the overall performance achieved by each architecture under identical experimental conditions.

| Model | Architecture | Accuracy | Precision | Recall | Macro F1 |
|--------|--------------|----------|-----------|--------|----------|
| 🥇 **ResNeSt50d** | CNN (Split Attention) | **98.23%** | **98.0%** | **98.0%** | **98.0%** |
| ECAResNet50d | CNN (Efficient Channel Attention) | 96.38% | 96.0% | 94.0% | 95.0% |
| ViT Base Patch16 224 DINO | Vision Transformer (Self-Supervised) | 94.00% | 91.0% | 91.0% | 91.0% |
| ResNet50ViT | CNN + Vision Transformer Hybrid | 93.00% | 90.0% | 91.0% | 90.0% |
| ResNet50 + CBAM | CNN (Attention Module) | 87.00% | 80.0% | 90.0% | 84.0% |

---

# 📖 Performance Analysis

The comparative study demonstrates that **ResNeSt50d** consistently achieved the strongest overall performance across all evaluation metrics, reaching **98.23% classification accuracy** while maintaining balanced precision, recall, and macro F1-score. Its split-attention mechanism enabled the network to capture fine-grained discriminative features that effectively distinguish visually similar invasive weed species under real-world field conditions.

Among the transformer-based architectures, the **ViT Base Patch16 224 DINO** model achieved the highest performance. The results indicate that self-supervised pretraining provides a measurable advantage over transformer models trained without self-supervised initialization, allowing the Vision Transformer to outperform the CNN–Transformer hybrid architecture evaluated in this study.

Although the Vision Transformer demonstrated competitive performance, the experimental results suggest that convolutional architectures remain better suited to the DeepWeeds dataset. CNNs incorporate spatial inductive biases directly into their architecture, enabling efficient feature learning from moderate-sized datasets. Vision Transformers, in contrast, generally benefit from substantially larger training datasets before consistently surpassing convolutional models.

Rather than demonstrating a limitation of self-supervised learning, these findings highlight the effectiveness of DINO pretraining in improving transformer performance while also illustrating that dataset scale remains an important factor influencing the relative performance of CNN and Vision Transformer architectures for agricultural image classification.

---

## 🔑 Key Findings

- ResNeSt50d achieved the highest overall classification accuracy (**98.23%**).
- Split-attention CNNs remained the most effective architecture for the DeepWeeds dataset.
- DINO self-supervised pretraining improved Vision Transformer performance compared with the CNN–Transformer hybrid.
- All models were evaluated under identical experimental conditions to ensure a fair comparison.
- Grad-CAM visualizations confirmed that the highest-performing models focused primarily on biologically meaningful plant structures.

---

# 🔍 Model Interpretability

Accurate predictions alone are insufficient for trustworthy AI systems. To better understand how each architecture reached its predictions, Grad-CAM (Gradient-weighted Class Activation Mapping) was applied across all evaluated models.

Grad-CAM generates visual explanations by highlighting the image regions that contribute most strongly to the model's prediction. These visualizations provide additional evidence that the classifier is learning meaningful botanical characteristics rather than relying on irrelevant background features.

Across the comparative study, **ResNeSt50d** consistently produced the most localized and biologically relevant activation maps, reinforcing the quantitative evaluation results and providing greater confidence in its suitability for deployment within practical agricultural applications.

---

## 📌 Experimental Summary

The experimental evaluation demonstrates that the proposed comparative framework successfully identified the most effective architecture for invasive weed classification on the DeepWeeds dataset. While self-supervised Vision Transformers showed promising performance and benefited substantially from DINO pretraining, the split-attention ResNeSt50d architecture achieved the best overall balance of accuracy, robustness, and interpretability. Consequently, ResNeSt50d was selected as the production model for deployment in the AgriScan web applications.

---

# 🚀 Deployment Architecture

<div align="center">

![AgriScan Deployment Architecture](docs/agriscan_deployment_architecture.png)

**Figure 2.** Shared deployment architecture illustrating how both web applications utilize the same trained ResNeSt50d inference engine while providing different user experiences.

</div>

---

## Overview

Following experimental evaluation, the highest-performing architecture (**ResNeSt50d**) was selected for deployment as the production model. Rather than developing separate machine learning models for different interfaces, AgriScan adopts a **shared inference architecture**, where both deployed applications use the same trained model and prediction pipeline.

Although the **Streamlit Dashboard** and **Flask Web Application** provide different user experiences, they perform identical preprocessing, inference, and prediction operations. This design guarantees consistent classification results regardless of which application is used while simplifying model maintenance and future updates.

The trained model is hosted on **Hugging Face Hub**, allowing both applications to automatically retrieve the latest production model during startup without requiring manual distribution of model weights.

---

# 💻 Deployed Applications

AgriScan provides two independent applications designed for different user groups while sharing the same AI inference engine.

---

## 🎛️ Streamlit Dashboard

The Streamlit application is intended for researchers, students, agricultural analysts, and demonstration purposes. It provides an interactive dashboard for exploring model predictions and exporting classification results.

### Features

- Image upload interface
- Interactive probability visualization
- Prediction confidence display
- Downloadable CSV prediction reports
- Bilingual interface (English / Bangla)
- Responsive dashboard layout
- Automatic model loading from Hugging Face Hub

### Typical Workflow

```text
Upload Image
      │
      ▼
Image Preprocessing
      │
      ▼
ResNeSt50d Inference
      │
      ▼
Probability Prediction
      │
      ▼
Interactive Dashboard
      │
      ▼
CSV Report Export
```

---

## 📱 Flask Web Application

The Flask application is designed primarily for practical field use, enabling users to identify weeds directly from smartphones or other mobile devices without installing a dedicated application.

### Features

- Image upload
- Browser-based live camera capture
- Mobile-friendly interface
- Interactive confidence visualization
- REST API support
- Bilingual interface (English / Bangla)
- Automatic model loading from Hugging Face Hub

### Typical Workflow

```text
Camera Capture
     or
Image Upload
      │
      ▼
Image Preprocessing
      │
      ▼
ResNeSt50d Inference
      │
      ▼
Probability Prediction
      │
      ▼
Prediction Results
```

---

# ⚙️ Shared AI Inference Engine

Both deployed applications rely on the same production inference pipeline to ensure identical prediction behaviour across platforms.

| Stage | Description |
|--------|-------------|
| **Model Loading** | Downloads the trained ResNeSt50d model from Hugging Face Hub during application startup. |
| **Image Preprocessing** | Resizes, crops, converts, and normalizes input images before inference. |
| **Model Inference** | Performs forward propagation using the trained PyTorch model. |
| **Probability Estimation** | Applies Softmax to generate confidence scores for all nine classes. |
| **Prediction Output** | Returns the predicted weed species together with class probabilities for presentation within each application. |

---

# 🔄 Shared Deployment Workflow

Although the Streamlit dashboard and Flask application provide different interfaces, both applications execute the same prediction pipeline.

```text
User Input
(Image Upload / Camera Capture)
              │
              ▼
Image Preprocessing
              │
              ▼
Shared ResNeSt50d Model
              │
              ▼
Softmax Probability Calculation
              │
              ▼
Prediction Results
              │
      ┌───────┴────────┐
      ▼                ▼
Streamlit UI      Flask UI
```

---

# 🌍 Deployment Highlights

- Single production model shared across both applications.
- Consistent prediction pipeline across deployment platforms.
- Automatic model retrieval from Hugging Face Hub.
- Cross-platform accessibility for desktop and mobile devices.
- Bilingual user interface for improved accessibility.
- Modular architecture that supports future deployment targets such as mobile applications, edge devices, or cloud-based APIs.

---

# 🛠️ Technology Stack

AgriScan combines modern deep learning frameworks, web technologies, and cloud deployment platforms to provide an end-to-end research and deployment pipeline.

---

## 🧠 Research & Model Development

| Category | Technology |
|----------|------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | PyTorch |
| **Model Library** | TIMM (PyTorch Image Models) |
| **CNN Architectures** | ResNeSt50d, ECAResNet50d, ResNet50 + CBAM |
| **Vision Transformer** | ViT Base Patch16 224 DINO |
| **Hybrid Architecture** | ResNet50ViT |
| **Dataset** | DeepWeeds |
| **Interpretability** | Grad-CAM |
| **Evaluation** | Scikit-learn |
| **Experiment Environment** | Jupyter Notebook |

---

## 🎛️ Streamlit Dashboard

| Category | Technology |
|----------|------------|
| **Framework** | Streamlit |
| **Inference Engine** | PyTorch |
| **Model Library** | TIMM |
| **Visualization** | Plotly Express |
| **Data Processing** | Pandas |
| **Image Processing** | Pillow |
| **Model Hosting** | Hugging Face Hub |
| **Deployment Platform** | Streamlit Community Cloud |
| **Localization** | English / Bangla |

---

## 📱 Flask Web Application

| Category | Technology |
|----------|------------|
| **Framework** | Flask |
| **Inference Engine** | PyTorch |
| **Model Library** | TIMM |
| **Image Processing** | Pillow |
| **Visualization** | Plotly.js |
| **Camera Integration** | Browser MediaDevices API |
| **Model Hosting** | Hugging Face Hub |
| **Deployment Platform** | Hugging Face Spaces |
| **Localization** | English / Bangla |

---

## ☁️ Deployment Services

| Service | Purpose |
|----------|---------|
| **Hugging Face Hub** | Model weight hosting |
| **Hugging Face Spaces** | Flask application deployment |
| **Streamlit Community Cloud** | Dashboard deployment |
| **GitHub** | Source code management |

---

# ⚡ Installation

The following instructions allow you to run both AgriScan applications locally.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/chihiihii71/agriscan-deepweeds.git

cd agriscan-deepweeds
```

---

## 2️⃣ Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

### Streamlit Dashboard

```bash
pip install streamlit \
torch \
torchvision \
timm \
pandas \
plotly \
Pillow \
huggingface_hub
```

---

### Flask Web Application

```bash
pip install flask \
torch \
torchvision \
timm \
Pillow \
huggingface_hub
```

---

## 4️⃣ Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 5️⃣ Run the Flask Application

```bash
python app.py
```

---

## Automatic Model Loading

Both applications automatically download the trained **ResNeSt50d** model from **Hugging Face Hub** during startup.

No manual download of model weights is required.

---

# 📂 Project Structure

The repository is organized into separate research and deployment components to simplify development and maintenance.

```text
agriscan-deepweeds/

│
├── 📁 streamlit_app/
│   └── app.py
│
├── 📁 flask_app/
│   ├── app.py
│   ├── templates/
│   │      ├── index.html
│   │      └── result.html
│   │
│   └── static/
│          └── style.css
│
├── 📁 research/
│   ├── grad-cam.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── 📁 docs/
│   ├── agriscan_research_pipeline.png
│   ├── agriscan_deployment_architecture.png
│   └── screenshots/
│
├── README.md
│
├── requirements.txt
│
└── LICENSE
```

---

## 📁 Repository Organization

| Directory | Purpose |
|-----------|---------|
| **streamlit_app/** | Research-oriented Streamlit dashboard |
| **flask_app/** | Mobile-friendly Flask web application |
| **research/** | Model training, evaluation, and Grad-CAM notebooks |
| **docs/** | Architecture diagrams, screenshots, and documentation assets |
| **README.md** | Project documentation |
| **requirements.txt** | Python package dependencies |

---

## 💡 Development Workflow

The repository follows a modular workflow that separates research from deployment.

```text
Research
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Model Upload
(Hugging Face Hub)
     │
     ▼
Deployment
     │
 ┌───┴────────────┐
 ▼                ▼

Streamlit      Flask
Dashboard      Web App
```

This modular organization allows future model improvements to be deployed simply by replacing the production model stored on Hugging Face Hub, without requiring significant modifications to either application.

---

# 🌍 Real-World Applications

Although AgriScan was developed as a comparative deep learning research project, its architecture and deployment strategy make it suitable for practical agricultural applications. By combining high-accuracy image classification with accessible web interfaces, the system demonstrates how modern artificial intelligence can assist farmers, researchers, and agricultural organizations in improving weed management and decision-making.

Potential application areas include:

- 🌱 Precision agriculture and smart farming
- 🚜 Early detection of invasive weed species
- 🌾 Crop monitoring and field inspection
- 📷 Mobile-assisted weed identification
- 🛰️ Future drone-based agricultural monitoring
- 🧪 Agricultural AI research and education
- 🏫 Teaching computer vision and deep learning concepts
- 🌍 Environmental and ecological monitoring

---

# ⚠️ Current Limitations

While AgriScan demonstrates strong classification performance, several limitations should be considered when interpreting the results.

- The model is trained exclusively on the **DeepWeeds** dataset.
- Only **nine image classes** are currently supported.
- Performance may decrease for severely blurred or low-resolution images.
- Weed species outside the DeepWeeds taxonomy cannot be recognized.
- The deployed model performs image classification rather than object detection or segmentation.
- Browser camera functionality depends on device compatibility and user permission.

These limitations provide opportunities for future improvements and broader real-world deployment.

---

# 🚀 Future Work

Several extensions can further improve the capabilities of AgriScan.

### Model Improvements

- Train on larger and more diverse agricultural datasets.
- Investigate newer Vision Transformer architectures.
- Explore ensemble learning between CNN and Transformer models.
- Apply model compression and quantization for faster inference.

---

### Application Improvements

- Native Android and iOS applications.
- Offline inference for remote agricultural environments.
- Cloud-based REST API for third-party integration.
- Multi-image batch prediction.
- User account management and prediction history.

---

### Research Directions

- Weed detection using object detection frameworks.
- Semantic segmentation for dense weed localization.
- Multi-label agricultural disease classification.
- Drone and UAV image analysis.
- Continual learning for newly emerging weed species.

---

# 📄 Publication

The research presented in this repository has been published in an IEEE international conference.

> **Vision Transformers for Image Classification with Self-Supervised Pretraining: A Comparative Study on DeepWeeds**

**Conference**

IEEE QPAIN 2026

**DOI**

```text
10.1109/QPAIN69676.2026.11545838
```

The publication provides a detailed discussion of the experimental methodology, comparative analysis, and evaluation results that form the foundation of the AgriScan system.

---

# 📚 Citation

If you use this repository, the trained models, or the accompanying methodology in your research, please cite our work.

```bibtex
@inproceedings{YOUR_CITATION_HERE,
  title={Vision Transformers for Image Classification with Self-Supervised Pretraining: A Comparative Study on DeepWeeds},
  author={...},
  booktitle={IEEE QPAIN},
  year={2026},
  doi={10.1109/QPAIN69676.2026.11545838}
}
```


# 👨‍💻 Authors

| Name | Contribution |
|------|--------------|
| **Your Name** | Research, model development, deployment, documentation |
| **Co-author(s)** | Research supervision / collaboration *(if applicable)* |

---

# 🙏 Acknowledgements

This project would not have been possible without the contributions of the following open-source communities and research initiatives.

### Dataset

- DeepWeeds Dataset

### Deep Learning

- PyTorch
- TIMM (PyTorch Image Models)

### Model Hosting

- Hugging Face Hub

### Web Frameworks

- Streamlit
- Flask

### Scientific Computing

- NumPy
- Pandas
- Scikit-learn
- Plotly
- Pillow

The authors also acknowledge the original DeepWeeds dataset creators and the maintainers of the open-source software used throughout this research.

---

<div align="center">

## ⭐ Support the Project

If you found AgriScan useful for your research, coursework, or agricultural applications, please consider giving this repository a **⭐ Star** on GitHub.

Your support helps improve the visibility of open-source research and encourages future development.

---

**Built with ❤️ using PyTorch, TIMM, Streamlit, Flask, and Hugging Face.**

</div>


