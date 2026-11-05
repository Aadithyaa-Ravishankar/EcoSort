# EcoSort: Visual Waste Classification Assistant

[![Course](https://img.shields.io/badge/Course-21CSE454T%20Computer%20Vision-blue.svg)](file:///Users/aadithyaaravishankar/Desktop/EcoSort%20CV%20Project/PROJECT_REPORT.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Pipeline-100%25%20Classical%20CV-orange.svg)](#key-methods--algorithms)

> **Academic Course Project for 21CSE454T — Computer Vision**  
> An interactive, fully explainable, classical computer vision system for automated waste item segmentation, feature extraction, and recycling bin routing.  
> **Strict Academic Constraint:** Built **without black-box deep learning** (no YOLO, CNNs, or external APIs). Every classification decision is mathematically derived from algorithms taught in Units 1 to 5 of the university syllabus.

---

## 📌 Project Overview

**EcoSort** demonstrates how classical image processing and shape analysis can replace deep learning in controlled industrial waste sorting environments (e.g., recycling conveyor belts). The application segments waste items, extracts morphological and photometric features, and routes items to standard color-coded recycling bins with natural-language engineering justifications.

### Recycling Bin Classification Matrix

| Waste Category | Target Bin | Color Code | Primary Physical & Geometric Cues |
| :--- | :--- | :---: | :--- |
| **Metal / Tin Can** | **Blue Bin** *(Recyclable Metal)* | `#1565C0` | Cylindrical aspect ratio ($0.45 \le AR \le 1.35$), Hough circles, metallic specular peaks ($V > 180, S < 80$), smooth GLCM texture. |
| **Plastic Bottle** | **Yellow Bin** *(Recyclable Plastic)* | `#FBC02D` | High elongation ($AR < 0.55$ or $AR > 1.80$), bottle-neck contour profile, bottle cap circle, PET specular glints. |
| **Paper / Cardboard** | **Green/Blue Bin** *(Recyclable Fiber)* | `#2E7D32` | Straight parallel Hough lines, high rectangular extent ($> 0.75$), crease corners (Shi-Tomasi), matte surface ($SR < 0.04$). |
| **Organic Waste** | **Green Bin** *(Compostable)* | `#4CAF50` | Irregular boundary (low solidity & circularity), absence of straight lines/circles, high boundary entropy, organic hue. |

---

## 🔬 Key Methods & Algorithms (Units 1–5 Mapping)

The image processing engine strictly mirrors the 5 units of course **21CSE454T**:

1. **Stage 1: Image Processing Foundations (Unit 1)**
   - **Preprocessing:** RGB to Grayscale & HSV conversion, Gaussian Smoothing (`cv2.GaussianBlur`), Median Filtering (`cv2.medianBlur`), and CLAHE for illumination normalization.
   - **Thresholding:** Otsu’s Global Optimal Binarization (`cv2.THRESH_OTSU`) and Adaptive Gaussian Thresholding with border-sampling polarity detection.
   - **Edge & Corner Detection:** Canny Edge Detection (`cv2.Canny`), Sobel Gradient Vectors, and Shi-Tomasi Corner Detection (`cv2.goodFeaturesToTrack`).
   - **Mathematical Morphology:** Structuring elements ($3\times3$ to $9\times9$) for Erosion, Dilation, Opening, and Closing (`cv2.morphologyEx`).
   - **Texture Analysis:** Gray-Level Co-occurrence Matrix (GLCM) computing Contrast, Dissimilarity, Homogeneity, Energy, and Correlation via `skimage.feature`.

2. **Stage 2: Shapes & Regions Analysis (Unit 2)**
   - **Segmentation & Labelling:** 8-connected Connected Component Analysis (`cv2.connectedComponentsWithStats`) with configurable minimum area filtering.
   - **Medial Representation:** Euclidean Distance Transform (`cv2.distanceTransform`) and Morphological Skeletonization (`skimage.morphology.skeletonize`).
   - **Geometric Descriptors:** Contour Boundary Tracking (`cv2.findContours`), Perimeter, Area, Circularity ($4\pi A / P^2$), Aspect Ratio, Solidity, Extent, 64-sample Centroidal Radial Profile, 8-directional Freeman Chain Code Entropy, and 10-coefficient Fourier Descriptors.
   - **Moment Invariants:** Spatial, Central, and Scale-Invariant Hu Moments ($\phi_1 \ldots \phi_7$) log-transformed.

3. **Stage 3: Hough Transforms for Geometric Verification (Unit 3)**
   - **Probabilistic Hough Line Transform (`cv2.HoughLinesP`):** Identifies straight structural edges of boxes, folded sheets, and bottle bodies.
   - **Circular Hough Transform (`cv2.HoughCircles`):** Pre-smoothed gradient voting detects circular rims of aluminum cans and plastic bottle caps.

4. **Stage 4: Photometric & Motion Cues (Units 4 & 5)**
   - **Shape-from-Shading / Specular Ratio:** HSV Value/Saturation thresholding ($V > 180, S < 80$) measures metallic and glossy plastic reflection.
   - **Background Separation:** Dual-strategy static binarization and MOG2 background subtractor hook (`cv2.createBackgroundSubtractorMOG2`).

5. **Stage 5: Template Matching & Explainable Classification (Unit 5)**
   - **Symmetric Chamfer Distance:** Two-way distance transform matching against canonical silhouette templates (Bottle, Can, Box, Organic).
   - **Explainable Weighted Scoring Engine:** Transparent multi-feature scoring system that outputs normalized scores, a soft confidence distribution, and an audit trail explaining which mathematical features triggered the decision.

---

## 🚀 How to Run

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- macOS, Linux, or Windows.

### 2. Installation
Open your terminal in the project directory and install the required dependencies:

```bash
cd "/Users/aadithyaaravishankar/Desktop/EcoSort CV Project"
pip install -r requirements.txt
```

*(If running on modern macOS Homebrew Python, you can use `pip install -r requirements.txt --break-system-packages` or run inside a virtual environment).*

### 3. Launch the Web Application
Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The web dashboard will launch automatically in your default browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Application Interface

The dashboard includes 4 dedicated interactive tabs:
- **Tab 1: 🗑️ EcoSort Classifier:** Real-time classification banner with bounding boxes, bin badges, confidence bars, parameter sliders, and natural-language "Why this classification?" decision cards. Supports Built-in Synthetic Samples, Custom File Uploads, and Live Webcam Capture (`st.camera_input`).
- **Tab 2: 🔬 Pipeline Visualizer:** A 20-panel visual grid displaying all intermediate image processing outputs from Unit 1 to Unit 5.
- **Tab 3: 📊 Feature Inspector:** Interactive Pandas DataFrame showing all extracted mathematical features for every detected object, with a one-click **Download CSV** export.
- **Tab 4: Reference:** Complete technical review featuring mathematical formulations, system architecture flowcharts, and technical Q&A reference.

---

## 📁 Repository Structure

```text
EcoSort CV Project/
├── README.md               # Quick overview & run guide (this file)
├── README_DETAILED.md      # Comprehensive end-to-end technical documentation
├── PROJECT_REPORT.md       # University project report formatted for 21CSE454T
├── VIVA_GUIDE.md           # 25 Viva Voce technical Q&As with syllabus citations
├── app.py                  # Streamlit web application & interactive UI
├── cv_pipeline.py          # Modular CV pipeline implementing Units 1 to 5
├── classifier.py           # Explainable weighted scoring engine & bin router
├── sample_generator.py     # Procedural test image generator for 6 waste scenes
└── requirements.txt        # Pinned lightweight dependencies
```

---

## 📖 Detailed Documentation

For the complete end-to-end technical specification, algorithmic formulas, tuning presets, and viva defense guidelines:
- Read the [Comprehensive Technical Guide](file:///Users/aadithyaaravishankar/Desktop/EcoSort%20CV%20Project/README_DETAILED.md).
- Read the [Formal Academic Project Report](file:///Users/aadithyaaravishankar/Desktop/EcoSort%20CV%20Project/PROJECT_REPORT.md).
- Review the [Viva Voce Technical Guide](file:///Users/aadithyaaravishankar/Desktop/EcoSort%20CV%20Project/VIVA_GUIDE.md).

