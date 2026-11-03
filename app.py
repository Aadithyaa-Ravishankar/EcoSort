"""
EcoSort: Visual Waste Classification Assistant
===============================================
Streamlit Application — app.py

Course: 21CSE454T — Computer Vision
A complete, interactive, academically explainable Classical Computer Vision pipeline
for waste classification and recycling bin routing.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import traceback

# ─── Local module imports ────────────────────────────────────────────────────
from sample_generator import get_sample_image
from cv_pipeline import run_full_pipeline
from classifier import classify_all_objects

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSort: Visual Waste Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header - Clean Light Theme */
    .eco-header {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #059669;
        color: #0f172a;
        padding: 1.25rem 1.6rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .eco-header h1 {
        margin: 0;
        font-size: 1.95rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.4px;
    }

    /* Bin badge */
    .bin-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 6px;
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        margin: 0.3rem 0;
    }

    /* Feature card */
    .feature-card {
        background: #f8fafc;
        border-left: 3px solid #0f766e;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.35rem 0;
        font-size: 0.88rem;
        color: #1e293b;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Stage header */
    .stage-header {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 0.45rem 0.85rem;
        border-radius: 6px;
        font-weight: 600;
        color: #1e293b;
        margin: 0.8rem 0 0.4rem;
        font-size: 0.88rem;
    }

    /* Metric pill */
    .metric-pill {
        display: inline-block;
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        margin: 0.15rem;
        font-family: inherit;
    }

    /* Info box */
    .info-box {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        color: #334155;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eco-header">
    <h1>EcoSort: Visual Waste Classification Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Global Parameter Controls & Presets
# ─────────────────────────────────────────────────────────────────────────────
PRESET_CONFIGS = {
    "Benchmark (Synthetic Samples)": {
        "thresh_method": "Otsu",
        "thresh_polarity": "Auto-Detect",
        "gaussian_kernel": 5,
        "adaptive_block": 11,
        "adaptive_C": 2,
        "canny_low": 50,
        "canny_high": 150,
        "morph_kernel": 5,
        "min_area": 500,
        "hough_threshold": 25,
        "min_line_length": 30,
        "circle_param2": 32,
    },
    "Real Camera / Light Background (Desk/Paper)": {
        "thresh_method": "Otsu",
        "thresh_polarity": "Auto-Detect",
        "gaussian_kernel": 7,
        "adaptive_block": 15,
        "adaptive_C": 3,
        "canny_low": 35,
        "canny_high": 110,
        "morph_kernel": 7,
        "min_area": 1000,
        "hough_threshold": 25,
        "min_line_length": 35,
        "circle_param2": 26,
    },
    "Real Camera / Dark Background (Black Mat/Floor)": {
        "thresh_method": "Otsu",
        "thresh_polarity": "Light Object on Dark Background",
        "gaussian_kernel": 7,
        "adaptive_block": 15,
        "adaptive_C": 3,
        "canny_low": 35,
        "canny_high": 110,
        "morph_kernel": 7,
        "min_area": 1000,
        "hough_threshold": 25,
        "min_line_length": 35,
        "circle_param2": 26,
    },
    "Metal Cans & Cylinders (Circle Focus)": {
        "thresh_method": "Otsu",
        "thresh_polarity": "Auto-Detect",
        "gaussian_kernel": 7,
        "adaptive_block": 11,
        "adaptive_C": 2,
        "canny_low": 40,
        "canny_high": 120,
        "morph_kernel": 5,
        "min_area": 800,
        "hough_threshold": 25,
        "min_line_length": 30,
        "circle_param2": 22,
    },
    "Cardboard & Flat Sheets (Line Focus)": {
        "thresh_method": "Adaptive",
        "thresh_polarity": "Auto-Detect",
        "gaussian_kernel": 5,
        "adaptive_block": 19,
        "adaptive_C": 3,
        "canny_low": 35,
        "canny_high": 120,
        "morph_kernel": 5,
        "min_area": 800,
        "hough_threshold": 25,
        "min_line_length": 35,
        "circle_param2": 35,
    },
    "Custom / Manual Tuning": None,
}

with st.sidebar:
    st.markdown("## Pipeline Parameters")
    st.caption("*Select an optimal preset or tune individual sliders manually.*")

    chosen_preset = st.selectbox(
        "Accuracy Preset",
        list(PRESET_CONFIGS.keys()),
        index=0,
        help="Loads scientifically calibrated parameter values for specific lighting and scene types."
    )

    defaults = PRESET_CONFIGS.get(chosen_preset)
    if defaults is None:
        defaults = PRESET_CONFIGS["Benchmark (Synthetic Samples)"]

    st.markdown("---")
    st.markdown("**Segmentation & Polarity**")

    thresh_method_opts = ["Otsu", "Adaptive", "Combined"]
    default_tm_idx = thresh_method_opts.index(defaults.get("thresh_method", "Otsu"))
    thresh_method = st.selectbox(
        "Segmentation Strategy",
        thresh_method_opts,
        index=default_tm_idx,
        help="Otsu: global bimodal optimal. Adaptive: robust to shadows/gradients. Combined: union."
    )

    polarity_opts = ["Auto-Detect", "Dark Object on Light Background", "Light Object on Dark Background"]
    default_pol_idx = polarity_opts.index(defaults.get("thresh_polarity", "Auto-Detect"))
    thresh_polarity = st.selectbox(
        "Background Polarity",
        polarity_opts,
        index=default_pol_idx,
        help="Auto-Detect analyzes border pixels to prevent background from inverting into foreground."
    )

    gaussian_kernel = st.slider("Gaussian Kernel Size (odd)", 3, 21, int(defaults.get("gaussian_kernel", 5)), step=2,
                                help="cv2.GaussianBlur kernel. Larger = more smoothing for webcam noise.")

    adaptive_block = st.slider("Adaptive Threshold Block Size", 7, 51, int(defaults.get("adaptive_block", 11)), step=2,
                               help="Neighbourhood block size for adaptive thresholding.")
    adaptive_C     = st.slider("Adaptive Threshold C", 0, 20, int(defaults.get("adaptive_C", 2)),
                               help="Constant subtracted from mean in adaptive threshold.")

    st.markdown("**Edge Detection & Morphology**")
    canny_low  = st.slider("Canny Low Threshold",  10, 150, int(defaults.get("canny_low", 50)),
                           help="Lower hysteresis threshold for Canny edge detection.")
    canny_high = st.slider("Canny High Threshold", 50, 300, int(defaults.get("canny_high", 150)),
                           help="Upper hysteresis threshold for Canny edge detection.")

    morph_kernel = st.slider("Morphology Kernel Size", 3, 21, int(defaults.get("morph_kernel", 5)), step=2,
                             help="Structuring element size for erosion/dilation/opening/closing.")

    st.markdown("**Object Area Filtering**")
    min_area = st.slider("Min Object Area (px)", 100, 5000, int(defaults.get("min_area", 500)), step=100,
                         help="Discard connected components smaller than this (noise filter).")

    st.markdown("**Hough Geometric Transforms**")
    hough_threshold = st.slider("Hough Line Threshold", 10, 200, int(defaults.get("hough_threshold", 25)),
                                help="Minimum votes for Hough line detection (25-35 recommended).")
    min_line_length = st.slider("Min Hough Line Length (px)", 10, 200, int(defaults.get("min_line_length", 30)),
                                help="Minimum line length for probabilistic Hough transform.")
    circle_param2   = st.slider("Hough Circle Sensitivity (param2)", 15, 50, int(defaults.get("circle_param2", 30)),
                                help="Lower value = more sensitive circle detector (detects faint rims/caps); higher = strict.")

    with st.expander("Parameter Tuning Guide"):
        st.markdown("""
        - **Can misclassified as Box?**  
          *Lower Hough Circle Sensitivity (to 22–26) and ensure Canny High is ≤ 120.*
        - **Bottle misclassified as Organic?**  
          *Increase Morphology Kernel (to 7–9) to seal transparent gaps and bridge contour holes.*
        - **Background detected as object?**  
          *Check Background Polarity. Use Auto-Detect or explicitly set Light Object on Dark Background.*
        - **Real Camera Shadow Specks?**  
          *Increase Min Object Area (to 1000–1500 px) and Gaussian Kernel (to 7).*
        """)

    st.markdown("---")
    st.caption("*EcoSort Visual Classifier — Interactive Controls*")

pol_code = "auto"
if "Dark Background" in thresh_polarity:
    pol_code = "light_on_dark"
elif "Light Background" in thresh_polarity:
    pol_code = "dark_on_light"

params = {
    "gaussian_kernel": gaussian_kernel,
    "canny_low":        canny_low,
    "canny_high":       canny_high,
    "morph_kernel":     morph_kernel,
    "min_area":         min_area,
    "hough_threshold":  hough_threshold,
    "adaptive_block":   adaptive_block,
    "adaptive_C":       adaptive_C,
    "min_line_length":  min_line_length,
    "circle_param2":    circle_param2,
    "thresh_method":    thresh_method.lower(),
    "thresh_polarity":  pol_code,
}

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "EcoSort Classifier",
    "Pipeline Visualizer",
    "Feature Inspector",
    "Viva & Academic Guide",
])

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

BIN_COLORS = {
    "Metal/Tin Can":      ("#1565C0", "Blue Bin (Recyclable Metal)"),
    "Plastic Bottle":     ("#D97706", "Yellow Bin (Recyclable Plastic)"),
    "Paper/Cardboard":    ("#2E7D32", "Green/Blue Bin (Recyclable Fiber)"),
    "Organic Waste":      ("#15803D", "Green Bin (Compostable Waste)"),
}

def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image for Streamlit display."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def gray_to_pil(gray_img: np.ndarray) -> Image.Image:
    """Convert single-channel grayscale to PIL Image."""
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Image.fromarray(gray_img)

def overlay_text(img_bgr, text, pos=(10, 30), color=(0, 255, 0), scale=0.6, thickness=2):
    """Draw text with dark shadow for visibility on any background."""
    cv2.putText(img_bgr, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness+1)
    cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)
    return img_bgr

def draw_annotated_result(bgr_image: np.ndarray,
                          pipeline_results: dict,
                          classifications: list) -> np.ndarray:
    """
    Draw bounding boxes, centroids, object IDs, category labels,
    and bin badges on the original image.
    """
    annotated = bgr_image.copy()
    contours_data = pipeline_results.get("contours", [])

    cat_colors = {
        "Metal/Tin Can":   (212,  85,   0),   # Blue (BGR)
        "Plastic Bottle":  ( 35, 168, 249),   # Yellow
        "Paper/Cardboard": ( 50, 126,  50),   # Green
        "Organic Waste":   ( 57, 142,  57),   # Green
    }

    for idx, (cdata, cls) in enumerate(zip(contours_data, classifications)):
        cat   = cls.get("category", "Unknown")
        conf  = cls.get("confidence", 0.0)
        color = cat_colors.get(cat, (200, 200, 200))

        x, y, w, h = cdata.get("bounding_rect", (0, 0, 50, 50))
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

        cx, cy = cdata.get("centroid", (x + w//2, y + h//2))
        cv2.circle(annotated, (int(cx), int(cy)), 5, color, -1)
        cv2.circle(annotated, (int(cx), int(cy)), 7, (255, 255, 255), 2)

        label = f"#{idx+1} {cat[:8]}.. {conf*100:.0f}%"
        overlay_text(annotated, label, (max(x, 2), max(y - 8, 20)), color)

    return annotated

def confidence_bar(score: float, color: str) -> str:
    """Return HTML for a colored confidence progress bar."""
    pct = int(score * 100)
    return (
        f'<div style="background:#E0E0E0;border-radius:6px;height:18px;width:100%;">'
        f'<div style="background:{color};width:{pct}%;height:18px;border-radius:6px;'
        f'text-align:center;color:white;font-size:0.8rem;line-height:18px;font-weight:bold;">'
        f'{pct}%</div></div>'
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STATE: IMAGE ACQUISITION + PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
# We compute pipeline results once and share them across tabs via st.session_state.

def run_pipeline_cached(bgr_image: np.ndarray, params: dict):
    """Run the full CV pipeline and classification, store results in session state."""
    with st.spinner("Executing Classical Computer Vision Pipeline..."):
        try:
            pipeline_results = run_full_pipeline(bgr_image, params)
            classifications  = classify_all_objects(pipeline_results)
            st.session_state["pipeline_results"] = pipeline_results
            st.session_state["classifications"]  = classifications
            st.session_state["bgr_image"]        = bgr_image
            return pipeline_results, classifications
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.code(traceback.format_exc())
            return None, []

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE / INTERACTIVE ECOSORT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Select Input Image")

    source_options = ["Built-in Sample Generator", "Upload Image", "Live Webcam Capture"]
    input_source   = st.radio("Input Source:", source_options, horizontal=True)

    bgr_image = None

    # ── Built-in Sample Generator ─────────────────────────────────────────────
    if input_source == "Built-in Sample Generator":
        sample_names = [
            "Plastic Bottle", "Aluminum Can", "Cardboard Box",
            "Crumpled Paper", "Organic Waste", "Multi-Object Tray",
        ]
        col_sel, col_prev = st.columns([2, 3])
        with col_sel:
            chosen = st.selectbox("Choose a synthetic test sample:", sample_names)
            if st.button("Generate Synthetic Sample", type="primary", use_container_width=True):
                bgr_image = get_sample_image(chosen)
                st.session_state["generated_image"] = bgr_image
                st.session_state["selected_sample"] = chosen
            elif "generated_image" in st.session_state:
                bgr_image = st.session_state["generated_image"]
        with col_prev:
            if bgr_image is not None:
                st.image(bgr_to_pil(bgr_image), caption=f"Generated: *{chosen}*",
                         use_container_width=True)
            else:
                st.info("Select a category and click *Generate Synthetic Sample* to proceed.")

    # ── Upload Image ──────────────────────────────────────────────────────────
    elif input_source == "Upload Image":
        uploaded = st.file_uploader("Upload a waste image (.jpg, .png):",
                                    type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_img   = Image.open(uploaded).convert("RGB")
            rgb_array = np.array(pil_img)
            bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            st.image(pil_img, caption="Uploaded Input Image", use_container_width=True)

    # ── Webcam Capture ────────────────────────────────────────────────────────
    else:
        st.info("Capture a frame from the connected camera device below.")
        camera_frame = st.camera_input("Capture waste item:")
        if camera_frame is not None:
            pil_img   = Image.open(camera_frame).convert("RGB")
            rgb_array = np.array(pil_img)
            bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # ── Run Pipeline Button ───────────────────────────────────────────────────
    st.markdown("---")
    if bgr_image is not None:
        if st.button("Execute EcoSort Classification Pipeline", type="primary",
                     use_container_width=True):
            run_pipeline_cached(bgr_image, params)

    # ── Results Display ───────────────────────────────────────────────────────
    if ("pipeline_results" in st.session_state
            and st.session_state["pipeline_results"] is not None):

        pr   = st.session_state["pipeline_results"]
        clss = st.session_state["classifications"]
        orig = st.session_state["bgr_image"]

        st.markdown("---")
        st.markdown("## Classification Results")

        annotated = draw_annotated_result(orig, pr, clss)

        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            st.image(bgr_to_pil(annotated),
                     caption="Annotated Result — Bounding Boxes, Centroids & Categories",
                     use_container_width=True)

        with res_col2:
            if not clss:
                st.warning("No objects detected above the minimum area threshold. "
                           "Try lowering the *Min Object Area* slider in the sidebar.")
            else:
                for idx, cls_result in enumerate(clss):
                    cat       = cls_result.get("category", "Unknown")
                    bin_label = cls_result.get("bin", "Unknown Bin")
                    bin_color = cls_result.get("bin_color", "#607D8B")
                    conf      = cls_result.get("confidence", 0.0)
                    reasons   = cls_result.get("reasons", [])
                    syl_feats = cls_result.get("syllabus_features", [])

                    with st.expander(f"Object #{idx+1} — {cat}", expanded=(idx == 0)):
                        # Bin badge
                        st.markdown(
                            f'<span class="bin-badge" style="background:{bin_color};">'
                            f'{bin_label}</span>',
                            unsafe_allow_html=True
                        )
                        # Confidence bar
                        st.markdown(f"**Confidence Score:**")
                        st.markdown(confidence_bar(conf, bin_color), unsafe_allow_html=True)
                        st.markdown(f"**Calculated Confidence:** `{conf*100:.1f}%`")

                        # All category scores
                        all_scores = cls_result.get("all_scores", {})
                        if all_scores:
                            st.markdown("**Category Scores:**")
                            score_df = pd.DataFrame(
                                list(all_scores.items()), columns=["Category", "Score"]
                            ).sort_values("Score", ascending=False)
                            st.dataframe(score_df, use_container_width=True, hide_index=True)

                        # Why this classification
                        st.markdown("**Classification Rationale:**")
                        for reason in reasons:
                            st.markdown(
                                f'<div class="feature-card"><em>{reason}</em></div>',
                                unsafe_allow_html=True
                            )

                        # Active feature descriptor mapping
                        if syl_feats:
                            st.markdown("**Active Feature Descriptors:**")
                            pills = " ".join(
                                f'<span class="metric-pill"><em>{f}</em></span>'
                                for f in syl_feats
                            )
                            st.markdown(pills, unsafe_allow_html=True)

        # ── Bin Summary ───────────────────────────────────────────────────────
        if clss:
            st.markdown("---")
            st.markdown("### Recycling Bin Allocation Summary")
            bin_cols = st.columns(len(clss))
            for col, cls_result in zip(bin_cols, clss):
                cat       = cls_result.get("category", "Unknown")
                bin_label = cls_result.get("bin", "Unknown")
                bin_color = cls_result.get("bin_color", "#607D8B")
                conf      = cls_result.get("confidence", 0.0)
                with col:
                    st.markdown(
                        f'<div style="background:{bin_color};color:white;padding:1.2rem;'
                        f'border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);">'
                        f'<div style="font-weight:700;font-size:1.05rem;letter-spacing:0.3px;">{cat}</div>'
                        f'<div style="font-size:0.85rem;margin-top:0.3rem;opacity:0.9;"><em>{bin_label}</em></div>'
                        f'<div style="font-size:1.15rem;font-weight:700;margin-top:0.5rem;border-top:1px solid rgba(255,255,255,0.25);padding-top:0.4rem;">'
                        f'{conf*100:.0f}% confidence</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STEP-BY-STEP PIPELINE VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Step-by-Step Pipeline Visualizer")
    st.caption("Sequential visualization of intermediate classical computer vision stages.")

    if "pipeline_results" not in st.session_state or st.session_state["pipeline_results"] is None:
        st.info("Navigate to *EcoSort Classifier* (Tab 1), select an input image, and click *Execute EcoSort Classification Pipeline* first.")
    else:
        pr   = st.session_state["pipeline_results"]
        orig = st.session_state["bgr_image"]

        def show_panel(col, pil_img, title, caption="", height=220):
            with col:
                st.markdown(f'<div class="stage-header">{title}</div>', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
                if caption:
                    st.caption(caption)

        # ── ROW 1: Original + Preprocessing ──────────────────────────────────
        st.markdown("#### Stage 1: Image Preprocessing & Filtering")
        r1c1, r1c2, r1c3 = st.columns(3)
        show_panel(r1c1, bgr_to_pil(orig),
                   "1. Original Image",
                   "Input BGR image before any processing.")

        prep = pr.get("preprocessing", {})
        if "clahe" in prep:
            show_panel(r1c2, gray_to_pil(prep["clahe"]),
                       "2. CLAHE Illumination Normalization",
                       "Contrast Limited Adaptive Histogram Equalization for illumination normalization.")
        if "gaussian_blur" in prep:
            show_panel(r1c3, gray_to_pil(prep["gaussian_blur"]),
                       "3. Gaussian & Median Filtering",
                       f"Gaussian kernel={gaussian_kernel}×{gaussian_kernel}, σ auto. Noise reduction.")

        # ── ROW 2: Thresholding + Morphology ─────────────────────────────────
        r2c1, r2c2, r2c3 = st.columns(3)
        thresh = pr.get("thresholding", {})
        if "otsu" in thresh:
            otsu_val = thresh.get("otsu_thresh_val", 0)
            show_panel(r2c1, gray_to_pil(thresh["otsu"]),
                       "4. Otsu Binarization",
                       f"Otsu threshold = {otsu_val}. Maximizes inter-class variance.")
        if "adaptive" in thresh:
            show_panel(r2c2, gray_to_pil(thresh["adaptive"]),
                       "5. Adaptive Thresholding",
                       f"Block={adaptive_block}, C={adaptive_C}. Local neighbourhood mean thresholding.")

        morph = pr.get("morphology", {})
        if "closed" in morph:
            show_panel(r2c3, gray_to_pil(morph["closed"]),
                       "6. Morphological Opening & Closing",
                       "Opening removes noise specks; Closing fills internal holes. Structuring element: Ellipse.")

        # ── ROW 3: Edges + Corners ────────────────────────────────────────────
        r3c1, r3c2, r3c3 = st.columns(3)
        ec = pr.get("edges_corners", {})
        if "canny" in ec:
            ed  = ec.get("edge_density", 0)
            cnt = ec.get("corner_count", 0)
            show_panel(r3c1, gray_to_pil(ec["canny"]),
                       "7. Canny Edge Detection",
                       f"Low={canny_low}, High={canny_high}. Edge density={ed:.3f}.")

        if "sobel_mag" in ec:
            show_panel(r3c2, gray_to_pil(ec["sobel_mag"]),
                       "8. Sobel Gradient Magnitude",
                       "√(Gx²+Gy²). Measures local intensity gradient.")

        # Draw Shi-Tomasi corners overlay
        if "shitomasi_corners" in ec and ec["shitomasi_corners"] is not None:
            corner_vis = cv2.cvtColor(
                prep.get("gaussian_blur", cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)),
                cv2.COLOR_GRAY2BGR
            )
            for pt in ec["shitomasi_corners"]:
                x_c, y_c = int(pt[0][0]), int(pt[0][1])
                cv2.circle(corner_vis, (x_c, y_c), 4, (0, 255, 0), -1)
            show_panel(r3c3, bgr_to_pil(corner_vis),
                       "9. Shi-Tomasi Corner Detection",
                       f"{ec.get('corner_count', 0)} corners detected (green markers). Irregular surfaces yield higher counts.")

        # ── ROW 4: Connected Components + Distance Transform ──────────────────
        st.markdown("#### Stage 2: Shape & Regional Boundary Analysis")
        r4c1, r4c2, r4c3 = st.columns(3)
        comp = pr.get("components", {})
        if "colored_labels" in comp:
            show_panel(r4c1, bgr_to_pil(comp["colored_labels"]),
                       "10. Connected Component Labeling",
                       f"{comp.get('num_components', 0)} components found. Distinct colors denote individual regions.")

        dist_sk = pr.get("distance_skeleton", {})
        if "distance_transform" in dist_sk:
            dt_vis = cv2.normalize(
                dist_sk["distance_transform"], None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            dt_color = cv2.applyColorMap(dt_vis, cv2.COLORMAP_JET)
            show_panel(r4c2, bgr_to_pil(dt_color),
                       "11. Euclidean Distance Transform",
                       "Distance of each foreground pixel to nearest background. Utilized in Chamfer matching.")

        if "skeleton" in dist_sk:
            skel_arr = (dist_sk["skeleton"].astype(np.uint8)) * 255
            show_panel(r4c3, gray_to_pil(skel_arr),
                       "12. Morphological Skeleton & Thinning",
                       "Medial axis of the object. Reveals elongated topological structure.")

        # ── ROW 5: Contour + Hough ────────────────────────────────────────────
        r5c1, r5c2, r5c3 = st.columns(3)
        contours_data = pr.get("contours", [])
        if contours_data:
            hull_vis = orig.copy()
            for cdata in contours_data:
                cnt_pts = cdata.get("contour")
                hull    = cdata.get("convex_hull")
                cx, cy  = cdata.get("centroid", (0, 0))
                if cnt_pts is not None:
                    cv2.drawContours(hull_vis, [cnt_pts], -1, (0, 255, 0), 2)
                if hull is not None:
                    cv2.drawContours(hull_vis, [hull], -1, (255, 0, 0), 2)
                cv2.circle(hull_vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(hull_vis, f"c=({int(cx)},{int(cy)})",
                            (int(cx)+8, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
            show_panel(r5c1, bgr_to_pil(hull_vis),
                       "13. Boundary Tracking & Convex Hull",
                       "Green = contour boundary, Blue = convex hull, Red marker = centroid.")

        hough = pr.get("hough", {})
        if "line_overlay" in hough:
            show_panel(r5c2, bgr_to_pil(hough["line_overlay"]),
                       "14. Hough Line Transform (Probabilistic)",
                       f"{hough.get('line_count', 0)} lines. Straight structural edges indicate paper or cardboard.")
        if "circle_overlay" in hough:
            show_panel(r5c3, bgr_to_pil(hough["circle_overlay"]),
                       "15. Circular Hough Transform",
                       f"{hough.get('circle_count', 0)} circles. Circular geometries indicate cans or caps.")

        # ── ROW 6: Texture + Specular + Chamfer ───────────────────────────────
        st.markdown("#### Stage 3: Surface Shading, Texture & Template Matching")
        r6c1, r6c2, r6c3 = st.columns(3)

        specular = pr.get("specular", {})
        if "specular_mask" in specular:
            spec_vis = orig.copy()
            spec_mask = specular["specular_mask"]
            spec_vis[spec_mask > 0] = [0, 255, 255]  # cyan overlay for specular pixels
            show_panel(r6c1, bgr_to_pil(spec_vis),
                       "16. Specular Highlight Detection",
                       f"Cyan = specular pixels (V>220, S<50). Specular ratio = {specular.get('specular_ratio',0):.3f}. "
                       f"Glossy = {'True' if specular.get('is_glossy') else 'False'}.")

        # Texture entropy visualization
        texture = pr.get("texture", {})
        gray_for_texture = prep.get("gaussian_blur", cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY))
        entropy_img = np.zeros_like(gray_for_texture, dtype=np.float32)
        try:
            from skimage.filters.rank import entropy as rank_entropy
            from skimage.morphology import disk
            entropy_img = rank_entropy(gray_for_texture, disk(5)).astype(np.float32)
            ent_norm    = cv2.normalize(entropy_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ent_color   = cv2.applyColorMap(ent_norm, cv2.COLORMAP_HOT)
            show_panel(r6c2, bgr_to_pil(ent_color),
                       "17. Local Texture Entropy",
                       f"High entropy indicates textural complexity. GLCM contrast = {texture.get('glcm_contrast',0):.3f}.")
        except Exception:
            show_panel(r6c2, gray_to_pil(gray_for_texture),
                       "17. Texture Analysis (Grayscale)",
                       "Local entropy map unavailable — displaying preprocessed grayscale.")

        # Chamfer matching bar chart
        chamfer = pr.get("chamfer", {})
        scores  = chamfer.get("scores", {})
        best    = chamfer.get("best_match", "—")
        if scores:
            fig_c, ax_c = plt.subplots(figsize=(4, 3))
            names  = list(scores.keys())
            values = [scores[n] for n in names]
            colors_bar = ["#1565C0" if n == best else "#90CAF9" for n in names]
            ax_c.barh(names, values, color=colors_bar)
            ax_c.set_xlabel("Chamfer Distance (lower indicates closer match)")
            ax_c.set_title(f"Chamfer Template Matching\nBest Match: {best}", fontsize=10)
            ax_c.invert_xaxis()
            fig_c.tight_layout()
            with r6c3:
                st.markdown('<div class="stage-header">18. Chamfer Distance Scores</div>',
                            unsafe_allow_html=True)
                st.pyplot(fig_c, use_container_width=True)
                st.caption(f"Best silhouette match: *{best}* (minimum Chamfer distance).")
            plt.close(fig_c)

        # ── Centroidal Profile + Chain Code ───────────────────────────────────
        if contours_data:
            st.markdown("#### Stage 4: Boundary Descriptors (Centroidal Profile & Chain Code)")
            rc1, rc2 = st.columns(2)

            first = contours_data[0]
            profile = first.get("centroidal_profile", np.array([]))
            if len(profile) > 0:
                fig_p, ax_p = plt.subplots(figsize=(5, 2.5))
                ax_p.plot(profile, color="#2E7D32", linewidth=1.5)
                ax_p.fill_between(range(len(profile)), profile, alpha=0.2, color="#81C784")
                ax_p.set_xlabel("Sample Angle Index (0–63)")
                ax_p.set_ylabel("Radial Distance from Centroid (px)")
                ax_p.set_title("Centroidal Radial Profile (1D Signature)", fontsize=10)
                ax_p.grid(alpha=0.3)
                fig_p.tight_layout()
                with rc1:
                    st.markdown('<div class="stage-header">19. Centroidal Distance Profile</div>',
                                unsafe_allow_html=True)
                    st.pyplot(fig_p, use_container_width=True)
                    st.caption("Radial distance signature sampled at 64 uniform angular intervals. "
                               "Uniform profiles indicate circularity; variances indicate irregular boundaries.")
                plt.close(fig_p)

            chain = first.get("chain_code", [])
            if chain:
                hist = np.zeros(8)
                for c in chain:
                    if 0 <= c < 8:
                        hist[c] += 1
                fig_ch, ax_ch = plt.subplots(figsize=(5, 2.5))
                ax_ch.bar(range(8), hist, color="#1565C0", alpha=0.8)
                ax_ch.set_xticks(range(8))
                ax_ch.set_xticklabels(["E","NE","N","NW","W","SW","S","SE"])
                ax_ch.set_ylabel("Frequency")
                ax_ch.set_title("Freeman 8-Direction Chain Code Histogram", fontsize=10)
                fig_ch.tight_layout()
                with rc2:
                    st.markdown('<div class="stage-header">20. Freeman Chain Code Histogram</div>',
                                unsafe_allow_html=True)
                    st.pyplot(fig_ch, use_container_width=True)
                    entropy_val = first.get("chain_code_entropy", 0)
                    st.caption(f"Distribution of 8-directional boundary transitions. "
                               f"Shannon entropy = *{entropy_val:.3f}* (elevated in irregular boundaries).")
                plt.close(fig_ch)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUANTITATIVE FEATURE INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Quantitative Feature Inspector")
    st.caption("Extracted numerical feature matrix for each detected object — exportable as CSV.")

    if "pipeline_results" not in st.session_state or st.session_state["pipeline_results"] is None:
        st.info("Execute the classification pipeline in *EcoSort Classifier* (Tab 1) first to inspect numerical features.")
    else:
        pr   = st.session_state["pipeline_results"]
        clss = st.session_state["classifications"]
        contours_data = pr.get("contours", [])
        texture = pr.get("texture", {})
        ec      = pr.get("edges_corners", {})
        hough   = pr.get("hough", {})
        specular = pr.get("specular", {})
        chamfer  = pr.get("chamfer", {})

        rows = []
        for idx, (cdata, cls_res) in enumerate(zip(contours_data, clss)):
            hu = cdata.get("hu_moments", np.zeros(7))
            fd = cdata.get("fourier_descriptors", np.zeros(10))

            row = {
                "Object #":         idx + 1,
                "Category":         cls_res.get("category", "—"),
                "Confidence (%)":   f"{cls_res.get('confidence', 0)*100:.1f}",
                "Bin":              cls_res.get("bin", "—"),
                # Shape descriptors
                "Area (px²)":       int(cdata.get("area", 0)),
                "Perimeter (px)":   f"{cdata.get('perimeter', 0):.1f}",
                "Circularity":      f"{cdata.get('circularity', 0):.4f}",
                "Solidity":         f"{cdata.get('solidity', 0):.4f}",
                "Aspect Ratio":     f"{cdata.get('aspect_ratio', 0):.4f}",
                "Extent":           f"{cdata.get('extent', 0):.4f}",
                # Hu Moments
                "Hu₁":  f"{hu[0]:.5f}" if len(hu) > 0 else "—",
                "Hu₂":  f"{hu[1]:.5f}" if len(hu) > 1 else "—",
                "Hu₃":  f"{hu[2]:.5f}" if len(hu) > 2 else "—",
                "Hu₄":  f"{hu[3]:.5f}" if len(hu) > 3 else "—",
                "Hu₅":  f"{hu[4]:.5f}" if len(hu) > 4 else "—",
                "Hu₆":  f"{hu[5]:.5f}" if len(hu) > 5 else "—",
                "Hu₇":  f"{hu[6]:.5f}" if len(hu) > 6 else "—",
                # Fourier Descriptors
                "FD₁":  f"{fd[0]:.3f}" if len(fd) > 0 else "—",
                "FD₂":  f"{fd[1]:.3f}" if len(fd) > 1 else "—",
                "FD₃":  f"{fd[2]:.3f}" if len(fd) > 2 else "—",
                # Chain code
                "Chain Code Entropy":   f"{cdata.get('chain_code_entropy', 0):.4f}",
                # Texture (GLCM)
                "GLCM Contrast":        f"{texture.get('glcm_contrast', 0):.4f}",
                "GLCM Dissimilarity":   f"{texture.get('glcm_dissimilarity', 0):.4f}",
                "GLCM Homogeneity":     f"{texture.get('glcm_homogeneity', 0):.4f}",
                "GLCM Energy":          f"{texture.get('glcm_energy', 0):.4f}",
                "GLCM Correlation":     f"{texture.get('glcm_correlation', 0):.4f}",
                "Local Entropy":        f"{texture.get('local_entropy', 0):.4f}",
                # Edge & Corner
                "Edge Density":         f"{ec.get('edge_density', 0):.4f}",
                "Corner Count":         ec.get('corner_count', 0),
                # Hough
                "Hough Line Count":     hough.get('line_count', 0),
                "Hough Circle Count":   hough.get('circle_count', 0),
                # Specular / Shading
                "Specular Ratio":       f"{specular.get('specular_ratio', 0):.4f}",
                "Mean HSV Value":       f"{specular.get('mean_value', 0):.2f}",
                "Is Glossy":            specular.get('is_glossy', False),
                # Chamfer
                "Chamfer Best Match":   chamfer.get('best_match', '—'),
                "Chamfer Distance":     f"{min(chamfer.get('scores', {None: 0}).values(), default=0):.2f}",
            }
            rows.append(row)

        if rows:
            feat_df = pd.DataFrame(rows)
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Download button
            csv_buf = io.StringIO()
            feat_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download Feature Table (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="ecosort_features.csv",
                mime="text/csv",
            )

            # Feature correlation heatmap
            st.markdown("---")
            st.markdown("#### Numeric Feature Matrix Overview")
            numeric_cols = ["Area (px²)", "Circularity", "Solidity", "Aspect Ratio",
                            "Extent", "Chain Code Entropy", "Edge Density", "Specular Ratio"]
            available_numeric = [c for c in numeric_cols if c in feat_df.columns]
            if available_numeric and len(rows) > 1:
                numeric_df = feat_df[available_numeric].apply(pd.to_numeric, errors="coerce")
                fig_heat, ax_h = plt.subplots(figsize=(8, 3))
                ax_h.imshow(numeric_df.T.values, aspect="auto", cmap="RdYlGn")
                ax_h.set_yticks(range(len(available_numeric)))
                ax_h.set_yticklabels(available_numeric, fontsize=9)
                ax_h.set_xticks(range(len(rows)))
                ax_h.set_xticklabels([f"Obj {i+1}" for i in range(len(rows))], fontsize=9)
                ax_h.set_title("Normalized Feature Heatmap Across Detected Objects", fontsize=10)
                fig_heat.tight_layout()
                st.pyplot(fig_heat, use_container_width=True)
                plt.close(fig_heat)

        else:
            st.warning("No objects found to display features for. "
                       "Adjust the *Min Object Area* threshold in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VIVA VOCE & PROJECT REPORT ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Theoretical Architecture & Reference Guide")
    st.caption("Comprehensive technical reference for classical computer vision algorithms and mathematical formulations.")

    viva_subtab1, viva_subtab2, viva_subtab3 = st.tabs([
        "System Architecture",
        "Mathematical Formulations",
        "Viva Voce Q&A Reference",
    ])

    # ── Architecture ──────────────────────────────────────────────────────────
    with viva_subtab1:
        st.markdown("## System Architecture — EcoSort Pipeline")
        st.markdown("""
```
+----------------------------------------------------------------------+
|                         INPUT ACQUISITION                            |
|    Synthetic Generator | Uploaded Image | Live Camera Capture        |
+----------------------------------+-----------------------------------+
                                   | BGR Image (H x W x 3, uint8)
                                   v
+----------------------------------------------------------------------+
|             STAGE 1 -- IMAGE PROCESSING FOUNDATIONS                  |
|  * Color space conversions: BGR to Grayscale, BGR to HSV             |
|  * Noise reduction: GaussianBlur (sigma=auto), MedianBlur            |
|  * Illumination normalization: CLAHE (Contrast Limited AHE)          |
|  * Thresholding: Otsu Binarization vs. Adaptive Local Thresholding   |
|  * Edge & Gradient: Canny Edge Detector + Sobel Gradient Magnitude   |
|  * Corner analysis: Harris Corner Response & Shi-Tomasi Detection    |
|  * Mathematical Morphology: Erosion, Dilation, Opening, Closing      |
|  * Texture: GLCM (Contrast, Homogeneity, Energy, Correlation)        |
+----------------------------------+-----------------------------------+
                                   | Binary Mask + Gradient Maps
                                   v
+----------------------------------------------------------------------+
|             STAGE 2 -- SHAPES AND REGIONS ANALYSIS                   |
|  * Connected Component Labeling (CCL) & Pixel Area Filtering         |
|  * Euclidean Distance Transform (EDT) & Inscribed Radius Cue         |
|  * Morphological Skeletonization / Medial Axis Thinning              |
|  * External Boundary Tracking (cv2.findContours)                     |
|  * Geometric Descriptors: Perimeter, Circularity, Solidity, Extent   |
|  * Convex Hull, Centroidal Radial Profile, Freeman 8-Direction Code  |
|  * Invariant Moments: Hu Moments (phi_1 to phi_7, log-transformed)  |
|  * Boundary Frequency Analysis: Low-Frequency Fourier Descriptors    |
+----------------------------------+-----------------------------------+
                                   | Object Features & Regions
                                   v
+----------------------------------------------------------------------+
|             STAGE 3 -- HOUGH TRANSFORM ANALYSIS                      |
|  * Probabilistic Hough Line Transform (HoughLinesP)                  |
|  * Circular Hough Transform (HoughCircles)                           |
|  * Structural cues: collinear edge sets and circular cross-sections  |
+----------------------------------+-----------------------------------+
                                   | Line & Circle Counts
                                   v
+----------------------------------------------------------------------+
|             STAGE 4 -- SHADING & SURFACE CUES                        |
|  * Specular Highlight Detection (HSV space: V > 220, S < 50)         |
|  * Specular Area Ratio: Metallic/Glossy vs. Matte Fibrous Surface    |
|  * Background Subtraction (MOG2) support for dynamic streams         |
+----------------------------------+-----------------------------------+
                                   | Surface Reflectance Metric
                                   v
+----------------------------------------------------------------------+
|             STAGE 5 -- CHAMFER TEMPLATE MATCHING                     |
|  * Procedural Canonical Silhouette Templates (Bottle, Can, Box, Blob)|
|  * Chamfer Distance computation via Distance Transform               |
|  * Global Geometric Fit: best_match = argmin(d_chamfer)              |
+----------------------------------+-----------------------------------+
                                   | Full Feature Matrix
                                   v
+----------------------------------------------------------------------+
|        EXPLAINABLE HEURISTIC CLASSIFIER (classifier.py)              |
|  * Multi-metric weighted scoring for all 4 waste categories          |
|  * Fully interpretable decision rationale per detected object        |
|  * Soft probability confidence calibration: w_i / sum(w)             |
+----------------------------------+-----------------------------------+
                                   |
                                   v
                 +-----------------------------------+
                 |        RECYCLING BIN ROUTING      |
                 |  [Blue Bin]   - Metal / Tin Cans  |
                 |  [Yellow Bin] - Plastic Bottles   |
                 |  [Green/Blue] - Paper / Cardboard |
                 |  [Green Bin]  - Organic Waste     |
                 +-----------------------------------+
```
""")

    # ── Mathematical Formulas ─────────────────────────────────────────────────
    with viva_subtab2:
        st.markdown("## Mathematical Formulations")

        st.markdown("### Image Processing Foundations")
        st.markdown("""
**Otsu's Threshold** — Maximizes inter-class variance $\\sigma_B^2$:

$$\\sigma_B^2(t) = \\omega_0(t)\\,\\omega_1(t)\\,[\\mu_0(t) - \\mu_1(t)]^2$$

where $\\omega_0, \\omega_1$ are class probabilities and $\\mu_0, \\mu_1$ are class means.

**Gaussian Smoothing:**
$$G(x,y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2+y^2}{2\\sigma^2}}$$

**Sobel Gradient Magnitude:**
$$|\\nabla f| = \\sqrt{G_x^2 + G_y^2}$$

**Harris Corner Response:**
$$R = \\det(M) - k\\,\\cdot\\,\\text{trace}(M)^2, \\quad M = \\begin{bmatrix}\\sum I_x^2 & \\sum I_x I_y \\\\ \\sum I_x I_y & \\sum I_y^2\\end{bmatrix}$$

**GLCM Features** (for co-occurrence matrix $P$):
$$\\text{Contrast} = \\sum_{i,j}(i-j)^2\\,P(i,j)$$
$$\\text{Homogeneity} = \\sum_{i,j}\\frac{P(i,j)}{1+|i-j|}$$
$$\\text{Energy} = \\sum_{i,j}P(i,j)^2$$
$$\\text{Correlation} = \\sum_{i,j}\\frac{(i-\\mu_i)(j-\\mu_j)\\,P(i,j)}{\\sigma_i\\,\\sigma_j}$$
""")

        st.markdown("### Shapes and Regions Analysis")
        st.markdown("""
**Circularity (Isoperimetric Quotient):**
$$C = \\frac{4\\pi A}{P^2}  \\quad (C=1 \\text{ for perfect circle, } 0 < C \\leq 1)$$

**Solidity:**
$$S = \\frac{A}{A_{\\text{convex hull}}}$$

**Aspect Ratio:**
$$AR = \\frac{W_{\\text{bounding box}}}{H_{\\text{bounding box}}}$$

**Extent:**
$$E = \\frac{A}{W \\times H}$$

**Hu Moments** (log-transformed, 7 invariants):
$$\\phi_k = -\\text{sign}(h_k)\\,\\log_{10}|h_k|, \\quad k=1,\\ldots,7$$

**Fourier Descriptors** (from complex boundary):
$$z_n = x_n + j\\,y_n, \\quad a_k = \\frac{1}{N}\\sum_{n=0}^{N-1} z_n\\,e^{-j2\\pi kn/N}$$

**Freeman Chain Code Entropy:**
$$H = -\\sum_{d=0}^{7} p_d \\log_2 p_d$$

**Centroidal Profile:** $r(\\theta_k) = \\|p(\\theta_k) - c\\|$ for $\\theta_k = k\\cdot\\frac{2\\pi}{64},\\ k=0,\\ldots,63$
""")

        st.markdown("### Hough Transforms")
        st.markdown("""
**Hough Line Transform** (Normal form):
$$\\rho = x\\cos\\theta + y\\sin\\theta$$

**Hough Circle Transform:**
$$(x - a)^2 + (y - b)^2 = r^2$$

Each edge pixel votes for $(a, b, r)$ in the accumulator space.
""")

        st.markdown("### Surface Shading & Chamfer Matching")
        st.markdown("""
**Specular Ratio:**
$$R_{\\text{spec}} = \\frac{|\\{(x,y) : V(x,y) > 220 \\wedge S(x,y) < 50\\}|}{|\\text{mask}|}$$

**Chamfer Distance:**
$$d_{\\text{chamfer}}(T, I) = \\frac{1}{|I|} \\sum_{p \\in I} DT_T(p)$$

where $DT_T$ is the distance transform of the inverted template $T$, and $I$ is the set of foreground pixels in the input mask.
""")

    # ── Viva Q&A ──────────────────────────────────────────────────────────────
    with viva_subtab3:
        st.markdown("## Technical Q&A — Classical Computer Vision")

        qa_pairs = [
            ("What is Otsu's thresholding? Why is it better than a fixed threshold?",
             "Otsu's method automatically finds the optimal threshold by maximizing the **inter-class variance** "
             "$\\sigma_B^2(t) = \\omega_0\\omega_1(\\mu_0 - \\mu_1)^2$ between foreground and background pixel classes. "
             "Unlike a fixed threshold, it adapts to the actual histogram of the image, making it robust to "
             "lighting changes. In code: `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)`.",
             "Image Thresholding", "cv_pipeline.py -> threshold_image()"),

            ("What is CLAHE and why is it needed?",
             "**CLAHE** (Contrast Limited Adaptive Histogram Equalization) applies histogram equalization "
             "to small local tiles (8x8 default) and clips the contrast to prevent over-amplification of noise "
             "(clip limit = 2.0). It normalizes uneven illumination across the image — critical when objects have "
             "shadows or non-uniform lighting. Code: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`.",
             "Image Filtering", "cv_pipeline.py -> preprocess_image()"),

            ("Explain the Canny edge detector step by step.",
             "Canny has 5 steps: **(1)** Gaussian smoothing to reduce noise. **(2)** Sobel gradient computation "
             "($G_x, G_y$, magnitude $M = \\sqrt{G_x^2+G_y^2}$, direction $\\theta = \\arctan(G_y/G_x)$). "
             "**(3)** Non-maximum suppression — retain only local maxima along gradient direction. "
             "**(4)** Double hysteresis thresholding with high/low thresholds. "
             "**(5)** Edge linking. In code: `cv2.Canny(blur, low, high)`.",
             "Edge Detection", "cv_pipeline.py -> detect_edges_and_corners()"),

            ("What is the Harris Corner Response and when does it fire?",
             "Harris computes the **structure tensor** $M = \\sum[\\nabla I \\nabla I^T]$ in a local window. "
             "The response $R = \\det(M) - k\\cdot\\text{trace}(M)^2$. When both eigenvalues of $M$ are large "
             "(intensity changes in all directions), $R$ is large -> corner. When one eigenvalue is large -> edge. "
             "When both are small -> flat region. In EcoSort, corner count > 20 is a strong indicator of "
             "crumpled paper. Code: `cv2.cornerHarris(gray, 2, 3, 0.04)`.",
             "Corner Detection", "cv_pipeline.py -> detect_edges_and_corners()"),

            ("What is a GLCM? What features does it provide?",
             "A **Gray-Level Co-occurrence Matrix (GLCM)** $P(i,j)$ counts how often intensity $i$ and $j$ "
             "co-occur at a given spatial offset $(d,\\theta)$. From it we derive: **Contrast** (local variation), "
             "**Homogeneity** (proximity to diagonal — uniform regions), **Energy** (textural uniformity/repetition), "
             "**Correlation** (linear dependency). Metallic cans have high homogeneity/energy; cardboard has high contrast. "
             "Code: `skimage.feature.graycomatrix` + `graycoprops`.",
             "Texture Analysis", "cv_pipeline.py -> compute_texture_features()"),

            ("Difference between Erosion, Dilation, Opening, and Closing?",
             "**Erosion**: shrinks foreground, removes small protrusions (min filter). "
             "**Dilation**: expands foreground, fills small holes (max filter). "
             "**Opening** = Erosion then Dilation: removes small foreground noise specks. "
             "**Closing** = Dilation then Erosion: fills small background holes in objects. "
             "In EcoSort, opening removes noise, closing ensures the waste mask is solid. "
             "Code: `cv2.morphologyEx(mask, cv2.MORPH_OPEN/CLOSE, kernel)`.",
             "Morphological Operations", "cv_pipeline.py -> apply_morphology()"),

            ("How does Connected Component Labeling work?",
             "CCL assigns a unique integer label to each group of connected foreground pixels. "
             "It uses a two-pass algorithm: first pass assigns provisional labels and records equivalences "
             "(union-find); second pass resolves equivalences and relabels. "
             "`cv2.connectedComponentsWithStats` returns label map, bounding boxes, centroids, and pixel areas. "
             "In EcoSort, components below `min_area` are discarded as noise.",
             "Connected Component Analysis", "cv_pipeline.py -> label_connected_components()"),

            ("What is the Euclidean Distance Transform used for in EcoSort?",
             "The **EDT** assigns to each foreground pixel the Euclidean distance to the nearest background pixel. "
             "It is used in two ways: (1) for Chamfer matching (sum EDT values at input mask pixels against a template), "
             "and (2) as a visual cue for object thickness/width (maximum EDT value approx radius of the inscribed circle). "
             "Code: `cv2.distanceTransform(mask, cv2.DIST_L2, 5)`.",
             "Distance Transform", "cv_pipeline.py -> compute_distance_transform_and_skeleton()"),

            ("What is Morphological Skeletonization?",
             "Skeletonization (thinning) reduces a binary shape to its **medial axis** — a 1-pixel-wide "
             "skeleton that preserves the topology and connectivity of the original shape. "
             "It is computed by iteratively removing boundary pixels that don't disconnect the shape. "
             "In EcoSort, elongated skeletons indicate bottles or straws; compact skeletons indicate cans. "
             "Code: `skimage.morphology.skeletonize(mask > 0)`.",
             "Skeletonization", "cv_pipeline.py -> compute_distance_transform_and_skeleton()"),

            ("Explain Circularity and Solidity as shape descriptors.",
             "**Circularity** $C = 4\\pi A / P^2$ measures how close a shape is to a perfect circle ($C=1$). "
             "Cans have high circularity; bottles and paper have low. "
             "**Solidity** $S = A / A_{convex}$ measures the fraction of the convex hull area filled by the shape. "
             "High solidity (approaching 1) means compact/convex (can, box). Low solidity means concave/irregular (organic waste). "
             "These are computed from `cv2.moments()` and `cv2.convexHull()`.",
             "Shape Descriptors", "cv_pipeline.py -> extract_contour_features()"),

            ("What are Hu Moments and why are they invariant?",
             "Hu Moments are 7 nonlinear combinations of **central normalized moments** $\\eta_{pq}$ "
             "that are invariant to **translation, scale, and rotation**. "
             "They characterize the overall shape distribution of an object. "
             "In code: `cv2.HuMoments(cv2.moments(contour))` then log-transformed as "
             "$\\phi_k = -\\text{sign}(h_k)\\log_{10}|h_k|$. In EcoSort, they help distinguish "
             "bottle profiles from can profiles.",
             "Invariant Moments", "cv_pipeline.py -> extract_contour_features()"),

            ("What are Fourier Descriptors?",
             "Fourier Descriptors represent the boundary as a complex signal $z_n = x_n + j y_n$ and apply "
             "the DFT: $a_k = \\sum z_n e^{-j2\\pi kn/N}$. The low-frequency coefficients (first 10, skip DC) "
             "capture the overall shape; high-frequency coefficients capture fine details. "
             "They are scale-normalized by dividing by $|a_1|$. High variance in FDs indicates irregular or organic boundaries.",
             "Fourier Descriptors", "cv_pipeline.py -> extract_contour_features()"),

            ("What is the Freeman Chain Code?",
             "The Freeman Chain Code encodes a boundary as a sequence of **8 directional codes** (0=E, 1=NE, ..., 7=SE) "
             "representing steps between consecutive boundary pixels. The histogram of these codes and its "
             "**Shannon entropy** $H = -\\sum p_d \\log_2 p_d$ characterize boundary regularity. "
             "Low entropy indicates smooth regular boundaries (can/bottle). High entropy indicates irregular boundaries (organic/crumpled paper).",
             "Chain Code Analysis", "cv_pipeline.py -> extract_contour_features()"),

            ("Explain the Hough Line Transform mathematically.",
             "Every edge point $(x,y)$ is transformed to the **parameter space** $(\\rho, \\theta)$ via "
             "$\\rho = x\\cos\\theta + y\\sin\\theta$. Each edge point votes for all $(\\rho,\\theta)$ pairs that pass "
             "through it. Lines are detected as peaks in the accumulator array. "
             "`HoughLinesP` (probabilistic) is faster — it randomly samples and verifies candidate lines. "
             "In EcoSort, line count > 4 strongly suggests cardboard/paper.",
             "Hough Line Transform", "cv_pipeline.py -> apply_hough_transforms()"),

            ("Explain the Hough Circle Transform.",
             "For each edge point $(x,y)$, circles of all valid radii are parameterized as "
             "$(a,b) = (x - r\\cos\\phi,\\; y - r\\sin\\phi)$. Peaks in the 3D accumulator $(a,b,r)$ correspond to circles. "
             "`cv2.HoughCircles` uses **gradient direction** to reduce it to a 2D problem (center only, then radius). "
             "In EcoSort, detected circles -> can tops/caps -> strong Metal/Tin Can indicator.",
             "Hough Circle Transform", "cv_pipeline.py -> apply_hough_transforms()"),

            ("How is specular ratio computed and what does it indicate?",
             "In HSV space, specular/metallic highlights appear as pixels with **high Value (V > 220)** and "
             "**low Saturation (S < 50)** — bright but colorless (desaturated). "
             "$R_{spec} = |\\{(x,y): V>220 \\wedge S<50\\}| / |\\text{mask}|$. "
             "High specular ratio indicates glossy metallic or plastic surfaces. "
             "Low specular ratio indicates matte surfaces characteristic of paper or organic waste.",
             "Shape-from-Shading", "cv_pipeline.py -> analyze_specular_shading()"),

            ("What is Chamfer Distance and how is it used for template matching?",
             "Chamfer matching compares an input binary mask $I$ against canonical templates $T$ using: "
             "$d = \\frac{1}{|I|} \\sum_{p \\in I} DT_T(p)$ where $DT_T$ is the distance transform of the "
             "inverted template. A small $d$ means input mask pixels land close to template boundary pixels -> closer match. "
             "In EcoSort, 4 templates (bottle, can, box, organic blob) are generated procedurally and the "
             "best match (argmin) is used as a classification feature.",
             "Chamfer Matching", "cv_pipeline.py -> chamfer_matching()"),

            ("Why did you NOT use deep learning for this project?",
             "This project focuses on classical Computer Vision algorithms — thresholding, morphology, "
             "feature descriptors, Hough transforms, and geometric analysis. Using a pre-trained CNN (YOLO/ResNet) "
             "would be an opaque black box that cannot be mathematically explained. "
             "Classical CV provides full algorithmic transparency: every classification decision traces back to an "
             "explicit mathematical formula and transparent feature extraction pipeline.",
             "Architectural Rationale", "classifier.py -> classify_object()"),

            ("What is the difference between Otsu's and Adaptive Thresholding?",
             "**Otsu's** computes a **single global threshold** from the image histogram by maximizing inter-class variance. "
             "It works well when illumination is uniform. "
             "**Adaptive Thresholding** computes a **local threshold** for each pixel based on the mean (or Gaussian-weighted mean) "
             "of its neighbourhood block. It handles uneven illumination and shadows significantly better. "
             "In EcoSort, adaptive thresholding is preferred for real-world camera images.",
             "Image Thresholding", "cv_pipeline.py -> threshold_image()"),

            ("How does the EcoSort classifier assign confidence scores?",
             "Each of 4 scoring functions assigns weighted points based on feature values "
             "(e.g., circularity > 0.7 yields +3 pts for Metal Can). Raw scores are normalized to [0,1]. "
             "**Confidence** = $\\text{winner\\_score} / \\sum \\text{all\\_scores}$. "
             "This yields a calibrated probability distribution across categories. "
             "Explanations are dynamically generated strings with actual measured feature values embedded.",
             "Classifier Design", "classifier.py -> classify_object()"),

            ("How would you extend EcoSort to a real-time conveyor belt system?",
             "Extensions would include: (1) MOG2 background subtraction for moving belt background removal. "
             "(2) Optical flow (Lucas-Kanade) for tracking objects across frames. "
             "(3) A trained Support Vector Machine (SVM) or Random Forest on the same classical feature vector. "
             "(4) Calibrated camera with known metric scale for physical dimension estimation. "
             "(5) Servo/pneumatic bin actuator control via serial port triggered by classification output.",
             "Motion & Tracking", "app.py -> MOG2 webcam mode"),
        ]

        for i, (question, answer, topic, code_ref) in enumerate(qa_pairs, 1):
            with st.expander(f"Q{i}: {question}"):
                st.markdown(f"**Answer:** {answer}")
                st.markdown(
                    f'<div class="info-box">'
                    f'<strong>Topic Area:</strong> <em>{topic}</em>&nbsp;&nbsp;|&nbsp;&nbsp;'
                    f'<strong>Code Reference:</strong> <code>{code_ref}</code>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Academic Reference Documents ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Academic Reference Documents")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        try:
            with open("PROJECT_REPORT.md", "r") as f:
                report_text = f.read()
            st.download_button("Download Project Report (Markdown)",
                               data=report_text.encode("utf-8"),
                               file_name="PROJECT_REPORT.md",
                               mime="text/markdown",
                               use_container_width=True)
        except FileNotFoundError:
            st.warning("PROJECT_REPORT.md not found in project directory.")
    with col_r2:
        try:
            with open("VIVA_GUIDE.md", "r") as f:
                viva_text = f.read()
            st.download_button("Download Viva Guide (Markdown)",
                               data=viva_text.encode("utf-8"),
                               file_name="VIVA_GUIDE.md",
                               mime="text/markdown",
                               use_container_width=True)
        except FileNotFoundError:
            st.warning("VIVA_GUIDE.md not found in project directory.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding:1rem 0; border-top:1px solid #e2e8f0; margin-top:2rem;">
    <em>EcoSort: Visual Waste Classification Assistant</em> &nbsp;|&nbsp;
    <em>Explainable Classical Computer Vision Architecture</em>
</div>
""", unsafe_allow_html=True)


