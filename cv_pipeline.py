"""
cv_pipeline.py
==============
EcoSort Computer Vision Pipeline — Course 21CSE454T
Implements all 5 units of the classical CV pipeline using only:
  cv2, numpy, scipy, scikit-image (skimage)

No deep learning. No external models.

Units covered:
  Unit 1 – Image Processing Foundations
  Unit 2 – Shapes and Regions
  Unit 3 – Hough Transforms
  Unit 4 – Motion & Shading Analysis
  Unit 5 – Chamfer Distance Matching
"""

import cv2
import numpy as np
import scipy.stats
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk


# =============================================================================
# UNIT 1 – IMAGE PROCESSING FOUNDATIONS
# =============================================================================

def preprocess_image(bgr_image, gaussian_kernel=5, use_clahe=True):
    """
    Unit 1: Preprocessing.

    Converts an input BGR image into several processed representations used
    throughout the pipeline.

    Operations
    ----------
    1. BGR → Grayscale:
         I_gray = 0.114·B + 0.587·G + 0.299·R   (OpenCV convention)
    2. BGR → HSV colour space (cylindrical: Hue ∈ [0,180], S,V ∈ [0,255])
    3. Gaussian blur (spatial low-pass filter):
         G(x,y) = (1/(2πσ²)) · exp(-(x²+y²)/(2σ²))
         σ chosen automatically by OpenCV when sigmaX=0
    4. Median blur (rank filter; robust to salt-and-pepper noise):
         output(x,y) = median{ I(x+i, y+j) | (i,j) ∈ k×k neighbourhood }
    5. CLAHE (Contrast-Limited Adaptive Histogram Equalisation):
         Limits amplification to clipLimit, operates on tileGridSize tiles.

    Parameters
    ----------
    bgr_image     : np.ndarray  H×W×3 uint8 in BGR order
    gaussian_kernel: int         Odd kernel size for Gaussian and median blur
    use_clahe     : bool         Whether to apply CLAHE to the grayscale image

    Returns
    -------
    dict with keys
        'gray'         – uint8 grayscale image
        'clahe'        – uint8 CLAHE-enhanced grayscale (or copy of gray)
        'gaussian_blur'– uint8 Gaussian-blurred grayscale
        'median_blur'  – uint8 median-blurred grayscale
        'hsv'          – uint8 HSV image
    """
    safe = {
        'gray': None, 'clahe': None,
        'gaussian_blur': None, 'median_blur': None, 'hsv': None
    }
    try:
        if bgr_image is None or bgr_image.size == 0:
            return safe

        # Ensure kernel is odd and >= 1
        k = max(1, int(gaussian_kernel))
        if k % 2 == 0:
            k += 1

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        gauss  = cv2.GaussianBlur(gray, (k, k), 0)
        median = cv2.medianBlur(gray, k)

        if use_clahe:
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe_obj.apply(gray)
        else:
            clahe_img = gray.copy()

        return {
            'gray':          gray,
            'clahe':         clahe_img,
            'gaussian_blur': gauss,
            'median_blur':   median,
            'hsv':           hsv
        }
    except Exception:
        return safe


def threshold_image(gray_image, otsu_offset=0, adaptive_block=11, adaptive_C=2,
                    method='otsu', polarity='auto'):
    """
    Unit 1: Thresholding.

    Binarises a grayscale image using classical binarization strategies with
    adaptive background polarity detection.

    Operations
    ----------
    Otsu's method (global, optimal threshold τ*):
        τ* = argmax_τ [ ω₀(τ)·ω₁(τ)·(μ₀(τ) − μ₁(τ))² ]
    where ω₀, ω₁ are class probabilities and μ₀, μ₁ are class means.

    Adaptive Gaussian thresholding:
        τ(x,y) = Gaussian_weighted_mean_of_block(x,y) − C
    This copes with uneven illumination and shadows.

    Polarity Handling:
        'auto'          – Examines outer 5% border pixels. If > 40% are 255,
                          inverts mask so the background is cleanly 0 (black).
        'dark_on_light' – Foreground object is darker than background.
        'light_on_dark' – Foreground object is lighter than background.

    Parameters
    ----------
    gray_image     : np.ndarray  H×W uint8
    otsu_offset    : int         Added to Otsu threshold (fine-tuning)
    adaptive_block : int         Odd block size for adaptive threshold
    adaptive_C     : int         Constant subtracted from adaptive mean
    method         : str         'otsu', 'adaptive', or 'combined'
    polarity       : str         'auto', 'dark_on_light', or 'light_on_dark'

    Returns
    -------
    dict with keys
        'otsu'           – uint8 binary mask (Otsu)
        'adaptive'       – uint8 binary mask (adaptive Gaussian)
        'combined'       – uint8 binary mask (Otsu | Adaptive)
        'otsu_thresh_val'– float, the Otsu threshold value (before offset)
        'used_polarity'  – str, 'inverted' or 'normal'
    """
    safe = {'otsu': None, 'adaptive': None, 'combined': None, 'otsu_thresh_val': 0.0, 'used_polarity': 'normal'}
    try:
        if gray_image is None or gray_image.size == 0:
            return safe

        # Ensure adaptive_block is odd and >= 3
        ab = max(3, int(adaptive_block))
        if ab % 2 == 0:
            ab += 1

        # Base thresholding (default binary inversion for conveyor setups)
        ret, otsu_raw = cv2.threshold(
            gray_image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        if otsu_offset != 0:
            adjusted_thresh = max(0, min(255, int(ret) + int(otsu_offset)))
            _, otsu_raw = cv2.threshold(
                gray_image, adjusted_thresh, 255, cv2.THRESH_BINARY_INV
            )

        adaptive_raw = cv2.adaptiveThreshold(
            gray_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ab,
            int(adaptive_C)
        )

        def _apply_polarity(mask):
            if mask is None:
                return mask, 'normal'
            if polarity == 'light_on_dark':
                return cv2.bitwise_not(mask), 'inverted'
            elif polarity == 'dark_on_light':
                return mask, 'normal'
            else:  # 'auto'
                h, w = mask.shape
                b = max(2, min(10, min(h, w) // 16))
                border = np.concatenate([
                    mask[:b, :].ravel(),
                    mask[-b:, :].ravel(),
                    mask[:, :b].ravel(),
                    mask[:, -b:].ravel()
                ])
                if np.mean(border > 0) > 0.40:
                    return cv2.bitwise_not(mask), 'inverted'
                return mask, 'normal'

        otsu_mask, pol = _apply_polarity(otsu_raw)
        adaptive_mask, _ = _apply_polarity(adaptive_raw)
        combined_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)

        return {
            'otsu':            otsu_mask,
            'adaptive':        adaptive_mask,
            'combined':        combined_mask,
            'otsu_thresh_val': float(ret),
            'used_polarity':   pol
        }
    except Exception:
        return safe


def detect_edges_and_corners(gray_image, canny_low=50, canny_high=150):
    """
    Unit 1: Edge and Corner Detection.

    Applies multiple edge/corner detectors and aggregates descriptive
    statistics useful for classification.

    Operations
    ----------
    Canny edge detector (multi-stage):
        1. Gaussian smoothing
        2. Sobel gradient magnitude & angle
        3. Non-maximum suppression
        4. Double thresholding + hysteresis (canny_low, canny_high)

    Sobel operators:
        G_x = [-1 0 +1; -2 0 +2; -1 0 +1] * I
        G_y = G_x^T
        |G| = sqrt(G_x² + G_y²)

    Harris corner detector:
        M = Σ_w [G_x² G_xG_y; G_xG_y G_y²]
        R = det(M) − k·trace(M)²,  k=0.04
        Corners where R > 0.01·max(R)

    Shi-Tomasi corner detector:
        score = min(λ₁, λ₂)  (minimum eigenvalue of M)

    Metrics
    -------
    edge_density  = |{pixels : Canny(p) > 0}| / (H·W)
    corner_count  = number of Shi-Tomasi corners returned

    Parameters
    ----------
    gray_image : np.ndarray  H×W uint8
    canny_low  : int         Canny lower hysteresis threshold
    canny_high : int         Canny upper hysteresis threshold

    Returns
    -------
    dict with keys
        'canny'            – uint8 edge map
        'sobel_x'          – float32 horizontal gradient
        'sobel_y'          – float32 vertical gradient
        'sobel_mag'        – float32 gradient magnitude
        'harris_corners'   – float32 Harris response map
        'shitomasi_corners'– Nx1x2 float32 corner array (or empty)
        'edge_density'     – float
        'corner_count'     – int
    """
    safe = {
        'canny': None, 'sobel_x': None, 'sobel_y': None,
        'sobel_mag': None, 'harris_corners': None,
        'shitomasi_corners': np.array([]), 'edge_density': 0.0, 'corner_count': 0
    }
    try:
        if gray_image is None or gray_image.size == 0:
            return safe

        canny = cv2.Canny(gray_image, canny_low, canny_high)

        sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.float32)

        gray_f32 = gray_image.astype(np.float32)
        harris = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
        harris_thresh = np.zeros_like(harris)
        if harris.max() > 0:
            harris_thresh[harris > 0.01 * harris.max()] = 255

        shitomasi = cv2.goodFeaturesToTrack(
            gray_image, maxCorners=200, qualityLevel=0.01, minDistance=10
        )
        if shitomasi is None:
            shitomasi = np.array([])

        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        edge_density = float(np.count_nonzero(canny)) / total_pixels if total_pixels > 0 else 0.0
        corner_count = int(len(shitomasi)) if shitomasi.size > 0 else 0

        return {
            'canny':             canny,
            'sobel_x':           sobel_x,
            'sobel_y':           sobel_y,
            'sobel_mag':         sobel_mag,
            'harris_corners':    harris_thresh.astype(np.float32),
            'shitomasi_corners': shitomasi,
            'edge_density':      edge_density,
            'corner_count':      corner_count
        }
    except Exception:
        return safe


def apply_morphology(binary_mask, kernel_size=5):
    """
    Unit 1: Mathematical Morphology.

    Applies a sequence of morphological operations on a binary mask using
    an elliptical structuring element B of size kernel_size × kernel_size.

    Operations
    ----------
    Erosion:     (A ⊖ B)(x,y) = 1  iff  B shifted to (x,y) ⊆ A
    Dilation:    (A ⊕ B)(x,y) = 1  iff  B shifted to (x,y) ∩ A ≠ ∅
    Opening:     A ∘ B = (A ⊖ B) ⊕ B   (removes small objects)
    Closing:     A • B = (A ⊕ B) ⊖ B   (fills small holes)

    final_mask is the closed result, which is used as the canonical
    working mask for downstream stages.

    Parameters
    ----------
    binary_mask : np.ndarray  H×W uint8 (0 or 255)
    kernel_size : int         Side length of the elliptical structuring element

    Returns
    -------
    dict with keys
        'eroded'     – uint8 eroded mask
        'dilated'    – uint8 dilated mask
        'opened'     – uint8 opened mask
        'closed'     – uint8 closed mask
        'final_mask' – uint8 (same as 'closed')
    """
    safe = {
        'eroded': None, 'dilated': None,
        'opened': None, 'closed': None, 'final_mask': None
    }
    try:
        if binary_mask is None or binary_mask.size == 0:
            return safe

        k = max(1, int(kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        eroded  = cv2.erode(binary_mask, kernel, iterations=1)
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        opened  = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
        closed  = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return {
            'eroded':     eroded,
            'dilated':    dilated,
            'opened':     opened,
            'closed':     closed,
            'final_mask': closed
        }
    except Exception:
        return safe


def compute_texture_features(gray_image, mask=None):
    """
    Unit 1: Texture Analysis using GLCM (Grey-Level Co-occurrence Matrix).

    The GLCM P(i,j|d,θ) counts the frequency that grey levels i and j
    co-occur at distance d apart in direction θ.

    Haralick features (averaged over distances [1,2] and
    angles [0, π/4, π/2, 3π/4]):

        Contrast     = Σ_{i,j} (i−j)² · p(i,j)
        Dissimilarity= Σ_{i,j} |i−j| · p(i,j)
        Homogeneity  = Σ_{i,j} p(i,j) / (1 + (i−j)²)
        Energy       = Σ_{i,j} p(i,j)²      (Angular Second Moment, ASM)
        Correlation  = Σ_{i,j} (i−μᵢ)(j−μⱼ)·p(i,j) / (σᵢ·σⱼ)

    Local entropy (information content of texture):
        H = −Σ p_k · log₂(p_k)
    computed over an 11×11 neighbourhood using skimage rank.entropy
    then averaged over the masked region.

    Parameters
    ----------
    gray_image : np.ndarray  H×W uint8
    mask       : np.ndarray  H×W uint8 (optional, non-zero = ROI)

    Returns
    -------
    dict with keys
        'glcm_contrast'      – float
        'glcm_dissimilarity' – float
        'glcm_homogeneity'   – float
        'glcm_energy'        – float
        'glcm_correlation'   – float
        'local_entropy'      – float
    """
    safe = {
        'glcm_contrast': 0.0, 'glcm_dissimilarity': 0.0,
        'glcm_homogeneity': 0.0, 'glcm_energy': 0.0,
        'glcm_correlation': 0.0, 'local_entropy': 0.0
    }
    try:
        if gray_image is None or gray_image.size == 0:
            return safe

        # Quantise to 64 levels for faster GLCM
        img_q = (gray_image // 4).astype(np.uint8)

        distances = [1, 2]
        angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(img_q, distances=distances, angles=angles,
                            levels=64, symmetric=True, normed=True)

        contrast      = float(np.mean(graycoprops(glcm, 'contrast')))
        dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
        homogeneity   = float(np.mean(graycoprops(glcm, 'homogeneity')))
        energy        = float(np.mean(graycoprops(glcm, 'energy')))
        correlation   = float(np.mean(graycoprops(glcm, 'correlation')))

        # Local entropy via rank filter (disk radius 5 → 11×11 approx)
        try:
            ent_img = rank_entropy(gray_image, disk(5))
            if mask is not None and mask.size > 0:
                roi = ent_img[mask > 0]
                local_entropy = float(roi.mean()) if roi.size > 0 else float(ent_img.mean())
            else:
                local_entropy = float(ent_img.mean())
        except Exception:
            # Fallback: entropy of global histogram
            hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
            hist_prob = hist / (hist.sum() + 1e-10)
            local_entropy = float(scipy.stats.entropy(hist_prob, base=2))

        return {
            'glcm_contrast':      contrast,
            'glcm_dissimilarity': dissimilarity,
            'glcm_homogeneity':   homogeneity,
            'glcm_energy':        energy,
            'glcm_correlation':   correlation,
            'local_entropy':      local_entropy
        }
    except Exception:
        return safe


# =============================================================================
# UNIT 2 – SHAPES AND REGIONS
# =============================================================================

def label_connected_components(binary_mask, min_area=500):
    """
    Unit 2: Connected Component Analysis (CCA).

    Uses 8-connectivity labelling. For a binary image A, two pixels p and q
    are in the same component iff there exists a path p=p₀,p₁,…,pₙ=q where
    each consecutive pair shares at least one of the 8 neighbours.

    cv2.connectedComponentsWithStats returns for each component:
        CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA

    Components whose area < min_area are discarded (label → 0 / background).
    A random colour is assigned to each surviving component for visualisation.

    Parameters
    ----------
    binary_mask : np.ndarray  H×W uint8 (0 or 255)
    min_area    : int         Minimum pixel area to keep a component

    Returns
    -------
    dict with keys
        'labeled_image'  – int32 label map (0 = background)
        'num_components' – int, number of kept components (excl. background)
        'stats'          – list of dicts {label, area, bbox, centroid}
        'centroids'      – Nx2 float32 array of (cx, cy)
        'colored_labels' – uint8 H×W×3 BGR visualisation
    """
    safe = {
        'labeled_image': None, 'num_components': 0,
        'stats': [], 'centroids': np.array([]), 'colored_labels': None
    }
    try:
        if binary_mask is None or binary_mask.size == 0:
            return safe

        # Ensure proper binary uint8
        bm = (binary_mask > 0).astype(np.uint8) * 255

        num_labels, labels, stats_arr, centroids_arr = \
            cv2.connectedComponentsWithStats(bm, connectivity=8)

        h, w = bm.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        kept_stats    = []
        kept_centroids = []
        new_label     = 0

        rng = np.random.default_rng(seed=42)

        for lbl in range(1, num_labels):  # skip background (label 0)
            area = int(stats_arr[lbl, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            new_label += 1
            color = rng.integers(50, 255, size=3).tolist()
            colored[labels == lbl] = color

            cx, cy = float(centroids_arr[lbl, 0]), float(centroids_arr[lbl, 1])
            kept_centroids.append([cx, cy])
            kept_stats.append({
                'label':    new_label,
                'area':     area,
                'bbox':     (
                    int(stats_arr[lbl, cv2.CC_STAT_LEFT]),
                    int(stats_arr[lbl, cv2.CC_STAT_TOP]),
                    int(stats_arr[lbl, cv2.CC_STAT_WIDTH]),
                    int(stats_arr[lbl, cv2.CC_STAT_HEIGHT])
                ),
                'centroid': (cx, cy)
            })

        centroids_out = np.array(kept_centroids, dtype=np.float32) \
            if kept_centroids else np.array([])

        return {
            'labeled_image':  labels.astype(np.int32),
            'num_components': new_label,
            'stats':          kept_stats,
            'centroids':      centroids_out,
            'colored_labels': colored
        }
    except Exception:
        return safe


def compute_distance_transform_and_skeleton(binary_mask):
    """
    Unit 2: Distance Transform and Skeletonization.

    Distance Transform (Euclidean, L2):
        DT(p) = min_{q ∈ background} ‖p − q‖₂
    Computed with a 5×5 mask approximation (cv2.DIST_MASK_5).

    Skeletonization (medial axis, Zhang-Suen thinning):
        Iteratively removes border pixels from foreground regions while
        preserving 8-connectivity and topology, until the image is 1-pixel
        wide (the skeleton or medial axis of the shape).

    Parameters
    ----------
    binary_mask : np.ndarray  H×W uint8 (0 or 255)

    Returns
    -------
    dict with keys
        'distance_transform' – float32 H×W DT values
        'skeleton'           – bool H×W skeleton mask
        'normalized_dt'      – float32 H×W in [0,1]
    """
    safe = {
        'distance_transform': None,
        'skeleton':           None,
        'normalized_dt':      None
    }
    try:
        if binary_mask is None or binary_mask.size == 0:
            return safe

        bm = (binary_mask > 0).astype(np.uint8) * 255

        dt = cv2.distanceTransform(bm, cv2.DIST_L2, 5)
        dt_max = dt.max()
        normalized_dt = (dt / dt_max).astype(np.float32) if dt_max > 0 \
            else dt.astype(np.float32)

        skel = skeletonize(bm > 0)

        return {
            'distance_transform': dt,
            'skeleton':           skel,
            'normalized_dt':      normalized_dt
        }
    except Exception:
        return safe


def _freeman_chain_code(contour):
    """
    Compute the 8-directional Freeman chain code for a contour.

    Direction encoding (dx, dy → code):
        (1,0)=0  (1,-1)=1  (0,-1)=2  (-1,-1)=3
        (-1,0)=4 (-1,1)=5  (0,1)=6   (1,1)=7
    where dy is measured in image coordinates (row increases downward).

    Parameters
    ----------
    contour : np.ndarray  Nx1x2 int32

    Returns
    -------
    codes : list of int (0-7), length N (closed chain)
    """
    dir_map = {
        ( 1,  0): 0,
        ( 1, -1): 1,
        ( 0, -1): 2,
        (-1, -1): 3,
        (-1,  0): 4,
        (-1,  1): 5,
        ( 0,  1): 6,
        ( 1,  1): 7,
    }
    pts = contour[:, 0, :]  # Nx2 (x,y)
    n   = len(pts)
    codes = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx = int(np.clip(p2[0] - p1[0], -1, 1))
        dy = int(np.clip(p2[1] - p1[1], -1, 1))
        codes.append(dir_map.get((dx, dy), 0))
    return codes


def _centroidal_profile(contour, cx, cy, n_samples=64):
    """
    Compute the centroidal profile: distances from centroid (cx, cy)
    to boundary points sampled at n_samples evenly spaced angles.

    For each angle θ_k = 2πk/n_samples, the closest contour point to the
    ray direction is selected and its distance from the centroid recorded.
    If no point is found at a given angle the previous sample is reused.

    Returns
    -------
    profile : np.ndarray float32, shape (n_samples,)
    """
    pts = contour[:, 0, :].astype(np.float32)
    # Angle of each contour point w.r.t. centroid
    dx  = pts[:, 0] - cx
    dy  = pts[:, 1] - cy
    angles     = np.arctan2(dy, dx)          # [-π, π]
    distances  = np.sqrt(dx**2 + dy**2)

    profile = np.zeros(n_samples, dtype=np.float32)
    step    = 2 * np.pi / n_samples
    for k in range(n_samples):
        target = -np.pi + k * step
        diffs  = np.abs(angles - target)
        # Wrap-around
        diffs  = np.minimum(diffs, 2 * np.pi - diffs)
        idx    = np.argmin(diffs)
        profile[k] = distances[idx]
    return profile


def extract_contour_features(binary_mask, original_gray=None, hsv_image=None, min_area=200):
    """
    Unit 2: Boundary Tracking and Shape Descriptors.

    Retrieves all external contours (cv2.RETR_EXTERNAL) using the full
    border-following algorithm (cv2.CHAIN_APPROX_NONE) so every boundary
    pixel is present (required for chain-code and Fourier descriptors).

    Shape descriptors computed per contour
    ---------------------------------------
    Circularity (isoperimetric ratio):
        C = 4π·A / P²     (C=1 for a circle; C→0 for elongated shapes)

    Aspect ratio of bounding rect:
        AR = W / H

    Solidity (convexity measure):
        S = A / A_hull     (S=1 for convex shapes)

    Extent:
        E = A / (W·H)

    Freeman chain code:
        8-directional codes between consecutive boundary pixels (see _freeman_chain_code)

    Chain-code entropy (texture/shape regularity):
        H_cc = −Σ_{k=0}^{7} p_k · log₂(p_k)     (base-2 Shannon entropy)

    Fourier descriptors (rotation/scale-normalised shape signature):
        z_k = x_k + i·y_k  (complex boundary representation)
        FD  = |FFT(z)|      (magnitude spectrum)
        Return first 10 non-DC coefficients (indices 1..10)

    Hu moments (7 invariants to translation, scale, rotation):
        η_{pq} = μ_{pq} / μ_{00}^{1+(p+q)/2}
        Log-transformed: h_k = -sign(Hu_k)·log₁₀(|Hu_k| + 1e-10)

    Parameters
    ----------
    binary_mask  : np.ndarray  H×W uint8
    original_gray: np.ndarray  H×W uint8 (unused, kept for API consistency)
    hsv_image    : np.ndarray  H×W×3 uint8 HSV image for color extraction
    min_area     : int         Minimum pixel area to keep a contour

    Returns
    -------
    list of dict, one entry per valid object contour, keys:
        contour, area, perimeter, circularity, bounding_rect,
        aspect_ratio, solidity, extent, convex_hull, centroid,
        centroidal_profile, chain_code, chain_code_entropy,
        fourier_descriptors, hu_moments, mean_hue, mean_saturation, mean_value
    """
    try:
        if binary_mask is None or binary_mask.size == 0:
            return []

        bm = (binary_mask > 0).astype(np.uint8) * 255
        img_h, img_w = bm.shape[:2]
        contours, _ = cv2.findContours(
            bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        effective_min_area = max(80, int(min_area))
        results = []

        for cnt in contours:
            try:
                area = cv2.contourArea(cnt)
                if area < effective_min_area:
                    continue

                perimeter = cv2.arcLength(cnt, closed=True)
                circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

                x, y, w, h = cv2.boundingRect(cnt)
                # Discard outer canvas/frame border artifact that encloses the entire image
                if w >= 0.92 * img_w and h >= 0.92 * img_h:
                    continue
                aspect_ratio = float(w) / float(h) if h > 0 else 0.0
                extent       = area / float(w * h) if (w * h) > 0 else 0.0

                hull     = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0

                M  = cv2.moments(cnt)
                cx = float(M['m10'] / M['m00']) if M['m00'] != 0 else float(x + w / 2)
                cy = float(M['m01'] / M['m00']) if M['m00'] != 0 else float(y + h / 2)

                # Centroidal profile (64 angular samples)
                if len(cnt) >= 3:
                    centroidal_profile = _centroidal_profile(cnt, cx, cy, n_samples=64)
                else:
                    centroidal_profile = np.zeros(64, dtype=np.float32)

                # Freeman chain code
                if len(cnt) >= 2:
                    chain_code = _freeman_chain_code(cnt)
                    hist = np.bincount(chain_code, minlength=8).astype(float)
                    p    = hist / (hist.sum() + 1e-10)
                    p    = p[p > 0]
                    cc_entropy = float(-np.sum(p * np.log2(p)))
                else:
                    chain_code = []
                    cc_entropy = 0.0

                # Fourier descriptors
                if len(cnt) >= 2:
                    z    = cnt[:, 0, 0].astype(np.float32) + \
                           1j * cnt[:, 0, 1].astype(np.float32)
                    Z    = np.fft.fft(z)
                    fd   = np.abs(Z[1:11]).astype(np.float32)   # skip DC, first 10
                else:
                    fd = np.zeros(10, dtype=np.float32)

                # Hu moments (log-transformed)
                hu_raw = cv2.HuMoments(M).flatten()
                hu     = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10)

                # Color & Specular cues inside this specific contour
                c_mask = np.zeros(bm.shape, dtype=np.uint8)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                c_pts = c_mask > 0
                if hsv_image is not None and np.count_nonzero(c_pts) > 0:
                    H_vals = hsv_image[:, :, 0][c_pts]
                    S_vals = hsv_image[:, :, 1][c_pts]
                    V_vals = hsv_image[:, :, 2][c_pts]
                    c_hue = float(np.median(H_vals))
                    c_sat = float(np.mean(S_vals))
                    c_val = float(np.mean(V_vals))
                    spec_pts = (V_vals > 180) & (S_vals < 80)
                    c_spec_ratio = float(np.count_nonzero(spec_pts)) / float(len(V_vals))
                else:
                    c_hue, c_sat, c_val, c_spec_ratio = -1.0, 0.0, 0.0, 0.0

                results.append({
                    'contour':             cnt,
                    'area':                float(area),
                    'perimeter':           float(perimeter),
                    'circularity':         float(circularity),
                    'bounding_rect':       (x, y, w, h),
                    'aspect_ratio':        float(aspect_ratio),
                    'solidity':            float(solidity),
                    'extent':              float(extent),
                    'convex_hull':         hull,
                    'centroid':            (float(cx), float(cy)),
                    'centroidal_profile':  centroidal_profile,
                    'chain_code':          chain_code,
                    'chain_code_entropy':  float(cc_entropy),
                    'fourier_descriptors': fd,
                    'hu_moments':          hu.astype(np.float32),
                    'mean_hue':            c_hue,
                    'mean_saturation':     c_sat,
                    'mean_value':          c_val,
                    'contour_specular_ratio': c_spec_ratio
                })
            except Exception:
                continue

        return results
    except Exception:
        return []


# =============================================================================
# UNIT 3 – HOUGH TRANSFORMS
# =============================================================================

def apply_hough_transforms(gray_image, edges_image,
                           hough_threshold=50, min_line_length=30,
                           max_line_gap=10, circle_param2=30):
    """
    Unit 3: Hough Transforms.

    Standard Hough Line Transform (accumulator space):
        ρ = x·cos θ + y·sin θ
    Each edge pixel (x,y) votes for all (ρ,θ) that pass through it.
    Peaks in the (ρ,θ) accumulator correspond to dominant lines.

    Probabilistic Hough Line Transform (HoughLinesP) — more efficient:
        Uses a random subset of edge pixels and finds line segments
        directly, controlled by minLineLength and maxLineGap.

    Circular Hough Transform (HOUGH_GRADIENT):
        Uses Canny edges + gradient direction; accumulates votes in (cx, cy)
        parameter space for each candidate radius.

    Parameters
    ----------
    gray_image     : np.ndarray  H×W uint8
    edges_image    : np.ndarray  H×W uint8 (Canny edge map)
    hough_threshold: int         Minimum accumulator votes for a line
    min_line_length: int         Minimum line segment length (pixels)
    max_line_gap   : int         Maximum gap within a line segment (pixels)

    Returns
    -------
    dict with keys
        'lines'         – list of (x1,y1,x2,y2) tuples or []
        'circles'       – Nx3 float32 array [cx,cy,r] or []
        'line_count'    – int
        'circle_count'  – int
        'line_overlay'  – uint8 H×W×3 BGR image with lines drawn
        'circle_overlay'– uint8 H×W×3 BGR image with circles drawn
    """
    safe = {
        'lines': [], 'circles': [], 'line_count': 0, 'circle_count': 0,
        'line_overlay': None, 'circle_overlay': None
    }
    try:
        if gray_image is None or gray_image.size == 0:
            return safe

        base_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        line_overlay   = base_bgr.copy()
        circle_overlay = base_bgr.copy()

        # Probabilistic Hough lines
        raw_lines = cv2.HoughLinesP(
            edges_image if edges_image is not None else np.zeros_like(gray_image),
            rho=1, theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        lines = []
        if raw_lines is not None:
            for seg in raw_lines:
                pts = np.asarray(seg).flatten()
                if len(pts) == 4:
                    x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                    lines.append((x1, y1, x2, y2))
                    cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Circular Hough Transform
        # Apply Gaussian smoothing to suppress high-frequency noise from voting
        blurred_for_circles = cv2.GaussianBlur(gray_image, (9, 9), 2)
        cp2 = max(15, min(60, int(circle_param2)))
        raw_circles = cv2.HoughCircles(
            blurred_for_circles, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=25,
            param1=50, param2=cp2,
            minRadius=12, maxRadius=140
        )

        circles = []
        if raw_circles is not None:
            rc = np.asarray(raw_circles).reshape(-1, 3)
            for pt in rc:
                cx, cy, r = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
                circles.append((cx, cy, r))
                cv2.circle(circle_overlay, (cx, cy), r,  (0, 0, 255), 2)
                cv2.circle(circle_overlay, (cx, cy), 2,  (0, 255, 0), 3)

        return {
            'lines':          lines,
            'circles':        circles,
            'line_count':     len(lines),
            'circle_count':   len(circles),
            'line_overlay':   line_overlay,
            'circle_overlay': circle_overlay
        }
    except Exception:
        return safe


# =============================================================================
# UNIT 4 – MOTION & SHADING
# =============================================================================

def analyze_specular_shading(bgr_image, mask=None):
    """
    Unit 4: Shape-from-Shading / Specular Highlight Analysis.

    Specular highlights arise from mirror-like (Fresnel) reflections on
    glossy or metallic surfaces. In HSV space they appear as:
        • High Value   (V > 220): bright
        • Low Saturation (S < 50): nearly achromatic / white

    This heuristic detects such regions and uses the ratio of specular pixels
    to total foreground pixels as a gloss indicator.

    Model
    -----
    specular_mask(x,y) = 1  iff  V(x,y) > 220  AND  S(x,y) < 50
    specular_ratio = |specular_mask ∩ ROI| / |ROI|
    is_glossy      = specular_ratio > 0.05

    Parameters
    ----------
    bgr_image : np.ndarray  H×W×3 uint8
    mask      : np.ndarray  H×W uint8 (optional; use full image if None)

    Returns
    -------
    dict with keys
        'specular_mask'  – uint8 H×W binary mask of specular pixels
        'specular_ratio' – float
        'mean_saturation'– float, mean S channel over ROI
        'mean_value'     – float, mean V channel over ROI
        'is_glossy'      – bool
    """
    safe = {
        'specular_mask': None, 'specular_ratio': 0.0,
        'mean_saturation': 0.0, 'mean_value': 0.0, 'is_glossy': False
    }
    try:
        if bgr_image is None or bgr_image.size == 0:
            return safe

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        S   = hsv[:, :, 1]
        V   = hsv[:, :, 2]

        # Build ROI mask
        if mask is not None and mask.size == bgr_image.shape[0] * bgr_image.shape[1]:
            roi = (mask > 0)
        else:
            roi = np.ones(bgr_image.shape[:2], dtype=bool)

        # Specular highlight detection.
        # Metallic/glossy surfaces reflect light as bright (V>180) and
        # near-achromatic (S<80) pixels in HSV space.
        # Thresholds are intentionally relaxed (V>180 instead of >220, S<80 instead of <50)
        # so that synthetic gradient highlights and real mild reflections are both captured.
        # For very dark images, 'mean_value' is still a useful matte-vs-glossy discriminator.
        specular_mask = ((V > 180) & (S < 80)).astype(np.uint8) * 255

        roi_pixels      = int(roi.sum())
        specular_pixels = int((specular_mask > 0)[roi].sum())

        specular_ratio  = specular_pixels / roi_pixels if roi_pixels > 0 else 0.0
        mean_saturation = float(S[roi].mean()) if roi_pixels > 0 else 0.0
        mean_value      = float(V[roi].mean()) if roi_pixels > 0 else 0.0
        is_glossy       = specular_ratio > 0.02

        return {
            'specular_mask':   specular_mask,
            'specular_ratio':  float(specular_ratio),
            'mean_saturation': mean_saturation,
            'mean_value':      mean_value,
            'is_glossy':       bool(is_glossy)
        }
    except Exception:
        return safe


# =============================================================================
# UNIT 5 – CHAMFER DISTANCE TEMPLATE MATCHING
# =============================================================================

def _make_template_bottle(h, w):
    """Tall ellipse + neck – plastic bottle silhouette."""
    img = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    brx, bry = int(w * 0.16), int(h * 0.30)
    cv2.ellipse(img, (cx, cy + int(h * 0.05)), (brx, bry), 0, 0, 360, 255, -1)
    nrx, nry = int(brx * 0.45), int(h * 0.12)
    cv2.rectangle(img, (cx - nrx, cy - bry - nry + int(h * 0.05)), (cx + nrx, cy - bry + int(h * 0.05)), 255, -1)
    return img


def _make_template_can(h, w):
    """Cylinder – aluminium can silhouette (aspect ratio ~ 0.75)."""
    img = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    crx, cry = int(w * 0.22), int(h * 0.26)
    cv2.rectangle(img, (cx - crx, cy - cry), (cx + crx, cy + cry), 255, -1)
    cv2.ellipse(img, (cx, cy - cry), (crx, int(crx * 0.35)), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (cx, cy + cry), (crx, int(crx * 0.35)), 0, 0, 360, 255, -1)
    return img


def _make_template_box(h, w):
    """Square/cube – cardboard box silhouette."""
    img  = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    s = int(min(w, h) * 0.30)
    cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), 255, -1)
    return img


def _make_template_organic(h, w):
    """Irregular natural blob – organic waste silhouette."""
    img = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.28)
    cv2.circle(img, (cx, cy), r, 255, -1)
    cv2.ellipse(img, (cx + int(r * 0.3), cy - int(r * 0.2)),
                (int(r * 0.6), int(r * 0.4)), 45, 0, 360, 255, -1)
    return img


def chamfer_matching(binary_mask):
    """
    Unit 5: Symmetric Chamfer Distance Template Matching.

    Computes the two-way (symmetric) Chamfer distance between the input
    binary mask and 4 canonical category templates:

        CD_sym(Q, T) = (1/|Q|)·Σ_{p ∈ Q} DT_T(p) + (1/|T|)·Σ_{t ∈ T} DT_Q(t)

    where DT_T and DT_Q are the Euclidean distance transforms of the inverted
    shapes. Symmetric formulation penalises both missing regions and spurious
    foreground pixels. Lower score → better match.

    Parameters
    ----------
    binary_mask : np.ndarray  H×W uint8 (0 or 255)

    Returns
    -------
    dict with keys
        'scores'     – dict {template_name: chamfer_score (float)}
        'best_match' – str, name of the template with minimum score
    """
    safe = {'scores': {}, 'best_match': 'unknown'}
    try:
        if binary_mask is None or binary_mask.size == 0:
            return safe

        bm = (binary_mask > 0).astype(np.uint8) * 255
        h, w = bm.shape

        query_pixels = np.count_nonzero(bm)
        if query_pixels == 0:
            return safe

        templates = {
            'bottle':  _make_template_bottle(h, w) if h >= w else np.rot90(_make_template_bottle(w, h)).copy(),
            'can':     _make_template_can(h, w),
            'box':     _make_template_box(h, w),
            'organic': _make_template_organic(h, w)
        }

        inv_bm = cv2.bitwise_not(bm)
        dt_q = cv2.distanceTransform(inv_bm, cv2.DIST_L2, 5)

        scores = {}
        for name, tmpl in templates.items():
            try:
                inv_tmpl = cv2.bitwise_not(tmpl)
                dt_t = cv2.distanceTransform(inv_tmpl, cv2.DIST_L2, 5)
                t_pix = np.count_nonzero(tmpl)
                fwd = float(dt_t[bm > 0].sum()) / float(query_pixels)
                bwd = float(dt_q[tmpl > 0].sum()) / float(t_pix) if t_pix > 0 else float('inf')
                scores[name] = float(fwd + bwd)
            except Exception:
                scores[name] = float('inf')

        best = min(scores, key=scores.get)
        return {'scores': scores, 'best_match': best}
    except Exception:
        return safe


# =============================================================================
# MASTER PIPELINE
# =============================================================================

def run_full_pipeline(bgr_image, params=None):
    """
    Master function: executes all five CV pipeline units in sequence.

    The pipeline mirrors the academic unit structure of 21CSE454T:
        Unit 1 → preprocess → threshold → edge/corner → morphology → texture
        Unit 2 → CCA → distance/skeleton → contour features
        Unit 3 → Hough transforms
        Unit 4 → Specular shading analysis
        Unit 5 → Chamfer template matching

    The binary mask used for Units 2-5 is the morphologically-closed
    result from Unit 1 morphology (best_mask = morphology['final_mask']).

    Parameters
    ----------
    bgr_image : np.ndarray  H×W×3 uint8
    params    : dict (optional) with keys:
                    gaussian_kernel  (default 5)
                    canny_low        (default 50)
                    canny_high       (default 150)
                    morph_kernel     (default 5)
                    min_area         (default 500)
                    hough_threshold  (default 50)

    Returns
    -------
    dict with keys:
        'preprocessing'     – output of preprocess_image
        'thresholding'      – output of threshold_image
        'edges_corners'     – output of detect_edges_and_corners
        'morphology'        – output of apply_morphology
        'texture'           – output of compute_texture_features
        'components'        – output of label_connected_components
        'distance_skeleton' – output of compute_distance_transform_and_skeleton
        'contours'          – list from extract_contour_features
        'hough'             – output of apply_hough_transforms
        'specular'          – output of analyze_specular_shading
        'chamfer'           – output of chamfer_matching
    """
    if params is None:
        params = {}

    gaussian_kernel  = int(params.get('gaussian_kernel',  5))
    canny_low        = int(params.get('canny_low',        50))
    canny_high       = int(params.get('canny_high',      150))
    morph_kernel     = int(params.get('morph_kernel',     5))
    min_area         = int(params.get('min_area',        500))
    # Default Hough threshold lowered to 20 so synthetic images reliably
    # produce line/circle detections; users can raise it via the sidebar.
    hough_threshold  = int(params.get('hough_threshold', 20))

    # ── Unit 1 ───────────────────────────────────────────────────────────────
    preprocessing = preprocess_image(bgr_image, gaussian_kernel=gaussian_kernel)

    gray = preprocessing.get('gray')
    if gray is None:
        gray = np.zeros(bgr_image.shape[:2], dtype=np.uint8) \
            if bgr_image is not None else np.zeros((1, 1), dtype=np.uint8)

    clahe_img = preprocessing.get('clahe', gray)

    thresh_method   = params.get('thresh_method', 'otsu')
    thresh_polarity = params.get('thresh_polarity', 'auto')
    otsu_offset     = int(params.get('otsu_offset', 0))
    adaptive_block  = int(params.get('adaptive_block', 11))
    adaptive_C      = int(params.get('adaptive_C', 2))
    circle_param2   = int(params.get('circle_param2', 30))

    thresholding = threshold_image(
        clahe_img if clahe_img is not None else gray,
        otsu_offset=otsu_offset,
        adaptive_block=adaptive_block,
        adaptive_C=adaptive_C,
        method=thresh_method,
        polarity=thresh_polarity
    )
    edges_corners = detect_edges_and_corners(gray, canny_low=canny_low,
                                             canny_high=canny_high)

    # Select base mask according to requested method
    if thresh_method == 'adaptive':
        init_mask = thresholding.get('adaptive')
    elif thresh_method == 'combined':
        init_mask = thresholding.get('combined')
    else:
        init_mask = thresholding.get('otsu')

    if init_mask is None:
        init_mask = thresholding.get('otsu', np.zeros_like(gray))

    morphology    = apply_morphology(init_mask, kernel_size=morph_kernel)
    best_mask     = morphology.get('final_mask', init_mask)
    if best_mask is None:
        best_mask = init_mask

    texture = compute_texture_features(gray, mask=best_mask)

    # ── Unit 2 ───────────────────────────────────────────────────────────────
    components       = label_connected_components(best_mask, min_area=min_area)
    distance_skeleton = compute_distance_transform_and_skeleton(best_mask)
    contours         = extract_contour_features(best_mask, original_gray=gray,
                                                 hsv_image=preprocessing.get('hsv'),
                                                 min_area=min_area)

    # ── Unit 3 ───────────────────────────────────────────────────────────────
    canny_map = edges_corners.get('canny')
    if canny_map is None:
        canny_map = np.zeros_like(gray)

    hough = apply_hough_transforms(
        gray, canny_map,
        hough_threshold=hough_threshold,
        min_line_length=int(params.get('min_line_length', 30)),
        max_line_gap=int(params.get('max_line_gap', 10)),
        circle_param2=circle_param2
    )

    # ── Unit 4 ───────────────────────────────────────────────────────────────
    specular = analyze_specular_shading(bgr_image, mask=best_mask)

    # ── Unit 5 ───────────────────────────────────────────────────────────────
    chamfer = chamfer_matching(best_mask)

    return {
        'preprocessing':     preprocessing,
        'thresholding':      thresholding,
        'edges_corners':     edges_corners,
        'morphology':        morphology,
        'texture':           texture,
        'components':        components,
        'distance_skeleton': distance_skeleton,
        'contours':          contours,
        'hough':             hough,
        'specular':          specular,
        'chamfer':           chamfer
    }


# =============================================================================
# Quick smoke-test (run this file directly to verify no import errors)
# =============================================================================
if __name__ == '__main__':
    import sys

    print("EcoSort CV Pipeline — 21CSE454T")
    print("Smoke-testing with a synthetic 300×300 test image …")

    # Create a simple synthetic scene: white ellipse on grey background
    test_img = np.full((300, 300, 3), 80, dtype=np.uint8)
    cv2.ellipse(test_img, (150, 150), (60, 100), 0, 0, 360, (200, 200, 200), -1)

    results = run_full_pipeline(test_img)

    for stage, val in results.items():
        if stage == 'contours':
            print(f"  {stage}: {len(val)} contour(s)")
        elif isinstance(val, dict):
            keys = list(val.keys())
            print(f"  {stage}: {keys}")
        else:
            print(f"  {stage}: {type(val)}")

    print("\nAll stages completed without exceptions.")
    sys.exit(0)
