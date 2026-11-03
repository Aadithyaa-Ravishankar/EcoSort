"""
classifier.py — EcoSort CV Project (Course: 21CSE454T)
=======================================================
Explainable Rule-Based Classifier with Weighted Feature Scoring.

This module classifies waste objects into four recycling categories using
handcrafted computer-vision features extracted from the CV pipeline. Each
scoring function is fully transparent: it returns both a normalised score
and a human-readable list of reasons that explain *why* the score was
assigned — making the system inherently explainable without any neural network.

Dependencies: numpy, scipy, sklearn (optional KNN fallback only).
NO deep learning / NO YOLO / NO pretrained models.

Author      : EcoSort Team
Course      : 21CSE454T — Computer Vision
"""

import logging
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants — Classification categories, bins, and UI badge colours
# ---------------------------------------------------------------------------
CATEGORY_METAL    = "Metal/Tin Can"
CATEGORY_PLASTIC  = "Plastic Bottle"
CATEGORY_PAPER    = "Paper/Cardboard"
CATEGORY_ORGANIC  = "Organic Waste"

BIN_METAL   = "Blue Bin - Recyclable Metal"
BIN_PLASTIC = "Yellow Bin - Recyclable Plastic"
BIN_PAPER   = "Green/Blue Bin - Recyclable Fiber"
BIN_ORGANIC = "Green Bin - Compostable"

BIN_SHORT_METAL   = "Blue Bin"
BIN_SHORT_PLASTIC = "Yellow Bin"
BIN_SHORT_PAPER   = "Green/Blue Bin"
BIN_SHORT_ORGANIC = "Green Bin"

COLOR_METAL   = "#1565C0"   # blue
COLOR_PLASTIC = "#F9A825"   # amber/yellow
COLOR_PAPER   = "#2E7D32"   # dark green
COLOR_ORGANIC = "#388E3C"   # medium green

# Maximum raw scores per category (used for normalisation)
MAX_RAW_METAL   = 20
MAX_RAW_PLASTIC = 20
MAX_RAW_PAPER   = 19
MAX_RAW_ORGANIC = 20

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_get(features: dict, key: str, default: Any = 0.0) -> Any:
    """
    Safely retrieve a value from the features dict.

    Parameters
    ----------
    features : dict
        Feature dictionary assembled by classify_all_objects().
    key : str
        The feature key to look up.
    default : Any
        Fallback value if the key is missing or the value is None.

    Returns
    -------
    Any
        The feature value, or *default* if not available.
    """
    val = features.get(key, default)
    return default if val is None else val


def _normalise(raw_score: float, max_raw: float) -> float:
    """
    Clamp and normalise a raw point-sum to the [0, 1] range.

    Parameters
    ----------
    raw_score : float
        Accumulated points from the scoring criteria.
    max_raw : float
        Maximum possible points for this classifier.

    Returns
    -------
    float
        Normalised score in [0.0, 1.0].
    """
    if max_raw <= 0:
        return 0.0
    return float(np.clip(raw_score / max_raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------

def score_metal_can(features: dict) -> tuple:
    """
    Score a set of CV features for the Metal/Tin Can category.

    The scoring is additive: each criterion contributes a fixed number of
    points when it is satisfied.  The raw total is then normalised to [0, 1].

    Scoring criteria
    ----------------
    +3  circularity > 0.7        → cylindrical can cross-section
    +4  hough_circle_count >= 1  → circular cross-sections detected
    +3  specular_ratio > 0.08    → metallic/glossy surface
    +2  glcm_homogeneity > 0.6   → smooth uniform metallic surface
    +2  glcm_energy > 0.1        → regular texture pattern
    +1  edge_density > 0.1       → sharp rigid boundary
    +1  aspect_ratio in [0.3, 1.5] → cylindrical can proportions
    +1  solidity > 0.85          → compact, non-crumpled object
    ────────────────────────────
    Max raw = 17

    Parameters
    ----------
    features : dict
        Feature dictionary (see classify_all_objects for key names).

    Returns
    -------
    tuple[float, list[str]]
        (normalised_score, list_of_reason_strings)
    """
    raw_score = 0.0
    reasons: list = []

    try:
        # ── Shape descriptors ────────────────────────────────────────────────
        circularity  = float(_safe_get(features, "circularity",  0.0))
        aspect_ratio = float(_safe_get(features, "aspect_ratio", 1.0))
        solidity     = float(_safe_get(features, "solidity",     0.0))
        corner_count = int(_safe_get(features, "corner_count", 0))

        # ── Hough features ───────────────────────────────────────────────────
        hough_circle_count = int(_safe_get(features, "hough_circle_count", 0))

        # ── Texture (GLCM) ───────────────────────────────────────────────────
        glcm_homogeneity = float(_safe_get(features, "glcm_homogeneity", 0.0))
        glcm_energy      = float(_safe_get(features, "glcm_energy",      0.0))

        # ── Specular / surface ───────────────────────────────────────────────
        specular_ratio = float(_safe_get(features, "specular_ratio", 0.0))

        # ── Edge density ─────────────────────────────────────────────────────
        edge_density = float(_safe_get(features, "edge_density", 0.0))

        # ── Chamfer template match ────────────────────────────────────────────
        chamfer_best = str(_safe_get(features, "chamfer_best", "")).lower()

        # ── Criterion 1 : Circularity ─────────────────────────────────────────
        # Unit 2: Shape Descriptors — Circularity C = 4πA/P² measures how
        # close the contour is to a perfect circle. A can's cross-section
        # is nearly circular (C ≈ 0.65–0.85 for a slightly imperfect render).
        if circularity > 0.65:
            raw_score += 3
            reasons.append(
                f"High circularity ({circularity:.2f}) matches cylindrical can cross-section"
            )

        # ── Criterion 2 : Hough Circle Transform ─────────────────────────────
        # Unit 3: Hough Transforms — Detecting circular cross-sections is
        # reliable for a tin can, provided the contour is relatively compact
        # (circularity > 0.55 or solidity > 0.85). If the shape is an elongated bottle
        # or irregular crushed scrap, a circle is a bottle cap or glare artifact.
        if hough_circle_count >= 1:
            if circularity > 0.55 or solidity > 0.85:
                raw_score += 4
                reasons.append(
                    f"Hough Circle Transform detected {hough_circle_count} circular cross-section(s) on compact body"
                )
            else:
                raw_score += 1
                reasons.append(
                    f"Hough Circle detected ({hough_circle_count}) on non-compact contour"
                )

        # ── Criterion 3 : Specular highlight ratio ────────────────────────────
        # Unit 4: Specular Analysis — Metal surfaces reflect light specularly;
        # a high specular-highlight ratio (V>180, S<80) suggests metal/gloss.
        if specular_ratio > 0.05:
            raw_score += 3
            reasons.append(
                f"Specular highlight ratio {specular_ratio:.3f} indicates metallic/glossy surface"
            )

        # ── Criterion 4 : GLCM Homogeneity ───────────────────────────────────
        # Unit 1: GLCM Texture Features — A smooth uniform metallic surface
        # yields high GLCM Homogeneity (nearby pixel-pairs similar in grey).
        if glcm_homogeneity > 0.45:
            raw_score += 2
            reasons.append(
                f"GLCM Homogeneity {glcm_homogeneity:.3f} indicates relatively smooth surface"
            )

        # ── Criterion 5 : GLCM Energy ─────────────────────────────────────────
        # Unit 1: GLCM Texture — High energy ↔ regularity of the texture.
        if glcm_energy > 0.10:
            raw_score += 2
            reasons.append(
                f"High GLCM Energy {glcm_energy:.3f} indicates regular texture (metallic ridges)"
            )

        # ── Criterion 6 : Many corners (ridges on can body) ───────────────────
        # Unit 1: Shi-Tomasi Corner Detection — Metallic cans have sine-wave
        # ridges that create many local corners in the image.
        if corner_count > 150:
            raw_score += 3
            reasons.append(
                f"{corner_count} Shi-Tomasi corners detected — consistent with metallic ridging pattern"
            )

        # ── Criterion 7 : Aspect ratio ────────────────────────────────────────
        # Unit 2: Shape Descriptors — Standard tin cans have aspect ratio
        # (width/height) broadly between 0.45 and 1.25 (elongation < 1.75).
        # Shapes with elongation > 1.80 are elongated bottles or tubes, not cylindrical cans!
        elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 0.001))
        if 0.45 <= aspect_ratio <= 1.25:
            raw_score += 2
            reasons.append(
                f"Aspect ratio {aspect_ratio:.2f} consistent with cylindrical can proportions"
            )
        elif elongation > 1.80:
            raw_score = max(0.0, raw_score - 4)

        # ── Criterion 8 : Solidity ────────────────────────────────────────────
        # Unit 2: Shape Descriptors — A non-crumpled can fills its convex
        # hull very well (solidity ≈ 0.90+).
        if solidity > 0.88:
            raw_score += 1
            reasons.append(
                f"High Solidity {solidity:.2f} indicates compact rigid metallic object"
            )

        # ── Criterion 9 : Chamfer match = 'can' ──────────────────────────────
        # Unit 5: Chamfer Distance Template Matching.
        if chamfer_best == "can":
            raw_score += 3
            reasons.append("Chamfer distance template matching best matches can silhouette")

        # ── Criterion 10 : Achromatic metallic aluminum surface ───────────────
        # Unit 4: Color Analysis — Silver aluminum has low saturation (S < 65).
        mean_sat = float(_safe_get(features, "mean_saturation", 100.0))
        if mean_sat < 65.0:
            raw_score += 2
            reasons.append(
                f"Low saturation ({mean_sat:.1f}) indicates achromatic silver metallic surface"
            )

        # Negative penalty: chamfer matching bottle strongly with very tall aspect ratio
        if chamfer_best == "bottle" and aspect_ratio < 0.42:
            raw_score = max(0.0, raw_score - 3)

    except Exception as exc:
        logger.warning("score_metal_can encountered an error: %s", exc)

    score = _normalise(raw_score, MAX_RAW_METAL)
    return score, reasons


# ---------------------------------------------------------------------------

def score_plastic_bottle(features: dict) -> tuple:
    """
    Score a set of CV features for the Plastic Bottle category.

    Scoring criteria
    ----------------
    +4  aspect_ratio < 0.5        → tall elongated bottle shape
    +2  hough_line_count >= 2     → bottle-neck parallel edges
    +2  specular_ratio > 0.04     → translucent/glossy plastic
    +2  solidity > 0.7            → mostly filled convex shape
    +2  circularity < 0.6         → elongated bottle profile
    +2  chain_code_entropy < 2.5  → smooth regular boundary
    +1  glcm_dissimilarity < 1.5  → uniform plastic surface
    +3  chamfer_best == 'bottle'  → template silhouette match
    ────────────────────────────
    Max raw = 18

    Parameters
    ----------
    features : dict
        Feature dictionary.

    Returns
    -------
    tuple[float, list[str]]
        (normalised_score, list_of_reason_strings)
    """
    raw_score = 0.0
    reasons: list = []

    try:
        # ── Shape descriptors ────────────────────────────────────────────────
        aspect_ratio = float(_safe_get(features, "aspect_ratio", 1.0))
        solidity     = float(_safe_get(features, "solidity",     0.0))
        circularity  = float(_safe_get(features, "circularity",  1.0))
        extent       = float(_safe_get(features, "extent",       0.0))

        # ── Hough features ───────────────────────────────────────────────────
        hough_line_count   = int(_safe_get(features, "hough_line_count", 0))
        hough_circle_count = int(_safe_get(features, "hough_circle_count", 0))

        # ── Surface / specular ───────────────────────────────────────────────
        specular_ratio = float(_safe_get(features, "specular_ratio", 0.0))

        # ── Contour / boundary features ──────────────────────────────────────
        chain_code_entropy = float(_safe_get(features, "chain_code_entropy", 99.0))

        # ── Texture (GLCM) ───────────────────────────────────────────────────
        glcm_dissimilarity = float(_safe_get(features, "glcm_dissimilarity", 99.0))
        glcm_homogeneity   = float(_safe_get(features, "glcm_homogeneity", 0.0))

        # ── Chamfer template matching ────────────────────────────────────────
        chamfer_best = str(_safe_get(features, "chamfer_best", "")).lower()

        # ── Criterion 1 : Aspect ratio / Elongation ───────────────────────────
        # Unit 2: Shape Descriptors — Bottles are elongated whether vertical (aspect_ratio < 0.55)
        # or horizontal (aspect_ratio > 1.80).
        elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 0.001))
        if elongation > 1.80 or aspect_ratio < 0.55:
            raw_score += 5
            reasons.append(
                f"Elongated aspect ratio (aspect={aspect_ratio:.2f}, elongation={elongation:.2f}) matches bottle profile"
            )

        # ── Criterion 2 : Chamfer template matching ───────────────────────────
        # Unit 5: Chamfer Distance Template Matching — best geometric match.
        if chamfer_best == "bottle":
            raw_score += 4
            reasons.append(
                "Chamfer distance template matching best matches bottle silhouette"
            )

        # ── Criterion 3 : Moderate-high solidity (filled but not rectangular) ─
        # Unit 2: Shape Descriptors — An upright bottle fills its convex hull
        # well; solidity 0.80–0.99 confirms a mostly solid elongated shape.
        if 0.80 <= solidity <= 0.99:
            raw_score += 3
            reasons.append(
                f"Solidity {solidity:.2f} indicates mostly filled elongated shape"
            )

        # ── Criterion 4 : Circularity range for elongated bottle profile ───────
        # Unit 2: Circularity — A tall bottle silhouette has 0.45 < C < 0.75.
        if 0.45 <= circularity <= 0.75:
            raw_score += 3
            reasons.append(
                f"Circularity {circularity:.2f} matches elongated oval/bottle profile"
            )

        # ── Criterion 5 : Low extent (bottle doesn't fill bounding box fully) ──
        # Unit 2: Extent — A bottle with neck & body has extent < 0.85.
        if extent < 0.85:
            raw_score += 2
            reasons.append(
                f"Extent {extent:.2f} — bottle shape doesn't fully fill bounding box (neck region)"
            )

        # ── Criterion 6 : Hough Lines & Bottle Cap Circles ────────────────────
        # Unit 3: Hough Transform — Parallel edges or circular bottle cap / neck opening.
        if hough_line_count >= 2:
            raw_score += 1
            reasons.append(
                f"{hough_line_count} Hough lines detected (bottle neck parallel edges)"
            )
        if hough_circle_count >= 1:
            raw_score += 2
            reasons.append(
                f"Detected {hough_circle_count} circular feature(s) consistent with bottle cap/opening"
            )

        # ── Criterion 7 : Specular ratio ──────────────────────────────────────
        # Unit 4: Specular Analysis — PET plastic produces moderate highlights (0.005 to 0.35).
        if 0.005 <= specular_ratio <= 0.35:
            raw_score += 2
            reasons.append(
                f"Specular highlights {specular_ratio:.3f} indicate translucent/glossy plastic"
            )

        # ── Criterion 8 : Chain-code entropy — smooth boundary ────────────────
        # Unit 2: Chain Code Entropy — Low entropy → smooth regular surface.
        if chain_code_entropy < 2.8:
            raw_score += 1
            reasons.append(
                f"Boundary entropy {chain_code_entropy:.2f} indicates smooth regular surface"
            )

    except Exception as exc:
        logger.warning("score_plastic_bottle encountered an error: %s", exc)

    score = _normalise(raw_score, MAX_RAW_PLASTIC)
    return score, reasons


# ---------------------------------------------------------------------------

def score_paper_cardboard(features: dict) -> tuple:
    """
    Score a set of CV features for the Paper/Cardboard category.

    Scoring criteria
    ----------------
    +4  hough_line_count >= 4     → strong linear structure
    +2  solidity > 0.75           → flat rectangular shape
    +2  extent > 0.6              → bounding-box-filling rectangle
    +2  specular_ratio < 0.05     → matte paper surface
    +2  glcm_contrast > 1.0       → fibrous/textured surface
    +2  corner_count > 20         → surface crumpling/creasing
    +1  0.6 < aspect_ratio < 2.0  → paper/box proportions
    +1  circularity < 0.5         → flat or irregular, non-circular
    ────────────────────────────
    Max raw = 16

    Parameters
    ----------
    features : dict
        Feature dictionary.

    Returns
    -------
    tuple[float, list[str]]
        (normalised_score, list_of_reason_strings)
    """
    raw_score = 0.0
    reasons: list = []

    try:
        # ── Shape descriptors ────────────────────────────────────────────────
        solidity     = float(_safe_get(features, "solidity",     0.0))
        extent       = float(_safe_get(features, "extent",       0.0))
        aspect_ratio = float(_safe_get(features, "aspect_ratio", 1.0))
        circularity  = float(_safe_get(features, "circularity",  1.0))

        # ── Hough / edge features ────────────────────────────────────────────
        hough_line_count = int(_safe_get(features, "hough_line_count", 0))
        corner_count     = int(_safe_get(features, "corner_count",     0))
        edge_density     = float(_safe_get(features, "edge_density",   0.0))

        # ── Surface / specular ───────────────────────────────────────────────
        specular_ratio = float(_safe_get(features, "specular_ratio", 0.0))

        # ── Texture (GLCM) ───────────────────────────────────────────────────
        glcm_contrast = float(_safe_get(features, "glcm_contrast", 0.0))

        # ── Chamfer template match ────────────────────────────────────────────
        chamfer_best = str(_safe_get(features, "chamfer_best", "")).lower()

        # ── Criterion 1 : Chamfer matches box ─────────────────────────────────
        # Unit 5: Chamfer Matching — Strongest geometric cue for flat box/paper.
        if chamfer_best == "box":
            raw_score += 4
            reasons.append(
                "Chamfer distance template matching best matches flat box/paper silhouette"
            )

        # ── Criterion 2 : Square-ish aspect ratio (flat box) ──────────────────
        # Unit 2: Shape Descriptors — Cardboard boxes are roughly square or
        # rectangular; aspect_ratio 0.65–1.6 is characteristic.
        if 0.65 <= aspect_ratio <= 1.6:
            raw_score += 3
            reasons.append(
                f"Aspect ratio {aspect_ratio:.2f} consistent with flat box/sheet proportions"
            )

        # ── Criterion 3 : High extent (fills bounding box) ────────────────────
        # Unit 2: Extent — A flat rectangular box/sheet fills its bounding box.
        if extent > 0.75:
            raw_score += 3
            reasons.append(
                f"High Extent {extent:.2f} indicates bounding-box-filling rectangular shape"
            )

        # ── Criterion 4 : High Solidity for rectangular / flat shape ──────────
        # Unit 2: Shape Descriptors — Flat cardboard sheets and rigid boxes fill their
        # convex hull tightly (solidity > 0.85).
        if solidity > 0.85:
            raw_score += 2
            reasons.append(
                f"High Solidity {solidity:.2f} indicates compact non-concave rectangular boundary"
            )

        # ── Criterion 5 : Matte surface (low specular) ────────────────────────
        # Unit 4: Specular Analysis — Uncoated paper/cardboard is matte.
        if specular_ratio < 0.04:
            raw_score += 2
            reasons.append(
                f"Low specular ratio {specular_ratio:.3f} confirms matte paper/cardboard surface"
            )

        # ── Criterion 6 : GLCM Contrast — fibrous texture ────────────────────
        # Unit 1: GLCM — Fibrous cardboard has high local contrast.
        if glcm_contrast > 5.0:
            raw_score += 2
            reasons.append(
                f"High GLCM Contrast {glcm_contrast:.1f} indicates fibrous/textured cardboard surface"
            )

        # ── Criterion 7 : Hough Lines ─────────────────────────────────────────
        # Unit 3: Hough Transform — Folded/corrugated cardboard has dense parallel lines.
        if hough_line_count >= 15:
            raw_score += 4
            reasons.append(
                f"{hough_line_count} Hough lines detected — strong linear structure of cardboard edges"
            )
        elif hough_line_count >= 4:
            raw_score += 2
            reasons.append(
                f"{hough_line_count} Hough lines detected — linear structure of cardboard edges"
            )

        # ── Criterion 8 : Corner count (crumpled paper case) ──────────────────
        # Unit 1: Shi-Tomasi Corners — Crumpled/folded paper has many creases.
        if corner_count > 50:
            raw_score += 2
            reasons.append(
                f"{corner_count} Shi-Tomasi corners — surface creasing/crumpling"
            )

        # ── Criterion 9 : Cardboard / Paper colour signature ──────────────────
        # Unit 4: Color Analysis — Kraft brown (10–38°) or achromatic white paper (low saturation).
        mean_hue = float(_safe_get(features, "mean_hue", -1.0))
        mean_sat = float(_safe_get(features, "mean_saturation", 0.0))
        if 10.0 <= mean_hue <= 38.0 and mean_sat > 50.0:
            raw_score += 3
            reasons.append(
                f"Cardboard brown/tan hue ({mean_hue:.1f}°) matches unbleached kraft paper fiber"
            )
        elif mean_sat < 35.0 and corner_count > 40:
            raw_score += 3
            reasons.append(
                f"Achromatic low-saturation surface ({mean_sat:.1f}) matches bleached white paper"
            )
        elif 45.0 <= mean_hue <= 125.0 and mean_sat > 60.0:
            # Vivid green hue indicates organic matter, NOT cardboard/paper
            raw_score = max(0.0, raw_score - 5)

    except Exception as exc:
        logger.warning("score_paper_cardboard encountered an error: %s", exc)

    score = _normalise(raw_score, MAX_RAW_PAPER)
    return score, reasons


# ---------------------------------------------------------------------------

def score_organic_waste(features: dict) -> tuple:
    """
    Score a set of CV features for the Organic Waste category.

    The organic class is the "catch-all" for irregular natural objects
    (food scraps, leaves, peels, etc.) and is characterised by the
    *absence* of manufactured geometric regularity.

    Scoring criteria
    ----------------
    +3  solidity < 0.75              → irregular organic boundary
    +3  circularity < 0.5            → irregular non-manufactured shape
    +2  hough_line_count == 0        → no manufactured straight lines
    +2  hough_circle_count == 0      → no regular circular geometry
    +2  chain_code_entropy > 2.5     → irregular organic outline
    +2  local_entropy > 5.0          → complex natural surface texture
    +1  specular_ratio < 0.03        → non-glossy organic material
    +1  hue in warm/organic range    → HSV hue consistent with organic
    ────────────────────────────
    Max raw = 16

    Parameters
    ----------
    features : dict
        Feature dictionary.

    Returns
    -------
    tuple[float, list[str]]
        (normalised_score, list_of_reason_strings)
    """
    raw_score = 0.0
    reasons: list = []

    try:
        # ── Shape descriptors ────────────────────────────────────────────────
        solidity    = float(_safe_get(features, "solidity",    1.0))
        circularity = float(_safe_get(features, "circularity", 1.0))
        extent      = float(_safe_get(features, "extent",      1.0))
        aspect_ratio = float(_safe_get(features, "aspect_ratio", 1.0))

        # ── Hough features ───────────────────────────────────────────────────
        hough_line_count   = int(_safe_get(features, "hough_line_count",   1))
        hough_circle_count = int(_safe_get(features, "hough_circle_count", 1))

        # ── Boundary / contour features ──────────────────────────────────────
        chain_code_entropy = float(_safe_get(features, "chain_code_entropy", 0.0))

        # ── Texture ──────────────────────────────────────────────────────────
        local_entropy = float(_safe_get(features, "local_entropy", 0.0))
        glcm_contrast = float(_safe_get(features, "glcm_contrast", 0.0))

        # ── Specular ─────────────────────────────────────────────────────────
        specular_ratio = float(_safe_get(features, "specular_ratio", 1.0))

        # ── Chamfer ──────────────────────────────────────────────────────────
        chamfer_best = str(_safe_get(features, "chamfer_best", "")).lower()

        # ── Colour (HSV hue) ─────────────────────────────────────────────────
        mean_hue = float(_safe_get(features, "mean_hue", -1.0))

        # ── Criterion 1 : Chamfer matches organic ─────────────────────────────
        # Unit 5: Chamfer Matching — Blob-like template → best organic match.
        if chamfer_best == "organic":
            raw_score += 4
            reasons.append(
                "Chamfer distance best matches organic blob silhouette template"
            )

        # ── Criterion 2 : Chain-code boundary entropy ─────────────────────────
        # Unit 2: Chain Code Entropy — High entropy means frequent direction
        # changes → irregular organic boundary.
        if chain_code_entropy > 2.7:
            raw_score += 3
            reasons.append(
                f"High boundary entropy {chain_code_entropy:.2f} indicates irregular organic outline"
            )

        # ── Criterion 3 : Aspect ratio near 1.0 (blob-like) ──────────────────
        # Unit 2: Shape Descriptors — Organic waste (apple, leaf) tends to be
        # blob-like (aspect ratio 0.6–1.4) but not as square as a box.
        if 0.65 <= aspect_ratio <= 1.4:
            raw_score += 2
            reasons.append(
                f"Blob-like aspect ratio {aspect_ratio:.2f} consistent with organic waste shape"
            )

        # ── Criterion 4 : Low extent (irregular fill of bounding box) ─────────
        # Unit 2: Extent — Organic shapes have irregular outlines; extent < 0.78.
        if extent < 0.78:
            raw_score += 3
            reasons.append(
                f"Low Extent {extent:.2f} indicates irregular organic boundary doesn't fill bounding box"
            )

        # ── Criterion 5 : Absence of dense linear structure ───────────────────
        # Unit 3: Hough Transform — Natural organic objects lack rigid parallel manufactured lines.
        # Genuine absence requires that it is not a rigid convex manufactured item (solidity < 0.88).
        if hough_line_count == 0 and solidity < 0.88:
            raw_score += 2
            reasons.append(
                f"Absence of manufactured parallel lines (lines={hough_line_count})"
            )
        elif hough_line_count <= 2 and solidity < 0.82:
            raw_score += 1
            reasons.append(
                f"Minimal linear structure ({hough_line_count} line) consistent with organic scrap"
            )

        # ── Criterion 6 : Absence of dominant circular geometry ───────────────
        # Regular manufactured circles are absent in organic food (and circularity < 0.70).
        if hough_circle_count == 0 and circularity < 0.70 and solidity < 0.88:
            raw_score += 2
            reasons.append(
                "Absence of manufactured circular cross-sections"
            )

        # ── Criterion 7 : Very low specular ratio ────────────────────────────
        # Unit 4: Specular Analysis — Organic materials are non-glossy.
        if specular_ratio < 0.03:
            raw_score += 2
            reasons.append(
                f"Very low specular ratio {specular_ratio:.3f} indicates non-glossy organic material"
            )

        # Negative penalty for high metallic specular highlight
        if specular_ratio > 0.08:
            raw_score = max(0.0, raw_score - 3)

        # ── Criterion 8 : Vivid organic plant / fruit hue range ───────────────
        # Unit 4: Colour Features (HSV) — Green (50-110) or rich natural fruit/vegetable hue
        if mean_hue >= 0:
            mean_sat = float(_safe_get(features, "mean_saturation", 0.0))
            if 50 <= mean_hue <= 110:
                raw_score += 5
                reasons.append(
                    f"Vivid green/yellow organic hue ({mean_hue:.1f}°) strongly indicates plant or food scrap material"
                )
            elif (10 <= mean_hue <= 35) and mean_sat > 140:
                raw_score += 3
                reasons.append(
                    f"Vivid warm natural hue ({mean_hue:.1f}°) consistent with fruit peel or organic waste"
                )

    except Exception as exc:
        logger.warning("score_organic_waste encountered an error: %s", exc)

    score = _normalise(raw_score, MAX_RAW_ORGANIC)
    return score, reasons


# ---------------------------------------------------------------------------
# Master classification function
# ---------------------------------------------------------------------------

def classify_object(features: dict) -> dict:
    """
    Classify a single waste object given its feature dictionary.

    Procedure
    ---------
    1. Run all four scoring functions.
    2. Determine the winner = category with the highest normalised score.
    3. Compute confidence = winner_score / (sum_of_all_scores + ε).
    4. Collect the *syllabus_features* — a list identifying which CV course
       units / algorithms contributed to the final decision.
    5. Return a structured result dictionary suitable for downstream UI
       rendering (bin colour badge, reasons, confidence bar, etc.).

    Parameters
    ----------
    features : dict
        Feature dictionary assembled by classify_all_objects().

    Returns
    -------
    dict with keys:
        category        : str   — winning category name
        bin             : str   — full bin label (e.g. "Blue Bin - Recyclable Metal")
        bin_color       : str   — hex colour for the UI badge
        confidence      : float — fraction of total score held by winner
        all_scores      : dict  — {category: normalised_score}
        reasons         : list  — human-readable explanation strings
        syllabus_features: list — CV unit / algorithm tags that fired
    """
    # ── Run all scorers ──────────────────────────────────────────────────────
    try:
        metal_score,   metal_reasons   = score_metal_can(features)
    except Exception as exc:
        logger.error("score_metal_can failed: %s", exc)
        metal_score, metal_reasons = 0.0, []

    try:
        plastic_score, plastic_reasons = score_plastic_bottle(features)
    except Exception as exc:
        logger.error("score_plastic_bottle failed: %s", exc)
        plastic_score, plastic_reasons = 0.0, []

    try:
        paper_score,   paper_reasons   = score_paper_cardboard(features)
    except Exception as exc:
        logger.error("score_paper_cardboard failed: %s", exc)
        paper_score, paper_reasons = 0.0, []

    try:
        organic_score, organic_reasons = score_organic_waste(features)
    except Exception as exc:
        logger.error("score_organic_waste failed: %s", exc)
        organic_score, organic_reasons = 0.0, []

    # ── Score mapping ────────────────────────────────────────────────────────
    scores = {
        CATEGORY_METAL:   metal_score,
        CATEGORY_PLASTIC: plastic_score,
        CATEGORY_PAPER:   paper_score,
        CATEGORY_ORGANIC: organic_score,
    }

    reasons_map = {
        CATEGORY_METAL:   metal_reasons,
        CATEGORY_PLASTIC: plastic_reasons,
        CATEGORY_PAPER:   paper_reasons,
        CATEGORY_ORGANIC: organic_reasons,
    }

    # ── Determine winner ─────────────────────────────────────────────────────
    winner_category = max(scores, key=lambda c: scores[c])
    winner_score    = scores[winner_category]

    # ── Confidence (share of total accumulated score) ────────────────────────
    total_score = sum(scores.values())
    confidence  = float(winner_score / (total_score + 1e-6))

    # ── Bin / colour lookup ──────────────────────────────────────────────────
    bin_map = {
        CATEGORY_METAL:   (BIN_METAL,   BIN_SHORT_METAL,   COLOR_METAL),
        CATEGORY_PLASTIC: (BIN_PLASTIC, BIN_SHORT_PLASTIC, COLOR_PLASTIC),
        CATEGORY_PAPER:   (BIN_PAPER,   BIN_SHORT_PAPER,   COLOR_PAPER),
        CATEGORY_ORGANIC: (BIN_ORGANIC, BIN_SHORT_ORGANIC, COLOR_ORGANIC),
    }
    full_bin, short_bin, bin_color = bin_map[winner_category]

    # ── Syllabus features — which CV units / algorithms fired ─────────────────
    # We inspect the winning reasons list plus the features themselves to tag
    # the relevant course units from 21CSE454T.
    syllabus_features = _build_syllabus_tags(features, winner_category, reasons_map[winner_category])

    result = {
        "category":         winner_category,
        "bin":              full_bin,
        "bin_short":        short_bin,
        "bin_color":        bin_color,
        "confidence":       round(confidence, 4),
        "all_scores":       {cat: round(s, 4) for cat, s in scores.items()},
        "reasons":          reasons_map[winner_category],
        "syllabus_features": syllabus_features,
    }

    logger.info(
        "classify_object → %s (conf=%.2f, scores=%s)",
        winner_category, confidence, scores
    )
    return result


# ---------------------------------------------------------------------------
# Syllabus tag builder (internal helper)
# ---------------------------------------------------------------------------

def _build_syllabus_tags(features: dict, category: str, reasons: list) -> list:
    """
    Build the list of algorithm and feature descriptor tags that contributed to the
    classification decision.

    Parameters
    ----------
    features : dict
        The feature dictionary for this object.
    category : str
        The winning category (used for category-specific tags).
    reasons : list[str]
        The list of reason strings generated for the winner.

    Returns
    -------
    list[str]
        Deduplicated, sorted list of feature descriptor strings.
    """
    tags = set()

    # Helper: check if a keyword appears in any reason string
    def in_reasons(keyword: str) -> bool:
        return any(keyword.lower() in r.lower() for r in reasons)

    # ── Texture / GLCM ────────────────────────────────────────────────────────
    if in_reasons("GLCM Homogeneity") or _safe_get(features, "glcm_homogeneity", None) is not None:
        if in_reasons("GLCM Homogeneity"):
            tags.add("GLCM Homogeneity")
    if in_reasons("GLCM Energy"):
        tags.add("GLCM Energy")
    if in_reasons("GLCM Contrast"):
        tags.add("GLCM Contrast")
    if in_reasons("GLCM Dissimilarity"):
        tags.add("GLCM Dissimilarity")
    if in_reasons("local texture entropy"):
        tags.add("Local Texture Entropy")

    # ── Shape Descriptors & Boundary ──────────────────────────────────────────
    if in_reasons("circularity"):
        tags.add("Circularity Descriptor")
    if in_reasons("Solidity"):
        tags.add("Solidity Descriptor")
    if in_reasons("Aspect ratio") or in_reasons("aspect ratio"):
        tags.add("Aspect Ratio Descriptor")
    if in_reasons("Extent"):
        tags.add("Extent Descriptor")
    if in_reasons("boundary entropy"):
        tags.add("Chain Code Boundary Entropy")
    if _safe_get(features, "hu_moments", None) is not None:
        tags.add("Hu Moments")
    if _safe_get(features, "fourier_descriptors", None) is not None:
        tags.add("Fourier Descriptors")
    if _safe_get(features, "centroidal_profile", None) is not None:
        tags.add("Centroidal Profile")

    # ── Hough Transforms & Corner Detection ───────────────────────────────────
    if in_reasons("Hough Circle"):
        tags.add("Hough Circle Transform")
    if in_reasons("Hough lines") or in_reasons("Hough line"):
        tags.add("Hough Line Transform")
    if in_reasons("No straight Hough lines"):
        tags.add("Hough Line Transform (absence)")
    if in_reasons("No Hough circles"):
        tags.add("Hough Circle Transform (absence)")
    if in_reasons("Sharp edges") or in_reasons("edge density"):
        tags.add("Edge Detection (Canny)")
    if in_reasons("Shi-Tomasi corners"):
        tags.add("Shi-Tomasi Corner Detection")

    # ── Specular / Colour Analysis ────────────────────────────────────────────
    if in_reasons("Specular highlight") or in_reasons("Specular highlights") or in_reasons("specular ratio"):
        tags.add("Specular Highlight Analysis")
    if in_reasons("HSV hue"):
        tags.add("HSV Colour Space Analysis")

    # ── Template Matching ─────────────────────────────────────────────────────
    if in_reasons("Chamfer template"):
        tags.add("Chamfer Distance Template Matching")

    return sorted(tags)


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def classify_all_objects(pipeline_results: dict) -> list:
    """
    Classify every detected object from the full CV pipeline output.

    This is the primary entry-point called by the main application after
    cv_pipeline.run_full_pipeline() returns.  It flattens per-contour data
    with global pipeline results into a unified features dict, then calls
    classify_object() for each contour.

    Parameters
    ----------
    pipeline_results : dict
        The dictionary returned by cv_pipeline.run_full_pipeline().  Expected
        top-level keys and their nested structure:

        pipeline_results = {
            'contours': [
                {
                    'circularity':         float,
                    'aspect_ratio':        float,
                    'solidity':            float,
                    'extent':              float,
                    'area':                float,
                    'perimeter':           float,
                    'hu_moments':          list | np.ndarray,
                    'fourier_descriptors': list | np.ndarray,
                    'chain_code_entropy':  float,
                    'centroidal_profile':  list | np.ndarray,
                },
                ...
            ],
            'hough': {
                'line_count':   int,
                'circle_count': int,
            },
            'texture': {
                'glcm_contrast':     float,
                'glcm_dissimilarity':float,
                'glcm_homogeneity':  float,
                'glcm_energy':       float,
                'glcm_correlation':  float,
                'local_entropy':     float,
            },
            'edges_corners': {
                'edge_density': float,
                'corner_count': int,
            },
            'specular': {
                'specular_ratio': float,
                'mean_value':     float,
            },
            'chamfer': {
                'best_match': str,   # e.g. 'bottle', 'can', 'box', ''
                'scores':     dict,  # {template_name: distance}
            },
        }

    Returns
    -------
    list[dict]
        One classification result dict per contour (same order as
        pipeline_results['contours']).  Each dict has the structure
        described in classify_object().
    """
    results: list = []

    # ── Validate top-level structure ─────────────────────────────────────────
    if not isinstance(pipeline_results, dict):
        logger.error("classify_all_objects: pipeline_results must be a dict.")
        return results

    contours = pipeline_results.get("contours", [])
    if not contours:
        logger.warning("classify_all_objects: no contours found in pipeline_results.")
        return results

    # ── Extract global (image-level) features once ───────────────────────────
    hough_data    = pipeline_results.get("hough",        {}) or {}
    texture_data  = pipeline_results.get("texture",      {}) or {}
    edges_data    = pipeline_results.get("edges_corners",{}) or {}
    specular_data = pipeline_results.get("specular",     {}) or {}
    chamfer_data  = pipeline_results.get("chamfer",      {}) or {}

    global_features = {
        # ── Hough ──────────────────────────────────────────────────────────
        "hough_line_count":   int(hough_data.get("line_count",   0) or 0),
        "hough_circle_count": int(hough_data.get("circle_count", 0) or 0),

        # ── GLCM Texture ───────────────────────────────────────────────────
        "glcm_contrast":      float(texture_data.get("glcm_contrast",     0.0) or 0.0),
        "glcm_dissimilarity": float(texture_data.get("glcm_dissimilarity",0.0) or 0.0),
        "glcm_homogeneity":   float(texture_data.get("glcm_homogeneity",  0.0) or 0.0),
        "glcm_energy":        float(texture_data.get("glcm_energy",       0.0) or 0.0),
        "glcm_correlation":   float(texture_data.get("glcm_correlation",  0.0) or 0.0),
        "local_entropy":      float(texture_data.get("local_entropy",     0.0) or 0.0),

        # ── Edge / Corner ──────────────────────────────────────────────────
        "edge_density": float(edges_data.get("edge_density", 0.0) or 0.0),
        "corner_count": int(  edges_data.get("corner_count", 0)   or 0),

        # ── Specular ───────────────────────────────────────────────────────
        "specular_ratio": float(specular_data.get("specular_ratio", 0.0) or 0.0),
        "mean_value":     float(specular_data.get("mean_value",     0.0) or 0.0),

        # ── Chamfer template matching ──────────────────────────────────────
        "chamfer_best":   str(chamfer_data.get("best_match", "") or ""),
        "chamfer_scores": chamfer_data.get("scores", {}) or {},
    }

    # ── Per-contour classification ───────────────────────────────────────────
    for idx, contour in enumerate(contours):
        try:
            if not isinstance(contour, dict):
                logger.warning("Contour %d is not a dict; skipping.", idx)
                continue

            # Contour-level specular and color features
            c_spec = float(contour.get("contour_specular_ratio", 0.0) or 0.0)
            g_spec = float(global_features.get("specular_ratio", 0.0) or 0.0)
            # In multi-object scenes, use strictly the contour's own specular ratio
            eff_spec = c_spec if (len(contours) > 1 or c_spec > 0) else g_spec

            bx, by, bw, bh = contour.get("bounding_rect", (0, 0, 0, 0))
            all_circles = hough_data.get("circles", [])
            all_lines   = hough_data.get("lines", [])

            # Localize circles and lines to this contour's bounding box when multiple objects exist
            if len(contours) > 1 and bw > 0 and bh > 0:
                local_circles = [c for c in all_circles if bx <= c[0] <= bx + bw and by <= c[1] <= by + bh]
                local_lines   = [l for l in all_lines   if bx <= (l[0]+l[2])/2.0 <= bx + bw and by <= (l[1]+l[3])/2.0 <= by + bh]
                eff_circle_count = len(local_circles)
                eff_line_count   = len(local_lines)
                eff_corner_count = min(100, max(5, int(float(contour.get("area", 0.0)) / 30.0)))
            else:
                eff_circle_count = global_features["hough_circle_count"]
                eff_line_count   = global_features["hough_line_count"]
                eff_corner_count = global_features["corner_count"]

            # Per-contour Chamfer shape heuristic for multi-object scenes
            contour_aspect = float(contour.get("aspect_ratio", 1.0) or 1.0)
            contour_elong  = max(contour_aspect, 1.0 / max(contour_aspect, 0.001))
            if len(contours) > 1:
                if contour_elong > 1.70:
                    local_chamfer = "bottle"
                elif 0.70 <= contour_aspect <= 1.45 and float(contour.get("solidity", 0.0) or 0.0) > 0.82:
                    local_chamfer = "box"
                elif float(contour.get("circularity", 0.0) or 0.0) > 0.65:
                    local_chamfer = "can"
                else:
                    local_chamfer = global_features["chamfer_best"]
            else:
                local_chamfer = global_features["chamfer_best"]

            features: dict = {
                # ── Per-contour shape descriptors ──────────────────────
                "circularity":         float(contour.get("circularity",  0.0) or 0.0),
                "aspect_ratio":        float(contour.get("aspect_ratio", 1.0) or 1.0),
                "solidity":            float(contour.get("solidity",     0.0) or 0.0),
                "extent":              float(contour.get("extent",       0.0) or 0.0),
                "area":                float(contour.get("area",         0.0) or 0.0),
                "perimeter":           float(contour.get("perimeter",    0.0) or 0.0),

                # ── Per-contour boundary descriptors ───────────────────
                "hu_moments":          contour.get("hu_moments",          None),
                "fourier_descriptors": contour.get("fourier_descriptors", None),
                "chain_code_entropy":  float(contour.get("chain_code_entropy", 0.0) or 0.0),
                "centroidal_profile":  contour.get("centroidal_profile",  None),

                # ── Contour color & shading cues ───────────────────────
                "mean_hue":            float(contour.get("mean_hue", -1.0)),
                "mean_saturation":     float(contour.get("mean_saturation", 0.0) or 0.0),
                "mean_value":          float(contour.get("mean_value", global_features.get("mean_value", 0.0)) or 0.0),

                # ── Merge global features with local overrides ─────────
                **global_features,
                "hough_circle_count":  eff_circle_count,
                "hough_line_count":    eff_line_count,
                "corner_count":        eff_corner_count,
                "chamfer_best":        local_chamfer,
                "specular_ratio":      eff_spec,
            }

            classification = classify_object(features)

            # Attach contour metadata for downstream rendering / debugging
            classification["contour_index"] = idx
            classification["contour_area"]  = features["area"]
            classification["bounding_rect"] = contour.get("bounding_rect", None)
            classification["centroid"]      = contour.get("centroid", None)

            results.append(classification)

        except Exception as exc:
            logger.error("classify_all_objects: error on contour %d: %s", idx, exc)
            # Append a safe fallback so list length matches contour count
            results.append({
                "category":          "Unknown",
                "bin":               "Unknown",
                "bin_short":         "Unknown",
                "bin_color":         "#9E9E9E",
                "confidence":        0.0,
                "all_scores":        {},
                "reasons":           [f"Classification error on contour {idx}: {exc}"],
                "syllabus_features": [],
                "contour_index":     idx,
                "contour_area":      0.0,
            })

    logger.info("classify_all_objects: classified %d object(s).", len(results))
    return results


# ---------------------------------------------------------------------------
# Quick self-test (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal smoke-test to verify the classifier produces sensible output
    for synthetic features representative of each waste category.
    """

    import json

    print("=" * 60)
    print("EcoSort Classifier — Self-Test")
    print("=" * 60)

    # Synthetic feature sets for each expected winner
    test_cases = [
        {
            "label": "Metal Tin Can",
            "features": {
                "circularity": 0.82,
                "aspect_ratio": 0.95,
                "solidity": 0.90,
                "extent": 0.75,
                "area": 5000.0,
                "perimeter": 280.0,
                "hough_line_count": 2,
                "hough_circle_count": 3,
                "glcm_contrast": 0.8,
                "glcm_dissimilarity": 0.9,
                "glcm_homogeneity": 0.75,
                "glcm_energy": 0.20,
                "glcm_correlation": 0.95,
                "local_entropy": 4.0,
                "edge_density": 0.18,
                "corner_count": 15,
                "specular_ratio": 0.15,
                "mean_value": 190.0,
                "chamfer_best": "can",
                "chamfer_scores": {},
                "chain_code_entropy": 1.8,
                "hu_moments": None,
                "fourier_descriptors": None,
                "centroidal_profile": None,
            },
        },
        {
            "label": "Plastic Bottle",
            "features": {
                "circularity": 0.45,
                "aspect_ratio": 0.35,
                "solidity": 0.82,
                "extent": 0.55,
                "area": 7000.0,
                "perimeter": 450.0,
                "hough_line_count": 5,
                "hough_circle_count": 0,
                "glcm_contrast": 0.6,
                "glcm_dissimilarity": 0.9,
                "glcm_homogeneity": 0.55,
                "glcm_energy": 0.08,
                "glcm_correlation": 0.90,
                "local_entropy": 3.5,
                "edge_density": 0.09,
                "corner_count": 8,
                "specular_ratio": 0.09,
                "mean_value": 160.0,
                "chamfer_best": "bottle",
                "chamfer_scores": {},
                "chain_code_entropy": 1.5,
                "hu_moments": None,
                "fourier_descriptors": None,
                "centroidal_profile": None,
            },
        },
        {
            "label": "Paper/Cardboard",
            "features": {
                "circularity": 0.40,
                "aspect_ratio": 1.30,
                "solidity": 0.88,
                "extent": 0.78,
                "area": 12000.0,
                "perimeter": 600.0,
                "hough_line_count": 8,
                "hough_circle_count": 0,
                "glcm_contrast": 2.5,
                "glcm_dissimilarity": 2.1,
                "glcm_homogeneity": 0.40,
                "glcm_energy": 0.05,
                "glcm_correlation": 0.70,
                "local_entropy": 4.5,
                "edge_density": 0.12,
                "corner_count": 35,
                "specular_ratio": 0.02,
                "mean_value": 130.0,
                "chamfer_best": "box",
                "chamfer_scores": {},
                "chain_code_entropy": 2.0,
                "hu_moments": None,
                "fourier_descriptors": None,
                "centroidal_profile": None,
            },
        },
        {
            "label": "Organic Waste",
            "features": {
                "circularity": 0.32,
                "aspect_ratio": 0.85,
                "solidity": 0.60,
                "extent": 0.45,
                "area": 3500.0,
                "perimeter": 320.0,
                "hough_line_count": 0,
                "hough_circle_count": 0,
                "glcm_contrast": 0.9,
                "glcm_dissimilarity": 1.1,
                "glcm_homogeneity": 0.45,
                "glcm_energy": 0.04,
                "glcm_correlation": 0.60,
                "local_entropy": 6.2,
                "edge_density": 0.06,
                "corner_count": 5,
                "specular_ratio": 0.01,
                "mean_value": 90.0,
                "chamfer_best": "",
                "chamfer_scores": {},
                "chain_code_entropy": 3.5,
                "mean_hue": 25.0,   # warm orange
                "hu_moments": None,
                "fourier_descriptors": None,
                "centroidal_profile": None,
            },
        },
    ]

    for tc in test_cases:
        result = classify_object(tc["features"])
        match  = "✓" if tc["label"] in result["category"] else "✗"
        print(f"\n[{match}] Input: {tc['label']}")
        print(f"    → Category  : {result['category']}")
        print(f"    → Bin       : {result['bin']}")
        print(f"    → Confidence: {result['confidence']:.2%}")
        print(f"    → All Scores: {result['all_scores']}")
        print("    → Reasons:")
        for r in result["reasons"]:
            print(f"        • {r}")
        print("    → Syllabus Features:")
        for sf in result["syllabus_features"]:
            print(f"        – {sf}")

    print("\n" + "=" * 60)
    print("Self-test complete.")
