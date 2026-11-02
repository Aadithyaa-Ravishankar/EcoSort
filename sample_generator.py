"""
sample_generator.py
===================
EcoSort Computer Vision Project — Procedural Test Image Generator

Generates realistic synthetic images of 6 waste-object categories using only
NumPy, OpenCV (cv2), and SciPy. No external datasets or deep-learning models
are required.

Each generator returns a BGR uint8 NumPy array suitable for direct use with
cv2.imshow / cv2.imwrite or as input to the EcoSort analysis pipeline.
"""

import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_noise(image: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Add realistic Gaussian noise to an image (in-place safe copy)."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def _make_canvas(size: tuple, color: tuple = (200, 200, 200)) -> np.ndarray:
    """Return a blank BGR canvas filled with *color* (B, G, R)."""
    h, w = size[1], size[0]
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    return canvas


def _perlin_like_noise(h: int, w: int, scale: float = 0.07,
                       octaves: int = 4) -> np.ndarray:
    """
    Generate a Perlin-like noise field via summed sine/cosine layers.

    Returns a float32 array in [0, 1].
    """
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    frequency = scale

    rng = np.random.default_rng(seed=42)

    for _ in range(octaves):
        ox = rng.uniform(0, 100)
        oy = rng.uniform(0, 100)
        xs = np.linspace(ox, ox + frequency * w, w, dtype=np.float32)
        ys = np.linspace(oy, oy + frequency * h, h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        layer = (np.sin(xg) * np.cos(yg) +
                 np.cos(xg * 1.3) * np.sin(yg * 0.9))
        noise += amplitude * layer
        total_amplitude += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    noise /= total_amplitude
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def _vertical_gradient(h: int, w: int,
                        top_val: float, bot_val: float) -> np.ndarray:
    """Return a (h, w) float32 gradient from top_val to bot_val."""
    col = np.linspace(top_val, bot_val, h, dtype=np.float32)
    return np.tile(col.reshape(-1, 1), (1, w))


# ---------------------------------------------------------------------------
# Generator 1 — Plastic Bottle
# ---------------------------------------------------------------------------

def generate_plastic_bottle(size: tuple = (400, 400), bg_color: tuple = None) -> np.ndarray:
    """
    Generate a synthetic plastic bottle image.

    Visual features engineered:
    - Elongated ellipse body → aspect-ratio feature, contour detection
    - Narrower neck ellipse + circular cap → Hough circles (small radius)
    - Vertical specular highlight strip → high-intensity vertical edge
    - Horizontal label rectangle → rectangle detection, texture region
    - Slight transparency gradient (lighter towards highlight side)
    - Gaussian noise for realism

    Returns
    -------
    np.ndarray
        BGR uint8 array of shape (size[1], size[0], 3).
    """
    w, h = size
    color = bg_color if bg_color is not None else (210, 210, 210)
    canvas = _make_canvas(size, color=color)

    cx, cy = w // 2, h // 2

    # --- Body (elongated ellipse) ---
    body_rx = int(w * 0.18)
    body_ry = int(h * 0.38)
    body_color = (180, 220, 255)   # light-blue PET tint (BGR)

    # Gradient on the body — lighter on the right (simulate transparency)
    body_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(body_mask, (cx, cy), (body_rx, body_ry), 0, 0, 360, 255, -1)

    grad_h = _vertical_gradient(h, w, 0.75, 1.0)          # left-right gradient
    x_col = np.linspace(0.7, 1.0, w, dtype=np.float32)
    grad_x = np.tile(x_col, (h, 1))

    for c_idx, base in enumerate(body_color):
        channel = canvas[:, :, c_idx].astype(np.float32)
        fill = (base * grad_x).clip(0, 255)
        mask_bool = body_mask > 0
        channel[mask_bool] = fill[mask_bool]
        canvas[:, :, c_idx] = channel.astype(np.uint8)

    # --- Neck ---
    neck_rx = int(body_rx * 0.55)
    neck_ry = int(body_ry * 0.18)
    neck_cy = cy - body_ry - neck_ry + int(neck_ry * 0.5)
    neck_color = (160, 200, 240)
    cv2.ellipse(canvas, (cx, neck_cy), (neck_rx, neck_ry),
                0, 0, 360, neck_color, -1)

    # --- Cap (circle — Hough-detectable) ---
    cap_r = int(neck_rx * 0.9)
    cap_cy = neck_cy - neck_ry
    cap_color = (50, 50, 200)   # red cap (BGR)
    cv2.circle(canvas, (cx, cap_cy), cap_r, cap_color, -1)
    cv2.circle(canvas, (cx, cap_cy), cap_r, (30, 30, 150), 2)

    # --- Label rectangle ---
    lbl_top = cy - int(body_ry * 0.35)
    lbl_bot = cy + int(body_ry * 0.35)
    lbl_left = cx - body_rx + 2
    lbl_right = cx + body_rx - 2
    label_color = (80, 200, 80)   # green label
    label_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(label_mask, (lbl_left, lbl_top),
                  (lbl_right, lbl_bot), 255, -1)
    # Only paint inside body
    label_region = (label_mask > 0) & (body_mask > 0)
    canvas[label_region] = label_color

    # Label border lines
    cv2.line(canvas, (lbl_left, lbl_top), (lbl_right, lbl_top),
             (60, 160, 60), 2)
    cv2.line(canvas, (lbl_left, lbl_bot), (lbl_right, lbl_bot),
             (60, 160, 60), 2)

    # --- Specular highlight strip ---
    hl_x = cx + int(body_rx * 0.45)
    hl_top = cy - int(body_ry * 0.6)
    hl_bot = cy + int(body_ry * 0.6)
    hl_width = max(3, int(body_rx * 0.12))
    hl_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(hl_mask, (hl_x, cy),
                (hl_width, int((hl_bot - hl_top) // 2)),
                0, 0, 360, 255, -1)
    hl_region = (hl_mask > 0) & (body_mask > 0)
    canvas[hl_region] = (255, 255, 255)

    # Soft-blur highlight edge
    hl_blur = cv2.GaussianBlur(
        (hl_mask * 255).astype(np.uint8), (15, 15), 0)
    hl_soft = (hl_blur / 255.0)
    for c_idx in range(3):
        ch = canvas[:, :, c_idx].astype(np.float32)
        ch = ch * (1 - hl_soft * 0.5) + 255 * (hl_soft * 0.5)
        canvas[:, :, c_idx] = ch.clip(0, 255).astype(np.uint8)

    # --- Outline ---
    cv2.ellipse(canvas, (cx, cy), (body_rx, body_ry),
                0, 0, 360, (120, 160, 200), 2)
    cv2.ellipse(canvas, (cx, neck_cy), (neck_rx, neck_ry),
                0, 0, 360, (120, 160, 200), 2)

    canvas = _add_noise(canvas, sigma=5)
    return canvas


# ---------------------------------------------------------------------------
# Generator 2 — Aluminum Can
# ---------------------------------------------------------------------------

def generate_aluminum_can(size: tuple = (400, 400), bg_color: tuple = None) -> np.ndarray:
    """
    Generate a synthetic aluminum can image.

    Visual features engineered:
    - Cylinder body with metallic silver-to-dark horizontal gradient
    - Sine-wave ridge lines → high-frequency horizontal texture
    - Circular top and bottom ellipses + concentric rims → Hough circle detection
    - Pull-tab detail (small rectangle + circle) at top
    - Strong vertical edges on cylinder sides
    - Gaussian noise for realism

    Returns
    -------
    np.ndarray
        BGR uint8 array of shape (size[1], size[0], 3).
    """
    w, h = size
    color = bg_color if bg_color is not None else (195, 195, 195)
    canvas = _make_canvas(size, color=color)

    cx, cy = w // 2, h // 2
    can_rx = int(w * 0.24)
    can_ry = int(h * 0.28)
    ellipse_ry = int(can_rx * 0.35)   # vertical radius of top/bottom ellipses

    top_cy = cy - can_ry
    bot_cy = cy + can_ry

    # --- Body gradient (metallic silver) ---
    body_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(body_mask,
                  (cx - can_rx, top_cy),
                  (cx + can_rx, bot_cy), 255, -1)

    # Metallic gradient: dark on sides, bright in centre
    x_arr = np.arange(w, dtype=np.float32)
    metallic = np.abs(x_arr - cx) / float(can_rx)
    metallic = np.clip(metallic, 0, 1)
    bright = 230
    dark = 75
    metal_val = (bright * (1 - metallic) + dark * metallic)
    metal_val = np.tile(metal_val, (h, 1)).astype(np.float32)

    body_bool = body_mask > 0
    for c_idx in range(3):
        ch = canvas[:, :, c_idx].astype(np.float32)
        ch[body_bool] = metal_val[body_bool]
        canvas[:, :, c_idx] = ch.clip(0, 255).astype(np.uint8)

    # Vertical specular highlight stripe (bright metallic reflection)
    hl_x = cx - int(can_rx * 0.35)
    hl_w = max(8, int(can_rx * 0.20))
    canvas[top_cy:bot_cy, hl_x:hl_x + hl_w] = (250, 250, 255)

    # Slight brand colour tint (red-ish)
    tint_region = body_bool
    canvas[:, :, 2][tint_region] = np.clip(
        canvas[:, :, 2][tint_region].astype(np.int32) + 20, 0, 255
    ).astype(np.uint8)

    # --- Sine-wave ridge lines ---
    ridge_spacing = max(8, h // 20)
    for ry_offset in range(top_cy + ridge_spacing,
                           bot_cy, ridge_spacing):
        pts = []
        for xi in range(cx - can_rx, cx + can_rx + 1, 2):
            yi = ry_offset + int(1.5 * math.sin((xi - cx) * 0.25))
            pts.append([xi, yi])
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_arr], False,
                      (60, 60, 70), 1, cv2.LINE_AA)

    # --- Top ellipse ---
    cv2.ellipse(canvas, (cx, top_cy), (can_rx, ellipse_ry),
                0, 0, 360, (160, 160, 170), -1)
    cv2.ellipse(canvas, (cx, top_cy), (can_rx, ellipse_ry),
                0, 0, 360, (70, 70, 80), 2)

    # --- Circular rim bevels on top face (Hough-detectable circles) ---
    cv2.circle(canvas, (cx, top_cy), int(can_rx * 0.70), (140, 140, 150), 2)
    cv2.circle(canvas, (cx, top_cy), int(can_rx * 0.35), (170, 170, 180), 2)

    # --- Bottom ellipse ---
    cv2.ellipse(canvas, (cx, bot_cy), (can_rx, ellipse_ry),
                0, 0, 360, (130, 130, 140), -1)
    cv2.ellipse(canvas, (cx, bot_cy), (can_rx, ellipse_ry),
                0, 0, 360, (70, 70, 80), 2)

    # --- Pull-tab ---
    tab_w = int(can_rx * 0.28)
    tab_h = int(ellipse_ry * 1.3)
    tab_tl = (cx - tab_w // 2, top_cy - tab_h)
    tab_br = (cx + tab_w // 2, top_cy)
    cv2.rectangle(canvas, tab_tl, tab_br, (100, 100, 110), -1)
    cv2.rectangle(canvas, tab_tl, tab_br, (60, 60, 70), 1)
    tab_hole_r = max(4, tab_w // 4)
    cv2.circle(canvas, (cx, top_cy - tab_h // 2), tab_hole_r,
               (195, 195, 195), -1)
    cv2.circle(canvas, (cx, top_cy - tab_h // 2), tab_hole_r,
               (60, 60, 70), 2)

    # --- Vertical side edges ---
    cv2.line(canvas, (cx - can_rx, top_cy), (cx - can_rx, bot_cy),
             (60, 60, 70), 2)
    cv2.line(canvas, (cx + can_rx, top_cy), (cx + can_rx, bot_cy),
             (60, 60, 70), 2)

    canvas = _add_noise(canvas, sigma=5)
    return canvas


# ---------------------------------------------------------------------------
# Generator 3 — Cardboard Box
# ---------------------------------------------------------------------------

def generate_cardboard_box(size: tuple = (400, 400), bg_color: tuple = None) -> np.ndarray:
    """
    Generate a synthetic cardboard box image.

    Visual features engineered:
    - Large brown rectangle (front face) → rectangle/corner detection
    - Perspective trapezoid (top face) → vanishing-point / Hough lines
    - Many parallel horizontal corrugation lines → strong Hough line signal
    - Flat matte brown colour palette (low saturation) → GLCM texture
    - Dark crease lines on edges
    - Gaussian noise for paper-texture realism

    Returns
    -------
    np.ndarray
        BGR uint8 array of shape (size[1], size[0], 3).
    """
    w, h = size
    color = bg_color if bg_color is not None else (190, 190, 190)
    canvas = _make_canvas(size, color=color)

    # Colours
    front_color = (30, 100, 140)   # matte brown (BGR)
    top_color   = (50, 130, 170)
    side_color  = (20,  80, 115)
    crease_color = (15, 60, 90)

    # Front face rectangle
    f_left  = int(w * 0.12)
    f_right = int(w * 0.80)
    f_top   = int(h * 0.35)
    f_bot   = int(h * 0.88)
    cv2.rectangle(canvas, (f_left, f_top), (f_right, f_bot), front_color, -1)

    # Corrugation lines on front face (many parallel horizontals)
    corr_gap = max(5, (f_bot - f_top) // 22)
    for ly in range(f_top + corr_gap, f_bot, corr_gap):
        shade = front_color[0] - 12
        cv2.line(canvas, (f_left, ly), (f_right, ly),
                 (max(0, shade), front_color[1] - 8,
                  front_color[2] - 8), 1)

    # Top face (perspective trapezoid)
    vp_x = int(w * 0.90)   # vanishing-point x (right side)
    vp_y = int(h * 0.10)   # vanishing-point y (above box)
    top_face = np.array([
        [f_left,  f_top],
        [f_right, f_top],
        [vp_x,    vp_y],
        [f_left + int((vp_x - f_left) * 0.18), vp_y + int((f_top - vp_y) * 0.25)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [top_face], top_color)

    # Corrugation on top face (follow perspective — lines from left to right)
    for ratio in np.linspace(0.1, 0.9, 12):
        y_left  = int(vp_y + (f_top - vp_y) * ratio +
                      (top_face[3][1] - vp_y) * (1 - ratio))
        y_right = int(vp_y + (f_top - vp_y) * ratio)
        x_left  = int(top_face[3][0] + (f_left - top_face[3][0]) * ratio)
        x_right = int(vp_x + (f_right - vp_x) * ratio)
        cv2.line(canvas, (x_left, y_left), (x_right, y_right),
                 (max(0, top_color[0] - 10),
                  top_color[1] - 8,
                  top_color[2] - 8), 1)

    # Right side face (narrow trapezoid)
    side_face = np.array([
        [f_right, f_top],
        [vp_x,    vp_y],
        [vp_x,    vp_y + (f_bot - f_top)],
        [f_right, f_bot],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [side_face], side_color)

    # Vertical corrugation on right face
    for ratio in np.linspace(0.1, 0.9, 8):
        x_top = int(f_right + (vp_x - f_right) * ratio)
        y_top = int(f_top  + (vp_y - f_top)  * ratio)
        x_bot = int(f_right + (vp_x - f_right) * ratio)
        y_bot = int(f_bot   + (vp_y + (f_bot - f_top) - f_bot) * ratio)
        cv2.line(canvas, (x_top, y_top), (x_bot, y_bot),
                 (max(0, side_color[0] - 8),
                  side_color[1] - 6,
                  side_color[2] - 6), 1)

    # Crease / edge lines
    cv2.line(canvas, (f_left, f_top), (f_right, f_top), crease_color, 2)
    cv2.line(canvas, (f_left, f_bot), (f_right, f_bot), crease_color, 2)
    cv2.line(canvas, (f_left, f_top), (f_left,  f_bot), crease_color, 2)
    cv2.line(canvas, (f_right, f_top), (f_right, f_bot), crease_color, 2)
    cv2.line(canvas, (f_right, f_top), (vp_x, vp_y), crease_color, 2)
    cv2.line(canvas, (f_right, f_bot), (vp_x, vp_y + (f_bot - f_top)),
             crease_color, 2)

    canvas = _add_noise(canvas, sigma=7)
    return canvas


# ---------------------------------------------------------------------------
# Generator 4 — Crumpled Paper
# ---------------------------------------------------------------------------

def generate_crumpled_paper(size: tuple = (400, 400), bg_color: tuple = None) -> np.ndarray:
    """
    Generate a synthetic crumpled paper image.

    Visual features engineered:
    - Irregular white/gray blob shape (many Shi-Tomasi corners)
    - Layered sine/cosine displacement for wrinkle texture (Perlin-like)
    - Random fold/crease lines (dark gray lines with slight randomness)
    - High local variance (wrinkles) → rich GLCM energy / contrast features
    - No dominant straight edges → low Hough line score

    Returns
    -------
    np.ndarray
        BGR uint8 array of shape (size[1], size[0], 3).
    """
    w, h = size
    color = bg_color if bg_color is not None else (215, 215, 215)
    canvas = _make_canvas(size, color=color)

    cx, cy = w // 2, h // 2
    rng = np.random.default_rng(seed=7)

    # --- Irregular polygon blob ---
    n_pts = 36
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    base_rx = int(w * 0.34)
    base_ry = int(h * 0.34)
    radii = (base_rx + rng.integers(-int(base_rx * 0.25),
                                     int(base_rx * 0.25),
                                     size=n_pts))
    pts_x = (cx + radii * np.cos(angles)).astype(np.int32)
    pts_y = (cy + (base_ry / base_rx) * radii * np.sin(angles)).astype(np.int32)
    blob_pts = np.stack([pts_x, pts_y], axis=1).reshape(-1, 1, 2)

    blob_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(blob_mask, [blob_pts], 255)

    # Base white-gray colour
    canvas[blob_mask > 0] = (230, 230, 230)

    # --- Wrinkle texture via Perlin-like noise ---
    noise_field = _perlin_like_noise(h, w, scale=0.06, octaves=5)
    wrinkle_val = (noise_field * 80 + 150).astype(np.uint8)   # 150–230 range
    wrinkle_bgr = np.stack([wrinkle_val] * 3, axis=2)

    blob_bool = blob_mask > 0
    canvas[blob_bool] = wrinkle_bgr[blob_bool]

    # Smooth the texture slightly
    smooth = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas[blob_bool] = smooth[blob_bool]

    # --- Fold / crease lines ---
    n_folds = rng.integers(6, 12)
    for _ in range(n_folds):
        # Random start/end within blob bounding box
        x0 = int(rng.uniform(cx - base_rx, cx + base_rx))
        y0 = int(rng.uniform(cy - base_ry, cy + base_ry))
        angle = rng.uniform(0, math.pi)
        length = int(rng.uniform(base_rx * 0.3, base_rx * 0.9))
        x1 = x0 + int(length * math.cos(angle))
        y1 = y0 + int(length * math.sin(angle))
        # Slight jitter on the fold line
        n_segs = 8
        prev = (x0, y0)
        for seg in range(1, n_segs + 1):
            t = seg / n_segs
            xm = int(x0 + (x1 - x0) * t + rng.integers(-4, 5))
            ym = int(y0 + (y1 - y0) * t + rng.integers(-4, 5))
            shade = int(rng.uniform(100, 160))
            cv2.line(canvas, prev, (xm, ym),
                     (shade, shade, shade), 1, cv2.LINE_AA)
            prev = (xm, ym)

    # --- Outline of blob ---
    cv2.polylines(canvas, [blob_pts], True, (160, 160, 160), 2, cv2.LINE_AA)

    canvas = _add_noise(canvas, sigma=8)
    return canvas


# ---------------------------------------------------------------------------
# Generator 5 — Organic Waste
# ---------------------------------------------------------------------------

def generate_organic_waste(size: tuple = (400, 400), bg_color: tuple = None) -> np.ndarray:
    """
    Generate a synthetic organic waste image (apple / leaf-like blob).

    Visual features engineered:
    - Irregular polygon perturbed by noise (low circularity)
    - Warm green / yellow-green HSV colour palette → HSV histogram signature
    - Natural texture noise (Perlin-like) for surface variation
    - Colour gradient (light in centre, darker at edges)
    - No straight lines → low Hough line score
    - Brown/dark patches simulating bruising or decay

    Returns
    -------
    np.ndarray
        BGR uint8 array of shape (size[1], size[0], 3).
    """
    w, h = size
    color = bg_color if bg_color is not None else (180, 190, 175)
    canvas = _make_canvas(size, color=color)

    cx, cy = w // 2, h // 2
    rng = np.random.default_rng(seed=13)

    # --- Irregular organic blob (apple-like) ---
    n_pts = 48
    angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    rx_base = int(w * 0.30)
    ry_base = int(h * 0.32)
    # Perturb radii with smooth noise
    noise_r = rng.uniform(-0.22, 0.22, size=n_pts)
    # Smooth the perturbation
    from scipy.ndimage import uniform_filter1d
    noise_r = uniform_filter1d(noise_r, size=5, mode='wrap')
    rx_arr = (rx_base * (1 + noise_r)).astype(np.int32)
    ry_arr = (ry_base * (1 + noise_r * 0.8)).astype(np.int32)

    pts_x = (cx + rx_arr * np.cos(angles)).astype(np.int32)
    pts_y = (cy + ry_arr * np.sin(angles)).astype(np.int32)
    blob_pts = np.stack([pts_x, pts_y], axis=1).reshape(-1, 1, 2)

    blob_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(blob_mask, [blob_pts], 255)

    # --- Colour: yellow-green HSV rendered to BGR ---
    # Base green-yellow hue (HSV: ~70–90 °)
    hue_base = 75
    sat_base = 180
    val_base = 170

    blob_bool = blob_mask > 0

    # Perlin noise for texture
    pnoise = _perlin_like_noise(h, w, scale=0.05, octaves=4)

    # Build HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.clip(hue_base + (pnoise * 20 - 10), 55, 100).astype(np.uint8)
    hsv[:, :, 1] = np.clip(sat_base + (pnoise * 40 - 20), 120, 220).astype(np.uint8)

    # Radial gradient for value (bright centre, darker edges)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xs - cx) / rx_base) ** 2 + ((ys - cy) / ry_base) ** 2)
    dist = np.clip(dist, 0, 1)
    val_field = (val_base + (1 - dist) * 50).clip(0, 255).astype(np.uint8)
    val_field = (val_field + (pnoise * 30 - 15)).clip(0, 255).astype(np.uint8)
    hsv[:, :, 2] = val_field

    bgr_organic = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    canvas[blob_bool] = bgr_organic[blob_bool]

    # --- Bruising / decay patches ---
    n_patches = rng.integers(2, 5)
    for _ in range(n_patches):
        px = int(rng.uniform(cx - rx_base * 0.6, cx + rx_base * 0.6))
        py = int(rng.uniform(cy - ry_base * 0.6, cy + ry_base * 0.6))
        prx = int(rng.uniform(rx_base * 0.06, rx_base * 0.18))
        pry = int(rng.uniform(ry_base * 0.06, ry_base * 0.16))
        patch_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(patch_mask, (px, py), (prx, pry), 0, 0, 360, 255, -1)
        patch_bool = (patch_mask > 0) & blob_bool
        # Dark brown-green
        canvas[patch_bool] = (20, 60, 40)

    # Smooth the whole organic region
    smooth = cv2.GaussianBlur(canvas, (5, 5), 0)
    canvas[blob_bool] = smooth[blob_bool]

    # --- Outline ---
    cv2.polylines(canvas, [blob_pts], True, (20, 80, 30), 2, cv2.LINE_AA)

    # --- Stem (thin dark line at top) ---
    stem_base = (cx, cy - ry_base + 5)
    stem_tip  = (cx + int(rx_base * 0.08),
                 cy - ry_base - int(ry_base * 0.18))
    cv2.line(canvas, stem_base, stem_tip, (20, 60, 20), 3, cv2.LINE_AA)

    canvas = _add_noise(canvas, sigma=6)
    return canvas


# ---------------------------------------------------------------------------
# Generator 6 — Multi-Object Tray
# ---------------------------------------------------------------------------

def generate_multi_object_tray(size: tuple = (600, 600)) -> np.ndarray:
    """
    Generate a scene containing 4 waste objects on a neutral tray background.

    Calls generate_plastic_bottle, generate_aluminum_can,
    generate_cardboard_box, and generate_crumpled_paper; generates each with
    the same neutral background colour, and composites them at 4 non-overlapping
    quadrant positions on the 600×600 tray canvas.

    Visual features engineered:
    - Multiple object classes in one scene → multi-label detection scenario
    - Connected component analysis segments all 4 objects independently
    - Neutral light-gray background → background/foreground separation
    - Each sub-image retains its class-specific feature signatures

    Returns
    -------
    np.ndarray
        BGR uint8 array of shape (size[1], size[0], 3).
    """
    tw, th = size
    tray_bg = (215, 215, 215)
    tray = _make_canvas(size, color=tray_bg)

    # Sub-image size for each quadrant (no dark outer border so tray is clean)
    sub_size = (240, 240)

    generators = [
        (generate_plastic_bottle,  35,  35),   # top-left
        (generate_aluminum_can,   325,  35),   # top-right
        (generate_cardboard_box,   35, 325),   # bottom-left
        (generate_crumpled_paper, 325, 325),   # bottom-right
    ]

    for gen_fn, px, py in generators:
        sub_img = gen_fn(size=sub_size, bg_color=tray_bg)
        tray[py:py + sub_size[1], px:px + sub_size[0]] = sub_img

    tray = _add_noise(tray, sigma=4)
    return tray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GENERATOR_MAP = {
    "Plastic Bottle":  generate_plastic_bottle,
    "Aluminum Can":    generate_aluminum_can,
    "Cardboard Box":   generate_cardboard_box,
    "Crumpled Paper":  generate_crumpled_paper,
    "Organic Waste":   generate_organic_waste,
    "Multi-Object Tray": generate_multi_object_tray,
}


def get_sample_image(name: str) -> np.ndarray:
    """
    Return a procedurally generated BGR sample image for the given waste class.

    Parameters
    ----------
    name : str
        One of:
        - "Plastic Bottle"
        - "Aluminum Can"
        - "Cardboard Box"
        - "Crumpled Paper"
        - "Organic Waste"
        - "Multi-Object Tray"

    Returns
    -------
    np.ndarray
        BGR uint8 array.  "Multi-Object Tray" returns a 600×600 image;
        all others return 400×400 by default.

    Raises
    ------
    ValueError
        If *name* is not one of the recognised class names.
    """
    if name not in _GENERATOR_MAP:
        valid = ", ".join(f'"{k}"' for k in _GENERATOR_MAP)
        raise ValueError(
            f"Unknown sample name '{name}'. "
            f"Valid options are: {valid}."
        )
    return _GENERATOR_MAP[name]()


# ---------------------------------------------------------------------------
# Quick self-test / demo  (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "sample_outputs")
    os.makedirs(out_dir, exist_ok=True)

    for cls_name in _GENERATOR_MAP:
        img = get_sample_image(cls_name)
        filename = cls_name.lower().replace(" ", "_") + ".png"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, img)
        print(f"Saved {cls_name:22s} → {path}  shape={img.shape}")

    print("\nAll sample images generated successfully.")
