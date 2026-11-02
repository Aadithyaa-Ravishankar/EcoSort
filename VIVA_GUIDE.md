# EcoSort CV Project — Viva Voce Preparation Guide
## Course: 21CSE454T — Computer Vision

> [!IMPORTANT]
> This guide covers all 5 syllabus units + project-specific questions. Each answer is concise (3–5 sentences), references the relevant formula, and points to the implementing function in the codebase.

---

## Q1: What is Otsu's thresholding? Derive the criterion.

**Answer:** Otsu's method selects a global threshold $t^*$ that maximizes the **between-class variance** of the image histogram, which equivalently minimizes the weighted within-class variance. The criterion is $\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,[\mu_0(t) - \mu_1(t)]^2$, where $\omega_0, \omega_1$ are the class probabilities and $\mu_0, \mu_1$ are their means. The optimal threshold is $t^* = \arg\max_t \sigma_B^2(t)$, found by scanning all 256 possible values. It requires no user parameter and works well when the histogram is bimodal (clear foreground/background separation).

**Syllabus Unit:** Unit 2 — Image Segmentation  
**Code Reference:** `cv_pipeline.py → segment_object()` — `cv2.threshold(..., cv2.THRESH_OTSU)`

---

## Q2: What is CLAHE and why is it used for illumination normalization?

**Answer:** CLAHE (Contrast Limited Adaptive Histogram Equalization) is an extension of histogram equalization that operates on small local tiles of the image rather than globally, preventing the over-amplification of noise in homogeneous regions. A clip limit caps the histogram height before redistribution, limiting noise gain. In EcoSort, CLAHE is applied to the L (luminance) channel of the LAB color space, preserving hue while correcting uneven lighting across the waste object. This ensures that subsequent thresholding and specular ratio computation are not confounded by shadows or hotspots.

**Syllabus Unit:** Unit 1 — Image Enhancement  
**Code Reference:** `cv_pipeline.py → apply_clahe()` — `cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))`

---

## Q3: Explain Canny edge detection step by step.

**Answer:** Canny detection has four stages: (1) **Gaussian smoothing** — convolve with $G_\sigma$ to suppress noise; (2) **Gradient computation** — apply Sobel operators to obtain magnitude $M = \sqrt{G_x^2 + G_y^2}$ and direction $\theta = \arctan(G_y/G_x)$; (3) **Non-Maximum Suppression (NMS)** — thin edges by retaining only pixels that are local maxima along the gradient direction; (4) **Hysteresis thresholding** — classify pixels as strong ($M > T_{high}$), weak ($T_{low} < M \leq T_{high}$), or suppressed, then accept weak pixels only if 8-connected to a strong pixel. The result is a thin, connected binary edge map.

**Syllabus Unit:** Unit 4 — Feature Detection  
**Code Reference:** `cv_pipeline.py → detect_edges()` — `cv2.Canny(gray, 50, 150)`

---

## Q4: What is the Harris Corner Response function?

**Answer:** The Harris detector constructs the **structure tensor** $M = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$ over a local window, capturing gradient distribution. The corner response is $R = \det(M) - k \cdot \operatorname{trace}(M)^2$, where $k \approx 0.04$–$0.06$. When both eigenvalues of $M$ are large (corner), $R \gg 0$; when one is large (edge), $R < 0$; when both are small (flat region), $R \approx 0$. In EcoSort, high corner counts indicate structured objects like cardboard boxes with printed edges.

**Syllabus Unit:** Unit 4 — Corner Detection  
**Code Reference:** `cv_pipeline.py → detect_harris_corners()` — `cv2.cornerHarris(gray, 2, 3, 0.04)`

---

## Q5: What is a GLCM and what features does it give?

**Answer:** A Gray-Level Co-occurrence Matrix (GLCM) $P(i,j)$ tabulates the frequency of adjacent pixel-intensity pairs $(i,j)$ at a given displacement $(\Delta x, \Delta y)$ and angle. From $P(i,j)$ we derive: **Contrast** $= \sum (i-j)^2 P(i,j)$ (local intensity variation), **Homogeneity** $= \sum P(i,j)/(1+|i-j|)$ (closeness to diagonal), **Energy** $= \sum P(i,j)^2$ (textural uniformity), and **Correlation** (linear dependency of gray levels). In EcoSort, GLCMs are computed at 0°, 45°, 90°, 135° to achieve rotation robustness, and high contrast distinguishes printed newspaper from smooth plastic.

**Syllabus Unit:** Unit 5 — Texture Analysis  
**Code Reference:** `cv_pipeline.py → compute_texture_features()` — `skimage.feature.graycomatrix` + `graycoprops`

---

## Q6: Explain the difference between Erosion, Dilation, Opening, and Closing with examples.

**Answer:** **Erosion** shrinks foreground regions by keeping a pixel only if the structuring element fits entirely within the foreground — useful for removing thin noise spurs. **Dilation** expands foreground regions, filling small holes. **Opening** (erosion then dilation) removes small isolated objects while preserving large ones — EcoSort uses it to strip noise after thresholding. **Closing** (dilation then erosion) fills small internal holes and bridges narrow gaps — EcoSort uses it to unify broken object masks. For example, opening a binary image of a crumpled bag removes stray foreground specks, while closing fills shadow-induced holes inside the bag silhouette.

**Syllabus Unit:** Unit 2 — Morphological Operations  
**Code Reference:** `cv_pipeline.py → apply_morphology()` — `cv2.morphologyEx` with `MORPH_OPEN`, `MORPH_CLOSE`

---

## Q7: What is Connected Component Labeling? How does it work?

**Answer:** Connected Component Labeling (CCL) partitions a binary image into groups of foreground pixels where every pixel in a group is reachable from every other via 4- or 8-connected paths. The standard two-pass algorithm first assigns provisional labels in a raster scan and records equivalences, then resolves equivalences using Union-Find in a second pass. OpenCV's `cv2.connectedComponentsWithStats` returns a label matrix, bounding boxes, areas, and centroids for each component. In EcoSort, the largest component by area is retained as the primary object, eliminating background artifacts.

**Syllabus Unit:** Unit 2 — Segmentation  
**Code Reference:** `cv_pipeline.py → label_connected_components()` — `cv2.connectedComponentsWithStats`

---

## Q8: What is the Euclidean Distance Transform used for?

**Answer:** The Euclidean Distance Transform (EDT) assigns to each foreground pixel its shortest distance to the nearest background pixel: $D(x,y) = \min_{(x',y') \in \text{Background}} \sqrt{(x-x')^2 + (y-y')^2}$. The maximum value $D_{\max}$ identifies the inscribed circle radius of the object, approximating its thickness. In EcoSort, $D_{\max}$ distinguishes flat paper (small thickness) from cylindrical cans or bottles (large inscribed radius). The EDT is also used to generate the distance map $D_Q$ required for Chamfer template matching.

**Syllabus Unit:** Unit 3 — Shape Analysis  
**Code Reference:** `cv_pipeline.py → compute_distance_transform()` — `cv2.distanceTransform(mask, cv2.DIST_L2, 5)`

---

## Q9: What is Morphological Skeletonization?

**Answer:** Skeletonization reduces a binary shape to its **medial axis** — a one-pixel-wide skeleton that preserves the object's topology and connectivity. It works by iteratively applying thinning operations: pixels on the boundary are removed if their removal does not disconnect the skeleton, until no more can be removed. The resulting skeleton passes through the "center" of the shape like the spine of a letter. In EcoSort, the number of **skeleton branches** (junction points) discriminates crumpled paper (many branches) from smooth bottles (few branches), captured by `skimage.morphology.skeletonize`.

**Syllabus Unit:** Unit 3 — Shape Representation  
**Code Reference:** `cv_pipeline.py → compute_skeleton()` — `skimage.morphology.skeletonize`

---

## Q10: Explain Circularity and Solidity as shape descriptors.

**Answer:** **Circularity** $C = 4\pi A / P^2$ measures how close a shape is to a perfect circle ($C = 1$); irregular or elongated shapes have $C \ll 1$. It is sensitive to contour roughness, so a jagged edge decreases $C$ even for a roughly circular object. **Solidity** $S = A / A_{\text{convex}}$ measures the ratio of the shape's area to its convex hull area; $S \approx 1$ for convex objects (bottles, cans) and $S \ll 1$ for concave or highly irregular ones (crumpled bags, wire). In EcoSort, high circularity ($> 0.75$) combined with high solidity ($> 0.85$) strongly indicates a cylindrical metallic can.

**Syllabus Unit:** Unit 3 — Shape Descriptors  
**Code Reference:** `cv_pipeline.py → extract_contour_features()` — computed from `cv2.contourArea`, `cv2.arcLength`, `cv2.convexHull`

---

## Q11: What are Hu Moments and why are they invariant?

**Answer:** Hu Moments are seven scalar values $\phi_1 \ldots \phi_7$ derived from normalized central moments of an image up to order 3. Central moments $\mu_{pq}$ are translation-invariant; dividing by $\mu_{00}^\gamma$ achieves scale invariance; specific nonlinear combinations of these yield rotation invariance. For example, $\phi_1 = \eta_{20} + \eta_{02}$ (sum of second-order moments) is unchanged under any rigid transformation. In practice, the log-transformed values $-\text{sign}(\phi_i)\log_{10}|\phi_i|$ are used to handle their wide dynamic range, and they serve as a compact, global shape fingerprint in EcoSort.

**Syllabus Unit:** Unit 3 — Moment Invariants  
**Code Reference:** `cv_pipeline.py → compute_hu_moments()` — `cv2.HuMoments(cv2.moments(contour))`

---

## Q12: What are Fourier Descriptors?

**Answer:** Fourier Descriptors represent a 2D contour as a sequence of complex numbers $z_k = x_k + jy_k$ and take the Discrete Fourier Transform: $Z_n = \frac{1}{N}\sum_{k=0}^{N-1} z_k e^{-j2\pi nk/N}$. The magnitudes $|Z_n|$ are rotation-invariant; normalizing by $|Z_1|$ gives scale invariance; and starting from a fixed point gives translation invariance. Truncating to the first $d$ coefficients yields a compact, low-frequency shape signature — higher coefficients encode fine boundary detail. In EcoSort, the first 20 normalized Fourier descriptors distinguish smooth bottle silhouettes from jagged paper shapes.

**Syllabus Unit:** Unit 3 — Shape Representation  
**Code Reference:** `cv_pipeline.py → compute_fourier_descriptors()` — `np.fft.fft` on complex contour array

---

## Q13: What is the Freeman Chain Code?

**Answer:** The Freeman Chain Code encodes a digital contour as a sequence of direction values $d \in \{0,1,2,3,4,5,6,7\}$, where each value represents one of eight compass directions (0 = East, increasing counter-clockwise by 45°). The first-difference (derivative) of the chain code is rotation-invariant. The **direction histogram entropy** $H = -\sum_d p_d \log_2 p_d$ measures shape complexity — low entropy for simple shapes (mostly one direction), high entropy for irregular shapes. In EcoSort, high chain code entropy characterizes crumpled bags and newspaper, helping route them to general waste or paper bins.

**Syllabus Unit:** Unit 3 — Shape Representation  
**Code Reference:** `cv_pipeline.py → compute_chain_code()` — custom 8-connectivity direction encoding

---

## Q14: Explain the Hough Line Transform mathematically.

**Answer:** The Hough Line Transform maps each edge point $(x_0, y_0)$ into a sinusoidal curve in $(\rho, \theta)$ parameter space via $\rho = x_0\cos\theta + y_0\sin\theta$, where $\rho$ is the perpendicular distance from the origin and $\theta \in [0°, 180°)$ is the line angle. Points lying on the same line produce sinusoids that intersect at a common $(\rho^*, \theta^*)$ in the accumulator. The cell with the maximum vote count identifies the dominant line. In EcoSort, many detected Hough lines indicate structured objects like cardboard boxes with straight edges, supporting paper bin classification.

**Syllabus Unit:** Unit 4 — Hough Transform  
**Code Reference:** `cv_pipeline.py → detect_hough_lines()` — `cv2.HoughLinesP(edges, 1, π/180, threshold=50)`

---

## Q15: Explain the Hough Circle Transform.

**Answer:** For circles, the parameter space is three-dimensional: centre $(a, b)$ and radius $r$. Each edge point $(x_0, y_0)$ votes for all circles satisfying $(x_0 - a)^2 + (y_0 - b)^2 = r^2$. OpenCV's `HOUGH_GRADIENT` method uses the gradient direction of each edge point to reduce the accumulator to 2D (for fixed $r$), then searches across a radius range. A peak in the accumulator at $(a^*, b^*, r^*)$ signifies a detected circle. In EcoSort, the presence of one or more Hough circles is a strong indicator of a cylindrical metal can or bottle cap, and is combined with the specular ratio to confirm metal.

**Syllabus Unit:** Unit 4 — Hough Transform  
**Code Reference:** `cv_pipeline.py → detect_hough_circles()` — `cv2.HoughCircles(..., cv2.HOUGH_GRADIENT)`

---

## Q16: What is Shape-from-Shading? How is specular ratio computed?

**Answer:** Shape-from-Shading (SfS) infers 3D surface normals from 2D intensity gradients under the assumption of a known reflectance model (e.g., Lambertian). While EcoSort uses a simplified proxy rather than full SfS, it exploits the fact that **specular highlights** — bright spots caused by mirror-like (specular) reflection — are distinctive of metallic and glossy surfaces. The specular ratio is computed as the fraction of pixels whose HSV Value channel exceeds a threshold (V > 200): $\text{SR} = |{p : V(p) > 200}| / N_{\text{total}}$. High SR (> 0.15) strongly predicts metal classification.

**Syllabus Unit:** Unit 5 — Photometric Analysis  
**Code Reference:** `cv_pipeline.py → analyze_specular_shading()` — HSV V-channel and S-channel thresholding

---

## Q17: What is Chamfer Distance? How is it used for template matching?

**Answer:** Chamfer distance measures the alignment between two edge maps by computing, for each template edge point $t$, its distance to the nearest query edge using the precomputed **distance transform** $D_Q$ of the query: $d_{\text{Chamfer}}(Q,T) = \frac{1}{|T|}\sum_{t \in T} D_Q(t)$. A small Chamfer distance means template edges closely overlap query edges. In EcoSort, reference silhouettes (can, bottle, box, bag) are matched against the object's Canny edge map; the silhouette with the lowest Chamfer distance contributes a shape-match score to the classification rule. This is a classical, learning-free form of template matching robust to small deformations.

**Syllabus Unit:** Unit 5 — Template Matching  
**Code Reference:** `cv_pipeline.py → chamfer_matching()` — `cv2.distanceTransform` + symmetric Chamfer distance against templates

---

## Q18: How does your classifier decide if something is a metal can vs. plastic bottle?

**Answer:** The classifier applies a prioritized rule hierarchy. For **metal**, it checks: circularity > 0.75 AND specular\_ratio > 0.15 AND (Hough circles ≥ 1 OR Chamfer\_can < 30 px). For **plastic**, it checks: solidity > 0.85 AND mean\_saturation < 60 AND corner\_count < 20 AND NOT metal. The key discriminators are specular ratio (metal surfaces reflect; plastic bottles are translucent/matte) and Hough circle detection (cylindrical cans produce strong circle evidence; bottles have an irregular profile above the label). If a bottle has a metallic cap that triggers specular detection, the Chamfer score to the bottle template overrides.

**Syllabus Unit:** Unit 5 (Classification Logic)  
**Code Reference:** `classifier.py → classify_waste_object(contour_features, scene_results)`

---

## Q19: Why did you NOT use deep learning for this project?

**Answer:** The 21CSE454T course specifically constrains the project to **classical computer vision techniques** covered across Units 1–5, ensuring students deeply understand the mathematical foundations — Fourier analysis, morphological theory, moment invariants — rather than treating them as black boxes. Deep learning models require labeled training datasets, GPU hardware, and are not interpretable at the feature level, making them unsuitable for demonstrating CV concepts. Furthermore, the project shows that for controlled industrial settings (conveyor belts with known object types), classical pipelines can achieve competitive performance with far lower computational cost and complete algorithmic transparency.

**Syllabus Unit:** All Units (Conceptual Justification)  
**Code Reference:** N/A (design rationale)

---

## Q20: What is the difference between Otsu's and Adaptive thresholding?

**Answer:** **Otsu's thresholding** computes a single optimal threshold for the entire image by maximizing between-class variance — it works well when the image histogram is bimodal and illumination is uniform. **Adaptive thresholding** computes a different threshold for each pixel based on the local neighbourhood statistics (mean or Gaussian-weighted mean minus a constant $C$), making it robust to spatially varying illumination. In EcoSort, both are applied and their binary masks fused via bitwise OR: Otsu handles well-lit objects against plain backgrounds, while adaptive thresholding recovers object regions in shadowed or gradient-illuminated areas.

**Syllabus Unit:** Unit 2 — Segmentation  
**Code Reference:** `cv_pipeline.py → segment_object()` — `cv2.THRESH_OTSU` and `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`

---

## Q21: What are Fourier Descriptors and how are they computed from contours?

**Answer:** Fourier Descriptors encode a 2D contour as a 1D complex signal $z_k = x_k + jy_k$ for $k = 0 \ldots N-1$ contour points, then apply the DFT to obtain frequency-domain coefficients $Z_n$. The magnitudes $|Z_n|$ form a shape spectrum; low-frequency components capture global shape (overall roundness), while high-frequency components encode fine detail (notches, ridges). Normalization — dividing all $|Z_n|$ by $|Z_1|$ — achieves scale invariance. In EcoSort, only the first 20 coefficients are retained, giving a compact 20-element descriptor that is compared between object classes.

**Syllabus Unit:** Unit 3 — Shape in Frequency Domain  
**Code Reference:** `cv_pipeline.py → compute_fourier_descriptors()` — `np.fft.fft(z_complex)[:20]`

---

## Q22: What is the Centroidal Profile of a shape?

**Answer:** The centroidal (radial) distance profile is a 1D shape signature computed by measuring the Euclidean distance from the shape's centroid $(c_x, c_y)$ to each boundary point as a function of angle $\theta$: $r(\theta) = \sqrt{(x(\theta)-c_x)^2 + (y(\theta)-c_y)^2}$. This converts the 2D contour into a 1D signal that can be analyzed using standard signal processing (Fourier, statistics). For a perfect circle, $r(\theta)$ is constant; for a bottle, it shows a periodic pattern corresponding to the shoulder and base. In EcoSort, the standard deviation of $r(\theta)$ measures shape irregularity and supplements the circularity metric.

**Syllabus Unit:** Unit 3 — Shape Representation  
**Code Reference:** `cv_pipeline.py → extract_contour_features()` — distance from centroid to each contour point

---

## Q23: How does background subtraction work with MOG2?

**Answer:** MOG2 (Mixture of Gaussians v2) models each pixel's intensity history as a mixture of Gaussian distributions, adaptively updating the mixture parameters over time. Pixels whose current values are poorly explained by the background model are classified as foreground. MOG2 handles gradual illumination changes (slow adaptation) and sudden background changes using a shadow detection mechanism that identifies darker foreground pixels. In a conveyor belt extension of EcoSort, MOG2 would continuously update the belt texture model, cleanly isolating each new waste item as it moves into frame, eliminating the need for manual background removal.

**Syllabus Unit:** Unit 2 — Segmentation (Video)  
**Code Reference:** `cv2.createBackgroundSubtractorMOG2()` (referenced in `app.py` for video/webcam mode)

---

## Q24: What is the role of the Sobel operator vs. Canny in your pipeline?

**Answer:** The **Sobel operator** is a simple 3×3 gradient filter that computes first-order image derivatives ($G_x, G_y$) to estimate local edge strength and direction; it produces a continuous gradient magnitude map, is fast, but sensitive to noise and produces thick edges. **Canny** builds on Sobel (using it internally for gradient computation) but adds Gaussian pre-smoothing for noise suppression, NMS for edge thinning, and hysteresis thresholding for connectivity. In EcoSort, Canny is used to generate the binary edge map for Hough transforms and Chamfer matching, while the raw Sobel response is used within the Harris corner detector to build the structure tensor.

**Syllabus Unit:** Unit 4 — Edge and Corner Detection  
**Code Reference:** `cv_pipeline.py → detect_edges()` (Canny); `detect_harris_corners()` uses `cv2.Sobel` internally

---

## Q25: How would you extend this project to a real-time conveyor belt sorting system?

**Answer:** A production extension would involve: (1) **Hardware**: a GigE camera mounted above the belt, triggered by a photoelectric sensor at 30–60 fps; (2) **Segmentation**: replace static thresholding with MOG2 background subtraction to handle moving belt texture; (3) **Speed**: port the morphological and contour pipeline to C++ OpenCV or FPGA (VHDL/HLS) for sub-10 ms per-frame latency; (4) **Actuation**: connect the classifier output to pneumatic air-jet sorters or robotic arm endpoints via PLC digital I/O; (5) **Learning Extension**: collect labeled images from the belt over several weeks and fine-tune a MobileNet-v3 classifier, using EcoSort's classical features as a fallback for edge cases. The interpretable rule layer would remain for regulatory audit trails.

**Syllabus Unit:** All Units — System Integration  
**Code Reference:** `app.py → live camera input mode`

---

## Quick Formula Reference Sheet

### Segmentation

$$\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,[\mu_0(t) - \mu_1(t)]^2 \quad \text{(Otsu between-class variance)}$$

$$T(x,y) = \frac{\sum_{(i,j)\in\mathcal{N}} w(i,j)\,I(i,j)}{\sum w(i,j)} - C \quad \text{(Adaptive threshold)}$$

### Edge & Corner Detection

$$M(x,y) = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\!\left(\frac{G_y}{G_x}\right) \quad \text{(Canny gradient)}$$

$$R = \det(M) - k\cdot\operatorname{trace}(M)^2, \quad k \approx 0.04 \quad \text{(Harris response)}$$

### Shape Descriptors

$$C = \frac{4\pi A}{P^2} \quad \text{(Circularity)}$$

$$S = \frac{A}{A_{\text{convex}}} \quad \text{(Solidity)}$$

$$\phi_1 = \eta_{20} + \eta_{02} \quad \text{(First Hu Moment)}$$

$$\phi_2 = (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2 \quad \text{(Second Hu Moment)}$$

$$Z_n = \frac{1}{N}\sum_{k=0}^{N-1}(x_k+jy_k)\,e^{-j2\pi nk/N} \quad \text{(Fourier Descriptors)}$$

$$H = -\sum_{d=0}^{7} p_d \log_2 p_d \quad \text{(Chain Code Entropy)}$$

### Hough Transforms

$$\rho = x\cos\theta + y\sin\theta \quad \text{(Hough Line)}$$

$$(x-a)^2 + (y-b)^2 = r^2 \quad \text{(Hough Circle)}$$

### Texture (GLCM)

$$\text{Contrast} = \sum_{i,j}(i-j)^2 P(i,j)$$

$$\text{Homogeneity} = \sum_{i,j}\frac{P(i,j)}{1+|i-j|}$$

$$\text{Energy} = \sum_{i,j} P(i,j)^2$$

$$\text{Correlation} = \sum_{i,j}\frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i\sigma_j}$$

### Template Matching

$$d_{\text{Chamfer}}(Q,T) = \frac{1}{|T|}\sum_{t \in T} D_Q(t) \quad \text{(Chamfer Distance)}$$

### Specular Ratio

$$\text{SR} = \frac{|\{p : V(p) > 200\}|}{N_{\text{total}}} \quad \text{(Specular pixel fraction in HSV)}$$

### Distance Transform

$$D(x,y) = \min_{(x',y')\in\text{BG}}\sqrt{(x-x')^2+(y-y')^2} \quad \text{(Euclidean DT)}$$

---

> [!TIP]
> **Viva Strategy**: For every answer, begin with the intuition ("what problem does it solve"), give the formula, then say "in our code, this is implemented in `function_name()`". Examiners reward structured, code-grounded answers over purely theoretical ones.

> [!NOTE]
> **Likely Follow-up Questions**: After Q1 (Otsu), expect "what if the image has multiple modes?" (answer: multi-level Otsu / Kapur entropy). After Q3 (Canny), expect "what do you choose for $T_{low}$ and $T_{high}$?" (answer: 2:1 or 3:1 ratio, typically 50 and 150). After Q18 (classifier logic), expect "what is your accuracy?" (answer: heuristically calibrated, no formal dataset — acknowledge limitation, cite future work with TrashNet).
