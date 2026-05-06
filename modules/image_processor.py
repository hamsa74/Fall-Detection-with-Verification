import cv2
import numpy as np
import os
from datetime import datetime


def process_fall_image(frame, output_dir='output'):
    """
    Takes a fall evidence frame and saves multiple processed versions:
    1. Original (color)
    2. Grayscale
    3. Edge Detection (Canny)
    4. Enhanced (Brightness + Contrast)
    5. Combined (2x2 grid of all versions)

    Returns dict of all saved paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = {}

    # ── 1. Original ──────────────────────────────────────────
    orig_path = os.path.join(output_dir, f"FALL_{ts}_1_original.jpg")
    cv2.imwrite(orig_path, frame)
    paths['original'] = orig_path

    # ── 2. Grayscale ─────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, f"FALL_{ts}_2_grayscale.jpg")
    cv2.imwrite(gray_path, gray)
    paths['grayscale'] = gray_path

    # ── 3. Edge Detection (Canny) ────────────────────────────
    # Blur first to reduce noise, then detect edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, threshold1=50, threshold2=150)
    # Convert back to BGR for consistent saving
    edges_bgr  = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_path = os.path.join(output_dir, f"FALL_{ts}_3_edges.jpg")
    cv2.imwrite(edges_path, edges_bgr)
    paths['edges'] = edges_path

    # ── 4. Enhanced (Brightness + Contrast + Denoise) ────────
    # alpha = contrast (1.0-3.0), beta = brightness (0-100)
    enhanced = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    enh_path = os.path.join(output_dir, f"FALL_{ts}_4_enhanced.jpg")
    cv2.imwrite(enh_path, enhanced)
    paths['enhanced'] = enh_path

    # ── 5. Combined 2x2 Grid ─────────────────────────────────
    h, w  = frame.shape[:2]
    half_h, half_w = h // 2, w // 2

    # Resize all to same size
    def resize(img, color=True):
        r = cv2.resize(img, (half_w, half_h))
        return r if color else cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    orig_s  = resize(frame)
    gray_s  = resize(gray,     color=False)
    edges_s = resize(edges,    color=False)
    enh_s   = resize(enhanced)

    # Add labels
    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (len(text)*9+10, 24), (0,0,0), -1)
        cv2.putText(img, text, (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return img

    orig_s  = label(orig_s,  "Original")
    gray_s  = label(gray_s,  "Grayscale")
    edges_s = label(edges_s, "Edge Detection")
    enh_s   = label(enh_s,   "Enhanced")

    # Combine into 2x2
    top    = np.hstack([orig_s, gray_s])
    bottom = np.hstack([edges_s, enh_s])
    grid   = np.vstack([top, bottom])

    # Add title bar
    title_bar = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, f"CareBot AI - Fall Evidence | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (249,115,22), 1)
    grid = np.vstack([title_bar, grid])

    grid_path = os.path.join(output_dir, f"FALL_{ts}_5_combined.jpg")
    cv2.imwrite(grid_path, grid)
    paths['combined'] = grid_path

    print(f"[Processor] ✅ Saved {len(paths)} versions for fall at {ts}")
    for k, v in paths.items():
        print(f"            {k:12s} → {os.path.basename(v)}")

    return paths