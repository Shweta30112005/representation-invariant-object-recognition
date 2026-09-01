"""
Script 3: Generate image variants from original images
Variants: outline, dotted, dashed, sketch, silhouette
ALL variants feature a clean WHITE background.

Reads from:  dataset/original/<class>/*.jpg
Writes to:   dataset/variants/<variant>/<class>/*.jpg
"""

import os
import sys
import cv2
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DIR = os.path.join(PROJECT_DIR, "dataset", "original")
VARIANTS_DIR = os.path.join(PROJECT_DIR, "dataset", "variants")

VARIANT_NAMES = ["outline", "dotted", "dashed", "sketch", "silhouette"]


# ─────────────────────────────────────────────────────────────
# Variant generation functions (All with WHITE background)
# ─────────────────────────────────────────────────────────────

def generate_outline(img):
    """
    Canny edge detection → black edges on white background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Invert: non-edges become 255 (white), edges become 0 (black)
    inv_edges = cv2.bitwise_not(edges)
    result = cv2.cvtColor(inv_edges, cv2.COLOR_GRAY2BGR)
    return result


def generate_dotted(img):
    """
    Extract contour points and draw small black filled circles (dots)
    along them on a white background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Create white canvas
    h, w = img.shape[:2]
    result = np.full((h, w, 3), 255, dtype=np.uint8)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Draw black dots along contours at regular intervals
    dot_spacing = 6  # pixels between dots
    dot_radius = 2

    for contour in contours:
        for i in range(0, len(contour), dot_spacing):
            x, y = contour[i][0]
            cv2.circle(result, (x, y), dot_radius, (0, 0, 0), -1)

    return result


def generate_dashed(img):
    """
    Extract contours and draw black dashed lines along them
    (alternating between drawing and skipping segments) on a white background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Create white canvas
    h, w = img.shape[:2]
    result = np.full((h, w, 3), 255, dtype=np.uint8)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    dash_length = 10  # pixels of drawn segment
    gap_length = 6    # pixels of gap

    for contour in contours:
        pts = contour.reshape(-1, 2)
        drawing = True
        segment_counter = 0

        for i in range(1, len(pts)):
            if drawing:
                cv2.line(result, tuple(pts[i - 1]), tuple(pts[i]), (0, 0, 0), 1)

            segment_counter += 1
            if drawing and segment_counter >= dash_length:
                drawing = False
                segment_counter = 0
            elif not drawing and segment_counter >= gap_length:
                drawing = True
                segment_counter = 0

    return result


def generate_sketch(img):
    """
    Pencil sketch effect using cv2.pencilSketch().
    Returns the grayscale sketch converted to 3 channels
    (dark pencil strokes on light/white paper background).
    """
    gray_sketch, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    result = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
    return result


def generate_silhouette(img):
    """
    Create a silhouette: solid black shape on white background.
    Uses adaptive/Otsu thresholding + morphological cleanup.
    Ensures background (corners/borders) is white (255) and subject is black (0).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check corners to determine whether background was mapped to 255 or 0
    h, w = binary.shape
    corners = [binary[0, 0], binary[0, w - 1], binary[h - 1, 0], binary[h - 1, w - 1]]
    if np.mean(corners) < 128:
        # Background was mapped to 0 (black), invert so background is white (255)
        binary = cv2.bitwise_not(binary)

    # At this point, background is 255 (white) and subject is 0 (black)
    # We invert temporarily for morphological closing on the foreground object
    foreground = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_fg = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Re-invert back to black subject on white background
    silhouette = cv2.bitwise_not(closed_fg)
    result = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
    return result


# ─────────────────────────────────────────────────────────────
# Worker function to process a single image for all 5 variants
# ─────────────────────────────────────────────────────────────

def process_single_image(args):
    """
    Worker task: reads original image once, generates all 5 variants,
    and saves each variant into its respective directory.
    """
    img_path, class_name, variants_dir = args
    filename = os.path.basename(img_path)

    img = cv2.imread(img_path)
    if img is None:
        return class_name, filename, False, "Failed to read image"

    try:
        outline_img = generate_outline(img)
        dotted_img = generate_dotted(img)
        dashed_img = generate_dashed(img)
        sketch_img = generate_sketch(img)
        silhouette_img = generate_silhouette(img)

        variant_images = {
            "outline": outline_img,
            "dotted": dotted_img,
            "dashed": dashed_img,
            "sketch": sketch_img,
            "silhouette": silhouette_img,
        }

        for v_name, v_img in variant_images.items():
            out_path = os.path.join(variants_dir, v_name, class_name, filename)
            cv2.imwrite(out_path, v_img)

        return class_name, filename, True, None
    except Exception as e:
        return class_name, filename, False, str(e)


def process_class(class_name, max_workers=None):
    """Generate all 5 variants for all images in a single class using multiprocessing."""
    src_dir = os.path.join(ORIGINAL_DIR, class_name)
    image_files = sorted(glob(os.path.join(src_dir, "*.jpg")))

    if not image_files:
        print(f"  [WARN] No images found for class '{class_name}' in {src_dir}")
        return

    # Ensure output directories exist
    for v in VARIANT_NAMES:
        os.makedirs(os.path.join(VARIANTS_DIR, v, class_name), exist_ok=True)

    tasks = [(p, class_name, VARIANTS_DIR) for p in image_files]

    success = 0
    errors = 0

    # Use ProcessPoolExecutor for multi-core parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]
        for f in as_completed(futures):
            _, filename, ok, err = f.result()
            if ok:
                success += 1
            else:
                errors += 1
                if errors <= 3:
                    print(f"    [WARN] Error processing {filename}: {err}")

    print(f"  [OK] '{class_name}': {success}/{len(image_files)} images processed for all 5 variants" +
          (f" ({errors} errors)" if errors else ""))


def main():
    print("=" * 60)
    print("Variant Generation (White Background)")
    print(f"Source: {ORIGINAL_DIR}")
    print(f"Output: {VARIANTS_DIR}")
    print(f"Variants: {', '.join(VARIANT_NAMES)} (White Background)")
    print("=" * 60)

    # Validate directory
    if not os.path.isdir(ORIGINAL_DIR):
        print(f"[FAIL] Original directory not found: {ORIGINAL_DIR}")
        print("  Run extraction scripts first!")
        sys.exit(1)

    classes = sorted([
        d for d in os.listdir(ORIGINAL_DIR)
        if os.path.isdir(os.path.join(ORIGINAL_DIR, d))
    ])

    if not classes:
        print("[FAIL] No class folders found in original directory!")
        sys.exit(1)

    print(f"\nFound {len(classes)} classes: {', '.join(classes)}\n")

    workers = min(os.cpu_count() or 4, 8)
    print(f"Using {workers} parallel worker processes.\n")

    for i, class_name in enumerate(classes, 1):
        print(f"[{i}/{len(classes)}] Processing '{class_name}' ...")
        process_class(class_name, max_workers=workers)
        print()

    # Print summary table
    print("=" * 75)
    print("SUMMARY OF GENERATED VARIANTS")
    print("=" * 75)
    print(f"{'Class':<12} {'Original':>9}", end="")
    for v in VARIANT_NAMES:
        print(f" {v:>10}", end="")
    print()
    print("-" * 75)

    total_orig = 0
    total_variants = {v: 0 for v in VARIANT_NAMES}

    for class_name in classes:
        orig_count = len(glob(os.path.join(ORIGINAL_DIR, class_name, "*.jpg")))
        total_orig += orig_count
        print(f"{class_name:<12} {orig_count:>9}", end="")
        for v in VARIANT_NAMES:
            v_count = len(glob(os.path.join(VARIANTS_DIR, v, class_name, "*.jpg")))
            total_variants[v] += v_count
            print(f" {v_count:>10}", end="")
        print()

    print("-" * 75)
    print(f"{'TOTAL':<12} {total_orig:>9}", end="")
    for v in VARIANT_NAMES:
        print(f" {total_variants[v]:>10}", end="")
    print("\n" + "=" * 75)
    print("[OK] All variants generated with white background successfully!")


if __name__ == "__main__":
    main()

