"""
Script 3: Generate image variants from original images
Variants: outline, dotted, dashed, sketch, silhouette, color_tint_red, color_tint_green, color_tint_blue
Structural variants feature a clean WHITE background.
Color tint variants preserve the original image but shift color channels.

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

VARIANT_NAMES = ["outline", "dotted", "dashed", "sketch", "silhouette",
                 "color_tint_red", "color_tint_green", "color_tint_blue"]


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


def extract_foreground_mask(img):
    """
    Extract a soft alpha mask (0.0 for background, 1.0 for main object)
    using Otsu thresholding, border analysis, and morphological closing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border_pixels) > 127:
        fg_mask = (thresh == 0).astype(np.uint8) * 255
    else:
        fg_mask = (thresh == 255).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_float = cv2.GaussianBlur(fg_mask, (7, 7), 0).astype(np.float32) / 255.0
    return np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)


def recolor_foreground_object(img, color_name, mask_3d=None):
    """
    Changes ONLY the color of the main object while preserving 100% of the original background.
    Uses HSV color transformation on the foreground object.
    """
    if mask_3d is None:
        mask_3d = extract_foreground_mask(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Target hues in OpenCV (0-180): Red: 0, Green: 60, Blue: 120
    if color_name == "red":
        hsv[:, :, 0] = 0.0
    elif color_name == "green":
        hsv[:, :, 0] = 60.0
    elif color_name == "blue":
        hsv[:, :, 0] = 120.0

    # Boost saturation on object so color is prominent, while keeping original Value (shading & details)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5 + 80, 0, 255)
    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    orig_bgr = img.astype(np.float32)

    # Foreground gets recolored, background keeps original pixels exactly
    blended = orig_bgr * (1.0 - mask_3d) + recolored_bgr * mask_3d
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_color_tint_red(img, mask_3d=None):
    return recolor_foreground_object(img, "red", mask_3d)


def generate_color_tint_green(img, mask_3d=None):
    return recolor_foreground_object(img, "green", mask_3d)


def generate_color_tint_blue(img, mask_3d=None):
    return recolor_foreground_object(img, "blue", mask_3d)


# ─────────────────────────────────────────────────────────────
# Worker function to process a single image for all variants
# ─────────────────────────────────────────────────────────────

def process_single_image(args):
    """
    Worker task: reads original image once, generates missing variants,
    and saves each variant into its respective directory.
    """
    img_path, class_name, variants_dir, overwrite_colors = args
    filename = os.path.basename(img_path)

    missing = []
    for v in VARIANT_NAMES:
        out_p = os.path.join(variants_dir, v, class_name, filename)
        if overwrite_colors and v in {"color_tint_red", "color_tint_green", "color_tint_blue"}:
            missing.append(v)
        elif not os.path.exists(out_p):
            missing.append(v)

    if not missing:
        return class_name, filename, True, None

    img = cv2.imread(img_path)
    if img is None:
        return class_name, filename, False, "Failed to read image"

    try:
        color_variants = {"color_tint_red", "color_tint_green", "color_tint_blue"}
        need_mask = any(v in missing for v in color_variants)
        mask_3d = extract_foreground_mask(img) if need_mask else None

        for v_name in missing:
            if v_name == "outline":
                v_img = generate_outline(img)
            elif v_name == "dotted":
                v_img = generate_dotted(img)
            elif v_name == "dashed":
                v_img = generate_dashed(img)
            elif v_name == "sketch":
                v_img = generate_sketch(img)
            elif v_name == "silhouette":
                v_img = generate_silhouette(img)
            elif v_name == "color_tint_red":
                v_img = generate_color_tint_red(img, mask_3d)
            elif v_name == "color_tint_green":
                v_img = generate_color_tint_green(img, mask_3d)
            elif v_name == "color_tint_blue":
                v_img = generate_color_tint_blue(img, mask_3d)
            else:
                continue

            out_path = os.path.join(variants_dir, v_name, class_name, filename)
            cv2.imwrite(out_path, v_img)

        return class_name, filename, True, None
    except Exception as e:
        return class_name, filename, False, str(e)


def process_class(class_name, max_workers=None, overwrite_colors=False):
    """Generate all variants for all images in a single class using multiprocessing."""
    src_dir = os.path.join(ORIGINAL_DIR, class_name)
    image_files = sorted(glob(os.path.join(src_dir, "*.jpg")))

    if not image_files:
        print(f"  [WARN] No images found for class '{class_name}' in {src_dir}")
        return

    # Ensure output directories exist
    for v in VARIANT_NAMES:
        os.makedirs(os.path.join(VARIANTS_DIR, v, class_name), exist_ok=True)

    tasks = [(p, class_name, VARIANTS_DIR, overwrite_colors) for p in image_files]

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

    print(f"  [OK] '{class_name}': {success}/{len(image_files)} images processed" +
          (f" ({errors} errors)" if errors else ""))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate structural and color tint variants")
    parser.add_argument("--overwrite-colors", action="store_true", default=False,
                        help="Force re-generation of color tint variants with foreground-only recoloring")
    args = parser.parse_args()

    print("=" * 60)
    print("Variant Generation (Structural + Foreground Color Tint)")
    print(f"Source: {ORIGINAL_DIR}")
    print(f"Output: {VARIANTS_DIR}")
    print(f"Variants: {', '.join(VARIANT_NAMES)}")
    if args.overwrite_colors:
        print("Mode: Overwriting foreground color tints (preserving background)")
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
        process_class(class_name, max_workers=workers, overwrite_colors=args.overwrite_colors)
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
    print("[OK] All variants generated successfully!")


if __name__ == "__main__":
    main()

