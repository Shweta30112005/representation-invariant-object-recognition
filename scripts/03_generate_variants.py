"""
generate_variants.py

Representation-Invariant Object Recognition
--------------------------------------------

Automatically processes the complete dataset.

Input:
    dataset/original/<class>/*.jpg

Output:
    dataset/variants/
        ├── outline/<class>/
        ├── dotted/<class>/
        ├── dashed/<class>/
        ├── sketch/<class>/
        ├── silhouette/<class>/
        ├── color_tint_red/<class>/
        ├── color_tint_green/<class>/
        └── color_tint_blue/<class>/

Usage:
    python3 generate_variants.py

Requirements:
    pip install opencv-python numpy rembg pillow
"""

import os
import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from rembg import remove


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"

ORIGINAL_DIR = DATASET_DIR / "original"

VARIANTS_DIR = DATASET_DIR / "variants"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# REPRESENTATION PARAMETERS
# ============================================================

# ----------------------------
# Segmentation
# ----------------------------

ALPHA_THRESHOLD = 127

MIN_COMPONENT_AREA_RATIO = 0.001


# ----------------------------
# Internal structural edges
# ----------------------------

CANNY_LOW = 60
CANNY_HIGH = 150

MIN_EDGE_LENGTH = 20

MIN_EDGE_COMPONENT_AREA = 15

EDGE_CLOSE_KERNEL = 3

EDGE_CLOSE_ITERATIONS = 1


# ----------------------------
# Contour simplification
# ----------------------------

CONTOUR_EPSILON = 0.002


# ----------------------------
# Outline
# ----------------------------

OUTLINE_THICKNESS = 2


# ----------------------------
# Dotted
# ----------------------------

DOT_SPACING = 8

DOT_RADIUS = 2


# ----------------------------
# Dashed
# ----------------------------

DASH_LENGTH = 12

GAP_LENGTH = 8

DASH_THICKNESS = 2


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    """
    Load an image using OpenCV.
    """

    img = cv2.imread(
        str(path)
    )

    if img is None:

        raise ValueError(
            f"Could not read image: {path}"
        )

    return img


# ============================================================
# BACKGROUND SEGMENTATION
# ============================================================

def get_foreground_mask(img):
    """
    Remove the background using rembg.

    Returns:
        Binary mask

        255 = object
        0   = background
    """

    # BGR -> RGB
    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    pil_img = Image.fromarray(
        img_rgb
    )

    # Background removal
    result = remove(
        pil_img
    )

    result_np = np.array(
        result
    )

    # Alpha channel
    alpha = result_np[:, :, 3]

    # Binary mask
    _, mask = cv2.threshold(
        alpha,
        ALPHA_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Smooth segmentation
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Remove tiny components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    h, w = mask.shape

    min_area = (
        h *
        w *
        MIN_COMPONENT_AREA_RATIO
    )

    clean_mask = np.zeros_like(
        mask
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            clean_mask[
                labels == label
            ] = 255

    return clean_mask


# ============================================================
# OUTER CONTOURS
# ============================================================

def get_outer_contours(mask):
    """
    Extract the external boundary of the object.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    contours = [
        c
        for c in contours
        if cv2.contourArea(c) > 20
    ]

    return contours


# ============================================================
# INTERNAL EDGES
# ============================================================

def get_internal_edges(img, mask):
    """
    Extract meaningful internal structural edges.

    Edges are detected from the original image but
    restricted to the segmented foreground.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Smooth texture/noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.2
    )

    # Canny
    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    # --------------------------------------------------------
    # Restrict edges to object
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    interior_mask = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    edges = cv2.bitwise_and(
        edges,
        interior_mask
    )

    # --------------------------------------------------------
    # Connect small breaks
    # --------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            EDGE_CLOSE_KERNEL,
            EDGE_CLOSE_KERNEL
        )
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=EDGE_CLOSE_ITERATIONS
    )

    # --------------------------------------------------------
    # Remove small edge components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            edges,
            connectivity=8
        )
    )

    clean_edges = np.zeros_like(
        edges
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_EDGE_COMPONENT_AREA:

            clean_edges[
                labels == label
            ] = 255

    return clean_edges


# ============================================================
# INTERNAL CONTOURS
# ============================================================

def get_internal_contours(edges):
    """
    Convert internal edge map into clean contours.
    """

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    good_contours = []

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            False
        )

        if perimeter < MIN_EDGE_LENGTH:
            continue

        epsilon = (
            CONTOUR_EPSILON *
            perimeter
        )

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            False
        )

        if len(contour) >= 2:

            good_contours.append(
                contour
            )

    return good_contours


# ============================================================
# STRUCTURAL CONTOURS
# ============================================================

def get_structural_contours(
    img,
    mask
):
    """
    Combine:

        1. Outer object contours
        2. Internal structural contours
    """

    outer_contours = (
        get_outer_contours(
            mask
        )
    )

    internal_edges = (
        get_internal_edges(
            img,
            mask
        )
    )

    internal_contours = (
        get_internal_contours(
            internal_edges
        )
    )

    return (
        outer_contours,
        internal_contours
    )


# ============================================================
# OUTLINE
# ============================================================

def make_outline(
    img,
    outer_contours,
    internal_contours
):
    """
    Generate clean black structural
    outline on white background.
    """

    h, w = img.shape[:2]

    canvas = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8
    )

    # Outer boundary
    if outer_contours:

        cv2.drawContours(
            canvas,
            outer_contours,
            -1,
            (0, 0, 0),
            OUTLINE_THICKNESS,
            lineType=cv2.LINE_AA
        )

    # Internal structure
    for contour in internal_contours:

        cv2.polylines(
            canvas,
            [contour],
            False,
            (0, 0, 0),
            OUTLINE_THICKNESS,
            lineType=cv2.LINE_AA
        )

    return canvas


# ============================================================
# CONTOUR SAMPLING
# ============================================================

def sample_contour_points(
    contour,
    spacing
):
    """
    Sample points at approximately uniform
    arc-length intervals.
    """

    pts = contour.reshape(
        -1,
        2
    ).astype(
        np.float32
    )

    if len(pts) < 2:
        return []

    sampled = []

    accumulated = 0.0

    next_sample = 0.0

    for i in range(
        len(pts) - 1
    ):

        p1 = pts[i]

        p2 = pts[i + 1]

        segment = p2 - p1

        length = np.linalg.norm(
            segment
        )

        if length < 1e-6:
            continue

        while (
            next_sample
            <= accumulated + length
        ):

            t = (
                next_sample
                - accumulated
            ) / length

            t = np.clip(
                t,
                0.0,
                1.0
            )

            point = (
                p1 +
                t *
                segment
            )

            sampled.append(
                (
                    int(round(point[0])),
                    int(round(point[1]))
                )
            )

            next_sample += spacing

        accumulated += length

    return sampled


# ============================================================
# DOTTED
# ============================================================

def make_dotted(
    img,
    outer_contours,
    internal_contours
):
    """
    Generate a clean dotted structural
    representation.
    """

    h, w = img.shape[:2]

    canvas = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8
    )

    all_contours = (
        outer_contours +
        internal_contours
    )

    for contour in all_contours:

        points = sample_contour_points(
            contour,
            DOT_SPACING
        )

        for x, y in points:

            cv2.circle(
                canvas,
                (x, y),
                DOT_RADIUS,
                (0, 0, 0),
                -1,
                lineType=cv2.LINE_AA
            )

    return canvas


# ============================================================
# DASHED
# ============================================================

def make_dashed(
    img,
    outer_contours,
    internal_contours
):
    """
    Generate clean dashed structural
    representation.
    """

    h, w = img.shape[:2]

    canvas = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8
    )

    all_contours = (
        outer_contours +
        internal_contours
    )

    for contour in all_contours:

        pts = contour.reshape(
            -1,
            2
        ).astype(
            np.float32
        )

        if len(pts) < 2:
            continue

        for i in range(
            len(pts) - 1
        ):

            p1 = pts[i]

            p2 = pts[i + 1]

            segment = p2 - p1

            length = np.linalg.norm(
                segment
            )

            if length < 1e-6:
                continue

            position = 0.0

            draw = True

            while position < length:

                step = (
                    DASH_LENGTH
                    if draw
                    else GAP_LENGTH
                )

                end = min(
                    position + step,
                    length
                )

                if draw:

                    q1 = (
                        p1 +
                        segment *
                        (
                            position /
                            length
                        )
                    )

                    q2 = (
                        p1 +
                        segment *
                        (
                            end /
                            length
                        )
                    )

                    cv2.line(
                        canvas,
                        tuple(
                            np.round(q1)
                            .astype(int)
                        ),
                        tuple(
                            np.round(q2)
                            .astype(int)
                        ),
                        (0, 0, 0),
                        DASH_THICKNESS,
                        lineType=cv2.LINE_AA
                    )

                position = end

                draw = not draw

    return canvas


# ============================================================
# SKETCH
# ============================================================

def make_sketch(
    img,
    mask
):
    """
    Generate a clean pencil-like sketch
    while keeping the background white.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Remove background
    foreground = np.full_like(
        gray,
        255
    )

    foreground[
        mask > 0
    ] = gray[
        mask > 0
    ]

    # Invert
    inverted = (
        255 -
        foreground
    )

    # Blur
    blur = cv2.GaussianBlur(
        inverted,
        (21, 21),
        0
    )

    # Pencil sketch
    sketch = cv2.divide(
        foreground,
        255 - blur,
        scale=256
    )

    # Structural edges
    edges = cv2.Canny(
        foreground,
        50,
        150
    )

    edges = cv2.bitwise_and(
        edges,
        mask
    )

    sketch = cv2.addWeighted(
        sketch,
        0.85,
        255 - edges,
        0.15,
        0
    )

    sketch[
        mask == 0
    ] = 255

    return cv2.cvtColor(
        sketch,
        cv2.COLOR_GRAY2BGR
    )


# ============================================================
# COLOR TINTS
# ============================================================

def make_color_tint(img, mask, color):
    """
    Generate a true monochromatic color-shift representation.

    The original luminance/shape information is retained,
    while all object pixels are mapped into one color family.

    Background = pure white.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize luminance
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # --------------------------------------------------------
    # Create white canvas
    # --------------------------------------------------------

    output = np.full(
        img.shape,
        255,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Map grayscale → monochromatic color
    #
    # gray = 0   → strongest color
    # gray = 255 → white
    # --------------------------------------------------------

    if color == "red":

        output[:, :, 0] = gray
        output[:, :, 1] = gray
        output[:, :, 2] = 255

    elif color == "green":

        output[:, :, 0] = gray
        output[:, :, 1] = 255
        output[:, :, 2] = gray

    elif color == "blue":

        output[:, :, 0] = 255
        output[:, :, 1] = gray
        output[:, :, 2] = gray

    else:

        raise ValueError(
            f"Unknown color: {color}"
        )

    # --------------------------------------------------------
    # Background remains white
    # --------------------------------------------------------

    output[mask == 0] = 255

    return output




# ============================================================
# SILHOUETTE
# ============================================================

def make_silhouette(mask):
    """
    Generate clean black silhouette
    on white background.
    """

    h, w = mask.shape

    canvas = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8
    )

    canvas[
        mask > 0
    ] = (
        0,
        0,
        0
    )

    return canvas


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    class_name
):
    """
    Process one image and save all 8 variants.
    """

    print(
        f"\n  Processing: "
        f"{image_path.name}"
    )

    try:

        img = load_image(
            image_path
        )

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        mask = get_foreground_mask(
            img
        )

        foreground_pixels = np.sum(
            mask > 0
        )

        if foreground_pixels < 100:

            print(
                "    WARNING: segmentation failed"
            )

            return False

        # ----------------------------------------------------
        # Structural contours
        # ----------------------------------------------------

        (
            outer_contours,
            internal_contours
        ) = get_structural_contours(
            img,
            mask
        )

        print(
            f"    Outer contours: "
            f"{len(outer_contours)}"
        )

        print(
            f"    Internal contours: "
            f"{len(internal_contours)}"
        )

        # ----------------------------------------------------
        # Generate variants
        # ----------------------------------------------------

        variants = {

            "outline":
                make_outline(
                    img,
                    outer_contours,
                    internal_contours
                ),

            "dotted":
                make_dotted(
                    img,
                    outer_contours,
                    internal_contours
                ),

            "dashed":
                make_dashed(
                    img,
                    outer_contours,
                    internal_contours
                ),

            "sketch":
                make_sketch(
                    img,
                    mask
                ),

            "silhouette":
                make_silhouette(
                    mask
                ),

            "color_tint_red":
                make_color_tint(
                    img,
                    mask,
                    "red"
                ),

            "color_tint_green":
                make_color_tint(
                    img,
                    mask,
                    "green"
                ),

            "color_tint_blue":
                make_color_tint(
                    img,
                    mask,
                    "blue"
                )
        }

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        for variant_name, image in variants.items():

            output_dir = (
                VARIANTS_DIR /
                variant_name /
                class_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                output_dir /
                image_path.name
            )

            cv2.imwrite(
                str(output_path),
                image
            )

        print(
            "    ✓ Generated 8 representations"
        )

        return True

    except Exception as e:

        print(
            f"    ✗ ERROR: {e}"
        )

        return False


# ============================================================
# PROCESS ENTIRE DATASET
# ============================================================

def process_dataset():
    """
    Automatically process every class and
    every image under dataset/original/.
    """

    print("\n")
    print("=" * 70)
    print(
        "REPRESENTATION-INVARIANT OBJECT RECOGNITION"
    )
    print(
        "AUTOMATIC VARIANT GENERATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not ORIGINAL_DIR.exists():

        raise FileNotFoundError(
            f"\nOriginal dataset not found:\n"
            f"{ORIGINAL_DIR}\n\n"
            f"Expected structure:\n"
            f"dataset/original/<class>/images"
        )

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    class_dirs = sorted([
        d
        for d in ORIGINAL_DIR.iterdir()
        if d.is_dir()
    ])

    if not class_dirs:

        raise RuntimeError(
            f"No class folders found in:\n"
            f"{ORIGINAL_DIR}"
        )

    print(
        f"\nDataset directory:"
        f"\n  {DATASET_DIR}"
    )

    print(
        f"\nClasses found: "
        f"{len(class_dirs)}"
    )

    for class_dir in class_dirs:

        print(
            f"  • {class_dir.name}"
        )

    print("\n" + "-" * 70)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_images = 0

    successful_images = 0

    failed_images = 0

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for class_dir in class_dirs:

        class_name = class_dir.name

        image_files = sorted([
            f
            for f in class_dir.iterdir()
            if f.is_file()
            and f.suffix.lower()
            in IMAGE_EXTENSIONS
        ])

        print("\n")
        print(
            f"CLASS: {class_name}"
        )

        print(
            f"Images: {len(image_files)}"
        )

        if not image_files:

            print(
                "  No images found."
            )

            continue

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for image_path in image_files:

            total_images += 1

            success = process_image(
                image_path,
                class_name
            )

            if success:

                successful_images += 1

            else:

                failed_images += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Classes processed : "
        f"{len(class_dirs)}"
    )

    print(
        f"Images found      : "
        f"{total_images}"
    )

    print(
        f"Successful        : "
        f"{successful_images}"
    )

    print(
        f"Failed            : "
        f"{failed_images}"
    )

    print(
        f"Representations   : "
        f"8 variants/image"
    )

    print(
        f"Generated images  : "
        f"{successful_images * 8}"
    )

    print(
        f"\nOutput:"
    )

    print(
        f"  {VARIANTS_DIR}"
    )

    print("\nRepresentations:")

    print(
        "  1. outline"
    )

    print(
        "  2. dotted"
    )

    print(
        "  3. dashed"
    )

    print(
        "  4. sketch"
    )

    print(
        "  5. silhouette"
    )

    print(
        "  6. color_tint_red"
    )

    print(
        "  7. color_tint_green"
    )

    print(
        "  8. color_tint_blue"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()