"""
Script 2: Extract images from Kaggle datasets
- khalidboussaroual/2d-geometric-shapes-17-shapes → circle, square, triangle, rectangle (300 each)
- dataclusterlabs/bottles-and-cups-dataset → cup (300)

Uses kagglehub to download datasets. Requires Kaggle API credentials.
If kagglehub is not set up, you can manually download and extract the datasets
to the paths printed by this script.
"""

import os
import sys
import shutil
import glob
from PIL import Image

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "original")
IMAGES_PER_CLASS = 300


def save_images_from_folder(src_folder, class_name, count, extensions=("*.jpg", "*.jpeg", "*.png", "*.bmp")):
    """Copy `count` images from src_folder into dataset/original/<class_name>/"""
    out_dir = os.path.join(BASE_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    # Gather all image files
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(src_folder, ext)))
        image_files.extend(glob.glob(os.path.join(src_folder, "**", ext), recursive=True))

    # Remove duplicates and sort
    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        print(f"  [FAIL] {class_name}: NO images found in {src_folder}")
        return 0

    actual_count = min(count, len(image_files))
    if actual_count < count:
        print(f"  [WARN] {class_name}: only {len(image_files)} images available (requested {count})")

    saved = 0
    for img_path in image_files[:actual_count]:
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            filepath = os.path.join(out_dir, f"{saved:05d}.jpg")
            img.save(filepath, "JPEG", quality=95)
            saved += 1
        except Exception as e:
            print(f"  [WARN] Skipping {img_path}: {e}")
            continue

    print(f"  [OK] {class_name}: saved {saved} images to {out_dir}")
    return saved


def extract_geometric_shapes():
    """Extract circle, square, triangle, rectangle from 2D geometric shapes dataset"""
    print("\n[1/2] Downloading 2D Geometric Shapes dataset from Kaggle ...")

    try:
        import kagglehub
        path = kagglehub.dataset_download("khalidboussaroual/2d-geometric-shapes-17-shapes")
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"  [FAIL] kagglehub download failed: {e}")
        print("  Attempting alternative: opendatasets ...")
        try:
            import opendatasets as od
            od.download("https://www.kaggle.com/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes",
                        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_kaggle_cache"))
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "_kaggle_cache", "2d-geometric-shapes-17-shapes")
        except Exception as e2:
            print(f"  [FAIL] opendatasets also failed: {e2}")
            print("  Please download manually from:")
            print("    https://www.kaggle.com/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes")
            print(f"  Extract to: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_kaggle_cache', '2d-geometric-shapes-17-shapes')}")
            return

    # The dataset has folders named by shape
    target_shapes = ["circle", "square", "triangle", "rectangle"]

    # Search for shape folders (case-insensitive)
    for shape in target_shapes:
        # Try common folder patterns
        candidates = [
            os.path.join(path, shape),
            os.path.join(path, shape.capitalize()),
            os.path.join(path, shape.upper()),
        ]
        
        # Also search recursively
        for root, dirs, files in os.walk(path):
            for d in dirs:
                if d.lower() == shape.lower():
                    candidates.append(os.path.join(root, d))

        found = False
        for candidate in candidates:
            if os.path.isdir(candidate):
                print(f"  Found {shape} folder: {candidate}")
                save_images_from_folder(candidate, shape, IMAGES_PER_CLASS)
                found = True
                break

        if not found:
            print(f"  [FAIL] Could not find folder for '{shape}' in {path}")
            # List available folders for debugging
            print(f"    Available folders: {os.listdir(path) if os.path.isdir(path) else 'N/A'}")


def extract_cups():
    """Extract cup images from bottles-and-cups-dataset"""
    print("\n[2/2] Downloading Bottles and Cups dataset from Kaggle ...")

    try:
        import kagglehub
        path = kagglehub.dataset_download("dataclusterlabs/bottles-and-cups-dataset")
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"  [FAIL] kagglehub download failed: {e}")
        try:
            import opendatasets as od
            od.download("https://www.kaggle.com/datasets/dataclusterlabs/bottles-and-cups-dataset",
                        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_kaggle_cache"))
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "_kaggle_cache", "bottles-and-cups-dataset")
        except Exception as e2:
            print(f"  [FAIL] opendatasets also failed: {e2}")
            print("  Please download manually from:")
            print("    https://www.kaggle.com/datasets/dataclusterlabs/bottles-and-cups-dataset")
            return

    # Search for cups/cup folder
    cup_candidates = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if "cup" in d.lower():
                cup_candidates.append(os.path.join(root, d))

    if cup_candidates:
        # Use the folder with the most images
        best_folder = max(cup_candidates, key=lambda f: len(os.listdir(f)))
        print(f"  Found cup folder: {best_folder} ({len(os.listdir(best_folder))} files)")
        save_images_from_folder(best_folder, "cup", IMAGES_PER_CLASS)
    else:
        # If no dedicated cup folder, look for images with 'cup' in name
        print(f"  No dedicated cup folder found. Searching for cup images...")
        cup_images = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if "cup" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    cup_images.append(os.path.join(root, f))

        if cup_images:
            # Save from list
            out_dir = os.path.join(BASE_DIR, "cup")
            os.makedirs(out_dir, exist_ok=True)
            saved = 0
            for img_path in cup_images[:IMAGES_PER_CLASS]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.save(os.path.join(out_dir, f"{saved:05d}.jpg"), "JPEG", quality=95)
                    saved += 1
                except Exception:
                    continue
            print(f"  [OK] cup: saved {saved} images to {out_dir}")
        else:
            # Just take images from whatever folder structure exists
            print(f"  Listing contents of {path}:")
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"    [DIR] {item} ({len(os.listdir(item_path))} items)")
                else:
                    print(f"    [FILE] {item}")
            print("  [WARN] Please check the folder structure and update the script.")


def main():
    print("=" * 60)
    print("Kaggle Dataset Extraction")
    print(f"Target: {IMAGES_PER_CLASS} images per class")
    print(f"Output: {BASE_DIR}")
    print("=" * 60)

    extract_geometric_shapes()
    extract_cups()

    print("\n" + "=" * 60)
    print("Kaggle extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()