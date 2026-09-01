"""
Script 1: Extract images from HuggingFace datasets
- microsoft/cats_vs_dogs → cat (300), dog (300)
- filnow/furniture-synthetic-dataset → chair (300), table (300)
- HumynLabs/car-images → car (300)
"""

import os
import sys
from datasets import load_dataset

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "original")
IMAGES_PER_CLASS = 300


def save_images(dataset_iter, class_name, count):
    """Save `count` images from a dataset iterator into dataset/original/<class_name>/"""
    out_dir = os.path.join(BASE_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for item in dataset_iter:
        if saved >= count:
            break
        img = item["image"]
        # Convert to RGB if necessary (some images may be RGBA or grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")
        filepath = os.path.join(out_dir, f"{saved:05d}.jpg")
        img.save(filepath, "JPEG", quality=95)
        saved += 1

    print(f"  [OK] {class_name}: saved {saved} images to {out_dir}")
    return saved


def extract_cats_vs_dogs():
    """Extract cat and dog images from microsoft/cats_vs_dogs"""
    print("\n[1/3] Loading microsoft/cats_vs_dogs ...")
    dataset = load_dataset("microsoft/cats_vs_dogs", split="train")
    print(f"  Total images: {len(dataset)}")

    # labels: 0 = cat, 1 = dog
    cats = dataset.filter(lambda x: x["labels"] == 0)
    dogs = dataset.filter(lambda x: x["labels"] == 1)

    print(f"  Cats available: {len(cats)}, Dogs available: {len(dogs)}")

    save_images(cats, "cat", IMAGES_PER_CLASS)
    save_images(dogs, "dog", IMAGES_PER_CLASS)


def extract_furniture():
    """Extract chair and table images from filnow/furniture-synthetic-dataset"""
    print("\n[2/3] Loading filnow/furniture-synthetic-dataset ...")
    dataset = load_dataset("filnow/furniture-synthetic-dataset")
    
    # Combine train and test splits
    train = dataset["train"]
    test = dataset["test"]
    print(f"  Train: {len(train)}, Test: {len(test)}")

    chairs = train.filter(lambda x: x["type"] == "chair")
    tables = train.filter(lambda x: x["type"] == "table")

    print(f"  Chairs available: {len(chairs)}, Tables available: {len(tables)}")

    save_images(chairs, "chair", IMAGES_PER_CLASS)
    save_images(tables, "table", IMAGES_PER_CLASS)


def extract_cars():
    """Extract car images from HumynLabs/car-images"""
    print("\n[3/3] Loading HumynLabs/car-images ...")
    dataset = load_dataset("HumynLabs/car-images", split="train")
    print(f"  Total car images available: {len(dataset)}")

    count = min(IMAGES_PER_CLASS, len(dataset))
    if count < IMAGES_PER_CLASS:
        print(f"  [WARN] Only {count} images available (requested {IMAGES_PER_CLASS})")

    save_images(dataset, "car", count)


def main():
    print("=" * 60)
    print("HuggingFace Dataset Extraction")
    print(f"Target: {IMAGES_PER_CLASS} images per class")
    print(f"Output: {BASE_DIR}")
    print("=" * 60)

    extract_cats_vs_dogs()
    extract_furniture()
    extract_cars()

    print("\n" + "=" * 60)
    print("HuggingFace extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
