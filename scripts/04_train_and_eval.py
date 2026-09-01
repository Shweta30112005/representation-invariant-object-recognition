"""
Script 4: Cross-Representation Training and Evaluation
Models:
  - ResNet-50 (Fine-tuned)
  - ViT-B/16 (Fine-tuned)
  - DINOv2 / DINOv3 (Linear probe / Classifier head)
  - CLIP ViT-B/32 (Zero-Shot)
  - EVA-CLIP (Zero-Shot)

Representations:
  - Original
  - Outline
  - Dotted
  - Dashed
  - Sketch
  - Silhouette

Generates full Cross-Representation Matrix:
  Train on variant X -> Test on ALL variants [Original, Outline, Dotted, Dashed, Sketch, Silhouette].
Saves all metrics to CSV and prints formatted accuracy matrices on terminal.
"""

import os
import sys
import argparse
import random
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ─────────────────────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR = PROJECT_DIR / "dataset" / "variants"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = ["Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette"]

REPRESENTATION_PATHS = {
    "Original": ORIGINAL_DIR,
    "Outline": VARIANTS_DIR / "outline",
    "Dotted": VARIANTS_DIR / "dotted",
    "Dashed": VARIANTS_DIR / "dashed",
    "Sketch": VARIANTS_DIR / "sketch",
    "Silhouette": VARIANTS_DIR / "silhouette",
}


# ─────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────

class RepresentationDataset(Dataset):
    """
    Dataset storing (image_path, label_idx).
    Loads images on-the-fly and applies the given transform.
    """
    def __init__(self, records, transform=None):
        self.records = records  # list of (image_path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, label = self.records[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────
# Data Splitting & Setup Builder
# ─────────────────────────────────────────────────────────────

def build_data_splits(classes, train_ratio=0.8, seed=42):
    """
    Splits images into train/test partitions consistently across representations.
    Ensures image index '00240.jpg' belongs to test in ALL representations for fair cross-testing.
    """
    random.seed(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Determine train/test file indices from Original directory
    class_train_files = {}
    class_test_files = {}

    for c in classes:
        orig_class_dir = ORIGINAL_DIR / c
        files = sorted([f.name for f in orig_class_dir.glob("*.jpg")])
        if not files:
            raise FileNotFoundError(f"No images found for class '{c}' in {orig_class_dir}")

        n_total = len(files)
        n_train = int(n_total * train_ratio)

        # Shuffle indices deterministically
        indices = list(range(n_total))
        random.shuffle(indices)
        train_indices = set(indices[:n_train])

        class_train_files[c] = [files[i] for i in range(n_total) if i in train_indices]
        class_test_files[c] = [files[i] for i in range(n_total) if i not in train_indices]

    # Build Training Sets for each representation
    training_setups = {}
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        records = []
        for c in classes:
            c_dir = base_dir / c
            for fname in class_train_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append((str(p), class_to_idx[c]))
        training_setups[f"Train_{rep}"] = records

    # Combined training setup (trained on all representations)
    all_combined_records = []
    for rep in REPRESENTATIONS:
        all_combined_records.extend(training_setups[f"Train_{rep}"])
    training_setups["Train_All_Combined"] = all_combined_records

    # Build Test Sets for each representation
    test_sets = {}
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        records = []
        for c in classes:
            c_dir = base_dir / c
            for fname in class_test_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append((str(p), class_to_idx[c]))
        test_sets[rep] = records

    return training_setups, test_sets, class_to_idx


# ─────────────────────────────────────────────────────────────
# Supervised Model Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_and_eval_model(model_name, classes, training_setups, test_sets, device, epochs=5, batch_size=32, lr=1e-4):
    num_classes = len(classes)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []

    print(f"\n{'='*70}")
    print(f"TRAINING AND EVALUATING MODEL: {model_name} ({num_classes} classes)")
    print(f"{'='*70}")

    for setup_name, train_records in training_setups.items():
        print(f"\n--- Model: {model_name} | Training Setup: {setup_name} ({len(train_records)} images) ---")

        # Instantiate fresh model for each training setup
        if model_name == "ResNet-50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "ViT-B/16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        elif model_name == "DINOv3" or model_name == "DINOv2":
            try:
                backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                embed_dim = 768
            except Exception as e:
                print(f"  [Notice] torch.hub dinov2 load failed ({e}), falling back to timm/vit_b_16")
                try:
                    import timm
                    backbone = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0)
                    embed_dim = 768
                except Exception:
                    base_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
                    backbone = base_vit
                    embed_dim = 768

            class DINOClassifier(nn.Module):
                def __init__(self, bb, dim, n_cls):
                    super().__init__()
                    self.bb = bb
                    for p in self.bb.parameters():
                        p.requires_grad = False
                    self.fc = nn.Linear(dim, n_cls)

                def forward(self, x):
                    if hasattr(self.bb, 'forward_features'):
                        feats = self.bb(x)
                    else:
                        feats = self.bb(x)
                    if hasattr(feats, 'logits'):
                        feats = feats.logits
                    return self.fc(feats)

            model = DINOClassifier(backbone, embed_dim, num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model = model.to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_dataset = RepresentationDataset(train_records, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2 if sys.platform != 'win32' else 0, pin_memory=True if device.type == 'cuda' else False)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%")

        # Cross-Representation Evaluation on all representations
        model.eval()
        for test_rep, test_records in test_sets.items():
            if len(test_records) == 0:
                print(f"  [WARN] Test set '{test_rep}' has 0 images. Skipping.")
                continue

            test_dataset = RepresentationDataset(test_records, transform=transform_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2 if sys.platform != 'win32' else 0)

            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f"    Eval on {test_rep:<12} -> Accuracy: {acc*100:6.2f}% | F1: {f1:.4f}")

            results.append({
                "model": model_name,
                "training_setup": setup_name,
                "test_representation": test_rep,
                "accuracy": float(acc),
                "f1_score": float(f1)
            })

    # Print Result Matrix
    df_m = pd.DataFrame(results)
    if len(df_m) > 0:
        pivot_table = df_m.pivot(index="training_setup", columns="test_representation", values="accuracy") * 100
        ordered_cols = [c for c in REPRESENTATIONS if c in pivot_table.columns]
        pivot_table = pivot_table[ordered_cols]
        print(f"\n--- {model_name} ACCURACY RESULT MATRIX (%) ---")
        print(pivot_table.round(2).to_string())

        safe_name = model_name.replace('/', '_').replace('-', '_')
        pivot_table.round(2).to_csv(RESULTS_DIR / f"{safe_name}_matrix.csv")

    return results


# ─────────────────────────────────────────────────────────────
# Zero-Shot CLIP / EVA-CLIP Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_zero_shot_clip(model_name, clip_model_name, pretrained_tag, classes, test_sets, training_setups, device, batch_size=32):
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_name} (Zero-Shot)")
    print(f"Model ID: {clip_model_name} | Pretrained: {pretrained_tag}")
    print(f"{'='*70}")

    try:
        import open_clip
    except ImportError:
        print("[FAIL] 'open_clip' is not installed! Run: pip install open_clip_torch")
        return []

    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained_tag)
    except Exception as e:
        print(f"[FAIL] Could not load {clip_model_name} ({pretrained_tag}): {e}")
        if "EVA" in model_name:
            print("  Trying alternative EVA-CLIP weights...")
            for fallback_model, fallback_tag in [('EVA02-B-16', 'merged2b_s8b_b131k'), ('EVA02-E-14-plus', 'laion2b_s9b_b144k')]:
                try:
                    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(fallback_model, pretrained=fallback_tag)
                    clip_model_name = fallback_model
                    pretrained_tag = fallback_tag
                    print(f"  Successfully loaded fallback {fallback_model}!")
                    break
                except Exception:
                    continue
            else:
                return []
        else:
            return []

    clip_model = clip_model.to(device)
    clip_model.eval()

    tokenizer = open_clip.get_tokenizer(clip_model_name)

    # Representation-aware zero-shot prompts
    def get_prompt(rep, cls_name):
        if rep == "Original":
            return f"a photo of a {cls_name}"
        elif rep == "Outline":
            return f"an outline drawing of a {cls_name}"
        elif rep == "Dotted":
            return f"a dotted drawing of a {cls_name}"
        elif rep == "Dashed":
            return f"a dashed line drawing of a {cls_name}"
        elif rep == "Sketch":
            return f"a pencil sketch of a {cls_name}"
        elif rep == "Silhouette":
            return f"a solid black silhouette of a {cls_name}"
        else:
            return f"a drawing of a {cls_name}"

    results = []

    for test_rep, test_records in test_sets.items():
        if len(test_records) == 0:
            print(f"  [WARN] Test set '{test_rep}' has 0 images. Skipping.")
            continue

        prompts = [get_prompt(test_rep, c) for c in classes]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        test_dataset = RepresentationDataset(test_records, transform=clip_preprocess)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2 if sys.platform != 'win32' else 0)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                preds = torch.argmax(similarity, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"  Test: {test_rep:<12} -> Accuracy: {acc*100:6.2f}% | F1: {f1:.4f}")

        results.append({
            "model": model_name,
            "training_setup": "Zero-Shot",
            "test_representation": test_rep,
            "accuracy": float(acc),
            "f1_score": float(f1)
        })

    return results


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-Representation Evaluation Pipeline")
    parser.add_argument("--classes", type=str, default="all",
                        help="Classes to evaluate: 'all' (all 10 classes), 'furniture' (chair, table), or comma-separated list")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--skip-supervised", action="store_true", help="Skip ResNet/ViT/DINO training")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP zero-shot evaluation")
    parser.add_argument("--skip-eva-clip", action="store_true", help="Skip EVA-CLIP zero-shot evaluation")
    args = parser.parse_args()

    # Determine Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("BTP Cross-Representation Learning & Generalization Pipeline")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 70)

    # Determine Classes
    if args.classes.lower() == "all":
        classes = sorted([d.name for d in ORIGINAL_DIR.iterdir() if d.is_dir()])
    elif args.classes.lower() == "furniture":
        classes = ["chair", "table"]
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Evaluating {len(classes)} classes: {', '.join(classes)}")
    print(f"Representations: {', '.join(REPRESENTATIONS)}\n")

    # Build Data Splits
    training_setups, test_sets, class_to_idx = build_data_splits(classes)
    print("Data splits created:")
    for rep, recs in test_sets.items():
        print(f"  Test [{rep}]: {len(recs)} images")
    for s_name, recs in training_setups.items():
        print(f"  Train [{s_name}]: {len(recs)} images")

    all_results = []

    # 1. Supervised Models (Trained on each variant, evaluated on ALL variants)
    if not args.skip_supervised:
        for model_name in ["ResNet-50", "ViT-B/16", "DINOv3"]:
            res = train_and_eval_model(
                model_name=model_name,
                classes=classes,
                training_setups=training_setups,
                test_sets=test_sets,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
            all_results.extend(res)

    # 2. CLIP ViT-B/32 Zero-Shot
    if not args.skip_clip:
        clip_res = evaluate_zero_shot_clip(
            model_name="CLIP ViT-B/32",
            clip_model_name="ViT-B-32",
            pretrained_tag="openai",
            classes=classes,
            test_sets=test_sets,
            training_setups=training_setups,
            device=device,
            batch_size=args.batch_size
        )
        all_results.extend(clip_res)

    # 3. EVA-CLIP Zero-Shot
    if not args.skip_eva_clip:
        eva_res = evaluate_zero_shot_clip(
            model_name="EVA-CLIP (EVA02-B/16)",
            clip_model_name="EVA02-B-16",
            pretrained_tag="merged2b_s8b_b131k",
            classes=classes,
            test_sets=test_sets,
            training_setups=training_setups,
            device=device,
            batch_size=args.batch_size
        )
        all_results.extend(eva_res)

    # Save consolidated results
    df_all = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "classification_results.csv"
    df_all.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("CONSOLIDATED SUMMARY RESULTS")
    print("=" * 70)
    print(f"Results saved to: {out_csv}\n")

    # Print summary pivot for all models
    if not df_all.empty:
        summary_pivot = df_all.pivot_table(
            index=["model", "training_setup"],
            columns="test_representation",
            values="accuracy"
        ) * 100
        ordered_cols = [c for c in REPRESENTATIONS if c in summary_pivot.columns]
        summary_pivot = summary_pivot[ordered_cols]
        print(summary_pivot.round(2).to_string())

    print("\n[OK] All training and evaluations completed successfully!")


if __name__ == "__main__":
    main()
