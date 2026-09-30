"""
Script 4: Cross-Representation Training and Evaluation
Models:
  - ResNet-50 (Fine-tuned)
  - ViT-B/16 (Fine-tuned)
  - DINOv2 / DINOv3 (Linear probe / Classifier head)
  - CLIP ViT-B/32 (Zero-Shot)
  - EVA-CLIP (Zero-Shot)
  - SigLIP-2 Base (Zero-Shot)

Representations:
  - Original
  - Outline
  - Dotted
  - Dashed
  - Sketch
  - Silhouette
  - ColorTint_Red
  - ColorTint_Green
  - ColorTint_Blue

Generates full Cross-Representation Matrix:
  Train on variant X -> Test on ALL variants.
Saves all metrics to CSV and prints formatted accuracy matrices on terminal.
Also saves class-wise metrics and model-wise inference time.
"""

import os
import sys
import re
import argparse
import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# SigLIP-2 (Hugging Face Transformers)
try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

# ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR = PROJECT_DIR / "dataset" / "variants"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette",
    "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
]

REPRESENTATION_PATHS = {
    "Original": ORIGINAL_DIR,
    "Outline": VARIANTS_DIR / "outline",
    "Dotted": VARIANTS_DIR / "dotted",
    "Dashed": VARIANTS_DIR / "dashed",
    "Sketch": VARIANTS_DIR / "sketch",
    "Silhouette": VARIANTS_DIR / "silhouette",
    "ColorTint_Red": VARIANTS_DIR / "color_tint_red",
    "ColorTint_Green": VARIANTS_DIR / "color_tint_green",
    "ColorTint_Blue": VARIANTS_DIR / "color_tint_blue",
}


# ─────────────────────────────────────────────────────────────
# Qwen3-VL-8B configuration
# ─────────────────────────────────────────────────────────────

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_MODEL_NAME = "Qwen3-VL-8B"
QWEN_MAX_NEW_TOKENS = 16

# Qwen output files are stored directly in results/, like all other models.

# Qwen uses the same root results directory as every other model.
QWEN_RESULTS_DIR = RESULTS_DIR

QWEN_CLASSIFICATION_INSTRUCTION = """
Identify the object represented in this image.

Choose exactly ONE class from the following list:
car, cat, chair, circle, cup, dog, rectangle, square, table, triangle.

Answer with ONLY the class name and nothing else.
""".strip()


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
        if not base_dir.exists():
            print(f"  [WARN] Representation directory not found: {base_dir}. Skipping Train_{rep}.")
            continue
        records = []
        for c in classes:
            c_dir = base_dir / c
            for fname in class_train_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append((str(p), class_to_idx[c]))
        if records:
            training_setups[f"Train_{rep}"] = records

    # Combined training setup (trained on all representations)
    all_combined_records = []
    for rep in REPRESENTATIONS:
        key = f"Train_{rep}"
        if key in training_setups:
            all_combined_records.extend(training_setups[key])
    training_setups["Train_All_Combined"] = all_combined_records

    # Build Test Sets for each representation
    test_sets = {}
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        if not base_dir.exists():
            print(f"  [WARN] Representation directory not found: {base_dir}. Skipping test set for {rep}.")
            continue
        records = []
        for c in classes:
            c_dir = base_dir / c
            for fname in class_test_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append((str(p), class_to_idx[c]))
        if records:
            test_sets[rep] = records

    return training_setups, test_sets, class_to_idx


# ─────────────────────────────────────────────────────────────
# Class-wise Metrics Helper
# ─────────────────────────────────────────────────────────────

def compute_classwise_metrics(all_labels, all_preds, classes, class_to_idx):
    """
    Compute per-class accuracy, precision, recall, F1.
    Returns a list of dicts, one per class.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(classes)

    # Per-class precision, recall, F1
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))

    # Per-class accuracy: correct predictions for class i / total samples of class i
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    class_totals = cm.sum(axis=1)
    class_correct = cm.diagonal()
    accuracy_per_class = np.where(class_totals > 0, class_correct / class_totals, 0.0)

    results = []
    for i in range(num_classes):
        class_name = idx_to_class.get(i, f"class_{i}")
        results.append({
            "class_name": class_name,
            "accuracy": float(accuracy_per_class[i]),
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        })

    return results


def print_classwise_table(classwise_metrics, model_name, setup_name, test_rep):
    """Print a formatted class-wise metrics table to terminal."""
    print(f"\n    {'Class':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"    {'-'*52}")
    for m in classwise_metrics:
        print(f"    {m['class_name']:<12} {m['accuracy']*100:>9.2f}% {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% {m['f1']:>10.4f}")
    print()


# ─────────────────────────────────────────────────────────────
# Inference Time Helper
# ─────────────────────────────────────────────────────────────

def measure_inference_time(model, dataloader, device):
    """
    Measure inference time for a model.
    Returns (total_images, total_time_sec).
    Uses torch.cuda.synchronize() for accurate GPU timing.
    """
    model.eval()

    # GPU warmup pass (3 batches or all data if fewer)
    warmup_count = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            warmup_count += 1
            if warmup_count >= 3:
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_images = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            total_images += images.size(0)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    return total_images, total_time


# ─────────────────────────────────────────────────────────────
# Supervised Model Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_and_eval_model(model_name, classes, training_setups, test_sets, device,
                         class_to_idx, epochs=5, batch_size=32, lr=1e-4):
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
    classwise_results = []
    inference_results = []
    model_inference_measured = False

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

            print(f"    Eval on {test_rep:<16} -> Accuracy: {acc*100:6.2f}% | F1: {f1:.4f}")

            # Class-wise metrics
            cw_metrics = compute_classwise_metrics(all_labels, all_preds, classes, class_to_idx)
            print_classwise_table(cw_metrics, model_name, setup_name, test_rep)

            for m in cw_metrics:
                row = {
                    "model": model_name,
                    "training_setup": setup_name,
                    "test_representation": test_rep,
                    "class_name": m["class_name"],
                    "accuracy": round(m["accuracy"], 4),
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "f1_score": round(m["f1"], 4)
                }
                classwise_results.append(row)
                results.append(row)

            # Also append overall macro summary row
            macro_prec = float(np.mean([m["precision"] for m in cw_metrics]))
            macro_rec = float(np.mean([m["recall"] for m in cw_metrics]))
            results.append({
                "model": model_name,
                "training_setup": setup_name,
                "test_representation": test_rep,
                "class_name": "ALL",
                "accuracy": round(float(acc), 4),
                "precision": round(macro_prec, 4),
                "recall": round(macro_rec, 4),
                "f1_score": round(float(f1), 4)
            })

            # Measure inference time ONCE per model (model-wise)
            if not model_inference_measured:
                n_imgs, t_sec = measure_inference_time(model, test_loader, device)
                avg_ms = (t_sec / n_imgs * 1000) if n_imgs > 0 else 0
                throughput = n_imgs / t_sec if t_sec > 0 else 0
                inference_results.append({
                    "model": model_name,
                    "architecture_type": "Supervised" if "DINO" not in model_name else "Linear Probe",
                    "device": f"{device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
                    "batch_size": batch_size,
                    "total_images": n_imgs,
                    "total_time_sec": round(t_sec, 4),
                    "avg_latency_ms": round(avg_ms, 4),
                    "avg_time_ms": round(avg_ms, 4),
                    "throughput_fps": round(throughput, 2),
                    "throughput_img_per_sec": round(throughput, 2)
                })
                print(f"    ⏱ [Model-Wise Benchmark] {model_name}: {avg_ms:.2f} ms/image | {throughput:.1f} FPS ({n_imgs} images)")
                model_inference_measured = True

    # Print Result Matrix (using macro ALL summary)
    df_m = pd.DataFrame(results)
    if len(df_m) > 0:
        df_summary = df_m[df_m["class_name"] == "ALL"]
        pivot_table = df_summary.pivot(index="training_setup", columns="test_representation", values="accuracy") * 100
        ordered_cols = [c for c in REPRESENTATIONS if c in pivot_table.columns]
        pivot_table = pivot_table[ordered_cols]
        print(f"\n--- {model_name} MACRO ACCURACY MATRIX (%) ---")
        print(pivot_table.round(2).to_string())

        safe_name = model_name.replace('/', '_').replace('-', '_')
        pivot_table.round(2).to_csv(RESULTS_DIR / f"{safe_name}_matrix.csv")

        # Also save model-specific class-wise metrics
        df_cw_m = pd.DataFrame([r for r in classwise_results if r["model"] == model_name])
        if len(df_cw_m) > 0:
            df_cw_m.to_csv(RESULTS_DIR / f"{safe_name}_classwise.csv", index=False)

    return results, classwise_results, inference_results


# ─────────────────────────────────────────────────────────────
# Zero-Shot CLIP / EVA-CLIP Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_zero_shot_clip(model_name, clip_model_name, pretrained_tag, classes, test_sets,
                            training_setups, device, class_to_idx, batch_size=32):
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_name} (Zero-Shot)")
    print(f"Model ID: {clip_model_name} | Pretrained: {pretrained_tag}")
    print(f"{'='*70}")

    try:
        import open_clip
    except ImportError:
        print("[FAIL] 'open_clip' is not installed! Run: pip install open_clip_torch")
        return [], [], []

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
                return [], [], []
        else:
            return [], [], []

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
        elif rep.startswith("ColorTint"):
            color = rep.split("_")[-1].lower()
            return f"a {color}-tinted photo of a {cls_name}"
        else:
            return f"a drawing of a {cls_name}"

    results = []
    classwise_results = []
    inference_results = []
    inference_measured = False

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

        print(f"  Test: {test_rep:<16} -> Accuracy: {acc*100:6.2f}% | F1: {f1:.4f}")

        # Class-wise metrics
        cw_metrics = compute_classwise_metrics(all_labels, all_preds, classes, class_to_idx)
        print_classwise_table(cw_metrics, model_name, "Zero-Shot", test_rep)

        for m in cw_metrics:
            row = {
                "model": model_name,
                "training_setup": "Zero-Shot",
                "test_representation": test_rep,
                "class_name": m["class_name"],
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1_score": round(m["f1"], 4)
            }
            classwise_results.append(row)
            results.append(row)

        # Also append overall macro summary row
        macro_prec = float(np.mean([m["precision"] for m in cw_metrics]))
        macro_rec = float(np.mean([m["recall"] for m in cw_metrics]))
        results.append({
            "model": model_name,
            "training_setup": "Zero-Shot",
            "test_representation": test_rep,
            "class_name": "ALL",
            "accuracy": round(float(acc), 4),
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(float(f1), 4)
        })

        # Measure inference time ONCE per model (model-wise)
        if not inference_measured:
            class CLIPImageEncoder(nn.Module):
                def __init__(self, clip_m):
                    super().__init__()
                    self.clip_m = clip_m
                def forward(self, x):
                    return self.clip_m.encode_image(x)

            wrapper = CLIPImageEncoder(clip_model)
            n_imgs, t_sec = measure_inference_time(wrapper, test_loader, device)
            avg_ms = (t_sec / n_imgs * 1000) if n_imgs > 0 else 0
            throughput = n_imgs / t_sec if t_sec > 0 else 0
            inference_results.append({
                "model": model_name,
                "architecture_type": "Zero-Shot VLM",
                "device": f"{device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
                "batch_size": batch_size,
                "total_images": n_imgs,
                "total_time_sec": round(t_sec, 4),
                "avg_latency_ms": round(avg_ms, 4),
                "avg_time_ms": round(avg_ms, 4),
                "throughput_fps": round(throughput, 2),
                "throughput_img_per_sec": round(throughput, 2)
            })
            print(f"    ⏱ [Model-Wise Benchmark] {model_name}: {avg_ms:.2f} ms/image | {throughput:.1f} FPS ({n_imgs} images)")
            inference_measured = True

    # Save model-specific class-wise metrics
    safe_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
    df_cw_m = pd.DataFrame([r for r in classwise_results if r["model"] == model_name])
    if len(df_cw_m) > 0:
        df_cw_m.to_csv(RESULTS_DIR / f"{safe_name}_classwise.csv", index=False)

    return results, classwise_results, inference_results


# ─────────────────────────────────────────────────────────────
# Zero-Shot SigLIP-2 Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_zero_shot_siglip2(model_name, model_id, classes, test_sets, training_setups,
                               device, class_to_idx, batch_size=32):
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_name} (Zero-Shot)")
    print(f"Model ID: {model_id}")
    print(f"{'='*70}")

    if AutoModel is None or AutoProcessor is None:
        print("[FAIL] 'transformers' is not installed! Run: pip install transformers")
        return [], [], []

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        siglip_model = AutoModel.from_pretrained(model_id)
    except Exception as e:
        print(f"[FAIL] Could not load {model_id}: {e}")
        return [], [], []

    siglip_model = siglip_model.to(device)
    siglip_model.eval()

    # Representation-aware zero-shot prompts.
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
        elif rep.startswith("ColorTint"):
            color = rep.split("_")[-1].lower()
            return f"a {color}-tinted photo of a {cls_name}"
        else:
            return f"a drawing of a {cls_name}"

    results = []
    classwise_results = []
    inference_results = []
    inference_measured = False

    for test_rep, test_records in test_sets.items():
        if len(test_records) == 0:
            print(f"  [WARN] Test set '{test_rep}' has 0 images. Skipping.")
            continue

        prompts = [get_prompt(test_rep, c) for c in classes]

        # SigLIP-2 was trained with fixed-length text padding.
        text_inputs = processor(
            text=prompts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

        text_inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in text_inputs.items()
        }

        with torch.no_grad():
            text_outputs = siglip_model.get_text_features(**text_inputs)

            if hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            elif isinstance(text_outputs, tuple):
                text_features = text_outputs[0]
            else:
                text_features = text_outputs

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        all_preds = []
        all_labels = []

        # Timing for SigLIP-2
        if not inference_measured:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_start = time.perf_counter()

        # Load PIL images directly in batches because the official SigLIP-2
        # processor performs its own image resizing/normalization.
        for start_idx in range(0, len(test_records), batch_size):
            batch_records = test_records[start_idx:start_idx + batch_size]

            images = [
                Image.open(path).convert("RGB")
                for path, _ in batch_records
            ]
            labels = torch.tensor(
                [label for _, label in batch_records],
                dtype=torch.long
            )

            # Process images using the official SigLIP-2 image processor.
            image_inputs = processor(
                images=images,
                return_tensors="pt"
            )

            image_inputs = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in image_inputs.items()
            }

            with torch.no_grad():
                image_outputs = siglip_model.get_image_features(**image_inputs)

                if hasattr(image_outputs, "pooler_output"):
                    image_features = image_outputs.pooler_output
                elif isinstance(image_outputs, tuple):
                    image_features = image_outputs[0]
                else:
                    image_features = image_outputs

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # SigLIP-2 computes normalized image/text cosine scores,
                # followed by the learned logit scale and bias.
                logits = image_features @ text_features.T
                logits = logits * siglip_model.logit_scale.exp() + siglip_model.logit_bias

                # SigLIP-2 is trained with sigmoid loss, but for mutually
                # exclusive object classes we select the highest-scoring class.
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Explicitly release PIL objects before the next batch.
            for image in images:
                image.close()

        # Capture inference time ONCE per model (model-wise)
        if not inference_measured:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - infer_start
            n_imgs = len(test_records)
            avg_ms = (infer_time / n_imgs * 1000) if n_imgs > 0 else 0
            throughput = n_imgs / infer_time if infer_time > 0 else 0
            inference_results.append({
                "model": model_name,
                "architecture_type": "Zero-Shot VLM",
                "device": f"{device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
                "batch_size": batch_size,
                "total_images": n_imgs,
                "total_time_sec": round(infer_time, 4),
                "avg_latency_ms": round(avg_ms, 4),
                "avg_time_ms": round(avg_ms, 4),
                "throughput_fps": round(throughput, 2),
                "throughput_img_per_sec": round(throughput, 2)
            })
            print(f"    ⏱ [Model-Wise Benchmark] {model_name}: {avg_ms:.2f} ms/image | {throughput:.1f} FPS ({n_imgs} images)")
            inference_measured = True

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"  Test: {test_rep:<16} -> Accuracy: {acc*100:6.2f}% | F1: {f1:.4f}")

        # Class-wise metrics
        cw_metrics = compute_classwise_metrics(all_labels, all_preds, classes, class_to_idx)
        print_classwise_table(cw_metrics, model_name, "Zero-Shot", test_rep)

        for m in cw_metrics:
            row = {
                "model": model_name,
                "training_setup": "Zero-Shot",
                "test_representation": test_rep,
                "class_name": m["class_name"],
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1_score": round(m["f1"], 4)
            }
            classwise_results.append(row)
            results.append(row)

        # Also append overall macro summary row
        macro_prec = float(np.mean([m["precision"] for m in cw_metrics]))
        macro_rec = float(np.mean([m["recall"] for m in cw_metrics]))
        results.append({
            "model": model_name,
            "training_setup": "Zero-Shot",
            "test_representation": test_rep,
            "class_name": "ALL",
            "accuracy": round(float(acc), 4),
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(float(f1), 4)
        })

    # Save model-specific class-wise metrics
    safe_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
    df_cw_m = pd.DataFrame([r for r in classwise_results if r["model"] == model_name])
    if len(df_cw_m) > 0:
        df_cw_m.to_csv(RESULTS_DIR / f"{safe_name}_classwise.csv", index=False)

    # Release the model before returning so subsequent models can use GPU memory.
    del siglip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results, classwise_results, inference_results


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# Qwen3-VL-8B Zero-Shot Generative Evaluation
# ─────────────────────────────────────────────────────────────

def build_qwen_test_sets(classes, train_ratio=0.8, seed=42):
    """
    Reproduce the same deterministic train/test split used by Script 4.

    The original-image filenames determine the split, and the same
    filenames are then used for every representation.
    """
    random.seed(seed)

    class_train_files = {}
    class_test_files = {}

    for c in classes:
        orig_class_dir = ORIGINAL_DIR / c

        files = sorted(
            [f.name for f in orig_class_dir.glob("*.jpg")]
        )

        if not files:
            raise FileNotFoundError(
                f"No images found for class '{c}' in {orig_class_dir}"
            )

        n_total = len(files)
        n_train = int(n_total * train_ratio)

        indices = list(range(n_total))
        random.shuffle(indices)

        train_indices = set(indices[:n_train])

        class_train_files[c] = [
            files[i] for i in range(n_total)
            if i in train_indices
        ]

        class_test_files[c] = [
            files[i] for i in range(n_total)
            if i not in train_indices
        ]

    test_sets = {}

    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]

        if not base_dir.exists():
            print(
                f"  [WARN] Representation directory not found: "
                f"{base_dir}. Skipping {rep}."
            )
            continue

        records = []

        for c in classes:
            c_dir = base_dir / c

            for fname in class_test_files[c]:
                p = c_dir / fname

                if p.exists():
                    records.append((str(p), c))

        if records:
            test_sets[rep] = records

    return test_sets

def qwen_get_prompt(rep):
    """
    Representation-aware prompts.

    These follow the same conceptual wording used by the existing
    CLIP / SigLIP-2 zero-shot implementation.
    """

    if rep == "Original":
        representation_description = (
            "This is a natural image of an object."
        )

    elif rep == "Outline":
        representation_description = (
            "This is an outline drawing of an object."
        )

    elif rep == "Dotted":
        representation_description = (
            "This is a dotted drawing of an object."
        )

    elif rep == "Dashed":
        representation_description = (
            "This is a dashed line drawing of an object."
        )

    elif rep == "Sketch":
        representation_description = (
            "This is a pencil sketch of an object."
        )

    elif rep == "Silhouette":
        representation_description = (
            "This is a solid black silhouette representation of an object."
        )

    elif rep == "ColorTint_Red":
        representation_description = (
            "This is a red-tinted image of an object."
        )

    elif rep == "ColorTint_Green":
        representation_description = (
            "This is a green-tinted image of an object."
        )

    elif rep == "ColorTint_Blue":
        representation_description = (
            "This is a blue-tinted image of an object."
        )

    else:
        representation_description = (
            "This image contains an object."
        )

    return f"""
{representation_description}

{QWEN_CLASSIFICATION_INSTRUCTION}
""".strip()

def normalize_prediction(response, classes):
    """
    Convert Qwen's generated text into exactly one benchmark class.

    The prompt asks for only the class name, but the parser is made
    tolerant of short responses such as:
        "chair"
        "The answer is chair."
        "I think it is a chair."

    If no valid class can be extracted, return "unknown".
    """

    if response is None:
        return "unknown"

    text = str(response).strip().lower()

    # Remove common formatting.
    text = text.replace("`", " ")
    text = text.replace("*", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Exact match first.
    for cls in classes:
        if text == cls.lower():
            return cls

    # Look for a class as a standalone word.
    # Sort longer names first to avoid accidental partial matching.
    sorted_classes = sorted(
        classes,
        key=len,
        reverse=True,
    )

    for cls in sorted_classes:
        pattern = rf"\b{re.escape(cls.lower())}\b"
        if re.search(pattern, text):
            return cls

    return "unknown"

def ask_qwen(model, processor, image_path, prompt):
    """
    Run one image through Qwen3-VL.

    One image = one independent request/context.
    This avoids interactions between different benchmark images.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Qwen's official Transformers examples move the processed
    # multimodal inputs to the model device before generation.
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip() if output_text else ""

def load_existing_predictions(path):
    """
    Load previously completed predictions so a stopped/crashed run
    can resume instead of starting from zero.
    """

    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)

        if df.empty:
            return {}

        required = {
            "image_path",
            "representation",
            "true_class",
            "raw_response",
            "prediction",
        }

        if not required.issubset(df.columns):
            print(
                "[WARN] Existing predictions.csv has an unexpected "
                "format. Starting without checkpoint recovery."
            )
            return {}

        completed = {}

        for _, row in df.iterrows():
            key = (
                str(row["representation"]),
                str(row["image_path"]),
            )

            completed[key] = row.to_dict()

        print(
            f"[Checkpoint] Loaded {len(completed)} existing predictions."
        )

        return completed

    except Exception as e:
        print(
            f"[WARN] Could not read existing checkpoint: {e}"
        )
        return {}

def append_prediction(path, row):
    """
    Append one prediction immediately to disk.

    This makes long Qwen runs resumable.
    """

    df = pd.DataFrame([row])

    write_header = not path.exists()

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )

def qwen_compute_classwise_metrics(
    all_labels,
    all_preds,
    classes,
):
    """
    Compute Qwen class-wise metrics using string class labels.

    Qwen produces textual class predictions such as "chair" or
    "triangle", so this function maps the strings to integer indices
    internally before using sklearn metrics. This mirrors the
    standalone Qwen evaluation implementation.
    """

    class_to_idx = {
        c: i for i, c in enumerate(classes)
    }

    labels_idx = [
        class_to_idx[x]
        for x in all_labels
    ]

    # Unknown/unparseable Qwen answers are mapped to -1.
    preds_idx = [
        class_to_idx.get(x, -1)
        for x in all_preds
    ]

    num_classes = len(classes)
    valid_labels = list(range(num_classes))

    precision_per_class = precision_score(
        labels_idx,
        preds_idx,
        average=None,
        zero_division=0,
        labels=valid_labels,
    )

    recall_per_class = recall_score(
        labels_idx,
        preds_idx,
        average=None,
        zero_division=0,
        labels=valid_labels,
    )

    f1_per_class = f1_score(
        labels_idx,
        preds_idx,
        average=None,
        zero_division=0,
        labels=valid_labels,
    )

    cm = confusion_matrix(
        labels_idx,
        preds_idx,
        labels=valid_labels,
    )

    class_totals = cm.sum(axis=1)
    class_correct = cm.diagonal()

    accuracy_per_class = np.where(
        class_totals > 0,
        class_correct / class_totals,
        0.0,
    )

    results = []

    for i, class_name in enumerate(classes):
        results.append(
            {
                "class_name": class_name,
                "accuracy": float(accuracy_per_class[i]),
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
        )

    return results


def qwen_print_classwise_table(classwise_metrics):
    print(
        f"\n    {'Class':<12}"
        f"{'Accuracy':>10}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )
    print(f"    {'-' * 52}")
    for m in classwise_metrics:
        print(
            f"    {m['class_name']:<12}"
            f"{m['accuracy'] * 100:>9.2f}% "
            f"{m['precision'] * 100:>9.2f}% "
            f"{m['recall'] * 100:>9.2f}% "
            f"{m['f1']:>10.4f}"
        )
    print()

def evaluate_qwen(
    classes,
    test_sets,
    device,
    limit_per_rep=None,
    resume=True,
):
    print(f"\n{'=' * 70}")
    print(f"EVALUATING MODEL: {QWEN_MODEL_NAME} (Zero-Shot Generative VLM)")
    print(f"Model ID: {QWEN_MODEL_ID}")
    print(f"{'=' * 70}")

    if AutoProcessor is None or Qwen3VLForConditionalGeneration is None:
        print(
            "[FAIL] Required Qwen/Transformers classes are unavailable."
        )
        print(
            'Install a current Transformers version with:'
        )
        print(
            '  pip install -U "transformers>=4.57.0" accelerate'
        )
        return [], [], []

    # ─────────────────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────────────────

    print("\nLoading Qwen3-VL processor...")

    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID
    )

    print("Loading Qwen3-VL-8B-Instruct...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )

    model.eval()

    print("Qwen3-VL loaded successfully.")
    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ─────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────

    predictions_csv = (
        RESULTS_DIR / "Qwen3_VL_8B_predictions.csv"
    )

    completed = (
        load_existing_predictions(predictions_csv)
        if resume
        else {}
    )

    all_predictions = []

    # Load checkpoint rows for final metrics.
    if completed:
        all_predictions.extend(
            completed.values()
        )

    total_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────
    # Evaluate every representation
    # ─────────────────────────────────────────────────────────

    for test_rep, test_records in test_sets.items():

        if len(test_records) == 0:
            print(
                f"  [WARN] Test set '{test_rep}' has 0 images. Skipping."
            )
            continue

        records_to_process = list(test_records)

        if limit_per_rep is not None:
            records_to_process = records_to_process[
                :limit_per_rep
            ]

        print(
            f"\n--- Representation: {test_rep} "
            f"({len(records_to_process)} images) ---"
        )

        prompt = qwen_get_prompt(test_rep)

        rep_start = time.perf_counter()
        processed_now = 0

        for idx, (image_path, true_class) in enumerate(
            records_to_process,
            start=1,
        ):

            checkpoint_key = (
                test_rep,
                image_path,
            )

            if checkpoint_key in completed:
                continue

            print(
                f"  [{idx}/{len(records_to_process)}] "
                f"{image_path}",
                end="",
                flush=True,
            )

            image_start = time.perf_counter()

            try:
                raw_response = ask_qwen(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt=prompt,
                )

                prediction = normalize_prediction(
                    raw_response,
                    classes,
                )

                elapsed = (
                    time.perf_counter()
                    - image_start
                )

                print(
                    f" -> {prediction}"
                    f" | {elapsed:.2f}s"
                )

                row = {
                    "model": QWEN_MODEL_NAME,
                    "image_path": image_path,
                    "representation": test_rep,
                    "true_class": true_class,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "prediction": prediction,
                    "latency_sec": round(
                        elapsed,
                        4,
                    ),
                }

                append_prediction(
                    predictions_csv,
                    row,
                )

                completed[checkpoint_key] = row
                all_predictions.append(row)

                processed_now += 1

            except Exception as e:
                print(
                    f" -> ERROR: {e}"
                )

                # Keep going so one problematic image does not
                # terminate the complete benchmark.
                continue

        rep_time = (
            time.perf_counter()
            - rep_start
        )

        print(
            f"  Representation time: "
            f"{rep_time:.2f}s"
        )

        if processed_now > 0:
            print(
                f"  Newly processed: "
                f"{processed_now}"
            )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # ─────────────────────────────────────────────────────────
    # Re-read the complete checkpoint for reliable metrics
    # ─────────────────────────────────────────────────────────

    if not predictions_csv.exists():
        print(
            "\n[FAIL] No predictions were generated."
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [], [], []

    df_predictions = pd.read_csv(
        predictions_csv
    )

    # Restrict metrics to requested representations.
    if test_sets:
        df_predictions = df_predictions[
            df_predictions["representation"].isin(
                test_sets.keys()
            )
        ].copy()

    # Restrict to requested classes.
    df_predictions = df_predictions[
        df_predictions["true_class"].isin(classes)
    ].copy()

    if df_predictions.empty:
        print(
            "\n[FAIL] No valid predictions available for metrics."
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [], [], []

    # ─────────────────────────────────────────────────────────
    # Overall results
    # ─────────────────────────────────────────────────────────

    results = []
    classwise_results = []

    for test_rep in test_sets.keys():

        df_rep = df_predictions[
            df_predictions["representation"] == test_rep
        ]

        if df_rep.empty:
            continue

        all_labels = df_rep["true_class"].tolist()
        all_preds = df_rep["prediction"].tolist()

        acc = accuracy_score(
            all_labels,
            all_preds,
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        precision = precision_score(
            all_labels,
            all_preds,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            all_labels,
            all_preds,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        print(
            f"\n  Test: {test_rep:<18}"
            f" -> Accuracy: {acc * 100:6.2f}%"
            f" | F1: {f1:.4f}"
        )

        # Qwen keeps labels as strings, so use the Qwen-specific
        # class-wise metric implementation.
        cw_metrics = qwen_compute_classwise_metrics(
            all_labels,
            all_preds,
            classes,
        )

        qwen_print_classwise_table(
            cw_metrics
        )

        for m in cw_metrics:
            classwise_results.append(
                {
                    "model": QWEN_MODEL_NAME,
                    "training_setup": "Zero-Shot",
                    "test_representation": test_rep,
                    "class_name": m["class_name"],
                    "accuracy": round(
                        m["accuracy"],
                        4,
                    ),
                    "precision": round(
                        m["precision"],
                        4,
                    ),
                    "recall": round(
                        m["recall"],
                        4,
                    ),
                    "f1_score": round(
                        m["f1"],
                        4,
                    ),
                }
            )

        results.append(
            {
                "model": QWEN_MODEL_NAME,
                "training_setup": "Zero-Shot",
                "test_representation": test_rep,
                "class_name": "ALL",
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "num_images": len(df_rep),
            }
        )

    # ─────────────────────────────────────────────────────────
    # Save metrics
    # ─────────────────────────────────────────────────────────

    results_csv = (
        RESULTS_DIR / "Qwen3_VL_8B_results.csv"
    )

    classwise_csv = (
        RESULTS_DIR / "Qwen3_VL_8B_classwise.csv"
    )

    pd.DataFrame(results).to_csv(
        results_csv,
        index=False,
    )

    pd.DataFrame(classwise_results).to_csv(
        classwise_csv,
        index=False,
    )

    # ─────────────────────────────────────────────────────────
    # Inference benchmark
    # ─────────────────────────────────────────────────────────

    valid_latency = pd.to_numeric(
        df_predictions["latency_sec"],
        errors="coerce",
    ).dropna()

    if len(valid_latency) > 0:
        total_inference_time = float(
            valid_latency.sum()
        )

        total_images = len(valid_latency)

        avg_latency_ms = (
            total_inference_time
            / total_images
            * 1000
        )

        throughput = (
            total_images
            / total_inference_time
            if total_inference_time > 0
            else 0
        )

        inference_df = pd.DataFrame(
            [
                {
                    "model": QWEN_MODEL_NAME,
                    "architecture_type": "Generative VLM",
                    "device": (
                        f"{device.type} "
                        f"({torch.cuda.get_device_name(0)})"
                        if device.type == "cuda"
                        else "CPU"
                    ),
                    "total_images": total_images,
                    "total_time_sec": round(
                        total_inference_time,
                        4,
                    ),
                    "avg_latency_ms": round(
                        avg_latency_ms,
                        4,
                    ),
                    "avg_time_ms": round(
                        avg_latency_ms,
                        4,
                    ),
                    "throughput_fps": round(
                        throughput,
                        4,
                    ),
                    "throughput_img_per_sec": round(
                        throughput,
                        4,
                    ),
                }
            ]
        )

        inference_csv = (
            RESULTS_DIR / "Qwen3_VL_8B_inference_time.csv"
        )

        inference_df.to_csv(
            inference_csv,
            index=False,
        )

        print(
            f"\n[Benchmark] Qwen3-VL:"
            f" {avg_latency_ms:.2f} ms/image"
            f" | {throughput:.2f} images/s"
        )

    # ─────────────────────────────────────────────────────────
    # Print final matrix
    # ─────────────────────────────────────────────────────────

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        matrix = (
            df_results
            .set_index("test_representation")["accuracy"]
            .to_frame()
            .T
            * 100
        )

        ordered_cols = [
            rep
            for rep in REPRESENTATIONS
            if rep in matrix.columns
        ]

        matrix = matrix[ordered_cols]

        print(
            f"\n{'=' * 90}"
        )
        print(
            "QWEN3-VL-8B ZERO-SHOT ACCURACY MATRIX (%)"
        )
        print(
            f"{'=' * 90}"
        )
        print(
            matrix.round(2).to_string(
                index=False
            )
        )

        matrix_csv = (
            RESULTS_DIR
            / "Qwen3_VL_8B_matrix.csv"
        )

        matrix.round(2).to_csv(
            matrix_csv,
            index=False,
        )

    print(
        f"\n[OK] Predictions saved to:"
        f" {predictions_csv}"
    )

    print(
        f"[OK] Results saved to:"
        f" {results_csv}"
    )

    print(
        f"[OK] Class-wise metrics saved to:"
        f" {classwise_csv}"
    )

    print(
        f"[OK] Total evaluation wall time:"
        f" {total_time:.2f}s"
    )

    # Convert Qwen summary rows to the common Script-4 schema.
    master_results = []
    for r in results:
        master_results.append({
            "model": r["model"],
            "training_setup": r["training_setup"],
            "test_representation": r["test_representation"],
            "class_name": "ALL",
            "accuracy": round(float(r["accuracy"]), 4),
            "precision": round(float(r["precision"]), 4),
            "recall": round(float(r["recall"]), 4),
            "f1_score": round(float(r["f1_score"]), 4),
        })

    # Save a Qwen-specific classwise file and return data for the master CSV.
    # Release GPU memory.
    del model
    del processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return master_results, classwise_results, inference_df.to_dict('records') if 'inference_df' in locals() else []


# ─────────────────────────────────────────────────────────────
# Safe master-result merge
# ─────────────────────────────────────────────────────────────

def merge_and_save_master_results(new_results, new_classwise, new_inference, models_ran):
    """
    Merge results from the current execution into the master CSVs.

    Important:
      - Models run in THIS execution replace only their own old rows.
      - Models skipped in THIS execution remain untouched.
      - This prevents a SigLIP/Qwen-only run from deleting ResNet/ViT/DINO/CLIP/EVA results.
    """
    def read_existing(path):
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] Could not read existing {path.name}: {e}")
        return pd.DataFrame()

    # ---- Classification results ----
    out_csv = RESULTS_DIR / "classification_results.csv"
    old = read_existing(out_csv)
    new_df = pd.DataFrame(new_results)

    if not new_df.empty:
        if not old.empty and "model" in old.columns:
            old = old[~old["model"].isin(models_ran)].copy()
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = old.copy()

    if not merged.empty:
        # Stable ordering: model, training setup, representation, class.
        model_order = [
            "ResNet-50", "ViT-B/16", "DINOv3",
            "CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)",
            "SigLIP-2 Base", "Qwen3-VL-8B"
        ]
        merged["_model_order"] = merged["model"].map(
            {m:i for i,m in enumerate(model_order)}
        ).fillna(999)
        merged["_class_order"] = merged["class_name"].apply(
            lambda x: 999999 if x == "ALL" else 0
        )
        merged = merged.sort_values(
            ["_model_order", "training_setup",
             "test_representation", "_class_order", "class_name"],
            kind="stable"
        ).drop(columns=["_model_order", "_class_order"])
        merged.to_csv(out_csv, index=False)

    print(f"[OK] Master classification results saved: {out_csv}")
    print(f"     Models present: {sorted(merged['model'].dropna().unique().tolist()) if not merged.empty else []}")

    # ---- Classwise metrics ----
    cw_csv = RESULTS_DIR / "classwise_metrics.csv"
    old_cw = read_existing(cw_csv)
    new_cw = pd.DataFrame(new_classwise)

    if not new_cw.empty:
        if not old_cw.empty and "model" in old_cw.columns:
            old_cw = old_cw[~old_cw["model"].isin(models_ran)].copy()
        merged_cw = pd.concat([old_cw, new_cw], ignore_index=True)
    else:
        merged_cw = old_cw.copy()

    if not merged_cw.empty:
        merged_cw.to_csv(cw_csv, index=False)
    print(f"[OK] Master classwise metrics saved: {cw_csv}")

    # ---- Macro summary ----
    if not merged.empty and "class_name" in merged.columns:
        macro_csv = RESULTS_DIR / "summary_metrics.csv"
        macro = merged[merged["class_name"] == "ALL"].copy()
        macro.to_csv(macro_csv, index=False)
        print(f"[OK] Master summary metrics saved: {macro_csv}")

    # ---- Inference time ----
    inf_csv = RESULTS_DIR / "inference_time.csv"
    old_inf = read_existing(inf_csv)
    new_inf = pd.DataFrame(new_inference)

    if not new_inf.empty:
        if not old_inf.empty and "model" in old_inf.columns:
            old_inf = old_inf[~old_inf["model"].isin(models_ran)].copy()
        merged_inf = pd.concat([old_inf, new_inf], ignore_index=True)
    else:
        merged_inf = old_inf.copy()

    if not merged_inf.empty:
        merged_inf = merged_inf.drop_duplicates(subset=["model"], keep="last")
        merged_inf.to_csv(inf_csv, index=False)
    print(f"[OK] Master inference-time results saved: {inf_csv}")

    return merged, merged_cw, merged_inf


def main():
    parser = argparse.ArgumentParser(
        description="Unified Cross-Representation Benchmark: ResNet-50, ViT-B/16, DINOv3, CLIP, EVA-CLIP, SigLIP-2, Qwen3-VL-8B"
    )
    parser.add_argument("--classes", type=str, default="all",
                        help="'all', 'furniture', or comma-separated class names")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto'")

    parser.add_argument("--skip-supervised", action="store_true",
                        help="Skip ResNet-50, ViT-B/16 and DINOv3")
    parser.add_argument("--skip-clip", action="store_true",
                        help="Skip CLIP ViT-B/32")
    parser.add_argument("--skip-eva-clip", action="store_true",
                        help="Skip EVA-CLIP")
    parser.add_argument("--skip-siglip2", action="store_true",
                        help="Skip SigLIP-2")
    parser.add_argument("--skip-qwen", action="store_true",
                        help="Skip Qwen3-VL-8B")

    parser.add_argument("--qwen-limit-per-rep", type=int, default=None,
                        help="Optional maximum Qwen images per representation")
    parser.add_argument("--no-qwen-resume", action="store_true",
                        help="Do not resume Qwen from results/Qwen3_VL_8B_predictions.csv")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("UNIFIED BTP REPRESENTATION-INVARIANT OBJECT RECOGNITION BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Classes
    if args.classes.lower() == "all":
        classes = sorted([d.name for d in ORIGINAL_DIR.iterdir() if d.is_dir()])
    elif args.classes.lower() == "furniture":
        classes = ["chair", "table"]
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Evaluating {len(classes)} classes: {', '.join(classes)}")
    print(f"Representations: {', '.join(REPRESENTATIONS)}\n")

    training_setups, test_sets, class_to_idx = build_data_splits(classes)

    print("Data splits created:")
    for rep, recs in test_sets.items():
        print(f"  Test [{rep}]: {len(recs)} images")
    for setup, recs in training_setups.items():
        print(f"  Train [{setup}]: {len(recs)} images")

    all_results = []
    all_classwise = []
    all_inference = []
    models_ran = []

    # 1. ResNet-50, ViT-B/16, DINOv3
    if not args.skip_supervised:
        for model_name in ["ResNet-50", "ViT-B/16", "DINOv3"]:
            res, cw_res, inf_res = train_and_eval_model(
                model_name=model_name,
                classes=classes,
                training_setups=training_setups,
                test_sets=test_sets,
                device=device,
                class_to_idx=class_to_idx,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
            all_results.extend(res)
            all_classwise.extend(cw_res)
            all_inference.extend(inf_res)
            models_ran.append(model_name)

    # 2. CLIP ViT-B/32
    if not args.skip_clip:
        clip_res, clip_cw, clip_inf = evaluate_zero_shot_clip(
            model_name="CLIP ViT-B/32",
            clip_model_name="ViT-B-32",
            pretrained_tag="openai",
            classes=classes,
            test_sets=test_sets,
            training_setups=training_setups,
            device=device,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size
        )
        all_results.extend(clip_res)
        all_classwise.extend(clip_cw)
        all_inference.extend(clip_inf)
        if clip_res:
            models_ran.append("CLIP ViT-B/32")

    # 3. EVA-CLIP
    if not args.skip_eva_clip:
        eva_res, eva_cw, eva_inf = evaluate_zero_shot_clip(
            model_name="EVA-CLIP (EVA02-B/16)",
            clip_model_name="EVA02-B-16",
            pretrained_tag="merged2b_s8b_b131k",
            classes=classes,
            test_sets=test_sets,
            training_setups=training_setups,
            device=device,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size
        )
        all_results.extend(eva_res)
        all_classwise.extend(eva_cw)
        all_inference.extend(eva_inf)
        if eva_res:
            models_ran.append("EVA-CLIP (EVA02-B/16)")

    # 4. SigLIP-2
    if not args.skip_siglip2:
        siglip2_res, siglip2_cw, siglip2_inf = evaluate_zero_shot_siglip2(
            model_name="SigLIP-2 Base",
            model_id="google/siglip2-base-patch16-224",
            classes=classes,
            test_sets=test_sets,
            training_setups=training_setups,
            device=device,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size
        )
        all_results.extend(siglip2_res)
        all_classwise.extend(siglip2_cw)
        all_inference.extend(siglip2_inf)
        if siglip2_res:
            models_ran.append("SigLIP-2 Base")

    # 5. Qwen3-VL-8B
    if not args.skip_qwen:
        # Qwen uses the same deterministic split, but keeps true labels as strings.
        qwen_full_test_sets = build_qwen_test_sets(classes, train_ratio=0.8, seed=42)
        qwen_test_sets = {
            rep: qwen_full_test_sets[rep]
            for rep in REPRESENTATIONS
            if rep in qwen_full_test_sets
        }

        qwen_res, qwen_cw, qwen_inf = evaluate_qwen(
            classes=classes,
            test_sets=qwen_test_sets,
            device=device,
            limit_per_rep=args.qwen_limit_per_rep,
            resume=not args.no_qwen_resume,
        )
        all_results.extend(qwen_res)
        all_classwise.extend(qwen_cw)
        all_inference.extend(qwen_inf)
        if qwen_res:
            models_ran.append(QWEN_MODEL_NAME)

    # Safe merge into the master CSVs.
    df_all, df_cw, df_inf = merge_and_save_master_results(
        all_results,
        all_classwise,
        all_inference,
        models_ran
    )

    # Create the requested clean Class x Variant x Model comparison.
    # Create clean Class x Variant x Model comparison CSV.
    # Uses Train_All_Combined for supervised models and Zero-Shot for VLMs.
    comparison_csv = RESULTS_DIR / "class_variant_model_comparison.csv"
    comparison_models = [
        "ResNet-50", "ViT-B/16", "DINOv3",
        "CLIP ViT-B/32", "EVA-CLIP (EVA02-B/16)",
        "SigLIP-2 Base", "Qwen3-VL-8B"
    ]
    comparison_setups = {
        "ResNet-50": "Train_All_Combined",
        "ViT-B/16": "Train_All_Combined",
        "DINOv3": "Train_All_Combined",
        "CLIP ViT-B/32": "Zero-Shot",
        "EVA-CLIP (EVA02-B/16)": "Zero-Shot",
        "SigLIP-2 Base": "Zero-Shot",
        "Qwen3-VL-8B": "Zero-Shot",
    }

    if not df_cw.empty:
        comparison = df_cw.copy()
        comparison = comparison[
            comparison["model"].isin(comparison_models)
        ].copy()
        comparison = comparison[
            comparison.apply(
                lambda r: r["training_setup"] == comparison_setups.get(r["model"], ""),
                axis=1
            )
        ].copy()

        # Accuracy is the performance measure in this compact comparison table.
        comparison["accuracy_pct"] = comparison["accuracy"] * 100.0
        comparison["model"] = comparison["model"].replace({
            "EVA-CLIP (EVA02-B/16)": "EVA-CLIP",
            "SigLIP-2 Base": "SigLIP-2",
        })

        comparison = comparison.pivot_table(
            index=["class_name", "test_representation"],
            columns="model",
            values="accuracy_pct",
            aggfunc="first"
        ).reset_index()

        comparison = comparison.rename(columns={
            "class_name": "class",
            "test_representation": "variant",
        })

        desired_columns = [
            "class", "variant", "ResNet-50", "ViT-B/16", "DINOv3",
            "CLIP ViT-B/32", "EVA-CLIP", "SigLIP-2", "Qwen3-VL-8B"
        ]
        comparison = comparison.reindex(
            columns=[c for c in desired_columns if c in comparison.columns]
        )

        representation_order = {rep: i for i, rep in enumerate(REPRESENTATIONS)}
        class_order = {c: i for i, c in enumerate(classes)}
        comparison["_class_order"] = comparison["class"].map(class_order).fillna(999)
        comparison["_variant_order"] = comparison["variant"].map(representation_order).fillna(999)
        comparison = comparison.sort_values(
            ["_class_order", "_variant_order"], kind="stable"
        ).drop(columns=["_class_order", "_variant_order"])

        comparison.to_csv(comparison_csv, index=False)
        print(f"[OK] Class-variant-model comparison saved: {comparison_csv}")
    else:
        print("[WARN] Could not create class-variant-model comparison: no classwise results available.")

    # Print consolidated matrix.
    print("\n" + "=" * 80)
    print("CONSOLIDATED SUMMARY MATRIX — MACRO ACCURACY (%)")
    print("=" * 80)

    if not df_all.empty and "class_name" in df_all.columns:
        df_macro = df_all[df_all["class_name"] == "ALL"].copy()
        if not df_macro.empty:
            summary_pivot = df_macro.pivot_table(
                index=["model", "training_setup"],
                columns="test_representation",
                values="accuracy"
            ) * 100
            ordered_cols = [c for c in REPRESENTATIONS if c in summary_pivot.columns]
            summary_pivot = summary_pivot[ordered_cols]
            print(summary_pivot.round(2).to_string())

    if not df_inf.empty:
        print("\n" + "=" * 95)
        print("MODEL-WISE INFERENCE TIME & COMPLEXITY BENCHMARKS")
        print("=" * 95)
        cols = ["model", "architecture_type", "avg_latency_ms", "throughput_fps"]
        available = [c for c in cols if c in df_inf.columns]
        print(df_inf[available].to_string(index=False))

    print("\n" + "=" * 80)
    print("[OK] Unified benchmark completed.")
    print(f"[OK] Models run this execution: {models_ran}")
    print("[OK] Existing results for skipped models were preserved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
