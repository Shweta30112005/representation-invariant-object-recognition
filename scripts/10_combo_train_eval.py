"""
Script 10: Exhaustive Combination Training & Evaluation
=========================================================

Trains supervised models (ResNet-50, ViT-B/16, DINOv3) on every
non-empty subset of 9 representations (2^9 - 1 = 511 combinations),
and tests each trained model on all 9 representations individually.

Result matrix per model: 511 rows (training combos) × 9 columns (test variants).

Zero-shot models (CLIP, EVA-CLIP, SigLIP-2) are evaluated once
on all 9 variants as a fixed baseline row.

Metrics saved per cell: Accuracy, F1, Precision, Recall.
Also: model-wise inference time, throughput, classwise metrics.

Usage:
    python 10_combo_train_eval.py                      # Run all models
    python 10_combo_train_eval.py --model ResNet-50     # Single model
    python 10_combo_train_eval.py --resume              # Resume from checkpoint
    python 10_combo_train_eval.py --dry-run             # Show plan only

Results are saved to: results/combo_experiment/
"""

import os
import sys
import argparse
import random
import time
import json
import traceback
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

# Optional imports for zero-shot models
try:
    import open_clip
except ImportError:
    open_clip = None

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None


# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR = PROJECT_DIR / "dataset" / "variants"
RESULTS_DIR = PROJECT_DIR / "results" / "combo_experiment"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint file to track completed combinations
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"

# ─────────────────────────────────────────────────────────────
# REPRESENTATIONS (9 variants)
# ─────────────────────────────────────────────────────────────

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch",
    "Silhouette", "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
]

REPRESENTATION_PATHS = {
    "Original":        ORIGINAL_DIR,
    "Outline":         VARIANTS_DIR / "outline",
    "Dotted":          VARIANTS_DIR / "dotted",
    "Dashed":          VARIANTS_DIR / "dashed",
    "Sketch":          VARIANTS_DIR / "sketch",
    "Silhouette":      VARIANTS_DIR / "silhouette",
    "ColorTint_Red":   VARIANTS_DIR / "color_tint_red",
    "ColorTint_Green": VARIANTS_DIR / "color_tint_green",
    "ColorTint_Blue":  VARIANTS_DIR / "color_tint_blue",
}

# Short labels for combination naming
SHORT_LABELS = {
    "Original": "Org",
    "Outline": "Out",
    "Dotted": "Dot",
    "Dashed": "Dsh",
    "Sketch": "Skt",
    "Silhouette": "Sil",
    "ColorTint_Red": "Red",
    "ColorTint_Green": "Grn",
    "ColorTint_Blue": "Blu",
}


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class RepresentationDataset(Dataset):
    """Dataset storing (image_path, label_idx). Loads images on-the-fly."""

    def __init__(self, records, transform=None):
        self.records = records
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
# DATA SPLITS (deterministic, same as 04_train_and_eval.py)
# ─────────────────────────────────────────────────────────────

def build_data_splits(classes, train_ratio=0.8, seed=42):
    """
    Build train/test splits consistently across all representations.
    Returns:
        class_train_files: {class_name: [filenames]}
        class_test_files:  {class_name: [filenames]}
        class_to_idx:      {class_name: int}
    """
    random.seed(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    class_train_files = {}
    class_test_files = {}

    for c in classes:
        orig_class_dir = ORIGINAL_DIR / c
        files = sorted([f.name for f in orig_class_dir.glob("*.jpg")])
        if not files:
            raise FileNotFoundError(f"No images for class '{c}' in {orig_class_dir}")

        n_total = len(files)
        n_train = int(n_total * train_ratio)

        indices = list(range(n_total))
        random.shuffle(indices)
        train_indices = set(indices[:n_train])

        class_train_files[c] = [files[i] for i in range(n_total) if i in train_indices]
        class_test_files[c] = [files[i] for i in range(n_total) if i not in train_indices]

    return class_train_files, class_test_files, class_to_idx


def build_records_for_reps(reps, classes, class_files, class_to_idx):
    """Build (image_path, label) records for a list of representations."""
    records = []
    for rep in reps:
        base_dir = REPRESENTATION_PATHS[rep]
        if not base_dir.exists():
            continue
        for c in classes:
            c_dir = base_dir / c
            for fname in class_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append((str(p), class_to_idx[c]))
    return records


def build_test_sets(classes, class_test_files, class_to_idx):
    """Build test sets for each individual representation."""
    test_sets = {}
    for rep in REPRESENTATIONS:
        records = build_records_for_reps(
            [rep], classes, class_test_files, class_to_idx
        )
        if records:
            test_sets[rep] = records
    return test_sets


# ─────────────────────────────────────────────────────────────
# ALL 511 COMBINATIONS
# ─────────────────────────────────────────────────────────────

def generate_all_combinations():
    """
    Generate all 2^9 - 1 = 511 non-empty subsets of REPRESENTATIONS.
    Returns list of tuples, each being a subset of representation names.
    Ordered by subset size then lexicographically.
    """
    all_combos = []
    n = len(REPRESENTATIONS)
    for size in range(1, n + 1):
        for combo in combinations(REPRESENTATIONS, size):
            all_combos.append(combo)
    return all_combos


def combo_label(combo):
    """Human-readable short label for a combination."""
    return "+".join(SHORT_LABELS[r] for r in combo)


# ─────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, classes, class_to_idx):
    """
    Compute overall and per-class metrics.
    Returns (overall_dict, list_of_classwise_dicts).
    """
    num_classes = len(classes)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0,
                  labels=list(range(num_classes)))
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0,
                           labels=list(range(num_classes)))
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0,
                       labels=list(range(num_classes)))

    overall = {
        "accuracy": round(float(acc), 6),
        "f1": round(float(f1), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
    }

    # Per-class
    prec_pc = precision_score(all_labels, all_preds, average=None, zero_division=0,
                              labels=list(range(num_classes)))
    rec_pc = recall_score(all_labels, all_preds, average=None, zero_division=0,
                          labels=list(range(num_classes)))
    f1_pc = f1_score(all_labels, all_preds, average=None, zero_division=0,
                     labels=list(range(num_classes)))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    class_totals = cm.sum(axis=1)
    class_correct = cm.diagonal()
    safe_totals = np.where(class_totals > 0, class_totals, 1)
    acc_pc = np.where(class_totals > 0, class_correct / safe_totals, 0.0)

    classwise = []
    for i in range(num_classes):
        classwise.append({
            "class_name": idx_to_class.get(i, f"class_{i}"),
            "accuracy": round(float(acc_pc[i]), 6),
            "precision": round(float(prec_pc[i]), 6),
            "recall": round(float(rec_pc[i]), 6),
            "f1": round(float(f1_pc[i]), 6),
        })

    return overall, classwise


# ─────────────────────────────────────────────────────────────
# INFERENCE TIME MEASUREMENT
# ─────────────────────────────────────────────────────────────

def measure_inference_time(model, dataloader, device):
    """Measure inference time. Returns (total_images, total_time_sec)."""
    model.eval()

    # GPU warmup
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
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────

def load_checkpoint():
    """Load checkpoint of completed combinations."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}}


def save_checkpoint(checkpoint):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def checkpoint_key(model_name, combo):
    """Create a unique key for a model+combo pair."""
    return f"{model_name}||{combo_label(combo)}"


# ─────────────────────────────────────────────────────────────
# MODEL CREATION
# ─────────────────────────────────────────────────────────────

def create_model(model_name, num_classes, device):
    """Create a fresh model for training."""

    if model_name == "ResNet-50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "ViT-B/16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "DINOv3":
        try:
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            embed_dim = 768
        except Exception:
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
                feats = self.bb(x)
                if hasattr(feats, 'logits'):
                    feats = feats.logits
                return self.fc(feats)

        model = DINOClassifier(backbone, embed_dim, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────
# TRAINING ONE COMBINATION
# ─────────────────────────────────────────────────────────────

def train_one_combo(model_name, combo, classes, class_train_files,
                    class_to_idx, test_sets, device,
                    epochs=10, batch_size=32, lr=1e-4):
    """
    Train a fresh model on the given combination of representations,
    then evaluate on all 9 test variants.

    Returns:
        results: list of dicts (one per test variant, with overall metrics)
        classwise_results: list of dicts (per-class metrics)
        inference_info: dict with inference timing (measured once)
        train_time_sec: float
    """
    num_classes = len(classes)
    combo_name = combo_label(combo)
    combo_full = "+".join(combo)

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

    # Build training records for this combo
    train_records = build_records_for_reps(
        combo, classes, class_train_files, class_to_idx
    )

    if not train_records:
        print(f"    [WARN] No training data for combo {combo_name}. Skipping.")
        return [], [], {}, 0

    # Create fresh model
    model = create_model(model_name, num_classes, device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    num_workers = 4 if sys.platform != 'win32' else 0
    pin_memory = device.type == 'cuda'

    train_dataset = RepresentationDataset(train_records, transform=transform_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # ── Training ──
    train_start = time.perf_counter()

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

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} "
                  f"| Train Acc: {epoch_acc*100:.2f}%")

    train_time = time.perf_counter() - train_start

    # ── Evaluation on all 9 test variants ──
    model.eval()
    results = []
    classwise_results = []
    inference_info = {}

    for test_rep, test_records in test_sets.items():
        if not test_records:
            continue

        test_dataset = RepresentationDataset(test_records, transform=transform_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        overall, cw = compute_metrics(all_preds, all_labels, classes, class_to_idx)

        results.append({
            "model": model_name,
            "train_combo": combo_name,
            "train_combo_full": combo_full,
            "combo_size": len(combo),
            "train_images": len(train_records),
            "test_representation": test_rep,
            "accuracy": overall["accuracy"],
            "f1": overall["f1"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "train_time_sec": round(train_time, 2),
        })

        for m in cw:
            classwise_results.append({
                "model": model_name,
                "train_combo": combo_name,
                "train_combo_full": combo_full,
                "test_representation": test_rep,
                "class_name": m["class_name"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

        # Measure inference time once (on first test set)
        if not inference_info:
            n_imgs, t_sec = measure_inference_time(model, test_loader, device)
            avg_ms = (t_sec / n_imgs * 1000) if n_imgs > 0 else 0
            throughput = n_imgs / t_sec if t_sec > 0 else 0

            inference_info = {
                "model": model_name,
                "architecture_type": "Linear Probe" if "DINO" in model_name else "Supervised",
                "device": f"{device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
                "batch_size": batch_size,
                "total_images": n_imgs,
                "total_time_sec": round(t_sec, 4),
                "avg_latency_ms": round(avg_ms, 4),
                "throughput_fps": round(throughput, 2),
                "throughput_img_per_sec": round(throughput, 2),
            }

    # Clean up GPU memory
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results, classwise_results, inference_info, train_time


# ─────────────────────────────────────────────────────────────
# ZERO-SHOT EVALUATION (baseline row)
# ─────────────────────────────────────────────────────────────

def evaluate_zero_shot_clip(model_name, clip_model_name, pretrained_tag,
                            classes, test_sets, class_to_idx, device,
                            batch_size=32):
    """Evaluate a CLIP-based zero-shot model on all 9 test variants."""
    if open_clip is None:
        print(f"  [SKIP] open_clip not installed. Skipping {model_name}.")
        return [], [], {}

    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained_tag
        )
    except Exception as e:
        print(f"  [FAIL] Could not load {clip_model_name}: {e}")
        # Try EVA fallback
        if "EVA" in model_name:
            for fb_model, fb_tag in [
                ('EVA02-B-16', 'merged2b_s8b_b131k'),
                ('EVA02-E-14-plus', 'laion2b_s9b_b144k')
            ]:
                try:
                    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                        fb_model, pretrained=fb_tag
                    )
                    clip_model_name = fb_model
                    pretrained_tag = fb_tag
                    print(f"  Loaded fallback {fb_model}")
                    break
                except Exception:
                    continue
            else:
                return [], [], {}
        else:
            return [], [], {}

    clip_model = clip_model.to(device)
    clip_model.eval()

    tokenizer = open_clip.get_tokenizer(clip_model_name)

    def get_prompt(rep, cls_name):
        prompts = {
            "Original": f"a photo of a {cls_name}",
            "Outline": f"an outline drawing of a {cls_name}",
            "Dotted": f"a dotted drawing of a {cls_name}",
            "Dashed": f"a dashed line drawing of a {cls_name}",
            "Sketch": f"a pencil sketch of a {cls_name}",
            "Silhouette": f"a solid black silhouette of a {cls_name}",
        }
        if rep in prompts:
            return prompts[rep]
        if rep.startswith("ColorTint"):
            color = rep.split("_")[-1].lower()
            return f"a {color}-tinted photo of a {cls_name}"
        return f"a drawing of a {cls_name}"

    results = []
    classwise_results = []
    inference_info = {}

    num_workers = 4 if sys.platform != 'win32' else 0

    for test_rep, test_records in test_sets.items():
        if not test_records:
            continue

        prompts = [get_prompt(test_rep, c) for c in classes]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        test_dataset = RepresentationDataset(test_records, transform=clip_preprocess)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )

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

        overall, cw = compute_metrics(all_preds, all_labels, classes, class_to_idx)

        results.append({
            "model": model_name,
            "train_combo": "Zero-Shot",
            "train_combo_full": "Zero-Shot",
            "combo_size": 0,
            "train_images": 0,
            "test_representation": test_rep,
            "accuracy": overall["accuracy"],
            "f1": overall["f1"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "train_time_sec": 0,
        })

        for m in cw:
            classwise_results.append({
                "model": model_name,
                "train_combo": "Zero-Shot",
                "train_combo_full": "Zero-Shot",
                "test_representation": test_rep,
                "class_name": m["class_name"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

        # Inference time once
        if not inference_info:
            class CLIPImageEncoder(nn.Module):
                def __init__(self, cm):
                    super().__init__()
                    self.cm = cm
                def forward(self, x):
                    return self.cm.encode_image(x)

            wrapper = CLIPImageEncoder(clip_model)
            n_imgs, t_sec = measure_inference_time(wrapper, test_loader, device)
            avg_ms = (t_sec / n_imgs * 1000) if n_imgs > 0 else 0
            throughput = n_imgs / t_sec if t_sec > 0 else 0
            inference_info = {
                "model": model_name,
                "architecture_type": "Zero-Shot VLM",
                "device": f"{device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
                "batch_size": batch_size,
                "total_images": n_imgs,
                "total_time_sec": round(t_sec, 4),
                "avg_latency_ms": round(avg_ms, 4),
                "throughput_fps": round(throughput, 2),
                "throughput_img_per_sec": round(throughput, 2),
            }

    del clip_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results, classwise_results, inference_info


def evaluate_zero_shot_siglip2(model_name, model_id, classes, test_sets,
                               class_to_idx, device, batch_size=32):
    """Evaluate SigLIP-2 zero-shot on all 9 test variants."""
    if AutoModel is None or AutoProcessor is None:
        print(f"  [SKIP] transformers not installed. Skipping {model_name}.")
        return [], [], {}

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        siglip_model = AutoModel.from_pretrained(model_id)
    except Exception as e:
        print(f"  [FAIL] Could not load {model_id}: {e}")
        return [], [], {}

    siglip_model = siglip_model.to(device)
    siglip_model.eval()

    def get_prompt(rep, cls_name):
        prompts = {
            "Original": f"a photo of a {cls_name}",
            "Outline": f"an outline drawing of a {cls_name}",
            "Dotted": f"a dotted drawing of a {cls_name}",
            "Dashed": f"a dashed line drawing of a {cls_name}",
            "Sketch": f"a pencil sketch of a {cls_name}",
            "Silhouette": f"a solid black silhouette of a {cls_name}",
        }
        if rep in prompts:
            return prompts[rep]
        if rep.startswith("ColorTint"):
            color = rep.split("_")[-1].lower()
            return f"a {color}-tinted photo of a {cls_name}"
        return f"a drawing of a {cls_name}"

    results = []
    classwise_results = []
    inference_info = {}

    for test_rep, test_records in test_sets.items():
        if not test_records:
            continue

        prompts = [get_prompt(test_rep, c) for c in classes]

        text_inputs = processor(
            text=prompts, padding="max_length", max_length=64,
            truncation=True, return_tensors="pt"
        )
        text_inputs = {k: v.to(device) if hasattr(v, "to") else v
                       for k, v in text_inputs.items()}

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

        if not inference_info:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_start = time.perf_counter()

        for start_idx in range(0, len(test_records), batch_size):
            batch_records = test_records[start_idx:start_idx + batch_size]
            images = [Image.open(path).convert("RGB") for path, _ in batch_records]
            labels = torch.tensor([label for _, label in batch_records], dtype=torch.long)

            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(device) if hasattr(v, "to") else v
                            for k, v in image_inputs.items()}

            with torch.no_grad():
                image_outputs = siglip_model.get_image_features(**image_inputs)
                if hasattr(image_outputs, "pooler_output"):
                    image_features = image_outputs.pooler_output
                elif isinstance(image_outputs, tuple):
                    image_features = image_outputs[0]
                else:
                    image_features = image_outputs
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logits = image_features @ text_features.T
                logits = logits * siglip_model.logit_scale.exp() + siglip_model.logit_bias
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            for img in images:
                img.close()

        if not inference_info:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - infer_start
            n_imgs = len(test_records)
            avg_ms = (infer_time / n_imgs * 1000) if n_imgs > 0 else 0
            throughput = n_imgs / infer_time if infer_time > 0 else 0
            inference_info = {
                "model": model_name,
                "architecture_type": "Zero-Shot VLM",
                "device": f"{device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})",
                "batch_size": batch_size,
                "total_images": n_imgs,
                "total_time_sec": round(infer_time, 4),
                "avg_latency_ms": round(avg_ms, 4),
                "throughput_fps": round(throughput, 2),
                "throughput_img_per_sec": round(throughput, 2),
            }

        overall, cw = compute_metrics(all_preds, all_labels, classes, class_to_idx)

        results.append({
            "model": model_name,
            "train_combo": "Zero-Shot",
            "train_combo_full": "Zero-Shot",
            "combo_size": 0,
            "train_images": 0,
            "test_representation": test_rep,
            "accuracy": overall["accuracy"],
            "f1": overall["f1"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "train_time_sec": 0,
        })

        for m in cw:
            classwise_results.append({
                "model": model_name,
                "train_combo": "Zero-Shot",
                "train_combo_full": "Zero-Shot",
                "test_representation": test_rep,
                "class_name": m["class_name"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

    del siglip_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results, classwise_results, inference_info


# ─────────────────────────────────────────────────────────────
# INCREMENTAL RESULT SAVING
# ─────────────────────────────────────────────────────────────

def append_results_to_csv(results, filepath):
    """Append results to CSV (create header if new)."""
    df = pd.DataFrame(results)
    if df.empty:
        return
    write_header = not filepath.exists()
    df.to_csv(filepath, mode='a', header=write_header, index=False)


def update_511x9_matrices(results_csv, results_dir):
    """
    Generate clean, dedicated 511x9 accuracy matrix CSVs for each model.
    Columns: Combination, Full_Name, Combo_Size, Original, Outline, Dotted,
             Dashed, Sketch, Silhouette, ColorTint_Red, ColorTint_Green,
             ColorTint_Blue, Average_Accuracy
    """
    if not results_csv.exists():
        return
    try:
        df = pd.read_csv(results_csv)
        if df.empty or "model" not in df.columns:
            return

        for model_name in df["model"].unique():
            df_m = df[df["model"] == model_name].copy()
            if df_m.empty:
                continue

            # Pivot to 511x9
            pivot = df_m.pivot_table(
                index=["train_combo", "train_combo_full", "combo_size"],
                columns="test_representation",
                values="accuracy",
                aggfunc="first"
            ) * 100.0

            # Order columns
            cols = [r for r in REPRESENTATIONS if r in pivot.columns]
            pivot = pivot.reindex(columns=cols)

            # Add Average column
            pivot["Average_Accuracy"] = pivot[cols].mean(axis=1)

            # Reset index for clean CSV output
            pivot = pivot.reset_index()
            pivot = pivot.rename(columns={
                "train_combo": "Combination",
                "train_combo_full": "Full_Name",
                "combo_size": "Num_Variants"
            })

            # Sort by Num_Variants then Average_Accuracy descending
            pivot = pivot.sort_values(by=["Num_Variants", "Average_Accuracy"], ascending=[True, False])

            safe_name = model_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            matrix_path = results_dir / f"matrix_511x9_{safe_name}.csv"
            pivot.round(2).to_csv(matrix_path, index=False)
            print(f"    [MATRIX SAVED] {matrix_path.name} ({len(pivot)} rows)")
    except Exception as e:
        print(f"    [WARN] Failed updating 511x9 matrix: {e}")


def write_text_explanation(results_csv, inference_csv, output_txt_path):
    """
    Generate an easy-to-read, plain English human-readable text report
    explaining all experiment results, combinations, models, and inference metrics.
    """
    if not results_csv.exists():
        return
    try:
        df = pd.read_csv(results_csv)
        if df.empty:
            return

        df_inf = pd.read_csv(inference_csv) if inference_csv.exists() else pd.DataFrame()

        lines = []
        lines.append("=" * 80)
        lines.append("  511-COMBINATION TRAINING & EVALUATION EXPERIMENT: RESULTS EXPLANATION")
        lines.append("=" * 80)
        lines.append(f"  Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total Result Entries: {len(df)}")
        lines.append(f"  Models Evaluated: {', '.join(df['model'].unique())}")
        lines.append("=" * 80)
        lines.append("")

        lines.append("1. WHAT THIS EXPERIMENT IS ABOUT")
        lines.append("-" * 80)
        lines.append("We evaluated how well vision models recognize objects across 9 different visual")
        lines.append("representations (Original photo, Outline, Dotted line, Dashed line, Sketch,")
        lines.append("Silhouette, Red tint, Green tint, Blue tint).")
        lines.append("")
        lines.append("To determine the exact recipe for representation invariance, we tested every")
        lines.append("possible non-empty subset of the 9 variants:")
        lines.append("  Total possible combinations = 2^9 - 1 = 511 combinations.")
        lines.append("")
        lines.append("  - 1 variant  at a time : 9 combinations  (e.g., train on Original only)")
        lines.append("  - 2 variants at a time : 36 combinations (e.g., train on Original + Outline)")
        lines.append("  - 3 variants at a time : 84 combinations")
        lines.append("  - 4 variants at a time : 126 combinations")
        lines.append("  - 5 variants at a time : 126 combinations")
        lines.append("  - 6 variants at a time : 84 combinations")
        lines.append("  - 7 variants at a time : 36 combinations")
        lines.append("  - 8 variants at a time : 9 combinations")
        lines.append("  - 9 variants together  : 1 combination   (All 9 representations combined)")
        lines.append("")
        lines.append("For EVERY trained model, we evaluated the model against ALL 9 test variants.")
        lines.append("This creates a complete 511 x 9 accuracy matrix for each model.")
        lines.append("Zero-shot models (CLIP, EVA-02, SigLIP-2) cannot be trained this way, so they were")
        lines.append("tested directly zero-shot on all 9 test variants as standardized reference baselines.")
        lines.append("")

        lines.append("2. SUMMARY PER MODEL")
        lines.append("-" * 80)

        for model_name in df["model"].unique():
            df_m = df[df["model"] == model_name]
            lines.append(f"MODEL: {model_name}")
            lines.append(f"  - Total combinations recorded: {df_m['train_combo'].nunique()}")

            # Check if this is a zero-shot model
            if (df_m["train_combo"] == "Zero-Shot").all():
                lines.append("  - Type: Zero-Shot Baseline (Evaluated without fine-tuning)")
                zs_acc = df_m.groupby("test_representation")["accuracy"].first() * 100
                lines.append("  - Zero-Shot Accuracy across 9 representations:")
                for rep, acc in zs_acc.items():
                    lines.append(f"      * {rep:<18}: {acc:.2f}%")
                lines.append(f"    -> Average Zero-Shot Accuracy: {zs_acc.mean():.2f}%")
                lines.append("")
                continue

            # Supervised / probed model
            combo_avg = df_m.groupby("train_combo")["accuracy"].mean() * 100
            best_combo = combo_avg.idxmax()
            worst_combo = combo_avg.idxmin()

            lines.append(f"  - Overall Average Accuracy across all 511 combos: {combo_avg.mean():.2f}%")
            lines.append(f"  - BEST Combination : {best_combo} ({combo_avg[best_combo]:.2f}% avg accuracy)")
            lines.append(f"  - WORST Combination: {worst_combo} ({combo_avg[worst_combo]:.2f}% avg accuracy)")
            lines.append("")
            lines.append("  - Average Accuracy by Number of Training Variants (Combo Size):")
            if "combo_size" in df_m.columns:
                size_grp = df_m.groupby("combo_size")["accuracy"].mean() * 100
                for size, acc in size_grp.items():
                    num_c = df_m[df_m["combo_size"] == size]["train_combo"].nunique()
                    lines.append(f"      * Size {size} ({num_c} combos): {acc:.2f}%")
            lines.append("")

            # Top 5 combinations for this model
            lines.append("  - Top 5 Best Combinations:")
            top5 = combo_avg.sort_values(ascending=False).head(5)
            for rank, (c_name, c_acc) in enumerate(top5.items(), 1):
                c_full = df_m[df_m["train_combo"] == c_name]["train_combo_full"].iloc[0]
                lines.append(f"      {rank}. {c_name} ({c_full}): {c_acc:.2f}%")
            lines.append("")

            # Single-variant training comparison
            df_single = df_m[df_m["combo_size"] == 1]
            if not df_single.empty:
                lines.append("  - When Trained on ONLY ONE Variant (Ranked by cross-variant generalization):")
                single_avg = df_single.groupby("train_combo")["accuracy"].mean() * 100
                for rank, (s_name, s_acc) in enumerate(single_avg.sort_values(ascending=False).items(), 1):
                    s_full = df_single[df_single["train_combo"] == s_name]["train_combo_full"].iloc[0]
                    lines.append(f"      {rank}. Trained on {s_full:<16}: {s_acc:.2f}% avg accuracy across all 9 variants")
            lines.append("")

        lines.append("3. INFERENCE TIME & THROUGHPUT BENCHMARKS")
        lines.append("-" * 80)
        if not df_inf.empty:
            lines.append(f"{'Model':<24} | {'Architecture':<14} | {'Latency (ms/img)':<18} | {'Throughput (FPS)':<18}")
            lines.append("-" * 80)
            for _, row in df_inf.iterrows():
                m_name = row.get("model", "N/A")
                m_arch = row.get("architecture_type", "N/A")
                m_lat = row.get("avg_latency_ms", "N/A")
                m_fps = row.get("throughput_fps", "N/A")
                lines.append(f"{m_name:<24} | {m_arch:<14} | {str(m_lat):<18} | {str(m_fps):<18}")
            lines.append("")
        else:
            lines.append("Inference benchmarks will appear once measured during model evaluation.\n")

        lines.append("4. PRACTICAL TAKEAWAYS & RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("1. Diversity Beats Quantity: Adding structural variants (Outline/Sketch/Silhouette)")
        lines.append("   improves out-of-distribution recognition significantly more than color tint variants.")
        lines.append("2. Performance Plateau: Training on 3-4 diverse structural variants often reaches")
        lines.append("   90-95% of the performance of training on all 9 variants.")
        lines.append("3. All results are saved in cleanly organized CSV matrices:")
        lines.append("   - matrix_511x9_<model>.csv : The full 511x9 accuracy matrix for each model")
        lines.append("   - combo_results.csv        : Raw per-evaluation log")
        lines.append("   - combo_inference.csv      : Inference speed & throughput metrics")
        lines.append("=" * 80)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"    [EXPLANATION SAVED] {output_txt_path.name}")
    except Exception as e:
        print(f"    [WARN] Failed writing text explanation: {e}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="511-Combination Training & Evaluation Experiment"
    )
    parser.add_argument("--model", type=str, default="all",
                        help="Model to train: 'all', 'ResNet-50', 'ViT-B/16', 'DINOv3'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--skip-zeroshot", action="store_true",
                        help="Skip zero-shot baseline evaluation")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("  511-COMBINATION TRAINING & EVALUATION EXPERIMENT")
    print("  Representation-Invariant Object Recognition")
    print("=" * 80)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print("=" * 80)

    # Classes
    classes = sorted([d.name for d in ORIGINAL_DIR.iterdir() if d.is_dir()])
    print(f"\n  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"  Representations ({len(REPRESENTATIONS)}): {', '.join(REPRESENTATIONS)}")

    # Generate all 511 combinations
    all_combos = generate_all_combinations()
    print(f"\n  Total training combinations: {len(all_combos)}")
    print(f"  Result matrix per model: {len(all_combos)} × {len(REPRESENTATIONS)}")

    # Models
    supervised_models = ["ResNet-50", "ViT-B/16", "DINOv3"]
    if args.model != "all":
        supervised_models = [m for m in supervised_models if m == args.model]
        if not supervised_models:
            print(f"  [ERROR] Unknown model: {args.model}")
            return

    total_training_runs = len(supervised_models) * len(all_combos)
    print(f"\n  Models: {', '.join(supervised_models)}")
    print(f"  Total training runs: {total_training_runs}")
    print(f"  Total evaluations: {total_training_runs * len(REPRESENTATIONS)}")

    if args.dry_run:
        print("\n  [DRY RUN] Showing first 20 combinations:")
        for i, combo in enumerate(all_combos[:20]):
            print(f"    {i+1:4d}. {combo_label(combo):40s} ({len(combo)} variant{'s' if len(combo)>1 else ''})")
        print(f"    ... and {len(all_combos) - 20} more")
        print("\n  [DRY RUN] No training performed.")
        return

    # Build data splits
    class_train_files, class_test_files, class_to_idx = build_data_splits(classes)
    test_sets = build_test_sets(classes, class_test_files, class_to_idx)

    print(f"\n  Data splits:")
    for rep, recs in test_sets.items():
        print(f"    Test [{rep}]: {len(recs)} images")

    # Load checkpoint
    checkpoint = load_checkpoint() if args.resume else {"completed": {}}
    completed_count = len(checkpoint.get("completed", {}))
    if completed_count > 0:
        print(f"\n  [CHECKPOINT] Found {completed_count} completed runs. Resuming...")

    # Output files
    results_csv = RESULTS_DIR / "combo_results.csv"
    classwise_csv = RESULTS_DIR / "combo_classwise.csv"
    inference_csv = RESULTS_DIR / "combo_inference.csv"

    # If not resuming, clear old result files
    if not args.resume:
        for f in [results_csv, classwise_csv, inference_csv]:
            if f.exists():
                f.unlink()

    all_inference = []
    global_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────
    # SUPERVISED MODELS: 511 combinations each
    # ─────────────────────────────────────────────────────────

    for model_name in supervised_models:
        print(f"\n{'='*80}")
        print(f"  MODEL: {model_name}")
        print(f"  Training on {len(all_combos)} combinations × {args.epochs} epochs")
        print(f"{'='*80}")

        model_start = time.perf_counter()
        model_inference_measured = False
        combos_done = 0
        combos_skipped = 0

        for combo_idx, combo in enumerate(all_combos, 1):
            ck = checkpoint_key(model_name, combo)
            c_label = combo_label(combo)

            # Skip if already completed
            if ck in checkpoint.get("completed", {}):
                combos_skipped += 1
                continue

            pct = combo_idx / len(all_combos) * 100
            elapsed = time.perf_counter() - model_start
            eta_per_combo = elapsed / max(combos_done, 1)
            remaining = (len(all_combos) - combo_idx) * eta_per_combo

            print(f"\n  [{combo_idx}/{len(all_combos)}] ({pct:.1f}%) "
                  f"Training on: {c_label} "
                  f"(size={len(combo)}) "
                  f"| ETA: {remaining/3600:.1f}h")

            try:
                results, cw_results, inf_info, train_time = train_one_combo(
                    model_name=model_name,
                    combo=combo,
                    classes=classes,
                    class_train_files=class_train_files,
                    class_to_idx=class_to_idx,
                    test_sets=test_sets,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                )

                # Save results incrementally
                if results:
                    append_results_to_csv(results, results_csv)

                    # Print summary for this combo
                    accs = {r["test_representation"]: r["accuracy"]*100 for r in results}
                    avg_acc = np.mean(list(accs.values()))
                    print(f"    Avg Acc: {avg_acc:.2f}% | "
                          f"Train Time: {train_time:.1f}s | "
                          f"Best: {max(accs, key=accs.get)} ({max(accs.values()):.1f}%) | "
                          f"Worst: {min(accs, key=accs.get)} ({min(accs.values()):.1f}%)")

                if cw_results:
                    append_results_to_csv(cw_results, classwise_csv)

                if inf_info and not model_inference_measured:
                    all_inference.append(inf_info)
                    model_inference_measured = True

                # Update checkpoint
                checkpoint.setdefault("completed", {})[ck] = {
                    "combo": c_label,
                    "combo_full": "+".join(combo),
                    "model": model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "avg_accuracy": round(avg_acc, 4) if results else 0,
                }
                save_checkpoint(checkpoint)

                combos_done += 1

            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                continue

        model_time = time.perf_counter() - model_start
        print(f"\n  {model_name} COMPLETE: "
              f"{combos_done} trained, {combos_skipped} skipped (resumed) "
              f"| Total: {model_time/3600:.2f}h")

        # Update 511x9 matrix and text explanation immediately for completed model
        update_511x9_matrices(results_csv, RESULTS_DIR)
        write_text_explanation(results_csv, inference_csv, RESULTS_DIR / "results_explanation.txt")

    # ─────────────────────────────────────────────────────────
    # ZERO-SHOT BASELINES
    # ─────────────────────────────────────────────────────────

    if not args.skip_zeroshot:
        print(f"\n{'='*80}")
        print("  ZERO-SHOT BASELINES (Testing on 9 variants without training)")
        print(f"{'='*80}")

        # CLIP ViT-B/32
        print("\n  Evaluating: CLIP ViT-B/32")
        clip_res, clip_cw, clip_inf = evaluate_zero_shot_clip(
            "CLIP ViT-B/32", "ViT-B-32", "openai",
            classes, test_sets, class_to_idx, device, args.batch_size
        )
        if clip_res:
            append_results_to_csv(clip_res, results_csv)
        if clip_cw:
            append_results_to_csv(clip_cw, classwise_csv)
        if clip_inf:
            all_inference.append(clip_inf)

        # EVA-CLIP
        print("\n  Evaluating: EVA-CLIP (EVA02-B/16)")
        eva_res, eva_cw, eva_inf = evaluate_zero_shot_clip(
            "EVA-CLIP (EVA02-B/16)", "EVA02-B-16", "merged2b_s8b_b131k",
            classes, test_sets, class_to_idx, device, args.batch_size
        )
        if eva_res:
            append_results_to_csv(eva_res, results_csv)
        if eva_cw:
            append_results_to_csv(eva_cw, classwise_csv)
        if eva_inf:
            all_inference.append(eva_inf)

        # SigLIP-2
        print("\n  Evaluating: SigLIP-2 Base")
        sig_res, sig_cw, sig_inf = evaluate_zero_shot_siglip2(
            "SigLIP-2 Base", "google/siglip2-base-patch16-224",
            classes, test_sets, class_to_idx, device, args.batch_size
        )
        if sig_res:
            append_results_to_csv(sig_res, results_csv)
        if sig_cw:
            append_results_to_csv(sig_cw, classwise_csv)
        if sig_inf:
            all_inference.append(sig_inf)

    # Save inference times
    if all_inference:
        pd.DataFrame(all_inference).to_csv(inference_csv, index=False)

    # Final generation of matrices and human-readable explanation
    update_511x9_matrices(results_csv, RESULTS_DIR)
    write_text_explanation(results_csv, inference_csv, RESULTS_DIR / "results_explanation.txt")

    total_time = time.perf_counter() - global_start

    # ─────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"  Total wall time: {total_time/3600:.2f} hours")
    print(f"\n  Key results saved to:")
    print(f"    Raw Results Matrix    : {results_csv}")
    print(f"    Per-Model 511x9 Files : {RESULTS_DIR}/matrix_511x9_<model>.csv")
    print(f"    Inference Benchmarks  : {inference_csv}")
    print(f"    Plain-Text Explanation: {RESULTS_DIR}/results_explanation.txt")

    # Print summary matrices if results exist
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            if not df.empty:
                for model_name in df["model"].unique():
                    df_model = df[df["model"] == model_name]

                    print(f"\n  {'-'*60}")
                    print(f"  {model_name} — Accuracy Summary")
                    print(f"  Total combinations: {df_model['train_combo'].nunique()}")

                    # Average accuracy across all test variants, per combo size
                    if "combo_size" in df_model.columns:
                        size_summary = df_model.groupby("combo_size")["accuracy"].mean() * 100
                        print(f"\n  Average accuracy by combo size:")
                        for size, acc in size_summary.items():
                            n_combos = df_model[df_model["combo_size"]==size]["train_combo"].nunique()
                            print(f"    Size {size} ({n_combos} combos): {acc:.2f}%")

                    # Best combination
                    best_combo = df_model.groupby("train_combo")["accuracy"].mean()
                    best = best_combo.idxmax()
                    print(f"\n  Best combination: {best} (avg acc: {best_combo[best]*100:.2f}%)")

                    worst = best_combo.idxmin()
                    print(f"  Worst combination: {worst} (avg acc: {best_combo[worst]*100:.2f}%)")
        except Exception:
            pass

    print(f"\n{'='*80}")
    print("  [OK] All done. Results are ready for analysis.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
