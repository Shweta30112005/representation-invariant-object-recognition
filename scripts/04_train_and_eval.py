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
    parser.add_argument("--skip-siglip2", action="store_true", help="Skip SigLIP-2 zero-shot evaluation")
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
    all_classwise = []
    all_inference = []

    # 1. Supervised Models (Trained on each variant, evaluated on ALL variants)
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

    # 2. CLIP ViT-B/32 Zero-Shot
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

    # 3. EVA-CLIP Zero-Shot
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

    # 4. SigLIP-2 Zero-Shot
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

    # 1. Save Primary Classification Results (Full Class-wise + Macro ALL)
    df_all = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "classification_results.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n[OK] Classification results (class-wise & summary) saved to: {out_csv}")

    # 2. Save Dedicated Class-wise Metrics (individual classes only)
    if all_classwise:
        df_cw = pd.DataFrame(all_classwise)
        cw_csv = RESULTS_DIR / "classwise_metrics.csv"
        df_cw.to_csv(cw_csv, index=False)
        print(f"[OK] Dedicated class-wise metrics saved to: {cw_csv}")

    # 3. Save Dedicated Macro Summary Results (ALL rows only)
    df_macro = df_all[df_all["class_name"] == "ALL"].copy() if "class_name" in df_all.columns else df_all.copy()
    if not df_macro.empty:
        macro_csv = RESULTS_DIR / "summary_metrics.csv"
        df_macro.to_csv(macro_csv, index=False)
        print(f"[OK] Summary macro metrics saved to: {macro_csv}")

    # 4. Save Model-Wise Inference Time Benchmarks
    if all_inference:
        df_inf = pd.DataFrame(all_inference)
        # Drop duplicates by model if any exist
        df_inf = df_inf.drop_duplicates(subset=["model"], keep="first")
        inf_csv = RESULTS_DIR / "inference_time.csv"
        df_inf.to_csv(inf_csv, index=False)
        print(f"[OK] Model-wise inference time benchmarks saved to: {inf_csv}")

        # Print model-wise inference time summary table
        print(f"\n{'='*95}")
        print("MODEL-WISE INFERENCE TIME & COMPLEXITY BENCHMARKS")
        print(f"{'='*95}")
        print(f"{'Model':<24} {'Type':<18} {'Avg Latency (ms)':>18} {'Throughput (FPS)':>18} {'Device':<14}")
        print("-" * 95)
        for _, row in df_inf.iterrows():
            arch = str(row.get("architecture_type", "N/A"))
            lat = float(row.get("avg_latency_ms", row.get("avg_time_ms", 0)))
            fps = float(row.get("throughput_fps", row.get("throughput_img_per_sec", 0)))
            dev = str(row.get("device", "cuda"))
            print(f"{row['model']:<24} {arch:<18} {lat:>15.2f} ms {fps:>15.1f} img/s {dev:<14}")
        print("=" * 95)

    print("\n" + "=" * 70)
    print("CONSOLIDATED SUMMARY MATRIX")
    print("=" * 70)

    # Print summary pivot for all models using macro ALL rows
    if not df_macro.empty:
        summary_pivot = df_macro.pivot_table(
            index=["model", "training_setup"],
            columns="test_representation",
            values="accuracy"
        ) * 100
        ordered_cols = [c for c in REPRESENTATIONS if c in summary_pivot.columns]
        summary_pivot = summary_pivot[ordered_cols]
        print(summary_pivot.round(2).to_string())

    print("\n[OK] All training, class-wise evaluations, and inference timing completed successfully!")


if __name__ == "__main__":
    main()
