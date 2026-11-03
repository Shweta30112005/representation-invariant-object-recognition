"""
Script 9: Resolution / Input Size Effect Analysis
══════════════════════════════════════════════════

Evaluates how input resolution affects model performance across all representations.

Strategy:
  1. For each target resolution (32, 64, 112, 224, 336, 448):
     - Resize test images DOWN to target resolution
     - Then resize BACK UP to 224×224 (model's expected input)
     - This simulates the information loss of lower resolution inputs
  2. Run inference with each model on each representation at each resolution
  3. Compute accuracy, precision, recall, F1
  4. Generate comprehensive visualizations

Models: ResNet-50, ViT-B/16, DINOv3, SigLIP-2
Representations: Original, Outline, Dotted, Dashed, Sketch, Silhouette, ColorTint_*
Resolutions: 32, 64, 112, 224, 336, 448

Outputs:
  results/resolution_analysis/resolution_results.csv
  results/resolution_analysis/plots/
  results/resolution_analysis/tables/resolution_summary.html
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from transformers import AutoModel, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent.parent
ORIGINAL_DIR  = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR  = PROJECT_DIR / "dataset" / "variants"
RESULTS_DIR   = PROJECT_DIR / "results"
RES_DIR       = RESULTS_DIR / "resolution_analysis"
PLOTS_DIR     = RES_DIR / "plots"
TABLES_DIR    = RES_DIR / "tables"

for d in [RES_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = [
    "Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette",
    "ColorTint_Red", "ColorTint_Green", "ColorTint_Blue"
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

CLASSES = ["car", "cat", "chair", "circle", "cup", "dog",
           "rectangle", "square", "table", "triangle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

RESOLUTIONS = [32, 64, 112, 224, 336, 448]

# Matplotlib style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight"
})


# ─────────────────────────────────────────────────────────────
# Dataset with Resolution Degradation
# ─────────────────────────────────────────────────────────────

class ResolutionDataset(Dataset):
    """
    Dataset that simulates resolution degradation:
    1. Load image
    2. Resize DOWN to target_resolution × target_resolution
    3. Resize BACK UP to 224 × 224
    4. Apply normalization
    """
    def __init__(self, records, target_resolution=224):
        self.records = records  # list of (path, label)
        self.target_res = target_resolution
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")

        # Step 1: Resize down to target resolution
        img = img.resize((self.target_res, self.target_res), Image.BILINEAR)

        # Step 2: Resize back up to 224×224 (model input size)
        if self.target_res != 224:
            img = img.resize((224, 224), Image.BILINEAR)

        # Step 3: Convert to tensor and normalize
        img = transforms.functional.to_tensor(img)
        img = self.normalize(img)

        return img, label


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def get_test_records(train_ratio=0.8, seed=42):
    """Get test records for each representation."""
    random.seed(seed)

    class_test_files = {}
    for c in CLASSES:
        orig_dir = ORIGINAL_DIR / c
        files = sorted([f.name for f in orig_dir.glob("*.jpg")] +
                       [f.name for f in orig_dir.glob("*.png")] +
                       [f.name for f in orig_dir.glob("*.jpeg")])
        if not files:
            continue
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        indices = list(range(n_total))
        random.shuffle(indices)
        train_idx = set(indices[:n_train])
        class_test_files[c] = [files[i] for i in range(n_total) if i not in train_idx]

    # Build train records (for training models)
    train_records = []
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        if not base_dir.exists():
            continue
        for c in CLASSES:
            if c not in class_test_files:
                continue
            c_dir = base_dir / c
            # Use train files
            orig_dir = ORIGINAL_DIR / c
            files = sorted([f.name for f in orig_dir.glob("*.jpg")] +
                           [f.name for f in orig_dir.glob("*.png")])
            n_total = len(files)
            n_train = int(n_total * train_ratio)
            random.seed(seed)
            indices = list(range(n_total))
            random.shuffle(indices)
            train_idx = set(indices[:n_train])
            for i in train_idx:
                if i < len(files):
                    p = c_dir / files[i]
                    if p.exists():
                        train_records.append((str(p), CLASS_TO_IDX[c]))

    # Build test records per representation
    test_records = {}
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        if not base_dir.exists():
            continue
        records = []
        for c in CLASSES:
            if c not in class_test_files:
                continue
            c_dir = base_dir / c
            for fname in class_test_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append((str(p), CLASS_TO_IDX[c]))
        if records:
            test_records[rep] = records

    return train_records, test_records


# ─────────────────────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────────────────────

def build_models(device):
    """Build all model architectures."""
    models_list = []

    # ResNet-50
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    resnet = resnet.to(device)
    models_list.append((resnet, "ResNet-50"))

    # ViT-B/16
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    vit.heads.head = nn.Linear(vit.heads.head.in_features, NUM_CLASSES)
    vit = vit.to(device)
    models_list.append((vit, "ViT-B_16"))

    # DINOv3
    try:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        embed_dim = 768
    except Exception:
        try:
            import timm
            backbone = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0)
            embed_dim = 768
        except Exception:
            backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
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

    dino = DINOClassifier(backbone, embed_dim, NUM_CLASSES)
    dino = dino.to(device)
    models_list.append((dino, "DINOv3"))

    # SigLIP-2
    if HAS_TRANSFORMERS:
        try:
            from transformers import AutoModel
            siglip_bb = AutoModel.from_pretrained("google/siglip2-base-patch16-224", trust_remote_code=True)

            class SigLIP2Classifier(nn.Module):
                def __init__(self, bb, n_cls):
                    super().__init__()
                    self.bb = bb
                    for p in self.bb.parameters():
                        p.requires_grad = False
                    if hasattr(bb, 'config') and hasattr(bb.config, 'vision_config'):
                        hidden = bb.config.vision_config.hidden_size
                    elif hasattr(bb, 'config') and hasattr(bb.config, 'hidden_size'):
                        hidden = bb.config.hidden_size
                    else:
                        hidden = 768
                    self.fc = nn.Linear(hidden, n_cls)

                def forward(self, x):
                    if hasattr(self.bb, 'vision_model'):
                        out = self.bb.vision_model(pixel_values=x)
                        feats = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:, 0]
                    elif hasattr(self.bb, 'get_image_features'):
                        feats = self.bb.get_image_features(pixel_values=x)
                    else:
                        feats = self.bb(x)
                        if hasattr(feats, 'pooler_output'):
                            feats = feats.pooler_output
                    return self.fc(feats)

            siglip = SigLIP2Classifier(siglip_bb, NUM_CLASSES)
            siglip = siglip.to(device)
            models_list.append((siglip, "SigLIP-2"))
        except Exception as e:
            print(f"  [WARN] Could not load SigLIP-2: {e}")

    # CLIP ViT-B/32
    try:
        import open_clip
        clip_m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_m = clip_m.to(device).eval()
        tok = open_clip.get_tokenizer('ViT-B-32')
        prompts = [f"a photo of a {c}" for c in CLASSES]
        text_tokens = tok(prompts).to(device)
        with torch.no_grad():
            t_feats = clip_m.encode_text(text_tokens)
            t_feats /= t_feats.norm(dim=-1, keepdim=True)

        class CLIPClassifier(nn.Module):
            def __init__(self, m, tf):
                super().__init__()
                self.m = m
                self.tf = nn.Parameter(tf, requires_grad=False)
            def forward(self, x):
                f = self.m.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                return 100.0 * f @ self.tf.T

        models_list.append((CLIPClassifier(clip_m, t_feats).to(device), "CLIP_ViT-B_32"))
    except Exception as e:
        print(f"  [WARN] Could not load CLIP ViT-B/32: {e}")

    # EVA-CLIP
    try:
        import open_clip
        eva_m, _, _ = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='merged2b_s8b_b131k')
        eva_m = eva_m.to(device).eval()
        tok = open_clip.get_tokenizer('EVA02-B-16')
        prompts = [f"a photo of a {c}" for c in CLASSES]
        text_tokens = tok(prompts).to(device)
        with torch.no_grad():
            t_feats = eva_m.encode_text(text_tokens)
            t_feats /= t_feats.norm(dim=-1, keepdim=True)

        class EVAClassifier(nn.Module):
            def __init__(self, m, tf):
                super().__init__()
                self.m = m
                self.tf = nn.Parameter(tf, requires_grad=False)
            def forward(self, x):
                f = self.m.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                return 100.0 * f @ self.tf.T

        models_list.append((EVAClassifier(eva_m, t_feats).to(device), "EVA-CLIP"))
    except Exception as e:
        print(f"  [WARN] Could not load EVA-CLIP: {e}")

    return models_list


def train_model(model, train_records, device, epochs=3, batch_size=32, lr=1e-4):
    """Train a model on Train_All_Combined at 224×224."""
    dataset = ResolutionDataset(train_records, target_resolution=224)
    num_workers = 4 if sys.platform != 'win32' else 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)
        print(f"    Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.1f}%")

    model.eval()
    return model


def evaluate_model(model, test_records_list, device, resolution, batch_size=32):
    """Evaluate model at a specific resolution."""
    dataset = ResolutionDataset(test_records_list, target_resolution=resolution)
    num_workers = 4 if sys.platform != 'win32' else 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    all_preds = []
    all_labels = []

    model.eval()

    # Warmup
    with torch.no_grad():
        warmup_count = 0
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)
            warmup_count += 1
            if warmup_count >= 2:
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            total_images += images.size(0)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    latency_ms = (elapsed / total_images) * 1000 if total_images > 0 else 0
    throughput = total_images / elapsed if elapsed > 0 else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "latency_ms": latency_ms,
        "throughput_fps": throughput,
        "total_images": total_images
    }


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def generate_accuracy_vs_resolution_plot(df, model_name, save_path):
    """Line plot: accuracy vs resolution for each representation."""
    model_df = df[df["model"] == model_name]
    if model_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(REPRESENTATIONS)))

    for i, rep in enumerate(REPRESENTATIONS):
        rep_df = model_df[model_df["representation"] == rep].sort_values("resolution")
        if rep_df.empty:
            continue
        ax.plot(rep_df["resolution"], rep_df["accuracy"] * 100,
                marker='o', linewidth=2, markersize=6,
                label=rep, color=colors[i], alpha=0.85)

    ax.set_xlabel("Input Resolution (pixels)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{model_name} — Accuracy vs. Input Resolution", fontsize=14, fontweight="bold")
    ax.set_xticks(RESOLUTIONS)
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_resolution_heatmap(df, model_name, save_path):
    """Heatmap: Resolution × Representation for a single model."""
    model_df = df[df["model"] == model_name]
    if model_df.empty:
        return

    pivot = model_df.pivot_table(
        index="representation", columns="resolution",
        values="accuracy", aggfunc="mean"
    )
    if pivot.max().max() <= 1.0:
        pivot = pivot * 100

    # Reorder
    row_order = [r for r in REPRESENTATIONS if r in pivot.index]
    col_order = sorted(pivot.columns)
    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"})
    ax.set_title(f"{model_name} — Accuracy Heatmap (Resolution × Representation)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("Representation")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_degradation_bar(df, save_path):
    """Bar chart: Performance drop from 224→32 for each model × representation."""
    results = []
    for model_name in df["model"].unique():
        for rep in REPRESENTATIONS:
            subset = df[(df["model"] == model_name) & (df["representation"] == rep)]
            acc_224 = subset[subset["resolution"] == 224]["accuracy"].values
            acc_32 = subset[subset["resolution"] == 32]["accuracy"].values
            if len(acc_224) > 0 and len(acc_32) > 0:
                drop = (acc_224[0] - acc_32[0]) * 100
                results.append({"model": model_name, "representation": rep, "accuracy_drop": drop})

    if not results:
        return

    drop_df = pd.DataFrame(results)
    pivot = drop_df.pivot_table(index="representation", columns="model", values="accuracy_drop")

    row_order = [r for r in REPRESENTATIONS if r in pivot.index]
    pivot = pivot.reindex(index=row_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Representation")
    ax.set_ylabel("Accuracy Drop (%) from 224→32")
    ax.set_title("Resolution Degradation: Accuracy Drop (224px → 32px)", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_combined_dashboard(df, save_path):
    """4-panel dashboard: one line plot per model."""
    model_names = sorted(df["model"].unique())
    n_models = len(model_names)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
    if n_models == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    colors = plt.cm.tab10(np.linspace(0, 1, len(REPRESENTATIONS)))

    for idx, model_name in enumerate(model_names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        model_df = df[df["model"] == model_name]

        for i, rep in enumerate(REPRESENTATIONS):
            rep_df = model_df[model_df["representation"] == rep].sort_values("resolution")
            if rep_df.empty:
                continue
            ax.plot(rep_df["resolution"], rep_df["accuracy"] * 100,
                    marker='o', linewidth=1.8, markersize=5,
                    label=rep, color=colors[i], alpha=0.8)

        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(RESOLUTIONS)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)

    # Hide unused axes
    for idx in range(n_models, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Resolution Effect Analysis — All Models", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_html_report(df, save_path):
    """Generate HTML summary report with all tables."""
    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resolution Effect Analysis</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); color: #e0e0e0; padding: 40px 20px; }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 2em; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
  h2 { font-size: 1.4em; color: #a78bfa; margin: 30px 0 15px; border-bottom: 2px solid rgba(167,139,250,0.3); padding-bottom: 8px; }
  p.sub { color: #9ca3af; font-size: 0.9em; margin-bottom: 20px; }
  .tw { overflow-x: auto; margin: 15px 0 30px; border-radius: 12px; background: rgba(30,30,60,0.6); border: 1px solid rgba(167,139,250,0.15); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
  table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
  th { background: rgba(88,66,150,0.4); color: #d8b4fe; font-weight: 600; padding: 10px 12px; text-align: center; border-bottom: 2px solid rgba(167,139,250,0.3); white-space: nowrap; }
  td { padding: 8px 10px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.06); }
  tr:hover td { background: rgba(167,139,250,0.08) !important; }
  td.rh { text-align: left; font-weight: 500; color: #c4b5fd; white-space: nowrap; }
  footer { margin-top: 50px; text-align: center; color: #6b7280; font-size: 0.8em; }
</style></head><body><div class="container">
<h1>Resolution Effect Analysis</h1>
<p class="sub">Impact of input resolution on model performance across all visual representations</p>
"""

    model_names = sorted(df["model"].unique())

    for model_name in model_names:
        model_df = df[df["model"] == model_name]
        pivot = model_df.pivot_table(
            index="representation", columns="resolution",
            values="accuracy", aggfunc="mean"
        )
        if pivot.max().max() <= 1.0:
            pivot = pivot * 100

        row_order = [r for r in REPRESENTATIONS if r in pivot.index]
        pivot = pivot.reindex(index=row_order)
        pivot = pivot.reset_index()
        pivot.columns = ["Representation"] + [f"{int(c)}×{int(c)}" for c in pivot.columns[1:]]

        html += f'<h2>{model_name}</h2>\n<div class="tw"><table>\n<thead><tr>'
        for col in pivot.columns:
            html += f'<th>{col}</th>'
        html += '</tr></thead>\n<tbody>\n'

        for _, row in pivot.iterrows():
            html += '<tr>'
            for ci, col in enumerate(pivot.columns):
                val = row[col]
                if ci == 0:
                    html += f'<td class="rh">{val}</td>'
                elif isinstance(val, (int, float)) and not pd.isna(val):
                    norm = max(0, min(1, val / 100))
                    if norm < 0.5:
                        r, g, b = 220, int(60 + norm * 2 * 160), 60
                    else:
                        r, g, b = int(220 - (norm-0.5)*2*180), 200, int(60 + (norm-0.5)*2*80)
                    html += f'<td style="background:rgba({r},{g},{b},0.25)">{val:.1f}%</td>'
                else:
                    html += f'<td>—</td>'
            html += '</tr>\n'
        html += '</tbody></table></div>\n'

    html += '<footer><p>Generated by 09_resolution_analysis.py</p></footer>\n</div></body></html>'

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resolution effect analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS)
    args = parser.parse_args()

    device = torch.device(args.device)
    resolutions = args.resolutions

    print(f"\n{'='*70}")
    print(f"  RESOLUTION EFFECT ANALYSIS")
    print(f"  Device: {device}")
    print(f"  Resolutions: {resolutions}")
    print(f"{'='*70}\n")

    # Load data
    print("  ► Loading dataset...")
    train_records, test_records = get_test_records()
    print(f"    Train records: {len(train_records)}")
    for rep, records in test_records.items():
        print(f"    Test [{rep}]: {len(records)} images")

    # Build and train models
    print("\n  ► Building models...")
    models_list = build_models(device)

    for i, (model, model_name) in enumerate(models_list):
        if model_name in ("ResNet-50", "ViT-B_16", "DINOv3"):
            print(f"\n  ► Training {model_name} on Train_All_Combined...")
            model = train_model(model, train_records, device,
                               epochs=args.epochs, batch_size=args.batch_size)
            models_list[i] = (model, model_name)

    # ── Evaluate at each resolution ──
    all_results = []

    for model, model_name in models_list:
        print(f"\n  {'─'*60}")
        print(f"  ► Evaluating {model_name} across resolutions")
        print(f"  {'─'*60}")

        for resolution in resolutions:
            for rep, records in test_records.items():
                print(f"    {model_name} | {resolution:>4}×{resolution:<4} | {rep:>18}", end="")

                metrics = evaluate_model(model, records, device, resolution,
                                        batch_size=args.batch_size)

                result = {
                    "model": model_name,
                    "resolution": resolution,
                    "representation": rep,
                    **metrics
                }
                all_results.append(result)

                print(f" | Acc: {metrics['accuracy']*100:5.1f}% | F1: {metrics['f1']:.3f} "
                      f"| Lat: {metrics['latency_ms']:.2f}ms ✓")

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = RES_DIR / "resolution_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  ✓ Results saved to: {csv_path}")

    # Copy to structured csvs/
    csvs_dest = RESULTS_DIR / "csvs" / "resolution_results.csv"
    df.to_csv(csvs_dest, index=False, float_format="%.6f")

    # ── Generate Visualizations ──
    print("\n  ► Generating visualizations...")

    for model_name in df["model"].unique():
        safe_name = model_name.replace("/", "_").replace(" ", "_")

        # Line plot
        generate_accuracy_vs_resolution_plot(
            df, model_name,
            PLOTS_DIR / f"accuracy_vs_resolution_{safe_name}.png"
        )

        # Heatmap
        generate_resolution_heatmap(
            df, model_name,
            PLOTS_DIR / f"resolution_heatmap_{safe_name}.png"
        )

    # Degradation bar chart
    generate_degradation_bar(df, PLOTS_DIR / "resolution_degradation_bar.png")

    # Combined dashboard
    generate_combined_dashboard(df, PLOTS_DIR / "resolution_dashboard.png")

    # HTML report
    html_report_path = TABLES_DIR / "resolution_summary.html"
    generate_html_report(df, html_report_path)

    # Copy HTML report to tables/
    import shutil
    shutil.copy2(str(html_report_path), str(RESULTS_DIR / "tables" / "resolution_summary.html"))

    # Copy plots to visualizations/
    vis_dir = RESULTS_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for p in PLOTS_DIR.glob("*.png"):
        shutil.copy2(str(p), str(vis_dir / p.name))

    print(f"\n{'='*70}")
    print(f"  DONE — Resolution analysis complete!")
    print(f"  Results: {RES_DIR}")
    print(f"  Plots:   {PLOTS_DIR} (copied to results/visualizations/)")
    print(f"  Tables:  {TABLES_DIR} (copied to results/tables/)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
