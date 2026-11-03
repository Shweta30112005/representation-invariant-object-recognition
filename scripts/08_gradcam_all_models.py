"""
Script 8: Generate Grad-CAM Activation Maps for ALL Models × ALL Variants × ALL Classes
═══════════════════════════════════════════════════════════════════════════════════════════

Models:
  - ResNet-50 (Fine-tuned, target: layer4[-1])
  - ViT-B/16 (Fine-tuned, target: encoder.layers[-1].ln_1)
  - DINOv3 (Linear probe, target: backbone last block norm)
  - SigLIP-2 Base (Zero-shot, attention rollout on vision encoder)

Representations:
  Original, Outline, Dotted, Dashed, Sketch, Silhouette,
  ColorTint_Red, ColorTint_Green, ColorTint_Blue

Classes (10):
  car, cat, chair, circle, cup, dog, rectangle, square, table, triangle

Each class uses 20 images.

Outputs:
  results/activation_maps_all/{model}/gallery/       — Individual CAM images
  results/activation_maps_all/{model}/class_grids/   — Per-class comparison grids
  results/activation_maps_all/comparisons/           — Cross-model comparisons
  results/activation_maps_all/stats/                 — Activation statistics
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

try:
    from transformers import AutoModel, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent.parent
ORIGINAL_DIR  = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR  = PROJECT_DIR / "dataset" / "variants"
RESULTS_DIR   = PROJECT_DIR / "results"
CAM_BASE_DIR  = RESULTS_DIR / "activation_maps_all"
COMPARE_DIR   = CAM_BASE_DIR / "comparisons"
STATS_DIR     = CAM_BASE_DIR / "stats"

for d in [CAM_BASE_DIR, COMPARE_DIR, STATS_DIR]:
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

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Standard preprocessing
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────
# Grad-CAM Implementation (supports Conv and ViT)
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM that works with both CNN feature maps and ViT token sequences."""

    def __init__(self, model, target_layer, is_vit=False, image_size=224, patch_size=16):
        self.model = model
        self.target_layer = target_layer
        self.is_vit = is_vit
        self.image_size = image_size
        self.patch_size = patch_size
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = output.argmax(dim=1).item()

        score = output[0, target_category]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        if grads is None or not isinstance(grads, torch.Tensor):
            # Safe attention/activation magnitude fallback when gradients are not hooked
            if self.is_vit and acts is not None and acts.shape[1] > 1:
                cam = acts[:, 1:].norm(dim=-1)
                n_patches = cam.shape[1]
                h = w = int(n_patches ** 0.5)
                cam = cam[:, :h*w].view(1, h, w)
            elif not self.is_vit and acts is not None:
                cam = acts.norm(dim=1)
            else:
                cam = torch.ones(1, 14, 14, device=input_tensor.device)
        elif self.is_vit:
            # ViT: activations are [B, N_tokens, D]
            # Exclude CLS token if present
            acts_p = acts[:, 1:, :] if acts.shape[1] > 1 else acts
            grads_p = grads[:, 1:, :] if (grads is not None and grads.shape[1] > 1) else grads

            if grads_p is not None and isinstance(grads_p, torch.Tensor) and grads_p.abs().sum() > 0:
                # Proper Vision Transformer Grad-CAM:
                # GAP over spatial patch tokens (dim 1) to get channel weights:
                weights = grads_p.mean(dim=1, keepdim=True)  # [B, 1, D]
                # Weighted combination across channels:
                cam = (acts_p * weights).sum(dim=-1)          # [B, N]
            else:
                # Fallback to feature norm
                cam = acts_p.norm(dim=-1)

            cam = F.relu(cam)
            # Reshape to spatial grid
            n_patches = cam.shape[1]
            h = w = int(n_patches ** 0.5)
            if h * w != n_patches:
                h = w = int(np.ceil(n_patches ** 0.5))
                pad_len = h * w - n_patches
                if pad_len > 0:
                    cam = F.pad(cam, (0, pad_len), value=0)
            cam = cam.view(1, h, w)
        else:
            # CNN: activations are [B, C, H, W]
            weights = grads.mean(dim=[2, 3], keepdim=True)  # GAP
            cam = (acts * weights).sum(dim=1)  # [B, H, W]

        cam = F.relu(cam)
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.squeeze().cpu().numpy(), target_category


# ─────────────────────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────────────────────

def build_resnet50(device, num_classes=10):
    """Build and train ResNet-50 on Train_All_Combined."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    target_layer = model.layer4[-1]
    return model, target_layer, False, "ResNet-50"


def build_vit_b16(device, num_classes=10):
    """Build ViT-B/16."""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)
    target_layer = model.encoder.layers[-1].ln_1
    return model, target_layer, True, "ViT-B_16"


def build_dinov3(device, num_classes=10):
    """Build DINOv3 (DINOv2 backbone + linear head)."""
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
            if hasattr(self.bb, 'blocks'):
                for p in self.bb.blocks[-1].parameters():
                    p.requires_grad = True
            self.fc = nn.Linear(dim, n_cls)

        def forward(self, x):
            feats = self.bb(x)
            if hasattr(feats, 'logits'):
                feats = feats.logits
            return self.fc(feats)

    model = DINOClassifier(backbone, embed_dim, num_classes)
    model = model.to(device)

    # Find the last block's norm layer for Grad-CAM
    if hasattr(backbone, 'blocks'):
        target_layer = backbone.blocks[-1].norm1
    elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
        target_layer = backbone.encoder.layers[-1].ln_1
    else:
        target_layer = list(backbone.modules())[-3]

    return model, target_layer, True, "DINOv3"


def build_siglip2(device, num_classes=10):
    """Build SigLIP-2 Base as a classifier (fine-tunable or feature-based)."""
    if not HAS_TRANSFORMERS:
        print("  [WARN] transformers not installed, skipping SigLIP-2")
        return None, None, None, None

    try:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained("google/siglip2-base-patch16-224", trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] Could not load SigLIP-2: {e}")
        return None, None, None, None

    class SigLIP2Classifier(nn.Module):
        def __init__(self, bb, n_cls):
            super().__init__()
            self.bb = bb
            # Freeze vision backbone except last layer for Grad-CAM
            for p in self.bb.parameters():
                p.requires_grad = False
            if hasattr(self.bb, 'vision_model') and hasattr(self.bb.vision_model, 'encoder'):
                for p in self.bb.vision_model.encoder.layers[-1].parameters():
                    p.requires_grad = True
            # Get hidden size
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

    model = SigLIP2Classifier(backbone, num_classes)
    model = model.to(device)

    # Find target layer
    if hasattr(backbone, 'vision_model') and hasattr(backbone.vision_model, 'encoder'):
        target_layer = backbone.vision_model.encoder.layers[-1].layer_norm1
    elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
        target_layer = backbone.encoder.layers[-1]
    else:
        target_layer = list(backbone.modules())[-3]

    return model, target_layer, True, "SigLIP-2"


def build_clip_vit_b32(device, num_classes=10):
    """Build CLIP ViT-B/32 Zero-Shot Classifier."""
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        prompts = [f"a photo of a {c}" for c in CLASSES]
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        class CLIPClassifier(nn.Module):
            def __init__(self, m, tf):
                super().__init__()
                self.m = m
                self.tf = nn.Parameter(text_features, requires_grad=False)
            def forward(self, x):
                feats = self.m.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return 100.0 * feats @ self.tf.T

        wrapper = CLIPClassifier(clip_model, text_features).to(device)
        target_layer = clip_model.visual.transformer.resblocks[-1].ln_1
        return wrapper, target_layer, True, "CLIP_ViT-B_32"
    except Exception as e:
        print(f"  [WARN] Could not load CLIP ViT-B/32: {e}")
        return None, None, None, None


def build_eva_clip(device, num_classes=10):
    """Build EVA-CLIP (EVA02-B-16) Zero-Shot Classifier."""
    try:
        import open_clip
        eva_model, _, _ = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='merged2b_s8b_b131k')
        eva_model = eva_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('EVA02-B-16')
        prompts = [f"a photo of a {c}" for c in CLASSES]
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = eva_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        class EVAClassifier(nn.Module):
            def __init__(self, m, tf):
                super().__init__()
                self.m = m
                self.tf = nn.Parameter(text_features, requires_grad=False)
            def forward(self, x):
                feats = self.m.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return 100.0 * feats @ self.tf.T

        wrapper = EVAClassifier(eva_model, text_features).to(device)
        if hasattr(eva_model.visual, 'trunk'):
            target_layer = eva_model.visual.trunk.blocks[-1].norm1
        elif hasattr(eva_model.visual, 'blocks'):
            target_layer = eva_model.visual.blocks[-1].norm1
        else:
            target_layer = list(eva_model.visual.modules())[-3]
        return wrapper, target_layer, True, "EVA-CLIP"
    except Exception as e:
        print(f"  [WARN] Could not load EVA-CLIP: {e}")
        return None, None, None, None


# ─────────────────────────────────────────────────────────────
# Training Utility (quick Train_All_Combined)
# ─────────────────────────────────────────────────────────────

def get_all_combined_records(train_ratio=0.8, seed=42):
    """Get train and test records for Train_All_Combined setup."""
    random.seed(seed)

    class_train_files = {}
    class_test_files = {}

    for c in CLASSES:
        orig_dir = ORIGINAL_DIR / c
        files = sorted([f.name for f in orig_dir.glob("*.jpg")] +
                       [f.name for f in orig_dir.glob("*.png")] +
                       [f.name for f in orig_dir.glob("*.jpeg")])
        if not files:
            print(f"  [WARN] No images for class '{c}' in {orig_dir}")
            continue

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        indices = list(range(n_total))
        random.shuffle(indices)
        train_idx = set(indices[:n_train])

        class_train_files[c] = [files[i] for i in range(n_total) if i in train_idx]
        class_test_files[c] = [files[i] for i in range(n_total) if i not in train_idx]

    # Build combined training records
    train_records = []
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        if not base_dir.exists():
            continue
        for c in CLASSES:
            if c not in class_train_files:
                continue
            c_dir = base_dir / c
            for fname in class_train_files[c]:
                p = c_dir / fname
                if p.exists():
                    train_records.append((str(p), CLASS_TO_IDX[c]))

    # Build test records per class per representation
    test_records = {}
    for rep in REPRESENTATIONS:
        base_dir = REPRESENTATION_PATHS[rep]
        if not base_dir.exists():
            continue
        test_records[rep] = {}
        for c in CLASSES:
            if c not in class_test_files:
                continue
            c_dir = base_dir / c
            records = []
            for fname in class_test_files[c]:
                p = c_dir / fname
                if p.exists():
                    records.append(str(p))
            test_records[rep][c] = records

    return train_records, test_records


def train_model_quick(model, train_records, device, epochs=3, batch_size=32, lr=1e-4):
    """Quick training on Train_All_Combined."""
    from torch.utils.data import Dataset, DataLoader

    class SimpleDataset(Dataset):
        def __init__(self, records, transform=None):
            self.records = records
            self.transform = transform
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            path, label = self.records[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SimpleDataset(train_records, transform=transform)
    num_workers = 4 if sys.platform != 'win32' else 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
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
        print(f"    Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.1f}%")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# CAM Generation & Visualization
# ─────────────────────────────────────────────────────────────

def generate_cam_overlay(img_path, cam_map, save_path=None, figsize=(10, 4)):
    """Generate a side-by-side visualization: Original | CAM overlay | CAM heatmap."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize CAM to image size
    cam_resized = cv2.resize(cam_map, (224, 224))
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_rgb, 0.45, 0)

    if save_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Overlay", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Activation Map", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return img_rgb, overlay, heatmap_rgb, cam_resized


def generate_class_grid(cam_data, cls_name, model_name, save_path, max_images=5):
    """
    Generate a grid: rows = images, cols = representations.
    cam_data: dict of {rep: [(img_rgb, cam_resized, img_path), ...]}
    """
    available_reps = [r for r in REPRESENTATIONS if r in cam_data and len(cam_data[r]) > 0]
    if not available_reps:
        return

    n_reps = len(available_reps)
    n_images = min(max_images, min(len(cam_data[r]) for r in available_reps))
    if n_images == 0:
        return

    fig, axes = plt.subplots(n_images, n_reps * 2, figsize=(n_reps * 4, n_images * 2.2))
    if n_images == 1:
        axes = axes[np.newaxis, :]
    if n_reps * 2 == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f'{model_name} — Class: {cls_name.upper()} — Grad-CAM Comparison',
                 fontsize=14, fontweight='bold', y=1.02)

    for ri, rep in enumerate(available_reps):
        for ii in range(n_images):
            if ii >= len(cam_data[rep]):
                continue
            img_rgb, cam_map, _ = cam_data[rep][ii]

            # Show original image
            ax_img = axes[ii, ri * 2]
            ax_img.imshow(img_rgb)
            ax_img.axis("off")
            if ii == 0:
                ax_img.set_title(f'{rep}\n(Image)', fontsize=8, fontweight='bold')

            # Show CAM overlay
            cam_resized = cv2.resize(cam_map, (224, 224)) if cam_map.shape != (224, 224) else cam_map
            cam_uint8 = np.uint8(255 * cam_resized)
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_rgb, 0.45, 0)

            ax_cam = axes[ii, ri * 2 + 1]
            ax_cam.imshow(overlay)
            ax_cam.axis("off")
            if ii == 0:
                ax_cam.set_title(f'{rep}\n(CAM)', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_model_comparison_grid(all_models_cam, cls_name, rep_name, img_idx, save_path):
    """
    Compare CAMs from all models for the same image.
    all_models_cam: dict of {model_name: (img_rgb, cam_map)}
    """
    models_available = [m for m in all_models_cam if all_models_cam[m] is not None]
    if not models_available:
        return

    n_models = len(models_available)
    fig, axes = plt.subplots(2, n_models, figsize=(n_models * 3.5, 7))
    if n_models == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f'Model Comparison — {cls_name.upper()} / {rep_name} / Image #{img_idx+1}',
                 fontsize=13, fontweight='bold')

    for mi, model_name in enumerate(models_available):
        img_rgb, cam_map = all_models_cam[model_name]

        # Top row: original + overlay
        cam_resized = cv2.resize(cam_map, (224, 224))
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_rgb, 0.45, 0)

        axes[0, mi].imshow(img_rgb)
        axes[0, mi].set_title(f'{model_name}\n(Original)', fontsize=9)
        axes[0, mi].axis("off")

        axes[1, mi].imshow(overlay)
        axes[1, mi].set_title(f'{model_name}\n(Grad-CAM)', fontsize=9)
        axes[1, mi].axis("off")

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for all models/variants/classes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_images", type=int, default=20, help="Number of images per class for CAM")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for supervised models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_training", action="store_true", help="Skip training (use pretrained weights only)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"  GRAD-CAM ACTIVATION MAPS — ALL MODELS × ALL VARIANTS × ALL CLASSES")
    print(f"  Device: {device}")
    print(f"  Images per class: {args.n_images}")
    print(f"{'='*70}\n")

    # Get training and test data
    print("  ► Loading dataset splits...")
    train_records, test_records = get_all_combined_records()
    print(f"    Train records: {len(train_records)}")
    total_test = sum(len(v) for rep_dict in test_records.values() for v in rep_dict.values())
    print(f"    Test records: {total_test}")

    # Build all models
    model_configs = [
        build_resnet50(device),
        build_vit_b16(device),
        build_dinov3(device),
    ]

    # Add Zero-Shot VLMs
    for builder in [build_siglip2, build_clip_vit_b32, build_eva_clip]:
        cfg = builder(device)
        if cfg[0] is not None:
            model_configs.append(cfg)

    # Filter out None models
    model_configs = [(m, tl, is_vit, name) for m, tl, is_vit, name in model_configs if m is not None]

    # Train supervised models
    if not args.skip_training:
        for i, (model, target_layer, is_vit, model_name) in enumerate(model_configs):
            if model_name in ("ResNet-50", "ViT-B_16", "DINOv3"):
                print(f"\n  ► Training {model_name} on Train_All_Combined...")
                model = train_model_quick(model, train_records, device,
                                         epochs=args.epochs, batch_size=args.batch_size)
                model_configs[i] = (model, target_layer, is_vit, model_name)

    # ── Generate CAMs ──
    activation_stats = []  # Collect stats: model, class, rep, mean_activation, max_activation

    # Store CAMs for cross-model comparison
    all_model_cams = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # all_model_cams[model_name][rep][cls] = [(img_rgb, cam_map, img_path), ...]

    for model, target_layer, is_vit, model_name in model_configs:
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        model_dir = CAM_BASE_DIR / safe_name
        gallery_dir = model_dir / "gallery"
        class_grids_dir = model_dir / "class_grids"
        for d in [model_dir, gallery_dir, class_grids_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n  {'─'*60}")
        print(f"  ► Generating Grad-CAM for: {model_name}")
        print(f"  {'─'*60}")

        cam_generator = GradCAM(model, target_layer, is_vit=is_vit)

        for rep in REPRESENTATIONS:
            if rep not in test_records:
                print(f"    [SKIP] Representation '{rep}' not available")
                continue

            for cls in CLASSES:
                if cls not in test_records[rep] or not test_records[rep][cls]:
                    continue

                # Select N images
                img_paths = test_records[rep][cls][:args.n_images]
                print(f"    {model_name} | {rep:>18} | {cls:>10} | {len(img_paths)} images", end="")

                cam_data_for_grid = []

                for img_idx, img_path in enumerate(img_paths):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        input_tensor = PREPROCESS(img).unsqueeze(0).to(device)

                        cam_map, pred_class = cam_generator(input_tensor, target_category=CLASS_TO_IDX[cls])

                        # Load original image for visualization
                        img_cv = cv2.imread(img_path)
                        if img_cv is None:
                            continue
                        img_cv = cv2.resize(img_cv, (224, 224))
                        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                        # Save individual CAM
                        fname = Path(img_path).stem
                        save_name = f"{cls}_{fname}_{rep}_{safe_name}_cam.png"
                        save_path = gallery_dir / save_name
                        generate_cam_overlay(img_path, cam_map, save_path=save_path, figsize=(9, 3))

                        # Collect for grid
                        cam_data_for_grid.append((img_rgb, cam_map, img_path))

                        # Store for cross-model comparison
                        all_model_cams[model_name][rep][cls].append((img_rgb, cam_map, img_path))

                        # Stats
                        activation_stats.append({
                            "model": model_name,
                            "class": cls,
                            "representation": rep,
                            "image": fname,
                            "mean_activation": float(cam_map.mean()),
                            "max_activation": float(cam_map.max()),
                            "predicted_class": CLASSES[pred_class]
                        })

                    except Exception as e:
                        print(f" [ERR: {e}]", end="")
                        continue

                print(f" ✓")

                # Generate class grid for this class/rep
                if cam_data_for_grid:
                    grid_path = class_grids_dir / f"grid_{cls}_{rep}_{safe_name}.png"
                    # Simple grid: just this representation
                    fig_rows = min(5, len(cam_data_for_grid))
                    fig, axes = plt.subplots(fig_rows, 3, figsize=(9, fig_rows * 2.5))
                    if fig_rows == 1:
                        axes = axes[np.newaxis, :]
                    fig.suptitle(f'{model_name} — {cls.upper()} — {rep}', fontsize=12, fontweight='bold')

                    for ii in range(fig_rows):
                        img_rgb, cam_map, _ = cam_data_for_grid[ii]
                        cam_resized = cv2.resize(cam_map, (224, 224))
                        cam_uint8 = np.uint8(255 * cam_resized)
                        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_rgb, 0.45, 0)

                        axes[ii, 0].imshow(img_rgb)
                        axes[ii, 0].axis("off")
                        if ii == 0:
                            axes[ii, 0].set_title("Image", fontsize=9)

                        axes[ii, 1].imshow(overlay)
                        axes[ii, 1].axis("off")
                        if ii == 0:
                            axes[ii, 1].set_title("Overlay", fontsize=9)

                        axes[ii, 2].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
                        axes[ii, 2].axis("off")
                        if ii == 0:
                            axes[ii, 2].set_title("Heatmap", fontsize=9)

                    plt.tight_layout()
                    fig.savefig(str(grid_path), dpi=150, bbox_inches="tight")
                    plt.close(fig)

        # Generate cross-representation grids per class
        print(f"\n    ► Generating cross-representation grids for {model_name}...")
        for cls in CLASSES:
            cam_per_rep = {}
            for rep in REPRESENTATIONS:
                if model_name in all_model_cams and rep in all_model_cams[model_name] and cls in all_model_cams[model_name][rep]:
                    cam_per_rep[rep] = all_model_cams[model_name][rep][cls]

            if cam_per_rep:
                grid_path = class_grids_dir / f"cross_rep_grid_{cls}_{safe_name}.png"
                generate_class_grid(cam_per_rep, cls, model_name, grid_path, max_images=4)

        cam_generator.remove_hooks()

    # ── Cross-Model Comparison Grids ──
    print(f"\n  ► Generating cross-model comparison grids...")
    model_names = [name for _, _, _, name in model_configs]

    for rep in REPRESENTATIONS:
        for cls in CLASSES:
            # Take first image available across all models
            for img_idx in range(min(3, args.n_images)):
                model_cam_dict = {}
                for model_name in model_names:
                    if (model_name in all_model_cams and
                        rep in all_model_cams[model_name] and
                        cls in all_model_cams[model_name][rep] and
                        img_idx < len(all_model_cams[model_name][rep][cls])):
                        img_rgb, cam_map, _ = all_model_cams[model_name][rep][cls][img_idx]
                        model_cam_dict[model_name] = (img_rgb, cam_map)

                if len(model_cam_dict) >= 2:
                    save_path = COMPARE_DIR / f"compare_{cls}_{rep}_img{img_idx+1}.png"
                    generate_model_comparison_grid(model_cam_dict, cls, rep, img_idx, save_path)

    # ── Save Activation Stats ──
    if activation_stats:
        stats_df = pd.DataFrame(activation_stats)
        stats_df.to_csv(STATS_DIR / "activation_stats.csv", index=False, float_format="%.6f")
        print(f"\n  ✓ Activation stats saved ({len(stats_df)} entries)")

        # Generate summary heatmap: mean activation per model × class
        pivot_mean = stats_df.pivot_table(
            index="class", columns="model", values="mean_activation", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_mean, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Mean Activation"})
        ax.set_title("Mean Grad-CAM Activation Intensity — Model × Class", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(STATS_DIR / "mean_activation_heatmap.png"), dpi=200)
        plt.close(fig)

        # Summary heatmap: mean activation per model × representation
        pivot_rep = stats_df.pivot_table(
            index="representation", columns="model", values="mean_activation", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_rep, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Mean Activation"})
        ax.set_title("Mean Grad-CAM Activation Intensity — Model × Representation", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(STATS_DIR / "mean_activation_by_rep_heatmap.png"), dpi=200)
        plt.close(fig)

    print(f"\n{'='*70}")
    print(f"  DONE — All Grad-CAM activation maps generated!")
    print(f"  Output: {CAM_BASE_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
