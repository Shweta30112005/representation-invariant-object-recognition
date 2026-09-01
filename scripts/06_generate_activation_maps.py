"""
Script 6: Generate Class Activation Maps (CAM)
Methods:
  1. Grad-CAM (Gradient-weighted Class Activation Mapping)
  2. Score-CAM (Score-based Class Activation Mapping - Gradient-free)
  3. Grad-CAM++ (Generalized Grad-CAM with higher-order partial derivatives)

Generates visual comparisons across:
  - All 10 Classes: car, cat, chair, circle, cup, dog, rectangle, square, table, triangle
  - All 6 Visual Representations: Original, Outline, Dotted, Dashed, Sketch, Silhouette
  - Multiple sample images per class
  - Multi-class comparisons for each visual representation style
  - Individual 4-panel publication-ready comparison cards

Outputs saved to: results/activation_maps/
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR = PROJECT_DIR / "dataset" / "variants"
RESULTS_DIR = PROJECT_DIR / "results"
CAM_DIR = RESULTS_DIR / "activation_maps"
GALLERY_DIR = CAM_DIR / "gallery"
CLASS_GRIDS_DIR = CAM_DIR / "class_grids"
STYLE_GRIDS_DIR = CAM_DIR / "style_grids"

for d in [CAM_DIR, GALLERY_DIR, CLASS_GRIDS_DIR, STYLE_GRIDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Image preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────
# Native Robust PyTorch Implementations of CAM Methods
# ─────────────────────────────────────────────────────────────

class BaseCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


class GradCAM(BaseCAM):
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    Weights are global average pooled gradients: alpha_k = (1/Z) * sum(grad_k)
    """
    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = output.argmax(dim=1).item()

        score = output[0, target_category]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]       # (C, H, W)
        activations = self.activations[0]   # (C, H, W)

        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)       # (H, W)
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_category


class GradCAMPlusPlus(BaseCAM):
    """
    Grad-CAM++: Generalized Grad-CAM with higher-order gradients
    Weights higher-order positive gradients for better multi-instance localization.
    """
    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = output.argmax(dim=1).item()

        score = output[0, target_category]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]       # (C, H, W)
        activations = self.activations[0]   # (C, H, W)

        # Higher-order gradients
        g = gradients
        g_2 = g.pow(2)
        g_3 = g.pow(3)

        # Alpha weights
        sum_act = torch.sum(activations, dim=(1, 2), keepdim=True)
        denom = 2.0 * g_2 + sum_act * g_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alpha = g_2 / denom

        weights = torch.sum(alpha * F.relu(g), dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_category


class ScoreCAM(BaseCAM):
    """
    Score-CAM: Score-weighted Class Activation Mapping (Gradient-free)
    Passes masked inputs through the model and weights activations by class confidence score.
    """
    def __call__(self, input_tensor, target_category=None, max_channels=32):
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_category is None:
                target_category = output.argmax(dim=1).item()

        activations = self.activations[0]   # (C, H, W)
        c, h, w = activations.shape

        # Select top channels by activation variance to keep it efficient and fast
        variances = activations.var(dim=(1, 2))
        top_k = min(c, max_channels)
        top_indices = torch.topk(variances, top_k).indices

        scores = []
        batch_masks = []

        for idx in top_indices:
            act = activations[idx].cpu().numpy()
            act = cv2.resize(act, (input_tensor.shape[3], input_tensor.shape[2]))
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            mask = torch.from_numpy(act).to(input_tensor.device).unsqueeze(0).unsqueeze(0)
            masked_input = input_tensor * mask
            batch_masks.append(masked_input)

        batch_masks = torch.cat(batch_masks, dim=0)

        with torch.no_grad():
            logits = []
            for i in range(0, len(batch_masks), 16):
                sub_batch = batch_masks[i:i+16]
                out = self.model(sub_batch)
                logits.append(out[:, target_category])
            logits = torch.cat(logits, dim=0)
            probs = F.softmax(logits, dim=0)

        cam = torch.zeros((input_tensor.shape[2], input_tensor.shape[3]), device=input_tensor.device)
        for i, idx in enumerate(top_indices):
            act = activations[idx].cpu().numpy()
            act = cv2.resize(act, (input_tensor.shape[3], input_tensor.shape[2]))
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            cam += probs[i] * torch.from_numpy(act).to(input_tensor.device)

        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_category


def overlay_cam(img_pil, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay CAM heatmap on PIL image."""
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * alpha + np.float32(img_np) * (1 - alpha)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return overlay


# ─────────────────────────────────────────────────────────────
# Visualizations Generator
# ─────────────────────────────────────────────────────────────

def generate_cam_visualizations(device):
    print("=" * 70)
    print("Generating Activation Maps for Multiple Images and Representations")
    print(f"Device: {device}")
    print(f"Output Root: {CAM_DIR}")
    print("=" * 70)

    # Load pretrained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    target_layer = model.layer4[-1]

    # Initialize the 3 CAM extractors
    grad_cam = GradCAM(model, target_layer)
    grad_cam_pp = GradCAMPlusPlus(model, target_layer)
    score_cam = ScoreCAM(model, target_layer)

    representations = ["Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette"]
    rep_paths = {
        "Original": ORIGINAL_DIR,
        "Outline": VARIANTS_DIR / "outline",
        "Dotted": VARIANTS_DIR / "dotted",
        "Dashed": VARIANTS_DIR / "dashed",
        "Sketch": VARIANTS_DIR / "sketch",
        "Silhouette": VARIANTS_DIR / "silhouette",
    }

    all_classes = ["cat", "dog", "car", "chair", "cup", "table", "circle", "square", "triangle", "rectangle"]
    col_titles = ["Input Image", "Grad-CAM", "Score-CAM", "Grad-CAM++"]

    # ─────────────────────────────────────────────────────────────
    # Part 1: Comprehensive Cross-Representation Grids for ALL 10 Classes
    # ─────────────────────────────────────────────────────────────
    print("\n--- Part 1: Generating Cross-Representation Grids for All 10 Classes ---")
    sample_files = ["00000.jpg", "00001.jpg"]

    for cls_name in all_classes:
        for s_idx, sample_filename in enumerate(sample_files):
            # Check if file exists
            if not (ORIGINAL_DIR / cls_name / sample_filename).exists():
                continue

            fig, axes = plt.subplots(len(representations), 4, figsize=(14, 18))
            fig.suptitle(f"Cross-Representation Activation Maps: '{cls_name.capitalize()}' (Sample #{s_idx + 1})\nGrad-CAM vs Score-CAM vs Grad-CAM++",
                         fontsize=15, fontweight="bold", y=0.99)

            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

            for row_idx, rep in enumerate(representations):
                img_path = rep_paths[rep] / cls_name / sample_filename
                if not img_path.exists():
                    continue

                img_pil = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

                cam_gc, _ = grad_cam(input_tensor)
                cam_sc, _ = score_cam(input_tensor)
                cam_gcpp, _ = grad_cam_pp(input_tensor)

                overlay_gc = overlay_cam(img_pil, cam_gc)
                overlay_sc = overlay_cam(img_pil, cam_sc)
                overlay_gcpp = overlay_cam(img_pil, cam_gcpp)

                # Col 0: Input Image
                axes[row_idx, 0].imshow(img_pil.resize((224, 224)))
                axes[row_idx, 0].set_ylabel(rep, fontsize=12, fontweight="bold", labelpad=10)
                axes[row_idx, 0].set_xticks([])
                axes[row_idx, 0].set_yticks([])

                # Col 1: Grad-CAM
                axes[row_idx, 1].imshow(overlay_gc)
                axes[row_idx, 1].axis("off")

                # Col 2: Score-CAM
                axes[row_idx, 2].imshow(overlay_sc)
                axes[row_idx, 2].axis("off")

                # Col 3: Grad-CAM++
                axes[row_idx, 3].imshow(overlay_gcpp)
                axes[row_idx, 3].axis("off")

            plt.tight_layout()
            out_file = CLASS_GRIDS_DIR / f"grid_{cls_name}_sample{s_idx + 1}.png"
            plt.savefig(out_file, bbox_inches="tight", dpi=250)
            plt.close()
            print(f"  [Class Grid] Saved: {out_file.name}")

    # ─────────────────────────────────────────────────────────────
    # Part 2: Multi-Class Grids for EACH Visual Representation Style
    # ─────────────────────────────────────────────────────────────
    print("\n--- Part 2: Generating Multi-Class Grids for Each Representation Style ---")
    sample_class_subset = ["cat", "dog", "car", "chair", "cup", "table", "circle", "triangle"]

    for rep in representations:
        fig, axes = plt.subplots(len(sample_class_subset), 4, figsize=(14, 24))
        fig.suptitle(f"Activation Maps Across Classes: Representation '{rep}'\nGrad-CAM vs Score-CAM vs Grad-CAM++",
                     fontsize=15, fontweight="bold", y=0.99)

        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

        for row_idx, cls_name in enumerate(sample_class_subset):
            img_path = rep_paths[rep] / cls_name / "00000.jpg"
            if not img_path.exists():
                continue

            img_pil = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

            cam_gc, _ = grad_cam(input_tensor)
            cam_sc, _ = score_cam(input_tensor)
            cam_gcpp, _ = grad_cam_pp(input_tensor)

            overlay_gc = overlay_cam(img_pil, cam_gc)
            overlay_sc = overlay_cam(img_pil, cam_sc)
            overlay_gcpp = overlay_cam(img_pil, cam_gcpp)

            axes[row_idx, 0].imshow(img_pil.resize((224, 224)))
            axes[row_idx, 0].set_ylabel(cls_name.capitalize(), fontsize=12, fontweight="bold", labelpad=10)
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])

            axes[row_idx, 1].imshow(overlay_gc)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(overlay_sc)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(overlay_gcpp)
            axes[row_idx, 3].axis("off")

        plt.tight_layout()
        out_file = STYLE_GRIDS_DIR / f"style_multiclass_{rep.lower()}.png"
        plt.savefig(out_file, bbox_inches="tight", dpi=250)
        plt.close()
        print(f"  [Style Grid] Saved: {out_file.name}")

    # ─────────────────────────────────────────────────────────────
    # Part 3: Individual 4-Panel Gallery Cards for Diverse Pictures
    # ─────────────────────────────────────────────────────────────
    print("\n--- Part 3: Generating Individual 4-Panel Comparison Gallery Cards ---")
    gallery_samples = [
        ("cat", "00001.jpg", ["Original", "Outline", "Dotted", "Silhouette"]),
        ("dog", "00001.jpg", ["Original", "Outline", "Dashed", "Sketch"]),
        ("car", "00001.jpg", ["Original", "Dashed", "Outline", "Silhouette"]),
        ("cup", "00000.jpg", ["Original", "Outline", "Dotted", "Silhouette"]),
        ("cup", "00001.jpg", ["Original", "Dashed", "Sketch", "Silhouette"]),
        ("chair", "00001.jpg", ["Original", "Outline", "Dotted", "Sketch"]),
        ("table", "00000.jpg", ["Original", "Outline", "Dashed", "Silhouette"]),
        ("table", "00001.jpg", ["Original", "Outline", "Sketch", "Silhouette"]),
        ("circle", "00000.jpg", ["Original", "Outline", "Dotted", "Silhouette"]),
        ("triangle", "00000.jpg", ["Original", "Outline", "Dashed", "Silhouette"]),
    ]

    card_count = 0
    for cls_name, fname, reps in gallery_samples:
        for rep in reps:
            img_path = rep_paths[rep] / cls_name / fname
            if not img_path.exists():
                continue

            img_pil = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

            cam_gc, _ = grad_cam(input_tensor)
            cam_sc, _ = score_cam(input_tensor)
            cam_gcpp, _ = grad_cam_pp(input_tensor)

            overlay_gc = overlay_cam(img_pil, cam_gc)
            overlay_sc = overlay_cam(img_pil, cam_sc)
            overlay_gcpp = overlay_cam(img_pil, cam_gcpp)

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
            fig.suptitle(f"{cls_name.capitalize()} | Representation: {rep} ({fname})",
                         fontsize=12, fontweight="bold", y=1.02)

            axes[0].imshow(img_pil.resize((224, 224)))
            axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(overlay_gc)
            axes[1].set_title("Grad-CAM", fontsize=11, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(overlay_sc)
            axes[2].set_title("Score-CAM", fontsize=11, fontweight="bold")
            axes[2].axis("off")

            axes[3].imshow(overlay_gcpp)
            axes[3].set_title("Grad-CAM++", fontsize=11, fontweight="bold")
            axes[3].axis("off")

            plt.tight_layout()
            card_name = f"{cls_name}_{fname.split('.')[0]}_{rep.lower()}_cam.png"
            card_file = GALLERY_DIR / card_name
            plt.savefig(card_file, bbox_inches="tight", dpi=200)
            plt.close()
            card_count += 1

    print(f"  [Gallery Cards] Saved {card_count} individual 4-panel comparison cards in {GALLERY_DIR}")

    # Clean up hooks
    grad_cam.remove_hooks()
    grad_cam_pp.remove_hooks()
    score_cam.remove_hooks()

    print("\n" + "=" * 70)
    print(f"[OK] Generated CAM visual comparisons across all images and representations!")
    print(f"  - Class grids: {CLASS_GRIDS_DIR}")
    print(f"  - Style grids: {STYLE_GRIDS_DIR}")
    print(f"  - Gallery cards: {GALLERY_DIR}")
    print("=" * 70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_cam_visualizations(device)


if __name__ == "__main__":
    main()
