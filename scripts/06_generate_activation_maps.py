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



# ─────────────────────────────────────────────────────────────
# SigLIP-2 CAM Add-on
# Existing ResNet-50 CAM implementation above is unchanged.
# ─────────────────────────────────────────────────────────────

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_CAM_DIR = CAM_DIR / "siglip2"
SIGLIP2_GALLERY_DIR = SIGLIP2_CAM_DIR / "gallery"
SIGLIP2_CLASS_GRIDS_DIR = SIGLIP2_CAM_DIR / "class_grids"
SIGLIP2_STYLE_GRIDS_DIR = SIGLIP2_CAM_DIR / "style_grids"

for d in [SIGLIP2_CAM_DIR, SIGLIP2_GALLERY_DIR,
          SIGLIP2_CLASS_GRIDS_DIR, SIGLIP2_STYLE_GRIDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class SigLIP2CAMBase:
    """Base helper for CAM methods on the SigLIP-2 vision transformer."""

    def __init__(self, model, processor, text_features, target_layer, device):
        self.model = model
        self.processor = processor
        self.text_features = text_features
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            gradient = grad_output[0]
            if isinstance(gradient, tuple):
                gradient = gradient[0]
            self.gradients = gradient

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def _get_image_inputs(self, image_pil):
        inputs = self.processor(
            images=image_pil,
            return_tensors="pt"
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _get_logits(self, image_inputs):
        # IMPORTANT: Do not call self.model(**image_inputs) here.
        # SigLIP-2's full forward pass expects both image and text inputs.
        # For CAM generation we only need the vision encoder.
        image_outputs = self.model.get_image_features(**image_inputs)

        # Depending on the installed Transformers version,
        # get_image_features() may return a tensor or BaseModelOutputWithPooling.
        if hasattr(image_outputs, "pooler_output"):
            image_features = image_outputs.pooler_output
        elif isinstance(image_outputs, tuple):
            image_features = image_outputs[0]
        else:
            image_features = image_outputs

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        logits = logits * self.model.logit_scale.exp() + self.model.logit_bias
        return logits

    @staticmethod
    def _tokens_to_spatial(features):
        """
        Convert ViT token features (B, N, C) into (B, C, H, W).
        SigLIP-2 patch tokens form a 14x14 spatial grid for the
        224x224 base-patch16 model.
        """
        if features.ndim == 4:
            return features

        if features.ndim != 3:
            raise RuntimeError(
                f"Unexpected SigLIP-2 activation shape: {tuple(features.shape)}"
            )

        batch, tokens, channels = features.shape

        # Remove CLS token if a model variant exposes one.
        spatial_tokens = tokens
        grid_size = int(spatial_tokens ** 0.5)

        if grid_size * grid_size != spatial_tokens:
            if (tokens - 1) > 0:
                grid_size = int((tokens - 1) ** 0.5)
                if grid_size * grid_size == tokens - 1:
                    features = features[:, 1:, :]
                    spatial_tokens = tokens - 1
                else:
                    raise RuntimeError(
                        f"Cannot reshape SigLIP-2 tokens to a spatial grid: {tokens}"
                    )
            else:
                raise RuntimeError(
                    f"Cannot reshape SigLIP-2 tokens to a spatial grid: {tokens}"
                )

        return features.transpose(1, 2).reshape(
            batch, channels, grid_size, grid_size
        )

    def _normalize_cam(self, cam, input_size=(224, 224)):
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, input_size)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class SigLIP2GradCAM(SigLIP2CAMBase):
    """Gradient-weighted CAM for SigLIP-2 image-text similarity."""

    def __call__(self, image_pil, target_category=None):
        image_inputs = self._get_image_inputs(image_pil)

        self.model.zero_grad()
        logits = self._get_logits(image_inputs)

        if target_category is None:
            target_category = logits.argmax(dim=1).item()

        score = logits[0, target_category]
        score.backward(retain_graph=True)

        activations = self._tokens_to_spatial(self.activations)
        gradients = self._tokens_to_spatial(self.gradients)

        weights = gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations[0], dim=0)
        cam = F.relu(cam)

        return self._normalize_cam(cam), target_category


class SigLIP2GradCAMPlusPlus(SigLIP2CAMBase):
    """Grad-CAM++ style implementation for SigLIP-2 similarity scores."""

    def __call__(self, image_pil, target_category=None):
        image_inputs = self._get_image_inputs(image_pil)

        self.model.zero_grad()
        logits = self._get_logits(image_inputs)

        if target_category is None:
            target_category = logits.argmax(dim=1).item()

        score = logits[0, target_category]
        score.backward(retain_graph=True)

        activations = self._tokens_to_spatial(self.activations)
        gradients = self._tokens_to_spatial(self.gradients)

        g = gradients[0]
        a = activations[0]

        g_2 = g.pow(2)
        g_3 = g.pow(3)

        sum_act = torch.sum(a, dim=(1, 2), keepdim=True)
        denom = 2.0 * g_2 + sum_act * g_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alpha = g_2 / denom
        weights = torch.sum(alpha * F.relu(g), dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * a, dim=0)
        cam = F.relu(cam)

        return self._normalize_cam(cam), target_category


class SigLIP2ScoreCAM(SigLIP2CAMBase):
    """Gradient-free Score-CAM style implementation for SigLIP-2."""

    def __call__(self, image_pil, target_category=None, max_channels=32):
        image_inputs = self._get_image_inputs(image_pil)

        with torch.no_grad():
            logits = self._get_logits(image_inputs)
            if target_category is None:
                target_category = logits.argmax(dim=1).item()

        activations = self._tokens_to_spatial(self.activations)[0]
        channels, height, width = activations.shape

        variances = activations.var(dim=(1, 2))
        top_k = min(channels, max_channels)
        top_indices = torch.topk(variances, top_k).indices

        original_np = np.array(image_pil.resize((224, 224)))
        original_tensor = torch.from_numpy(
            original_np.astype(np.float32) / 255.0
        ).permute(2, 0, 1).to(self.device)

        masks = []
        valid_indices = []

        for idx in top_indices:
            act = activations[idx].detach().cpu().numpy()
            act = cv2.resize(act, (224, 224))
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)

            mask = torch.from_numpy(act).to(self.device).float()
            masks.append(mask)
            valid_indices.append(idx)

        if not masks:
            return np.zeros((224, 224), dtype=np.float32), target_category

        masked_images = []
        for mask in masks:
            masked = original_tensor * mask.unsqueeze(0)
            masked = masked.clamp(0, 1)
            masked_images.append(
                Image.fromarray(
                    (masked.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
            )

        scores = []
        with torch.no_grad():
            for start in range(0, len(masked_images), 16):
                batch_images = masked_images[start:start + 16]
                batch_inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt"
                )
                batch_inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in batch_inputs.items()
                }
                batch_logits = self._get_logits(batch_inputs)
                scores.append(batch_logits[:, target_category])

        scores = torch.cat(scores, dim=0)
        weights = F.softmax(scores, dim=0)

        cam = torch.zeros((224, 224), device=self.device)

        for weight, mask in zip(weights, masks):
            cam += weight * mask

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_category


def _siglip2_target_layer(model):
    """Find the final spatial transformer block for SigLIP-2."""
    vision = model.vision_model

    if hasattr(vision.encoder, "layers"):
        return vision.encoder.layers[-1]

    raise RuntimeError("Could not locate SigLIP-2 vision encoder layers.")


def _siglip2_text_features(model, processor, classes, device):
    """Create normalized text embeddings for the 10 object classes."""
    prompts = [f"a photo of a {cls_name}" for cls_name in classes]

    text_inputs = processor(
        text=prompts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )

    text_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in text_inputs.items()
    }

    with torch.no_grad():
        text_outputs = model.get_text_features(**text_inputs)

        # Depending on the installed Transformers version,
        # get_text_features() may return a tensor or BaseModelOutputWithPooling.
        if hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        elif isinstance(text_outputs, tuple):
            text_features = text_outputs[0]
        else:
            text_features = text_outputs

    return text_features


def _generate_siglip2_card(
    model,
    grad_cam,
    score_cam,
    grad_cam_pp,
    rep_paths,
    cls_name,
    fname,
    rep,
    output_dir
):
    img_path = rep_paths[rep] / cls_name / fname
    if not img_path.exists():
        return False

    img_pil = Image.open(img_path).convert("RGB")

    cam_gc, target_gc = grad_cam(img_pil)
    cam_sc, target_sc = score_cam(img_pil, target_category=target_gc)
    cam_gcpp, _ = grad_cam_pp(img_pil, target_category=target_gc)

    overlay_gc = overlay_cam(img_pil, cam_gc)
    overlay_sc = overlay_cam(img_pil, cam_sc)
    overlay_gcpp = overlay_cam(img_pil, cam_gcpp)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    fig.suptitle(
        f"SigLIP-2 | {cls_name.capitalize()} | Representation: {rep} ({fname})",
        fontsize=12,
        fontweight="bold",
        y=1.02
    )

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

    card_name = f"{cls_name}_{fname.split('.')[0]}_{rep.lower()}_siglip2_cam.png"
    card_file = output_dir / card_name
    plt.savefig(card_file, bbox_inches="tight", dpi=200)
    plt.close()

    return True


def generate_siglip2_cam_visualizations(device):
    """
    Additional SigLIP-2 CAM pipeline.

    This function is intentionally separate from the existing ResNet-50
    pipeline so none of the original CAM code or outputs are changed.
    """
    print("\n" + "=" * 70)
    print("Generating Additional SigLIP-2 Activation Maps")
    print(f"Model: {SIGLIP2_MODEL_ID}")
    print(f"Device: {device}")
    print(f"Output Root: {SIGLIP2_CAM_DIR}")
    print("=" * 70)

    if AutoModel is None or AutoProcessor is None:
        print("[FAIL] 'transformers' is not installed.")
        print("  Install it with: pip install transformers")
        return

    try:
        processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
        model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"[FAIL] Could not load SigLIP-2: {e}")
        return

    representations = [
        "Original", "Outline", "Dotted", "Dashed", "Sketch", "Silhouette"
    ]

    rep_paths = {
        "Original": ORIGINAL_DIR,
        "Outline": VARIANTS_DIR / "outline",
        "Dotted": VARIANTS_DIR / "dotted",
        "Dashed": VARIANTS_DIR / "dashed",
        "Sketch": VARIANTS_DIR / "sketch",
        "Silhouette": VARIANTS_DIR / "silhouette",
    }

    all_classes = [
        "cat", "dog", "car", "chair", "cup",
        "table", "circle", "square", "triangle", "rectangle"
    ]

    col_titles = [
        "Input Image", "Grad-CAM", "Score-CAM", "Grad-CAM++"
    ]

    text_features = _siglip2_text_features(
        model, processor, all_classes, device
    )

    target_layer = _siglip2_target_layer(model)

    grad_cam = SigLIP2GradCAM(
        model, processor, text_features, target_layer, device
    )
    score_cam = SigLIP2ScoreCAM(
        model, processor, text_features, target_layer, device
    )
    grad_cam_pp = SigLIP2GradCAMPlusPlus(
        model, processor, text_features, target_layer, device
    )

    # ─────────────────────────────────────────────────────────────
    # Part 1: Cross-Representation Grids for ALL 10 Classes
    # ─────────────────────────────────────────────────────────────
    print("\n--- SigLIP-2 Part 1: Cross-Representation Grids ---")

    sample_files = ["00000.jpg", "00001.jpg"]

    for cls_name in all_classes:
        for s_idx, sample_filename in enumerate(sample_files):
            if not (ORIGINAL_DIR / cls_name / sample_filename).exists():
                continue

            fig, axes = plt.subplots(
                len(representations), 4, figsize=(14, 18)
            )

            fig.suptitle(
                f"SigLIP-2 Cross-Representation Activation Maps: "
                f"'{cls_name.capitalize()}' (Sample #{s_idx + 1})\n"
                f"Grad-CAM vs Score-CAM vs Grad-CAM++",
                fontsize=15,
                fontweight="bold",
                y=0.99
            )

            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(
                    title, fontsize=12, fontweight="bold", pad=10
                )

            for row_idx, rep in enumerate(representations):
                img_path = rep_paths[rep] / cls_name / sample_filename
                if not img_path.exists():
                    continue

                img_pil = Image.open(img_path).convert("RGB")

                cam_gc, target_category = grad_cam(img_pil)
                cam_sc, _ = score_cam(
                    img_pil, target_category=target_category
                )
                cam_gcpp, _ = grad_cam_pp(
                    img_pil, target_category=target_category
                )

                overlay_gc = overlay_cam(img_pil, cam_gc)
                overlay_sc = overlay_cam(img_pil, cam_sc)
                overlay_gcpp = overlay_cam(img_pil, cam_gcpp)

                axes[row_idx, 0].imshow(img_pil.resize((224, 224)))
                axes[row_idx, 0].set_ylabel(
                    rep, fontsize=12, fontweight="bold", labelpad=10
                )
                axes[row_idx, 0].set_xticks([])
                axes[row_idx, 0].set_yticks([])

                axes[row_idx, 1].imshow(overlay_gc)
                axes[row_idx, 1].axis("off")

                axes[row_idx, 2].imshow(overlay_sc)
                axes[row_idx, 2].axis("off")

                axes[row_idx, 3].imshow(overlay_gcpp)
                axes[row_idx, 3].axis("off")

            plt.tight_layout()

            out_file = (
                SIGLIP2_CLASS_GRIDS_DIR /
                f"grid_{cls_name}_sample{s_idx + 1}.png"
            )
            plt.savefig(out_file, bbox_inches="tight", dpi=250)
            plt.close()

            print(f"  [SigLIP-2 Class Grid] Saved: {out_file.name}")

    # ─────────────────────────────────────────────────────────────
    # Part 2: Multi-Class Grids for Each Representation
    # ─────────────────────────────────────────────────────────────
    print("\n--- SigLIP-2 Part 2: Multi-Class Representation Grids ---")

    sample_class_subset = [
        "cat", "dog", "car", "chair",
        "cup", "table", "circle", "triangle"
    ]

    for rep in representations:
        fig, axes = plt.subplots(
            len(sample_class_subset), 4, figsize=(14, 24)
        )

        fig.suptitle(
            f"SigLIP-2 Activation Maps Across Classes: "
            f"Representation '{rep}'\n"
            f"Grad-CAM vs Score-CAM vs Grad-CAM++",
            fontsize=15,
            fontweight="bold",
            y=0.99
        )

        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(
                title, fontsize=12, fontweight="bold", pad=10
            )

        for row_idx, cls_name in enumerate(sample_class_subset):
            img_path = rep_paths[rep] / cls_name / "00000.jpg"
            if not img_path.exists():
                continue

            img_pil = Image.open(img_path).convert("RGB")

            cam_gc, target_category = grad_cam(img_pil)
            cam_sc, _ = score_cam(
                img_pil, target_category=target_category
            )
            cam_gcpp, _ = grad_cam_pp(
                img_pil, target_category=target_category
            )

            overlay_gc = overlay_cam(img_pil, cam_gc)
            overlay_sc = overlay_cam(img_pil, cam_sc)
            overlay_gcpp = overlay_cam(img_pil, cam_gcpp)

            axes[row_idx, 0].imshow(img_pil.resize((224, 224)))
            axes[row_idx, 0].set_ylabel(
                cls_name.capitalize(),
                fontsize=12,
                fontweight="bold",
                labelpad=10
            )
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])

            axes[row_idx, 1].imshow(overlay_gc)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(overlay_sc)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(overlay_gcpp)
            axes[row_idx, 3].axis("off")

        plt.tight_layout()

        out_file = (
            SIGLIP2_STYLE_GRIDS_DIR /
            f"style_multiclass_{rep.lower()}.png"
        )
        plt.savefig(out_file, bbox_inches="tight", dpi=250)
        plt.close()

        print(f"  [SigLIP-2 Style Grid] Saved: {out_file.name}")

    # ─────────────────────────────────────────────────────────────
    # Part 3: Individual Gallery Cards
    # ─────────────────────────────────────────────────────────────
    print("\n--- SigLIP-2 Part 3: Individual Gallery Cards ---")

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
            if _generate_siglip2_card(
                model,
                grad_cam,
                score_cam,
                grad_cam_pp,
                rep_paths,
                cls_name,
                fname,
                rep,
                SIGLIP2_GALLERY_DIR
            ):
                card_count += 1

    print(
        f"  [SigLIP-2 Gallery Cards] Saved {card_count} "
        f"individual 4-panel comparison cards in {SIGLIP2_GALLERY_DIR}"
    )

    grad_cam.remove_hooks()
    score_cam.remove_hooks()
    grad_cam_pp.remove_hooks()

    del model
    del processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("[OK] SigLIP-2 CAM visual comparisons generated!")
    print(f"  - Class grids: {SIGLIP2_CLASS_GRIDS_DIR}")
    print(f"  - Style grids: {SIGLIP2_STYLE_GRIDS_DIR}")
    print(f"  - Gallery cards: {SIGLIP2_GALLERY_DIR}")
    print("=" * 70)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_cam_visualizations(device)
    generate_siglip2_cam_visualizations(device)


if __name__ == "__main__":
    main()
