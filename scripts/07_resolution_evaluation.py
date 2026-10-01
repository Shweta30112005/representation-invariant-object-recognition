"""
Script 7: Resolution Sensitivity Evaluation — Full 7-Model Benchmark
====================================================================

Research question
-----------------
How does input/source resolution affect recognition performance across
models and visual representations?

Models
------
1. ResNet-50            — supervised fine-tuning on Train_All_Combined
2. ViT-B/16             — supervised fine-tuning on Train_All_Combined
3. DINOv2              — Script-4-compatible DINOv2 ViT-B/14 backbone + linear classifier
4. CLIP ViT-B/32        — zero-shot
5. EVA-CLIP             — zero-shot
6. SigLIP-2 Base        — zero-shot
7. Qwen3-VL-8B          — zero-shot generative VLM

Important methodological rules
------------------------------
- Same classes and representations as Script 4.
- Same deterministic 80/20 split as Script 4: sorted original filenames,
  seed=42, split by ORIGINAL filenames, reused for every representation.
- Supervised models are trained ONCE at 224x224 on Train_All_Combined and
  the SAME learned weights are evaluated at every requested resolution.
- CLIP/EVA-CLIP/SigLIP-2 are zero-shot. Their official preprocessing is kept;
  the source image is resized to the requested resolution before preprocessing.
  Therefore these results measure source-resolution robustness, not a claim
  that their internal fixed encoder size changed.
- Qwen3-VL uses its official processor with a controlled image pixel budget
  corresponding to the requested resolution.
- Existing Script-4 CSVs are never modified.

Outputs
-------
results/resolution_experiment/
    resolution_results.csv
    resolution_summary.csv
    resolution_delta_from_224.csv
    plots/
        accuracy_vs_resolution_all_models.png
        f1_vs_resolution_all_models.png
        delta_accuracy_from_224_all_models.png
        accuracy_heatmap_model_resolution.png
        f1_heatmap_model_resolution.png
        <model>_representation_resolution_heatmap.png
        <model>_representation_delta_from_224_heatmap.png

Recommended final run
---------------------
python3 scripts/07_resolution_evaluation_all_models.py \
    --models all \
    --resolutions 64,128,224,384,512 \
    --epochs 5 \
    --batch-size 32

For an initial validation before the expensive full run:
python3 scripts/07_resolution_evaluation_all_models.py \
    --models resnet,vit,dino,clip,eva,siglip,qwen \
    --resolutions 64,128,224 \
    --max-images-per-rep 20 \
    --epochs 1

Qwen warning
------------
Qwen3-VL performs one generation per image. A full 5-resolution run is
therefore substantially more expensive than the other six models. You can
run the six encoder/classifier models first and Qwen separately if desired.
"""

import argparse
import gc
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Project configuration — aligned with Script 4
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_DIR / "dataset" / "original"
VARIANTS_DIR = PROJECT_DIR / "dataset" / "variants"
OUTPUT_DIR = PROJECT_DIR / "results" / "resolution_experiment"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIONS = [
    "Original",
    "Outline",
    "Dotted",
    "Dashed",
    "Sketch",
    "Silhouette",
    "ColorTint_Red",
    "ColorTint_Green",
    "ColorTint_Blue",
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42
BASELINE_RESOLUTION = 224

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_MODEL_NAME = "Qwen3-VL-8B"
QWEN_MAX_NEW_TOKENS = 16


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Exact Script-4 split logic
# ============================================================

def build_split(classes, train_ratio=0.8, seed=SEED, max_images_per_rep=None):
    """Same deterministic split methodology as Script 4."""
    random.seed(seed)

    class_train_files = {}
    class_test_files = {}

    for cls in classes:
        files = sorted(p.name for p in (ORIGINAL_DIR / cls).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(
                f"No images found for class '{cls}' in {ORIGINAL_DIR / cls}"
            )

        indices = list(range(len(files)))
        random.shuffle(indices)
        n_train = int(len(files) * train_ratio)
        train_idx = set(indices[:n_train])

        class_train_files[cls] = [
            files[i] for i in range(len(files)) if i in train_idx
        ]
        class_test_files[cls] = [
            files[i] for i in range(len(files)) if i not in train_idx
        ]

    train_records = []
    for rep in REPRESENTATIONS:
        base = REPRESENTATION_PATHS[rep]
        if not base.exists():
            print(f"[WARN] Missing representation directory: {base}")
            continue

        for cls in classes:
            class_dir = base / cls
            for fname in class_train_files[cls]:
                path = class_dir / fname
                if path.exists():
                    train_records.append((str(path), classes.index(cls)))

    test_sets = {}
    for rep in REPRESENTATIONS:
        base = REPRESENTATION_PATHS[rep]
        if not base.exists():
            continue

        records = []
        for cls in classes:
            class_dir = base / cls
            for fname in class_test_files[cls]:
                path = class_dir / fname
                if path.exists():
                    records.append((str(path), classes.index(cls)))

        if max_images_per_rep is not None:
            records = records[:max_images_per_rep]

        if records:
            test_sets[rep] = records

    return train_records, test_sets


# ============================================================
# Generic torchvision dataset
# ============================================================

class ResolutionDataset(Dataset):
    def __init__(self, records, resolution, train=False):
        self.records = records
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), label


# ============================================================
# Supervised models
# ============================================================

def create_resnet(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_vit(num_classes):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


class DINOClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs.get("pixel_values")
        feats = self.backbone(x)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        elif hasattr(feats, "logits"):
            feats = feats.logits
        return self.fc(feats)


def train_supervised_model(model, train_records, device, epochs, batch_size):
    """Train once at 224x224; the resulting weights are reused at all test resolutions."""
    dataset = ResolutionDataset(
        train_records,
        BASELINE_RESOLUTION,
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=1e-4)

    print(
        f"Training {len(train_records)} images at "
        f"{BASELINE_RESOLUTION}x{BASELINE_RESOLUTION} "
        f"for {epochs} epoch(s)..."
    )

    for epoch in range(epochs):
        model.train()
        # Linear-probe DINO: keep the frozen backbone in eval mode.
        if isinstance(model, DINOClassifier):
            model.backbone.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"  Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {running_loss / max(total, 1):.4f} "
            f"| Train Acc: {100.0 * correct / max(total, 1):.2f}%"
        )

    return model


def prepare_torchvision_vit_for_resolution(
    model,
    resolution,
    original_pos_embedding,
    original_image_size=224,
):
    """
    Change torchvision ViT-B/16 to a new input size by interpolating the
    ORIGINAL 224x224 positional embeddings independently for every resolution.

    This prevents cumulative 14x14 -> 4x4 -> 8x8 -> ... interpolation drift.
    """
    patch_size = model.patch_size
    if resolution % patch_size != 0:
        raise ValueError(
            f"ViT resolution {resolution} must be divisible by "
            f"patch size {patch_size}."
        )

    pos = original_pos_embedding.to(device=model.encoder.pos_embedding.device,
                                    dtype=model.encoder.pos_embedding.dtype)

    if resolution == original_image_size:
        model.encoder.pos_embedding = nn.Parameter(pos.clone())
        model.image_size = original_image_size
        return

    cls_pos = pos[:, :1, :]
    patch_pos = pos[:, 1:, :]
    old_grid = int(patch_pos.shape[1] ** 0.5)
    if old_grid * old_grid != patch_pos.shape[1]:
        raise RuntimeError(
            f"Unexpected ViT positional embedding shape: {tuple(pos.shape)}"
        )

    new_grid = resolution // patch_size
    dim = patch_pos.shape[-1]

    patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_pos = torch.nn.functional.interpolate(
        patch_pos,
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=True,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
        1, new_grid * new_grid, dim
    )

    model.encoder.pos_embedding = nn.Parameter(
        torch.cat([cls_pos, patch_pos], dim=1)
    )
    model.image_size = resolution


def evaluate_torchvision_model(
    model,
    model_name,
    test_sets,
    resolutions,
    device,
    batch_size,
):
    rows = []
    model.eval()

    original_pos_embedding = None
    if model_name == "ViT-B/16":
        original_pos_embedding = model.encoder.pos_embedding.detach().clone()

    for resolution in resolutions:
        if model_name == "ViT-B/16":
            prepare_torchvision_vit_for_resolution(
                model, resolution, original_pos_embedding
            )

        for rep, records in test_sets.items():
            dataset = ResolutionDataset(records, resolution, train=False)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2 if device.type == "cuda" else 0,
                pin_memory=(device.type == "cuda"),
            )

            preds, labels = [], []
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                for images, y in loader:
                    images = images.to(device, non_blocking=True)
                    output = model(images)
                    preds.extend(output.argmax(1).cpu().numpy())
                    labels.extend(y.numpy())

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)

            print(
                f"[{model_name}] {rep:<18} {resolution:>4}x{resolution:<4} "
                f"Accuracy={acc * 100:6.2f}% F1={f1:.4f}"
            )

            rows.append({
                "model": model_name,
                "mode": "Train_All_Combined",
                "representation": rep,
                "resolution": resolution,
                "accuracy": float(acc),
                "accuracy_pct": float(acc * 100.0),
                "f1": float(f1),
                "num_images": len(records),
                "inference_time_sec": float(elapsed),
            })

    return rows


def create_dino_v2(num_classes):
    """
    EXACT DINO implementation used by Script 4:
      torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    with a frozen 768-dim backbone and trainable linear classifier.

    No DINOv3 checkpoint is used here.
    """
    try:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14"
        )
        embed_dim = 768
        print("Loaded DINOv2 ViT-B/14 from facebookresearch/dinov2")
    except Exception as e:
        print(f"  [Notice] torch.hub DINOv2 load failed: {e}")
        try:
            import timm
            backbone = timm.create_model(
                "vit_base_patch14_dinov2",
                pretrained=True,
                num_classes=0
            )
            embed_dim = 768
            print("Loaded DINOv2 ViT-B/14 via timm fallback")
        except Exception as e2:
            raise RuntimeError(
                "Could not load the same DINOv2 backbone used by Script 4. "
                "Install/enable the DINOv2 dependency rather than silently "
                "substituting another architecture.\n"
                f"torch.hub error: {e}\n"
                f"timm error: {e2}"
            )

    return DINOClassifier(backbone, embed_dim, num_classes), None


def _dino_prepare_resolution(image, resolution):
    """
    Keep the requested SOURCE resolution exact, then pad only as needed to
    satisfy DINOv2 ViT-B/14's patch-size requirement. Padding is not a resize.

    64  -> 70, 128 -> 140, 224 -> 224, 384 -> 392, 512 -> 518.
    The CSV still records 64/128/224/384/512 as the experimental resolutions.
    """
    patch = 14
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    model_resolution = int(np.ceil(resolution / patch) * patch)

    if model_resolution != resolution:
        canvas = Image.new(
            "RGB",
            (model_resolution, model_resolution),
            (255, 255, 255),
        )
        canvas.paste(image, (0, 0))
        image = canvas

    return image, model_resolution


def evaluate_dino_v2(model, processor, test_sets, resolutions, device, batch_size):
    """Evaluate the Script-4 DINOv2 linear-probe model at each source resolution."""
    rows = []
    model.eval()

    for resolution in resolutions:
        for rep, records in test_sets.items():
            preds, labels = [], []
            start = time.perf_counter()

            for begin in range(0, len(records), batch_size):
                batch = records[begin:begin + batch_size]
                tensors = []

                for path, _ in batch:
                    with Image.open(path) as im:
                        image = im.convert("RGB")
                        image, _ = _dino_prepare_resolution(image, resolution)
                        # IMPORTANT: no second Resize here.
                        tensor = transforms.functional.to_tensor(image)
                        tensor = transforms.functional.normalize(
                            tensor, IMAGENET_MEAN, IMAGENET_STD
                        )
                    tensors.append(tensor)

                x = torch.stack(tensors).to(device, non_blocking=True)

                with torch.inference_mode():
                    output = model(x)
                    pred = output.argmax(dim=1)

                preds.extend(pred.cpu().numpy())
                labels.extend(y for _, y in batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)

            print(
                f"[DINOv2] {rep:<18} {resolution:>4}x{resolution:<4} "
                f"Accuracy={acc * 100:6.2f}% F1={f1:.4f}"
            )

            rows.append({
                "model": "DINOv2",
                "mode": "Linear Probe",
                "representation": rep,
                "resolution": resolution,
                "accuracy": float(acc),
                "accuracy_pct": float(acc * 100.0),
                "f1": float(f1),
                "num_images": len(records),
                "inference_time_sec": float(elapsed),
                "implementation_note": "Script-4-compatible DINOv2 ViT-B/14 backbone",
                "actual_model_resolution": int(np.ceil(resolution / 14) * 14),
            })

    return rows


# ============================================================
# Shared zero-shot prompts
# ============================================================

def get_prompt(rep, cls_name):
    if rep == "Original":
        return f"a photo of a {cls_name}"
    if rep == "Outline":
        return f"an outline drawing of a {cls_name}"
    if rep == "Dotted":
        return f"a dotted drawing of a {cls_name}"
    if rep == "Dashed":
        return f"a dashed line drawing of a {cls_name}"
    if rep == "Sketch":
        return f"a pencil sketch of a {cls_name}"
    if rep == "Silhouette":
        return f"a solid black silhouette of a {cls_name}"
    if rep.startswith("ColorTint"):
        color = rep.split("_")[-1].lower()
        return f"a {color}-tinted photo of a {cls_name}"
    return f"a drawing of a {cls_name}"


# ============================================================
# CLIP / EVA-CLIP
# ============================================================

def evaluate_open_clip(
    model_name,
    clip_model_name,
    pretrained_tag,
    classes,
    test_sets,
    resolutions,
    device,
    batch_size,
):
    import open_clip

    print(f"\nLoading {model_name}: {clip_model_name} / {pretrained_tag}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained_tag
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    rows = []

    for resolution in resolutions:
        for rep, records in test_sets.items():
            prompts = [get_prompt(rep, cls) for cls in classes]
            text_tokens = tokenizer(prompts).to(device)
            with torch.inference_mode():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            preds, labels = [], []
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for begin in range(0, len(records), batch_size):
                batch = records[begin:begin + batch_size]
                images = []
                for path, _ in batch:
                    with Image.open(path) as im:
                        image = im.convert("RGB").resize(
                            (resolution, resolution), Image.Resampling.LANCZOS
                        )
                    images.append(preprocess(image))

                image_tensor = torch.stack(images).to(device)
                with torch.inference_mode():
                    image_features = model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100.0 * image_features @ text_features.T
                    pred = logits.argmax(dim=-1)

                preds.extend(pred.cpu().numpy())
                labels.extend(y for _, y in batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            print(
                f"[{model_name}] {rep:<18} {resolution:>4}x{resolution:<4} "
                f"Accuracy={acc*100:6.2f}% F1={f1:.4f}"
            )
            rows.append({
                "model": model_name,
                "mode": "Zero-Shot",
                "representation": rep,
                "resolution": resolution,
                "accuracy": float(acc),
                "accuracy_pct": float(acc * 100),
                "f1": float(f1),
                "num_images": len(records),
                "inference_time_sec": float(elapsed),
            })

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# ============================================================
# SigLIP-2
# ============================================================

def evaluate_siglip2(classes, test_sets, resolutions, device, batch_size):
    from transformers import AutoModel, AutoProcessor

    model_id = "google/siglip2-base-patch16-224"
    print(f"\nLoading SigLIP-2: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    rows = []

    for resolution in resolutions:
        for rep, records in test_sets.items():
            prompts = [get_prompt(rep, cls) for cls in classes]
            text_inputs = processor(
                text=prompts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in text_inputs.items()}

            with torch.inference_mode():
                text_features = model.get_text_features(**text_inputs)
                if hasattr(text_features, "pooler_output"):
                    text_features = text_features.pooler_output
                elif isinstance(text_features, tuple):
                    text_features = text_features[0]
                text_features /= text_features.norm(dim=-1, keepdim=True)

            preds, labels = [], []
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            for begin in range(0, len(records), batch_size):
                batch = records[begin:begin + batch_size]
                images = []
                for path, _ in batch:
                    with Image.open(path) as im:
                        images.append(im.convert("RGB").resize(
                            (resolution, resolution), Image.Resampling.LANCZOS
                        ))

                image_inputs = processor(images=images, return_tensors="pt")
                image_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in image_inputs.items()}

                with torch.inference_mode():
                    image_features = model.get_image_features(**image_inputs)
                    if hasattr(image_features, "pooler_output"):
                        image_features = image_features.pooler_output
                    elif isinstance(image_features, tuple):
                        image_features = image_features[0]
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = image_features @ text_features.T
                    if hasattr(model, "logit_scale"):
                        logits = logits * model.logit_scale.exp()
                    if hasattr(model, "logit_bias"):
                        logits = logits + model.logit_bias
                    pred = logits.argmax(dim=-1)

                preds.extend(pred.cpu().numpy())
                labels.extend(y for _, y in batch)
                for im in images:
                    im.close()

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            print(
                f"[SigLIP-2 Base] {rep:<18} {resolution:>4}x{resolution:<4} "
                f"Accuracy={acc*100:6.2f}% F1={f1:.4f}"
            )
            rows.append({
                "model": "SigLIP-2 Base",
                "mode": "Zero-Shot",
                "representation": rep,
                "resolution": resolution,
                "accuracy": float(acc),
                "accuracy_pct": float(acc * 100),
                "f1": float(f1),
                "num_images": len(records),
                "inference_time_sec": float(elapsed),
            })

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# ============================================================
# Qwen3-VL-8B
# ============================================================

def normalize_qwen_prediction(response, classes):
    if response is None:
        return "unknown"
    text = str(response).strip().lower().replace("`", " ").replace("*", " ")
    text = re.sub(r"\s+", " ", text).strip()
    for cls in classes:
        if text == cls.lower():
            return cls
    for cls in sorted(classes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(cls.lower())}\b", text):
            return cls
    return "unknown"


def qwen_prompt(rep, classes):
    descriptions = {
        "Original": "This is a natural image of an object.",
        "Outline": "This is an outline drawing of an object.",
        "Dotted": "This is a dotted drawing of an object.",
        "Dashed": "This is a dashed line drawing of an object.",
        "Sketch": "This is a pencil sketch of an object.",
        "Silhouette": "This is a solid black silhouette representation of an object.",
        "ColorTint_Red": "This is a red-tinted image of an object.",
        "ColorTint_Green": "This is a green-tinted image of an object.",
        "ColorTint_Blue": "This is a blue-tinted image of an object.",
    }
    return (
        f"{descriptions.get(rep, 'This image contains an object.')}\n\n"
        "Identify the object represented in this image.\n"
        f"Choose exactly ONE class from the following list: {', '.join(classes)}.\n"
        "Answer with ONLY the class name and nothing else."
    )


def evaluate_qwen(classes, test_sets, resolutions, device, resume=True):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"\nLoading Qwen3-VL: {QWEN_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    checkpoint = OUTPUT_DIR / "Qwen3_VL_8B_resolution_predictions.csv"
    completed = {}
    if resume and checkpoint.exists():
        old = pd.read_csv(checkpoint)
        for _, row in old.iterrows():
            key = (str(row["representation"]), int(row["resolution"]), str(row["image_path"]))
            completed[key] = row.to_dict()
        print(f"Resuming Qwen from {len(completed)} saved predictions.")

    rows = list(completed.values())

    for resolution in resolutions:
        # Qwen3-VL controls image visual-token budget via image_processor.size.
        # shortest_edge/longest_edge correspond to min/max pixel budgets.
        pixel_budget = int(resolution * resolution)
        processor.image_processor.size = {
            "longest_edge": pixel_budget,
            "shortest_edge": pixel_budget,
        }

        for rep, records in test_sets.items():
            prompt = qwen_prompt(rep, classes)
            print(f"[Qwen3-VL-8B] {rep} @ {resolution}x{resolution}: {len(records)} images")

            for image_path, true_label_idx in records:
                key = (rep, int(resolution), str(image_path))
                if key in completed:
                    continue

                true_label = classes[int(true_label_idx)]
                with Image.open(image_path) as im:
                    image = im.convert("RGB").resize(
                        (resolution, resolution), Image.Resampling.LANCZOS
                    )

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }]

                start = time.perf_counter()
                try:
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=QWEN_MAX_NEW_TOKENS,
                            do_sample=False,
                        )
                    trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated)
                    ]
                    response = processor.batch_decode(
                        trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    predicted = normalize_qwen_prediction(response, classes)
                    error = ""
                except Exception as exc:
                    response = ""
                    predicted = "unknown"
                    error = repr(exc)
                    print(f"  [WARN] Qwen failed on {image_path}: {exc}")

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                row = {
                    "model": QWEN_MODEL_NAME,
                    "mode": "Zero-Shot Generative VLM",
                    "representation": rep,
                    "resolution": int(resolution),
                    "image_path": str(image_path),
                    "true_class": true_label,
                    "prediction": predicted,
                    "raw_response": response,
                    "inference_time_sec": float(elapsed),
                    "error": error,
                }
                rows.append(row)
                completed[key] = row

                # Incremental checkpoint after every image.
                pd.DataFrame(rows).to_csv(checkpoint, index=False)

    # Convert Qwen predictions to common resolution-results schema.
    qwen_rows = []
    qdf = pd.DataFrame(rows)
    if not qdf.empty:
        for (rep, resolution), g in qdf.groupby(["representation", "resolution"]):
            labels = g["true_class"].tolist()
            preds = g["prediction"].tolist()
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, labels=classes, average="macro", zero_division=0)
            qwen_rows.append({
                "model": QWEN_MODEL_NAME,
                "mode": "Zero-Shot Generative VLM",
                "representation": rep,
                "resolution": int(resolution),
                "accuracy": float(acc),
                "accuracy_pct": float(acc * 100),
                "f1": float(f1),
                "num_images": len(g),
                "inference_time_sec": float(g["inference_time_sec"].sum()),
            })
    return qwen_rows


# ============================================================
# Plotting
# ============================================================

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def make_plots(df, baseline=BASELINE_RESOLUTION):
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    if plot_df.empty:
        return

    # 1. Mean accuracy vs resolution — all models.
    mean_acc = (
        plot_df.groupby(["model", "resolution"], as_index=False)["accuracy_pct"]
        .mean()
    )
    plt.figure(figsize=(12, 7))
    for model, g in mean_acc.groupby("model"):
        g = g.sort_values("resolution")
        plt.plot(g["resolution"], g["accuracy_pct"], marker="o", linewidth=2, label=model)
    plt.xlabel("Input resolution (pixels)")
    plt.ylabel("Mean accuracy (%)")
    plt.title("Accuracy vs Input Resolution — All Models")
    plt.xticks(sorted(plot_df["resolution"].unique()))
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_vs_resolution_all_models.png", dpi=300)
    plt.close()

    # 2. Mean F1 vs resolution — all models.
    mean_f1 = (
        plot_df.groupby(["model", "resolution"], as_index=False)["f1"]
        .mean()
    )
    plt.figure(figsize=(12, 7))
    for model, g in mean_f1.groupby("model"):
        g = g.sort_values("resolution")
        plt.plot(g["resolution"], g["f1"] * 100, marker="o", linewidth=2, label=model)
    plt.xlabel("Input resolution (pixels)")
    plt.ylabel("Mean Macro F1 (%)")
    plt.title("Macro F1 vs Input Resolution — All Models")
    plt.xticks(sorted(plot_df["resolution"].unique()))
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "f1_vs_resolution_all_models.png", dpi=300)
    plt.close()

    # 3. Delta accuracy relative to 224 — all models.
    baseline_df = plot_df[plot_df["resolution"] == baseline][
        ["model", "representation", "accuracy_pct"]
    ].rename(columns={"accuracy_pct": "baseline_accuracy_pct"})
    delta = plot_df.merge(baseline_df, on=["model", "representation"], how="left")
    delta["delta_accuracy_pp"] = delta["accuracy_pct"] - delta["baseline_accuracy_pct"]
    delta.to_csv(OUTPUT_DIR / "resolution_delta_from_224.csv", index=False)

    mean_delta = (
        delta.groupby(["model", "resolution"], as_index=False)["delta_accuracy_pp"]
        .mean()
    )
    plt.figure(figsize=(12, 7))
    for model, g in mean_delta.groupby("model"):
        g = g.sort_values("resolution")
        plt.plot(g["resolution"], g["delta_accuracy_pp"], marker="o", linewidth=2, label=model)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Input resolution (pixels)")
    plt.ylabel("Accuracy change from 224 px (percentage points)")
    plt.title("Resolution Sensitivity Relative to 224×224")
    plt.xticks(sorted(plot_df["resolution"].unique()))
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "delta_accuracy_from_224_all_models.png", dpi=300)
    plt.close()

    # 4. Model × resolution accuracy heatmap.
    import numpy as np
    pivot = mean_acc.pivot(index="model", columns="resolution", values="accuracy_pct")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=100)
    plt.colorbar(label="Mean accuracy (%)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Input resolution (pixels)")
    plt.ylabel("Model")
    plt.title("Mean Accuracy Heatmap — Model × Resolution")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_heatmap_model_resolution.png", dpi=300)
    plt.close()

    # 5. Model × resolution F1 heatmap.
    pivot_f1 = mean_f1.pivot(index="model", columns="resolution", values="f1") * 100
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_f1.values, aspect="auto", vmin=0, vmax=100)
    plt.colorbar(label="Mean Macro F1 (%)")
    plt.xticks(range(len(pivot_f1.columns)), pivot_f1.columns)
    plt.yticks(range(len(pivot_f1.index)), pivot_f1.index)
    plt.xlabel("Input resolution (pixels)")
    plt.ylabel("Model")
    plt.title("Mean Macro F1 Heatmap — Model × Resolution")
    for i in range(pivot_f1.shape[0]):
        for j in range(pivot_f1.shape[1]):
            val = pivot_f1.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "f1_heatmap_model_resolution.png", dpi=300)
    plt.close()

    # 6. One accuracy heatmap per model: representation × resolution.
    for model in sorted(plot_df["model"].unique()):
        sub = plot_df[plot_df["model"] == model]
        p = sub.pivot_table(
            index="representation", columns="resolution", values="accuracy_pct", aggfunc="mean"
        ).reindex(REPRESENTATIONS)
        plt.figure(figsize=(10, 7))
        plt.imshow(p.values, aspect="auto", vmin=0, vmax=100)
        plt.colorbar(label="Accuracy (%)")
        plt.xticks(range(len(p.columns)), p.columns)
        plt.yticks(range(len(p.index)), p.index)
        plt.xlabel("Input resolution (pixels)")
        plt.ylabel("Representation")
        plt.title(f"{model} — Accuracy by Representation and Resolution")
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                val = p.iloc[i, j]
                if pd.notna(val):
                    plt.text(j, i, f"{val:.1f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{safe_filename(model)}_representation_resolution_heatmap.png", dpi=300)
        plt.close()

        # Delta heatmap relative to 224 for this model.
        b = sub[sub["resolution"] == baseline][["representation", "accuracy_pct"]].rename(
            columns={"accuracy_pct": "baseline_accuracy_pct"}
        )
        d = sub.merge(b, on="representation", how="left")
        d["delta"] = d["accuracy_pct"] - d["baseline_accuracy_pct"]
        pdelta = d.pivot_table(
            index="representation", columns="resolution", values="delta", aggfunc="mean"
        ).reindex(REPRESENTATIONS)
        vmax = np.nanmax(np.abs(pdelta.values)) if np.isfinite(pdelta.values).any() else 1
        vmax = max(float(vmax), 1.0)
        plt.figure(figsize=(10, 7))
        plt.imshow(pdelta.values, aspect="auto", vmin=-vmax, vmax=vmax)
        plt.colorbar(label="Accuracy change (percentage points)")
        plt.xticks(range(len(pdelta.columns)), pdelta.columns)
        plt.yticks(range(len(pdelta.index)), pdelta.index)
        plt.xlabel("Input resolution (pixels)")
        plt.ylabel("Representation")
        plt.title(f"{model} — Accuracy Change Relative to 224×224")
        for i in range(pdelta.shape[0]):
            for j in range(pdelta.shape[1]):
                val = pdelta.iloc[i, j]
                if pd.notna(val):
                    plt.text(j, i, f"{val:+.1f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{safe_filename(model)}_representation_delta_from_224_heatmap.png", dpi=300)
        plt.close()

    print(f"\nPlots saved to: {PLOTS_DIR}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full 7-model resolution sensitivity benchmark"
    )
    parser.add_argument(
        "--models", default="all",
        help="all or comma-separated: resnet,vit,dino,clip,eva,siglip,qwen"
    )
    parser.add_argument("--classes", default="all")
    parser.add_argument("--resolutions", default="64,128,224,384,512")
    parser.add_argument(
        "--max-images-per-rep", type=int, default=None,
        help="Optional TEST cap only; leave unset for the full 20%% test split."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="Resume completed models from resolution_results.csv")
    parser.add_argument("--no-qwen-resume", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    if args.classes.lower() == "all":
        classes = sorted(p.name for p in ORIGINAL_DIR.iterdir() if p.is_dir())
    else:
        classes = [x.strip() for x in args.classes.split(",") if x.strip()]

    resolutions = [int(x.strip()) for x in args.resolutions.split(",") if x.strip()]
    if any(r <= 0 for r in resolutions):
        raise ValueError("All resolutions must be positive.")
    if any(r % 16 != 0 for r in resolutions):
        raise ValueError(
            "Resolutions must be divisible by 16 because ViT-B/16 is part of the benchmark. "
            "Use values such as 64,128,224,384,512."
        )

    requested = {x.strip().lower() for x in args.models.split(",") if x.strip()}
    if "all" in requested:
        requested = {"resnet", "vit", "dino", "clip", "eva", "siglip", "qwen"}

    print("=" * 90)
    print("SCRIPT 7 — FULL 7-MODEL RESOLUTION SENSITIVITY EXPERIMENT")
    print("=" * 90)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Classes: {classes}")
    print(f"Representations: {REPRESENTATIONS}")
    print(f"Resolutions: {resolutions}")
    print(f"Max TEST images/representation: {args.max_images_per_rep}")
    print(f"Supervised training resolution: {BASELINE_RESOLUTION}x{BASELINE_RESOLUTION}")
    print(f"Seed: {SEED}")
    print("=" * 90)

    train_records, test_sets = build_split(
        classes=classes,
        train_ratio=0.8,
        seed=SEED,
        max_images_per_rep=args.max_images_per_rep,
    )

    print(f"Train records: {len(train_records)}")
    for rep, records in test_sets.items():
        print(f"Test [{rep}]: {len(records)}")

    all_rows = []

    # Resume support: preserve models already completed in a previous run.
    existing_results_path = OUTPUT_DIR / "resolution_results.csv"
    completed_models = set()
    if args.resume and existing_results_path.exists():
        try:
            existing_df = pd.read_csv(existing_results_path)
            if not existing_df.empty and "model" in existing_df.columns:
                all_rows.extend(existing_df.to_dict("records"))
                completed_models = set(existing_df["model"].dropna().astype(str).unique())
                print(f"[RESUME] Loaded existing results for: {sorted(completed_models)}")
        except Exception as e:
            print(f"[RESUME] Could not load existing results: {e}")

    # --------------------------------------------------------
    # 1. ResNet-50
    # --------------------------------------------------------
    if "resnet" in requested and "ResNet-50" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 1/7 — ResNet-50")
        print("=" * 90)
        model = create_resnet(len(classes))
        model = train_supervised_model(model, train_records, device, args.epochs, args.batch_size)
        all_rows.extend(evaluate_torchvision_model(
            model, "ResNet-50", test_sets, resolutions, device, args.batch_size
        ))
        del model
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 2. ViT-B/16
    # --------------------------------------------------------
    if "vit" in requested and "ViT-B/16" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 2/7 — ViT-B/16")
        print("=" * 90)
        model = create_vit(len(classes))
        model = train_supervised_model(model, train_records, device, args.epochs, args.batch_size)
        all_rows.extend(evaluate_torchvision_model(
            model, "ViT-B/16", test_sets, resolutions, device, args.batch_size
        ))
        del model
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 3. DINOv2
    # --------------------------------------------------------
    if "dino" in requested and "DINOv2" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 3/7 — DINOv2")
        print("=" * 90)
        model, processor = create_dino_v2(len(classes))
        # Train only the classifier head; backbone remains frozen.
        # The classifier training uses standard 224x224 ImageNet tensors.
        model = train_supervised_model(model, train_records, device, args.epochs, args.batch_size)
        all_rows.extend(evaluate_dino_v2(
            model, processor, test_sets, resolutions, device, args.batch_size
        ))
        del model, processor
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 4. CLIP ViT-B/32
    # --------------------------------------------------------
    if "clip" in requested and "CLIP ViT-B/32" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 4/7 — CLIP ViT-B/32")
        print("=" * 90)
        all_rows.extend(evaluate_open_clip(
            "CLIP ViT-B/32", "ViT-B-32", "openai",
            classes, test_sets, resolutions, device, args.batch_size
        ))

    # --------------------------------------------------------
    # 5. EVA-CLIP
    # --------------------------------------------------------
    if "eva" in requested and "EVA-CLIP" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 5/7 — EVA-CLIP")
        print("=" * 90)
        all_rows.extend(evaluate_open_clip(
            "EVA-CLIP", "EVA02-B-16", "merged2b_s8b_b131k",
            classes, test_sets, resolutions, device, args.batch_size
        ))

    # --------------------------------------------------------
    # 6. SigLIP-2
    # --------------------------------------------------------
    if "siglip" in requested and "SigLIP-2 Base" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 6/7 — SigLIP-2 Base")
        print("=" * 90)
        all_rows.extend(evaluate_siglip2(
            classes, test_sets, resolutions, device, args.batch_size
        ))

    # --------------------------------------------------------
    # 7. Qwen3-VL-8B
    # --------------------------------------------------------
    if "qwen" in requested and "Qwen3-VL-8B" not in completed_models:
        print("\n" + "=" * 90)
        print("MODEL 7/7 — Qwen3-VL-8B")
        print("=" * 90)
        all_rows.extend(evaluate_qwen(
            classes, test_sets, resolutions, device,
            resume=not args.no_qwen_resume,
        ))

    if not all_rows:
        raise RuntimeError("No model results were generated.")

    df = pd.DataFrame(all_rows)
    results_path = OUTPUT_DIR / "resolution_results.csv"
    df.to_csv(results_path, index=False)

    summary = (
        df.groupby(["model", "resolution"], as_index=False)
        .agg(
            mean_accuracy_pct=("accuracy_pct", "mean"),
            mean_f1=("f1", "mean"),
            total_images=("num_images", "sum"),
        )
    )
    summary_path = OUTPUT_DIR / "resolution_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_plots(df)

    print("\n" + "=" * 90)
    print("RESOLUTION EXPERIMENT COMPLETE")
    print("=" * 90)
    print(f"Detailed results: {results_path}")
    print(f"Summary:          {summary_path}")
    print(f"Plots:            {PLOTS_DIR}")
    print("\nMean accuracy by model and resolution:")
    print(summary.pivot(index="model", columns="resolution", values="mean_accuracy_pct").round(2).to_string())


if __name__ == "__main__":
    main()
