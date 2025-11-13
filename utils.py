"""
Utility functions for loading YOLO datasets from disk and helper math
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image


def _load_split(image_dir: Path, label_dir: Path, img_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Internal helper to load one split (train/val) of a YOLO dataset.

    Returns lists of image tensors and per-image target tensors. Targets are in
    format (img_idx_placeholder, class, x_center_px, y_center_px, w_px, h_px).
    """
    images: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    if not image_dir.exists():
        return images, targets

    image_files = sorted([p for p in image_dir.rglob("*.jpg")])
    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size))
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Convert to torch tensor (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            images.append(img_tensor)

        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with label_path.open("r") as f:
                entries = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_c, y_c, w, h = map(float, parts)
                    entries.append([
                        0.0,
                        cls,
                        x_c * img_size,
                        y_c * img_size,
                        w * img_size,
                        h * img_size,
                    ])
                if entries:
                    targets.append(torch.tensor(entries, dtype=torch.float32))
                else:
                    targets.append(torch.empty((0, 6), dtype=torch.float32))
        else:
            targets.append(torch.empty((0, 6), dtype=torch.float32))

    return images, targets


def load_client_yolo_dataset(client_root: Path, img_size: int = 320) -> Dict[str, object]:
    """
    Load train/val splits for a client in YOLO directory format.

    Args:
        client_root: Path to the client folder containing data.yaml, images/, labels/
        img_size: Target image resize (square)

    Returns:
        dict with keys:
            train: tuple(List[images], List[targets])
            val: tuple(List[images], List[targets])
            num_classes: int
            class_names: List[str]
    """
    client_root = Path(client_root)
    data_yaml = client_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

    with data_yaml.open("r") as f:
        data_cfg = yaml.safe_load(f)

    train_rel = data_cfg.get("train", "images/train")
    val_rel = data_cfg.get("val", "images/val")
    num_classes = data_cfg.get("nc", 0)
    class_names = data_cfg.get("names", [])

    train_images_dir = (client_root / train_rel).resolve()
    val_images_dir = (client_root / val_rel).resolve()
    train_labels_dir = client_root / "labels" / "train"
    val_labels_dir = client_root / "labels" / "val"

    train_images, train_targets = _load_split(train_images_dir, train_labels_dir, img_size)
    val_images, val_targets = _load_split(val_images_dir, val_labels_dir, img_size)

    return {
        "train": (train_images, train_targets),
        "val": (val_images, val_targets),
        "num_classes": num_classes,
        "class_names": class_names,
    }


def cosine_similarity_numpy(vec1, vec2):
    """
    Compute cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

