"""Test-time augmentation (TTA) for the Task 4 ResNet checkpoints.

Each image is decoded once, turned into several deterministic views, and the
model's softmax probabilities are kept per view. Any combination of views can
then be averaged without running the network again. Nothing here trains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.task4.models.image_training import MEAN, STD

# Mild versions of the geometry seen in training (rotation up to 50 deg, scale 0.8-1.18).
VIEWS: dict[str, dict[str, float]] = {
    "identity": {"angle": 0.0, "scale": 1.0},
    "rot_m10": {"angle": -10.0, "scale": 1.0},
    "rot_p10": {"angle": 10.0, "scale": 1.0},
    "scale_090": {"angle": 0.0, "scale": 0.9},
    "scale_110": {"angle": 0.0, "scale": 1.1},
}


def view_transform(angle: float, scale: float) -> transforms.Compose:
    """Same resize/pad/normalise as image_transform('none'), plus one fixed affine view."""
    steps = [
        transforms.Resize((112, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.Pad((0, 56, 0, 56), fill=0),
    ]
    if angle != 0.0 or scale != 1.0:
        steps.append(transforms.Lambda(
            lambda image: TF.affine(image, angle=angle, translate=[0, 0], scale=scale,
                                    shear=[0.0], interpolation=InterpolationMode.BILINEAR, fill=0)))
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class MultiViewImages(Dataset):
    def __init__(self, paths: list[Path], view_names: list[str]):
        self.paths = paths
        self.transforms = [view_transform(**VIEWS[name]) for name in view_names]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return torch.stack([transform(image) for transform in self.transforms])


def view_probabilities(model, paths: list[Path], view_names: list[str] | None = None,
                       batch_size: int = 32) -> dict[str, np.ndarray]:
    """Return {view name: (n_images, n_classes) softmax probabilities}."""
    view_names = list(VIEWS) if view_names is None else view_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(MultiViewImages(paths, view_names), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    chunks = []
    with torch.inference_mode():
        for batch in loader:                                   # (B, V, 3, H, W)
            b, v = batch.shape[:2]
            logits = model(batch.flatten(0, 1).to(device, non_blocking=True))
            chunks.append(logits.softmax(1).view(b, v, -1).cpu().numpy())
    stacked = np.concatenate(chunks)                           # (N, V, C)
    model.cpu()
    return {name: stacked[:, k] for k, name in enumerate(view_names)}


def tta_predict(probabilities: dict[str, np.ndarray], view_names: list[str]) -> np.ndarray:
    """Average the chosen views' probabilities and return class indices."""
    return np.mean([probabilities[name] for name in view_names], axis=0).argmax(1)
