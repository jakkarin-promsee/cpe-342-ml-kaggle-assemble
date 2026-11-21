"""Reproduce the friend's Task 4 column from resnet18_cosine18_pretrained.pt with their 5-view TTA."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode, functional as TF
from PIL import Image

# Usage: python repro_friend_t4.py <out_dir> <kaggle_final_dir>
# <kaggle_final_dir> holds cpe342-ml-kaggle/, of_friends/ and kaggle_t12345_081771.csv
OUT = Path(sys.argv[1])
KF = Path(sys.argv[2])
RAW = KF / "cpe342-ml-kaggle/src/task4/data/raw"
PT = KF / "of_friends/resnet18_cosine18_pretrained.pt"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
VIEWS = ("identity", "rotate_minus8", "rotate_plus8", "scale95", "scale105")

class View:
    def __init__(self, v): self.v = v; self.base = transforms.Compose([transforms.Resize((112, 224), interpolation=InterpolationMode.BILINEAR), transforms.Pad((0, 56, 0, 56), fill=0)])
    def __call__(self, im):
        im = self.base(im)
        if self.v.startswith("rotate_"):
            im = TF.rotate(im, -8 if self.v == "rotate_minus8" else 8, interpolation=InterpolationMode.BILINEAR, fill=0)
        elif self.v.startswith("scale"):
            im = TF.affine(im, angle=0, translate=[0, 0], scale=.95 if self.v == "scale95" else 1.05, shear=[0, 0], interpolation=InterpolationMode.BILINEAR, fill=0)
        return TF.normalize(TF.to_tensor(im), MEAN, STD)

class Imgs(Dataset):
    def __init__(self, files, tf): self.files, self.tf = files, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        with Image.open(RAW / "test" / self.files[i]) as im: return self.tf(im.convert("RGB"))

pkg = torch.load(PT, map_location="cpu", weights_only=True)
print({k: v for k, v in pkg.items() if k != "state_dict"})
assert pkg["architecture"] == "resnet18"
model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(pkg["state_dict"]); dev = torch.device("cuda"); model = model.to(dev).eval()

test = pd.read_csv(RAW / "test.csv")
probs = {}
for v in VIEWS:
    t = time.time(); parts = []
    with torch.inference_mode():
        for x in DataLoader(Imgs(test.file_name.tolist(), View(v)), batch_size=64, num_workers=0, pin_memory=True):
            parts.append(model(x.to(dev)).softmax(1).cpu().numpy())
    probs[v] = np.concatenate(parts); print(v, f"{time.time()-t:.0f}s", flush=True)
np.savez_compressed(OUT / "friend_cosine18_test_view_probs.npz", **probs)

single = probs["identity"].argmax(1); tta = np.mean([probs[v] for v in VIEWS], 0).argmax(1)
best = pd.read_csv(KF / "kaggle_t12345_081771.csv")
assert best.id.tolist() == test.id.tolist()
col = best.task4.to_numpy()
print("agree tta vs 0.81771:", (tta == col).mean(), "mismatches", int((tta != col).sum()))
print("agree single vs 0.81771:", (single == col).mean())
print("tta counts", np.bincount(tta, minlength=5).tolist(), "| file counts", np.bincount(col, minlength=5).tolist())
ours = pd.read_csv(KF / "cpe342-ml-kaggle/src/task4/submission/t4_resnet18_geometry_full_v3_tta_all5.csv").task4.to_numpy()
print("agree our v3 all5 vs friend tta:", (ours == tta).mean())
