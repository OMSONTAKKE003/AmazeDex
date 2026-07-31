"""
train_face_cnn.py
==================
Trains the MobileNetV3-Small cube-face classifier consumed by
cube_face_cnn_detector.py / cube_tag_pose_cnn_detector.py.

Changes vs the previous version of this script (bugs found on review):

  1. FRAGILE INDEXING FIXED: IndexSubset used `df.iloc[self.indices[i]]` but
     `self.indices` are `df.index` values (from a filtered/shuffled slice),
     not positional row numbers. This only "worked" because pd.read_csv
     happens to produce a default 0..N-1 RangeIndex today. Any future
     re-indexing (dropping rows, concatenating CSVs, etc.) would silently
     pair images with the WRONG labels with no error raised. Fixed to
     `.loc[...]`, which is index-value-based and therefore correct
     regardless of how the DataFrame's index is structured.

  2. CORRUPT / MISSING IMAGE PROTECTION: a single unreadable or missing
     file used to crash the entire multi-hour run. __getitem__ now catches
     that and logs + skips to a neighboring sample instead of dying.

  3. CLASS SORT NO LONGER SILENTLY FALLS BACK TO ALPHABETICAL for the
     WHOLE label set just because one label (e.g. "unknown") isn't
     numeric. Numeric labels are now sorted numerically and any
     non-numeric labels (e.g. "unknown") are appended after them in
     alphabetical order. This keeps digit classes in the confusion
     matrix in a sane 1,2,3...6,unknown order instead of 1,10,2,3...
     (this bug wasn't currently visible because the labels are single
     digits 1-6 plus "unknown", but it would silently misorder e.g. a
     10+ class set, and it's a footgun worth removing regardless).

  4. MINIMUM-SAMPLES-PER-CLASS WARNING: a class with too few images to
     produce a meaningful validation split used to fail silently
     (n_val=1 leaves ~nothing to validate that class on). Now warns.

IMPORTANT (not a code bug, a data issue -- see conversation): the
training images sampled from this dataset are full occluded-gripper
scene renders, not the rectified, face-only, front-on crops that
cube_tag_pose_cnn_detector.py actually feeds the model at inference
time via warp_face_crop(). That distribution mismatch is very likely
the real cause of confident "unknown" misclassifications on real
digit faces, independent of anything in this script. Recommend
regenerating the dataset using the same face_corners_local() +
project_points() + warp_face_crop() rectification used at inference
time, so training and inference see the same kind of image. This
script trains correctly on whatever images/labels it's given --
it can't fix a data distribution problem for you.
"""

import os
import io
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models

# ==========================================
# 0. HARDCODED PATHS & CONFIGURATION
# ==========================================
CSV_PATH  = r"C:\Users\luvja\Desktop\updatedwithstand\dataset\labels.csv"
IMG_DIR   = r"C:\Users\luvja\Desktop\updatedwithstand\dataset\images"
SAVE_PATH = r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\face_cnn.pt"

IMG_SIZE      = 224
BATCH_SIZE    = 128 if torch.cuda.is_available() else 32  # Adaptive batch sizing
EPOCHS        = 40
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15
NUM_WORKERS   = 0      # 0 for Windows CUDA multiprocessing stability
SEED          = 42
LABEL_SMOOTH  = 0.05
MAX_GRAD_NORM = 5.0    # Gradient clipping threshold

MIN_SAMPLES_PER_CLASS_WARN = 20  # warn if any class has fewer than this

# Early Stopping & Delta Accuracy Threshold
PATIENCE         = 7      # N consecutive epochs allowed without threshold gain
ACC_DELTA_THRESH = 0.001  # Minimum accuracy gain required (0.1%)

USE_PRETRAINED_BACKBONE = True
FREEZE_BACKBONE_EPOCHS  = 3

# Photometric / Domain Randomization Parameters
BG_SWAP_PROB     = 0.6
NOISE_PROB       = 0.5
JPEG_PROB        = 0.4
MOTION_BLUR_PROB = 0.3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Hardware Acceleration Setup (TF32 + cuDNN Benchmark)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for param in model.features.parameters():
        param.requires_grad = trainable


# ==========================================
# 2. DOMAIN RANDOMIZATION TRANSFORMS
# ==========================================
class RandomBackgroundSwap:
    def __init__(self, prob=0.6, tolerance=28, feather=3):
        self.prob, self.tolerance, self.feather = prob, tolerance, feather

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        arr = np.array(img.convert("RGB")).astype(np.int16)
        h, w, _ = arr.shape
        corner_px = np.concatenate([
            arr[0:8, 0:8].reshape(-1, 3),
            arr[0:8, w-8:w].reshape(-1, 3),
            arr[h-8:h, 0:8].reshape(-1, 3),
            arr[h-8:h, w-8:w].reshape(-1, 3)
        ], axis=0)
        bg_color = np.median(corner_px, axis=0)

        dist = np.linalg.norm(arr - bg_color, axis=2)
        mask = (dist < self.tolerance).astype(np.float32)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(self.feather))
        mask = np.array(mask_img).astype(np.float32) / 255.0
        mask = mask[..., None]

        choice = random.random()
        if choice < 0.4:
            new_bg = np.tile(np.random.randint(0, 256, size=3), (h, w, 1)).astype(np.float32)
        elif choice < 0.75:
            c1 = np.random.randint(0, 256, size=3).astype(np.float32)
            c2 = np.random.randint(0, 256, size=3).astype(np.float32)
            ramp = np.linspace(0, 1, w)[None, :, None]
            new_bg = np.repeat(c1[None, None, :] * (1 - ramp) + c2[None, None, :] * ramp, h, axis=0)
        else:
            new_bg = np.random.randint(0, 256, size=(h, w, 3)).astype(np.float32)

        out = np.clip(arr.astype(np.float32) * (1 - mask) + new_bg * mask, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


class RandomMotionBlur:
    def __init__(self, prob=0.3, max_radius=3):
        self.prob, self.max_radius = prob, max_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, self.max_radius))) if random.random() <= self.prob else img


class RandomJPEGCompression:
    def __init__(self, prob=0.4, quality_range=(30, 80)):
        self.prob, self.quality_range = prob, quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality_range))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class AddGaussianNoise:
    def __init__(self, prob=0.5, std_range=(0.01, 0.05)):
        self.prob, self.std_range = prob, std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * random.uniform(*self.std_range), 0.0, 1.0)


def center_square_crop_pil(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


# ==========================================
# 3. DATASET & DATALOADERS
# ==========================================
def _sort_classes(raw_labels):
    """Numeric labels sorted numerically, any non-numeric labels (e.g.
    'unknown') appended afterward in alphabetical order. Avoids silently
    falling back to a pure alphabetical sort for the whole label set just
    because one label isn't numeric (which would misorder e.g. 1,10,2,3...)."""
    numeric, non_numeric = [], []
    for v in raw_labels:
        try:
            float(v)
            numeric.append(v)
        except ValueError:
            non_numeric.append(v)
    numeric.sort(key=lambda v: float(v))
    non_numeric.sort()
    return numeric + non_numeric


class CubeCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_col, self.label_col = self.df.columns[0], self.df.columns[1]

        raw_labels = self.df[self.label_col].astype(str).unique().tolist()
        self.classes = _sort_classes(raw_labels)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)


class IndexSubset(Dataset):
    """indices are DataFrame INDEX VALUES (from base.df.index[...]), not
    positional row numbers -- so lookups must use .loc, not .iloc. Using
    .iloc here would silently misalign images and labels the moment the
    DataFrame's index isn't a plain 0..N-1 RangeIndex (e.g. after
    filtering, concatenating CSVs, or dropping rows upstream)."""

    def __init__(self, base_dataset, indices, transform):
        self.base, self.indices, self.transform = base_dataset, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # Try the requested sample; on a corrupt/missing file, fall back to
        # the next sample in this subset instead of crashing the whole run.
        n = len(self.indices)
        for attempt in range(n):
            idx = self.indices[(i + attempt) % n]
            row = self.base.df.loc[idx]
            img_path = os.path.join(self.base.img_dir, str(row[self.base.img_col]))
            try:
                image = center_square_crop_pil(Image.open(img_path).convert("RGB"))
                label = self.base.class_to_idx[str(row[self.base.label_col])]
                return self.transform(image), label
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Skipping unreadable image '{img_path}': {e}")
                continue
        raise RuntimeError(f"All {n} samples in this subset failed to load; check IMG_DIR/CSV_PATH.")


def get_data_loaders(csv_file, img_dir, crop_size, batch_size, val_split, num_workers, seed):
    train_transforms = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        RandomBackgroundSwap(prob=BG_SWAP_PROB),
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        RandomMotionBlur(prob=MOTION_BLUR_PROB),
        transforms.RandomGrayscale(p=0.05),
        RandomJPEGCompression(prob=JPEG_PROB),
        transforms.ToTensor(),
        AddGaussianNoise(prob=NOISE_PROB),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    base_dataset = CubeCSVDataset(csv_file=csv_file, img_dir=img_dir)

    counts = base_dataset.df[base_dataset.label_col].astype(str).value_counts()
    print("\n--- Class Distribution ---")
    for cls_name in base_dataset.classes:
        n = int(counts.get(cls_name, 0))
        print(f"  Class {cls_name:>7}: {n:5d} ({n/len(base_dataset)*100:5.1f}%)")
        if n < MIN_SAMPLES_PER_CLASS_WARN:
            print(f"    [WARN] Class '{cls_name}' has only {n} samples (< {MIN_SAMPLES_PER_CLASS_WARN}); "
                  f"its validation split will be tiny/unreliable and it may not train well.")

    # Stratified Split (index-value based, matches IndexSubset's .loc lookups)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls_name in base_dataset.classes:
        idxs = base_dataset.df.index[base_dataset.df[base_dataset.label_col].astype(str) == cls_name].tolist()
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_split))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    # Weighted Sampler for Class Balance
    train_labels = base_dataset.df.loc[train_idx, base_dataset.label_col].astype(str).tolist()
    train_counts = pd.Series(train_labels).value_counts()
    per_class_weight = {c: 1.0 / max(train_counts.get(c, 1), 1) for c in base_dataset.classes}
    sample_weights = [per_class_weight[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
    }

    train_loader = DataLoader(
        IndexSubset(base_dataset, train_idx, train_transforms),
        sampler=sampler, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        IndexSubset(base_dataset, val_idx, val_transforms),
        shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, base_dataset.classes, counts


# ==========================================
# 4. EVALUATION & CONFIDENCE MONITORING
# ==========================================
def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct, total, loss_sum, total_conf = 0, 0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            total_conf += max_probs.sum().item()

            correct += (preds == labels).sum().item()
            total += inputs.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1

    avg_loss = loss_sum / max(total, 1)
    avg_acc = correct / max(total, 1)
    avg_conf = total_conf / max(total, 1)

    return avg_loss, avg_acc, avg_conf, confusion


def print_confusion(confusion, classes):
    print("       " + " ".join(f"{c:>7}" for c in classes) + "   <- predicted")
    for i, cls_name in enumerate(classes):
        row = confusion[i]
        acc = row[i] / max(row.sum(), 1) * 100
        print(f"  {cls_name:>7} " + " ".join(f"{v:7d}" for v in row) + f"   acc={acc:5.1f}%")


# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")
    if device.type == "cuda":
        print(f"--> GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"--> TF32 Acceleration: Enabled | cuDNN Benchmark: Enabled")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH non-existent: {CSV_PATH}")
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"IMG_DIR non-existent: {IMG_DIR}")

    train_loader, val_loader, classes, counts = get_data_loaders(
        csv_file=CSV_PATH, img_dir=IMG_DIR, crop_size=IMG_SIZE,
        batch_size=BATCH_SIZE, val_split=VAL_SPLIT, num_workers=NUM_WORKERS, seed=SEED
    )

    model = build_model(len(classes), pretrained=USE_PRETRAINED_BACKBONE).to(device)

    if hasattr(torch, "compile") and device.type == "cuda" and os.name != "nt":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    freqs = np.array([counts.get(c, 1) for c in classes], dtype=np.float32)
    weights = torch.tensor(freqs.sum() / (len(classes) * freqs), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if USE_PRETRAINED_BACKBONE and FREEZE_BACKBONE_EPOCHS > 0:
        set_backbone_trainable(model, False)
        print(f"--- Backbone frozen for initial {FREEZE_BACKBONE_EPOCHS} epochs ---")

    best_acc = 0.0
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        if USE_PRETRAINED_BACKBONE and epoch == FREEZE_BACKBONE_EPOCHS + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.3, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch + 1)
            print("--- Backbone unfrozen, fine-tuning full network ---")

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)

        scheduler.step()

        val_loss, val_acc, val_conf, confusion = evaluate(model, val_loader, device, len(classes))
        train_acc = train_correct / max(train_total, 1)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss/train_total:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | Avg Conf: {val_conf*100:.1f}%")

        acc_improvement = val_acc - best_acc

        if acc_improvement >= ACC_DELTA_THRESH:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0

            torch.save({
                "epoch": epoch,
                "classes": classes,
                "model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "arch": "mobilenet_v3_small",
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "best_acc": best_acc,
                "val_conf": val_conf,
            }, SAVE_PATH)
            print(f"  [Checkpoint Saved -> Val Acc: {val_acc*100:.2f}% | Avg Conf: {val_conf*100:.1f}% (Gain: +{acc_improvement*100:.2f}%)]")
            print_confusion(confusion, classes)
        else:
            no_improve_epochs += 1
            print(f"  [No threshold gain (Delta: {acc_improvement*100:+.2f}% < Target: {ACC_DELTA_THRESH*100:.2f}%). "
                  f"Counter: {no_improve_epochs}/{PATIENCE}]")

            if no_improve_epochs >= PATIENCE:
                print(f"\n[!] Early Stopping Triggered: Accuracy change fell below threshold "
                      f"({ACC_DELTA_THRESH*100:.2f}%) for {PATIENCE} consecutive epochs.")
                print(f"[!] Saved production model: {SAVE_PATH}")
                break

    print(f"\nTraining completed. Best Validation Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()