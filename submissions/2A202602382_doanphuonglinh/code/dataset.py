"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Hoàn thiện bởi: Doan Phuong Linh — 2A202602382
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df   = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df  = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1)."""
    images_dir = Path(images_dir)

    # 1. Số ảnh mỗi tập
    n = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    total = n["train"] + n["val"] + n["test"]
    print(f"[check_split] Số ảnh: train={n['train']}, val={n['val']}, test={n['test']}, tổng={total}")

    # 2. Giao từng cặp tập
    s_train = set(train_df["Filename"])
    s_val   = set(val_df["Filename"])
    s_test  = set(test_df["Filename"])
    overlap = {
        "train_val":  len(s_train & s_val),
        "train_test": len(s_train & s_test),
        "val_test":   len(s_val & s_test),
    }
    print(f"[check_split] Giao: {overlap}")
    assert overlap["train_val"]  == 0, "train ∩ val phải rỗng!"
    assert overlap["train_test"] == 0, "train ∩ test phải rỗng!"
    assert overlap["val_test"]   == 0, "val ∩ test phải rỗng!"

    # 3. Hợp 3 tập = 17509
    union = len(s_train | s_val | s_test)
    print(f"[check_split] Hợp 3 tập: {union}")
    assert union == 17509, f"Hợp 3 tập phải là 17509, đang là {union}"

    # 4. Mọi Filename tồn tại
    all_files = pd.concat([train_df, val_df, test_df])["Filename"].tolist()
    missing = [f for f in all_files if not (images_dir / f).exists()]
    print(f"[check_split] File thiếu: {len(missing)}")
    assert len(missing) == 0, f"Có {len(missing)} file thiếu, ví dụ: {missing[:5]}"

    # 5. Phân bố lớp mỗi tập
    per_class = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        per_class[name] = df["Label"].value_counts().sort_index().to_dict()

    print("\n[check_split] Phân bố lớp:")
    for name, d in per_class.items():
        print(f"  {name}: {d}")

    return {"n": n, "per_class": per_class, "overlap": overlap,
            "union": union, "missing": len(missing)}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform.

    aug: "basic" | "color" | "trivial" | "randaug"
    Train (basic): RandomResizedCrop + lật ngang + ToTensor + Normalize.
    Val/test: resize 256 -> CenterCrop(img_size) + ToTensor + Normalize.
    KHÔNG lật dọc cho ảnh cỏ dại (bất biến tự nhiên, lật dọc không hợp lệ).
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    # Train — augmentation tuỳ `aug`
    ops = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip()]

    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    elif aug == "trivial":
        ops.append(transforms.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")

    ops += [transforms.ToTensor(), normalize]
    return transforms.Compose(ops)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        fname = row["Filename"]
        label = int(row["Label"])
        img = Image.open(self.images_dir / fname).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, fname


def _worker_init_fn(worker_id):
    import numpy as np, random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    train=True: shuffle (hoặc sampler); train=False: giữ thứ tự df.
    sampler="balanced": WeightedRandomSampler với trọng số 1/(số ảnh của lớp).
    drop_last=True khi train để BatchNorm ổn định.
    """
    ds = DeepWeedsDataset(df, images_dir, transform)

    if sampler == "balanced" and train:
        counts = df["Label"].value_counts().to_dict()
        weights = df["Label"].map(lambda c: 1.0 / counts[c]).tolist()
        sampler_obj = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler_obj = None
        shuffle = train

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train and len(ds) > batch_size,
        worker_init_fn=_worker_init_fn,
    )
    return loader
