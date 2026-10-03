"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Hoàn thiện bởi: Doan Phuong Linh — 2A202602382
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

# Cho phép import eval.py từ thư mục gốc repo
_THIS_DIR = Path(__file__).resolve().parent
for _p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]:
    if (_p / "eval.py").exists():
        sys.path.insert(0, str(_p))
        break

from eval import save_predictions, compute_metrics  # noqa: E402

import dataset as ds  # noqa: E402
import model as model_module  # noqa: E402
import losses as losses_module  # noqa: E402


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Cân bằng giữa tốc độ và tái lập: benchmark=False để ổn định hơn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, cfg: Config):
    groups = model_module.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính + cosine về 0 theo bước."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số, xử lý cả buffer (BN)."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    if cfg.init == "frozen":
        # Giữ BN của backbone ở eval mode
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    total_loss = 0.0
    n = 0
    last_lr = optimizer.param_groups[0]["lr"]

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.amp):
            if cfg.mix is not None:
                imgs_mix, targets = losses_module.mix_batch(
                    imgs, labels, alpha=cfg.mix_alpha, mode=cfg.mix
                )
                logits = model(imgs_mix)
                loss = losses_module.mixed_loss(criterion, logits, targets)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        last_lr = optimizer.param_groups[0]["lr"]

    return {"train_loss": total_loss / max(1, n), "lr": last_lr}


def evaluate(model, loader, criterion, device):
    model.eval()
    all_files, all_y, all_logits = [], [], []
    total_loss, n = 0.0, 0

    with torch.inference_mode():
        for imgs, labels, fnames in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=True):
                logits = model(imgs)
                loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            all_logits.append(logits.float().cpu().numpy())
            all_y.append(labels.cpu().numpy())
            all_files.extend(list(fnames))

    return (
        all_files,
        np.concatenate(all_y, axis=0),
        np.concatenate(all_logits, axis=0),
        total_loss / max(1, n),
    )


def _compute_macro_f1(y_true, logits):
    y_pred = logits.argmax(axis=1)
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1 val")
    axes[1].set_title("Macro-F1 val"); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [h["lr"] for h in history])
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].set_title("Learning rate"); axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, fold=cfg.fold)
    split_info = ds.check_split(train_df, val_df, test_df, cfg.images_dir)
    with open(rd / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # 3. Loaders
    tf_train = ds.build_transforms(train=True,  img_size=cfg.img_size, aug=cfg.aug)
    tf_eval  = ds.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds.make_loader(train_df, cfg.images_dir, tf_train,
                                  batch_size=cfg.batch_size, train=True,
                                  sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = ds.make_loader(val_df, cfg.images_dir, tf_eval,
                                batch_size=cfg.batch_size, train=False,
                                sampler=None, num_workers=cfg.num_workers)

    steps_per_epoch = max(1, len(train_loader))

    # 4. Model + loss + opt + sched + scaler + ema
    model = model_module.build_model(cfg.backbone, pretrained=True, num_classes=9,
                                     drop_rate=cfg.drop_rate, init=cfg.init).to(device)

    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().tolist()
        weight = losses_module.class_weights(counts,
                                             beta=cfg.class_weight_beta or 0.0).to(device)
        criterion = losses_module.build_criterion("ce_weighted", weight=weight)
    elif cfg.loss == "ls":
        criterion = losses_module.build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        criterion = losses_module.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = losses_module.build_criterion("ce")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = GradScaler(device="cuda", enabled=cfg.amp)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay else None

    # 5. Vòng huấn luyện
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt = rd / "best.pt"
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                             scaler, cfg, device, ema)

        # Đánh giá bằng trọng số EMA nếu có
        eval_model = model
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            eval_model = model

        fnames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_macro_f1 = _compute_macro_f1(y_true, logits)

        if ema is not None:
            model.load_state_dict(backup)

        epoch_time = time.time() - ep_t0
        history.append({
            "epoch": epoch,
            "train_loss": tr["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "lr": tr["lr"],
            "epoch_time_s": epoch_time,
        })
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        print(f"[{cfg.exp_id} seed{cfg.seed}] epoch {epoch:2d}/{cfg.epochs} "
              f"train_loss={tr['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_macro_f1={val_macro_f1:.4f} time={epoch_time:.1f}s")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt)

    total_time = time.time() - t0

    # 6. Load best, predict val, save
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    fnames, y_true, logits, val_loss = evaluate(model, val_loader, criterion, device)

    pred_path(cfg, "val").parent.mkdir(parents=True, exist_ok=True)
    probs_val = torch.from_numpy(logits).softmax(dim=1).numpy()
    save_predictions(pred_path(cfg, "val"), fnames, y_true, probs_val)

    # 7. Test (chỉ Bước 4)
    test_macro_f1 = None
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(test_df, cfg.images_dir, tf_eval,
                                     batch_size=cfg.batch_size, train=False,
                                     sampler=None, num_workers=cfg.num_workers)
        fnames_t, y_true_t, logits_t, _ = evaluate(model, test_loader, criterion, device)
        probs_test = torch.from_numpy(logits_t).softmax(dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), fnames_t, y_true_t, probs_test)
        test_macro_f1 = _compute_macro_f1(y_true_t, logits_t)
        print(f"[{cfg.exp_id}] TEST macro-F1 = {test_macro_f1:.4f}")

    # 8. History + curves
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(rd / "history.csv", index=False)

    curves_dir = Path(cfg.out_dir).parent / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(history, curves_dir / f"{cfg.exp_id}_seed{cfg.seed}.png",
                title=f"{cfg.exp_id} seed{cfg.seed} — {cfg.backbone}")

    params_m = model_module.count_params(model)
    try:
        gmacs = model_module.count_gmacs(model, img_size=cfg.img_size)
    except Exception:
        gmacs = float("nan")

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "macro_f1_val": best_macro_f1,
        "val_loss": val_loss,
        "test_macro_f1": test_macro_f1,
        "total_time_s": total_time,
        "epoch_time_s": total_time / cfg.epochs,
        "params_m": params_m,
        "gmacs": gmacs,
    }
    with open(rd / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{cfg.exp_id} seed{cfg.seed}] DONE — best_epoch={best_epoch} "
          f"macro_f1_val={best_macro_f1:.4f} time={total_time:.1f}s")
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    field_types = {f.name: f.type for f in fields(Config)}
    out = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Thiếu '=' trong: {item}")
        k, v = item.split("=", 1)
        if k not in field_types:
            raise KeyError(f"Key không có trong Config: {k}")
        if v.lower() in ("none", "null"):
            out[k] = None
            continue
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[], help="KEY=VALUE ...")
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    result = run(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
