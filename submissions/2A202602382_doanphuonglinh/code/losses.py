"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Hoàn thiện bởi: Doan Phuong Linh — 2A202602382
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls", "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    elif kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", None))
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        if weight is None:
            raise ValueError("ce_weighted cần weight=tensor")
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"kind không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing.

    Dùng torch.nn.CrossEntropyLoss(label_smoothing=eps) — torch đã cài đúng
    công thức q'(k) = (1 - eps) * 1[k == y] + eps / K.
    eps = 0 phải cho đúng CE thường.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, target):
        return self.ce(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    gamma = 0 phải cho đúng CE (đã kiểm tra bằng unit test).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)              # [B, K]
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        pt = log_pt.exp()                                     # [B]
        loss = -(1 - pt) ** self.gamma * log_pt               # [B]

        if self.alpha is not None:
            at = self.alpha.to(logits.device)[target]
            loss = at * loss

        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c = 1 / n_c, chuẩn hoá trung bình = 1.
    - beta > 0: class-balanced: w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng = K.
    """
    counts = np.asarray(counts, dtype=np.float64)
    K = len(counts)

    if beta == 0.0:
        w = 1.0 / counts
        w = w / w.mean()
    else:
        effective = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / effective
        w = w / w.sum() * K

    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    B, C, H, W = x.shape
    device = x.device
    perm = torch.randperm(B, device=device)

    lam = float(np.random.beta(alpha, alpha))

    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        # Chọn tỉ lệ cạnh hộp: sqrt(1 - lam)
        cut_rat = float(np.sqrt(1.0 - lam))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Tâm hộp
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

        # Điều chỉnh lam theo DIỆN TÍCH thực của hộp sau khi cắt biên
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")

    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    `targets` là tuple (y_a, y_b, lam).
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
