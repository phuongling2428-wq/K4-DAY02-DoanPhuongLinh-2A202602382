"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hoàn thiện bởi: Doan Phuong Linh — 2A202602382
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    init: "scratch" | "frozen" | "finetune"
    """
    if init == "scratch":
        pretrained = False

    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head. BN của backbone phải để eval mode."""
    # Lấy tên tham số của head
    head = model.get_classifier()
    head_param_ids = {id(p) for p in head.parameters()}

    for p in model.parameters():
        if id(p) not in head_param_ids:
            p.requires_grad = False

    # Đặt BN của backbone ở eval mode — sẽ được duy trì trong train loop
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm (slide Day 2, trang 52).

    - backbone ndim > 1   : lr_backbone, wd = weight_decay
    - backbone ndim <= 1  : lr_backbone, wd = 0  (norm + bias)
    - head                : lr_head,     wd = weight_decay
    """
    head = model.get_classifier()
    head_param_ids = {id(p) for p in head.parameters()}

    backbone_decay, backbone_no_decay, head_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            head_params.append(p)
        elif p.ndim > 1:
            backbone_decay.append(p)
        else:
            backbone_no_decay.append(p)

    groups = [
        {"params": backbone_decay,    "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_params,       "lr": lr_head,     "weight_decay": weight_decay},
    ]
    # Bỏ nhóm rỗng (ví dụ khi freeze backbone → backbone groups rỗng)
    groups = [g for g in groups if len(g["params"]) > 0]
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Ưu tiên ptflops; fallback fvcore; fallback trả NaN.
    Lưu ý: ptflops báo MAC, timm hay báo FLOPs = 2*MAC.
    """
    model.eval()
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, img_size, img_size),
            as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        return macs / 1e9
    except ImportError:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, 3, img_size, img_size)
        flops = FlopCountAnalysis(model, x).total()
        return flops / 2 / 1e9
    except ImportError:
        pass

    print("[count_gmacs] Không có ptflops/fvcore. Cài: pip install ptflops")
    return float("nan")
