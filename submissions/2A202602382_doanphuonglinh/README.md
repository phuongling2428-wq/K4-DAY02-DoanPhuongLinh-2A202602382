# Lab Day 2 — DeepWeeds Classification

**Sinh viên:** Doan Phuong Linh
**MSSV:** 2A202602382

## Cấu trúc

- `code/` — Code đã hoàn thiện (dataset.py, model.py, losses.py, train.py, inference.py, benchmark.py)
- `results.xlsx` — 7 sheet: Backbones, Training, Inference, Latency, Final, PerClass, Summary
- `report.md` — Báo cáo đầy đủ (9 mục theo GUIDE)
- `curves/` — Ảnh biểu đồ training của mỗi thí nghiệm
- `predictions/` — Predictions test/val của F01, F02, T00, T12

## Kết quả chính

| Chỉ số | Giá trị |
|---|---|
| Test macro-F1 | 0.9698 ± 0.0003 (2 seed) |
| Test top-1 | 0.9758 ± 0.0004 |
| ECE (sau T-scaling) | 0.0072 |
| p95 latency batch-1 | 11.3 ms (FP32) |

## Cách chạy lại

### Môi trường

- Google Colab (T4 GPU)
- PyTorch 2.11.0+cu130, timm 1.0.29, Python 3.13

### Bước 1: Tải data

    wget https://zenodo.org/records/7939060/files/images.zip
    md5sum images.zip
    unzip -q images.zip -d data/

### Bước 2: Setup

    pip install -q timm openpyxl ptflops
    import sys; sys.path.insert(0, "code")
    import train

### Bước 3: Chạy 1 experiment

    cfg = train.Config(
        exp_id="F01", seed=0,
        backbone="convnext_tiny",
        aug="trivial", loss="ls", label_smoothing=0.1,
        epochs=12, batch_size=64,
        images_dir="data", labels_dir="data/labels",
        out_dir="runs", pred_dir="predictions",
        save_test_predictions=True,
    )
    result = train.run(cfg)

## Liên kết

- Colab notebook: https://colab.research.google.com/drive/10J3sqicufPyTbzuIkfNNYvv11HCqqUu7?usp=sharing
- Data: Zenodo 10.5281/zenodo.7939060
- Labels: github.com/AlexOlsen/DeepWeeds
