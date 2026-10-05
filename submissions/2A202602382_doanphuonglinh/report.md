# Báo cáo Lab Day 2 — DeepWeeds Classification

**Sinh viên:** Doan Phuong Linh — MSSV 2A202602382
**Ngày nộp:** 05/10/2026
**Repository:** github.com/phuongling2428-wq/K4-DAY02-DoanPhuongLinh-2A202602382

---

## 1. Tóm tắt

Bài lab so sánh 5 backbone (ResNet-50, ResNeXt-50, ConvNeXt-Tiny, Swin-Tiny, EfficientNet-B0) trên DeepWeeds (17.509 ảnh, 9 lớp, mất cân bằng 52% Negative) với cùng công thức nền.

Kết quả:
- ConvNeXt-Tiny tốt nhất: macro-F1 val = 0.9650 (ResNet-50: 0.8170, +0.148)
- 11 ablation / 5 trục: Label Smoothing (+0.005) và TrivialAugment (+0.004) cải thiện rõ
- T12 combo = ConvNeXt + TrivialAug + LS: 0.9733 val
- Chung kết (2 seed F01+F02): test macro-F1 = 0.9698 ± 0.0003, top-1 = 0.9758 ± 0.0004
- Temperature scaling giảm ECE từ 0.0851 → 0.0072 (91%)
- p95 latency batch-1 = 11.3 ms (FP32) — đủ cho robot real-time

---

## 2. Dữ liệu và thiết lập

### 2.1 Dataset DeepWeeds

- 17.509 ảnh RGB 256x256, 9 lớp (8 loài cỏ + Negative)
- Mất cân bằng: Negative 9.106 (~52%); mỗi loài 1.009–1.125 ảnh
- Chia fold 0 của tác giả: 10.501 train / 3.501 val / 3.507 test
- Giao train/val/test = 0, hợp = 17.509

### 2.2 Chỉ số đánh giá

- Chỉ số chính: macro-F1
- Phụ: top-1, balanced acc, F1 từng lớp, ECE (15 bin)
- Test chỉ chạy 1 lần cho mỗi seed; mọi quyết định trên val

### 2.3 Công thức nền (T00)

| Thành phần | Giá trị |
|---|---|
| Init | ImageNet pretrained, head 9 lớp |
| Aug | RandomResizedCrop(224) + HorizontalFlip |
| Optimizer | AdamW (backbone 1e-4, head 1e-3, wd 0.05) |
| LR schedule | Warmup 1 epoch + Cosine |
| Loss | CrossEntropy |
| Batch | 64 |
| Epoch | 12 |
| AMP | True |

### 2.4 Phần cứng

- GPU: Tesla T4 15GB
- PyTorch 2.11.0+cu130, timm 1.0.29, Python 3.13

---

## 3. So sánh backbone

| exp_id | Backbone | Macro-F1 val | Params (M) | GMAC | Time/epoch (s) |
|---|---|---|---|---|---|
| B03 | ConvNeXt-Tiny | 0.9650 | 27.83 | 4.48 | 63 |
| B04 | Swin-Tiny | 0.9529 | 27.53 | 4.38 | 73 |
| B02 | ResNeXt-50 | 0.8370 | 25.00 | 4.20 | 65 |
| B01 | ResNet-50 | 0.8170 | 23.53 | 4.13 | 59 |
| B05 | EfficientNet-B0 | 0.8048 | 4.02 | 0.39 | 50 |

**Nhận xét:**
- ConvNeXt-Tiny thắng tuyệt đối (+0.148 so ResNet-50)
- Swin-Tiny (transformer) tốt nhưng chậm hơn
- ResNeXt-50 > ResNet-50 (+0.020) — cardinality giúp
- EfficientNet-B0 nhẹ nhất nhưng F1 thấp
- Thứ hạng khác ImageNet — do dataset nhỏ 10k ảnh, inductive bias CNN có lợi

**Chọn ConvNeXt-Tiny + EfficientNet-B0 cho Bước 2/3.**

---

## 4. Công thức huấn luyện (ablation)

| exp_id | Trục | Thay đổi | Macro-F1 | Delta vs T00 |
|---|---|---|---|---|
| T12 | B+C | TrivialAug + LS(0.1) | 0.9733 | +0.0083 |
| T07 | C | LabelSmoothing 0.1 | 0.9700 | +0.0050 |
| T04 | B | TrivialAugment | 0.9685 | +0.0035 |
| T08 | C | Focal Loss (gamma=2) | 0.9669 | +0.0019 |
| T09 | C | Class-Weighted CE | 0.9651 | +0.0001 |
| T11 | F | EMA 0.999 | 0.9643 | -0.0007 |
| T10 | D | Balanced Sampler | 0.9640 | -0.0010 |
| T05 | B | Mixup | 0.9638 | -0.0012 |
| T03 | B | Color Jitter | 0.9597 | -0.0053 |
| T01 | A | Frozen | 0.8539 | -0.1111 |
| T02 | A | Scratch | 0.3286 | -0.6364 |

**Phân tích:**
- Trục A (khởi tạo) QUAN TRỌNG NHẤT: Pretrained bắt buộc. Scratch = -64 điểm, frozen = -11 điểm.
- Trục C (loss): Label Smoothing tốt nhất (+0.005, không overfit). Focal có overfit nhẹ. Class weight không giúp.
- Trục B (aug): TrivialAugment +0.004. Mixup/Color jitter không giúp.
- Trục D (sampler): Balanced sampler không giúp.
- Trục F (EMA): Không giúp vì model đã overfit.
- Kết hợp T12: TrivialAug + LS = +0.0083 → cộng dồn gần hoàn hảo.

---

## 5. Suy luận

| exp_id | Phương pháp | Macro-F1 val | ECE | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| I00 | 1-view 224 | 0.9733 | 0.0891 | 6.26 | 11.27 |
| I01 | TTA hflip | 0.9737 | 0.0907 | - | - |
| I02 | Multi-scale | 0.9744 | 0.1087 | - | - |
| I04 | res=256 | 0.9749 | 0.1059 | - | - |
| I04 | res=288 | 0.9738 | 0.1308 | - | - |
| I04 | res=320 | 0.9737 | 0.1544 | - | - |
| I07 | Temperature T=0.6263 | 0.9733 | 0.0040 | - | - |
| I08 | FP16 b1 | 0.9733 | - | 8.12 | 9.04 |
| I08 | FP32 b1 | 0.9733 | - | 6.26 | 11.27 |

**Nhận xét:**
- I04 res=256 cải thiện accuracy (+0.0016) — FixRes effect
- I02 multi-scale +0.0011 — tốt nhưng 3x chi phí
- I01 TTA flip +0.0004 — dưới nhiễu
- I07 Temperature scaling — ECE giảm 91% (0.0891 → 0.0040). T = 0.6263 < 1 → model under-confident
- FP16 batch-1 chậm hơn FP32 (8.12 vs 6.26ms) — T4 không tối ưu AMP batch nhỏ
- FP16 ổn định hơn ở p95 (9.04 vs 11.27ms)
- Throughput batch-32 = 614 ảnh/s

---

## 6. Cấu hình tốt nhất

### 6.1 Cấu hình chung kết

- backbone: convnext_tiny (pretrained ImageNet)
- init: finetune
- aug: TrivialAugmentWide
- loss: CrossEntropy + label_smoothing=0.1
- optimizer: AdamW (backbone lr=1e-4, head lr=1e-3, wd=0.05)
- lr_schedule: Warmup 1 epoch + Cosine
- epochs: 12, batch_size: 64, amp: True
- inference: 1-view 224 + Temperature scaling T=0.6263

### 6.2 Kết quả test (2 seed F01+F02)

| Chỉ số | Mean ± Std |
|---|---|
| Top-1 | 0.9758 ± 0.0004 |
| Macro-F1 | 0.9698 ± 0.0003 |
| Balanced acc | 0.9707 ± 0.0040 |
| ECE (sau TS) | 0.0072 |
| NLL | 0.1651 |

### 6.3 F1 từng lớp (test)

| Lớp | Precision | Recall | F1 | Mốc bài báo |
|---|---|---|---|---|
| Chinee apple | 0.971 | 0.942 | 0.956 | 88.5% |
| Lantana | 0.981 | 0.960 | 0.970 | - |
| Parkinsonia | 0.981 | 0.983 | 0.982 | 97.2% |
| Parthenium | 0.985 | 0.973 | 0.979 | - |
| Prickly acacia | 0.935 | 0.979 | 0.956 | - |
| Rubber vine | 0.971 | 0.980 | 0.975 | - |
| Siam weed | 0.959 | 0.991 | 0.975 | - |
| Snake weed | 0.956 | 0.946 | 0.951 | 88.8% |
| Negative | 0.984 | 0.982 | 0.983 | 97.6% |

Vượt mốc bài báo ở cả 2 lớp khó nhất.

### 6.4 Phân tích lỗi

Confusion matrix: curves/confusion_matrix_final.png. Nhầm lẫn chính ở cặp Chinee apple ↔ Snake weed (~3-5%/chiều). Nguyên nhân: 2 loài có hình dạng tương tự khi chụp ở góc/ánh sáng khác nhau.

---

## 7. Kết luận và khuyến nghị

**Backbone tốt nhất:** ConvNeXt-Tiny (0.9650 val). Đánh đổi: 27.83M params, 4.48 GMAC, 63s/epoch.

**Yếu tố đóng góp nhiều nhất:**
1. Pretrained: +0.636 (scratch → finetune)
2. Label Smoothing: +0.005
3. TrivialAugment: +0.004

**Suy luận tốt nhất:**
- Accuracy: res=256 (+0.0016)
- Calibration: Temperature scaling (ECE 0.085 → 0.007)

**Triển khai robot (30-100ms/khung):**
- Accuracy cao: ConvNeXt-T + TrivialAug + LS + T-scaling, p95 = 11.3ms
- Nhanh: EfficientNet-B0, p95 ~5-7ms, F1 ~0.80

---

## 8. Hạn chế và việc tiếp theo

### Hạn chế

1. Chỉ 2 seed chung kết (F01, F02). Rubric yêu cầu >= 3. Do Colab hết quota GPU và Kaggle lỗi.
2. Chỉ fold 0, không kiểm tra độ ổn định qua fold.
3. Delta test âm (-0.0018) so với mốc T00 — do test set nhỏ (3507 ảnh) + T00 may mắn. Delta val = +0.0066 rõ ràng.
4. Chia ngẫu nhiên không theo địa điểm → kết quả có thể lạc quan.
5. 12 epoch ít hơn bài báo (100 epoch + augmentation mạnh).

### Việc tiếp theo

1. Chạy đủ 3+ seed chung kết
2. Chạy 5 fold
3. Linear probe DINOv2
4. Knowledge distillation ConvNeXt-T → EfficientNet-B0
5. Grad-CAM phân tích lớp Chinee apple/Snake weed

---

## 9. Phụ lục

### 9.1 Danh sách exp_id

- B01–B05: 5 backbone, công thức T00
- T01–T12: 11 ablation / 5 trục
- I00–I08: 8 phương pháp suy luận trên T12
- F01–F02: 2 seed chung kết (T12)

### 9.2 Cấu trúc thư mục nộp bài

submissions/2A202602382_doanphuonglinh/
- README.md
- results.xlsx (7 sheet)
- report.md
- curves/ (20+ ảnh)
- predictions/ (F01, F02, T00, T12)
- code/ (dataset.py, model.py, losses.py, train.py, inference.py, benchmark.py)

### 9.3 Link

- Colab notebook: [điền link của bạn]
- Data: Zenodo 10.5281/zenodo.7939060
- Labels: github.com/AlexOlsen/DeepWeeds
- GitHub: github.com/phuongling2428-wq/K4-DAY02-DoanPhuongLinh-2A202602382

---

*Báo cáo hoàn thành ngày 05/10/2026.*
