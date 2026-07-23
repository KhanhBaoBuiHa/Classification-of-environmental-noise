# 🔊 Environmental Sound Classification (ESC-50)

Phân loại âm thanh môi trường sử dụng Log-Mel Spectrogram và Convolutional Neural Network (CNN), với baseline so sánh MFCC + kNN.

---

## 📌 Tổng quan

Dự án xây dựng pipeline hoàn chỉnh để phân loại 50 lớp âm thanh thực tế từ bộ dữ liệu **ESC-50**, bao gồm:

- Tiền xử lý tín hiệu âm thanh thô → Log-Mel Spectrogram 3 kênh
- Huấn luyện CNN với data augmentation (SpecAugment)
- Đánh giá và giải thích dự đoán qua Grad-CAM
- So sánh với baseline truyền thống: MFCC + kNN

---

## 🗂️ Cấu trúc dự án

```
Classification-of-environmental-noise-main/
│
├── CNN.ipynb                         # Pipeline chính: CNN trên Log-Mel Spectrogram
├── baseline_KNN+MFCC.ipynb           # Baseline: kNN trên đặc trưng MFCC
├── Nhom16_BaoCaoCuoiKy_NDM.pdf.pdf   # Báo cáo cuối kỳ
└── README.md
```

---

## 📦 Dataset

**ESC-50** — Environmental Sound Classification Dataset

- 2.000 audio clips, 50 lớp âm thanh môi trường
- Mỗi clip dài 5 giây, chia thành 5 fold (dùng cho cross-validation)
- Download: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

Sau khi tải về, đặt theo cấu trúc:

```
ESC-50-master/
├── audio/          # 2000 file .wav
└── meta/
    └── esc50.csv   # Metadata (filename, category, fold)
```

---

## ⚙️ Yêu cầu môi trường

Dự án chạy trên **Google Colab** (khuyến nghị, có GPU).

```
tensorflow >= 2.x
librosa
numpy
pandas
scikit-learn
matplotlib
tqdm
soundfile
scipy
```

Cài đặt nhanh:

```bash
pip install librosa soundfile tqdm
```

---

## 🚀 Hướng dẫn chạy

### 1. CNN (mô hình chính) — `CNN.ipynb`

| Giai đoạn | Nội dung |
|-----------|----------|
| **0** | Mount Google Drive, cấu hình đường dẫn |
| **1** | Load metadata ESC-50, chia train/val/test theo fold (fold 5 = test) |
| **2** | Trích xuất đặc trưng: Log-Mel + Delta + Delta-Delta (3 kênh), chuẩn hoá per-channel, cache `.npz` |
| **3** | Xây dựng CNN, huấn luyện với SpecAugment, lưu checkpoint tốt nhất |
| **3.5** | Đánh giá trên tập test: Accuracy, Top-5 Accuracy, Confusion Matrix, Classification Report |
| **4** | Explainable AI: Grad-CAM trực quan hoá vùng thời gian–tần số CNN tập trung |

### 2. Baseline — `baseline_KNN+MFCC.ipynb`

- Trích xuất MFCC từ file âm thanh
- Huấn luyện và đánh giá mô hình kNN
- Kết quả dùng để so sánh với CNN

---

## 🧠 Kiến trúc mô hình

**Input:** Log-Mel Spectrogram 3 kênh `(H × W × 3)` — kênh 1: Log-Mel, kênh 2: Delta, kênh 3: Delta-Delta

**CNN Architecture:**
```
Conv2D → BatchNorm → ReLU → MaxPool
Conv2D → BatchNorm → ReLU → MaxPool
Conv2D → BatchNorm → ReLU → MaxPool
GlobalAveragePooling2D
Dropout
Dense(50, softmax)
```

**Optimizer:** AdamW + CosineDecay learning rate schedule  
**Loss:** Categorical Crossentropy + Label Smoothing  
**Augmentation:** SpecAugment (Time Masking + Frequency Masking)

---

## 📊 Kết quả

| Mô hình | Đặc trưng | Test Accuracy |
|---------|-----------|--------------|
| kNN | MFCC | ~baseline |
| **CNN** | **Log-Mel 3-channel** | **cao hơn đáng kể** |

> Kết quả chi tiết (confusion matrix, classification report, top confusions) được lưu trong thư mục `models/eval_artifacts/` sau khi chạy notebook.

---

## 🔍 Giải thích mô hình (Grad-CAM)

Notebook CNN bao gồm phần **Grad-CAM** (Giai đoạn 4) để trực quan hoá vùng thời gian–tần số mà CNN chú ý khi phân loại. Hai trường hợp được hiển thị:
- Dự đoán **đúng**: vùng heatmap tập trung vào đặc trưng âm thanh nổi bật
- Dự đoán **sai**: phân tích nguyên nhân nhầm lẫn giữa các lớp tương tự

---

## 📝 Ghi chú

- **Split strategy:** Theo fold chuẩn ESC-50 (fold 5 = test, fold 1–4 = train/val) để đảm bảo tái lập và so sánh công bằng.
- **Caching:** Đặc trưng được cache ra file `.npz` để tránh trích xuất lại khi chạy nhiều lần.
- **Seed cố định:** `SEED = 42` cho TensorFlow và NumPy để tái lập kết quả.

---

## 👥 Nhóm thực hiện

Nhóm 16 — Báo cáo cuối kỳ môn học  
Chi tiết xem tại: `Nhom16_BaoCaoCuoiKy_NDM.pdf.pdf`
