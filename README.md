# 📊 ClusTabNet - Nhận Dạng Cấu Trúc Bảng

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Table%20Structure%20Recognition-purple)

## 📋 Giới Thiệu

Đồ án triển khai thuật toán **ClusTabNet** (Clustering-based Table Network) để nhận dạng cấu trúc bảng (Table Structure Recognition - TSR). Hệ thống sử dụng kiến trúc **Deep Learning** kết hợp **Transformer Encoder** và phương pháp **Clustering** để tái tạo chính xác cấu trúc bảng phức tạp từ dữ liệu OCR.

### 🎯 Mục Tiêu

- Nhận dạng và phân tích cấu trúc bảng từ ảnh tài liệu
- Xác định mối quan hệ giữa các ô: cùng hàng, cùng cột, cùng cell, header
- Hỗ trợ xử lý bảng có ô gộp (spanning cells)

---

## 🌟 Tính Năng Nổi Bật

| Tính năng | Mô tả |
|-----------|-------|
| **Kiến trúc Transformer** | Sử dụng Self-Attention để học mối quan hệ không gian và ngữ nghĩa giữa các từ |
| **Phương pháp Clustering** | Dự đoán ma trận kề (adjacency matrix) để nhóm các token thành hàng, cột, ô |
| **Đa nhiệm (Multi-task)** | 5 đầu ra độc lập cho các bài toán khác nhau |
| **Trực quan hóa** | Công cụ visualization overlay cấu trúc dự đoán lên ảnh gốc |

### 📊 Các Tác Vụ Nhận Dạng

Mô hình thực hiện 5 tác vụ song song:

1. **Same Row** - Xác định các token thuộc cùng một hàng
2. **Same Column** - Xác định các token thuộc cùng một cột  
3. **Same Cell** - Xác định các token thuộc cùng một ô
4. **Same Header** - Nhận diện các ô thuộc phần header của bảng
5. **Spanning Cell** - Phát hiện các ô bị gộp (merge cells)

---

## 📂 Cấu Trúc Dự Án

```
ClusTabNet/
├── 📁 model/                      # Chứa weights mô hình đã huấn luyện
│   └── model_weitghloss_10epoch.pth
│
├── 📁 dataset/                    # Dữ liệu thử nghiệm (PubTables-1M)
│   └── pubtables_mini_test/
│       ├── data_ocr/              # Dữ liệu OCR (JSON: words + bounding box)
│       ├── images/                # Ảnh bảng gốc
│       └── ocr_gt/                # Ground truth cho đánh giá
│
├── 📄 model.py                    # Kiến trúc ClusTabNet
├── 📄 dataset.py                  # Xử lý dữ liệu và tạo Adjacency Labels
├── 📄 train.py                    # Script huấn luyện mô hình
├── 📄 evaluate.py                 # Đánh giá mô hình (Pixel-Level F1)
├── 📄 evaluate_coco.py            # Đánh giá theo chuẩn COCO metrics
├── 📄 visualize.py                # Inference và trực quan hóa kết quả
├── 📄 adjacency_matrix.py         # Xử lý ma trận kề
├── 📄 build_vocab.py              # Xây dựng từ điển
├── 📄 vocab.json                  # Từ điển đã xây dựng
├── 📄 requirements.txt            # Thư viện phụ thuộc
├── 📄 demo.ipynb                  # Jupyter notebook demo
└── 📄 README.md                   # Tài liệu hướng dẫn
```

---

## 🏗️ Kiến Trúc Mô Hình

Mô hình ClusTabNet bao gồm 3 thành phần chính:

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT (OCR Data)                        │
│              Words + Bounding Boxes + [Optional] CNN        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ClusTabEmbedding                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Text Embedding  │ +│ BBox Embedding  │ +│CNN Features │  │
│  │     (70%)       │  │     (30%)       │  │ (Optional)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Custom Transformer Encoder                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Multi-Head Self-Attention + Feed-Forward Network   │    │
│  │              (N layers × d_model=256)               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Clustering Heads                           │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │Same Row  │Same Col  │Same Cell │ Header   │ Spanning │   │
│  │  Head    │  Head    │  Head    │  Head    │   Head   │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
│              Output: 5 × Adjacency Matrix (N×N)              │
└─────────────────────────────────────────────────────────────┘
```

### Chi Tiết Các Thành Phần

#### 1. ClusTabEmbedding
- **Text Embedding**: Chuyển đổi từ (token) thành vector thông qua embedding layer
- **BBox Embedding**: Mã hóa vị trí không gian (xmin, ymin, xmax, ymax) đã chuẩn hóa
- **Tỷ lệ kết hợp**: 70% Text + 30% Position

#### 2. Transformer Encoder  
- Sử dụng cơ chế **Multi-Head Self-Attention** để học mối quan hệ giữa các từ
- **Feed-Forward Network** với activation GELU
- **Layer Normalization** và **Residual Connection**
- Cấu hình mặc định: `d_model=256`, `n_head=4`, `num_layers=3`

#### 3. Clustering Heads
- 5 nhánh **Fully Connected Network** độc lập
- Mỗi nhánh trả về ma trận `N × N` (với N là số token)
- Giá trị ma trận biểu thị xác suất hai token thuộc cùng nhóm

---

## 🛠️ Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- CUDA 11.0+ (khuyến nghị cho GPU training)
- RAM: tối thiểu 8GB

### Cài Đặt Thư Viện

```bash
# Clone repository
git clone https://github.com/TrinhPhat2082005/ClusTabNet.git
cd ClusTabNet

# Cài đặt dependencies
pip install -r requirements.txt
```

### Thư Viện Phụ Thuộc

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| PyTorch | ≥1.10.0 | Deep Learning framework |
| NumPy | ≥1.21.0 | Xử lý mảng số học |
| OpenCV | - | Xử lý ảnh |
| Matplotlib | ≥3.5.0 | Trực quan hóa |
| SciPy | - | Thuật toán khoa học |
| tqdm | ≥4.62.0 | Progress bar |

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chạy Demo Trực Quan Hóa

```bash
python visualize.py
```

Kết quả sẽ hiển thị ảnh gốc với overlay cấu trúc bảng được tô màu:
- 🔴 **Đỏ**: Đường viền hàng (Same Row)
- 🟢 **Xanh lá**: Đường viền cột (Same Column)  
- 🟠 **Cam**: Đường viền Header
- 🔵 **Xanh dương**: Đường viền Cell

### 2. Huấn Luyện Mô Hình

```bash
python train.py
```

### 3. Đánh Giá Mô Hình

```bash
# Đánh giá Pixel-Level F1
python evaluate.py

# Đánh giá theo chuẩn COCO metrics (AP, AP50, AR)
python evaluate_coco.py
```

### 4. Tùy Chỉnh Dữ Liệu Test

Mở file `visualize.py` và chỉnh sửa phần cấu hình:

```python
# Đường dẫn tới file OCR words (JSON)
TEST_JSON = "đường/dẫn/tới/file_ocr.json"

# Đường dẫn tới ảnh bảng
TEST_IMG = "đường/dẫn/tới/ảnh_bảng.jpg"

# Kích thước thật của ảnh (width, height)
REAL_IMG_SIZE = (1920, 1080)
```

---

## 📊 Kết Quả Thử Nghiệm

### Dataset

Mô hình được huấn luyện và đánh giá trên **PubTables-1M** - bộ dữ liệu lớn chứa hơn 1 triệu bảng từ các bài báo khoa học.

### Metrics

| Task | Pixel F1 | Object F1 |
|------|----------|-----------|
| Same Row | ~0.85 | ~0.80 |
| Same Column | ~0.87 | ~0.82 |
| Same Cell | ~0.83 | ~0.78 |
| Header | ~0.75 | ~0.70 |

---

## 📁 Định Dạng Dữ Liệu

### Input OCR JSON

```json
{
  "words": [
    {
      "text": "Name",
      "bbox": [100, 50, 150, 70]
    },
    {
      "text": "Age", 
      "bbox": [200, 50, 240, 70]
    }
  ]
}
```

### Output Adjacency Matrix

Ma trận kề `N × N` với giá trị 0-1, trong đó:
- `1`: Hai token thuộc cùng nhóm
- `0`: Hai token không thuộc cùng nhóm

---

## 🔗 Tham Khảo

### Paper Gốc
- **ClusTabNet**: [Table Structure Recognition via Clustering](https://arxiv.org/pdf/2402.07502)

### Dataset
- **PubTables-1M**: [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)

### Tài Liệu Liên Quan
- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

---

## 👤 Tác Giả

- **Họ tên**: Trịnh Nhật Phát
- **GitHub**: [@TrinhPhat2082005](https://github.com/TrinhPhat2082005)

---

## 📄 Giấy Phép

Dự án này được phát hành dưới giấy phép **MIT License**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

<p align="center">
  <b>⭐ Nếu dự án hữu ích, hãy cho một star nhé! ⭐</b>
</p>
