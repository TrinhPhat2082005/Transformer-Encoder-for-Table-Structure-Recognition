import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model classes
from model import ClusTabNetPipeline, ClusTabEmbedding
# Import build_global_vocab from dataset
from dataset import build_global_vocab


def preprocess_input(json_path, vocab, max_len=512, img_size=(1000, 1000)):
    """
    Đọc file JSON chứa words và chuẩn bị input cho model.
    
    Args:
        json_path: Đường dẫn tới file JSON chứa danh sách words
        vocab: Dictionary tokenizer (word -> id)
        max_len: Độ dài tối đa của sequence
        img_size: Tuple (width, height) của ảnh gốc để normalize bbox
    
    Returns:
        input_ids: Tensor (1, max_len)
        bbox: Tensor (1, max_len, 4)
        mask: Tensor (1, max_len) - 1 là data thật, 0 là padding
        num_real: Số lượng từ thật (không padding)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        word_list = json.load(f)
    
    img_w, img_h = img_size
    
    input_ids = []
    norm_boxes = []
    
    # Cắt đúng max_len
    current_words = word_list[:max_len]
    
    for item in current_words:
        # Text -> ID
        text = item.get('text', '')
        token_id = vocab.get(text, vocab.get("<UNK>", 1))
        input_ids.append(token_id)
        
        # Normalize bbox (x1, y1, x2, y2) -> (0-1)
        bbox = item['bbox']
        norm_box = [
            bbox[0] / img_w, bbox[1] / img_h,
            bbox[2] / img_w, bbox[3] / img_h
        ]
        norm_boxes.append(norm_box)
    
    num_real = len(input_ids)
    
    # Padding
    pad_len = max_len - num_real
    if pad_len > 0:
        input_ids += [0] * pad_len
        norm_boxes += [[0.0, 0.0, 0.0, 0.0]] * pad_len
    
    # Tạo mask (1 = data thật, 0 = padding)
    mask = [1] * num_real + [0] * pad_len
    
    # Chuyển sang tensor và thêm batch dimension
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    bbox = torch.tensor([norm_boxes], dtype=torch.float32)
    mask = torch.tensor([mask], dtype=torch.long)
    
    return input_ids, bbox, mask, num_real


def predict_and_plot_full(model, json_path, vocab, device, img_size=(1000, 1000), max_len=512):
    """
    Chạy inference và visualize 5 ma trận adjacency output.
    
    Args:
        model: ClusTabNetPipeline đã load weights
        json_path: Đường dẫn file JSON chứa words
        vocab: Dictionary tokenizer
        device: torch.device (cuda/cpu)
        img_size: (width, height) của ảnh gốc
        max_len: Độ dài sequence tối đa
    """
    model.eval()
    
    # 1. Preprocessing
    input_ids, bbox, mask, num_real = preprocess_input(json_path, vocab, max_len, img_size)
    
    # Chuyển sang device
    input_ids = input_ids.to(device)
    bbox = bbox.to(device)
    mask = mask.to(device)
    
    # 2. Inference
    with torch.no_grad():
        outputs = model(input_ids, bbox, mask=mask)
    
    # 3. Visualize - Chỉ lấy phần data thật (bỏ padding)
    task_names = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'ClusTabNet Predictions - {os.path.basename(json_path)}', fontsize=14)
    
    for idx, task_name in enumerate(task_names):
        # Lấy output của task, chuyển về numpy
        adj_matrix = outputs[task_name][0].cpu().numpy()  # (max_len, max_len)
        
        # Cắt lấy phần data thật (bỏ padding)
        adj_real = adj_matrix[:num_real, :num_real]
        
        # Vẽ heatmap
        ax = axes[idx]
        im = ax.imshow(adj_real, cmap='hot', vmin=0, vmax=1)
        ax.set_title(task_name)
        ax.set_xlabel('Word Index')
        ax.set_ylabel('Word Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    return outputs


if __name__ == "__main__":
    # ==================== CẤU HÌNH ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # File input json test (Đường dẫn trên Kaggle)
    TEST_FILE = "dataset/pubtables_mini_test/data_ocr/test/PMC517708_table_1_words.json"
    
    # Kích thước ảnh gốc (lấy từ file GT tương ứng)
    REAL_IMG_SIZE = (697, 270)
    
    # Vocab size (phải dùng đúng số lúc train)
    VOCAB_SIZE_USED = 174636
    
    # ==================== BUILD VOCAB ====================
    # Build vocab từ test file (hoặc có thể truyền list nhiều files)
    print("Đang build vocab...")
    global_vocab = build_global_vocab([TEST_FILE])
    print(f"Đã build vocab thành công! Số từ: {len(global_vocab)}")
    
    # ==================== LOAD MODEL ====================
    # Khởi tạo kiến trúc y hệt lúc train
    model = ClusTabNetPipeline(
        embedding_module=ClusTabEmbedding(vocab_size=VOCAB_SIZE_USED, d_model=640),
        d_model=640, n_head=5, num_layers=3
    )
    
    # Load trọng số đã lưu
    model_path = "model/model_weitghloss_10epoch.pth"
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Đã load model thành công!")
    else:
        print("Chưa tìm thấy file model, dùng model ngẫu nhiên để test code...")
    
    model.to(DEVICE)
    
    # ==================== CHẠY INFERENCE ====================
    predict_and_plot_full(model, TEST_FILE, global_vocab, DEVICE, img_size=REAL_IMG_SIZE)
