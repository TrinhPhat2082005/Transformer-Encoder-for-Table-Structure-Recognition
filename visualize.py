import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Import model classes
from model import ClusTabNetPipeline, ClusTabEmbedding
# Import build_global_vocab from dataset
from dataset import build_global_vocab


def preprocess_input(json_path, vocab, max_len=512, img_size=(1000, 1000)):
    """
    Đọc file JSON chứa words và chuẩn bị input cho model.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        word_list = json.load(f)
    
    img_w, img_h = img_size
    
    input_ids = []
    norm_boxes = []
    
    current_words = word_list[:max_len]
    
    for item in current_words:
        text = item.get('text', '')
        token_id = vocab.get(text, vocab.get("<UNK>", 1))
        input_ids.append(token_id)
        
        bbox = item['bbox']
        norm_box = [
            bbox[0] / img_w, bbox[1] / img_h,
            bbox[2] / img_w, bbox[3] / img_h
        ]
        norm_boxes.append(norm_box)
    
    num_real = len(input_ids)
    
    pad_len = max_len - num_real
    if pad_len > 0:
        input_ids += [0] * pad_len
        norm_boxes += [[0.0, 0.0, 0.0, 0.0]] * pad_len
    
    mask = [1] * num_real + [0] * pad_len
    
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    bbox = torch.tensor([norm_boxes], dtype=torch.float32)
    mask = torch.tensor([mask], dtype=torch.long)
    
    return input_ids, bbox, mask, num_real


def visualize_single_task(image, raw_word_boxes, adj_matrix, task_name, threshold=0.5, color=(255, 0, 0)):
    """
    Visualize một task riêng lẻ để debug và xem chi tiết.
    """
    img = image.copy()
    
    matrix_bin = (adj_matrix > threshold).astype(int)
    graph = csr_matrix(matrix_bin)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    groups_info = []
    
    for i in range(n_components):
        indices = np.where(labels == i)[0]
        if len(indices) < 2:
            continue
        
        group_boxes = np.array([raw_word_boxes[idx] for idx in indices])
        x1 = int(np.min(group_boxes[:, 0]))
        y1 = int(np.min(group_boxes[:, 1]))
        x2 = int(np.max(group_boxes[:, 2]))
        y2 = int(np.max(group_boxes[:, 3]))
        
        # Vẽ bbox cho nhóm
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ số nhóm
        cv2.putText(img, str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        groups_info.append({
            'group_id': i,
            'num_words': len(indices),
            'bbox': [x1, y1, x2, y2]
        })
    
    return img, groups_info


def visualize_all_tasks_separate(image_path, raw_word_boxes, adj_matrices, threshold=0.5, save_path=None):
    """
    Vẽ riêng từng task trên ảnh riêng để dễ debug và so sánh.
    """
    # Load ảnh
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path.copy()
    
    # Cấu hình
    COLORS = {
        'same_row': (255, 50, 50),
        'same_col': (50, 200, 50),
        'same_header': (50, 50, 255),
        'same_cell': (180, 100, 255),
        'extract_cell': (255, 180, 50)
    }
    
    LABELS = {
        'same_row': 'Hàng (Row)',
        'same_col': 'Cột (Column)', 
        'same_header': 'Header',
        'same_cell': 'Ô (Cell)',
        'extract_cell': 'Spanning Cell'
    }
    
    # Tạo figure với 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH RIÊNG TỪNG TASK (Threshold = {threshold})")
    print(f"{'='*60}")
    
    for idx, (task_name, matrix) in enumerate(adj_matrices.items()):
        if idx >= 5:
            break
            
        color = COLORS.get(task_name, (128, 128, 128))
        result_img, groups = visualize_single_task(img, raw_word_boxes, matrix, task_name, threshold, color)
        
        axes[idx].imshow(result_img)
        axes[idx].set_title(f'{LABELS.get(task_name, task_name)}\n({len(groups)} nhóm)', fontsize=11)
        axes[idx].axis('off')
        
        print(f"\n{LABELS.get(task_name, task_name)}:")
        print(f"  Số nhóm có ≥2 từ: {len(groups)}")
        for g in groups[:5]:  # In 5 nhóm đầu
            print(f"    - Nhóm {g['group_id']}: {g['num_words']} từ, bbox={g['bbox']}")
        if len(groups) > 5:
            print(f"    ... và {len(groups)-5} nhóm khác")
    
    # Ô cuối: Ảnh gốc
    axes[5].imshow(img)
    axes[5].set_title('Ảnh gốc', fontsize=11)
    axes[5].axis('off')
    
    plt.suptitle(f'Phân tích từng Task - Threshold: {threshold}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nĐã lưu: {save_path}")
    
    plt.show()
    print(f"{'='*60}\n")


def visualize_table_combined(image_path, raw_word_boxes, adj_matrices, 
                              row_threshold=0.7, col_threshold=0.7, cell_threshold=0.5,
                              show_cells=True, show_rows=True, show_cols=True, show_header=True,
                              save_path=None):
    """
    Vẽ tổng hợp cấu trúc bảng với threshold riêng cho từng task.
    
    Args:
        row_threshold: Ngưỡng cho Row (cao hơn để lọc bớt)
        col_threshold: Ngưỡng cho Column
        cell_threshold: Ngưỡng cho Cell
        show_*: Bật/tắt hiển thị từng loại
    """
    # Load ảnh
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path.copy()
    
    overlay = img.copy()
    output_img = img.copy()

    # Cấu hình style
    COLORS = {
        'same_row': (255, 50, 50),
        'same_col': (50, 200, 50),
        'same_header': (50, 50, 255),
        'same_cell': (200, 180, 255),  # Tím nhạt
        'extract_cell': (255, 180, 50)
    }
    
    THRESHOLDS = {
        'same_row': row_threshold,
        'same_col': col_threshold,
        'same_header': row_threshold,
        'same_cell': cell_threshold,
        'extract_cell': cell_threshold
    }
    
    THICKNESS = {
        'same_row': 1,
        'same_col': 1,
        'same_header': 1,
        'same_cell': -1,  # Fill
        'extract_cell': 1  # Border
    }

    OFFSET = {
        'same_row': 2,
        'same_col': 4,
        'same_header': 6,
        'same_cell': 0,
        'extract_cell': 0
    }
    
    LABELS = {
        'same_row': 'Hàng (Row)',
        'same_col': 'Cột (Column)', 
        'same_header': 'Header',
        'same_cell': 'Ô (Cell)',
        'extract_cell': 'Spanning Cell'
    }
    
    # Task nào cần vẽ
    tasks_to_draw = []
    if show_cells:
        tasks_to_draw.extend(['same_cell', 'extract_cell'])
    if show_rows:
        tasks_to_draw.append('same_row')
    if show_cols:
        tasks_to_draw.append('same_col')
    if show_header:
        tasks_to_draw.append('same_header')
    
    # Thứ tự vẽ: nền trước, viền sau
    draw_order = ['same_cell', 'extract_cell', 'same_row', 'same_col', 'same_header']
    draw_order = [t for t in draw_order if t in tasks_to_draw]

    print(f"\n{'='*60}")
    print("KHÔI PHỤC CẤU TRÚC BẢNG (Threshold tùy chỉnh)")
    print(f"{'='*60}")

    stats = {}

    for task in draw_order:
        if task not in adj_matrices:
            continue
        
        threshold = THRESHOLDS.get(task, 0.5)
        matrix_prob = adj_matrices[task]
        matrix_bin = (matrix_prob > threshold).astype(int)
        
        graph = csr_matrix(matrix_bin)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        meaningful_groups = 0

        for i in range(n_components):
            indices = np.where(labels == i)[0]
            
            # Bỏ qua nhóm 1 từ (trừ cell)
            if len(indices) < 2 and task not in ['same_cell', 'extract_cell']:
                continue
            
            # Bỏ qua nhóm quá lớn (có thể là lỗi)
            if len(indices) > len(raw_word_boxes) * 0.8:
                print(f"  ⚠ Bỏ qua nhóm {task} với {len(indices)} từ (quá lớn, có thể lỗi)")
                continue
            
            meaningful_groups += 1
            group_boxes = np.array([raw_word_boxes[idx] for idx in indices])
            
            x1 = int(np.min(group_boxes[:, 0]))
            y1 = int(np.min(group_boxes[:, 1]))
            x2 = int(np.max(group_boxes[:, 2]))
            y2 = int(np.max(group_boxes[:, 3]))
            
            color = COLORS.get(task, (0, 0, 0))
            thickness = THICKNESS.get(task, 2)
            offset = OFFSET.get(task, 0)
            
            # Apply offset
            x1 -= offset
            y1 -= offset
            x2 += offset
            y2 += offset
            
            if thickness == -1:
                # Fill màu có trong suốt
                sub_overlay = output_img.copy()
                cv2.rectangle(sub_overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(sub_overlay, 0.4, output_img, 0.6, 0, output_img)
            else:
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)
        
        stats[task] = meaningful_groups
        print(f"  {LABELS[task]:20s}: {meaningful_groups:3d} nhóm (threshold={threshold})")

    # Trộn overlay
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)

    # Hiển thị
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(18, 10), 
                                              gridspec_kw={'width_ratios': [4, 1]})
    
    ax_main.imshow(output_img)
    ax_main.axis('off')
    ax_main.set_title('Khôi phục Cấu trúc Bảng', fontsize=14, fontweight='bold')
    
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    legend_y = 0.85
    ax_legend.text(0.5, 0.95, 'CHÚ THÍCH', fontsize=12, fontweight='bold', 
                   ha='center', transform=ax_legend.transAxes)
    
    for task in draw_order:
        if task not in COLORS:
            continue
        color = tuple(c/255 for c in COLORS[task])
        label = LABELS[task]
        count = stats.get(task, 0)
        thresh = THRESHOLDS.get(task, 0.5)
        
        if THICKNESS.get(task, 1) == -1:
             rect = plt.Rectangle((0.05, legend_y - 0.02), 0.12, 0.05, 
                                 facecolor=color, edgecolor='none')
        else:
             rect = plt.Rectangle((0.05, legend_y - 0.02), 0.12, 0.05, 
                                 facecolor='white', edgecolor=color, linewidth=2)
        ax_legend.add_patch(rect)
        
        ax_legend.text(0.2, legend_y, f'{label}\n({count}, t={thresh})', fontsize=9, 
                      verticalalignment='center')
        legend_y -= 0.14
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nĐã lưu: {save_path}")
    
    plt.show()
    print(f"{'='*60}\n")


def run_visualization_pipeline(model, json_path, img_path, tokenizer, device, img_size=(1000, 1000),
                                mode='separate', row_threshold=0.7, col_threshold=0.7, cell_threshold=0.5, save_path=None):
    """
    Pipeline chạy inference và visualize.
    
    Args:
        mode: 'separate' = vẽ riêng từng task, 'combined' = vẽ tổng hợp
        row_threshold, col_threshold, cell_threshold: Ngưỡng cho từng loại
    """
    # 1. Load Json gốc để lấy RAW BBOX
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    raw_boxes = [item['bbox'] for item in raw_data[:512]] 
    
    # 2. Chạy Model
    input_ids, bbox_norm, mask, num_real = preprocess_input(json_path, tokenizer, img_size=img_size)
    
    input_ids = input_ids.to(device)
    bbox_norm = bbox_norm.to(device)
    mask = mask.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, bbox_norm, mask=mask)
    
    # 3. Trích xuất Ma trận Numpy
    clean_matrices = {}
    for task_name, tensor in outputs.items():
        mat = tensor[0].cpu().numpy()
        
        if mat.max() > 1.0 or mat.min() < 0:
            mat = 1 / (1 + np.exp(-mat))
            
        clean_matrices[task_name] = mat[:num_real, :num_real]
    
    # 4. Visualize
    raw_boxes_cut = raw_boxes[:num_real]
    
    if mode == 'separate':
        visualize_all_tasks_separate(img_path, raw_boxes_cut, clean_matrices, threshold=0.5, save_path=save_path)
    else:
        visualize_table_combined(img_path, raw_boxes_cut, clean_matrices,
                                  row_threshold=row_threshold,
                                  col_threshold=col_threshold,
                                  cell_threshold=cell_threshold,
                                  save_path=save_path)


if __name__ == "__main__":
    # ==================== CẤU HÌNH ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TEST_JSON = "dataset/pubtables_mini_test/data_ocr/test/PMC3423433_table_1_words.json"
    TEST_IMG = "dataset/pubtables_mini_test/images/test/PMC3423433_table_1.jpg"
    
    if os.path.exists(TEST_IMG):
        tmp_img = cv2.imread(TEST_IMG)
        if tmp_img is not None:
            h, w, _ = tmp_img.shape
            REAL_IMG_SIZE = (w, h)
            print(f"Detected size: {REAL_IMG_SIZE}")
        else:
             REAL_IMG_SIZE = (1000, 1000)
    else:
         REAL_IMG_SIZE = (1000, 1000)

    VOCAB_SIZE_USED = 174636
    
    # ==================== BUILD VOCAB ====================
    print("Đang build vocab...")
    global_vocab = build_global_vocab([TEST_JSON])
    print(f"Đã build vocab! Số từ: {len(global_vocab)}")
    
    # ==================== LOAD MODEL ====================
    model = ClusTabNetPipeline(
        embedding_module=ClusTabEmbedding(vocab_size=VOCAB_SIZE_USED, d_model=640),
        d_model=640, n_head=5, num_layers=3
    )
    
    model_path = "model/model_weitghloss_10epoch.pth"
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Đã load model!")
    else:
        print("Chưa tìm thấy model, dùng random weights...")
    
    model.to(DEVICE)
    
    # ==================== KIỂM TRA ẢNH ====================
    if not os.path.exists(TEST_IMG):
        print(f"Không tìm thấy ảnh: {TEST_IMG}")
        dummy_img = np.ones((REAL_IMG_SIZE[1], REAL_IMG_SIZE[0], 3), dtype=np.uint8) * 255
        TEST_IMG = dummy_img
    
    # ==================== CHẠY VISUALIZATION ====================
    # Chỉ chạy mode combined (tất cả trong 1 ảnh)
    save_path = os.path.join("report", "figures", "visualization_result.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    run_visualization_pipeline(
        model, TEST_JSON, TEST_IMG, global_vocab, DEVICE, 
        img_size=REAL_IMG_SIZE,
        mode='combined',
        row_threshold=0.7,
        col_threshold=0.7,
        cell_threshold=0.5,
        save_path=save_path
    )
