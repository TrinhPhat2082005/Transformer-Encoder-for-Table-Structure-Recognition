import numpy as np
import json
from torch.utils.data import Dataset
import torch
import glob
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader



def build_global_vocab(file_paths, save_path="vocab.json"):
    word_set = set()
    # Chỉ quét qua để lấy từ vựng (nhanh hơn load full)
    for path in tqdm(file_paths, desc="Building Vocab"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                word_set.add(item['text'])
    
    # Map từ -> ID
    vocab = {word: idx + 2 for idx, word in enumerate(word_set)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    
    return vocab


# --- CẤU HÌNH ---
IOA_THRESHOLD = 0.7 

# 1. Thứ tự 5 Heads của Model 
# Index: 0: Row, 1: Col, 2: Header, 3: Cell, 4: Extract
MODEL_HEADS = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']

# 2. Map từ tên trong JSON -> Index của Model
# JSON của : 'row', 'column', 'header'
LABEL_TO_HEAD_IDX = {
    'row': 0,           # Map vào 'same_row'
    'column': 1,        # Map vào 'same_col'
    'header': 2,        # Map vào 'same_header'
    # 'table': Bỏ qua hoặc dùng làm mask nền
}
class AdjacencyLabelGenerator:
    def __init__(self, num_heads=5):
        self.num_heads = num_heads

    def compute_ioa_batch(self, boxes1, boxes2):
        """
        Tính IOA: Giữ nguyên logic numpy.
        Input: Raw Coordinates [x1, y1, x2, y2]
        """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))

        # Tính giao (Intersection)
        inter_x1 = np.maximum(boxes1[:, 0, None], boxes2[:, 0])
        inter_y1 = np.maximum(boxes1[:, 1, None], boxes2[:, 1])
        inter_x2 = np.minimum(boxes1[:, 2, None], boxes2[:, 2])
        inter_y2 = np.minimum(boxes1[:, 3, None], boxes2[:, 3])
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Diện tích Words (boxes1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        
        # Tránh chia cho 0
        return inter_area / (area1[:, None] + 1e-6)

    def create_adjacency_matrices(self, word_boxes, structure_boxes, structure_labels, cell_pointers_list):
        """
        word_boxes: List bbox của từ (Raw)
        structure_boxes: List bbox của cấu trúc (Raw)
        structure_labels: List index tương ứng với MODEL_HEADS (0=Row, 1=Col...)
        cell_pointers_list: List các 'extra_cells' tương ứng với structure
        """
        num_words = len(word_boxes)
        # Khởi tạo 5 ma trận 0
        adj_matrices = [
            np.zeros((num_words, num_words), dtype=np.float32)
            for _ in range(self.num_heads)
        ]

        if len(structure_boxes) == 0:
            return adj_matrices

        # --- DUYỆT QUA TỪNG LABEL ---
        for obj_box, obj_label, cell_ptrs in zip(structure_boxes, structure_labels, cell_pointers_list):
            
            # 1. Tìm các từ thuộc về object này (Row/Col/Header)
            ioa = self.compute_ioa_batch(word_boxes, [obj_box])
            belong_to_object = ioa[:, 0] > IOA_THRESHOLD
            indices = np.where(belong_to_object)[0]

            if len(indices) > 0:
                # a. Liên kết Mạnh (Full connection): Tất cả từ trong object nối với nhau
                # Dùng Grid Mesh để gán nhanh thay vì vòng lặp
                grid_x, grid_y = np.meshgrid(indices, indices)
                adj_matrices[obj_label][grid_x, grid_y] = 1.0

                # b. Liên kết Yếu (Spanning Cells)
                # Head 4: extract_cell
                if cell_ptrs and len(cell_ptrs) > 0:
                    ioa_cell = self.compute_ioa_batch(word_boxes, cell_ptrs)
                    belong_cells = ioa_cell > IOA_THRESHOLD
                    
                    # Logic: Nếu từ A thuộc Row X, và từ B thuộc Cell Y (mà Cell Y liên kết với Row X)
                    # Thì tạo nối A -> B trong ma trận 'extract_cell' (Index 4)
                    for cell_idx in range(belong_cells.shape[1]):
                        cell_word_indices = np.where(belong_cells[:, cell_idx])[0]
                        
                        if len(cell_word_indices) > 0:
                            # Nối từ (Row/Col) -> Từ (Spanning Cell)
                            # Sử dụng ma trận số 4 (extract_cell)
                            gx, gy = np.meshgrid(indices, cell_word_indices)
                            adj_matrices[4][gx, gy] = 1.0
                            adj_matrices[4][gy, gx] = 1.0 # Symmetric

        # --- BỔ SUNG: TỰ ĐỘNG TÍNH CELL (Index 3) ---
        # Vì script không có nhãn 'cell' riêng, ta lấy giao của Row và Col
        # adj[3] = adj[0] (Row) * adj[1] (Col)
        adj_matrices[3] = adj_matrices[0] * adj_matrices[1]

        return adj_matrices
    
class ClusTabTrainDataset(Dataset):
    def __init__(self, word_json_dir, gt_json_dir, tokenizer, max_len=512):
        self.word_files = glob.glob(os.path.join(word_json_dir, '*.json'))
        self.gt_json_dir = gt_json_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.label_gen = AdjacencyLabelGenerator(num_heads=5)

    def __len__(self):
        return len(self.word_files)

    def normalize_box(self, bbox, width, height):
        # Hàm chuẩn hóa dùng cho Input Model Embedding
        return [
            bbox[0] / width, bbox[1] / height,
            bbox[2] / width, bbox[3] / height
        ]

    def __getitem__(self, idx):
        # 1. Đường dẫn file
        word_path = self.word_files[idx]
        file_name = os.path.basename(word_path)
        
        # Giả định file GT tên giống hệt file Word
        # Nếu tên file khác (VD: _words.json vs .json), cần xử lý string ở đây
        # Ví dụ: gt_name = file_name.replace('_words.json', '.json')
        gt_name = file_name.replace('_words.json', '.json') if '_words' in file_name else file_name
        gt_path = os.path.join(self.gt_json_dir, gt_name)

        # 2. Load Dữ liệu Raw
        with open(word_path, 'r', encoding='utf-8') as f:
            word_list = json.load(f)
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        # Lấy thông tin ảnh để chuẩn hóa sau này
        img_w, img_h = gt_data.get('image_size', [1000, 1000])

        # 3. Chuẩn bị dữ liệu cho Generator (DÙNG RAW COORDINATES)
        raw_word_boxes = []
        input_ids = []
        norm_word_boxes = [] # Cái này để đưa vào Model

        # Cắt đúng max_len
        current_words = word_list[:self.max_len]

        for item in current_words:
            # a. Text ID
            text = item.get('text', '')
            token_id = self.tokenizer.get(text, self.tokenizer.get("<UNK>", 1))
            input_ids.append(token_id)
            
            # b. Bbox Raw (Cho Generator)
            raw_box = item['bbox'] 
            raw_word_boxes.append(raw_box)
            
            # c. Bbox Norm (Cho Model Embedding)
            norm_box = self.normalize_box(raw_box, img_w, img_h)
            norm_word_boxes.append(norm_box)

        # 4. Parse GT Structure (DÙNG RAW COORDINATES)
        struct_boxes = []
        struct_labels = [] # Index 0, 1, 2...
        struct_ptrs = []   # extra_cells list
        
        if 'labels' in gt_data:
            for item in gt_data['labels']:
                name = item['name']
                # Chỉ lấy những label có trong map (row, col, header)
                if name in LABEL_TO_HEAD_IDX:
                    struct_boxes.append(item['bbox']) # RAW
                    struct_labels.append(LABEL_TO_HEAD_IDX[name])
                    struct_ptrs.append(item.get('extra_cells', []))

        # 5. SINH MA TRẬN (Dùng toàn bộ là Raw Coordinates như script test)
        # adj_list là list 5 ma trận (num_real_words x num_real_words)
        adj_list = self.label_gen.create_adjacency_matrices(
            raw_word_boxes, 
            struct_boxes, 
            struct_labels,
            struct_ptrs
        )

        # 6. Padding & Đóng gói
        num_real = len(input_ids)
        pad_len = self.max_len - num_real
        
        if pad_len > 0:
            input_ids += [0] * pad_len
            # Pad bbox bằng 0
            norm_word_boxes += [[0.0, 0.0, 0.0, 0.0]] * pad_len

        # Chuyển Ma trận thành Tensor và Pad
        targets_tensor = torch.zeros((5, self.max_len, self.max_len), dtype=torch.float32)
        
        for head_idx, matrix in enumerate(adj_list):
            # matrix đang là numpy array (num_real, num_real)
            # Chuyển sang tensor
            mat_tensor = torch.tensor(matrix, dtype=torch.float32)
            
            # Gán vào góc trên trái của tensor đích (Padding tự động là 0 do khởi tạo zeros)
            targets_tensor[head_idx, :num_real, :num_real] = mat_tensor
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'bbox': torch.tensor(norm_word_boxes, dtype=torch.float32),
            'targets': targets_tensor
            }