
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataset import ClusTabTrainDataset
from model import ClusTabNetPipeline, ClusTabEmbedding

# Cấu hình đánh giá
IMG_SIZE = 1000  # Kích thước ảnh giả định để chuyển tọa độ sang pixel tuyệt đối
IOU_THRESHOLDS = [0.5]  # Ngưỡng IoU cho AP50

def matrix_to_boxes(adj_matrix, word_boxes, threshold=0.5):
    """
    Converts an NxN adjacency matrix into a list of bounding boxes (Clusters).
    
    Args:
        adj_matrix (np.ndarray): NxN probability matrix (0-1).
        word_boxes (torch.Tensor): (N, 4) Normalized boxes [x1, y1, x2, y2].
        threshold (float): Threshold to binarize the matrix.
        
    Returns:
        boxes (torch.Tensor): (K, 4) Absolute pixel coordinates.
        scores (torch.Tensor): (K,) Confidence scores.
        labels (torch.Tensor): (K,) Class labels (always 0 for single class).
    """
    # 1. Binarize & Cluster
    # adj_matrix is (N, N)
    bin_mat = (adj_matrix > threshold).astype(int)
    n_components, labels = connected_components(csgraph=bin_mat, directed=False, return_labels=True)
    
    cluster_boxes = []
    cluster_scores = []
    
    # Scale word boxes to absolute pixels first
    abs_word_boxes = word_boxes * IMG_SIZE
    
    for label_id in range(n_components):
        indices = np.where(labels == label_id)[0]
        if len(indices) == 0: continue
        
        # 2. Box Construction
        # Get all boxes for words in this cluster
        current_boxes = abs_word_boxes[indices] # (M, 4)
        
        # Find enclosing box: [min_x, min_y, max_x, max_y]
        x1 = torch.min(current_boxes[:, 0])
        y1 = torch.min(current_boxes[:, 1])
        x2 = torch.max(current_boxes[:, 2])
        y2 = torch.max(current_boxes[:, 3])
        
        cluster_boxes.append([x1, y1, x2, y2])
        
        # Tính điểm confidence = trung bình xác suất trong cluster
        if len(indices) > 1:
            # Lấy trung bình ma trận con của cluster
            score = np.mean(sub_mat)
        else:
            # Cluster chỉ có 1 từ, lấy giá trị đường chéo
            score = adj_matrix[indices[0], indices[0]]
            
        cluster_scores.append(score)
        
    if not cluster_boxes:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
        
    return (
        torch.tensor(cluster_boxes, dtype=torch.float32), 
        torch.tensor(cluster_scores, dtype=torch.float32),
        torch.zeros(len(cluster_boxes), dtype=torch.long) # Label 0 for all
    )

def evaluate_coco_standard():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_OCR_DIR = os.path.join(base_dir, "dataset", "pubtables_mini_test", "data_ocr", "test")
    TEST_GT_DIR = os.path.join(base_dir, "dataset", "pubtables_mini_test", "ocr_gt", "test")
    VOCAB_PATH = os.path.join(base_dir, "vocab.json")
    MODEL_PATH = os.path.join(base_dir, "model", "model_weitghloss_10epoch.pth")
    
    # Load Resources
    print("Loading vocab...")
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    dataset = ClusTabTrainDataset(TEST_OCR_DIR, TEST_GT_DIR, vocab, max_len=512)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    real_vocab_size = len(vocab)
    model = ClusTabNetPipeline(
        embedding_module=ClusTabEmbedding(vocab_size=real_vocab_size + 100, d_model=640),
        d_model=640, n_head=4, num_layers=3
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Model not found!")
        return
    model.eval()

    tasks = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']
    
    # Initialize Metrics per task
    # We use independent metric objects to keep state clean
    # max_detection_thresholds=[1, 10, 1000] because tables can have >100 cells
    metrics = {task: MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False, max_detection_thresholds=[1, 10, 1000]).to(device) 
               for task in tasks}
    
    print("\nStarting COCO Evaluation Loop...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device) # (B, Seq, 4) Normalized
            # Targets: (B, 5, Seq, Seq)
            targets_matrix = batch['targets'].to(device)
            
            mask = (input_ids != 0)
            seq_len = mask.sum(dim=1).item()
            
            # Forward
            outputs = model(input_ids, bbox, mask=mask)
            
            # --- PROCESS PER TASK ---
            for i, task in enumerate(tasks):
                # Output đã qua sigmoid, giá trị trong khoảng [0, 1]
                
                pred_mat = outputs[task][0, :seq_len, :seq_len].cpu().numpy()
                word_boxes = bbox[0, :seq_len, :].cpu() # (Seq, 4)
                
                pred_boxes, pred_scores, pred_labels = matrix_to_boxes(pred_mat, word_boxes, threshold=0.5)
                
                preds = [
                    {
                        "boxes": pred_boxes.to(device),
                        "scores": pred_scores.to(device),
                        "labels": pred_labels.to(device),
                    }
                ]
                
                # 2. Ground Truth
                # Extract GT matrix for this task
                gt_mat = targets_matrix[0, i, :seq_len, :seq_len].cpu().numpy()
                
                # Convert GT Matrix to Boxes (Score = 1.0)
                # Binarize GT at 0.5 (it should be binary already)
                gt_boxes, _, gt_labels = matrix_to_boxes(gt_mat, word_boxes, threshold=0.5)
                
                target = [
                    {
                        "boxes": gt_boxes.to(device),
                        "labels": gt_labels.to(device),
                    }
                ]
                
                # Cập nhật metric (torchmetrics xử lý cả trường hợp empty)
                metrics[task].update(preds, target)

    # Tính toán và in kết quả
    print("\n=== FINAL RESULTS (COCO) ===")
    final_results = {}
    
    print(f"{'Task':<15} | {'mAP (0.5:0.95)':<15} | {'AP50':<10} | {'AR':<10}")
    print("-" * 60)
    
    total_map = 0
    total_ap50 = 0
    total_ar = 0
    
    for task in tasks:
        m = metrics[task].compute()
        
        # map: mean Average Precision
        # map_50: AP at IoU=0.50
        # mar_100: Mean Average Recall given 100 detections per image
        
        ap = m['map'].item()
        ap50 = m['map_50'].item()
        ar = m['mar_1000'].item()
        
        final_results[task] = {
            "AP": ap,
            "AP50": ap50,
            "AR": ar
        }
        
        print(f"{task:<15} | {ap:.4f}          | {ap50:.4f}     | {ar:.4f}")
        
        total_map += ap
        total_ap50 += ap50
        total_ar += ar
        
    print("-" * 60)
    print(f"{'AVERAGE':<15} | {total_map/5:.4f}          | {total_ap50/5:.4f}     | {total_ar/5:.4f}")
    
    with open("eval_coco_results_standard.json", "w") as f:
        json.dump(final_results, f)
        
if __name__ == "__main__":
    evaluate_coco_standard()
