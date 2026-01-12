
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from dataset import ClusTabTrainDataset

def calculate_sparsity():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    val_ocr_dir = os.path.join(base_dir, "dataset", "pubtables_mini_test", "data_ocr", "val")
    val_gt_dir = os.path.join(base_dir, "dataset", "pubtables_mini_test", "ocr_gt", "val")
    vocab_path = os.path.join(base_dir, "vocab.json")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    dataset = ClusTabTrainDataset(val_ocr_dir, val_gt_dir, vocab)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    tasks = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']
    total_pixels = {task: 0 for task in tasks}
    positives = {task: 0 for task in tasks}
    
    print("Calculating positive ratios (sparsity)...")
    
    max_batches = 50 
    for i, batch in enumerate(tqdm(dataloader)):
        if i >= max_batches: break
        
        targets = batch['targets'] # (B, 5, Seq, Seq)
        input_ids = batch['input_ids']
        valid_mask = (input_ids != 0) # (B, Seq)
        
        # Chúng ta chỉ quan tâm đến các tương tác hợp lệ (từ thật vs từ thật)
        # Tạo mask 2D
        mask_2d = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2) # (B, Seq, Seq)
        
        for idx, task in enumerate(tasks):
            task_target = targets[:, idx, :, :]
            
            valid_pixels = task_target[mask_2d]
            
            total_elements = valid_pixels.numel()
            num_ones = valid_pixels.sum().item()
            
            total_pixels[task] += total_elements
            positives[task] += num_ones
            
    print("\n--- POSITIVE RATIO (Sparcity) ---")
    for task in tasks:
        if total_pixels[task] > 0:
            ratio = positives[task] / total_pixels[task]
            print(f"Task {task}: {ratio:.4f} ({ratio*100:.2f}%) positives")
        else:
            print(f"Task {task}: No data")

if __name__ == "__main__":
    calculate_sparsity()
