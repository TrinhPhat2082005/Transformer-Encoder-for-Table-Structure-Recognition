import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import glob
import json
from model import ClusTabNetPipeline, ClusTabEmbedding, WeightedClusTabNetLoss
from dataset import ClusTabTrainDataset, build_global_vocab

def train_one_epoch(model, dataloader, optimizer, criterion, device, task_names):
    """
    Huấn luyện mô hình qua một epoch.
    
    Args:
        model: Mô hình ClusTabNet
        dataloader: DataLoader chứa dữ liệu huấn luyện
        optimizer: Optimizer (AdamW)
        criterion: Hàm loss (WeightedClusTabNetLoss)
        device: CPU hoặc CUDA
        task_names: Danh sách tên 5 tasks
    
    Returns:
        float: Loss trung bình của epoch
    """
    model.train()
    epoch_loss = 0
    steps = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        
        # Xử lý Targets
        if isinstance(batch['targets'], dict):
                targets = {k: v.to(device) for k, v in batch['targets'].items()}
        else:
            raw_targets = batch['targets'].to(device)
            targets = {name: raw_targets[:, i] for i, name in enumerate(task_names)}

        mask = (input_ids != 0)
        
        # Forward
        outputs = model(input_ids, bbox, mask=mask)
        
        # Calculate Loss (Hàm này đã trả về Mean)
        loss = criterion(outputs, targets, mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        epoch_loss += current_loss
        steps += 1
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Avg Loss: {current_loss:.6f}")

    return epoch_loss / steps if steps > 0 else 0

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Dataset & Dataloader
    # Định nghĩa path local
    base_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(base_dir, "dataset", "pubtables_mini_test", "data_ocr", "train")
    GT_DIR = os.path.join(base_dir, "dataset", "pubtables_mini_test", "ocr_gt", "train")
    
    print(f"Input Dir: {INPUT_DIR}")
    print(f"GT Dir: {GT_DIR}")

    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
        
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    if not json_files:
        print("No json files found in input directory.")
        return

    # Load hoặc Build Vocab
    vocab_path = "vocab.json"
    if os.path.exists(vocab_path):
        print(f"Loading vocab from {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            global_vocab = json.load(f)
    else:
        print("Building vocab...")
        global_vocab = build_global_vocab(json_files, save_path=vocab_path) 
    
    real_vocab_size = len(global_vocab)
    print(f"Vocab Size: {real_vocab_size}")

    dataset = ClusTabTrainDataset(INPUT_DIR, GT_DIR, global_vocab)
    # Batch size nhỏ khi chạy local do giới hạn VRAM, trên Kaggle dùng batch_size=4
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Khởi tạo Model với d_model=640, chia hết cho n_head=4 (d_k=160)
    model = ClusTabNetPipeline(
        embedding_module=ClusTabEmbedding(vocab_size=real_vocab_size + 100, d_model=640),
        d_model=640, n_head=4, num_layers=3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = WeightedClusTabNetLoss().to(device) 
    
    # Task names constant
    TASK_NAMES = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']

    # 5. Train Loop
    print("\n--- Bắt đầu Train ---")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, TASK_NAMES)
        print(f"--> Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # Lưu model
        save_name = f"model_weitghloss_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved model to {save_name}")

if __name__ == "__main__":
    train()
