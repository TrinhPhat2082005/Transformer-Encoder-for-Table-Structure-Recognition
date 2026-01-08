
import torch
import numpy as np
import os
import glob
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from model import ClusTabNetPipeline, ClusTabEmbedding
from dataset import ClusTabTrainDataset

from scipy.sparse.csgraph import connected_components

def calculate_object_metrics(pred_matrix, target_matrix, threshold=0.5):
    """
    Calculate Precision, Recall, F1 at Object Level (Connected Components).
    This treats each Row/Column/Cell as an object.
    Stronger metric than pixel-level.
    """
    # 1. Binarize
    pred_bin = (pred_matrix > threshold).astype(int)
    target_bin = (target_matrix > 0.5).astype(int)
    
    # 2. Extract Components (Clusters)
    # n_components, labels = connected_components(...)
    # We need to construct adjacency graph. Input is NxN adjacency.
    
    # helper to get sets of frozensets
    def get_clusters(adj_mat):
        n_comp, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters: clusters[label] = []
            clusters[label].append(idx)
        
        # Filter out "singleton" clusters if they are just background noise? 
        # In Table structure, a single word CAN be a cell/row. So keep them.
        # But for 'same_header', usually we care if they form a group.
        # Let's keep all.
        return set(frozenset(c) for c in clusters.values())

    pred_clusters = get_clusters(pred_bin)
    target_clusters = get_clusters(target_bin)
    
    # 3. Match
    # Exact match of sets
    true_positives = len(pred_clusters.intersection(target_clusters))
    
    precision = true_positives / len(pred_clusters) if len(pred_clusters) > 0 else 0.0
    recall = true_positives / len(target_clusters) if len(target_clusters) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_OCR_DIR = os.path.join(base_dir, "dataset", "pubtables_mini_test", "data_ocr", "test")
    TEST_GT_DIR = os.path.join(base_dir, "dataset", "pubtables_mini_test", "ocr_gt", "test")
    VOCAB_PATH = os.path.join(base_dir, "vocab.json")
    MODEL_PATH = os.path.join(base_dir, "model", "model_weitghloss_10epoch.pth")
    
    # Load Vocab
    print("Loading vocab...")
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Vocab size: {len(vocab)}")
    
    # Dataset
    dataset = ClusTabTrainDataset(TEST_OCR_DIR, TEST_GT_DIR, vocab)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    real_vocab_size = len(vocab)
    model = ClusTabNetPipeline(
        embedding_module=ClusTabEmbedding(vocab_size=real_vocab_size + 100, d_model=640),
        d_model=640, n_head=4, num_layers=3
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Model file not found!")
        return
        
    model.eval()
    
    tasks = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']
    metrics = {task: {'p': [], 'r': [], 'f1': []} for task in tasks}
    
    print("\nStarting Evaluation (Object-Level with Connected Components)...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            targets = batch['targets'].to(device)
            mask = (input_ids != 0)
            
            outputs = model(input_ids, bbox, mask=mask)
            
            # Prediction thresholds per task
            thresholds = {
                'same_row': 0.7,
                'same_col': 0.7,
                'same_header': 0.7,
                'same_cell': 0.5,
                'extract_cell': 0.5
            }
            
            for i, task in enumerate(tasks):
                pred_prob = outputs[task]
                
                seq_len = mask.sum(dim=1).item()
                
                pred_mat = pred_prob[0, :seq_len, :seq_len].cpu().numpy()
                target_mat = targets[0, i, :seq_len, :seq_len].cpu().numpy()
                
                # Use Task-specific Threshold
                thresh = thresholds.get(task, 0.5)
                
                p, r, f1 = calculate_object_metrics(pred_mat, target_mat, threshold=thresh)
                
                metrics[task]['p'].append(p)
                metrics[task]['r'].append(r)
                metrics[task]['f1'].append(f1)
    
    # Compute Averages
    print("\n--- Evaluation Results ---")
    final_results = {}
    for task in tasks:
        avg_p = np.mean(metrics[task]['p'])
        avg_r = np.mean(metrics[task]['r'])
        avg_f1 = np.mean(metrics[task]['f1'])
        final_results[task] = {'p': avg_p, 'r': avg_r, 'f1': avg_f1}
        print(f"Task {task}: P={avg_p:.4f}, R={avg_r:.4f}, F1={avg_f1:.4f}")
        
    # Save results to file to read later
    with open('eval_results.json', 'w') as f:
        json.dump(final_results, f)

if __name__ == "__main__":
    evaluate()
