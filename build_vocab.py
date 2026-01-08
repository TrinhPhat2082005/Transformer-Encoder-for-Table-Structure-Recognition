import json
import glob
import os
from dataset import build_global_vocab

# Define paths
dataset_root = "dataset/pubtables_mini_test/data_ocr"
train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "val")
test_dir = os.path.join(dataset_root, "test")

# Collect all JSON files
print("Collecting files...")
train_files = glob.glob(os.path.join(train_dir, "*.json"))
val_files = glob.glob(os.path.join(val_dir, "*.json"))
test_files = glob.glob(os.path.join(test_dir, "*.json"))

# Build Vocab (Train Only as requested)
print("Building vocabulary from TRAIN set only...")
vocab = build_global_vocab(train_files)

# Save to file
save_path = "vocab.json"
print(f"Saving vocabulary to {save_path}...")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Done! Vocab size: {len(vocab)}")
