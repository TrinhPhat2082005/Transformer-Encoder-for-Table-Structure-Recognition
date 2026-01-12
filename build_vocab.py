import json
import glob
import os
from dataset import build_global_vocab

# Định nghĩa đường dẫn
dataset_root = "dataset/pubtables_mini_test/data_ocr"
train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "val")
test_dir = os.path.join(dataset_root, "test")

# Thu thập tất cả các file JSON
print("Collecting files...")
train_files = glob.glob(os.path.join(train_dir, "*.json"))
val_files = glob.glob(os.path.join(val_dir, "*.json"))
test_files = glob.glob(os.path.join(test_dir, "*.json"))

# Xây dựng bộ từ vựng (Chỉ dùng tập Train như yêu cầu)
print("Building vocabulary from TRAIN set only...")
vocab = build_global_vocab(train_files)

# Lưu vào file
save_path = "vocab.json"
print(f"Saving vocabulary to {save_path}...")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Done! Vocab size: {len(vocab)}")
