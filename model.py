import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class ClusTabEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size,        # Tổng số lượng từ trong từ điển
                 d_model=256,       # Kích thước vector đầu ra cuối cùng (đưa vào Transformer)
                 dropout=0.1,
                 max_len=1000,      # Độ dài tối đa của sequence
                 use_cnn=False,     # Có dùng vector CNN không?
                 cnn_input_dim=2048 # Kích thước vector CNN gốc (ví dụ ResNet50 ra 2048)
                 ):
        super().__init__()
        
        # --- 1. Cấu hình kích thước các thành phần ---
        # Chiến thuật chia d_model: 
        # Nếu không dùng CNN: 70% cho Text, 30% cho Vị trí
        # Nếu dùng CNN: 50% Text, 25% Vị trí, 25% Visual
        
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.d_text = int(d_model * 0.5)    
            self.d_box = int(d_model * 0.25)
            self.d_cnn = d_model - self.d_text - self.d_box
        else:
            self.d_text = int(d_model * 0.7)
            self.d_box = d_model - self.d_text
            self.d_cnn = 0

        # --- 2. Các lớp Embedding thành phần ---
        
        # A. Text Embedding: Chuyển ID từ -> Vector
        self.text_embed = nn.Embedding(vocab_size, self.d_text)
        
        # B. BBox Embedding: Chuyển 4 tọa độ (x1, y1, x2, y2) -> Vector
        # Input là 4 số thực đã chuẩn hóa
        self.box_embed = nn.Sequential(
            nn.Linear(4, self.d_box),
            nn.ReLU(),
            nn.Linear(self.d_box, self.d_box) # Thêm 1 lớp nữa để học phi tuyến tính tốt hơn
        )
        
        # C. CNN Embedding (Optional)
        if use_cnn:
            self.cnn_proj = nn.Linear(cnn_input_dim, self.d_cnn)
            
        # --- 3. Fusion & Normalization ---
        # LayerNorm cực kỳ quan trọng để cân bằng giá trị giữa Text và Box
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, bbox, cnn_features=None):
        """
        Input:
            - input_ids: Tensor (Batch, Seq_Len) chứa ID của chữ.
            - bbox: Tensor (Batch, Seq_Len, 4) chứa (xmin, ymin, xmax, ymax) đã chuẩn hóa 0-1.
            - cnn_features: Tensor (Batch, Seq_Len, cnn_dim) hoặc None.
        Output:
            - Tensor (Batch, Seq_Len, d_model)
        """
        
        # 1. Lấy đặc trưng Text
        # x_text shape: (Batch, Seq_Len, d_text)
        x_text = self.text_embed(input_ids)
        
        # 2. Lấy đặc trưng BBox
        # x_box shape: (Batch, Seq_Len, d_box)
        x_box = self.box_embed(bbox)
        
        # 3. Kết hợp (Concatenate)
        if self.use_cnn and cnn_features is not None:
            x_cnn = self.cnn_proj(cnn_features) # (Batch, Seq_Len, d_cnn)
            # Nối 3 phần lại theo chiều cuối cùng (dim=-1)
            x_final = torch.cat([x_text, x_box, x_cnn], dim=-1)
        else:
            # Nối 2 phần
            x_final = torch.cat([x_text, x_box], dim=-1)
            
        # 4. Chuẩn hóa đầu ra
        return self.dropout(self.layer_norm(x_final))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model phải chia hết cho n_head"
        
        self.d_k = d_model // n_head # Kích thước của mỗi head
        self.n_head = n_head
        self.d_model = d_model
        
        # Các lớp Linear để chiếu Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Lớp Linear cuối cùng sau khi nối các heads
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None):
        """
        Tính Scaled Dot-Product Attention
        input: (Batch, Head, Seq_Len, d_k)
        """
        # 1. Matmul Q * K_transpose
        # (B, h, L, d_k) * (B, h, d_k, L) -> (B, h, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. Masking (Che đi padding)
        if mask is not None:
            # mask shape: (Batch, 1, 1, Seq_Len) hoặc (Batch, 1, Seq_Len, Seq_Len)
            # Điền giá trị rất nhỏ vào chỗ mask == 0 (hoặc 1 tùy quy ước, ở đây giả sử mask=0 là pad)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Softmax
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 4. Matmul Score * V
        # (B, h, L, L) * (B, h, L, d_k) -> (B, h, L, d_k)
        output = torch.matmul(attn_probs, v)
        return output, attn_probs

    def forward(self, x, mask=None):
        # x shape: (Batch, Seq_Len, d_model)
        batch_size, seq_len, _ = x.size()
        
        # 1. Linear Projection & Split Heads
        # Biến đổi: (B, L, D) -> (B, L, h, d_k) -> (B, h, L, d_k)
        q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        
        # 2. Apply Attention
        # x_attn shape: (B, h, L, d_k)
        x_attn, _ = self.attention(q, k, v, mask)
        
        # 3. Concat Heads
        # (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, D)
        x_attn = x_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 4. Final Linear
        return self.w_o(x_attn)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() # GELU thường tốt hơn ReLU cho BERT/Transformer

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Norm và Dropout cho từng sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: Input tensor (Batch, Seq_Len, d_model)
        mask: Mask che padding
        """
        
        # 1. Tính Attention
        attn_output = self.self_attn(x, mask)
        # 2. Add & Norm (Residual Connection)
        # x + Dropout(Attention(x))
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 1. Tính FFN
        ffn_output = self.ffn(x)
        # 2. Add & Norm
        # x + Dropout(FFN(x))
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_head=4, num_layers=3, d_ff=1024, dropout=0.1):
        super().__init__()
        
        # Tạo list chứa N lớp EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # LayerNorm cuối cùng (Optional nhưng recommend)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: (Batch, Seq_Len, d_model) - Từ Embedding đi lên
        mask: (Batch, Seq_Len) - True là giữ, False/0 là Padding (cần xử lý shape)
        """
        
        # Xử lý mask để phù hợp với MultiHeadAttention
        # Input Mask: (Batch, Seq_Len) -> 1 là data, 0 là pad
        # Cần đổi sang: (Batch, 1, 1, Seq_Len) để broadcast qua các heads
        if mask is not None:
            extended_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_mask = None

        # Cho chạy qua từng tầng Encoder
        for layer in self.layers:
            x = layer(x, mask=extended_mask)
            
        return self.norm(x)
    

class ClusteringLinearHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        # Giúp giá trị không bị quá to. Ví dụ vector dài 640 thì chia cho căn(320)
        self.d_k = d_model // 2
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x):
        projected = self.linear(x)
        q, k = projected.chunk(2, dim=-1)
        
        # q và k nhân nhau ra giá trị rất lớn, cần scale nhỏ lại
        # (Batch, N, d/2) * (Batch, d/2, N) -> (Batch, N, N)
        adj_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        
        # Dùng Sigmoid để chuyển điểm số thành xác suất 0-1
        adj_probs = torch.sigmoid(adj_logits)
        
        return adj_probs 
    
class WeightedClusTabNetLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        self.criterion = nn.BCELoss(reduction='none')
        
        # Cấu hình trọng số ưu tiên: Ép học Row và Cell gấp đôi/gấp ba Col
        if task_weights is None:
            self.task_weights = {
                'same_col': 1.0,      # Cột dễ, giữ nguyên
                'same_row': 3.0,      # Hàng khó, nhân 3
                'same_cell': 5.0,     # Cell rất khó và thưa, nhân 5
                'same_header': 2.0,
                'extract_cell': 2.0
            }
        else:
            self.task_weights = task_weights

    def forward(self, predictions, targets, mask):
        total_loss = 0
        total_pixels = 0
        
        mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
        
        for task_name in predictions.keys():
            if task_name not in targets: continue
            
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Lọc dữ liệu thật
            valid_pred = pred[mask_2d]
            valid_target = target[mask_2d]
            
            if len(valid_pred) > 0:
                # Tính Loss cơ bản
                loss = self.criterion(valid_pred, valid_target)
                
                # Áp dụng trọng số theo độ khó của task
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = loss.sum() * weight
                
                total_loss += weighted_loss
                total_pixels += len(valid_pred)
        
        if total_pixels > 0:
            return total_loss / total_pixels
        else:
            return torch.tensor(0.0, device=mask.device, requires_grad=True)
class ClusTabNetPipeline(nn.Module):
    def __init__(self, embedding_module, d_model=256, n_head=4, num_layers=3):
        super().__init__()
        self.embedding = embedding_module
        self.encoder = CustomTransformerEncoder(d_model=d_model, n_head=n_head, num_layers=num_layers)
        
        # Thêm LayerNorm để ổn định luồng dữ liệu
        self.norm = nn.LayerNorm(d_model)
        
        self.task_names = ['same_row', 'same_col', 'same_header', 'same_cell', 'extract_cell']
        
        # 5 Clustering Heads - mỗi head dự đoán một loại mối quan hệ
        self.heads = nn.ModuleDict({
            name: ClusteringLinearHead(d_model) for name in self.task_names
        })

    def forward(self, input_ids, bbox, mask=None):
        x = self.embedding(input_ids, bbox)
        memory = self.encoder(x, mask=mask)
        memory = self.norm(memory) 
        
        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(memory) # Trả về 0->1
        return outputs
    
