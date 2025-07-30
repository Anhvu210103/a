import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
import math
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import cv2

class PatchEmbedding(nn.Module):
    """Video patch embedding cho Video Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_frames=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2 * num_frames
        
        # Sử dụng 3D convolution để xử lý video
        self.projection = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = self.projection(x)  # (B, embed_dim, T, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VideoTransformer(pl.LightningModule):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        num_frames=8,
        mlp_ratio=4.0,
        dropout=0.1,
        learning_rate=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.learning_rate = learning_rate
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Khởi tạo weights
        self.init_weights()
        
        # Metrics tracking
        self.train_acc = []
        self.val_acc = []
        self.test_predictions = []
        self.test_targets = []
        
    def init_weights(self):
        # Khởi tạo positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Khởi tạo linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Thêm class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        
        # Thêm positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Qua các transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Lấy class token để classification
        x = x[:, 0]  # (B, embed_dim)
        x = self.head(x)  # (B, num_classes)
        
        return x
    
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = F.cross_entropy(outputs, labels)
        
        # Tính accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = F.cross_entropy(outputs, labels)
        
        # Tính accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        
        # Lưu predictions và targets để tính metrics sau
        preds = torch.argmax(outputs, dim=1)
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        
        # Tính accuracy cho batch này
        acc = (preds == labels).float().mean()
        self.log('test_acc', acc)
        
        return {'test_acc': acc}
    
    def on_test_epoch_end(self):
        if len(self.test_predictions) > 0:
            # Tính các metrics chi tiết
            accuracy = accuracy_score(self.test_targets, self.test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.test_targets, self.test_predictions, average='weighted'
            )
            
            print(f"\n=== Test Results ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(self.test_targets, self.test_predictions)
            print(f"Confusion Matrix:\n{cm}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def predict_video(self, video_path):
        """Dự đoán cho một video duy nhất sử dụng CPU"""
        import time  # Import locally to avoid global dependency
        
        self.eval()
        
        # Đọc video
        start_time = time.time()
        video_frames = self.load_video(video_path)
        if video_frames is None:
            return None, None
        
        load_time = time.time() - start_time
        
        # Chuẩn bị input cho CPU
        video_tensor = torch.FloatTensor(video_frames).unsqueeze(0)
        
        # Thực hiện inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = self(video_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        inference_time = time.time() - inference_start
        
        print(f"📊 Video processing stats: Load time: {load_time:.4f}s, Inference time: {inference_time:.4f}s")
            
        return predicted_class, confidence
    
    @classmethod
    def load_from_pth(cls, pth_path, **model_kwargs):
        """Load mô hình từ file .pth"""
        # Tạo instance model với các tham số default hoặc từ kwargs
        model = cls(**model_kwargs)
        
        # Load state dict với CPU
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # Nếu checkpoint chứa 'state_dict' key
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Nếu checkpoint chỉ là state_dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def load_video(self, video_path, target_frames=None):
        """Load và preprocess video với tối ưu hiệu suất"""
        if target_frames is None:
            target_frames = self.num_frames
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return None
        
        # Optimize video capture settings
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Lấy thông tin video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video info: {width}x{height} @ {fps}fps, {frame_count} frames")
        
        # Pre-allocate numpy array cho frames để tối ưu memory
        frames = np.zeros((target_frames, 224, 224, 3), dtype=np.float32)
        
        # Lấy frame đều đặn qua video
        frame_indices = np.linspace(0, frame_count-1, target_frames, dtype=int)
        
        # Mean và std cho ImageNet normalization - lấy ra ngoài loop để tối ưu
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Lấy mẫu các frames theo các indices
        last_valid_frame = None
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize và normalize
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Normalize nhanh hơn với broadcasting và in-place operations
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - mean) / std
                
                frames[i] = frame
                last_valid_frame = frame
            else:
                # Nếu không đọc được frame, duplicate frame cuối
                if last_valid_frame is not None:
                    frames[i] = last_valid_frame
                else:
                    print(f"Không thể đọc frame {frame_idx}")
                    cap.release()
                    return None
        
        cap.release()
        
        # Transpose để có format (C, T, H, W) cho PyTorch
        frames = frames.transpose(3, 0, 1, 2)  # (C, T, H, W)
        
        return frames
