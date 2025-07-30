import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import cv2
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class TubeletEmbedding(nn.Module):
    """Tubelet embedding layer for ViViT"""
    def __init__(self, img_size=224, patch_size=40, tubelet_size=10, in_channels=3, embed_dim=100):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        
        self.linear_project = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.linear_project(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class SpatialTransformer(nn.Module):
    """Spatial Transformer module"""
    def __init__(self, embed_dim=100, num_heads=4, num_layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        
        # Tubelet embedding
        self.tubelet_embedding = TubeletEmbedding(embed_dim=embed_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.encoderLayer = encoder_layer
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B = x.shape[0]
        
        # Tubelet embedding
        x = self.tubelet_embedding(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, 1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        
        # Transformer encoding
        x = self.encoderLayer(x)  # (B, num_patches+1, embed_dim)
        
        return x

class TemporalTransformer(nn.Module):
    """Temporal Transformer module"""
    def __init__(self, embed_dim=100, num_heads=4, num_layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.encoderLayer = encoder_layer
        
    def forward(self, x):
        # x shape: (B, num_temporal_tokens, embed_dim)
        B = x.shape[0]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, 1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_temporal_tokens+1, embed_dim)
        
        # Transformer encoding
        x = self.encoderLayer(x)  # (B, num_temporal_tokens+1, embed_dim)
        
        return x

class ViViTModel(pl.LightningModule):
    """ViViT Model tương thích với checkpoint của bạn"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=40,
        tubelet_size=10,
        in_channels=3,
        embed_dim=100,
        num_heads=4,
        num_classes=2,
        dropout=0.1,
        learning_rate=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Spatial transformer
        self.spatial_transformer = SpatialTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # Temporal transformer  
        self.temporal_transformer = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, num_classes)
        )
        
        # Metrics tracking
        self.train_acc = []
        self.val_acc = []
        self.test_predictions = []
        self.test_targets = []
    
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        # Spatial encoding - process each frame
        spatial_features = []
        for t in range(T):
            frame = x[:, :, t:t+1, :, :]  # (B, C, 1, H, W)
            # Repeat frame to match tubelet_size if needed
            frame = frame.repeat(1, 1, 10, 1, 1)  # (B, C, 10, H, W)
            
            spatial_out = self.spatial_transformer(frame)  # (B, num_patches+1, embed_dim)
            cls_token = spatial_out[:, 0]  # (B, embed_dim)
            spatial_features.append(cls_token)
        
        # Stack temporal features
        temporal_input = torch.stack(spatial_features, dim=1)  # (B, T, embed_dim)
        
        # Temporal encoding
        temporal_out = self.temporal_transformer(temporal_input)  # (B, T+1, embed_dim)
        
        # Use class token for classification
        cls_token = temporal_out[:, 0]  # (B, embed_dim)
        
        # Classification
        output = self.mlp(cls_token)  # (B, num_classes)
        
        return output
    
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = F.cross_entropy(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = F.cross_entropy(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        
        preds = torch.argmax(outputs, dim=1)
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        
        acc = (preds == labels).float().mean()
        self.log('test_acc', acc)
        
        return {'test_acc': acc}
    
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
        """Dự đoán cho một video duy nhất"""
        self.eval()
        
        # Đọc video
        start_time = time.time()
        video_frames = self.load_video(video_path)
        if video_frames is None:
            return None, None
        
        load_time = time.time() - start_time
        
        # Chuẩn bị input
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
    
    def load_video(self, video_path, target_frames=8):
        """Load và preprocess video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return None
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Lấy frame đều đặn qua video
        if frame_count < target_frames:
            step = 1
        else:
            step = frame_count // target_frames
        
        frame_indices = np.linspace(0, frame_count-1, target_frames, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize và normalize
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                
                # Normalize theo ImageNet
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame = (frame - mean) / std
                
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    print(f"Không thể đọc frame {frame_idx}")
                    cap.release()
                    return None
        
        cap.release()
        
        # Chuyển về format (C, T, H, W)
        frames = np.array(frames)  # (T, H, W, C)
        frames = frames.transpose(3, 0, 1, 2)  # (C, T, H, W)
        
        return frames
    
    @classmethod
    def load_from_checkpoint_compatible(cls, ckpt_path, device='cpu'):
        """Load model từ checkpoint với compatibility"""
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Tạo model với default params
        model = cls()
        
        # Load state dict với strict=False
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        
        model.eval()
        return model
