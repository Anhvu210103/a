import torch
import numpy as np
import cv2
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from video_transformer_model import VideoTransformer
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VideoTransformerTester:
    """Class để test hiệu suất của VideoTransformer model"""
    
    def __init__(self, model_path: str, device: str = 'auto', model_type: str = 'auto'):
        """
        Args:
            model_path: Đường dẫn tới model file (.ckpt hoặc .pth)
            device: 'auto', 'cpu', hoặc 'cuda'
            model_type: 'auto', 'ckpt', hoặc 'pth' - tự động detect nếu 'auto'
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model_type = self._detect_model_type(model_path, model_type)
        self.model = None
        self.class_names = ['Bình thường', 'Gian lận']
        
        print(f"🚀 Khởi tạo VideoTransformer Tester")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Model path: {model_path}")
        print(f"📄 Model type: {self.model_type}")
        
    def _get_device(self, device: str) -> str:
        """Tự động chọn device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _detect_model_type(self, model_path: str, model_type: str) -> str:
        """Tự động detect loại model từ extension"""
        if model_type != 'auto':
            return model_type
        
        if model_path.endswith('.ckpt'):
            return 'ckpt'
        elif model_path.endswith('.pth') or model_path.endswith('.pt'):
            return 'pth'
        else:
            print(f"⚠️ Không thể detect model type từ {model_path}, sử dụng mặc định 'ckpt'")
            return 'ckpt'
    
    def load_model(self) -> bool:
        """Load model từ checkpoint hoặc pth file"""
        try:
            print(f"📥 Đang load model từ {self.model_path}...")
            
            if not os.path.exists(self.model_path):
                print(f"❌ Không tìm thấy file model: {self.model_path}")
                return False
            
            size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"📊 Model size: {size_mb:.1f} MB")
            
            if self.model_type == 'ckpt':
                # Load model từ PyTorch Lightning checkpoint
                print(f"🔧 Loading PyTorch Lightning checkpoint...")
                try:
                    # Thử load với Lightning loader trước
                    self.model = VideoTransformer.load_from_checkpoint(
                        self.model_path,
                        map_location=self.device,
                        strict=False
                    )
                except Exception as e1:
                    print(f"⚠️ Lightning loader failed: {e1}")
                    print(f"🔄 Trying manual load...")
                    
                    # Fallback: load manual
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    # Tạo model với hyperparameters
                    if 'hyper_parameters' in checkpoint:
                        hparams = checkpoint['hyper_parameters']
                        print(f"📋 Using hyperparameters from checkpoint")
                    else:
                        # Default hyperparameters
                        hparams = {
                            'img_size': 224,
                            'patch_size': 16,
                            'in_channels': 3,
                            'num_classes': 2,
                            'embed_dim': 768,
                            'num_heads': 12,
                            'num_layers': 12,
                            'num_frames': 8,
                            'mlp_ratio': 4.0,
                            'dropout': 0.1,
                            'learning_rate': 1e-4
                        }
                        print(f"� Using default hyperparameters")
                    
                    self.model = VideoTransformer(**hparams)
                    
                    # Load state dict
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Remove 'model.' prefix if exists
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('model.'):
                            new_state_dict[k[6:]] = v
                        else:
                            new_state_dict[k] = v
                    
                    self.model.load_state_dict(new_state_dict, strict=False)
                    
            elif self.model_type == 'pth':
                # Load model từ PyTorch pth file
                print(f"🔧 Loading từ .pth file...")
                model_kwargs = {
                    'img_size': 224,
                    'patch_size': 16,
                    'in_channels': 3,
                    'num_classes': 2,
                    'embed_dim': 768,
                    'num_heads': 12,
                    'num_layers': 12,
                    'num_frames': 8,
                    'mlp_ratio': 4.0,
                    'dropout': 0.1,
                    'learning_rate': 1e-4
                }
                self.model = VideoTransformer.load_from_pth(
                    self.model_path, 
                    device=self.device,
                    **model_kwargs
                )
            else:
                print(f"❌ Không hỗ trợ model type: {self.model_type}")
                return False
            
            self.model.eval()
            self.model.to(self.device)
            
            # Test forward pass
            print(f"🧪 Testing model forward pass...")
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 8, 224, 224).to(self.device)
                try:
                    output = self.model(dummy_input)
                    print(f"✅ Forward test successful! Output shape: {output.shape}")
                except Exception as e:
                    print(f"❌ Forward test failed: {e}")
                    return False
            
            print(f"✅ Load model thành công!")
            print(f"🔧 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"🎮 Model device: {next(self.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_single_video(self, video_path: str, true_label: int = None) -> Dict:
        """Test một video đơn lẻ"""
        if self.model is None:
            print("❌ Model chưa được load!")
            return {}
        
        print(f"\n🎬 Testing video: {os.path.basename(video_path)}")
        
        start_time = time.time()
        
        try:
            # Dự đoán
            predicted_class, confidence = self.model.predict_video(video_path, self.device)
            
            if predicted_class is None:
                print(f"❌ Không thể xử lý video: {video_path}")
                return {}
            
            inference_time = time.time() - start_time
            
            result = {
                'video_path': video_path,
                'predicted_class': predicted_class,
                'predicted_label': self.class_names[predicted_class],
                'confidence': confidence,
                'inference_time': inference_time
            }
            
            if true_label is not None:
                result['true_label'] = true_label
                result['true_class_name'] = self.class_names[true_label]
                result['correct'] = predicted_class == true_label
            
            # Hiển thị kết quả
            print(f"🎯 Dự đoán: {self.class_names[predicted_class]} ({confidence:.4f})")
            if true_label is not None:
                print(f"🏷️  Thực tế: {self.class_names[true_label]}")
                print(f"✅ Chính xác: {'Đúng' if result['correct'] else 'Sai'}")
            print(f"⏱️  Thời gian: {inference_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Lỗi khi test video: {str(e)}")
            return {}
    
    def test_video_folder(self, folder_path: str, label_mapping: Dict[str, int] = None) -> Dict:
        """
        Test toàn bộ folder video
        
        Args:
            folder_path: Đường dẫn tới folder chứa video
            label_mapping: Dict mapping tên subfolder -> label (0: bình thường, 1: gian lận)
                          Ví dụ: {'normal': 0, 'fraud': 1}
        """
        if self.model is None:
            print("❌ Model chưa được load!")
            return {}
        
        print(f"\n📁 Testing folder: {folder_path}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        all_results = []
        
        if label_mapping:
            # Test theo structure subfolder
            for class_folder, label in label_mapping.items():
                class_path = os.path.join(folder_path, class_folder)
                if not os.path.exists(class_path):
                    print(f"⚠️ Không tìm thấy folder: {class_path}")
                    continue
                
                print(f"\n📂 Testing class: {class_folder} (label: {label})")
                
                video_files = [
                    f for f in os.listdir(class_path) 
                    if Path(f).suffix.lower() in video_extensions
                ]
                
                for video_file in video_files:
                    video_path = os.path.join(class_path, video_file)
                    result = self.test_single_video(video_path, label)
                    if result:
                        all_results.append(result)
        else:
            # Test tất cả video trong folder (không có ground truth)
            video_files = [
                f for f in os.listdir(folder_path) 
                if Path(f).suffix.lower() in video_extensions
            ]
            
            for video_file in video_files:
                video_path = os.path.join(folder_path, video_file)
                result = self.test_single_video(video_path)
                if result:
                    all_results.append(result)
        
        return self._analyze_results(all_results)
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Phân tích kết quả test"""
        if not results:
            print("❌ Không có kết quả để phân tích!")
            return {}
        
        analysis = {
            'total_videos': len(results),
            'results': results,
            'avg_inference_time': np.mean([r['inference_time'] for r in results]),
            'avg_confidence': np.mean([r['confidence'] for r in results])
        }
        
        # Nếu có ground truth labels
        if 'true_label' in results[0]:
            true_labels = [r['true_label'] for r in results]
            pred_labels = [r['predicted_class'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            # Tính metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted'
            )
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            # Classification report
            class_report = classification_report(
                true_labels, pred_labels, 
                target_names=self.class_names,
                output_dict=True
            )
            
            analysis.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'correct_predictions': sum(r['correct'] for r in results)
            })
            
            # ROC AUC nếu có đủ classes
            if len(set(true_labels)) == 2:
                try:
                    # Sử dụng confidence của class 1 (gian lận) làm probability
                    probs = [conf if pred == 1 else 1-conf for pred, conf in zip(pred_labels, confidences)]
                    auc = roc_auc_score(true_labels, probs)
                    analysis['auc'] = auc
                except:
                    pass
        
        self._print_analysis(analysis)
        return analysis
    
    def _print_analysis(self, analysis: Dict):
        """In kết quả phân tích"""
        print(f"\n{'='*60}")
        print(f"📊 KẾT QUẢ PHÂN TÍCH HIỆU SUẤT")
        print(f"{'='*60}")
        
        print(f"📈 Tổng số video: {analysis['total_videos']}")
        print(f"⏱️  Thời gian trung bình: {analysis['avg_inference_time']:.3f}s/video")
        print(f"🎯 Confidence trung bình: {analysis['avg_confidence']:.4f}")
        
        if 'accuracy' in analysis:
            print(f"\n🎯 HIỆU SUẤT PHÂN LOẠI:")
            print(f"   • Accuracy: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)")
            print(f"   • Precision: {analysis['precision']:.4f}")
            print(f"   • Recall: {analysis['recall']:.4f}")
            print(f"   • F1-Score: {analysis['f1_score']:.4f}")
            
            if 'auc' in analysis:
                print(f"   • AUC: {analysis['auc']:.4f}")
            
            print(f"   • Dự đoán đúng: {analysis['correct_predictions']}/{analysis['total_videos']}")
            
            # Confusion Matrix
            cm = np.array(analysis['confusion_matrix'])
            print(f"\n📋 CONFUSION MATRIX:")
            print(f"                    Predicted")
            print(f"                Bình thường  Gian lận")
            print(f"Actual Bình thường    {cm[0,0]:4d}      {cm[0,1]:4d}")
            print(f"       Gian lận       {cm[1,0]:4d}      {cm[1,1]:4d}")
            
            # Per-class metrics
            print(f"\n📊 CHI TIẾT THEO CLASS:")
            for class_name in self.class_names:
                class_idx = self.class_names.index(class_name)
                if str(class_idx) in analysis['classification_report']:
                    metrics = analysis['classification_report'][str(class_idx)]
                    print(f"   {class_name}:")
                    print(f"     - Precision: {metrics['precision']:.4f}")
                    print(f"     - Recall: {metrics['recall']:.4f}")
                    print(f"     - F1-Score: {metrics['f1-score']:.4f}")
                    print(f"     - Support: {metrics['support']}")
    
    def save_results(self, results: Dict, output_path: str):
        """Lưu kết quả ra file JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Đã lưu kết quả vào: {output_path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file: {str(e)}")
    
    def plot_confusion_matrix(self, results: Dict, save_path: str = None):
        """Vẽ confusion matrix"""
        if 'confusion_matrix' not in results:
            print("❌ Không có confusion matrix để vẽ!")
            return
        
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Đã lưu confusion matrix: {save_path}")
        
        plt.show()
    
    def benchmark_speed(self, video_path: str, num_runs: int = 10):
        """Benchmark tốc độ xử lý"""
        if self.model is None:
            print("❌ Model chưa được load!")
            return
        
        print(f"\n⚡ BENCHMARK TỐC ĐỘ ({num_runs} lần chạy)")
        print(f"📹 Video: {os.path.basename(video_path)}")
        
        times = []
        for i in range(num_runs):
            start_time = time.time()
            self.model.predict_video(video_path, self.device)
            inference_time = time.time() - start_time
            times.append(inference_time)
            print(f"Run {i+1:2d}: {inference_time:.3f}s")
        
        print(f"\n📊 THỐNG KÊ TỐC ĐỘ:")
        print(f"   • Trung bình: {np.mean(times):.3f}s")
        print(f"   • Tối thiểu: {np.min(times):.3f}s")
        print(f"   • Tối đa: {np.max(times):.3f}s")
        print(f"   • Độ lệch chuẩn: {np.std(times):.3f}s")
        print(f"   • FPS tương đương: {1/np.mean(times):.2f} video/s")

def demo_test():
    """Demo function để test model"""
    print("🎬 VideoTransformer Model Tester Demo")
    print("="*50)
    
    # Cấu hình - Thử các đường dẫn model có sẵn
    MODEL_PATHS = [
        r"C:\Users\anhvu\Documents\linh tinh\epoch=4-step=75.ckpt",  # PyTorch Lightning checkpoint
        r"C:\Users\anhvu\Documents\linh tinh\vivit_model.pth",       # PyTorch model file
        "video_transformer_model.ckpt",  # Fallback
    ]
    
    DEVICE = 'auto'  # 'auto', 'cpu', hoặc 'cuda'
    
    # Tìm model file có sẵn
    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Tìm thấy model: {os.path.basename(path)}")
            break
    
    if not model_path:
        print("❌ Không tìm thấy model file nào!")
        print("💡 Các đường dẫn đã thử:")
        for path in MODEL_PATHS:
            print(f"   - {path}")
        return
    
    # Khởi tạo tester
    tester = VideoTransformerTester(model_path, DEVICE)
    
    # Load model
    if not tester.load_model():
        print("❌ Không thể load model. Vui lòng kiểm tra đường dẫn.")
        return
    
    # Test cases khác nhau
    print("\n" + "="*50)
    print("🎯 CHỌN LOẠI TEST:")
    print("1. Test một video đơn lẻ")
    print("2. Test folder video có cấu trúc")
    print("3. Test folder video không có nhãn")
    print("4. Benchmark tốc độ")
    
    choice = input("\nNhập lựa chọn (1-4): ").strip()
    
    if choice == '1':
        # Test single video
        video_path = input("Nhập đường dẫn video: ").strip()
        has_label = input("Video có ground truth label không? (y/n): ").lower().startswith('y')
        
        true_label = None
        if has_label:
            true_label = int(input("Nhập label (0: Bình thường, 1: Gian lận): "))
        
        result = tester.test_single_video(video_path, true_label)
        
        # Lưu kết quả
        if result:
            tester.save_results(result, 'single_video_result.json')
    
    elif choice == '2':
        # Test folder with structure
        folder_path = input("Nhập đường dẫn folder: ").strip()
        print("\nCấu trúc folder cần có dạng:")
        print("folder/")
        print("  ├── normal/     (video bình thường)")
        print("  └── fraud/      (video gian lận)")
        
        label_mapping = {
            'normal': 0,
            'fraud': 1
        }
        
        results = tester.test_video_folder(folder_path, label_mapping)
        
        # Lưu kết quả và vẽ biểu đồ
        if results:
            tester.save_results(results, 'folder_test_results.json')
            tester.plot_confusion_matrix(results, 'confusion_matrix.png')
    
    elif choice == '3':
        # Test folder without labels
        folder_path = input("Nhập đường dẫn folder: ").strip()
        results = tester.test_video_folder(folder_path)
        
        if results:
            tester.save_results(results, 'unlabeled_test_results.json')
    
    elif choice == '4':
        # Benchmark speed
        video_path = input("Nhập đường dẫn video để benchmark: ").strip()
        num_runs = int(input("Số lần chạy (mặc định 10): ") or "10")
        tester.benchmark_speed(video_path, num_runs)
    
    else:
        print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    demo_test()
