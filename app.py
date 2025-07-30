from flask import Flask, jsonify, render_template_string, request
import torch
import cv2
import numpy as np
import os
import tempfile
import time
from video_transformer_model import VideoTransformer
from vivit_model import ViViTModel
import base64
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Cấu hình
MODEL_PATHS = [
    r"C:\Users\anhvu\Documents\linh tinh\vivit_model.pth",       # PyTorch model file (ưu tiên đầu tiên)
    r"C:\Users\anhvu\Documents\linh tinh\epoch=4-step=75.ckpt",  # PyTorch Lightning checkpoint  
    "video_transformer_model.ckpt",  # Fallback
]

# Luôn sử dụng CPU vì đã phát hiện hiệu suất tốt hơn với mô hình nhỏ
DEVICE = 'cpu'

# Thông báo về việc chọn device
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"� GPU {gpu_name} khả dụng nhưng không được sử dụng (CPU nhanh hơn với mô hình nhỏ)")
print(f"🚀 Đang sử dụng CPU cho video processing để có hiệu suất tối ưu")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Global model instance
model = None
class_names = ['Bình thường', 'Gian lận']
current_model_path = None

def allowed_file(filename):
    """Kiểm tra file extension hợp lệ"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load model khi khởi động app"""
    global model, current_model_path
    
    # Tìm model file có sẵn
    model_path = None
    print("🔍 Tìm kiếm model files...")
    for path in MODEL_PATHS:
        print(f"   Checking: {path}")
        if os.path.exists(path):
            model_path = path
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ Tìm thấy model: {os.path.basename(path)} ({size_mb:.1f} MB)")
            break
        else:
            print(f"   ❌ Không tồn tại")
    
    if not model_path:
        print("❌ Không tìm thấy model file nào!")
        print("💡 Các đường dẫn đã thử:")
        for path in MODEL_PATHS:
            print(f"   - {path}")
        return False
    
    try:
        start_time = time.time()
        print(f"🚀 Đang load model từ {model_path}... ({DEVICE} mode)")
        current_model_path = model_path
        
        # Tối ưu GPU memory nếu đang sử dụng CUDA
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            
        if model_path.endswith('.ckpt'):
            # Kiểm tra loại checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Kiểm tra xem có keys của ViViT không
            if any(key.startswith('spatial_transformer') for key in state_dict.keys()):
                print(f"🔧 Detected ViViT architecture, loading with ViViT model...")
                model = ViViTModel.load_from_checkpoint_compatible(model_path, device=DEVICE)
            else:
                print(f"🔧 Loading standard VideoTransformer checkpoint...")
                try:
                    model = VideoTransformer.load_from_checkpoint(
                        model_path, 
                        map_location='cpu',  # Trước tiên load vào CPU
                        strict=False
                    )
                except Exception as e1:
                    print(f"⚠️ Lightning load failed: {e1}")
                    print(f"🔄 Trying manual load...")
                    
                    if 'hyper_parameters' in checkpoint:
                        hparams = checkpoint['hyper_parameters']
                        print(f"📋 Using hyperparameters from checkpoint")
                    else:
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
                        print(f"📋 Using default hyperparameters")
                    
                    model = VideoTransformer(**hparams)
                    
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                
        elif model_path.endswith('.pth') or model_path.endswith('.pt'):
            # Load PyTorch pth file - kiểm tra architecture trước
            print(f"🔧 Loading PyTorch .pth file...")
            
            # Kiểm tra structure của pth file
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Kiểm tra xem có keys của ViViT không
            if any(key.startswith('spatial_transformer') for key in checkpoint.keys()):
                print(f"🔧 Detected ViViT architecture in .pth file, loading with ViViT model...")
                # Tạo model ViViT và load state dict
                model = ViViTModel()
                model.load_state_dict(checkpoint, strict=False)
            else:
                print(f"🔧 Detected standard VideoTransformer architecture in .pth file...")
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
                model = VideoTransformer.load_from_pth(model_path, device='cpu', **model_kwargs)
        else:
            print(f"❌ Không hỗ trợ định dạng file: {model_path}")
            return False
        
        # Di chuyển model sang GPU và set eval mode
        model.eval()
        
        if DEVICE == 'cuda':
            # Optimized GPU memory usage with mixed precision
            print(f"🔧 Moving model to GPU and optimizing with mixed precision...")
            model = model.to(DEVICE)
            
            # Enable automatic mixed precision for faster inference if using PyTorch 1.6+
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                print(f"✅ Mixed precision inference được bật")
        else:
            model = model.to(DEVICE)
        
        # Test forward pass đơn giản
        print(f"🧪 Testing model forward pass...")
        with torch.no_grad():
            if DEVICE == 'cuda' and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast():
                    dummy_input = torch.randn(1, 3, 8, 224, 224).to(DEVICE)
                    try:
                        output = model(dummy_input)
                        print(f"✅ Model forward test successful! Output shape: {output.shape}")
                    except Exception as e:
                        print(f"❌ Model forward test failed: {e}")
                        return False
            else:
                dummy_input = torch.randn(1, 3, 8, 224, 224).to(DEVICE)
                try:
                    output = model(dummy_input)
                    print(f"✅ Model forward test successful! Output shape: {output.shape}")
                except Exception as e:
                    print(f"❌ Model forward test failed: {e}")
                    return False
        
        load_time = time.time() - start_time
        print(f"✅ Model đã load thành công trên {DEVICE}! Thời gian: {load_time:.2f}s")
        print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Thực hiện inference warm-up để tối ưu performance trên GPU
        if DEVICE == 'cuda':
            print(f"🔥 Thực hiện inference warm-up...")
            for _ in range(3):
                with torch.no_grad():
                    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                        with torch.cuda.amp.autocast():
                            _ = model(dummy_input)
                    else:
                        _ = model(dummy_input)
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# HTML template cho web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Fraud Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }
        .upload-section {
            border: 2px dashed #ddd;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
            transition: border-color 0.3s;
        }
        .upload-section:hover {
            border-color: #007bff;
        }
        .file-input {
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: 100%;
            max-width: 400px;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .btn:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .result.normal {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.fraud {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            margin: 5px;
            flex: 1;
            min-width: 120px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .api-info {
            margin-top: 30px;
            padding: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
        }
        .api-info h3 {
            margin-top: 0;
            color: #495057;
        }
        .code-block {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Video Fraud Detection System</h1>
            <p>Hệ thống phát hiện hành vi gian lận trong video sử dụng VideoTransformer</p>
        </div>

        <div class="upload-section">
            <h3>📤 Upload Video để Phân Tích</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="videoFile" name="file" class="file-input" accept="video/*" required>
                <br>
                <button type="submit" class="btn" id="analyzeBtn">🔍 Phân Tích Video</button>
            </form>
        </div>

        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Đang phân tích video... Vui lòng đợi</p>
        </div>

        <div id="result" class="result">
            <h3 id="resultTitle"></h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="prediction">-</div>
                    <div class="stat-label">Dự đoán</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="confidence">-</div>
                    <div class="stat-label">Độ tin cậy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="processTime">-</div>
                    <div class="stat-label">Thời gian xử lý</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="fileSize">-</div>
                    <div class="stat-label">Kích thước file</div>
                </div>
            </div>
            <div id="details" style="margin-top: 20px;"></div>
        </div>

        <div class="api-info">
            <h3>🔧 API Documentation</h3>
            <p>Bạn có thể sử dụng API để tích hợp vào ứng dụng của mình:</p>
            
            <h4>Endpoint: POST /predict</h4>
            <div class="code-block">
curl -X POST -F "file=@video.mp4" {{ request.url_root }}predict
            </div>

            <h4>Python Example:</h4>
            <div class="code-block">
import requests

url = "{{ request.url_root }}predict"
files = {"file": open("video.mp4", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
            </div>

            <h4>Response JSON:</h4>
            <div class="code-block">
{
  "success": true,
  "prediction": {
    "class": 0,
    "label": "Bình thường",
    "confidence": 0.8542
  },
  "processing_time": 2.345,
  "filename": "video.mp4",
  "file_size": "15.2 MB",
  "timestamp": "2025-01-29T10:30:45"
}
            </div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Vui lòng chọn file video!');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('analyzeBtn').textContent = '⏳ Đang xử lý...';
            
            formData.append('file', file);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('analyzeBtn').textContent = '🔍 Phân Tích Video';
                
                // Show result
                if (data.success) {
                    const prediction = data.prediction;
                    const resultDiv = document.getElementById('result');
                    
                    // Set result class and title
                    resultDiv.className = 'result ' + (prediction.class === 1 ? 'fraud' : 'normal');
                    document.getElementById('resultTitle').textContent = 
                        (prediction.class === 1 ? '⚠️ Phát hiện hành vi gian lận!' : '✅ Video bình thường');
                    
                    // Update stats
                    document.getElementById('prediction').textContent = prediction.label;
                    document.getElementById('confidence').textContent = (prediction.confidence * 100).toFixed(1) + '%';
                    document.getElementById('processTime').textContent = data.processing_time.toFixed(2) + 's';
                    document.getElementById('fileSize').textContent = data.file_size;
                    
                    // Show additional details
                    document.getElementById('details').innerHTML = `
                        <p><strong>Tên file:</strong> ${data.filename}</p>
                        <p><strong>Thời gian xử lý:</strong> ${data.timestamp}</p>
                        <p><strong>Mô hình:</strong> VideoTransformer</p>
                    `;
                    
                    resultDiv.style.display = 'block';
                } else {
                    alert('Lỗi: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('analyzeBtn').textContent = '🔍 Phân Tích Video';
                alert('Có lỗi xảy ra khi xử lý video!');
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Trang chủ với web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán video"""
    start_time = time.time()
    
    try:
        # Kiểm tra model
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model chưa được load. Vui lòng khởi động lại server.'
            }), 500
        
        # Kiểm tra file upload
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy file trong request'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Không có file được chọn'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File không được hỗ trợ. Chỉ chấp nhận: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Lưu file tạm thời
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, filename)
        file.save(temp_file_path)
        
        # Lấy thông tin file
        file_size = os.path.getsize(temp_file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        try:
            # Dự đoán - bỏ tham số DEVICE vì model đã được cấu hình chỉ dùng CPU
            predicted_class, confidence = model.predict_video(temp_file_path)
            
            if predicted_class is None:
                return jsonify({
                    'success': False,
                    'error': 'Không thể xử lý video. Vui lòng kiểm tra định dạng file.'
                }), 400
            
            processing_time = time.time() - start_time
            
            # Tạo response
            response = {
                'success': True,
                'prediction': {
                    'class': predicted_class,
                    'label': class_names[predicted_class],
                    'confidence': float(confidence)
                },
                'processing_time': processing_time,
                'filename': filename,
                'file_size': f"{file_size_mb:.1f} MB",
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'name': 'VideoTransformer',
                    'device': DEVICE,
                    'classes': class_names
                }
            }
            
            return jsonify(response)
            
        finally:
            # Xóa file tạm thời
            try:
                os.remove(temp_file_path)
                os.rmdir(temp_dir)
            except:
                pass
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': DEVICE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Thông tin về model"""
    if model is None:
        return jsonify({
            'error': 'Model chưa được load'
        }), 500
    
    return jsonify({
        'model_name': 'VideoTransformer',
        'classes': class_names,
        'device': DEVICE,
        'parameters': sum(p.numel() for p in model.parameters()),
        'input_format': {
            'video_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size': '100MB',
            'recommended_resolution': '224x224',
            'recommended_fps': '30'
        }
    })

@app.errorhandler(413)
def too_large(e):
    """Handler cho file quá lớn"""
    return jsonify({
        'success': False,
        'error': 'File quá lớn. Kích thước tối đa là 100MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handler cho lỗi server"""
    return jsonify({
        'success': False,
        'error': 'Lỗi server nội bộ'
    }), 500

if __name__ == '__main__':
    print("🚀 Khởi động Video Fraud Detection API Server...")
    print(f"📱 Device: {DEVICE}")
    print("🎯 Tìm kiếm model files...")
    
    # Load model
    if not load_model():
        print("❌ Không thể khởi động server do lỗi load model")
        exit(1)
    
    print(f"✅ Đã load model: {os.path.basename(current_model_path)}")
    print("\n✅ Server sẵn sàng!")
    print("🌐 Web interface: http://localhost:5000")
    print("🔧 API endpoint: http://localhost:5000/predict")
    print("❤️  Health check: http://localhost:5000/health")
    
    # Chạy Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
