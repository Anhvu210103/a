# Video Fraud Detection System

Hệ thống phát hiện gian lận trong video sử dụng AI và Explainable AI để phân tích hành vi bất thường.

## 🎯 Tính năng chính

- **Phân tích video offline**: Xử lý file video và tạo video kết quả với đánh dấu gian lận
- **Phát hiện real-time**: Phân tích trực tiếp từ camera
- **Explainable AI**: Sử dụng Grad-CAM để giải thích quyết định của AI
- **Phát hiện người**: Vẽ bounding box xung quanh người trong video
- **Giao diện thân thiện**: GUI dễ sử dụng với các tooltip hướng dẫn
- **Thống kê chi tiết**: Hiển thị thông tin phân tích và độ tin cậy

## 1. Setup Môi Trường (Tự Động)

### Cách 1: Sử dụng script tự động (Khuyến nghị)
```powershell
# Chạy trong PowerShell
.\setup.ps1
```

Script này sẽ:
- ✅ Kiểm tra Python version
- 🏠 Tạo môi trường ảo `videotransformer_env`
- 📦 Cài đặt tất cả dependencies
- 🎮 Kiểm tra GPU support
- 🚀 Chạy quick test

### Cách 2: Setup thủ công

```bash
# Tạo môi trường ảo
python -m venv videotransformer_env

# Kích hoạt môi trường ảo (Windows)
.\videotransformer_env\Scripts\Activate.ps1

# Cài đặt packages
pip install -r requirements.txt
```

## 2. Kích Hoạt Môi Trường Ảo

### Windows PowerShell:
```powershell
# Cách 1: Sử dụng script helper
.\activate_env.ps1

# Cách 2: Kích hoạt trực tiếp
.\videotransformer_env\Scripts\Activate.ps1
```

### Windows Command Prompt:
```cmd
REM Sử dụng batch file
activate_env.bat

REM Hoặc kích hoạt trực tiếp
videotransformer_env\Scripts\activate.bat
```

### Kiểm tra môi trường đã active:
```bash
# Python path sẽ trỏ đến môi trường ảo
python -c "import sys; print(sys.executable)"

# Kiểm tra packages
pip list
```

## 3. Cài đặt Dependencies

## 3. Cài đặt Dependencies

### Nếu đã chạy setup.ps1:
Dependencies đã được cài đặt tự động trong môi trường ảo.

### Nếu cài thủ công:
```bash
# Đảm bảo môi trường ảo đã được kích hoạt
pip install -r requirements.txt

# Hoặc cài đặt từng package
pip install torch torchvision pytorch-lightning
pip install opencv-python matplotlib seaborn scikit-learn
pip install flask werkzeug numpy pandas tqdm requests
```

## 4. Chuẩn bị Model

Đảm bảo bạn có một trong các file model sau đã được huấn luyện:

### Các đường dẫn model được hỗ trợ:
1. **`C:\Users\anhvu\Documents\linh tinh\vivit_model.pth`** - PyTorch model file (ưu tiên đầu tiên)
2. **`C:\Users\anhvu\Documents\linh tinh\epoch=4-step=75.ckpt`** - PyTorch Lightning checkpoint
3. **`video_transformer_model.ckpt`** - Fallback model file

### Định dạng model:
- **`.ckpt`**: PyTorch Lightning checkpoint format
- **`.pth`** hoặc **`.pt`**: PyTorch standard format

System sẽ tự động tìm và load model theo thứ tự ưu tiên trên. Nếu không tìm thấy, bạn có thể:
- Đặt file model vào thư mục dự án
- Hoặc sửa đường dẫn trong code

## 5. Cách sử dụng

### ⚠️ Quan trọng: Luôn kích hoạt môi trường ảo trước khi làm việc!

```powershell
# Kích hoạt môi trường ảo
.\activate_env.ps1
```

### 5.1. Test Model với Script

```bash
# Chạy script test tương tác
python test_video_transformer_model.py
```

Chọn loại test:
1. **Test một video đơn lẻ**: Dự đoán cho 1 video cụ thể
2. **Test folder có cấu trúc**: Test nhiều video với ground truth
3. **Test folder không nhãn**: Test nhiều video không biết kết quả
4. **Benchmark tốc độ**: Đo tốc độ xử lý

### 5.2. Chạy Web App

```bash
# Khởi động Flask server
python app.py
```

### 5.3. Định nghĩa "Gian lận" trong hệ thống

#### Các loại hành vi được coi là gian lận:
1. **Deepfake/Face Swap**: Khuôn mặt được thay thế bằng AI
2. **Video Manipulation**: Video bị chỉnh sửa, cắt ghép
3. **Synthetic Content**: Video được tạo hoàn toàn bằng AI
4. **Identity Fraud**: Sử dụng danh tính giả mạo
5. **Document Fraud**: Giả mạo giấy tờ trong video
6. **Staged/Scripted**: Video dàn dựng giả tạo tình huống

#### Các dấu hiệu nhận biết gian lận:
- **Bất thường về ánh sáng**: Ánh sáng không đồng nhất trên khuôn mặt
- **Chuyển động không tự nhiên**: Cử chỉ, biểu cảm bất thường
- **Chất lượng không đồng đều**: Một phần video rõ nét, phần khác mờ
- **Artifacts kỹ thuật số**: Nhiễu, méo, glitch không tự nhiên
- **Mismatch âm thanh**: Giọng nói không khớp với chuyển động môi
- **Temporal inconsistency**: Không nhất quán về thời gian

#### Model phân loại:
- **Class 0**: Video bình thường (Normal/Authentic)
- **Class 1**: Video gian lận (Fraud/Manipulated)
- **Confidence Score**: Độ tin cậy từ 0.0 đến 1.0

#### Ngưỡng phân loại:
- **Confidence > 0.7**: Kết quả đáng tin cậy
- **0.5 < Confidence ≤ 0.7**: Cần xem xét thủ công
- **Confidence ≤ 0.5**: Không chắc chắn, cần kiểm tra lại

#### Các trường hợp thực tế:
1. **Video selfie xác thực danh tính**: 
   - ✅ Bình thường: Người thật, ánh sáng tự nhiên, chuyển động mượt
   - ❌ Gian lận: Deepfake, ảnh in, màn hình phát video

2. **Video phỏng vấn/họp online**:
   - ✅ Bình thường: Tương tác tự nhiên, đồng bộ âm thanh
   - ❌ Gian lận: Pre-recorded video, AI-generated avatar

3. **Video xác thực tài liệu**:
   - ✅ Bình thường: Giấy tờ thật, chữ ký tự nhiên
   - ❌ Gian lận: Photoshop documents, fake signatures

### 5.4. Sử dụng API

#### Python Example:
```python
import requests

# Upload video qua API
url = "http://localhost:5000/predict"
files = {"file": open("test_video.mp4", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']['label']}")
print(f"Confidence: {result['prediction']['confidence']:.4f}")
```

#### cURL Example:
```bash
curl -X POST -F "file=@test_video.mp4" http://localhost:5000/predict
```

## 6. Cấu trúc dữ liệu test

### Cho test có ground truth:
```
test_data/
├── normal/          # Video bình thường (label = 0)
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fraud/           # Video gian lận (label = 1)
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### Cho test không có ground truth:
```
test_videos/
├── video1.mp4
├── video2.mp4
├── video3.avi
└── ...
```

## 7. Kết quả đầu ra

### Test Script:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Thời gian xử lý trung bình
- Kết quả chi tiết cho từng video
- File JSON với kết quả đầy đủ

### Web App:
- Giao diện web thân thiện
- Upload và xem kết quả trực tuyến
- Hiển thị confidence score
- Thống kê thời gian xử lý

### API Response:
```json
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
```

## 8. Troubleshooting

### Lỗi thường gặp:

1. **Import Error**: 
   - Kiểm tra môi trường ảo đã được kích hoạt chưa
   - Cài đặt đầy đủ dependencies từ requirements.txt
   - Kiểm tra Python version (3.8+)

2. **Môi trường ảo không hoạt động**:
   - Chạy lại: `.\setup.ps1`
   - Hoặc tạo thủ công: `python -m venv videotransformer_env`
   - Kiểm tra execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

3. **CUDA Error**:
   - Kiểm tra CUDA version tương thích với PyTorch
   - Sử dụng CPU nếu không có GPU: device='cpu'

4. **Model Load Error**:
   - Kiểm tra đường dẫn file .ckpt
   - Đảm bảo model architecture phù hợp

5. **Video Processing Error**:
   - Kiểm tra format video được hỗ trợ
   - Đảm bảo video không bị corrupted

6. **Memory Error**:
   - Giảm batch size
   - Sử dụng video độ phân giải thấp hơn
   - Đóng các ứng dụng khác

## 9. Tối ưu hiệu suất

### Để tăng tốc độ:
1. Sử dụng GPU (CUDA)
2. Giảm resolution video input
3. Giảm số frames xử lý
4. Sử dụng model nhỏ hơn nếu có

### Để tăng độ chính xác phát hiện gian lận:
1. **Chất lượng video input**:
   - Sử dụng video HD (1080p trở lên)
   - Tốc độ frame ổn định (30fps)
   - Ánh sáng đầy đủ, không quá tối

2. **Cài đặt model**:
   - Tăng số frames xử lý (8-16 frames)
   - Sử dụng ensemble nhiều models
   - Fine-tune trên dữ liệu cụ thể của bạn

3. **Kỹ thuật preprocessing**:
   - Chuẩn hóa ánh sáng
   - Giảm nhiễu video
   - Stabilization nếu video bị rung

### Xử lý false positive/negative:
- **False Positive** (Báo gian lận nhầm):
  - Video chất lượng kém bị nhận diện nhầm
  - Ánh sáng bất thường (backlight, shadow)
  - Makeup đậm, filter camera

- **False Negative** (Bỏ sót gian lận):
  - Deepfake chất lượng cao
  - Partial manipulation (chỉ sửa một phần)
  - Sophisticated AI generation

## 10. Monitoring và Logs

### Script logs:
- Kết quả in ra console
- File JSON lưu kết quả chi tiết
- Confusion matrix PNG

### Web app logs:
- Request/response logs
- Processing time tracking
- Error logging

## 11. Production Deployment

### Để deploy production:
1. Sử dụng WSGI server (Gunicorn, uWSGI)
2. Reverse proxy (Nginx)
3. Load balancing cho multiple instances
4. Database logging
5. Monitoring system

### Docker deployment:
```dockerfile
FROM python:3.8-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 12. Quản lý Môi trường Ảo

### Các lệnh hữu ích:

```powershell
# Kích hoạt môi trường ảo
.\videotransformer_env\Scripts\Activate.ps1
# hoặc
.\activate_env.ps1

# Hủy kích hoạt
deactivate

# Kiểm tra packages đã cài
pip list

# Cập nhật package
pip install --upgrade package_name

# Xóa môi trường ảo (nếu cần)
Remove-Item -Recurse -Force videotransformer_env

# Tạo lại môi trường ảo
python -m venv videotransformer_env
```

### Backup môi trường:
```bash
# Xuất danh sách packages
pip freeze > requirements_backup.txt

# Khôi phục từ backup
pip install -r requirements_backup.txt
```

## 13. Best Practices cho phát hiện Fraud

### Khi nào nên sử dụng:
1. **KYC (Know Your Customer)**: Xác thực danh tính khách hàng
2. **Video calls quan trọng**: Phỏng vấn, họp kinh doanh
3. **Content moderation**: Kiểm tra video upload
4. **Digital forensics**: Điều tra pháp lý

### Quy trình khuyến nghị:
1. **Pre-processing**: Kiểm tra chất lượng video trước
2. **AI Detection**: Chạy qua model phát hiện
3. **Human Review**: Xem xét thủ công kết quả nghi ngờ
4. **Decision Making**: Quyết định cuối cùng dựa trên tổng hợp

### Giới hạn và lưu ý:
- ⚠️ **Không 100% chính xác**: Luôn cần xác nhận thủ công
- ⚠️ **Privacy concerns**: Tuân thủ quy định bảo mật dữ liệu
- ⚠️ **Bias potential**: Model có thể thiên vị với một số nhóm
- ⚠️ **Technology evolution**: Cần cập nhật thường xuyên

### Compliance và Legal:
- Tuân thủ GDPR, CCPA về privacy
- Thông báo rõ việc sử dụng AI detection
- Lưu trữ audit logs cho truy vết
- Có quy trình appeal cho false positive

## 14. Support

Nếu gặp vấn đề, kiểm tra:
1. Python và package versions
2. Model file integrity
3. Video file formats
4. System resources (RAM, GPU memory)
5. Network connectivity (cho API calls)

Happy testing! 🎬✨
