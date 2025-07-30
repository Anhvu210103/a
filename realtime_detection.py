import cv2
import torch
import numpy as np
import time
import os
import datetime
from collections import deque
from threading import Thread
import queue
import json
from video_transformer_model import VideoTransformer
from vivit_model import ViViTModel
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Cấu hình
MODEL_PATH = r"C:\Users\anhvu\Documents\linh tinh\vivit_model.pth"  # Đường dẫn đến model
FRAME_BUFFER_SIZE = 150  # Số frame giữ trong bộ đệm (~5 giây với 30fps)
DETECTION_INTERVAL = 15  # Giảm để có nhiều dự đoán hơn (cứ 15 frames = 0.5s)
CONFIDENCE_THRESHOLD = 0.3  # Giảm ngưỡng tin cậy để dễ phát hiện hơn
SAVE_FOLDER = "detected_violations"  # Thư mục lưu video
CLASS_NAMES = ["Bình thường", "Gian lận"]  # Tên các lớp

# Tạo thư mục lưu video nếu chưa tồn tại
os.makedirs(SAVE_FOLDER, exist_ok=True)

class RealtimeDetector:
    def __init__(self, model_path, source=0, detection_interval=8, frame_buffer_size=150, 
                 confidence_threshold=0.3, save_folder="detected_violations", progress_callback=None, 
                 debug_mode=True):
        """
        Khởi tạo detector real-time
        Args:
            model_path: Đường dẫn đến model
            source: Camera ID hoặc đường dẫn video (0 là webcam mặc định)
            detection_interval: Số frame giữa các lần dự đoán
            frame_buffer_size: Số frame giữ trong buffer (để lưu trước/sau khi phát hiện)
            confidence_threshold: Ngưỡng tin cậy để xác định sai phạm
            save_folder: Thư mục lưu video sai phạm
            progress_callback: Hàm callback để cập nhật tiến độ (nhận giá trị 0-100)
            debug_mode: Bật chế độ debug để hiển thị thông tin chi tiết
        """
        self.model_path = model_path
        self.source = source
        self.detection_interval = detection_interval
        self.frame_buffer_size = frame_buffer_size
        self.confidence_threshold = confidence_threshold
        self.save_folder = save_folder
        self.progress_callback = progress_callback
        self.debug_mode = debug_mode
        
        # Khởi tạo các biến
        self.model = None
        self.cap = None
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.frame_count = 0
        self.running = False
        self.current_violation = False
        self.violation_start_time = None
        self.detection_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        # Khởi tạo HOG descriptor cho việc phát hiện người
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Lưu trữ vị trí người được phát hiện
        self.person_bboxes = []
        self.last_gradcam = None
        
        # Thống kê cho debug
        self.total_predictions = 0
        self.fraud_predictions = 0
        self.max_confidence = 0.0
        
        # Tải model
        self._load_model()
        
    def _load_model(self):
        """Tải model cho dự đoán"""
        print(f"[INFO] Dang tai model tu {self.model_path}...")
        
        try:
            # Kiểm tra xem là model ViViT hay VideoTransformer
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Kiểm tra các key để xác định loại model
            if any(key.startswith('spatial_transformer') for key in checkpoint.keys()):
                print("[INFO] Phat hien model ViViT...")
                self.model = ViViTModel()
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                print("[INFO] Phat hien model VideoTransformer...")
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
                }
                self.model = VideoTransformer(**model_kwargs)
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            print("[SUCCESS] Da tai model thanh cong!")
            
        except Exception as e:
            print(f"[ERROR] Loi khi tai model: {e}")
            raise e
    
    def _detect_people(self, frame):
        """
        Phát hiện người trong frame
        Args:
            frame: Frame hình ảnh (BGR)
        Returns:
            List các bounding boxes của người phát hiện được [(x, y, w, h)]
        """
        # Giảm kích thước để tăng tốc độ xử lý
        scale = min(1.0, 800 / max(frame.shape[0], frame.shape[1]))
        if scale < 1.0:
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            frame_resized = cv2.resize(frame, (width, height))
        else:
            frame_resized = frame
            
        # Phát hiện người
        boxes, weights = self.hog.detectMultiScale(
            frame_resized, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        # Lọc kết quả có độ tin cậy cao
        filtered_boxes = []
        for i, box in enumerate(boxes):
            if weights[i][0] > 0.5:  # Lọc dựa trên confidence
                # Đưa boxes về kích thước frame gốc
                if scale < 1.0:
                    box = (box / scale).astype(int)
                filtered_boxes.append(tuple(box))
                
        return filtered_boxes
    
    def _generate_gradcam(self, frame_tensor, target_class=1):
        """
        Tạo Grad-CAM để giải thích dự đoán
        Args:
            frame_tensor: Tensor đầu vào cho model
            target_class: Lớp mục tiêu để tạo Grad-CAM (mặc định là lớp gian lận)
        Returns:
            Heatmap Grad-CAM
        """
        # Đảm bảo model ở chế độ eval và gradient được tính
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        frame_tensor.requires_grad_()
        
        # Lấy ra activation từ lớp cuối cùng (tùy vào kiến trúc model)
        activations = None
        gradients = None
        
        # Định nghĩa hook để lấy activation và gradient
        def save_activation(module, input, output):
            nonlocal activations
            activations = output
        
        def save_gradient(grad):
            nonlocal gradients
            gradients = grad
        
        # Đăng ký hook (có thể cần điều chỉnh tên lớp tùy thuộc vào kiến trúc model)
        if hasattr(self.model, 'transformer'):
            # Hook cho VideoTransformer
            last_layer = self.model.transformer.layers[-1]
        else:
            # Giả định cho ViViT
            last_layer = self.model.spatial_transformer.layers[-1]
            
        handle = last_layer.register_forward_hook(save_activation)
        
        # Forward pass
        output = self.model(frame_tensor)
        
        # Tính gradient cho lớp mục tiêu
        if output.shape[1] > target_class:
            score = output[0, target_class]
            score.backward()
        else:
            print("[WARNING] Target class nằm ngoài phạm vi output")
            handle.remove()
            return None
        
        # Gỡ hook
        handle.remove()
        
        # Kiểm tra xem activation và gradient đã được lưu chưa
        if activations is None or gradients is None:
            print("[ERROR] Không thể lấy được activation hoặc gradient")
            return None
        
        # Tính toán trọng số của các feature maps
        weights = torch.mean(gradients, dim=[0, 2, 3])  # Global Average Pooling
        
        # Tạo CAM
        batch_size, num_channels, height, width = activations.shape
        cam = torch.zeros((height, width), dtype=torch.float32)
        
        # Trọng số nhân với activation
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
            
        # Áp dụng ReLU
        cam = F.relu(cam)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Chuyển về numpy và resize về kích thước gốc
        cam = cam.detach().cpu().numpy()
        return cam
    
    def _preprocess_frames(self, frames):
        """
        Tiền xử lý frames cho model
        Args:
            frames: List các frames cần xử lý
        Returns:
            Tensor đã xử lý cho model
        """
        processed_frames = []
        
        # ImageNet mean và std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        for frame in frames:
            # Resize và chuyển sang RGB nếu cần
            frame = cv2.resize(frame, (224, 224))
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 3:  # BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            
            processed_frames.append(frame)
        
        # Chuyển thành tensor
        # Format: (C, T, H, W) cho PyTorch
        processed_frames = np.array(processed_frames)
        processed_frames = processed_frames.transpose(3, 0, 1, 2)  # (C, T, H, W)
        
        return torch.FloatTensor(processed_frames).unsqueeze(0)  # Add batch dim
    
    def _detection_worker(self):
        """Worker thread để thực hiện dự đoán"""
        while self.running:
            try:
                if not self.detection_queue.empty():
                    frames = self.detection_queue.get(timeout=1)
                    
                    # Phát hiện người trong frame gốc
                    self.person_bboxes = []
                    for frame in frames:
                        # Lấy một số frame để phát hiện người
                        if len(self.person_bboxes) == 0:
                            self.person_bboxes = self._detect_people(frame)
                    
                    # Xử lý frames và dự đoán
                    start_time = time.time()
                    video_tensor = self._preprocess_frames(frames)
                    
                    # Dự đoán với gradient
                    self.model.zero_grad()
                    
                    # Dự đoán không có gradient cho hiệu suất
                    with torch.no_grad():
                        outputs = self.model(video_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Nếu phát hiện vi phạm, tạo Grad-CAM
                    gradcam = None
                    if predicted_class == 1 and confidence >= self.confidence_threshold:
                        # Tạo Grad-CAM cho việc giải thích
                        try:
                            gradcam = self._generate_gradcam(video_tensor, target_class=1)
                            self.last_gradcam = gradcam
                        except Exception as e:
                            print(f"[WARNING] Khong the tao Grad-CAM: {e}")
                    
                    processing_time = time.time() - start_time
                    
                    # Đưa kết quả vào queue
                    self.result_queue.put({
                        'class': predicted_class,
                        'confidence': confidence,
                        'time': processing_time,
                        'gradcam': gradcam,
                        'person_bboxes': self.person_bboxes
                    })
                else:
                    time.sleep(0.01)  # Tránh CPU cao khi không có việc
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Loi trong detection worker: {e}")
    
    def _apply_gradcam_to_frame(self, frame, gradcam):
        """
        Áp dụng heatmap Grad-CAM lên frame
        Args:
            frame: Frame hình ảnh
            gradcam: Heatmap Grad-CAM
        Returns:
            Frame với heatmap Grad-CAM
        """
        if gradcam is None:
            return frame
        
        # Resize gradcam về kích thước frame
        height, width = frame.shape[:2]
        gradcam_resized = cv2.resize(gradcam, (width, height))
        
        # Chuyển đổi heatmap sang màu
        heatmap = np.uint8(255 * gradcam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap lên frame với alpha blending
        alpha = 0.4
        output = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return output
    
    def _get_explanation_text(self, confidence):
        """
        Tạo văn bản giải thích dựa trên độ tin cậy
        Args:
            confidence: Độ tin cậy của dự đoán
        Returns:
            Văn bản giải thích
        """
        if confidence >= 0.9:
            return "Muc do nghiem trong: CAO - He thong phat hien duoc cac dau hieu gian lan ro rang"
        elif confidence >= 0.8:
            return "Muc do nghiem trong: TRUNG BINH - He thong phat hien duoc cac hanh vi dang nghi ngo"
        else:
            return "Muc do nghiem trong: THAP - Co dau hieu bat thuong can xem xet them"
    
    def _save_violation_video(self, frames, timestamp, confidence):
        """
        Lưu video vi phạm với bounding box và giải thích AI
        Args:
            frames: List các frames cần lưu
            timestamp: Thời điểm phát hiện
            confidence: Độ tin cậy của dự đoán
        """
        # Tạo tên file với timestamp
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        conf_str = f"{confidence:.2f}"
        filename = os.path.join(self.save_folder, f"violation_{time_str}_{conf_str}.mp4")
        
        # Lấy kích thước frame
        height, width = frames[0].shape[:2]
        
        # Tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        
        # Văn bản giải thích
        explanation = self._get_explanation_text(confidence)
        
        # Ghi các frame với bounding box và giải thích
        for i, frame in enumerate(frames):
            # Phát hiện người mỗi 15 frames để tăng tốc độ
            if i % 15 == 0 and len(self.person_bboxes) == 0:
                self.person_bboxes = self._detect_people(frame)
            
            # Vẽ bounding box cho người
            display_frame = frame.copy()
            for (x, y, w, h) in self.person_bboxes:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Thêm nhãn cho bounding box
                cv2.putText(display_frame, "Person", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Áp dụng Grad-CAM nếu có
            if self.last_gradcam is not None:
                display_frame = self._apply_gradcam_to_frame(display_frame, self.last_gradcam)
            
            # Thêm khung thông tin giải thích
            cv2.rectangle(display_frame, (10, height - 120), (width - 10, height - 20), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, height - 120), (width - 10, height - 20), (0, 0, 255), 2)
            
            # Thêm văn bản giải thích
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            color = (255, 255, 255)
            
            cv2.putText(display_frame, f"PHAT HIEN GIAN LAN - Do tin cay: {confidence:.2f}", 
                        (20, height - 90), font, scale, color, thickness)
                        
            cv2.putText(display_frame, explanation, 
                        (20, height - 60), font, scale, color, thickness)
                        
            cv2.putText(display_frame, "Khuyen nghi: Kiem tra lai video va xac minh hanh vi", 
                        (20, height - 30), font, scale, color, thickness)
            
            # Ghi frame
            out.write(display_frame)
        
        out.release()
        print(f"[SUCCESS] Da luu video vi pham: {filename}")
        
        # Lưu một hình ảnh thumbnail
        thumbnail_file = os.path.join(self.save_folder, f"violation_{time_str}_{conf_str}.jpg")
        middle_frame_idx = len(frames) // 2
        if middle_frame_idx < len(frames):
            # Chọn frame giữa để làm thumbnail
            middle_frame = frames[middle_frame_idx].copy()
            
            # Vẽ thông tin lên thumbnail
            for (x, y, w, h) in self.person_bboxes:
                cv2.rectangle(middle_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Thêm thông tin violation lên thumbnail
            cv2.putText(middle_frame, f"Gian lan - {confidence:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Áp dụng Grad-CAM nếu có
            if self.last_gradcam is not None:
                middle_frame = self._apply_gradcam_to_frame(middle_frame, self.last_gradcam)
                
            cv2.imwrite(thumbnail_file, middle_frame)
        
        # Tạo file metadata
        meta_file = os.path.join(self.save_folder, f"violation_{time_str}_{conf_str}.json")
        metadata = {
            "timestamp": time_str,
            "confidence": confidence,
            "class": "Gian lận",
            "file": filename,
            "thumbnail": thumbnail_file,
            "explanation": explanation,
            "num_persons_detected": len(self.person_bboxes)
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def start(self):
        """Bắt đầu phát hiện real-time"""
        # Mở camera hoặc video
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở camera hoặc video: {self.source}")
        
        # Lấy thông tin video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Video info: {width}x{height} @ {fps}fps")
        
        if fps <= 0:  # Nếu không đọc được fps từ camera
            fps = 30
        
        # Khởi động detection thread
        self.running = True
        detection_thread = Thread(target=self._detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Biến cho tracking
        last_detection_frame = 0
        violation_frames = []
        current_violation = False
        violation_start_time = None
        
        print("[INFO] Bat dau phat hien real-time...")
        print("[INFO] Huong dan: Nhan 'q' de thoat, 's' de luu frame hien tai")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    # Nếu là video file, có thể reset để loop
                    if isinstance(self.source, str) and os.path.exists(self.source):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Thêm frame vào buffer
                self.frame_buffer.append(frame.copy())
                self.frame_count += 1
                
                # Chuẩn bị frame để hiển thị
                display_frame = frame.copy()
                
                # Thực hiện dự đoán mỗi detection_interval frames
                if self.frame_count - last_detection_frame >= self.detection_interval:
                    # Nếu đủ frames trong buffer và queue trống
                    if len(self.frame_buffer) >= 8 and self.detection_queue.empty():
                        # Lấy 8 frames đều nhau từ buffer
                        indices = np.linspace(0, len(self.frame_buffer) - 1, 8, dtype=int)
                        frames_for_detection = [list(self.frame_buffer)[i] for i in indices]
                        
                        # Đặt vào queue để xử lý
                        self.detection_queue.put(frames_for_detection)
                        last_detection_frame = self.frame_count
                
                # Kiểm tra kết quả từ detection thread
                if not self.result_queue.empty():
                    result = self.result_queue.get()
                    predicted_class = result['class']
                    confidence = result['confidence']
                    processing_time = result['time']
                    gradcam = result.get('gradcam')
                    person_bboxes = result.get('person_bboxes', [])
                    
                    # Hiển thị kết quả
                    label = CLASS_NAMES[predicted_class]
                    color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
                    
                    # Lưu lại gradcam để sử dụng cho các frame khác
                    if gradcam is not None:
                        self.last_gradcam = gradcam
                    
                    # Lưu lại vị trí người được phát hiện
                    if person_bboxes:
                        self.person_bboxes = person_bboxes
                    
                    # Phát hiện vi phạm
                    if predicted_class == 1 and confidence >= self.confidence_threshold:
                        if not current_violation:
                            # Bắt đầu vi phạm mới
                            current_violation = True
                            violation_start_time = datetime.datetime.now()
                            print(f"[WARNING] Phat hien gian lan: {confidence:.2f}")
                            
                            # Bắt đầu thu thập frames
                            violation_frames = list(self.frame_buffer)
                        else:
                            # Tiếp tục thu thập frames trong vi phạm
                            violation_frames.extend([frame.copy()])
                    else:
                        # Nếu không còn vi phạm nhưng đang trong quá trình vi phạm
                        if current_violation:
                            # Kết thúc vi phạm, lưu video
                            # Thêm một số frame sau khi kết thúc vi phạm
                            violation_frames.extend([frame.copy() for _ in range(min(15, len(self.frame_buffer)))])
                            
                            # Lưu video vi phạm với các bounding box và giải thích
                            self._save_violation_video(violation_frames, violation_start_time, confidence)
                            
                            # Reset trạng thái
                            current_violation = False
                            violation_start_time = None
                            violation_frames = []
                    
                    # Vẽ bounding box cho người được phát hiện
                    for (x, y, w, h) in self.person_bboxes:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(display_frame, "Person", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Áp dụng Grad-CAM nếu phát hiện vi phạm
                    if current_violation and self.last_gradcam is not None:
                        display_frame = self._apply_gradcam_to_frame(display_frame, self.last_gradcam)
                    
                    # Hiển thị thông tin trên frame
                    cv2.putText(display_frame, f"Class: {label}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display_frame, f"Conf: {confidence:.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display_frame, f"Time: {processing_time:.3f}s", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Hiển thị trạng thái vi phạm
                if current_violation:
                    cv2.putText(display_frame, "VIOLATION DETECTED", (width - 300, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 3)
                    
                    # Hiển thị giải thích nếu đang có vi phạm
                    explanation = self._get_explanation_text(confidence)
                    cv2.putText(display_frame, explanation, (10, height - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Hiển thị frame
                cv2.imshow("Realtime Fraud Detection", display_frame)
                
                # Kiểm tra phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Lưu frame hiện tại
                    screenshot_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"{self.save_folder}/screenshot_{screenshot_time}.jpg", frame)
                    print(f"[INFO] Da luu screenshot: screenshot_{screenshot_time}.jpg")
        
        finally:
            # Dừng và dọn dẹp
            self.running = False
            detection_thread.join(timeout=1)
            self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Da dung phat hien real-time")
    
    def process_video_file(self, video_path, output_path=None):
        """
        Xử lý video file và tạo video output với đánh dấu các khoảng thời gian vi phạm
        Args:
            video_path: Đường dẫn đến video cần xử lý
            output_path: Đường dẫn lưu video output (nếu None sẽ tự động tạo)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")
        
        if output_path is None:
            filename = os.path.basename(video_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.save_folder, f"{name}_analyzed.mp4")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")
        
        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video info: {width}x{height} @ {fps}fps, {total_frames} frames (~{total_frames/fps:.1f} giây)")
        
        # Tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Biến cho tracking
        # Tăng frame_buffer_size để có thể xử lý toàn bộ video
        extended_buffer_size = max(self.frame_buffer_size, total_frames)
        frame_buffer = deque(maxlen=extended_buffer_size)
        frame_count = 0
        last_detection_frame = 0
        current_violations = []  # Danh sách các vi phạm đã phát hiện [(start_frame, end_frame, confidence)]
        current_violation = False
        violation_start_frame = 0
        violation_confidence = 0
        
        # Lưu tất cả frames để xử lý
        all_frames = []
        
        # Tạo file kết quả JSON
        results_file = os.path.splitext(output_path)[0] + "_results.json"
        results = {
            "video": video_path,
            "analysis_time": datetime.datetime.now().isoformat(),
            "violations": [],
            "total_frames": total_frames,
            "fps": fps,
            "duration": total_frames / fps
        }
        
        print(f"[INFO] Dang phan tich video {video_path}...")
        print(f"[INFO] Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        try:
            # Đọc tất cả frames trước
            print("[INFO] Đang đọc toàn bộ video...")
            all_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                all_frames.append(frame.copy())
                frame_buffer.append(frame.copy())
                frame_count += 1
                
                # Hiển thị tiến độ
                if frame_count % 100 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    print(f"[INFO] Dang doc video: {progress:.1f}% ({frame_count}/{total_frames})")
                    
                    # Gọi callback nếu có
                    if self.progress_callback:
                        self.progress_callback(progress / 2)  # Đọc chiếm 50% tiến độ
            
            # Reset lại biến đếm và cap để xử lý video
            frame_count = 0
            total_frames = len(all_frames)
            
            print(f"[INFO] Da doc xong {total_frames} frames. Dang xu ly...")
            
            # Xử lý từng frame
            for frame_idx, frame in enumerate(all_frames):
                frame_count = frame_idx + 1
                
                # Hiển thị tiến độ
                if frame_count % 100 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    print(f"[INFO] Dang xu ly: {progress:.1f}% ({frame_count}/{total_frames})")
                    
                    # Gọi callback nếu có
                    if self.progress_callback:
                        self.progress_callback(50 + progress / 2)  # Xử lý chiếm 50% tiến độ còn lại
                
                # Thực hiện dự đoán mỗi detection_interval frames
                if frame_count - last_detection_frame >= self.detection_interval:
                    # Chuẩn bị frames cho detection
                    start_idx = max(0, frame_idx - 7)
                    if frame_idx - start_idx + 1 >= 8:
                        # Lấy 8 frames đều nhau từ vị trí hiện tại trở về trước
                        indices = np.linspace(start_idx, frame_idx, 8, dtype=int)
                        frames_for_detection = [all_frames[i] for i in indices]
                        
                        # Tiền xử lý và dự đoán
                        video_tensor = self._preprocess_frames(frames_for_detection)
                        
                        with torch.no_grad():
                            outputs = self.model(video_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            predicted_class = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                            
                            # Cập nhật thống kê
                            self.total_predictions += 1
                            if predicted_class == 1:
                                self.fraud_predictions += 1
                            self.max_confidence = max(self.max_confidence, confidence)
                            
                            # Debug information - hiển thị mọi dự đoán
                            if self.debug_mode:
                                normal_conf = probabilities[0][0].item()
                                fraud_conf = probabilities[0][1].item()
                                print(f"[DEBUG] Frame {frame_count}: Normal={normal_conf:.3f}, Fraud={fraud_conf:.3f}, Pred={predicted_class}")
                            
                            # Nếu phát hiện vi phạm, tạo Grad-CAM
                            if predicted_class == 1 and confidence >= self.confidence_threshold:
                                try:
                                    gradcam = self._generate_gradcam(video_tensor, target_class=1)
                                    self.last_gradcam = gradcam
                                except Exception as e:
                                    print(f"[WARNING] Khong the tao Grad-CAM: {e}")
                        
                        # Điều chỉnh logic phát hiện vi phạm - dễ dàng hơn
                        # Sử dụng fraud confidence thay vì predicted class
                        fraud_confidence = probabilities[0][1].item()
                        
                        if fraud_confidence >= self.confidence_threshold:
                            if not current_violation:
                                # Bắt đầu vi phạm mới
                                current_violation = True
                                violation_start_frame = frame_count
                                violation_confidence = fraud_confidence
                                
                                # Phát hiện người trong frame hiện tại
                                self.person_bboxes = self._detect_people(frame)
                                
                                print(f"[WARNING] Phat hien gian lan tai frame {frame_count}: {fraud_confidence:.3f}")
                        else:
                            # Nếu không còn vi phạm nhưng đang trong quá trình vi phạm
                            if current_violation:
                                # Kết thúc vi phạm, lưu thông tin
                                current_violations.append((violation_start_frame, frame_count, violation_confidence))
                                
                                # Thêm vào kết quả JSON
                                start_time = violation_start_frame / fps
                                end_time = frame_count / fps
                                results["violations"].append({
                                    "start_frame": violation_start_frame,
                                    "end_frame": frame_count,
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "duration": end_time - start_time,
                                    "confidence": float(violation_confidence)
                                })
                                
                                # Reset trạng thái
                                current_violation = False
                                
                                print(f"[INFO] Ket thuc gian lan tai frame {frame_count}")
                        
                        last_detection_frame = frame_count
                
                # Vẽ thông tin lên frame
                display_frame = frame.copy()
                
                # Phát hiện người mỗi 30 frames để tăng tốc độ
                if frame_count % 30 == 0:
                    self.person_bboxes = self._detect_people(frame)
                
                # Vẽ bounding box cho người được phát hiện
                for (x, y, w, h) in self.person_bboxes:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Person", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Đánh dấu các vi phạm đã phát hiện
                in_violation = False
                current_confidence = 0
                
                for start_frame, end_frame, conf in current_violations:
                    if start_frame <= frame_count <= end_frame:
                        in_violation = True
                        current_confidence = conf
                        
                        # Áp dụng Grad-CAM nếu có
                        if self.last_gradcam is not None:
                            display_frame = self._apply_gradcam_to_frame(display_frame, self.last_gradcam)
                
                        cv2.putText(display_frame, "VIOLATION DETECTED", (width - 300, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 3)
                        cv2.putText(display_frame, f"Confidence: {conf:.2f}", (width - 300, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Thêm khung giải thích nếu đang có vi phạm
                if in_violation:
                    # Thêm khung thông tin giải thích
                    cv2.rectangle(display_frame, (10, height - 120), (width - 10, height - 20), (0, 0, 0), -1)
                    cv2.rectangle(display_frame, (10, height - 120), (width - 10, height - 20), (0, 0, 255), 2)
                    
                    # Thêm văn bản giải thích
                    explanation = self._get_explanation_text(current_confidence)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.7
                    thickness = 2
                    color = (255, 255, 255)
                    
                    cv2.putText(display_frame, f"PHAT HIEN GIAN LAN - Do tin cay: {current_confidence:.2f}", 
                                (20, height - 90), font, scale, color, thickness)
                                
                    cv2.putText(display_frame, explanation, 
                                (20, height - 60), font, scale, color, thickness)
                                
                    cv2.putText(display_frame, "Khuyen nghi: Kiem tra lai video va xac minh hanh vi", 
                                (20, height - 30), font, scale, color, thickness)
                
                # Hiển thị thời gian video
                time_sec = frame_count / fps
                cv2.putText(display_frame, f"Time: {int(time_sec//60):02d}:{int(time_sec%60):02d}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Ghi frame vào video output
                out.write(display_frame)
            
            # Xử lý vi phạm cuối cùng nếu video kết thúc trong khi đang có vi phạm
            if current_violation:
                current_violations.append((violation_start_frame, total_frames, violation_confidence))
                
                # Thêm vào kết quả JSON
                start_time = violation_start_frame / fps
                end_time = total_frames / fps
                results["violations"].append({
                    "start_frame": violation_start_frame,
                    "end_frame": total_frames,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "confidence": float(violation_confidence)
                })
                
                print(f"[INFO] Ket thuc gian lan tai cuoi video (frame {total_frames})")
        
        finally:
            # Đóng video và lưu kết quả
            cap.release()
            out.release()
            # cv2.destroyAllWindows()
            
            # Hiển thị thống kê debug
            if self.debug_mode:
                print(f"[DEBUG] Thong ke phan tich:")
                print(f"[DEBUG] - Tong so du doan: {self.total_predictions}")
                print(f"[DEBUG] - So lan du doan gian lan: {self.fraud_predictions}")
                print(f"[DEBUG] - Ti le gian lan: {self.fraud_predictions/max(1, self.total_predictions)*100:.1f}%")
                print(f"[DEBUG] - Do tin cay cao nhat: {self.max_confidence:.3f}")
                print(f"[DEBUG] - Nguong tin cay hien tai: {self.confidence_threshold}")
            
            # Lưu kết quả JSON
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] Da phan tich xong video!")
            print(f"[INFO] Ket qua: Phat hien {len(current_violations)} vi pham")
            print(f"[INFO] Video output: {output_path}")
            print(f"[INFO] File ket qua: {results_file}")
            
            # Nếu không có vi phạm nào được phát hiện, đưa ra gợi ý
            if len(current_violations) == 0:
                print(f"[INFO] Khong phat hien vi pham nao. Co the:")
                print(f"[INFO] - Giam nguong tin cay (hien tai: {self.confidence_threshold})")
                print(f"[INFO] - Kiem tra lai model va du lieu")
                print(f"[INFO] - Video khong chua hanh vi gian lan")
            
            return output_path, results_file

# Hàm main để chạy trực tiếp từ command line
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Fraud Detection")
    parser.add_argument("--source", default=0, help="Camera ID hoặc đường dẫn video")
    parser.add_argument("--model", default=MODEL_PATH, help="Đường dẫn đến model")
    parser.add_argument("--mode", default="realtime", choices=["realtime", "analyze"],
                       help="Chế độ: realtime hoặc analyze video file")
    parser.add_argument("--output", default=None, help="Đường dẫn output khi phân tích video")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, 
                       help="Ngưỡng tin cậy để xác định vi phạm")
    
    args = parser.parse_args()
    
    detector = RealtimeDetector(
        model_path=args.model,
        source=args.source if args.source == 0 else args.source,  # Convert to int if possible
        confidence_threshold=args.threshold
    )
    
    if args.mode == "realtime":
        detector.start()
    else:
        detector.process_video_file(args.source, args.output)

if __name__ == "__main__":
    main()
