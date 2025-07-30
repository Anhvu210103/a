#!/usr/bin/env python3
"""
Video Fraud Detection System - Main Launcher
Hệ thống phát hiện gian lận trong video sử dụng AI

Tác giả: Anh Vũ
Ngày tạo: 2025
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Thêm thư mục hiện tại vào Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_analyzer_ui import VideoAnalyzerUI
except ImportError as e:
    print(f"[ERROR] Không thể import module cần thiết: {e}")
    print("Vui lòng cài đặt các thư viện cần thiết bằng: pip install -r requirements.txt")
    sys.exit(1)

def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    missing_deps = []
    
    try:
        import torch
        import cv2
        import numpy
        import matplotlib
        from PIL import Image
    except ImportError as e:
        missing_deps.append(str(e))
    
    if missing_deps:
        error_msg = "Thiếu các thư viện sau:\n" + "\n".join(missing_deps)
        error_msg += "\n\nVui lòng cài đặt bằng: pip install -r requirements.txt"
        messagebox.showerror("Lỗi Dependencies", error_msg)
        return False
    
    return True

def main():
    """Hàm main chạy ứng dụng"""
    print("=" * 60)
    print("      VIDEO FRAUD DETECTION SYSTEM")
    print("      Hệ thống phát hiện gian lận video")
    print("=" * 60)
    print()
    
    # Kiểm tra dependencies
    if not check_dependencies():
        return
    
    # Tạo thư mục kết quả nếu chưa có
    result_dir = "detected_violations"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"[INFO] Đã tạo thư mục kết quả: {result_dir}")
    
    # Khởi tạo và chạy GUI
    try:
        root = tk.Tk()
        app = VideoAnalyzerUI(root)
        
        # Thiết lập icon và tiêu đề cửa sổ
        root.title("Video Fraud Detection System v1.0")
        
        # Căn giữa cửa sổ
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        print("[INFO] Ứng dụng đã khởi động thành công!")
        print("[INFO] Sử dụng giao diện để phân tích video gian lận")
        print()
        
        # Chạy main loop
        root.mainloop()
        
    except Exception as e:
        error_msg = f"Lỗi khi khởi động ứng dụng: {str(e)}"
        print(f"[ERROR] {error_msg}")
        messagebox.showerror("Lỗi", error_msg)
        return
    
    print("[INFO] Ứng dụng đã đóng.")

if __name__ == "__main__":
    main()
