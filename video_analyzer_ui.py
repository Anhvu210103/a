import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from threading import Thread
import time
import subprocess
import platform
from realtime_detection import RealtimeDetector, MODEL_PATH, CONFIDENCE_THRESHOLD

# Class hiển thị tooltip khi hover chuột qua widget
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)
    
    def show_tip(self, event=None):
        """Hiển thị tooltip khi di chuột qua widget"""
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Tạo cửa sổ tooltip
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(tw, text=self.text, background="#FFFFEA", 
                        relief="solid", borderwidth=1, padding=(5, 3))
        label.pack()
    
    def hide_tip(self, event=None):
        """Ẩn tooltip khi di chuột ra khỏi widget"""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class VideoAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phan tich video gian lan")
        self.root.geometry("700x500")
        self.root.minsize(600, 450)
        
        # Bien theo doi qua trinh
        self.detection_running = False
        self.video_path = None
        self.output_path = None
        self.result_video_path = None  # Lưu đường dẫn video kết quả
        
        self.setup_ui()
    
    def setup_ui(self):
        # Khung chinh
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tieu de
        title_label = ttk.Label(main_frame, text="PHAN TICH VIDEO GIAN LAN", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Khung chua cac lua chon
        options_frame = ttk.LabelFrame(main_frame, text="Tuy chon", padding="10")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Hang 1: Chon che do
        mode_frame = ttk.Frame(options_frame)
        mode_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(mode_frame, text="CHE DO:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="realtime")
        realtime_radio = ttk.Radiobutton(mode_frame, text="Camera truc tiep", variable=self.mode_var, 
                       value="realtime", command=self.update_ui)
        realtime_radio.pack(side=tk.LEFT, padx=15)
        ToolTip(realtime_radio, "Phan tich truc tiep tu camera")
        
        analyze_radio = ttk.Radiobutton(mode_frame, text="Phan tich video", variable=self.mode_var, 
                       value="analyze", command=self.update_ui)
        analyze_radio.pack(side=tk.LEFT, padx=15)
        ToolTip(analyze_radio, "Phan tich tu file video co san")
        
        # Hang 2: Chon camera hoac video
        self.source_frame = ttk.Frame(options_frame)
        self.source_frame.pack(fill=tk.X, pady=10)
        
        # Hang 3: Threshold
        threshold_frame = ttk.Frame(options_frame)
        threshold_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(threshold_frame, text="NGUONG TIN CAY:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        self.threshold_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, length=250,
                                  variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, padx=5)
        ToolTip(threshold_scale, "Dieu chinh nguong tin cay: gia tri thap hon se phat hien nhieu hon (co the co false positive)")
        
        # Hiển thị giá trị threshold
        def show_threshold_value(*args):
            threshold_value_label.config(text=f"{self.threshold_var.get():.2f}")
            # Hiển thị gợi ý dựa trên giá trị threshold
            if self.threshold_var.get() < 0.3:
                suggestion = " (Rat nhay cam)"
            elif self.threshold_var.get() < 0.5:
                suggestion = " (Nhay cam)"
            elif self.threshold_var.get() < 0.7:
                suggestion = " (Can bang)"
            else:
                suggestion = " (Kat khe)"
            threshold_value_label.config(text=f"{self.threshold_var.get():.2f}{suggestion}")
        
        self.threshold_var.trace("w", show_threshold_value)
        threshold_value_label = ttk.Label(threshold_frame, text=f"{self.threshold_var.get():.2f}", 
                                        width=15, font=("Arial", 10, "bold"))
        threshold_value_label.pack(side=tk.LEFT, padx=5)
        
        # Khung trang thai
        status_frame = ttk.LabelFrame(main_frame, text="Trang thai", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        self.status_text.insert(tk.END, "San sang phat hien gian lan.\n")
        self.status_text.insert(tk.END, "Chon che do va nhan 'BAT DAU' de khoi chay.\n")
        self.status_text.config(state=tk.DISABLED)
        
        # Thanh tien trinh
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Khung nut
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=15)
        
        # Nút BAT DAU to và rõ ràng
        self.start_button = ttk.Button(button_frame, text="BAT DAU", command=self.start_detection, 
                                      style="Accent.TButton", width=15)
        self.start_button.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Thêm gợi ý khi di chuột qua nút
        ToolTip(self.start_button, "Nhan vao day de bat dau phan tich video")
        
        self.stop_button = ttk.Button(button_frame, text="DUNG", command=self.stop_detection, 
                                    state=tk.DISABLED, width=15)
        self.stop_button.pack(side=tk.LEFT, padx=20)
        ToolTip(self.stop_button, "Dung qua trinh phan tich dang chay")
        
        open_folder_button = ttk.Button(button_frame, text="MO THU MUC KET QUA", 
                                      command=self.open_result_folder, width=20)
        open_folder_button.pack(side=tk.RIGHT, padx=20)
        ToolTip(open_folder_button, "Mo thu muc chua video ket qua da phan tich")
        
        # Nút phát video kết quả
        self.play_video_button = ttk.Button(button_frame, text="PHAT VIDEO KET QUA", 
                                          command=self.play_result_video, width=20, 
                                          state=tk.DISABLED)
        self.play_video_button.pack(side=tk.RIGHT, padx=5)
        ToolTip(self.play_video_button, "Phat video ket qua sau khi phan tich xong")
        
        # Style cho nút bấm và giao diện
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Arial", 11, "bold"))
        
        # Thêm màu nền cho các nút quan trọng
        try:
            # Thử sử dụng theme có sẵn để giao diện đẹp hơn
            if "vista" in self.style.theme_names():
                self.style.theme_use("vista")
            elif "clam" in self.style.theme_names():
                self.style.theme_use("clam")
                
            # Tùy chỉnh màu sắc cho nút "BAT DAU" nổi bật
            self.style.map("Accent.TButton",
                background=[('active', '#0078D7'), ('pressed', '#005A9E'), ('!disabled', '#0078D7')],
                foreground=[('pressed', 'white'), ('active', 'white'), ('!disabled', 'white')])
                
        except Exception:
            # Nếu không thể thay đổi theme, giữ nguyên mặc định
            pass
        
        # Cap nhat UI dua tren che do
        self.update_ui()
    
    def update_ui(self):
        """Cap nhat UI dua tren che do da chon"""
        mode = self.mode_var.get()
        
        # Xoa cac widget cu trong source_frame
        for widget in self.source_frame.winfo_children():
            widget.pack_forget()
        
        if mode == "realtime":
            # Hiển thị lựa chọn camera
            ttk.Label(self.source_frame, text="CAMERA ID:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
            
            self.camera_var = tk.StringVar(value="0")
            self.camera_entry = ttk.Combobox(self.source_frame, textvariable=self.camera_var, 
                                         values=["0", "1", "2", "3"], width=5)
            self.camera_entry.pack(side=tk.LEFT, padx=10)
            ToolTip(self.camera_entry, "Chon ID cua camera: 0 (camera mac dinh), 1, 2, 3,...")
        else:
            # Hiển thị lựa chọn file video
            ttk.Label(self.source_frame, text="FILE VIDEO:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
            
            # Nút chọn file to và rõ ràng
            browse_button = ttk.Button(self.source_frame, text="CHON FILE VIDEO", 
                                     command=self.browse_video,
                                     style="Accent.TButton")
            browse_button.pack(side=tk.LEFT, padx=5)
            ToolTip(browse_button, "Nhan vao day de chon file video can phan tich")
            
            # Hiển thị tên file đã chọn
            self.video_path_var = tk.StringVar(value="Chua chon video")
            video_path_label = ttk.Label(self.source_frame, textvariable=self.video_path_var, 
                                      width=35)
            video_path_label.pack(side=tk.LEFT, padx=10)
    
    def browse_video(self):
        """Mo hop thoai chon file video"""
        file_path = filedialog.askopenfilename(
            title="Chon file video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            # Hiển thị tên file, không phải đường dẫn đầy đủ
            filename = os.path.basename(file_path)
            self.video_path_var.set(filename)
            self.add_log(f"Da chon video: {filename}")
    
    def add_log(self, message):
        """Them thong bao vao o trang thai"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def update_progress(self, value):
        """Cap nhat thanh tien trinh"""
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def play_result_video(self):
        """Phát video kết quả"""
        if not self.result_video_path or not os.path.exists(self.result_video_path):
            messagebox.showwarning("Canh bao", "Khong tim thay video ket qua")
            return
        
        try:
            # Phát video bằng chương trình mặc định của hệ thống
            if platform.system() == "Windows":
                os.startfile(self.result_video_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.result_video_path])
            else:  # Linux
                subprocess.run(["xdg-open", self.result_video_path])
                
            self.add_log(f"Dang phat video: {os.path.basename(self.result_video_path)}")
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the phat video: {str(e)}")
    
    def open_result_folder(self):
        """Mo thu muc chua ket qua"""
        result_dir = os.path.abspath("detected_violations")
        if os.path.exists(result_dir):
            os.startfile(result_dir)
        else:
            messagebox.showinfo("Thong bao", "Thu muc ket qua chua duoc tao")
    
    def start_detection(self):
        """Bat dau qua trinh phat hien"""
        if self.detection_running:
            return
            
        mode = self.mode_var.get()
        threshold = self.threshold_var.get()
        
        if mode == "realtime":
            try:
                camera_id = int(self.camera_var.get())
                source = camera_id
            except ValueError:
                messagebox.showerror("Loi", "Camera ID khong hop le")
                return
        else:  # analyze
            if not hasattr(self, 'video_path') or not self.video_path or not os.path.exists(self.video_path):
                messagebox.showerror("Loi", "Vui long chon file video hop le")
                return
            source = self.video_path
        
        # Cap nhat UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.detection_running = True
        self.add_log(f"Bat dau phat hien ({mode})...")
        
        # Chay phat hien trong thread rieng
        detection_thread = Thread(target=self._run_detection, 
                                  args=(mode, source, threshold))
        detection_thread.daemon = True
        detection_thread.start()
    
    def _run_detection(self, mode, source, threshold):
        """Chay qua trinh phat hien trong thread rieng"""
        try:
            # Khoi tao detector
            detector = RealtimeDetector(
                model_path=MODEL_PATH,
                source=source,
                confidence_threshold=threshold,
                progress_callback=self.update_progress
            )
            
            # Ghi de phuong thuc in cua detector de cap nhat UI
            original_print = print
            def ui_print(message):
                original_print(message)  # Van in ra console
                self.root.after(0, lambda: self.add_log(message))
            
            # Gan ham moi
            import builtins
            builtins.print = ui_print
            
            # Chay detector
            if mode == "realtime":
                detector.start()
            else:
                output_path, results_file = detector.process_video_file(source)
                
                # Lưu đường dẫn video kết quả
                self.result_video_path = output_path
                
                # Kích hoạt nút phát video
                self.root.after(0, lambda: self.play_video_button.config(state=tk.NORMAL))
                
                # Tự động phát video kết quả
                self.root.after(0, lambda: self.auto_play_result(output_path))
                
                # Hiển thị thông báo hoàn thành với tùy chọn xem ngay
                self.root.after(0, lambda: self.show_completion_dialog(output_path))
                
            # Khoi phuc ham print
            builtins.print = original_print
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Loi", str(e)))
        finally:
            # Cap nhat UI
            self.root.after(0, self._reset_ui)
    
    def auto_play_result(self, video_path):
        """Tự động phát video kết quả"""
        if video_path and os.path.exists(video_path):
            # Hỏi người dùng có muốn xem video ngay không
            response = messagebox.askyesno("Phat video", 
                                         "Ban co muon xem video ket qua ngay bay gio khong?")
            if response:
                self.result_video_path = video_path
                self.play_result_video()
    
    def show_completion_dialog(self, video_path):
        """Hiển thị dialog hoàn thành với các tùy chọn"""
        result = messagebox.askquestion("Hoan thanh", 
                                      "Da phan tich xong video!\n\n" +
                                      "Ban muon lam gi tiep theo?\n\n" +
                                      "Yes: Mo thu muc ket qua\n" +
                                      "No: Dong thong bao")
        
        if result == 'yes':
            self.open_result_folder()
    
    def _reset_ui(self):
        """Reset UI sau khi dung phat hien"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.detection_running = False
        self.add_log("Da dung phat hien.")
    
    def stop_detection(self):
        """Dung qua trinh phat hien"""
        if not self.detection_running:
            return
            
        self.add_log("Dang dung phat hien...")
        # Note: Detector duoc dung tu ben trong bang cach dong cua so OpenCV
        
        # Reset UI
        self._reset_ui()

def main():
    root = tk.Tk()
    app = VideoAnalyzerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
