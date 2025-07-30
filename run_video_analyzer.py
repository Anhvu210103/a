#!/usr/bin/env python3
"""
Launcher cho ung dung phat hien gian lan
"""

import tkinter as tk
from video_analyzer_ui import VideoAnalyzerUI

def main():
    # Khoi tao giao dien
    root = tk.Tk()
    app = VideoAnalyzerUI(root)
    
    # Chay ung dung
    root.mainloop()

if __name__ == "__main__":
    main()
