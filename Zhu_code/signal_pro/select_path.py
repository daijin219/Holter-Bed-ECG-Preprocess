import tkinter as tk
from tkinter import filedialog
def select_path():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 选择文件夹
    folder_path = filedialog.askdirectory(title="请选择文件夹")
    if folder_path:
        print("选择的文件夹:", folder_path)

    
    return folder_path

def select_path_file():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 选择文件（取消注释使用）
    file_path = filedialog.askopenfilename(title="请选择文件")
    if file_path:
        print("选择的文件:", file_path)
    
    return file_path