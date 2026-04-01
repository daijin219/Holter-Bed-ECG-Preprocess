# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt

def two_average_detector_err_ret(unfiltered_ecg, FS, correct):
    """Detects R-peaks in ECG using dual moving average method"""
    # 初始化返回参数
    err_code = 0
    qrs = np.array([])
    
    # 带通滤波器参数
    F1 = 8 / FS
    F2 = 20 / FS
    low = F1 * 2
    high = F2 * 2
    
    # 设计滤波器
    b, a = butter(2, [low, high], btype='bandpass')
    
    try:
        # 应用滤波器（使用零相位滤波）
        filtered_ecg = filtfilt(b, a, unfiltered_ecg)
    except ValueError as e:
        err_code = -2
        return err_code, qrs

    # 计算移动平均窗口
    window1 = int(np.floor(0.12 * FS))
    mwa_qrs = moving_window_ave(np.abs(filtered_ecg), window1)

    window2 = int(np.floor(0.6 * FS))
    mwa_beat = moving_window_ave(np.abs(filtered_ecg), window2)
    import matplotlib.pyplot as plt 
    plt.plot(mwa_beat)
    plt.plot(mwa_qrs,color="yellow")
    # 创建块信号
    block_height = np.max(filtered_ecg)
    blocks = np.where(mwa_qrs > mwa_beat, block_height, 0)

    # 检测R峰
    qrs_buffer = np.zeros(int(3.5 * len(unfiltered_ecg) / FS) + 1, dtype=int)
    qrs_idx = 0
    start = 0

    for i in range(1, len(blocks)):
        if blocks[i-1] == 0 and blocks[i] == block_height:
            start = i
        elif blocks[i-1] == block_height and blocks[i] == 0:
            end = i - 1
            if end - start > int(0.08 * FS):
                segment = filtered_ecg[start:end+2]  # +2补偿索引差异
                detection = np.argmax(segment) + start
                
                if qrs_idx > 0:
                    if detection - qrs_buffer[qrs_idx-1] > int(0.3 * FS):
                        if qrs_idx >= len(qrs_buffer):
                            qrs_buffer = np.append(qrs_buffer, np.zeros_like(qrs_buffer))
                        qrs_buffer[qrs_idx] = detection
                        qrs_idx += 1
                else:
                    qrs_buffer[qrs_idx] = detection
                    qrs_idx += 1

    valid_qrs = qrs_buffer[:qrs_idx]
    plt.scatter(valid_qrs, mwa_qrs[valid_qrs],color = "black")
    # 执行校正
    if correct and len(valid_qrs) > 0:
        winlen = 11
        err_code, corrected = Rpeak_correction(unfiltered_ecg, valid_qrs - 18, FS, winlen)
        if err_code != 0:
            return err_code, np.array([])
        qrs = np.unique(corrected)
    else:
        qrs = valid_qrs

    # 错误检查
    if len(qrs) < 2:
        err_code = -1
        return err_code, qrs

    return err_code, qrs

def moving_window_ave(input_array, window_size):
    """滑动平均计算（优化版）"""
    n = len(input_array)
    mwa = np.zeros(n)
    
    # 使用cumsum优化计算
    cumsum = np.cumsum(np.insert(input_array, 0, 0))
    
    for i in range(1, n+1):
        if i <= window_size:
            mwa[i-1] = cumsum[i] / i
        else:
            mwa[i-1] = (cumsum[i] - cumsum[i-window_size]) / window_size
    
    return mwa

def Rpeak_correction(signal, rpeaks, fs, winlen):
    """R峰位置校正"""
    if len(rpeaks) < 2:
        return -1, np.array([-1, -1, -1])
    
    RRI = np.diff(rpeaks)
    RRI_median = np.median(RRI)
    
    # 计算窗口长度
    win = int(0.1 * RRI_median)
    peaks_corrected = []
    
    for peak in rpeaks:
        start = max(0, peak - win)
        end = min(len(signal)-1, peak + win)
        
        segment = signal[start:end+1]
        if len(segment) == 0:
            continue
            
        new_peak = np.argmax(segment) + start
        peaks_corrected.append(new_peak)
    
    # 去除重复并排序
    peaks_corrected = np.unique(peaks_corrected)
    
    # 幅度筛选
    filtered_peaks = []
    signal = signal - np.mean(signal)
    for peak in peaks_corrected:
        if 0.5 * np.median(signal[peaks_corrected]) <= signal[peak] <= 1.5 * np.median(signal[peaks_corrected]):
            filtered_peaks.append(peak)
    
    return 0, np.array(filtered_peaks)