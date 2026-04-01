
from scipy.signal import butter, iirnotch, filtfilt
def apply_filters(data, fs=500):
    """
    应用3-30Hz带通滤波和50Hz陷波滤波
    
    参数:
        data: 输入信号
        fs: 采样频率(Hz)
    
    返回:
        滤波后的信号
    """
    # 先进行带通滤波(3-30Hz)
    filtered_data = bandpass_filter(data, 3, 40, fs)
    
    # 再进行50Hz陷波滤波
    filtered_data = notch_filter(filtered_data, 50, fs)
    
    return filtered_data
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    零相位带通滤波器(3-30Hz)
    
    参数:
        data: 输入信号
        lowcut: 低截止频率(Hz)
        highcut: 高截止频率(Hz)
        fs: 采样频率(Hz)
        order: 滤波器阶数
    
    返回:
        滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)  # 使用filtfilt实现零相位滤波
    return y

def notch_filter(data, freq, fs, quality=30):
    """
    50Hz陷波滤波器
    
    参数:
        data: 输入信号
        freq: 要滤除的频率(Hz)
        fs: 采样频率(Hz)
        quality: 滤波器品质因数
    
    返回:
        滤波后的信号
    """
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, quality)
    y = filtfilt(b, a, data)  # 使用filtfilt实现零相位滤波
    return y

