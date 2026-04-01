import struct
import numpy as np

def read_dat_file(filename, num_channels=12):
    """
    读取.dat文件，每个数据2字节无符号整数，小端序，12通道
    
    参数:
        filename: 要读取的.dat文件路径
        num_channels: 数据通道数(默认为12)
        
    返回:
        一个形状为(n_samples, num_channels)的numpy数组
    """
    with open(filename, 'rb') as f:
        # 读取整个文件内容
        raw_data = f.read()
    
    # 计算总的数据点数
    num_samples = len(raw_data) // (2 * num_channels)
    
    # 使用struct解包二进制数据
    # '<'表示小端序，'H'表示无符号短整型(2字节)
    format_str = '<' + 'H' * num_channels * num_samples
    data_flat = struct.unpack(format_str, raw_data)
    
    # 转换为numpy数组并重塑为(n_samples, num_channels)
    data_array_u16 = np.array(data_flat, dtype=np.uint16).reshape(num_samples, num_channels)
    data_array = data_array_u16.astype(np.float32)
    del data_array_u16
    np.subtract(data_array, np.float32(32768.0), out=data_array)  # 原地减
    np.multiply(data_array, np.float32(1.0/218.0), out=data_array)  # 原地乘
    return data_array