from signal_pro.read_dat import read_dat_file  
from scipy.signal import find_peaks
import numpy as np
from lib_get_score.lib.qrs_detect2_err_ret_sign import qrs_detect2_err_ret_sign
def cut_data(data,ch,fs=500):
    data_ch = data[-90*500:-60*500,ch]
    _,_,sign = qrs_detect2_err_ret_sign(data_ch)
    if sign > 0:
        sign = 1
    else:
        sign = -1
    data = data*sign
    index_pro = get_R_peak_standard(data,fs,ch)
    for i in range(len(index_pro)-3):
        if abs(index_pro[i+1] - index_pro[i] - 500) < 5 and abs(index_pro[i+2] - index_pro[i+1] - 500) < 5:
            start_index = index_pro[i+2]
            break
    for i in range(len(index_pro)-3):
        if abs(index_pro[-i-1] - index_pro[-i-2] - 500) < 5 and abs(index_pro[-i-2] - index_pro[-i-3] - 500) < 5:
            end_index = index_pro[-i-3]
            break
    output = data[start_index-2*fs//5:end_index+3*fs//5,:]
    output = output[71501+600000:71501+1200000]
    return output
def get_R_peak_standard(data,fs,channel):
    data_ch = data[:,channel]   
    index = find_peaks(data_ch,distance=200)[0]
    index_pro = []
    for i in index:
        try:
            if data_ch[i] - data_ch[i-fs//5] > 1 and data_ch[i] - data_ch[i+fs//5] > 1:
                index_pro.append(i)
        except:
            print("find R_peak error")  
    return index_pro




 

if __name__ == '__main__':
    print("I love you,Sir")