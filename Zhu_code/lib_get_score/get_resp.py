import numpy as np
import time
from scipy.signal import filtfilt, detrend, kaiserord, firwin, sosfiltfilt,butter
from scipy.interpolate import UnivariateSpline, make_interp_spline

from .lib.dwt_reconstruction_RT import dwt_reconstruction_RT
from .lib.RR_estimator_rms_RT import RR_estimator_rms_RT
from .lib.ref_cto_RT import ref_cto_RT
from .lib.two_average_detector_err_ret import two_average_detector_err_ret
from .lib.qrs_detect2_err_ret_sign import qrs_detect2_err_ret_sign

def get_resp(ecg, fs=500):
    ecg_arr = np.asarray(ecg)
    if ecg_arr.ndim not in (1, 2):
        raise ValueError(f"ecg 只能是 1D 或 2D 数组，实际 ndim={ecg_arr.ndim}")
    if ecg_arr.ndim == 1:
        ecg_arr = ecg_arr[:, np.newaxis]
    a,b = ecg_arr.shape
    if a<b:
        ecg_arr = ecg_arr.T
    ch_num = ecg_arr.shape[1]
    err_code_list = []
    resp_rms_list = []
    resp_rsa_list = [] 
    for i in range(ch_num):
        err_code,rr_RSA, rr_RMS  = get_resp_onechannel(ecg_arr[:,i], fs)
        err_code_list.append(err_code)
        resp_rms_list.append(rr_RMS)
        resp_rsa_list.append(rr_RSA)
    return err_code_list,resp_rms_list,resp_rsa_list
def get_resp_onechannel(ecg, fs):
    """
    Extract real-time respiratory waves and rates from single-lead ECG.

    Parameters:
        ecg: numpy 数组，经过滤波的 ECG 信号 (N×1), 要求 N >= 20000
        fs: 采样率，单位 Hz

    Returns:
        err_code: 0 表示无错误，非0表示错误
        rr_RSA: RSA 算法得到的呼吸率，可能为 np.nan
        rr_RMS: RMS 算法得到的呼吸率，可能为 np.nan
    """
    RMS_cal = 0
    RSA_cal = 1

    err_code = 0
    rr_RSA = -1
    rr_RMS = -1


    # 确保 ecg 为一维数组
    ecg = np.array(ecg).flatten()
    # 确保 ecg 长度大于等于 20000
    if len(ecg) < 10000:
        err_code = -1
        return err_code, rr_RSA, rr_RMS
    level = 10
    wname = "db6"
    # dwt_reconstruction_RT 需自行实现或导入
    dwt_signal = dwt_reconstruction_RT(ecg, level, wname)

    # MATLAB 中 ecg_denoise = dwt_signal(:,1) - dwt_signal(:,9)
    # Python 中下标从 0 开始
    ecg_denoise = dwt_signal[:, 0] - dwt_signal[:, 8]

    # qrs_detect2_err_ret 和 two_average_detector_err_ret 需自行实现或导入
    err_code, qrs_cECG, _ = qrs_detect2_err_ret_sign(ecg_denoise, 0.3, 0.25, fs)

    if err_code != 0:
        err_code, qrs_cECG = two_average_detector_err_ret(ecg_denoise, fs, 1)
    if err_code != 0:
        return err_code, rr_RSA, rr_RMS

    time_vec = np.arange(1, len(ecg_denoise) + 1)

    # RSA 计算部分
    if RSA_cal == 1:
        RRI = np.diff(qrs_cECG) / fs
        HRV = 60.0 / RRI
        position = qrs_cECG[1:]
        if len(position) <= 3:
            return err_code, rr_RSA, rr_RMS

        # 使用 spline 进行插值
        spl_RSA = UnivariateSpline(position, HRV, s=0)
        RSA_EDR_interpled = spl_RSA(time_vec)

        spl_RSA = make_interp_spline(position, HRV, bc_type="not-a-knot")
        RSA_EDR_interpled = spl_RSA(time_vec)

    # RMS 计算部分
    if RMS_cal == 1:
        # RR_estimator_rms_RT 需自行实现或导入，返回 rms 和 rpeaks
        rms, rpeaks = RR_estimator_rms_RT(ecg_denoise, qrs_cECG, fs)
        # spl_RMS = UnivariateSpline(rpeaks, rms, s=0)
        # RMS_EDR_interpled = spl_RMS(time_vec)

        # 假设 rpeaks, rms, time 已经定义，其中
        # rpeaks: x 轴数据（采样点或时间），
        # rms: 对应的 y 轴数据，
        # time: 新的插值点序列。
        spl_RMS = make_interp_spline(rpeaks, rms, bc_type="not-a-knot")
        RMS_EDR_interpled = spl_RMS(time_vec)

    # 滤波部分
    Fs = fs  # 采样频率, 单位 Hz
    Fpass = 2  # 通带频率 Hz
    Fstop = 2.5  # 阻带频率 Hz
    Dpass = 0.057501127785  # 通带纹波 (线性值)
    Dstop = 0.0031622776602  # 阻带衰减 (线性值)

    # 转换阻带要求为 dB
    attenuation_db = -20 * np.log10(Dstop)
    # 计算归一化的过渡带宽度
    width = (Fstop - Fpass) / (Fs / 2)
    N, beta = kaiserord(attenuation_db, width)
    # firwin 中 cutoff 为归一化截止频率
    cutoff = Fpass / (Fs / 2)
    # b = firwin(N, cutoff, window=("kaiser", beta), pass_zero=True)
    sos = butter(4, Fpass, btype="low", fs=fs, output="sos")
    # 去趋势后双向滤波
    if RSA_cal == 1:
        xr = detrend(RSA_EDR_interpled)
        RSA_cECG = sosfiltfilt(sos, xr)
        # RSA_cECG = filtfilt(b, [1.0], detrend(RSA_EDR_interpled))
    if RMS_cal == 1:
        xr = detrend(RMS_EDR_interpled)
        RMS_cECG = sosfiltfilt(sos, xr)
        # RMS_cECG = filtfilt(b, [1.0], detrend(RMS_EDR_interpled))

    # 创建平均滤波器
    window_size = int(2 * fs)
    avg_filter = np.ones(window_size) / window_size

    # 使用卷积实现滤波，mode='same' 模拟 filter 函数的作用
    if RSA_cal == 1:
        RSA_cECG = np.convolve(RSA_cECG, avg_filter, mode="same")
    # RSA_cECG输出csv
    # np.savetxt("RSA_cECG.csv", RSA_cECG, delimiter=",")
    if RMS_cal == 1:
        RMS_cECG = np.convolve(RMS_cECG, avg_filter, mode="same")
    # RMS_cECG输出csv
    # np.savetxt("RMS_cECG.csv", RMS_cECG, delimiter=",")

    # 计算呼吸率，ref_cto_RT 需自行实现或导入
    if RSA_cal == 1:
        data = {
            "t": np.arange(1, len(RSA_cECG) + 1) / fs,
            "v": RSA_cECG - np.mean(RSA_cECG),
        }
        rr_RSA = ref_cto_RT(data)
    if RMS_cal == 1:
        data = {
            "t": np.arange(1, len(RMS_cECG) + 1) / fs,
            "v": RMS_cECG - np.mean(RMS_cECG),
        }
    rr_RMS = ref_cto_RT(data)
    return err_code, rr_RSA, rr_RMS


if __name__ == '__main__':
    import pandas as pd
    # ecg = pd.read_csv(r"C:\Users\rewalk\Documents\WXWork\1688857197660272\Cache\File\2025-04\20250425114503544.csv",index_col=False)
    # ecg = ecg.columns
    ecg = pd.read_csv(r"D:\work_file\床垫数据分析\Bio-Mattress_V2.1.1-20250408\02冯建成\session_20250408-182411-仰睡-A-监控区\ECGData_1.csv").values
    err_code, rr_RSA, rr_RMS = get_resp(ecg[:,:], 500)
    print(err_code, rr_RSA, rr_RMS)