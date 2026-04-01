# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import find_peaks,correlate
from scipy.stats import kurtosis, skew
from .qrs_detect2_err_ret_sign import qrs_detect2_err_ret_sign
from .two_average_detector_err_ret import two_average_detector_err_ret
from .template_matching import template_matching 
def feature_get(ecg, window, step, fs):
    err_code = 0
    features = np.ones(26)

    if len(ecg) <= window or len(ecg) <= step or step <= 0 or fs <= 0:
        err_code = -101
        return err_code, features
    # 基于R波检测的特征
    [err_code, qrs_i_raw, qrs_sign] = qrs_detect2_err_ret_sign(ecg, 0.3, 0.25, fs)
    if err_code != 0:
        [err_code, qrs_i_raw] = two_average_detector_err_ret(ecg, fs, 1)
        if err_code != 0:
            err_code = -102
            return err_code, features
        qrs_sign = 1
    if qrs_sign >= 0:
        x = ecg
    else:
        x = -ecg
    n = len(x)

    # 时域特征
    if max(abs(x)) == 0:
        err_code = -103
        return err_code, features
    
    a = x / max(abs(x))
    x_max = max(a)
    x_mean = np.mean(a)
    x_rms = np.sqrt(np.mean(a**2))
    x_var = np.var(a)
    x_kurt = kurtosis(a,fisher=False)
    x_skew = skew(a)

    x_abs = abs(a)
    x_peak = max(a) - min(a)

    if x_rms == 0 or np.mean(x_abs) == 0 or np.mean(np.sqrt(x_abs)) == 0:
        err_code = -104
        return err_code, features

    # 峰值因子
    crestfactor = x_peak / x_rms
    # 波形因子
    shapefactor = x_rms / np.mean(x_abs)
    # 脉冲因子
    impulsefactor = x_peak / np.mean(x_abs)
    # 裕度因子
    marginfactor = x_peak / np.mean(np.sqrt(x_abs))**2

    # 滑窗功率
    window_sample = np.floor(window * fs).astype(int)
    step_sample = np.floor(step * fs).astype(int)

    if window_sample == 0:
        err_code = -105
        return err_code, features
    
    fea_idx = 0
    pn = np.floor((n / fs - (window - step)) / step).astype(int)
    power = np.zeros(pn)
    for i in range(0, n - window_sample + 1, step_sample):
        fea_idx += 1
        power[fea_idx - 1] = np.sum(a[i:i + window_sample]**2) / window_sample
    max_window_power = np.max(power)

    # 波形形态长度
    wl = np.sum(np.abs(a[1:] - a[:-1]))

    # 频域特征
    a = a - x_mean
    cxn = correlate(a, a, mode='full', method='auto')
    F = np.abs(np.fft.fft(cxn))
    N = len(F)

    if N == 0:
        err_code = -106
        return err_code, features
    
    real_f = (np.arange(N // 2) * fs / N).reshape(-1, 1)
    real_y = np.asarray(2 * F[:N // 2] / N).reshape(-1, 1)
    
    if np.sum(real_y) == 0:
        err_code = -107
        return err_code, features

    # 频域熵
    energy = real_y / (np.sum(real_y) + 1e-12)
    fft_entropy = -np.sum(energy * np.log(energy + 1e-12))
    if np.isnan(fft_entropy):
        fft_entropy = -108
    
    # 中心频率
    fft_fc = np.sum(real_f * real_y) / np.sum(real_y)

    # 均方频率
    f2 = real_f**2
    fft_msf = np.sum(f2 * real_y) / np.sum(real_y)

    # 频率方差
    f3 = (real_f - fft_fc)**2
    fft_vf = np.sum(f3 * real_y) / np.sum(real_y)

    # basSQI, rate of power in band 1~40Hz and 0~40Hz 
    power1 = np.sum(real_y[2 * np.ceil(N / fs).astype(int):40 * np.ceil(N / fs).astype(int)])
    power2 = np.sum(real_y[:40 * np.ceil(N / fs).astype(int)])

    if power2 == 0:
        err_code = -109
        return err_code, features
    basSQI = power1 / power2

    # pSQI, rate of power in band 5~15Hz and 5~40Hz
    power3 = np.sum(real_y[5 * np.ceil(N / fs).astype(int):15 * np.ceil(N / fs).astype(int)])
    power4 = np.sum(real_y[5 * np.ceil(N / fs).astype(int):40 * np.ceil(N / fs).astype(int)])
    if power4 == 0:
        err_code = -110
        return err_code, features
    pSQI = power3 / power4

    # ACF feature, calculate mean of the second and third peak
    n_peak = 5
    pks, _ = find_peaks(cxn, distance=300)
    pks = np.sort(pks)[::-1]
    if len(pks) < n_peak:
        acf_peak = 0
    else:
        if pks[0] == 0:
            err_code = -111
            return err_code, features
        acf_peak = np.mean(pks[1:n_peak]) / pks[0]

    qrs_amp_raw = a[qrs_i_raw]
    # eSQI, power in (R-0.07s~R+0.08s) / total power 
    # rsdSQI, mean(std(R-0.07s~R+0.08s) / std(R-0.2s~R+0.2s))
    n_r = len(qrs_i_raw)

    b1 = round(0.07 * fs)
    b2 = round(0.08 * fs)
    b3 = round(0.2 * fs)

    e = 0
    r1 = 0
    r2 = 0
    r = np.zeros(n_r)
    
    # 判断边界
    for k in range(n_r):
        if qrs_i_raw[k] < b1:
            e += np.sum(a[:qrs_i_raw[k] + b2]**2)
        elif qrs_i_raw[k] > n - b2:
            e += np.sum(a[qrs_i_raw[k] - b1:]**2)
        else:
            e += np.sum(a[qrs_i_raw[k] - b1 + 1:qrs_i_raw[k] + b2]**2)

        if qrs_i_raw[k] < b1:
            r1 = np.std(a[:qrs_i_raw[k] + b2])
            r2 = np.std(a[:qrs_i_raw[k] + b3])
            
        elif qrs_i_raw[k] < 0.2 * fs and qrs_i_raw[k] > b1:
            r1 = np.std(a[qrs_i_raw[k] - b1:qrs_i_raw[k] + b2])
            r2 = np.std(a[:qrs_i_raw[k] + b3])
        
        elif qrs_i_raw[k] > b3 and qrs_i_raw[k] < n - b3:
            r1 = np.std(a[qrs_i_raw[k] - b1:qrs_i_raw[k] + b2])
            r2 = np.std(a[qrs_i_raw[k] - b3:qrs_i_raw[k] + b3])
        
        elif qrs_i_raw[k] > n - b3 and qrs_i_raw[k] < n - b2:
            r1 = np.std(a[qrs_i_raw[k] - b1:qrs_i_raw[k] + b2])
            r2 = np.std(a[qrs_i_raw[k] - b3:])
        
        elif qrs_i_raw[k] > n - b2:
            r1 = np.std(a[qrs_i_raw[k] - b1:])
            r2 = np.std(a[qrs_i_raw[k] - b3:])

        if r2 == 0:
            err_code = -112
            return err_code, features
        r[k] = r1 / r2

    if np.sum(a**2) == 0:
        err_code = -113
    eSQI = e / np.sum(a**2)

    rsdSQI = np.mean(r) if len(r) > 0 else 0

    if len(qrs_i_raw) <= 3 or len(qrs_amp_raw) <= 3:
        R = 0
        RR = 0
        err_code = -114
        return err_code, features
    else:
        r = np.sort(qrs_amp_raw)
        R = np.mean(r[:2])
        RR = (np.diff(qrs_i_raw)) / fs

    R_var = np.var(qrs_amp_raw)
    if len(RR) == 0:
        RR_var, RR_kurt, RR_skew = 0, 0, 0
    else:
        RR_var = np.var(RR)
        if len(RR) < 4 or np.allclose(RR, RR[0]):
            RR_kurt = 0
            RR_skew = 0
        else:
            RR_kurt = kurtosis(RR, fisher=True) if not np.isnan(kurtosis(RR, fisher=True)) else 0
            RR_skew = skew(RR) if not np.isnan(skew(RR)) else 0
    coe = template_matching(a, qrs_i_raw)

    features = np.array([x_max, x_rms, x_var, x_kurt, x_skew, crestfactor,
                         shapefactor, impulsefactor, marginfactor, max_window_power, wl, fft_entropy,
                         fft_fc, fft_msf, fft_vf, basSQI, pSQI, acf_peak, eSQI, rsdSQI, R, R_var, RR_var, RR_kurt, RR_skew, coe])

    if np.any(np.isnan(features)):
        print(np.where(np.isnan(features))[0])
        err_code = -115

    return err_code, features

