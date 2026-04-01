import numpy as np

def ref_cto_RT(sum_both):
    """
    计算呼吸率（RR），基于信号 extrema 的检测

    参数:
        sum_both: dict，包含两个键：
            't': 时间向量
            'v': 信号向量

    返回:
        rr_cto: 计算得到的呼吸率（单位：次/分钟），当无法计算时返回 np.nan
    """
    v = np.array(sum_both['v'])
    t = np.array(sum_both['t'])
    
    # 识别峰值
    diff_v = np.diff(v)
    # 左侧斜率：取 diff_v[0:-1]，判断是否 > 0
    left_slopes = diff_v[:-1] > 0
    # 右侧斜率：取 diff_v[1:]，判断是否 < 0
    right_slopes = diff_v[1:] < 0
    # 两侧条件同时满足，索引需要加1（MATLAB中 +1）
    peaks = np.where(left_slopes & right_slopes)[0] + 1

    # 识别谷值
    left_slopes_trough = diff_v[:-1] < 0
    right_slopes_trough = diff_v[1:] > 0
    troughs = np.where(left_slopes_trough & right_slopes_trough)[0] + 1

    # 定义阈值
    if peaks.size == 0:
        return np.nan
    q3 = np.quantile(v[peaks], 0.75)
    thresh = 0.2 * q3

    # 找到相关的峰和谷
    rel_peaks = peaks[v[peaks] > thresh]
    rel_troughs = troughs[v[troughs] < 0]

    # 如果相关峰数量不足，返回 NaN
    if rel_peaks.size <= 1:
        return np.nan
    else:
        valid_cycles = np.zeros(rel_peaks.size - 1, dtype=bool)
        cycle_durations = np.full(rel_peaks.size - 1, np.nan)
        for i in range(rel_peaks.size - 1):
            # 在两个相邻的相关峰间找到所有相关谷
            cycle_troughs = rel_troughs[(rel_troughs > rel_peaks[i]) & (rel_troughs < rel_peaks[i+1])]
            if cycle_troughs.size == 1:
                valid_cycles[i] = True
                cycle_durations[i] = t[rel_peaks[i+1]] - t[rel_peaks[i]]
        if np.sum(valid_cycles) == 0:
            rr_cto = np.nan
        else:
            ave_breath_duration = np.nanmean(cycle_durations[valid_cycles])
            rr_cto = 60 / ave_breath_duration
        return rr_cto