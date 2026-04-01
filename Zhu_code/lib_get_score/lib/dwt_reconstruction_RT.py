import numpy as np
import pywt


def rigrsure_threshold(detail):
    """
    实现 MATLAB thselect('rigrsure') 的逻辑，用于计算细节系数的阈值。
    参考：Donoho & Johnstone 的 SURE 方法。

    Parameters:
        detail: 一维数组，细节系数

    Returns:
        thresh: 计算得到的阈值
    """
    detail = np.asarray(detail)
    n = detail.size
    if n == 0:
        return 0.0

    # 以绝对值（平方）排序
    sorted_abs = np.sort(np.abs(detail))
    sorted_sq = sorted_abs**2
    # 累积和
    csum = np.cumsum(sorted_sq)

    # 计算 SURE 风险值
    risks = np.empty(n)
    for i in range(n):
        # i+1 为当前计数（MATLAB 下标从1开始）
        risks[i] = n - 2 * (i + 1) + csum[i] + (n - (i + 1)) * sorted_sq[i]

    # 当阈值为0时风险为 n
    risk0 = n
    min_risk = np.min(risks)
    if risk0 <= min_risk:
        thresh = 0.0
    else:
        idx = np.argmin(risks)
        thresh = sorted_abs[idx]
    return thresh


def dwt_reconstruction_RT(raw_signal, level, wname):
    """
    使用小波分解对信号进行重构，
    过程参考 MATLAB 代码：
      1. 小波分解 (wavedec)
      2. 对各级细节系数用 thselect('rigrsure') 计算阈值
      3. 分层通过 wdencmp 重构（对前 i 级细节系数作软阈值处理）

    Parameters:
        raw_signal: 输入原始信号（数组）
        level: 分解的层数
        wname: 小波基名称，如 'db6'

    Returns:
        dwt_signal: 重构后的信号矩阵，每一列对应不同层次的重构结果
    """
    # 转换为 numpy 数组
    x = np.array(raw_signal)
    N = level
    # 小波分解，coeffs[0] 为近似系数，其余为各级细节系数
    # pywt.wavedec 返回 [cA_n, cD_n, cD_{n-1}, …, cD1]
    coeffs = pywt.wavedec(x, wname, level=N)

    # 计算每一级细节系数的阈值，使用 rigrsure 方法
    TR = []
    # MATLAB 中 for i=1:N,  detcoef(C,L,i) 返回第 i 级细节
    # 对应 Python 中 coeffs[-i]（注意：coeffs[-1] 为最低分辨率细节，即第1级）
    for i in range(1, N + 1):
        detail = coeffs[-i]
        thr = rigrsure_threshold(detail)
        TR.append(thr)

    # 重构：对每一层分别对前 i 级细节系数进行软阈值处理，然后重构信号
    Y_list = []
    for i in range(1, N + 1):
        new_coeffs = coeffs.copy()
        for j in range(1, i + 1):
            # 对第 j 级（从低频到高频）的细节系数进行软阈值处理，
            # 注意：TR[j-1] 顺序与 MATLAB 的 TR(1:i) 一致
            new_coeffs[-j] = pywt.threshold(new_coeffs[-j], TR[j - 1], mode="soft")
        # 重构信号
        rec = pywt.waverec(new_coeffs, wname)
        # 保证重构信号长度与原始信号一致
        rec = rec[: len(x)]
        Y_list.append(rec)

    # 将各层结果按列组合为矩阵
    dwt_signal = np.column_stack(Y_list)
    return dwt_signal
