import numpy as np

from .lib.qrs_detect2_err_ret_sign import qrs_detect2_err_ret_sign
from .lib.two_average_detector_err_ret import two_average_detector_err_ret

# 旧版hr选择策略
# def get_hr(ecg, fs):
#     err_code2, qrs_pos2, sign = qrs_detect2_err_ret_sign(ecg, 0.3, 0.25, fs)
#     if sign > 0 and err_code2 == 0:
#         err_code1, qrs_pos1 = two_average_detector_err_ret(ecg, fs, True)
#     elif sign < 0 and err_code2 == 0:
#         err_code1, qrs_pos1 = two_average_detector_err_ret(-ecg, fs, True)

#     # 计算 err_code
#     err_code = int(err_code1 or err_code2)

#     # 根据 err_code1 和 err_code2 的值设置 hr_code 和 qrs_pos
#     if err_code1 == 0 and err_code2 == 0:
#         hr_code = 0
#         qrs_pos = np.concatenate([qrs_pos1, [0], qrs_pos2])  # 合并数组
#     elif err_code1 == 0:
#         hr_code = 1
#         qrs_pos = np.concatenate([qrs_pos1, [0]])
#     elif err_code2 == 0:
#         hr_code = 2
#         qrs_pos = np.concatenate([[0], qrs_pos2])
#     else:
#         hr_code = -105
#         err_code = -105

#     # 如果 qrs_pos1 是空的
#     if qrs_pos1.size == 0:  # 检查 qrs_pos1 是否为空
#         hr = 0
#     else:
#         # 计算 hr1 和 hr2
#         hr1 = (len(qrs_pos1) - 1) / (((qrs_pos1[-1] - qrs_pos1[0]) / fs) / 60)
#         hr2 = (len(qrs_pos2) - 1) / (((qrs_pos2[-1] - qrs_pos2[0]) / fs) / 60)

#         if hr_code == 0:
#             if hr2 < 50 or hr2 > 190:
#                 hr = hr2
#             else:
#                 hr = np.mean([hr1, hr2])
#         elif hr_code == 1:
#             hr = hr1
#         elif hr_code == 2:
#             hr = hr2
#         else:
#             hr = 0
#         qrs_count = qrs_pos.size
#         qrs_pos = np.asarray(qrs_pos)
#     return err_code, hr, qrs_count, qrs_pos
def get_hr(ecg, fs):
    ecg_arr = np.asarray(ecg)
    if ecg_arr.ndim not in (1, 2):
        raise ValueError(f"ecg 只能是 1D 或 2D 数组，实际 ndim={ecg_arr.ndim}")
    if ecg_arr.ndim == 1:
        ecg_arr = ecg_arr[:, np.newaxis]
    a,b = ecg_arr.shape
    if a<b:
        ecg_arr = ecg_arr.T
    ch_num = ecg_arr.shape[1]
    all_err_code1 = []
    all_err_code2 = []
    all_hr1 = []
    all_hr2 = []
    all_qrs_pos1 = []
    all_qrs_pos2 = []
    
    for i in range(ch_num):
        err_code1, err_code2, hr1, hr2, qrs_pos1, qrs_pos2 = get_hr_onechannel(ecg_arr[:,i], fs)
        all_err_code1.append(err_code1)
        all_err_code2.append(err_code2)
        all_hr1.append(hr1)
        all_hr2.append(hr2)
        all_qrs_pos1.append(qrs_pos1)
        all_qrs_pos2.append(qrs_pos2)
    
    return all_err_code1, all_err_code2, all_hr1, all_hr2, all_qrs_pos1, all_qrs_pos2
def get_hr_onechannel(ecg, fs):
    try:
        if ecg is None or len(ecg) < 10:
            raise ValueError("ECG 信号无效或长度过短")
        if fs <= 0:
            raise ValueError("采样率必须为正数")

        err_code2, qrs_pos2, sign = qrs_detect2_err_ret_sign(ecg, 0.3, 0.25, fs)

        if sign > 0 and err_code2 == 0:
            err_code1, qrs_pos1 = two_average_detector_err_ret(ecg, fs, True)
        elif sign < 0 and err_code2 == 0:
            err_code1, qrs_pos1 = two_average_detector_err_ret(-ecg, fs, True)
        else:
            err_code1 = 1
            qrs_pos1 = []

        hr1 = calculate_hr(qrs_pos1, fs)
        hr2 = calculate_hr(qrs_pos2, fs)
        if hr1 == -1 or hr2 == -1:
            raise ValueError('[HR计算错误]')

        qrs_pos1 = np.asarray(qrs_pos1, dtype=float)
        qrs_pos2 = np.asarray(qrs_pos2, dtype=float)

        return err_code1, err_code2, hr1, hr2, qrs_pos1, qrs_pos2

    except Exception as e:
        # 输出错误信息到 .NET 控制台或日志系统
        print(f"[get_hr ERROR] {str(e)}")
        # 返回错误码，其他值设为默认
        return 1, 1, 0, 0, np.array([]), np.array([])

def calculate_hr(qrs_pos, fs):
    try:
        if not isinstance(qrs_pos, (list, np.ndarray)) or len(qrs_pos) <= 1:
            return 0
        duration = (qrs_pos[-1] - qrs_pos[0]) / fs
        if duration <= 0:
            return 0
        hr = (len(qrs_pos) - 1) / (duration / 60)
        return hr
    except Exception as e:
        print(f"[HR计算错误] {e}")
        return -1

if __name__ == "__main__":
    import scipy.io as sio
    import pandas as pd
    # data = sio.loadmat('D:\work_file\算法\SCORE\Score_Cyhon_Demo\python\dataset\ECG_20230716\ECG_segment_20230716\seg_44.mat')
    data = pd.read_csv(r"C:\Users\rewalk\Documents\WXWork\1688857197660272\Cache\File\2025-04\WriteData(1)\20250428101610070.csv",header=None)
    data = data.values
    number_array = [float(x) for x in data[0]]
    number_array = np.asarray(number_array)
    print(get_hr(-1*number_array, 500))
    # for i in range(8):
    #     ecg = data['ECG'][i]
    #     print(get_hr(ecg, 500))
