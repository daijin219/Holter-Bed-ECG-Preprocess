import os
os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"       # MKL

# -------------------------------------------------------
# 1. 常规 import
# -------------------------------------------------------
import math
import pathlib
import tempfile
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List
from tqdm import tqdm
from .lib_get_score.lib.qrs_detect2_err_ret_sign import qrs_detect2_err_ret_sign
from .lib_get_score.lib.two_average_detector_err_ret import two_average_detector_err_ret
from .lib_get_score.get_score import get_score
from .signal_pro.select_path import select_path
from .signal_pro.read_dat import read_dat_file  
from .signal_pro.read_edf import read_edf_file
from .signal_pro.apply_filters import apply_filters
from .signal_pro.find_files_with_suffix import find_files_with_suffix




# -------------------------------------------------------
# 2. 你的算法依赖（示例用占位函数代替）
# -------------------------------------------------------
def get_qrs(data):
    ecg = data.ravel()
    ecg = ecg/np.max(np.abs(ecg))
    fs = 500
    err_code2, qrs_pos2, sign = qrs_detect2_err_ret_sign(ecg, 0.3, 0.25, fs)
    
    if sign > 0 and err_code2 == 0:
        err_code1, qrs_pos1 = two_average_detector_err_ret(ecg, fs, True)
    elif sign < 0 and err_code2 == 0:
        err_code1, qrs_pos1 = two_average_detector_err_ret(-ecg, fs, True)
    else:
        err_code = 1
        err_code1 = 1
    # 计算 err_code
    err_code =  int(err_code1 or err_code2)
    # 根据 err_code1 和 err_code2 的值设置 hr_code 和 qrs_pos
    if err_code1 == 0 and err_code2 == 0:
        # qrs_pos = merge_r_peaks(qrs_pos1, qrs_pos2,30)  # 合并数组
        qrs_pos = qrs_pos2
    elif err_code1 == 0:
        qrs_pos = qrs_pos1
    elif err_code2 == 0:
        qrs_pos = qrs_pos2
    else:
        err_code = -105
        qrs_pos = []
    return err_code, np.asarray(qrs_pos)
def merge_r_peaks(r1, r2, tolerance=30):
# 合并所有点
    all_r = np.sort(np.concatenate((r1, r2)))
    
    merged = []
    i = 0
    while i < len(all_r):
        group = [all_r[i]]
        j = i + 1
        # 合并距离在容差内的后续点
        while j < len(all_r) and all_r[j] - all_r[i] <= tolerance:
            group.append(all_r[j])
            j += 1
        # 平均这些点作为融合点
        merged.append(int(np.mean(group)))
        i = j
    return np.array(merged)

# -------------------------------------------------------
# 3. 全局常量 & memmap 生成工具
# -------------------------------------------------------
TMP_DIR = pathlib.Path(tempfile.gettempdir())
TMP_DIR.mkdir(exist_ok=True)

MEMMAP_PATH = TMP_DIR / "holter_data_memmap.pkl"

def _prepare_memmap(raw_path: str, index_path: str) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    加载原始数据 + 索引，生成只读 memmap 形式，返回两个 pkl 路径。
    """
    if raw_path.endswith(".dat"):  # 索引文件与原始文件相同
        r_data = read_dat_file(raw_path)
        fs_data = 500
    elif raw_path.endswith(".edf"):  # 索引文件与原始文件不同
        r_data,fs_data = read_edf_file(raw_path,1)
    elif raw_path.endswith(".npz"):  # 索引文件与原始文件不同
        r_data = np.load(raw_path)['arr_0'][:,2:10]
        fs_data = 500
    else:
        print("原始数据文件格式错误")
    cut_indices = joblib.load(index_path)

    mmap_raw_path = TMP_DIR / f"{pathlib.Path(raw_path).stem}_memmap.pkl"
    mmap_index_path = TMP_DIR / f"{pathlib.Path(index_path).stem}_memmap.pkl"
    r_data = apply_filters(r_data.T, fs_data).T
    joblib.dump(r_data, mmap_raw_path, compress=False)
    joblib.dump(cut_indices, mmap_index_path, compress=False)

    del r_data, cut_indices
    return mmap_raw_path, mmap_index_path



# -------------------------------------------------------
# 4. 子进程初始化 & 工作函数
# -------------------------------------------------------

def _worker_init(mmap_path_data: str,mmap_path_index: str):
    """ProcessPool 的 initializer——每个子进程只运行一次。"""
    global RAW_DATA,DATA_LIST
    RAW_DATA = joblib.load(mmap_path_data, mmap_mode="r")  # 共享只读 memmap
    DATA_LIST = joblib.load(mmap_path_index, mmap_mode="r")

def _score_batch(args) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """批量处理一段窗口。

    Parameters
    ----------
    args : Tuple
        (start_idx, end_idx, winlen, steplen, fs, thr)
    """
    start_idx, end_idx, fs, thr = args

    rt_scores, rt_channels, rt_ch_scores = [], [], []
    for i in range(start_idx, end_idx):
        seg = RAW_DATA[DATA_LIST[i]:DATA_LIST[i+1],:]
        _, temp_score = get_score(seg.T, 1.5, 1, fs)#加入error code及错误处理，确定error输出

        best_score = float(temp_score.max())
        if best_score > thr:
            best_ch = int(temp_score.argmax())
        else:
            best_ch = -1
            best_score = -1.0

        rt_scores.append(temp_score.astype(np.float32))
        rt_channels.append(best_ch)
        rt_ch_scores.append(best_score)
    return rt_scores, rt_channels, rt_ch_scores

def _rpeaks_batch(args) -> List[List[np.ndarray]]:
    start_idx, end_idx,batchsize, fs = args
    rt_rpeaks_batch = []
    ch_num = RAW_DATA.shape[1]
    for seg_idx in range(start_idx, end_idx):
        rpeaks_ch_all = []

        for ch in range(ch_num):
            # 构造带上下文的窗口
            curr = RAW_DATA[DATA_LIST[seg_idx]:DATA_LIST[seg_idx+1], ch]
            prev = RAW_DATA[DATA_LIST[seg_idx-1]:DATA_LIST[seg_idx], ch] if seg_idx > 1 else np.zeros_like(curr)
            nxt  = RAW_DATA[DATA_LIST[seg_idx+1]:DATA_LIST[seg_idx+2], ch] if seg_idx < len(DATA_LIST) - 2 else np.zeros_like(curr)
            
            context_len = 100
            data_test = np.concatenate([
                prev[-context_len:], curr, nxt[:context_len]
            ])

            err_code, qrs = get_qrs(data_test)

            if len(qrs) > 0:
                # 保留属于当前窗口的 QRS
                qrs_in_curr = qrs[
                    (qrs >= context_len) &
                    (qrs < context_len + len(curr))
                ] - context_len
            else:
                qrs_in_curr = []

            rpeaks_ch_all.append(qrs_in_curr)
        
        rt_rpeaks_batch.append(rpeaks_ch_all)

    return rt_rpeaks_batch
# -------------------------------------------------------
# 5. 主类
# -------------------------------------------------------
class Exp_ana_0625:
    def __init__(self, path_data, path_index, fs: int = 500):
        self.fs = fs
        # ------ 1⃣️ 预处理：确保有 memmap 文件 ------
        print("1⃣️ 预处理：确保有 memmap 文件")
        mmap_raw_path, mmap_index_path = _prepare_memmap(path_data,path_index)
        # ------ 2⃣️ 只读方式加载数据（不占内存） ------
        print("2⃣️ 只读方式加载数据（不占内存）")
        # with open(mmap_path, "rb") as f:
        #     data = pickle.load(f)
        temp_data = joblib.load(mmap_index_path, mmap_mode="r")
        print("OKK")
        self.win_num = len(temp_data)
        # ------ 3⃣️ 并行计算分数 ------
        print("3⃣️ 并行计算分数")
        self.get_rt_rpeaks_batch(mmap_raw_path,mmap_index_path)
        self.get_rt_score(mmap_raw_path,mmap_index_path)
        

    # ---------------------------------------------------
    # 核心：并行窗口评分
    # ---------------------------------------------------
    def get_rt_score(self, mmap_path_data: pathlib.Path, mmap_path_index: pathlib.Path,batchsize: int = 500, thr: float = 3.0):
        n_seg = self.win_num-1
        batches = math.ceil(n_seg / batchsize)

        # 结果占位
        rt_scores_total = []
        rt_channels_total = []
        rt_ch_scores_total = []

        # --------- ⚡ 并行执行 ---------
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_worker_init,
            initargs=(str(mmap_path_data),str(mmap_path_index)),
        ) as pool:
            tasks = (
                (
                    b * batchsize,
                    min((b + 1) * batchsize, n_seg),
                    self.fs,
                    thr,
                )
                for b in range(batches)
            )
            for batch_ret in tqdm(pool.map(_score_batch, tasks), total=batches, desc="rt_score"):
                rs, rc, rcs = batch_ret
                rt_scores_total.extend(rs)
                rt_channels_total.extend(rc)
                rt_ch_scores_total.extend(rcs)

        # --------- 汇总为 numpy ---------
        self.rt_score = np.stack(rt_scores_total)  # shape = (n_seg, ch_num)
        self.rt_channel = np.array(rt_channels_total, dtype=np.int16)  # (n_seg,)
        self.rt_channel_score = np.array(rt_ch_scores_total, dtype=np.float32)  # (n_seg,)
        print("[Exp] get_rt_score success →", self.rt_score.shape)

    def get_rt_rpeaks_batch(self,mmap_path_data: pathlib.Path, mmap_path_index: pathlib.Path, batchsize: int = 500):
        n_seg = self.win_num-1
        batches = math.ceil(n_seg / batchsize)

        # 结果占位
        rt_rpeaks_total = []

        # --------- ⚡ 并行执行 ---------
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_worker_init,
            initargs=(str(mmap_path_data),str(mmap_path_index)),
        ) as pool:
            tasks = (
                (
                    b * batchsize,
                    min((b + 1) * batchsize, n_seg),
                    batchsize,
                    self.fs
                )
                for b in range(batches)
            )
            for batch_ret in tqdm(pool.map(_rpeaks_batch, tasks), total=batches, desc="rt_rpeaks"):
                rt_rpeaks_total.extend(batch_ret)
        self.rt_rpeaks = rt_rpeaks_total

        # --------- 汇总为 numpy ---------
        # self.rt_rpeaks = np.stack(rt_rpeaks_total)  # shape = (n_seg, ch_num)
        print("[Exp] get_rt_score success")

def data_analysis(path_input):
    path = path_input
    if os.path.exists(os.path.join(path,"bed_cut_list.pkl")) and os.path.exists(os.path.join(path,"data.npz")):
        path_data = os.path.join(path,"data.npz")
        path_index = os.path.join(path,"bed_cut_list.pkl")
        print("[Exp]load bed data")
        exp_obj_bed = Exp_ana_0625(path_data,path_index, fs=500)
        with open(path + "\\" + "bed.pickle", "wb") as file:
            pickle.dump(exp_obj_bed, file)
        
    else:
        print("bed data not exist") 
        
    if os.path.exists(os.path.join(path,"holter_cut_list.pkl")):
        holter_files = find_files_with_suffix(path, '.dat')
        if not holter_files:
            holter_files = find_files_with_suffix(path, '.edf')
            holter_file_path = os.path.join(path,holter_files[0])
            edf_data_temp,fs_holter = read_edf_file(holter_file_path,read_fs=1)
        else:
            fs_holter = 500
        if len(holter_files) == 1:
            print("[Exp]load holter data")
            path_data = os.path.join(path,holter_files[0])
            print("holter fs:",fs_holter)
            path_index = os.path.join(path,"holter_cut_list.pkl")
            exp_obj_holter = Exp_ana_0625(path_data,path_index, fs=fs_holter)
            with open(path + "\\" + "holter.pickle", "wb") as file:
                pickle.dump(exp_obj_holter, file)
        else:
            print("holter data error")
    else:
        print("holter data not exist")
# -------------------------------------------------------
# 6. CLI 演示
# -------------------------------------------------------
if __name__ == "__main__":
    
    # path = r"D:\work_file\阳光\20250605-B\export"
    data_analysis()
    print("处理完成")

            