# save all the segment and downsample to 100Hz
import argparse
import numpy as np
from scipy import signal
from scipy.signal import filtfilt
import pandas as pd
import os
import mne
import sys
import pickle
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import neurokit2 as nk
from neurokit2.ecg.ecg_quality import (
    _ecg_quality_kSQI,
    _ecg_quality_pSQI,
    _ecg_quality_basSQI,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from Zhu_code import pipeline3_data_analysis_speedup
sys.modules['pipeline3_data_analysis_speedup'] = pipeline3_data_analysis_speedup
# from .fir_kaiser_bandpass import b_filter

if not sys.stdout.isatty():
    tqdm.disable = True
    
def ecg_quality_scores(ecg_cleaned, rpeaks,fs):
    quality1 = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks,
                sampling_rate=fs,
                method="zhao2018",
                approach="fuzzy")
    quality2 = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks,
                sampling_rate=fs,
                method="zhao2018")
    # NeuroKit2 internal API changed across versions: some releases accept
    # method="fisher", newer ones do not.
    try:
        kSQI = _ecg_quality_kSQI(ecg_cleaned)  # newer signature
    except TypeError:
        kSQI = _ecg_quality_kSQI(ecg_cleaned, method="fisher")  # older signature
    pSQI = _ecg_quality_pSQI(
            ecg_cleaned, sampling_rate=fs, window=1024
        )
    basSQI = _ecg_quality_basSQI(
            ecg_cleaned, sampling_rate=fs, window=1024
        )
    # if len(rpeaks) <=3:
    #     quality3 =-1*np.ones_like(ecg_cleaned)
    # else:
    #     quality3 = nk.ecg_quality(
    #                 ecg_cleaned, rpeaks=rpeaks, method="averageQRS", sampling_rate=fs)
    return quality1, quality2 , kSQI, pSQI, basSQI #, quality3

def cut_signal(signal, segindex, seglength):
    signal_list = []
    for i in tqdm(range(len(segindex)-1), total=len(segindex), desc="Cutting signal", ncols=75):
        cutthis = signal[int(segindex[i]):int(segindex[i]+seglength)]
        if cutthis.shape[0] < seglength:
            cutthis = np.append(cutthis, np.zeros([seglength-cutthis.shape[0], cutthis.shape[1]]), axis=0)
        signal_list.append(cutthis)
    return np.stack(signal_list, axis=0)

def cut_rpeak(rpeak, segindex, seglength):
    rpeak_list = -1*np.ones([len(segindex), 50, len(rpeak)])
    for i in range(len(segindex)-1):
        rpeak_this = []
        for ch_idx in range(len(rpeak)):
            tmp = rpeak[ch_idx][(rpeak[ch_idx]>=segindex[i]) & (rpeak[ch_idx]<segindex[i]+seglength)]
            rpeak_list[i, :len(tmp), ch_idx] = tmp - segindex[i]
        #     rpeak_this.append(tmp - segindex[i])
        # rpeak_list.append(rpeak_this)
    return rpeak_list

def clean_rpeaks(peaks, sampling_rate, min_dist_ms=150):
    """移除距离过近的 R 波 (生理不应期)"""
    if len(peaks) < 2:
        return peaks
    
    # 将毫秒转换为样本数
    min_dist_samples = int(min_dist_ms / 1000 * sampling_rate)
    
    # 简单去重
    peaks = np.unique(peaks)
    
    # 移除过近的波峰 (简单的迭代过滤)
    cleaned = [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - cleaned[-1] > min_dist_samples:
            cleaned.append(peaks[i])
            
    return np.array(cleaned)

def ecg_clean_all_channels(ecg_raw, fs):
    """
    ecg_raw: np.ndarray, shape (n_samples, n_channels)
    fs: sampling rate (Hz)
    """
    ecg_raw = np.asarray(ecg_raw)

    # Ensure 2D: (n_channels, n_samples)
    if ecg_raw.ndim == 1:
        ecg_raw = ecg_raw[None, :]   # add channel axis

    n_samples, n_channels = ecg_raw.shape
    ecg_cleaned = np.zeros_like(ecg_raw, dtype=float)
    rpeaks_all = []

    for ch in range(n_channels):
        sig = ecg_raw[:, ch]

        # 1) Clean this channel
        cleaned = nk.ecg_clean(sig, sampling_rate=fs)
        # cleaned = filtfilt(b_filter, 1, sig)
        # 2) R-peak detection on cleaned ECG
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs) #, correct_artifacts=True)
        rpeaks_idx = info["ECG_R_Peaks"]  # 1D array of indices
        rpeaks_idx = clean_rpeaks(rpeaks_idx, fs)
        ecg_cleaned[:, ch] = cleaned
        rpeaks_all.append(rpeaks_idx)

    return ecg_cleaned, rpeaks_all  # rpeaks_all is a list, length = n_channels

def nkqualitystr_num(quality:str):
    if quality in ["Unnacceptable", "Unacceptable"]:
        return 0
    elif quality == "Barely acceptable":
        return 1
    elif quality == "Excellent":
        return 2
    else:
        raise ValueError(f"Unknown quality type {quality}")

def select_limited_indices(label: np.ndarray,
                           max_class0: int = 500,
                           max_class3: int = 500) -> np.ndarray:
    """
    label: (n, 4) 的 one-hot numpy 数组
    返回：满足要求的样本索引（int numpy array）
    """
    # 各类的布尔 mask（假设严格 one-hot）
    mask0 = (label[:, 0] == 1) & (label[:, 1] == 0) & (label[:, 2] == 0)
    mask1 = (label[:, 1] == 1)
    mask2 = (label[:, 2] == 1)
    mask3 = (label[:, 3] == 1) & (label[:, 1] == 0) & (label[:, 2] == 0)

    # 找到每一类的索引
    idx0 = np.where(mask0)[0]
    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]
    idx3 = np.where(mask3)[0]

    # 对 class 0 和 class 3 做数量限制：最多 500 个
    if len(idx0) > max_class0:
        sep = int(np.ceil(len(idx0) / max_class0))
        idx0 = idx0[::sep]      # 如果想随机抽，可以改成 np.random.choice(idx0, max_class0, replace=False)
    if len(idx3) > max_class3:
        sep = int(np.ceil(len(idx3) / max_class3))
        idx3 = idx3[::sep]      # 同上可以改为随机

    # class 1 和 class 2 的全部保留
    selected_idx = np.concatenate([idx0, idx1, idx2, idx3])
    selected_idx = np.unique(selected_idx)
    selected_idx = np.sort(selected_idx).astype(int)

    return selected_idx

def compute_one(i, ecg_cleaned, rpeaks_all):
    nkquality_this = np.zeros([ecg_cleaned.shape[2], 5])
    # quality3_this = []
    for ch_idx in range(ecg_cleaned.shape[2]):
        r_this = rpeaks_all[i, :, ch_idx]
        r_this = r_this[r_this!=-1]
        if len(r_this) < 5:
            nkquality_this[ch_idx] = [
                -1,-1,-1,-1,-1
                # quality3.min()
            ]
            continue
        quality1, quality2, kSQI, pSQI, basSQI= ecg_quality_scores(
            ecg_cleaned[i][:, ch_idx],
            r_this,
            fs=100
        )
        nkquality_this[ch_idx] = [
            nkqualitystr_num(quality1),
            nkqualitystr_num(quality2),
            kSQI, pSQI, basSQI,
            # quality3.min()
        ]
        # quality3_this.append(quality3)
    return nkquality_this # quality3_this, 


def save_data_for_index(i, savepath, data_dict):
    """
    保存单个索引 i 对应的所有数据文件。
    
    参数:
        i (int): 数据的索引。
        savepath (str): 基础存储路径。
        data_dict (dict): 包含所有待保存数据的字典。
    """
    try:
        # 使用 f"{i:06d}" 格式化文件名
        filename_base = f"{i:06d}"
        
        # 1. np.save 文件
        np.save(os.path.join(savepath, 'ecg', f"ecg{filename_base}.npy"), data_dict['ecg_cleaned'][i])
        np.save(os.path.join(savepath, 'label', f"label{filename_base}.npy"), data_dict['label_list'][i])
        np.save(os.path.join(savepath, 'score', f"score{filename_base}.npy"), data_dict['bed_score'][i])
        np.save(os.path.join(savepath, 'nkqualitysix', f"nkqualitysix{filename_base}.npy"), data_dict['nkquality_alllist'][i])
        
        # 2. np.savez (使用 *args) 文件
        np.savez(os.path.join(savepath, 'rpeak', f"rpeak{filename_base}.npz"), *data_dict['rpeaks_all'][i])
        np.savez(os.path.join(savepath, 'nkqualityavgQRS', f"nkqualityavgQRS{filename_base}.npz"), *data_dict['quality3_alllist'][i])
        
        # 3. np.savez (使用 kwargs) 文件
        np.savez(
            os.path.join(savepath, 'labelorg', f"labelorg{filename_base}.npz"), 
            labellist=np.array(data_dict['labelorg_list'][i]), 
            beatidxlist=np.array(data_dict['beatidx_list'][i])
        )
        
        return i, "SUCCESS"
    except Exception as e:
        # 返回失败信息，以便在主线程中跟踪错误
        return i, f"ERROR: {e}"

def filter_cut_bedholter(ID, root_dir='/mnt/sdb1/Data/',beat_dir='/hdd/daijing/ECGbed_data/ECGbeats_df/', save_dir = '/hdd/daijing/ECGbed_data/ECG_cut/segment/'):
    print(ID, "Start")
    base_path = os.path.join(root_dir, ID.upper(), 'export')
    if os.path.exists(os.path.join(base_path, "bed_cut_list.pkl")) == False:
        raise FileNotFoundError(os.path.join(base_path, "bed_cut_list.pkl") + " not found")
    if os.path.exists(os.path.join(base_path, "holter_cut_list.pkl")) == False:
        raise FileNotFoundError(os.path.join(base_path, "holter_cut_list.pkl") + " not found")
    bed_index = joblib.load(os.path.join(base_path, "bed_cut_list.pkl"))
    bed_index = np.array(bed_index)
    holter_index = joblib.load(os.path.join(base_path, "holter_cut_list.pkl"))
    holter_index = np.array(holter_index)

    with open(os.path.join(base_path, "holter.pickle"), "rb") as file:
        Zhu = pickle.load(file)
    holter_fs = Zhu.fs
    holter_score = Zhu.rt_score
    holter_index = (holter_index * 100 / holter_fs).astype(int) # convert to 100Hz
    with open(os.path.join(base_path, "bed.pickle"), "rb") as file:
        Zhu = pickle.load(file)
    bed_fs = Zhu.fs
    bed_index =  (bed_index * 100 / bed_fs).astype(int) # convert to 100 Hz
    bed_score = Zhu.rt_score
    assert holter_index.shape[0] == bed_index.shape[0], f"holter{holter_index.shape} != bed{bed_index.shape}"
    beat_df = pd.read_csv(os.path.join(beat_dir, ID.lower()+'.csv'))
    beat_df['Beat Num'] = (beat_df['Beat Num']/10).astype(int) # downsample to 100Hz
    # beat_df = beat_df[(beat_df['Beat Num'] > holter_index[0]) & (beat_df['Beat Num'] < holter_index[1])]
    other_label = beat_df['Label'].unique()
    other_label = [i for i in other_label if i not in [1,5,8]]
    label_list = np.zeros([len(holter_index)-1, 4])
    labelorg_list = []
    beatidx_list = []
    if bed_fs == 250 or holter_fs == 250:
        raise ValueError(f"Sampling rate error: bed_fs={bed_fs}, holter_fs={holter_fs}.")
    print("Preload Done")
    for i in range(len(holter_index)-1):
        beat_df_this = beat_df[(beat_df['Beat Num'] > holter_index[i]) & (beat_df['Beat Num'] < holter_index[i+1])]
        if len(beat_df_this) ==0:
            label_list[i, 0] = -1
            labelorg_list.append(-1*np.ones([50]))
            beatidx_list.append(-1*np.ones([50]))
            continue
        beat_df_unique = beat_df_this['Label'].unique()
        del beat_df_this
        if 5 in beat_df_unique:
            label_list[i, 1] = 1
        if 8 in beat_df_unique:
            label_list[i, 2] = 1
        if len(np.intersect1d(other_label, beat_df_unique)) > 0:
            label_list[i, 3] = 1
        if len(beat_df_unique) == 1 and beat_df_unique == 1:
            label_list[i, 0] = 1
        beat_df_this = beat_df[(beat_df['Beat Num'] > holter_index[i]) & (beat_df['Beat Num'] < holter_index[i+1])]
        tmp = beat_df_this[beat_df_this['Label'].isin([1, 5, 8])]
        labelorg_this = tmp['Label'].tolist()
        beatorg_this = tmp['Beat Num'] - holter_index[i]
        # append -1 at the end of the list to make it the same length as labelorg
        labelorg_this = np.append(np.array(labelorg_this), -1*np.ones([50 - len(labelorg_this),])).reshape([50]).astype(int)
        beatorg_this = np.append(np.array(beatorg_this), -1*np.ones([50 - len(beatorg_this),])).reshape([50]).astype(int)
        labelorg_list.append(labelorg_this)
        beatidx_list.append(beatorg_this)
    labelorg_list = np.stack(labelorg_list, axis=0)
    beatidx_list = np.stack(beatidx_list, axis=0)
    print(f"Total{label_list.shape[0]}, 0={np.sum(label_list[:, 0])}, 1={np.sum(label_list[:, 1])}, 2={np.sum(label_list[:, 2])}, 3={np.sum(label_list[:, 3])}")
    # selected_idx = select_limited_indices(label_list)
    # selected_idx = np.arange(0, len(label_list), dtype=int) # select all segment
    # label_list = label_list[selected_idx]
    # labelorg_list = [labelorg_list[i] for i in selected_idx]
    # beatidx_list = [beatidx_list[i] for i in selected_idx]
    print("Label check Done")
    ######################### bed segment cut #########################
    if os.path.exists(os.path.join(base_path, 'data.npz')) == False:
        raise FileNotFoundError(os.path.join(base_path, 'data.npz') + " not found")
    ecgbed = np.load(os.path.join(base_path, 'data.npz'))['arr_0']
    ecgbed = ecgbed[:, 2:10]
    print("Load bed done")
    ecg_cleaned, rpeaks_all = ecg_clean_all_channels(ecgbed, bed_fs)
    del ecgbed
    # downsample
    ds_step_bed = int(bed_fs / 100)
    ecg_cleaned = ecg_cleaned[::ds_step_bed, :]
    rpeaks_all_ds = [(rpeaks_all[i] / ds_step_bed).astype(int) for i in range(len(rpeaks_all))]
    rpeaks_all = []
    for ch_idx in range(len(rpeaks_all_ds)):
        rpeaks_all.append(clean_rpeaks(rpeaks_all_ds[ch_idx], 100))
    print("Filter bed done")
    ecg_cleaned = cut_signal(ecg_cleaned, bed_index, 100*10)
    # ecg_cleaned = ecg_cleaned[selected_idx]
    rpeaks_all = cut_rpeak(rpeaks_all, bed_index, 100*10)
    
    # rpeaks_all = [rpeaks_all[i] for i in selected_idx]
    print("Cut bed done")
    
    # quality3_alllist = []
    nkquality_alllist = []

    worker = partial(compute_one, ecg_cleaned=ecg_cleaned, rpeaks_all=rpeaks_all)

    with ProcessPoolExecutor() as executor:
        results = executor.map(worker, range(ecg_cleaned.shape[0]))
        for nkquality_this in tqdm(
            results,
            total=ecg_cleaned.shape[0],
            ncols=75,
            desc="Bed quality"
        ):
            # quality3_alllist.append(quality3_this)
            nkquality_alllist.append(nkquality_this)
    nkquality_alllist = np.stack(nkquality_alllist, axis=0)
    mean_this = np.mean(ecg_cleaned, axis=(0, 1), keepdims= True)
    std_this = np.std(ecg_cleaned, axis=(0, 1), keepdims=True)
    ecg_cleaned = (ecg_cleaned - mean_this) / (std_this)
    print("Quality bed Done")
    savepath = os.path.join(save_dir,'segment_10_whole')
    os.makedirs(os.path.join(savepath, 'nkqualityfive'), exist_ok=True)
    # os.makedirs(os.path.join(savepath, 'nkqualityavgQRS'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'ecg'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'label'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'labelorg'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'beatorg'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'score'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'rpeak'), exist_ok=True)
    np.save(os.path.join(savepath, 'ecg', f'ecg_{ID.upper()}.npy'), ecg_cleaned)
    np.save(os.path.join(savepath, 'nkqualityfive', f"nkqualityfive_{ID.upper()}.npy"), nkquality_alllist)
    np.save(os.path.join(savepath, 'label', f"label_{ID.upper()}.npy"), label_list)
    np.save(os.path.join(savepath, 'labelorg', f"labelorg_{ID.upper()}.npy"), labelorg_list)
    np.save(os.path.join(savepath, 'beatorg', f"beatorg_{ID.upper()}.npy"), beatidx_list)
    np.save(os.path.join(savepath, 'score', f"score_{ID.upper()}.npy"), bed_score)
    np.save(os.path.join(savepath, 'rpeak', f"rpeak_{ID.upper()}.npy"), rpeaks_all)
    print("Save Done")
    # for i in range(ecg_cleaned.shape[0]):
    #     np.save(os.path.join(savepath, 'ecg', f"ecg{i:06d}.npy"), ecg_cleaned[i])
    #     np.save(os.path.join(savepath, 'label', f"label{i:06d}.npy"), label_list[i])
    #     np.save(os.path.join(savepath, 'score', f"score{i:06d}.npy"), bed_score[i])
    #     np.savez(os.path.join(savepath, 'rpeak', f"rpeak{i:06d}.npz"), *rpeaks_all[i])
    #     np.save(os.path.join(savepath, 'nkqualitysix',   f"nkqualitysix{i:06d}.npy"), nkquality_alllist[i])
    #     np.savez(os.path.join(savepath, 'nkqualityavgQRS', f"nkqualityavgQRS{i:06d}.npz"), *quality3_alllist[i])
    #     np.savez(os.path.join(savepath, 'labelorg', f"labelorg{i:06d}.npz"), labellist=np.array(labelorg_list[i]), beatidxlist=np.array(beatidx_list[i]))
    # del ecg_cleaned

    ######################### holter segment cut ######################### 
    files = os.listdir(base_path)
    ecgfilename = None
    Findedf = False
    for f in files:
        if f.endswith('edf'):
            ecgfilename = f
            Findedf = True
            break
    if not Findedf:
        raise FileNotFoundError(f"No EDF file found in {base_path}")
    ecgholter = mne.io.read_raw_edf(os.path.join(base_path,ecgfilename), preload=False, verbose=False)
    if len(ecgholter.ch_names) == 4:
        ecgholter = ecgholter.get_data()[0:-1,:].T
    else:
        ecgholter = ecgholter.get_data(['I', 'II', 'III']).T
    print("Load Holter Done")
    ecg_cleaned, rpeaks_all = ecg_clean_all_channels(ecgholter, holter_fs)
    print("Filter Holter Done")
    del ecgholter
    # downsample to 100Hz
    ds_step_holter = int(holter_fs / 100)
    ecg_cleaned = ecg_cleaned[::ds_step_holter, :]
    rpeaks_all_ds = [(rpeaks_all[i] / ds_step_holter).astype(int) for i in range(len(rpeaks_all))]
    rpeaks_all = []
    for ch_idx in range(len(rpeaks_all_ds)):
        rpeaks_all.append(clean_rpeaks(rpeaks_all_ds[ch_idx], 100))
    ecg_cleaned = cut_signal(ecg_cleaned, holter_index, 100*10)
    # ecg_cleaned = ecg_cleaned[selected_idx]
    rpeaks_all = cut_rpeak(rpeaks_all, holter_index, 100*10)
    print("Cut Holter Done")
    # rpeaks_all = [rpeaks_all[i] for i in selected_idx]
    nkquality_alllist = []
    worker = partial(compute_one, ecg_cleaned=ecg_cleaned, rpeaks_all=rpeaks_all)
    with ProcessPoolExecutor() as executor:
        results = executor.map(worker, range(ecg_cleaned.shape[0]))
        for nkquality_this in tqdm(
                results,
                total=ecg_cleaned.shape[0],
                ncols=75,
                desc="Holter quality"):
            nkquality_alllist.append(nkquality_this)
    nkquality_alllist = np.stack(nkquality_alllist, axis=0)
    mean_this = np.mean(ecg_cleaned, axis=(0, 1), keepdims= True)
    std_this = np.std(ecg_cleaned, axis=(0, 1), keepdims=True)
    ecg_cleaned = (ecg_cleaned - mean_this) / (std_this)
    print("Quality holter Done")
    savepath = os.path.join(save_dir,'Hsegment_10_whole')
    os.makedirs(os.path.join(savepath, 'nkqualityfive'), exist_ok=True)
    # os.makedirs(os.path.join(savepath, 'nkqualityavgQRS'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'ecg'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'label'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'labelorg'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'beatorg'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'score'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'rpeak'), exist_ok=True)
    np.save(os.path.join(savepath, 'ecg', f'ecg_{ID.upper()}.npy'), ecg_cleaned)
    np.save(os.path.join(savepath, 'nkqualityfive', f"nkqualityfive_{ID.upper()}.npy"), nkquality_alllist)
    np.save(os.path.join(savepath, 'label', f"label_{ID.upper()}.npy"), label_list)
    np.save(os.path.join(savepath, 'labelorg', f"labelorg_{ID.upper()}.npy"), labelorg_list)
    np.save(os.path.join(savepath, 'beatorg', f"beatorg_{ID.upper()}.npy"), beatidx_list)
    np.save(os.path.join(savepath, 'score', f"score_{ID.upper()}.npy"), holter_score)
    np.save(os.path.join(savepath, 'rpeak', f"rpeak_{ID.upper()}.npy"), rpeaks_all)
    print("Save holter Done")
    # for i in range(ecg_cleaned.shape[0]):
    #     np.save(os.path.join(savepath, 'ecg', f"ecg{i:06d}.npy"), ecg_cleaned[i])
    #     np.save(os.path.join(savepath, 'label', f"label{i:06d}.npy"), label_list[i])
    #     np.save(os.path.join(savepath, 'score', f"score{i:06d}.npy"), holter_score[i])
    #     np.savez(os.path.join(savepath, 'rpeak', f"rpeak{i:06d}.npz"), *rpeaks_all[i])
    #     np.save(os.path.join(savepath, 'nkqualitysix',   f"nkqualitysix{i:06d}.npy"), nkquality_alllist[i])
    #     np.savez(os.path.join(savepath, 'nkqualityavgQRS', f"nkqualityavgQRS{i:06d}.npz"), *quality3_alllist[i])
    #     np.savez(os.path.join(savepath, 'labelorg', f"labelorg{i:06d}.npz"), labellist=np.array(labelorg_list[i]), beatidxlist=np.array(beatidx_list[i]))
    # np.save(os.path.join(savepath, "selected_idx.npy"), selected_idx)
        
        
if __name__ == "__main__":
    # with open("/hdd/daijing/ECGbed_data/ECG_cut/f500.txt", "r") as f:
    #     IDs = [line.strip() for line in f]
    # IDs.sort()
    # for ID_idx, ID in tqdm(enumerate(IDs)):
    #     if ID =='20251028-1':
    #         continue
    #     save_dir = f'/hdd/daijing/ECGbed_data/ECG_cut/'
    #     filter_cut_bedholter(ID, save_dir=save_dir)
    #     print(f'{ID_idx}, Processing ID: {ID}')
    # parser = argparse.ArgumentParser(description='Preprocess')
    # parser.add_argument('--index', type=int, required=True)
    # args = parser.parse_args()
    # ID_idx = int(args.index)
    
    # # IDs = os.listdir('/hdd/daijing/ECGbed_data/ECGbeats_df/')
    # # IDs.sort()
    # # IDs = [ID.split('.')[0] for ID in IDs if 'all' not in ID]
    # IDs = pd.read_csv("/hdd/daijing/ECGbed_data/RunningLog/ErrorIDs_p1.csv", index_col=0)
    # IDs = IDs['ErrorIDs'].to_list()
    # ID = IDs[ID_idx]
    ID_list = [
        '20250606-B',
        '20250607-B',
        '20250611-A',
        '20250721-A',
        '20250723-A',
        '20250802',
        '20250806-A',
        '20250809-A',
        '20250810-A',
        '20250812-A',
        '20250816-7',
        '20250822-7',
        '20250824-7',
        '20250828-7',
        '20250906-4',
        '20250906-6',
        '20250908-4',
        '20250912-6',
        '20250913-1',
        '20250915-7',
        '20250919-2',
        '20250921-6',
        '20250926-4',
        '20250927-2',
        '20251009-1',
        '20251010-4',
        '20251011-2',
        '20251011-4',   
        '20251014-4',
        '20250521-A',
        '20250608-B',
        '20250618-A',
        '20250622-A',
        '20250630-A',
        '20250730-A',
        '20250807-A',
        '20250808-A',
        '20250814-6',
        '20250819-6',
        '20250823-6',
        '20250901-1',
        '20250902-5',
        '20250904-6',
        '20250904-7',
        '20250905-1',
        '20250905-7',
        '20250911-7',
        '20250914-1',
        '20250917-1',
        '20250917-6',       
        '20250927-4',
        '20250927-6',
        '20250928-1',
        '20251013-2',
        '20251015-2',
        '20251015-6',
        '20250524-B',
        '20250629-A',
        '20250725-A',
        '20250729-A',
        '20250731-A',
        '20250803-A',
        '20250804-A',
        '20250813-6',
        '20250814-4',
        '20250817-6',
        '20250820-7',   
        '20250826-7',
        '20250827-6',       
        '20250902-4',
        '20250904-1',
        '20250911-4',
        '20250915-6',
        '20250920-2',
        '20250920-4',
        '20250922-4',
        '20250925-6',
        '20250928-4',
        '20250929-4',
        '20251010-2',
        '20251011-6',
        '20251015-1',
        '20251016-2',
        '20251017-1',
        '20250520-A',
        '20250621-A',
        '20250702-A',
        '20250717-A',
        '20250719-A',
        '20250805-A',
        '20250815-4',
        '20250817-7',
        '20250824-6',   
        '20250901-7',
        '20250903-1',
        '20250904-4',
        '20250906-1',
        '20250906-7',
        '20250907-6',
        '20250909-6',
        '20250912-4',
        '20250916-2',
        '20250916-6',
        '20250917-4',
        '20250921-1',
        '20250922-6',
        '20251009-2',
        '20251009-4',
        '20251009-6',
        '20251013-4',
        '20251016-1',   
        '20251017-2',       
        '20250525-B',
        '20250605-B',
        '20250617-A',
        '20250620-A',
        '20250623-A',
        '20250625-A',
        '20250703-A',
        '20250722-A',
        '20250726-A',
        '20250816-5',
        '20250818',
        '20250822-6',
        '20250823-7',
        '20250826-6',
        '20250902-1',
        '20250902-7',
        '20250911-1',
        '20250915-1',
        '20250915-4',
        '20250918-4',
        '20250920-6',
        '20250923-6',
        '20250925-4',
        '20250928-2',
        '20251010-1',
        '20251012-2',
        '20251014-6',
        '20251016-6',
        '20251017-4',
    ]
    # ID_idx = 0
    # ID = '20250825-6'
    save_dir = f'/bigdata/daijin/ECGbed_data/ECG_cut_0331/'
    for ID_idx, ID in tqdm(enumerate(ID_list)):     
        filter_cut_bedholter(ID, \
            root_dir='/nas/data_ECG/Shuzhou_bed_holter/Data/',\
            beat_dir='/bigdata/daijin/ECGbed_data/ECGbeats_df/',\
            save_dir=save_dir) 
        print(f'{ID_idx}, Processing ID: {ID} FINISHED')
        break