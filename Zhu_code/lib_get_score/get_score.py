# -*- coding: utf-8 -*-
import numpy as np
import joblib
from .lib.feature_get import feature_get 
from pathlib import Path
import time



global _model_class, _model_score, _column_means, _column_stds
_base = Path(__file__).parent
_model_class  = joblib.load(_base / 'model/model_class.joblib')
_model_score  = joblib.load(_base / 'model/model_score.joblib')
_column_means, _column_stds = np.load(_base / 'model/norm_para.npy', allow_pickle=True)

def get_score(ecg, window=1.5, step=1, fs=500):
    ch = min(ecg.shape)
    error_code_list = np.zeros(ch)
    if ecg.ndim != 2:
        error_code_list = np.ones(ch)*-100
        raise ValueError("ECG must be 2-D array [ch, samples]")
    
    if ecg.shape[0] > ecg.shape[1]:
        ecg_num,ch = ecg.shape
    else:
        ecg = ecg.T
        ecg_num,ch = ecg.shape
    score  = np.zeros(ch)
    # if ecg_num >= 5000:
    #     ecg = ecg[:,-5000:]
    # else:
    #     error_code_list = -101*np.ones(ch)
    #     return error_code_list,score
    
    for i in range(ch):   
        error_code,feature = feature_get(ecg[:,i],window,step,fs)
        if error_code !=0:
            score[i]=-1
            continue
        if max(ecg[:,i])-min(ecg[:,i])>5:
            score[i]=-1
            continue
        try:
            normalized_data = (feature - _column_means) / _column_stds
            if error_code != 0: 
                error_code_list[i] = error_code
            else:
                
                MA = _model_class.predict(normalized_data.reshape(1,-1))[0] 
                
                if  MA == 1: 
                    score[i]=-1
                else:
                    score[i] = _model_score.predict(normalized_data.reshape(1,-1))[0]
        except:
            error_code_list[i] = -111
    try:
        score = np.asarray(score)
        error_code_list = np.asarray(error_code_list) 
    except:
        error_code_list = -112*np.ones(ch)
    finally:
        return error_code_list,score



if __name__ == '__main__':

    import scipy.io as sio
    import pandas as pd
    import numpy as np
    from scipy import signal
    ecg = sio.loadmat(r"C:\Users\rewalk\Documents\WXWork\1688857197660272\Cache\File\2025-06\testecg.mat")
    ecg = ecg['testecg']
    ecg_cut = ecg.reshape(len(ecg),-1).T
    b= sio.loadmat(r"D:\SynologyDrive_Fdu_co\Bio-Mattress-Project\算法\Score_20241204\dataset\Data_20240825\ECG_BPF.mat")['ECG_BPF'].flatten()
    score_list = []
    feature_list = []
    data_list = []
    for i in range(len(ecg_cut.T)//5000):
        temp = ecg_cut[:,i*5000:i*5000+5000]
        data_list.append(temp)
        # temp = signal.lfilter(b,[1.0],temp, axis=0)
        _,temp_feature = feature_get(temp,1.5,1,500)
        feature_list.append(temp_feature)
        _,score = get_score(temp, 1.5, 1, 500)
        score_list.append(score)
    print(score)
    # np.save('score_temp.npy',score_list)
    # np.save('feature_temp.npy',np.asarray(feature_list))
    # np.save('data_temp.npy',data_list)
    import matplotlib.pyplot as plt
    plt.plot(5000*np.arange((len(score_list)),dtype="int"),score_list)
    plt.plot(ecg)
    plt.show()