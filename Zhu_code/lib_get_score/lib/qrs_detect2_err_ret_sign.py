# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import filtfilt,resample,medfilt,lfilter
def circshift(u,shiftnum1,shiftnum2=0):
    need_reshape = False
    if u.ndim == 1:
        u = u.reshape(-1, 1)
        need_reshape = True
    
    # 使用np.roll进行循环移位
    shifted = np.roll(u, shift=shiftnum1, axis=0)
    shifted = np.roll(shifted, shift=shiftnum2, axis=1)
    
    return shifted.ravel() if need_reshape else shifted

def qrs_detect2_err_ret_sign(ecg,REF_PERIOD = 0.250,THRES = 0.48,fs = 256,fid_vec = [],SIGN_FORCE = [],debug = 0,WIN_SAMP_SZ = 7):

    fs = int(fs)
    MED_SMOOTH_NB_COEFF = round(fs/200)*2+1
    # length is 30 for fs=256Hz
    INT_NB_COEFF = round(WIN_SAMP_SZ*fs/256)
    #perform search back (FIX ME: should be in function param)
    SEARCH_BACK = 1
    #if you want to force the energy threshold value (FIX ME: should be in function param)
    MAX_FORCE = []
    # if the median of the filtered ECG is inferior to MINAMP then it is likely to be a flatline
    sign = 0
    # note the importance of the units here for the ECG (mV) 
    MIN_AMP = 0.1
    # number of input samples
    NB_SAMP = len(ecg.T); 
    err_code = 0
    tm = np.arange(1/fs, np.ceil(NB_SAMP/fs) + 1/fs, 1/fs)
    
    b1=[-7.757327341237223e-05  ,-2.357742589814283e-04 ,-6.689305101192819e-04 ,-0.001770119249103 ,
        -0.004364327211358 ,-0.010013251577232 ,-0.021344241245400 ,-0.042182820580118 ,-0.077080889653194,
        -0.129740392318591 ,-0.200064921294891 ,-0.280328573340852 ,-0.352139052257134 ,-0.386867664739069 ,
        -0.351974030208595 ,-0.223363323458050 ,0 ,0.286427448595213 ,0.574058766243311 ,
        0.788100265785590 ,0.867325070584078 ,0.788100265785590 ,0.574058766243311 ,0.286427448595213 ,0 ,
        -0.223363323458050 ,-0.351974030208595 ,-0.386867664739069 ,-0.352139052257134,
        -0.280328573340852 ,-0.200064921294891 ,-0.129740392318591 ,-0.077080889653194 ,-0.042182820580118 ,
        -0.021344241245400 ,-0.010013251577232 ,-0.004364327211358 ,-0.001770119249103 ,-6.689305101192819e-04,
        -2.357742589814283e-04 ,-7.757327341237223e-05]

    
    target_num = int(len(b1) * fs / 250)  # 根据采样率计算目标点数
    b1 = resample(b1, target_num)

    # filtfilt参数修正
    bpfecg = filtfilt(b1, [1.0], ecg)  # 分母系数设为[1.0]
    
    if (np.sum(abs(ecg-np.median(ecg))>MIN_AMP)/NB_SAMP)>0.01:
        # if 20% of the samples have an absolute amplitude which is higher
        # than MIN_AMP then we are good to go.
        
        # == P&T operations
        dffecg = np.diff(bpfecg);  # (4) differentiate (one datum shorter)
        sqrecg = pow(dffecg,2); # (5) square ecg
        # intecg = filter(np.ones(INT_NB_COEFF),1,sqrecg); # (6) integrate
        intecg = lfilter(np.ones(INT_NB_COEFF), [1.0], sqrecg)
        mdfint = medfilt(intecg,MED_SMOOTH_NB_COEFF);  # (7) smooth
        delay  = np.ceil(INT_NB_COEFF/2); 
        mdfint = circshift(mdfint, int(-delay), 0)  # 添加列维度后移位
        # % look for some measure of signal quality with signal fid_vec? (FIX ME)
        mdfintFidel = mdfint.reshape(-1, 1)
        mdfintFidel = mdfintFidel[np. isfinite (mdfintFidel)]
        # == P&T threshold
        if NB_SAMP/fs>90:
            xs = np.sort(mdfintFidel[fs:fs*90]); 
        else:
            xs = np.sort(mdfintFidel[fs:]); 
        
        if len(MAX_FORCE)==0:
           if NB_SAMP/fs>10:
                ind_xs = np.ceil(98/100*len(xs)); 
                en_thres = xs[int(ind_xs)]; # if more than ten seconds of ecg then 98% CI
           else:
                ind_xs = np.ceil(99/100*len(xs)); 
                en_thres = xs[int(ind_xs)] # else 99% CI  
        else:
           en_thres = MAX_FORCE
        

        # build an array of segments to look into
        poss_reg = mdfint>(THRES*en_thres)
        # in case empty because force threshold and crap in the signal
        if len(poss_reg) == 0:
             poss_reg = np.zeros(10,dtype=bool) 
             poss_reg[9] = True
        # == P&T QRS detection & search back
        if SEARCH_BACK:
            indAboveThreshold = np.where(poss_reg)[0] # ind of samples above threshold
            RRv = np.diff(tm[indAboveThreshold])  # compute RRv
            
            RRv_valid = RRv[RRv > 0.01]
            if len(RRv_valid) == 0:
                err_code = -1
                return err_code, [] , sign
            else:
                medRRv = np.median(RRv_valid)

            indMissedBeat = np.where(RRv>1.5*medRRv)[0] # missed a peak?
            # find interval onto which a beat might have been missed
            indStart = indAboveThreshold[indMissedBeat]
            indEnd = indAboveThreshold[indMissedBeat+1]

            for i in range(len(indStart)):
                current_slice = slice(indStart[i], min(indEnd[i] + 1, len(poss_reg)))
                # 获取信号片段（无需转置，自动广播）
                segment = mdfint[current_slice]
                # 阈值判断并更新区域标记
                poss_reg[current_slice] = segment > (0.5 * THRES * en_thres)
            
        
        # find indices into boudaries of each segment
        padded_reg = np.concatenate(([False], poss_reg, [False]))  # 等效MATLAB的[0 poss_reg' 0]
        diff_reg = np.diff(padded_reg.astype(int))
        left = np.where(diff_reg == 1)[0]
        right = np.where(diff_reg == -1)[0]
        # looking for max/min?
        if len(SIGN_FORCE) > 0:
            sign = SIGN_FORCE
        else:
            valid_segments = left < 30*fs
            nb_s = np.sum(valid_segments)  # 统计True的数量
            loc = np.zeros(nb_s, dtype=int)
            for j in range(nb_s):
                segment = bpfecg[left[j]:right[j]]
                loc[j] = np.argmax(np.abs(segment)) + left[j]  # 自动0-based索引
        if len(loc) > 0:
            sign = np.mean(ecg[loc]) - np.mean(ecg)
        else:
            return -1,[],0  # 或其他你认为合理的默认值  # FIX ME: change to median?  
        # if len(SIGN_FORCE) > 0:
        #     sign = SIGN_FORCE
        # else:
        #     if np.median(bpfecg)<np.mean(bpfecg):
        #         sign = 1
        #     else:
        #         sign = -1

        

        # loop through all possibilities         
        compt=1;        
        NB_PEAKS = len(left)
        # 初始化部分修改
        maxval = []  # 改为列表
        maxloc = []   # 改为列表
        compt = 0     # 从0开始计数更安全

        for i in range(NB_PEAKS):
            # 峰值检测逻辑保持不变
            if sign > 0:
                current_max = np.max(ecg[left[i]:right[i]])
                current_loc = np.argmax(ecg[left[i]:right[i]])
            else:
                current_max = np.min(ecg[left[i]:right[i]])
                current_loc = np.argmin(ecg[left[i]:right[i]])
            
            # 位置校正
            current_loc = current_loc + left[i] - 1  # MATLAB转Python索引修正
            
            # 添加当前峰值
            maxval.append(current_max)
            maxloc.append(current_loc)
            compt += 1
            
            # 不应期处理
            if compt >= 2:  # 至少有两个峰值时处理
                last_idx = compt-1
                prev_idx = compt-2
                
                # 计算时间差
                delta_t = maxloc[last_idx] - maxloc[prev_idx]
                
                if delta_t < fs*REF_PERIOD:
                    if abs(maxval[last_idx]) < abs(maxval[prev_idx]):
                        # 删除当前峰值
                        del maxval[last_idx]
                        del maxloc[last_idx]
                        compt -= 1
                    else:
                        # 删除前一个峰值
                        del maxval[prev_idx]
                        del maxloc[prev_idx]
                        compt -= 1
        qrs_pos = maxloc # datapoints QRS positions 
        R_t = tm[maxloc] # timestamps QRS positions
        R_amp = maxval # amplitude at QRS positions
    else:
        qrs_pos = [-1,-1,-1]
        err_code = -1
    return err_code, qrs_pos, sign