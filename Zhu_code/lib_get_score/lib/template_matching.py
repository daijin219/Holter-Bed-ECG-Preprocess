# -*- coding: utf-8 -*-
import numpy as np

def template_matching(ecg, qrs_i_raw):
    n = len(ecg)
    qrs_i_raw = np.array(qrs_i_raw)
    
    # R峰检测及心跳间隔计算
    if len(qrs_i_raw) < 2:
        coe = 0.0
    else:
        # 计算RR间期中值
        RR = int(np.floor(np.median(np.diff(qrs_i_raw))))
        
        # 提取QRS复合波窗口
        start_indices = qrs_i_raw - RR//2
        end_indices = qrs_i_raw + RR - RR//2
        
        # 筛选有效窗口
        valid_mask = (start_indices > 0) & (end_indices < n)
        start_valid = start_indices[valid_mask]
        end_valid = end_indices[valid_mask]
        n_r = len(start_valid)
        
        if n_r == 0:
            coe = 0.0
        else:
            # 创建QRS模板
            qrs_template = np.zeros(RR + 1)
            for i in range(n_r):
                s = int(start_valid[i])
                e = int(end_valid[i])
                qrs_template += ecg[s:e+1]  # +1包含结束索引
                
            qrs_template /= n_r
            
            # 计算相关系数
            correlations = []
            template_mean = np.mean(qrs_template)
            for i in range(n_r):
                s = int(start_valid[i])
                e = int(end_valid[i])
                segment = ecg[s:e+1]
                segment_mean = np.mean(segment)
                
                numerator = np.sum((qrs_template - template_mean) * (segment - segment_mean))
                denominator = np.sqrt(np.sum((qrs_template - template_mean)**2) * np.sum((segment - segment_mean)**2))
                
                if denominator != 0:
                    correlations.append(numerator / denominator)
                else:
                    correlations.append(0.0)
            
            coe = np.mean(correlations) if correlations else 0.0
            
    return coe