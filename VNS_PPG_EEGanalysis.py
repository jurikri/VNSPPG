# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:01:29 2023

@author: PC
"""

import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.layers import BatchNormalization, Dropout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random, pickle, cv2
from shapely.geometry import Point, Polygon
import cv2
from PIL import Image,ImageFilter, ImageEnhance, ImageOps
from tqdm import tqdm
# from PIL import Image,ImageEnhance
import os
import matplotlib.image as img
import pandas as pd
import gc
import time
import datetime
import ast

import sys;
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction
from scipy.signal import find_peaks

#%% listup

path2 = r'D:\SOSO_datasave_headband\\'; msFunction.createFolder(path2)
flist = os.listdir(path2)
nlist = []
for i in range(len(flist)):
    if os.path.splitext(flist[i])[1] == '.pickle' in flist[i]:
        nlist.append(flist[i])
nlist = list(set(nlist))
print(nlist)
nlist.sort()

#%% index
def find_char_indexes(string, char):
    indexes = []
    for i in range(len(string)):
        if string[i] == char:
            indexes.append(i)
    return indexes

def smooth_array(arr, window_size=1):
    smoothed_array = np.zeros_like(arr, dtype=float)
    n = len(arr)

    for i in range(n):
        # Determine the start and end indices for averaging
        start_index = max(i - window_size, 0)
        end_index = min(i + window_size + 1, n)
        
        # Calculate the average
        smoothed_array[i] = np.mean(arr[start_index:end_index])

    return smoothed_array

#%%
i = 0
label_n_ix = np.array(['baseline', 'VNS_para', 'VNS_sym', 'VNS_para_after', 'VNS_sym_after', 'VNS_sym_sham'])
mssave_array = msFunction.msarray([len(label_n_ix), 2])
# for i in range(len(nlist)):
# for i in [2,4,6,8,17,19,20,24,26,29,31,34]:
# for i in [2,4,6,8,17,19,20, 23,24,26, 27,28,29,31,32,33,34, 35, 36, 37]:
    
for i in [35, 36, 37]:
    if nlist[i] in ['20231214_1_msbak_headband_baseline.pickle', \
                    '20231214_2_msbak_headband_baseline.pickle']: # PPG 손끝 측정이므로, 제거함
        continue
    
    print(i, nlist[i])
    
    u_ixs = find_char_indexes(nlist[i], '_')
    end_ixs = nlist[i].find('.pickle')
    
    date_n = nlist[i][:8]
    session_n = nlist[i][9:10]
    state = nlist[i][u_ixs[2]+1:u_ixs[3]]
    subject = nlist[i][u_ixs[0]+1:u_ixs[1]]
    
    print('subject', subject)
    
    #%grouping
    label = None
    
    if 'baseline' in nlist[i]:
        print('baseline')
        label = 'baseline'
        
    elif 'VNS' in nlist[i]:
        if 'para' in nlist[i]:
            if '3hz' in nlist[i]:
                if 'after' in nlist[i]: 
                    print('VNSpara, 3hz, after')
                else:
                    print('VNSpara, 3hz')
            else:
                if 'after' in nlist[i]:
                    print('VNSpara, after')
                    label = 'VNS_para_after'
                else:
                    print('VNSpara')
                    label = 'VNS_para'
                    
        elif 'sym' in nlist[i]:
            if '3hz' in nlist[i]:
                if 'after' in nlist[i]: 
                    print('VNSsym, 3hz, after')
                else:
                    print('VNSsym, 3hz')
            else:
                if 'after' in nlist[i]:
                    print('VNSsym, after')
                    label = 'VNS_sym_after'
                else:
                    if 'sham' in nlist[i]:
                        print('VNSsym, sham')
                        label = 'VNS_sym_sham'
                    else:
                        print('VNSsym')
                        label = 'VNS_sym'
                    
    if not(label is None):
        print()
        print(i, nlist[i])
        print('label >>', label)
    
    label_n = None
    if not label is None:
        label_n = np.where(label_n_ix==label)[0][0]
        print('label_n', label_n)
        # print('date_n', date_n, 'session_n', session_n, 'etype', etype)
    
        with open(path2 + nlist[i], 'rb') as f:  # Python 3: open(..., 'wb')
            msdict = pickle.load(f)
    
        SR = msdict['SR']
        wavelet_data = msdict['template']
        phase_data = msdict['template_phase']
        wavelet_data = wavelet_data.swapaxes(0,2)
        phase_data = phase_data.swapaxes(0,2)
        
        nancheck = np.mean(np.mean(wavelet_data, axis=1), axis=1)
        nanvix = np.isnan(nancheck)==0
        if False:
            plt.figure()
            plt.plot(np.isnan(nancheck))
            # plt.title('nancheck _' +  msid)
            print('After nan fix', wavelet_data.shape)
        
        ##
        ppg_data = msdict['msdata_PPG'][nanvix]
        wavelet_data = wavelet_data[nanvix]
        phase_data = phase_data[nanvix]
        
        print(wavelet_data.shape, phase_data.shape)
        
        #% load pre
         
        prename = nlist[i][:u_ixs[0]] + '_1_'
        if prename == '20231214_1_': prename = '20231214_3_'
        for j in range(len(nlist)):
            if prename in nlist[j]: break
        print('prename', nlist[j])
        
        prename = nlist[i] # self
        
        with open(path2 + nlist[j], 'rb') as f:  # Python 3: open(..., 'wb')
            msdict = pickle.load(f)
            
        # with open(path2 + nlist[i], 'rb') as f:  # Python 3: open(..., 'wb')
        #     msdict = pickle.load(f) # self

        wavelet_data_pre = msdict['template']
        wavelet_data_pre = wavelet_data_pre.swapaxes(0,2)
        
        nancheck = np.mean(np.mean(wavelet_data_pre, axis=1), axis=1)
        nanvix = np.isnan(nancheck)==0

        wavelet_data_pre = wavelet_data_pre[nanvix]
    
        #%
        
        #%
        
        sampling_rate = 250  # in Hz
        # Window size and step size for slicing
        window_size = 10 * sampling_rate  # 10 seconds window
        step_size = 10 * sampling_rate
        
        # msbins = np.arange(0, wavelet_data.shape[0]-window_size, step_size, dtype=int)
        
        # import sys; sys.exit()
        msbins = np.arange(wavelet_data.shape[0] - (SR * 240), \
                           wavelet_data.shape[0] - step_size - (SR * 10), step_size, dtype=int)
        
        
        # msbins = np.arange(0, SR * 30, step_size, dtype=int)

        
        # mssave = [[],[]]
        
        for ch in range(2):
            for bi in range(len(msbins)):
                tmp = wavelet_data[msbins[bi] : msbins[bi] + window_size]
                if tmp.shape[0] == window_size:
                    tmp2 = np.mean(tmp[:,:,ch], axis=0)[:]
                    nmr = tmp2/np.sum(tmp2)
                    
                    tmp_pre = np.mean(wavelet_data_pre[int(SR*10):int(SR*70),:,ch], axis=0)[:]
                    nmr_pre = tmp_pre/np.sum(tmp_pre)
 
                    # pre = np.median(nmr_pre[int(SR*10):int(SR*70),:], axis=0)
                    db = 10 * np.log10(np.power(nmr - nmr_pre, 2))
                    
                    mssave_array[label_n][ch].append(db)
                    
                # mssave[ch].append(nmr)
                
#%%

# def plot_semshade(tmp):
#     import numpy as np
#     import matplotlib.pyplot as plt
    
#     mean_tmp = np.mean(tmp, axis=0)
#     sem_tmp = np.std(tmp, axis=0) / np.sqrt(len(tmp))
    
#     time = range(len(mean_tmp))  # Replace with actual time points if available
    
#     plt.plot(time, mean_tmp, label=label_n_ix[fn] + '_' + str(ch))
#     plt.fill_between(time, mean_tmp - sem_tmp, mean_tmp + sem_tmp, alpha=0.2)
    
#     plt.legend()
#     plt.show()


for fn in range(10):
    try:
        print(label_n_ix[fn])
        for ch in [0, 1]:
            tmp = np.array(mssave_array[fn][ch])
            # print(tmp.shape)
            plt.figure(ch)
            mean_tmp = np.median(tmp, axis=0)
            sem_tmp = np.std(tmp, axis=0) / np.sqrt(len(tmp))
            time = range(len(mean_tmp))  # Replace with actual time points if available
            
            plt.plot(time, mean_tmp, label=label_n_ix[fn] + '_' + str(ch))
            plt.fill_between(time, mean_tmp - sem_tmp, mean_tmp + sem_tmp, alpha=0.2)
            
            delta = np.mean(tmp[:, 1:3], axis=1)
            theta = np.mean(tmp[:, 4:8], axis=1)
            alpha = np.mean(tmp[:, 8:12], axis=1)
            beta = np.mean(tmp[:, 15:18], axis=1)
            gamma = np.mean(tmp[:, 30:], axis=1)
            
            print(np.round([np.mean(delta), np.mean(theta), \
                  np.mean(alpha), np.mean(beta), np.mean(gamma)], 4))
                
            print(np.round(np.median(beta/theta), 4))
            print(np.round(np.median(delta/theta), 4))
            
    except: pass

plt.legend()
        
    
                
    # plt.figure()
    # # for ch in range(2):
    # plt.title(nlist[i])
    # plt.plot(np.mean(np.array(mssave), axis=(0, 1)), label=nlist[i])
    # plt.legend()

    
                
            # plt.plot(np.mean(phase_data[msbins[bi] : msbins[bi] + window_size,:,1], axis=0))
        
    
    
    #%%
    
    if False:
        plt.plot(ppg_data[SR*75:SR*85])
        plt.plot(ppg_data)

    ##
    for ws in [20]:
    
        #%
        sampling_rate = 250  # in Hz
        # Window size and step size for slicing
        window_size = ws * sampling_rate  # 10 seconds window
        step_size = int(window_size / 4)
        
        # Slicing the data
        sliced_data, sliced_time = [], []
        for start in range(0, len(ppg_data) - window_size + 1, step_size):
            window = ppg_data[start:start + window_size]
            sliced_data.append(window)
            sliced_time.append(list(range(start, start + window_size)))
            
        # print(len(sliced_data[2]), len(sliced_time[2]))
        
        # Checking the total number of slices
        total_slices = len(sliced_data)
        accumulated_timeline = np.zeros((total_slices, int(ppg_data.shape[0])))
        # rsample = random.sample(list(range(len(sliced_data))), 20)
        
        for j in range(total_slices):
            
            
            
            slice_number = j  # 10th slice (indexing starts from 0)
            slice_to_plot = sliced_data[slice_number]
            
            
            n = 1
            diff1 = (slice_to_plot[n:] - slice_to_plot[:-n])
            if False:
                plt.plot(slice_to_plot/np.abs(np.sum(slice_to_plot))); np.median(slice_to_plot)
                plt.plot(diff1/np.abs(np.sum(diff1)))
            
            marker = np.zeros(len(slice_to_plot))
            # msmedian = np.median(np.sort(slice_to_plot)[:int(len(slice_to_plot)*0.6)])
            wait_sw = 1
            gapsw = np.inf
            recent_mark2 = None
            for c in range(len(diff1)):
                if gapsw < SR * 0.05: 
                    gapsw += 1; continue

                if wait_sw == 1 and diff1[c] > 1:
                    swindown_mean = np.mean(slice_to_plot[np.max([c-SR, 0]):np.min([c+SR, len(slice_to_plot)])])
                    if slice_to_plot[c-1] < swindown_mean:
                        s, e = int(np.max([c-(SR*0.2), 0])), int(np.min([c+(SR*0.2), len(slice_to_plot)]))
                        
                        sswindow = slice_to_plot[s:e]
                        mix = np.argmin(sswindow)
                        point = list(range(s, e))[mix]
                        marker[point-1] = 1
                        # print(c, 'marker 1')
                        wait_sw = 2
                        gapsw = 0
                        
                        # if not(recent_mark2 is None):
                        #     import sys; sys.exit()
                            
                        #     diff2 = diff1[recent_mark2:point-1]
                        #     diff3 = diff2[1:] - diff2[:-1]
                            
                            
                        #     sdiff1 = smooth_array(diff1[recent_mark2:point-1], window_size=2)
                            
                        #     plt.plot(slice_to_plot[recent_mark2-100:point-1+100])
                        #     plt.plot(slice_to_plot[recent_mark2:point-1])
                        #     plt.plot(diff1[recent_mark2:point-1])
                        #     plt.plot(sdiff1)
                            
                        # recent_mark2 = None
                        
                        
                    # import sys; sys.exit()
                        
                elif wait_sw == 2 and diff1[c] < -0.3: 
                    s, e = int(np.max([c-(SR*0.2), 0])), int(np.min([c+(SR*0.2), len(slice_to_plot)]))
                    
                    sswindow = slice_to_plot[s:e]
                    mix = np.argmax(sswindow)
                    point = list(range(s, e))[mix]
                    marker[point-1] = 2
                    # print(c, 'marker 2')
                    wait_sw = 1
                    gapsw = 0
                    recent_mark2 = point-1
                    
                   
                    
                # elif wait_sw == 2.5 and diff1[c] > -0.5:
                #     wait_sw = 3
                #     gapsw = 0
                    
                # elif wait_sw == 3 and diff1[c] < 0:
                #     marker[c-1] = 3
                #     # print(c, 'marker 3')
                #     wait_sw = 1
                #     gapsw = 0
                    
                #     slice_to_plot[]
                    
                #     recent_mark1 = None
                    
                    
                    
            if True:
                plt.figure()
                plt.title(str(j))
                plt.plot(slice_to_plot)
                ix1 = np.where(marker==1)[0]
                plt.scatter(ix1, slice_to_plot[ix1], color='red', label='Detected Peaks', s=10)
                
                ix1 = np.where(marker==2)[0]
                plt.scatter(ix1, slice_to_plot[ix1], color='purple', label='Detected Peaks', s=10)
                
                ix1 = np.where(marker==3)[0]
                plt.scatter(ix1, slice_to_plot[ix1], color='orange', label='Detected Peaks', s=10)
    
        
            #%%
                    
                    
                    
                
            
            
            
            
            if False:  
                plt.plot(slice_to_plot[0+n:300+n])
                plt.plot(diff1[0:300])

        
            
            # Assuming slice_to_plot contains the PPG data of interest
            # Peak detection using scipy's find_peaks
            peaks, _ = find_peaks(diff1, height=np.median(diff1), distance=SR*0.5, prominence=0.1)
            peaks = peaks + n
            accumulated_timeline[slice_number, np.array(sliced_time[slice_number])[peaks[1:-1]]] = 1
            
            # Extracting time points of the peaks
            # time_axis = np.linspace(0, 10, len(slice_to_plot))
            # peak_times = time_axis[peaks]
            
            
        # plt.hist(np.sum(accumulated_timeline, axis=0))
        acuumulated_cnt = np.sum(accumulated_timeline, axis=0)
        
        # pre post cut
        ppg_data = ppg_data[int(SR*5) : -int(SR*5)]
        acuumulated_cnt = acuumulated_cnt[int(SR*5) : -int(SR*5)]
        peak_times = np.where(acuumulated_cnt > 1)[0]
        
        # print(np.sum(acuumulated_cnt==1), np.sum(acuumulated_cnt==2), np.sum(acuumulated_cnt==3), np.sum(acuumulated_cnt==4))
 
        
        # gap = int(round(accumulated_timeline.shape[1]/len(peak_times)))
        # distance = gap * 0.7
        
        # for j in range(total_slices):
        #     slice_number = j  # 10th slice (indexing starts from 0)
        #     slice_to_plot = sliced_data[slice_number]
        
        #     from scipy.signal import find_peaks
        #     # Assuming slice_to_plot contains the PPG data of interest
        #     # Peak detection using scipy's find_peaks

        #     peaks, _ = find_peaks(slice_to_plot, height=np.mean(slice_to_plot), distance=distance, prominence=0.1)
            
        #     accumulated_timeline[slice_number, np.array(sliced_time[slice_number])[peaks[:]]] = 1
            
        #     # Extracting time points of the peaks
        #     time_axis = np.linspace(0, 10, len(slice_to_plot))
        #     peak_times = time_axis[peaks]
        
        #     if False:
        #         # Plotting the PPG data and the detected peaks
        #         plt.figure(figsize=(12, 6))
        #         plt.plot(time_axis, slice_to_plot, label='PPG Data')
        #         plt.scatter(peak_times, slice_to_plot[peaks], color='red', label='Detected Peaks')
        #         plt.title("PPG Data with Detected Peaks_" + str(j))
        #         plt.xlabel("Time (seconds)")
        #         plt.ylabel("Signal Amplitude")
        #         plt.legend()
        #         plt.grid(True)
        #         plt.show()
            
        # # plt.plot(np.sum(accumulated_timeline, axis=0))
        
        
        # peak_times = np.where(np.sum(accumulated_timeline, axis=0) > 0)[0]
        # print(ws, 'len(peak_times)', len(peak_times))
        if False:
            t = 5
            for i in range(0, int(ppg_data.shape[0]/SR), t):
                s = SR * i
                e = SR * (i + t)
                slice_to_plot = ppg_data[s:e]
                # pix =  np.where((np.sum(accumulated_timeline, axis=0) > 0)[s:e] == 1)[0] + n
                pix = np.where(acuumulated_cnt[s:e] > 1)[0]
                time_axis = np.linspace(0, t, len(slice_to_plot))
                
                plt.figure()
                plt.title(str(i))
                plt.plot(time_axis, slice_to_plot)
                peak_times = time_axis[pix]
                plt.scatter(peak_times, slice_to_plot[pix], color='red', label='Detected Peaks', s=10)
    
    #%
    
    peak_times = peak_times/SR*1000 # 단위 ms로
    
    
    
    def peak_times_in(peak_times, tmp):
      
        
    
        # 앞 2번째 부터, 뒤 2번째 peak 까지 잘라서 유효 범위 설정
        # 유효 범위 내에 peak들을 total map에 check,
        # total map에서 2번 이상 중복 체크된것만 유효값으로 check.
        
        # peak 간의 거리 값으로 환산
        # peak time point, 그 peak로 부터 다음 peak로의 거리로 저장
        
        # 거리의 std
        
        # peak_times는 감지된 peak들의 시간 좌표를 나타냅니다.
        NN_intervals = np.diff(peak_times)  # 연속된 peak들 사이의 시간 차이 계산
        
        exn = int(len(NN_intervals)*0.01)
        NN_intervals_SDNN = np.sort(NN_intervals)[exn:-exn]
        SDNN = np.std(NN_intervals_SDNN)  # NN 간격의 표준 편차 계산
        
        # # RMSSD 계산
        
        NN_intervals_diff = np.diff(NN_intervals)
        exn = int(len(NN_intervals_diff)*0.01)
        NN_intervals_diff_ex = np.sort(NN_intervals_diff)[exn:-exn]
        
        RMSSD = np.sqrt(np.mean(np.square(NN_intervals_diff_ex)))
        # 부교감신경계가 활성화될 때, 심박수는 빠르게 변화합니다 (예를 들어, 휴식 시 빠르게 감소). 이러한 빠른 변화는 연속된 NN 간격의 차이가 크게 되며, 이는 RMSSD 값이 증가하게 합니다.
        # 반면, 교감신경계의 영향은 보통 더 장기적이며, 연속된 NN 간격의 차이에 덜 민감하게 반응합니다.
    
        
        # # pNN50 계산
        differences = np.abs(NN_intervals_diff_ex)
        pNN50 = np.sum(differences > 50) / len(differences) * 100  # 50ms는 0.05초에 해당
        # 네, 말씀하신 대로 pNN50을 짧은 시간 윈도우의 관점에서 해석하면, 실제로 심장이 상대적으로 느리게 뛰는 순간들을 카운팅하는 것과 유사하게 볼 수 있습니다. pNN50 지표는 연속된 심박 간의 시간 차이가 50 밀리초 이상인 경우를 카운트하며, 이러한 큰 시간 차이는 심박수가 감소하는 순간을 반영할 수 있습니다.
    
        # # RRHRV 계산 
        # RR_intervals = np.diff(peak_times)  # RR 간격 계산
        # weighted_diffs = np.diff(RR_intervals) / ((RR_intervals[:-1] + RR_intervals[1:]) / 2)  # 연속적인 RR 간격의 차이를 평균으로 가중치 적용
        # rrHRV = np.mean(weighted_diffs)  # 가중치가 적용된 차이들의 평균
        # RRHRV는 연속적인 RR 간격의 차이를 그들의 평균으로 가중치를 두어 계산하는 방식을 기반으로 합니다. 이 방법은 특히 짧은 RR 시퀀스에 적합합니다

        BPM = np.round(np.sum(peak_times>0)/(tmp.shape[1]/SR/60), 1)
        
        if False:
            print('SDNN', np.round(SDNN, 2), 'RMSSD', np.round(RMSSD, 2), 'pNN50', np.round(pNN50, 2)) #, 'RRHRV 준비중..')
            print('BPM', BPM)
        
        return SDNN, RMSSD, pNN50, BPM
    
    mssave2 = []
    window_size = SR*60
    msbins = np.arange(0, accumulated_timeline.shape[1]-window_size, SR*5)
    for k in range(len(msbins)):
        tmp = accumulated_timeline[:, msbins[k]: msbins[k]+window_size]
        peak_times = np.where(np.sum(tmp, axis=0) > 0)[0]
        peak_times = peak_times/SR*1000 # 단위 ms로
        
        SDNN, RMSSD, pNN50, BPM = peak_times_in(peak_times, tmp)
        
        mssave2.append([SDNN, RMSSD, pNN50, BPM])
    mssave2 = np.array(mssave2)

    titles = ['SDNN', 'RMSSD', 'pNN50', 'BPM']
    if False:
        for fn in range(4):
            plt.figure()
            plt.title(nlist[i] + '_' + titles[fn])
            plt.plot(mssave2[:,fn])
            if fn in [0, 1]: plt.ylim([0,400])
            if fn in [2, 3]: plt.ylim([20,100])
            
        
    tmp = accumulated_timeline[:, :]
    peak_times = np.where(np.sum(tmp, axis=0) > 0)[0]
    peak_times = peak_times/SR*1000 # 단위 ms로
    
    SDNN, RMSSD, pNN50, BPM = peak_times_in(peak_times, tmp)
    print('SDNN, RMSSD, pNN50, BPM', np.round([SDNN, RMSSD, pNN50, BPM], 1))
    
        

            
        
    
    
 



























