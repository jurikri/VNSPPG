# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:01:29 2023

@author: PC
"""

import sys;
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
sys.path.append(r'C:\SynologyDrive\worik in progress\SOSOdevice\VNS_PPG_1')

import VNS_PPG_def
import msFunction
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

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

# def smooth_array(arr, window_size=1):
#     smoothed_array = np.zeros_like(arr, dtype=float)
#     n = len(arr)

#     for i in range(n):
#         # Determine the start and end indices for averaging
#         start_index = max(i - window_size, 0)
#         end_index = min(i + window_size + 1, n)
        
#         # Calculate the average
#         smoothed_array[i] = np.mean(arr[start_index:end_index])

#     return smoothed_array

#%%
i = 0
label_n_ix = np.array(['baseline', 'baseline_after', 'VNS_para', 'VNS_sym', \
                       'VNS_para_after', 'VNS_sym_after', 'VNS_sym_sham'])
    
mssave_array = msFunction.msarray([len(label_n_ix), 3])
# for i in range(len(nlist)):
# for i in [2,4,6,8,17,19,20,24,26,29,31,34]:
# for i in [2,4,6,8,17,19,20, 23,24,26, 27,28,29,31,32,33,34, 35, 36, 37]:
for i in [0,1,2,4,5,6,10,11,13,14,15,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37]:
    
# for i in [35, 36, 37]:
    # exlist = ['20231214_1_msbak_headband_baseline.pickle', \
    #           '20231214_2_msbak_headband_baseline.pickle']
    # if nlist[i] in exlist: continue # PPG 손끝 측정이므로, 제거함
    print(i, nlist[i])
    if not 'msbak' in nlist[i]: continue
    
    
    
    u_ixs = find_char_indexes(nlist[i], '_')
    end_ixs = nlist[i].find('.pickle')
    
    date_n = nlist[i][:8]
    session_n = nlist[i][9:10]
    state = nlist[i][u_ixs[2]+1:u_ixs[3]]
    subject = nlist[i][u_ixs[0]+1:u_ixs[1]]
    
    # if session_n == '1' or '20231214_3_msbak_heaband_baseline' in nlist[i]: 
    # if session_n == '1' in nlist[i]: 
    #     print(nlist[i], 'excluded')
    #     continue
    
    # print('subject', subject)
    
    #%grouping
    label = None
    
    sw_baseline = 'baseline' in nlist[i]
    sw_vns = 'VNS' in nlist[i]
    sw_para = 'para' in nlist[i]
    sw_sym = 'sym' in nlist[i]
    sw_3hz = '3hz' in nlist[i]
    sw_after = 'after' in nlist[i]
    sw_3hz = '3hz' in nlist[i]
    sw_sham = 'sham' in nlist[i]

    if sw_baseline and not(sw_vns) and not(sw_after): 
        label = 'baseline' 
        
    elif sw_baseline and not(sw_3hz) and sw_after and not(sw_sham):
        label = 'baseline_after' 
        
    elif sw_para and not(sw_3hz) and not(sw_after) and not(sw_sham):
        label = 'VNS_para'
        
    elif sw_sym and not(sw_3hz) and not(sw_after) and not(sw_sham):
        label = 'VNS_sym'
        
    elif sw_para and not(sw_3hz) and sw_after and not(sw_sham):
        label = 'VNS_para_after'
        
    elif sw_sym and not(sw_3hz) and sw_after and not(sw_sham):
        label = 'VNS_sym_after'
               
    if not(label is None):
        # print()
        # print(i, nlist[i])
        print('label >>', label)
        print()

#%
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
        # if prename == '20231214_1_': prename = '20231214_3_'
        for j in range(len(nlist)):
            if prename in nlist[j]: break
        print('prename', nlist[j])
        
        # prename = nlist[i] # self
        
        with open(path2 + nlist[j], 'rb') as f:  # Python 3: open(..., 'wb')
            msdict_pre = pickle.load(f)
            
        # with open(path2 + nlist[i], 'rb') as f:  # Python 3: open(..., 'wb')
        #     msdict = pickle.load(f) # self

        wavelet_data_pre = msdict_pre['template']
        wavelet_data_pre = wavelet_data_pre.swapaxes(0,2)
        nancheck = np.mean(np.mean(wavelet_data_pre, axis=1), axis=1)
        nanvix = np.isnan(nancheck)==0
        wavelet_data_pre = wavelet_data_pre[nanvix]
        
        ppg_data_pre = msdict_pre['msdata_PPG'][nanvix]
        _, _, _, _, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=ppg_data_pre)
        peak_times = peaks
        _, _, _, BPM_pre, _ = VNS_PPG_def.msmain(SR=SR, \
                            ppg_data=ppg_data_pre[-int(SR*310):-int(SR*10)])

        
        #%
        
        sampling_rate = 250  # in Hz
        
        # PPG analysis ###
        # _, _, _, _, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=ppg_data)
        # peak_times = peaks
        SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, \
                            ppg_data=ppg_data[-int(SR*310):-int(SR*10)])
        # PPG 분석시 마지막 -5:10 ~ -0:10 사용

        # SDNN, RMSSD, pNN50, BPM, _ = VNS_PPG_def.msmain(SR=SR, ppg_data='', peak_times=peak_times)
        BPM_ratio = ((BPM / BPM_pre) - 1) * 100
        print('SDNN, RMSSD, pNN50, BPM', np.round([SDNN, RMSSD, pNN50, BPM_ratio], 1))
        mssave_array[label_n][2].append([SDNN, RMSSD, pNN50, BPM_ratio])
            # mssave_array[][2] 는 PPG data allocation
            
        if label_n == 0 and BPM_ratio != 0:
            import sys; sys.exit()
        
        # Window size and step size for slicing
        window_size = 10 * sampling_rate  # 10 seconds window
        step_size = 10 * sampling_rate
        
        # msbins = np.arange(0, wavelet_data.shape[0]-window_size, step_size, dtype=int)
        
        # import sys; sys.exit()
        msbins = np.arange(wavelet_data.shape[0] - (SR * 310), \
                           wavelet_data.shape[0] - step_size - (SR * 10), step_size, dtype=int)
        
        for ch in range(2):
            # baseline 기준점 +10s ~ +70s 으로 고정
            tmp_pre = np.mean(wavelet_data_pre[int(SR*10):int(SR*70),:,ch], axis=0)
            nmr_pre = tmp_pre/np.sum(tmp_pre)
            
            for bi in range(len(msbins)):
                tmp = wavelet_data[msbins[bi] : msbins[bi] + window_size]
                if tmp.shape[0] == window_size:
                    tmp2 = np.mean(tmp[:,:,ch], axis=0)
                    nmr = tmp2/np.sum(tmp2)

                    # pre = np.median(nmr_pre[int(SR*10):int(SR*70),:], axis=0)
                    db = 10 * np.log10(np.power(nmr - nmr_pre, 2))
                    
                    mssave_array[label_n][ch].append(db)
                    
                # mssave[ch].append(nmr)
            

#%%
for laben_n in range(10):
    try:
        print(label_n_ix[laben_n])
        for ch in [0, 1]:
            tmp = np.array(mssave_array[laben_n][ch])
            # print(tmp.shape)
            plt.figure(ch)
            mean_tmp = np.median(tmp, axis=0)
            sem_tmp = np.std(tmp, axis=0) / np.sqrt(len(tmp))
            xaxis = range(len(mean_tmp))  # Replace with actual time points if available
            
            plt.plot(xaxis, mean_tmp, label=label_n_ix[laben_n] + '_' + str(ch))
            plt.fill_between(xaxis, mean_tmp - sem_tmp, mean_tmp + sem_tmp, alpha=0.2)
            
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
        
#%%
#% PPG VIS
fn = 0
xlabel = ['SDNN', 'RMSSD', 'pNN50', 'BPM']
for xn in range(len(xlabel)):
    plt.figure()
    # plt.title(xlabel[xn])
    
    mean_array, sem_array, n_values = [], [], []
    
    for laben_n in range(5):
        try:
            #%
            tmp = np.array(mssave_array[laben_n][2])
            mean_array.append(np.mean(tmp, axis=0)[xn])
            sem_array.append((np.std(tmp, axis=0) / np.sqrt(len(tmp)))[xn])
            n_values.append(len(tmp))  # N 값 저장
        except: pass
    
    # 평균값을 scatter plot으로 표시
    mean_array, sem_array = np.array(mean_array), np.array(sem_array)
    colors = ['gray', 'darkgreen', 'darkred', 'mediumseagreen', 'lightcoral']
    
    bars = plt.bar(label_n_ix[:len(mean_array)], mean_array, color=colors)

    # 각 막대에 N값 표시
    # meansave = []
    # for bar, ix in zip(bars, range(len(bars))):
    #     meansave.append((bar.get_height() - (sem_array[j]*10) / 2))
    ypos = [10, 5, 1, 1][xn]
    
    for bar, n in zip(bars, n_values):
        plt.text(bar.get_x() + bar.get_width() / 2, ypos, str(n), 
                 ha='center', va='bottom', fontsize=20)
    
    for i in range(len(mean_array)):
        plt.errorbar(label_n_ix[i], mean_array[i], yerr=sem_array[i], \
                     fmt='none', color='black', capsize=20)
    
        
    plt.xticks(rotation=45)
        # Y축 범위 설정
    y_min = np.max([np.min(mean_array - (sem_array*10)),-20])  # 여유를 두어 조정
    y_max = np.max(mean_array + (sem_array*3))
    plt.ylim([y_min, y_max])
    plt.xticks(rotation=45)
    plt.ylabel(xlabel[xn])
    
    spath = r'C:\SynologyDrive\worik in progress\SOSOdevice\VNS_PPG_1' + '\\'
    ptitle = xlabel[xn]
    #%%
    plt.savefig(spath + ptitle, dpi=200, bbox_inches='tight', transparent=True)
    
    
    # plt.show()
    # plt.legend()
     
    
    # BPM은 baseline 대비로 보아야한다
    # baseline-after, sym after sample들이 필요하다.
    # 자극은 20분 가하고, 마지막 5분 보는것으로 통일하자
    # baseline과 after는 6분 10초 recording으로 통일
    
        













