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
sys.path.append(r'C:\SynologyDrive\worik in progress\SOSOdevice\VNS_PPG_1')
import VNS_PPG_def
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

for i in range(len(nlist)):

    if nlist[i] in ['20231214_1_msbak_headband_baseline', \
                    '20231214_2_msbak_headband_baseline.pickle']: # PPG 손끝 측정이므로, 제거함
        continue
    #%
    print(i, nlist[i])
    
    u_ixs = find_char_indexes(nlist[i], '_')
    end_ixs = nlist[i].find('.pickle')
    
    date_n = nlist[i][:8]
    session_n = nlist[i][9:10]
    state = nlist[i][u_ixs[2]+1:u_ixs[3]]
    subject = nlist[i][u_ixs[3]+1:end_ixs]
    
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
    
    if False:
        for s in [200,300,400,500]:
            # s = 200
            plt.figure()
            plt.plot(ppg_data[SR*s:SR*(s+7)])
        # plt.plot(ppg_data)

    ##
    sampling_rate = 250  # in Hz
        
    SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=ppg_data)
    peak_times = peaks
   
    mssave2 = []
    window_size = SR*60
    msbins = np.arange(0, ppg_data.shape[0]-window_size, SR*5)
    for k in range(len(msbins)):
        peak_times_tw = peak_times[np.logical_and(peak_times>msbins[k], peak_times<msbins[k]+window_size)]
        SDNN, RMSSD, pNN50, BPM, _ = VNS_PPG_def.msmain(SR=SR, ppg_data='', peak_times=peak_times_tw)
        mssave2.append([SDNN, RMSSD, pNN50, BPM])
    mssave2 = np.array(mssave2)

    titles = ['SDNN', 'RMSSD', 'pNN50', 'BPM']
    if False:
        for fn in range(4):
            plt.figure()
            plt.title(nlist[i] + '_' + titles[fn])
            plt.plot(mssave2[:,fn])
            if fn in [0, 1]: plt.ylim([0,400])
            if fn in [2, 3]: plt.ylim([50,90])
            
            #%
        
    # tmp = peak_times
    # peak_times = np.where(np.sum(tmp, axis=0) > 0)[0]
    # peak_times = peak_times/SR*1000 # 단위 ms로
    
    SDNN, RMSSD, pNN50, BPM, _ = VNS_PPG_def.msmain(SR=SR, ppg_data='', peak_times=peak_times)
    print('SDNN, RMSSD, pNN50, BPM', np.round([SDNN, RMSSD, pNN50, BPM], 1))
    
        

            
        
    
    
 



























