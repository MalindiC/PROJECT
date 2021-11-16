# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:23:04 2021

@author: cmbbd
"""

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join


FEATURES = np.zeros((1000,80))
jcout = -1

for j in ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
       'metal', 'pop', 'reggae', 'rock']:
    jcout+=1
    for i in range(0,100):
        if (i<10):
            filename = j+'.0000'+str(i)+'.wav'
        else:
            filename = j+'.000'+str(i)+'.wav'
        dirpath = 'Data/genres_original/'
        dirpath +=j + '/'
        filepath =dirpath + filename
        signal, sr = librosa.load(filepath)
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=20, sr=sr)
        delta = librosa.feature.delta(mfccs)
        FEATURES[i+(jcout*100),[2*k for k in range(0,20)]]=np.mean(mfccs,axis=1)
        FEATURES[i+(jcout*100), [2*k+1 for k in range(0,20)]]=np.var(mfccs,axis=1)        
        FEATURES[i+(jcout*100),[2*k +40 for k in range(0,20)]]=np.mean(delta,axis=1)
        FEATURES[i+(jcout*100), [2*k+1 +40 for k in range(0,20)]]=np.var(delta,axis=1) 
