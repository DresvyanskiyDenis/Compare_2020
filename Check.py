import sys
import wave

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import pandas as pd

path='C:\\Users\\Dresvyanskiy\\Desktop\\ComParE2020_Elderly\\wav\\'


# evaluate min size of audiofile among all audiofiles
files=os.listdir(path)
filename=''
min_duration=1.7976931348623157e+308
min_filename=''
max_duration=0
max_filename=''
for file in files:
    if os.path.isfile(file):
        wf = wave.open(path + file, 'r')
        RATE = wf.getframerate()

        y, sr = librosa.load(path+file, sr=RATE)
        if y.shape[0]<min_duration:
            min_duration=y.shape[0]
            min_filename=file
        if y.shape[0]>max_duration:
            max_duration=y.shape[0]
            max_filename=file

# create and save mel-spectrogram for each sound file
path_to_save='C:\\Users\\Dresvyanskiy\\Desktop\\ComParE2020_Elderly\\Mel_spectrogram\\'
hop_length = 256 # step per window
n_fft=1024   # size of window
n_mels=256  # num of windows

if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)

for file in files:
    if os.path.isfile(path+file):
        wf = wave.open(path + file, 'r')
        RATE = wf.getframerate()

        sound, sr = librosa.load(path + file, sr=RATE)
        sound = sound[:min_duration]
        S = librosa.feature.melspectrogram(sound, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length,
                                           n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        temp=pd.DataFrame(S_DB)
        transpose=temp.T
        transpose.to_csv(path_to_save+file.split(sep='.wav')[0]+'.csv', sep=',', header=False, index=False)
        print(transpose.shape)
