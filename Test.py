import wave

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

filename = 'train_00496.wav'

wf = wave.open(filename, 'r')
RATE = wf.getframerate()

y, sr = librosa.load(filename, sr=RATE)
devel=y
hop_length = 256
n_fft=1024
n_mels=256
'''plt.subplot(1,2,1)
D = np.abs(librosa.stft(devel, n_fft=n_fft,
                        hop_length=hop_length))
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')'''
plt.subplot(1,1,1)
S = librosa.feature.melspectrogram(devel, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')


plt.show()