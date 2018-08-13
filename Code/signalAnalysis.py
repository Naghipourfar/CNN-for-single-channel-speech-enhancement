"""
    Created by Mohsen Naghipourfar on 8/12/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

import sys
import wave

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

audio_path = '../Data/TIMIT/TRAIN/DR1/FCJF0/SA1.wav'
noise_path = '../Data/Noise/train/noise-1.wav'
# spf = wave.open(audio_path, 'r')
audio_signal, audio_sr = librosa.load(audio_path, sr=None)
noise_signal, noise_sr = librosa.load(noise_path, sr=None)

print(audio_sr)
print(noise_sr)

# Extract Raw Audio from Wav File
# signal = spf.readframes(-1)
# signal = np.fromstring(signal, 'Int16')

# If Stereo
# if spf.getnchannels() == 2:
#     print('Just mono files')
#     sys.exit(0)

plt.close("all")
plt.figure(1)
plt.title('Signal Wave...')
plt.plot(noise_signal[2300:2355], label='Noise')
# plt.plot(audio_signal, label='Real speech')

plt.legend()
plt.show()


print(pd.DataFrame(noise_signal[2300:2355]))