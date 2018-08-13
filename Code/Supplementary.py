import fnmatch
import os
import librosa
import numpy as np
import pandas as pd
"""
    Created by Mohsen Naghipourfar on 8/11/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


val_loss = np.load("../val_loss2.npy")
print(pd.DataFrame(val_loss))
# plt.plot(np.arange(0, len(val_loss)), val_loss)
# plt.show()

# def fix_all_voices(directory):
#     files = []
#     pattern = ['*.WAV', '*.wav']
#     for root, dirs, filenames in os.walk(directory):
#         for filename in fnmatch.filter(filenames, pattern[0]):
#             files.append(root + '/' + filename)
#     for file in files:
#         print(file)
#         audio_org, sr = librosa.load(file, sr=None)
#         print(file[:-3] + "wav")
#         librosa.output.write_wav(file[:-3] + "wav", audio_org, sr=sr)


# fix_all_voices("../Data/TIMIT/TRAIN/")
