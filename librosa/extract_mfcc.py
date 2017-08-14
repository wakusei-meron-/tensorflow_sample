import librosa
import os
import numpy as np

_D = os.path.dirname(__file__)
print(_D)

# 初期値
filenames = ["fujitou_happy_001.wav", "tsuchiya_happy_001.wav"]
hop_length = 1024

for filename in filenames:
  y, sr = librosa.load(librosa.util.example_audio_file())
  mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 2, hop_length = hop_length)
  print(mfccs.shape)
  np.save('{}/mfcc/{}'.format(_D, filename), mfccs.T)


