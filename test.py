#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.io.wavfile
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
GENRE_DIR = "genre"

def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)

def read_fft(genre_list, base_dir=GENRE_DIR):
    x = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        for fn in file_list:
            fft_features = np.load(fn)
            x.append(fft_features[:1000])
            y.append(label)

    return np.array(X), np.array(y)
# specgram(x, Fs=sample_rate, xextent=(0,30))

def plot_confugion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True Class')
    pylab.grid(False)
    pylab.show()

if __name__ == "__main__":
    # rie = "voice_sample/k_rie_ch1.wav"
    # ikue = "voice_sample/o_ikue_ch1.wav"
    # ayana = "voice_sample/t_ayana_ch1.wav"
    # create_fft(rie)
    # create_fft(ikue)
    # create_fft(ayana)

    # x, y = read_fft(fft)
    # cm = confusion_matrix(x, y)
    g_list = ["a", "b", "c"]
    cm = [[1 2 3][2 3 4][3 4 5]]
    plot_confugion_matrix(cm, g_list, "name", "title")


    # cm = confusion_matrix()
