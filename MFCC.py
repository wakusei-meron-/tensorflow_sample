#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scikits.talkbox.features import mfcc
import scipy
from scipy import io
from scipy.io import wavfile
import glob
import numpy as np
import os
import wave
from pylab import *
import pyaudio

# import scipy
# from scipy import io
from scipy.io import wavfile
# import glob
# import numpy as np
# import os

BASE_DIR = "/Users/a13887/Desktop/PPP/VoiceProject/voice_sample"

def write_ceps(ceps, fn):
    base_fn, ext = os.path.splintext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)

def create_ceps(fn):
    sample_rate, X = io.wavefile.read(fn)
    ceps, mspec, spec = mfcc(X)
    isNan = False
    for num in ceps:
        if np.isnan(num[1]):
            isNan = True
    if isNan == False:
        write_ceps(ceps, fn)


def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print "チャンネル数:", wf.getnchannels()
    print "サンプル幅:", wf.getsampwidth()
    print "サンプリング周波数:", wf.getframerate()
    print "フレーム数:", wf.getnframes()
    print "パラメータ:", wf.getparams()
    print "長さ（秒）:", float(wf.getnframes()) / wf.getframerate()

def listen_wave(filename):
    wf = wave.open(filename, "r")
    printWaveInfo(wf)

    # ストリームを開く
    # p = pyaudio.PyAudio()
    # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                 channels=wf.getnchannels(),
    #                 rate=wf.getframerate(),
    #                 output=True)
    #
    # # チャンク単位でストリームに出力し音声を再生
    # chunk = 1024
    # data = wf.readframes(chunk)
    # while data != '':
    #     stream.write(data)
    #     data = wf.readframes(chunk)
    # stream.close()
    # p.terminate()

def read_ceps(name_list, base_dir = BASE_DIR):
    X, y = [], []
    for label, name in enumerate(name_list):
        # print label, name
        # print np.load(glob.glob(os.path.join(base_dir, name + ".wav")))
        # for fn in glob.glob(os.path.join(base_dir, name + ".wav")):
        #     print fn
        ceps = np.load(name)
            # ceps = wavfile.read(fn)
            # num_ceps = len(ceps)
            # X.append(np.mean(ceps[:], axis = 0))
            # y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # name_list = ["a.wav"]#, "taketatsu_ayana", "otani_ikue"]
    # x, y = read_ceps(name_list)
    listen_wave("voice_sample/k_rie.wav")
    # listen_wave("a.wav")
#     print "start"
#     wav, fs = wav_read("a.wav")
#     # wav, fs = wav_read("voice_sample/k_rie.wav")
#     t = np.arange(0.0, len(wav) / fs, 1 / fs)
#
#     center = len(wav) / 2
#     cuttime = 0.04
#     wavdata = wav[center - cuttime/2*fs : center + cuttime/2*fs]
#     time = t[center - cuttime/2*fs : center + cuttime/2*fs]
#
#     plot(time * 1000, wavdata)
#     xlabel("time [ms]")
#     ylabel("amplitude")
#     # savefig("waveform.png")
#     show()
#     # print "aiueo"
