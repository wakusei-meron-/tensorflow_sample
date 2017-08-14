import MFCC
from matplotlib.pyplot import specgram
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from matplotlib import pylab
import numpy as np

name_list = ["kugimiya_rie", "taketatsu_ayana", "otani_ikue"]

x, y = MFCC.read_ceps(name_list)
print x, y
# svc = LinearSVC(C=1.0)
# x, y = resample(x, y, len(y))
# svc.fit(x[150:], y[150:])
# prediction = svc.predict(x[:150])
# cm = confusion_matrix(y[:150], prediction)
