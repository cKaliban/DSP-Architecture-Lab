import numpy as np
from scipy import signal
import sklearn as sk
import matplotlib.pyplot as plt
import librosa

src_signal = librosa.load("LAB5_500HzFHR.wav", sr=None)

sample_rate = src_signal[1]
source_signal = src_signal[0]

N = source_signal.size

Ts = 1.0 / sample_rate

X = source_signal[:N-1].reshape(-1, 2)
# print(X)

b, a =  signal.iirdesign(2, 4, gpass=1, gstop=40, ftype='butter', fs=sample_rate)
signal_filtered = signal.lfilter(b, a, source_signal)
X_p = signal_filtered[:N-1].reshape(-1, 2)

ica = sk.decomposition.FastICA(n_components=2)
S_ = ica.fit_transform(X_p)
A_ = ica.mixing_
print(A_)

S1 = np.dot(S_[:, 0].T, A_[0, :])
print(S1)

plt.plot(S_)
plt.show()
# W, S = sk.decomposition.fastica(X, n_components=2, whiten=True)

