#!/usr/bin/python
import sys
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import sklearn as sk
import librosa 
import soundfile as sf
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("HelloWorld")

src_signal = librosa.load("LAB5_500HzFHR.wav", sr=None)

sample_rate = src_signal[1]
source_signal = src_signal[0]

N = source_signal.size

Ts = 1.0 / sample_rate
fft_src = fft(source_signal)
yf = 2.0/N * np.abs(fft_src[:N//2])
x = np.linspace(0.0, N*Ts, N)
xf = np.linspace(0.0, 1.0/(2.0*Ts), N//2)

# W, S = sk.decomposition.fastica((source_signal, source_signal), n_components=2, whiten=False)
# print(W)
# print(S)



b, a =  signal.iirdesign(2, 4, gpass=1, gstop=40, ftype='butter', fs=sample_rate)
signal_filtered = signal.lfilter(b, a, source_signal)


fft_flt = fft(signal_filtered)
print(b)
print(a)
plt.plot(fft_flt)
plt.show()

plt.plot(x, signal_filtered)
plt.show()

sf.write("test.wav", signal_filtered.astype(np.int16), sample_rate)