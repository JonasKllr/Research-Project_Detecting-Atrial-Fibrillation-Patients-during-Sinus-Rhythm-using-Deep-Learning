# code for filter taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb

from scipy.signal import butter, lfilter, freqz


def butter_bandpass(lowcut, highcut, FREQUENCY, order):
    nyq = 0.5 * FREQUENCY
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, FREQUENCY, order):
    b, a = butter_bandpass(lowcut, highcut, FREQUENCY, order=order)
    y = lfilter(b, a, data)
    return y


# Sample rate and desired cutoff frequencies (in Hz).
FREQUENCY = 128.0
lowcut = 10.0
highcut = 60.0

# filter order
order = 8   # value taken from paper


# Filter a noisy signal.
DURATION_SECONDS = 10.0
nsamples = int(DURATION_SECONDS * FREQUENCY)
t = np.linspace(0, DURATION_SECONDS, nsamples, endpoint=False)

record = wfdb.rdrecord('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/1.0.0/n02', sampto=1280, channels=[1])
x = record.p_signal


plt.figure(2)
plt.clf()
plt.plot(t, x, label='Original signal')

y = butter_bandpass_filter(x, lowcut, highcut, FREQUENCY, order=order)
print(np.shape(y))

plt.plot(t, y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()