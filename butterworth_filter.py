# code for filter taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb

from scipy.signal import butter, lfilter, freqz


def butter_bandpass(lowcut, highcut, FREQUENCY_HERTZ, order):
    nyq = 0.5 * FREQUENCY_HERTZ
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, FREQUENCY_HERTZ, order):
    b, a = butter_bandpass(lowcut, highcut, FREQUENCY_HERTZ, order=order)
    return lfilter(b, a, data)


# set record parameters
RECORD_DURATION_SECONDS = 10
FREQUENCY_HERTZ = 128.0

# set filter parameters
lowcut = 10.0
highcut = 60.0
order = 5   # value 8 taken from paper (but might be too high for my usecase)

#set path 
ONE_SECOND = 128
record = wfdb.rdrecord('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/n02',sampfrom=int(0), sampto=int(RECORD_DURATION_SECONDS*ONE_SECOND), channels=[0, 1])



# from here only for plotting the signal
nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

x = record.p_signal


plt.figure(2)
plt.clf()
plt.plot(t, x, label='Original signal')

y = butter_bandpass_filter(x, lowcut, highcut, FREQUENCY_HERTZ, order=order)
print(np.shape(y))
print(type(y))

plt.plot(t, y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()