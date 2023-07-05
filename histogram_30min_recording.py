import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from sklearn.model_selection import train_test_split

from butterworth_filter import butter_bandpass_filter, butter_bandpass


DS_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/'
signal = wfdb.rdsamp(DS_DIRECTORY + 'n01')[0]

# fix ouliers in the last sample
LAST_SAMPLE = 230399
signal[LAST_SAMPLE] = signal[LAST_SAMPLE-3]
signal[LAST_SAMPLE-1] = signal[LAST_SAMPLE-3]
signal[LAST_SAMPLE-2] = signal[LAST_SAMPLE-3]


# histogram
signals_added, bins = np.histogram(signal, bins=500)
#print(signals_added.min())
#print(signals_added.max())

print(signal.min())
print(signal.max())


plt.bar(bins[:-1], signals_added, width=np.diff(bins))
plt.ylim(0, 10)
plt.show()


# from here only for plotting the signal
RECORD_DURATION_SECONDS = 1800
FREQUENCY_HERTZ = 128
nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

x = signal


#plt.figure(2)
#plt.clf()
#plt.plot(t, x[:, 0], label='Original signal')

#y = butter_bandpass_filter(x, lowcut, highcut, FREQUENCY_HERTZ, order)
#print(np.shape(y))
#print(type(y))

#plt.plot(t, x[:, 1], label='Filtered signal')
#plt.xlabel('time (seconds)')
#plt.grid(True)
#plt.axis('tight')
#plt.legend(loc='upper left')

#SAFE_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/figures/paf_pred_challenge/'
#plt.savefig(SAFE_DIRECTORY + 'n01_filtered.png')

#plt.figure(2)
fig, axs = plt.subplots(2)
#plt.clf()
axs[0].plot(t, signal[:, 0])
axs[0].set_title('ECG 0')
axs[0].grid(True)
axs[0].set_xlim(1265, 1277)
axs[0].set_ylim(-1.1, 1.1)


axs[1].plot(t, signal[:, 1])
axs[1].set_title('ECG 1')
#axs[1].xlabel('time (seconds)')
axs[1].grid(True)
axs[1].set_xlim(1265, 1277)
axs[1].set_ylim(-1.1, 1.1)
#axs[1].axis('tight')
#axs[1].legend(loc='upper left')

plt.tight_layout()
plt.show()