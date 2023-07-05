import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from load_dataset_without_filter import load_dataset_PAF
from butterworth_filter import butter_bandpass_filter, butter_bandpass
from filter_clipped_segments import clipping_filter_normalized_signal_sliding_window

def normalize_ecg(raw_signals):
    return normalize(raw_signals, norm='max')


file_directory = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
signals, labels = load_dataset_PAF(file_directory)

print(np.shape(signals))

# data normalization into range [-1.0, 1.0]
signals[:,:,0] = normalize(signals[:,:,0], norm='max', axis=1)
signals[:,:,1] = normalize(signals[:,:,1], norm='max', axis=1)

print(signals.max())
print(signals.min())

# deleting 10 sec segments which contain signal clipping
signals, labels = clipping_filter_normalized_signal_sliding_window(signals, labels)

print(np.shape(signals))
print(np.shape(labels))



train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
        signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=21
        )

# set record parameters
RECORD_DURATION_SECONDS = 10
FREQUENCY_HERTZ = 128.0

# set filter parameters
lowcut = 0.3
highcut = 50.0
order = 5   # value 8 taken from paper (but might be too high for my usecase)

train_data_1 = butter_bandpass_filter(train_data_1, lowcut=lowcut, highcut=highcut, FREQUENCY_HERTZ=FREQUENCY_HERTZ, order=order)
train_data_2 = butter_bandpass_filter(train_data_2, lowcut=lowcut, highcut=highcut, FREQUENCY_HERTZ=FREQUENCY_HERTZ, order=order)
test_data_1 = butter_bandpass_filter(test_data_1, lowcut=lowcut, highcut=highcut, FREQUENCY_HERTZ=FREQUENCY_HERTZ, order=order)
test_data_2 = butter_bandpass_filter(test_data_2, lowcut=lowcut, highcut=highcut, FREQUENCY_HERTZ=FREQUENCY_HERTZ, order=order)

print(np.shape(train_data_1))
print(np.shape(train_data_2))
print(np.shape(test_data_1))
print(np.shape(test_data_2))
print(np.shape(train_labels))
print(np.shape(test_labels))

#train_data_1 = normalize(train_data_1, norm='max')
#train_data_2 = normalize(train_data_2, norm='max')
#test_data_1 = normalize(test_data_1, norm='max')
#test_data_2 = normalize(test_data_2, norm='max')

#print(np.shape(train_data_1))
#print(np.shape(train_data_2))
#print(np.shape(test_data_1))
#print(np.shape(test_data_2))
#print(np.shape(train_labels))
#print(np.shape(test_labels))

#train_data_1, train_data_2, train_labels = clipping_filter_normalized_signal(train_data_1, train_data_2, train_labels)



# from here only for plotting the signal
nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

x = train_data_1[0, :]


#plt.figure(2)
fig, axs = plt.subplots(2)
#plt.clf()
axs[0].plot(t, x)
axs[0].set_title('ECG 0')

y = train_data_2[0, :]

axs[1].plot(t, y)
axs[1].set_title('ECG 1')
#axs[1].xlabel('time (seconds)')
#axs[1].grid(True)
#axs[1].axis('tight')
#axs[1].legend(loc='upper left')

plt.show()