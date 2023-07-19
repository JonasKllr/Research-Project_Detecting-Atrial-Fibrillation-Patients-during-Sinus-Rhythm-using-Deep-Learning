import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline.normalize import normalize_ecg



# load data
FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
signals, labels = load_dataset_PAF(FILE_DIRECTORY)

# data normalization into range [0.0, 1.0]
signals_norm = normalize_ecg(signals)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels = delete_clipped_segments(signals, labels, clipped_segments)
print(np.shape(signals))
print(np.shape(labels))

# Butterworth filter
LOWCUT = 0.3
HIGHCUT = 50
FREQUENCY_HERTZ = 128
ORDER = 5

signals[:,:,0] = butter_bandpass_filter(signals[:,:,0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
signals[:,:,1] = butter_bandpass_filter(signals[:,:,1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

# final normalization
signals = normalize_ecg(signals)

print(np.shape(signals))
print(np.shape(labels))


train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
        signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=21
        )

print(np.shape(train_data_1))
print(np.shape(train_data_2))
print(np.shape(test_data_1))
print(np.shape(test_data_2))
print(np.shape(train_labels))
print(np.shape(test_labels))


# set record parameters
RECORD_DURATION_SECONDS = 10
FREQUENCY_HERTZ = 128.0

# from here only for plotting the signal
SEGMENT_NUMBER = 5

print(train_labels[SEGMENT_NUMBER])

nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

fig, axs = plt.subplots(2)

# plot normalized signals
z = train_data_1[SEGMENT_NUMBER, :]
axs[0].plot(t, z)
axs[0].set_title('lead 0 normalized', loc='left')
axs[0].set_xlabel('seconds')

w = train_data_2[SEGMENT_NUMBER, :]
axs[1].plot(t, w)
axs[1].set_title('lead 1 normalized', loc='left')
axs[1].set_xlabel('seconds')


plt.tight_layout()
plt.show()