import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

#from data_pipeline.load_dataset import load_dataset_PAF
from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter, butter_bandpass
from data_pipeline.filter_clipped_segments import clipping_filter_normalized_signal
from data_pipeline.normalize import normalize_ecg



file_directory = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
LOWCUT = 0.3
HIGHCUT = 50
FREQUENCY_HERTZ = 128
ORDER = 5

signals, labels = load_dataset_PAF(file_directory)
#signals, labels = load_dataset_PAF(file_directory, LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

print(np.shape(signals))

# data normalization into range [-1.0, 1.0]
#signals[:,:,0] = normalize(signals[:,:,0], norm='max', axis=1)
#signals[:,:,1] = normalize(signals[:,:,1], norm='max', axis=1)
#signals[:,:,0] = normalize_ecg(signals[:,:,0])
signals = normalize_ecg(signals)


print(signals.max())
print(signals.min())
print(signals)

# deleting 10 sec segments which contain signal clipping
signals, labels = clipping_filter_normalized_signal(signals, labels)

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