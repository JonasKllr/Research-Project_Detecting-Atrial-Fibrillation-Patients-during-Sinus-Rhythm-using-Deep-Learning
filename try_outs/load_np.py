import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.normalize import normalize_ecg
from data_pipeline.filter_butterworth import butter_bandpass_filter

SAVE_DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'

signals = np.load(SAVE_DIR + 'CinC_signals.npy', allow_pickle=False)
labels = np.load(SAVE_DIR + 'CinC_labels_5.npy', allow_pickle=False)

print(labels)



# Butterworth filter
LOWCUT = 0.3
HIGHCUT = 50
FREQUENCY_HERTZ = 128
ORDER = 5

signals[:,:,0] = butter_bandpass_filter(signals[:,:,0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
signals[:,:,1] = butter_bandpass_filter(signals[:,:,1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

signals = normalize_ecg(signals)

# set record parameters
RECORD_DURATION_SECONDS = 10
FREQUENCY_HERTZ = 128.0


# from here only for plotting the signal
nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

#plt.figure(2)
fig, axs = plt.subplots(2)

# plot normalized signals
signal_number = 500

z = signals[signal_number, :, 0]
axs[0].plot(t, z)
axs[0].set_title('lead 0 normalized', loc='left')
axs[0].set_xlabel('seconds')

w = signals[signal_number, :, 1]
axs[1].plot(t, w)
axs[1].set_title('lead 1 normalized', loc='left')
axs[1].set_xlabel('seconds')


plt.tight_layout()
plt.show()