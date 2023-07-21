import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.normalize import normalize_ecg

file_directory = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
signals, labels = load_dataset_PAF(file_directory)

signals_norm = normalize_ecg(signals)



SEGMENT_NUMBER = 25

# set record parameters
RECORD_DURATION_SECONDS = 10
FREQUENCY_HERTZ = 128.0


# from here only for plotting the signal
nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

x = signals[SEGMENT_NUMBER,:, 0]


#plt.figure(2)
fig, axs = plt.subplots(2)

#plot original signals
#axs[0].plot(t, x)
#axs[0].set_title('ECG 0')
#
#y = signals[SEGMENT_NUMBER,:, 1]
#
#axs[1].plot(t, y)
#axs[1].set_title('ECG 1')
#axs[1].xlabel('time (seconds)')
#axs[1].grid(True)
#axs[1].axis('tight')
#axs[1].legend(loc='upper left')


# plot normalized signals
z = signals_norm[SEGMENT_NUMBER, :, 0]
#
axs[0].plot(t, z)
axs[0].set_title('lead 0 normalized', loc='left')
axs[0].set_xlabel('seconds')
#
w = signals_norm[SEGMENT_NUMBER, :, 1]
#
axs[1].plot(t, w)
axs[1].set_title('lead 1 normalized', loc='left')
axs[1].set_xlabel('seconds')

#axs.set_xlabel('seconds')

plt.tight_layout()
plt.show()