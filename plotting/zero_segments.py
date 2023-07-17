import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.load_dataset_without_filter import load_dataset_PAF


file_directory = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
signals, labels = load_dataset_PAF(file_directory)

file_name = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/figures/paf_pred_challenge/deleted_segments/np_arrays/normalized_with_nan_1.txt'

normalized_signals = np.loadtxt(file_name)
print(np.shape(normalized_signals))
print(normalized_signals)

#rows_with_nan = np.where((np.any(nan)))
#rows_with_nan = np.where(np.isnan(normalized_signals).any(axis=0))
#rows_with_nan = np.where((np.any(np.isnan(normalized_signals))))[0]
rows_with_nan = np.argwhere(np.isnan(normalized_signals).any(axis=1))

print(rows_with_nan)

# set record parameters
RECORD_DURATION_SECONDS = 10
FREQUENCY_HERTZ = 128.0


# from here only for plotting the signal
nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

x = signals[2880,:, 0]


#plt.figure(2)
fig, axs = plt.subplots(2)
#plt.clf()
axs[0].plot(t, x)
axs[0].set_title('ECG 0')

y = signals[2880,:, 1]

axs[1].plot(t, y)
axs[1].set_title('ECG 1')
#axs[1].xlabel('time (seconds)')
#axs[1].grid(True)
#axs[1].axis('tight')
#axs[1].legend(loc='upper left')

plt.show()