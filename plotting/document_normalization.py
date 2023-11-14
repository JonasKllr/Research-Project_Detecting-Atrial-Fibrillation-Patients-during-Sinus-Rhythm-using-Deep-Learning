import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline.normalize import normalize_ecg

# force tensorflow to use CPU (for my laptop)
tf.config.set_visible_devices([], 'GPU')




FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/validation_split"
#FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned"
signals, labels, patient_number_array = load_dataset_PAF(FILE_DIRECTORY)
print(np.shape(signals))

# data normalization into range [0.0, 1.0] to filter for signal clipping
signals_norm = normalize_ecg(signals)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels, patient_number_array = delete_clipped_segments(signals, labels, patient_number_array, clipped_segments)
patient_number_array_unique = np.unique(patient_number_array, return_index=False)

print(np.shape(signals))
print(np.shape(labels))
print(np.shape(patient_number_array))
print(np.shape(patient_number_array_unique))

# Butterworth filter
LOWCUT = 0.3
HIGHCUT = 50
FREQUENCY_HERTZ = 128
ORDER = 5

signals[:,:,0] = butter_bandpass_filter(signals[:,:,0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
signals[:,:,1] = butter_bandpass_filter(signals[:,:,1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

# final normalization
signals = normalize_ecg(signals)





########################################################PLOTTING#######################################################################
PLOTTING = True

if PLOTTING:
    # set record parameters
    RECORD_DURATION_SECONDS = 10
    FREQUENCY_HERTZ = 128.0


    # from here only for plotting the signal
    nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
    t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

    #plt.figure(2)
    fig, axs = plt.subplots(2)

    # plot normalized signals
    signal_number = 0
    print('Patient ID:')
    print(patient_number_array[signal_number])

    print(labels[signal_number])

    z = signals[signal_number, :, 0]
    axs[0].plot(t, z)
    axs[0].set_title('lead 0', loc='left')
    axs[0].set_xlabel('time [seconds]')
    axs[0].set_ylabel('normalized amplitude [-]')

    w = signals[signal_number, :, 1]
    axs[1].plot(t, w)
    axs[1].set_title('lead 1', loc='left')
    axs[1].set_xlabel('time [seconds]')
    axs[1].set_ylabel('normalized amplitude [-]')


    plt.tight_layout()
    plt.show()

else:
    pass




########################################################PLOTTING#######################################################################



