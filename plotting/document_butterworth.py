import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb

from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter

FREQUENCY_HERTZ = 128.0

# set filter parameters
LOWCUT = 0.3
HIGHCUT = 50.0
ORDER = 5   # value 8 taken from paper (but might be too high for my usecase)




DS_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/'

signals, labels, patient_number_array = load_dataset_PAF(DS_DIRECTORY)



signals_filtered = np.empty_like(signals)
signals_filtered[:,:,0] = butter_bandpass_filter(signals[:,:,0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
signals_filtered[:,:,1] = butter_bandpass_filter(signals[:,:,1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

print(np.shape(signals))
print(np.shape(labels))
print(np.shape(patient_number_array))

########################################################PLOTTING#######################################################################
PLOTTING = True

if PLOTTING:
    # set record parameters
    RECORD_DURATION_SECONDS = 10
    FREQUENCY_HERTZ = 128.0

    # set filter parameters
    LOWCUT = 0.3
    HIGHCUT = 50.0
    ORDER = 5   # value 8 taken from paper (but might be too high for my usecase)


    # from here only for plotting the signal
    nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
    t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

    signal_number = 360
    print('Patient ID:')
    print(patient_number_array[signal_number] + 1)
    print(labels[signal_number])

    used_signal = signals[signal_number,:,:]
    filtered_signal = signals_filtered[signal_number,:,:]
    print(np.shape(used_signal[:,0]))
    #used_signal_temp = used_signal[np.newaxis,:,:]

    #filtered_signal = np.empty_like(used_signal)
    #filtered_signal[:, 0] = butter_bandpass_filter(used_signal_temp[:, :, 0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
    #filtered_signal[:, 1] = butter_bandpass_filter(used_signal_temp[:, :, 1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

    fig, axs = plt.subplots(2)
    axs[0].plot(t, used_signal[:,0], label='Original signal')
    axs[0].plot(t, filtered_signal[:,0], label='Filtered signal')
    axs[0].set_title('lead 0', loc='left')
    axs[0].set_xlabel('time [seconds]')
    axs[0].set_ylabel('voltage [mV]')
    #axs[0].legend(loc='upper left')
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)


    axs[1].plot(t, used_signal[:,1], label='Original signal')
    axs[1].plot(t, filtered_signal[:,1], label='Filtered signal')
    axs[1].set_title('lead 1', loc='left')
    axs[1].set_xlabel('time [seconds]')
    axs[1].set_ylabel('voltage [mV]')
    #axs[1].legend(loc='upper left')

    #fig.legend(handles=[] loc='upper center')
    plt.tight_layout()
    plt.show()

else:
    pass


########################################################PLOTTING#######################################################################