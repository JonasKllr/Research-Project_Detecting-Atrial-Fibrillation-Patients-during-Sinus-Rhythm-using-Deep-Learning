import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb

from data_pipeline.load_dataset_without_filter import load_dataset_PAF






DS_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/'

signals, labels, patient_number_array = load_dataset_PAF(DS_DIRECTORY)

print(np.shape(signals))
print(np.shape(labels))
print(np.shape(patient_number_array))

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
    signal_number = 504
    print('Patient ID:')
    print(patient_number_array[signal_number] + 1)

    print(labels[signal_number])

    z = signals[signal_number, :, 0]
    axs[0].plot(t, z)
    axs[0].set_title('lead 0', loc='left')
    axs[0].set_xlabel('time [seconds]')
    axs[0].set_ylabel('voltage [mV]')

    w = signals[signal_number, :, 1]
    axs[1].plot(t, w)
    axs[1].set_title('lead 1', loc='left')
    axs[1].set_xlabel('time [seconds]')
    axs[1].set_ylabel('voltage [mV]')


    plt.tight_layout()
    plt.show()

else:
    pass




########################################################PLOTTING#######################################################################