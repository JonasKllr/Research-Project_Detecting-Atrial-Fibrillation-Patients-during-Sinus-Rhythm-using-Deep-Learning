import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb

from butterworth_filter import butter_bandpass_filter, butter_bandpass


DS_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/'
SECONDS_10 = 1280


record_1 = wfdb.rdrecord(DS_DIRECTORY + 'n01')
#record_1 = wfdb.rdrecord(DS_DIRECTORY + 'n01', sampfrom=int(SECONDS_10*180 - SECONDS_10), sampto=int(SECONDS_10*180))
#record_2 = wfdb.rdrecord(DS_DIRECTORY + 'n02', sampto=int(SECONDS_10 * 2))
#record_3 = wfdb.rdrecord(DS_DIRECTORY + 'n03', sampto=int(SECONDS_10 * 2))
#record_4 = wfdb.rdrecord(DS_DIRECTORY + 'n04', sampto=int(SECONDS_10 * 2))
#record_5 = wfdb.rdrecord(DS_DIRECTORY + 'n05', sampto=int(SECONDS_10 * 2))
#record_6 = wfdb.rdrecord(DS_DIRECTORY + 'n06', sampto=int(SECONDS_10 * 2))

#save_figure = wfdb.plot_wfdb(record=record_1, return_fig=True)
wfdb.plot_wfdb(record=record_1)
#wfdb.plot_wfdb(record=record_2)
#wfdb.plot_wfdb(record=record_3)
#wfdb.plot_wfdb(record=record_4)
#wfdb.plot_wfdb(record=record_5)
#wfdb.plot_wfdb(record=record_6)


SAFE_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/figures/paf_pred_challenge/'
#save_figure.savefig(SAFE_DIRECTORY + 'n01.png')


ONE_SECOND = 128
RECORD_START = 896
RECORD_END = 898

#print(np.shape(record_1.p_signal))
print(record_1.p_signal[RECORD_START*ONE_SECOND : RECORD_END*ONE_SECOND, 0])

print(record_1.p_signal.min())