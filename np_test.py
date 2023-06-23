import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from sklearn.model_selection import train_test_split

file_directory = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/n01"
n_records = 18000

signals = np.empty((0, 1280, 2))
labels = np.empty(n_records)

signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1800*128)[0]
signals_temp = np.split(signals_temp, indices_or_sections=180)

print(np.shape(signals_temp))

signals = np.append(signals, signals_temp, axis=0)

print(np.shape(signals))

print(signals[0, :, :])