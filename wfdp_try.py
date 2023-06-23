import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath

import wfdb

record = wfdb.rdrecord('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/n01', sampto=1280) 
wfdb.plot_wfdb(record=record)
 
#signals, fields = wfdb.rdsamp('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/1.0.0/n01', channels=[0, 1])
#print(np.shape(signals))
#print(fields)

#wfdb.plot_wfdb(record=signals)