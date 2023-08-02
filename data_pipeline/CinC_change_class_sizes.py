import matplotlib.pyplot as plt
import numpy as np

dir = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'
age_file = dir + 'age_array.txt'

age = np.loadtxt(age_file)
labels = np.empty(0)


for i in range(len(age)):

    print(i)
    if age[i] in range(0,18):
        labels = np.append(labels, 0)
    elif age[i] in range(18,35):
        labels = np.append(labels, 1)
    elif age[i] in range(35,53):
        labels = np.append(labels, 2)
    elif age[i] in range(53,71):
        labels = np.append(labels, 3)
    elif age[i] in range(71,90):
        labels = np.append(labels, 4)
    
    else:
        pass


print(labels)

#np.save(dir + 'CinC_labels_5.npy', labels, allow_pickle=False)