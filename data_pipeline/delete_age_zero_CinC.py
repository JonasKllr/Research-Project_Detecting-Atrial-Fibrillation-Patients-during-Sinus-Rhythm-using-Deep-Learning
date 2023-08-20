import numpy as np


def delete_age_zero_CinC(signals, labels):
    
    zero_index = np.where(labels == 0)
    signals = np.delete(signals, zero_index, axis=0)
    labels = np.delete(labels, zero_index, axis=0)

    return signals, labels

    


if __name__ == '__main__':
    # load data
    FILE_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'
    signals_temp = np.load(FILE_DIRECTORY + 'CinC_signals.npy', allow_pickle=False)
    labels_temp = np.loadtxt(FILE_DIRECTORY + 'age_array.txt')
    labels_temp = labels_temp.astype(int)
    print(np.shape(signals_temp))
    print(np.shape(labels_temp))
    print(labels_temp[19890])

    signals_temp, labels_temp = delete_age_zero_CinC(signals_temp, labels_temp)
    print(np.shape(signals_temp))
    print(np.shape(labels_temp))
    