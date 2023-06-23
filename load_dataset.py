import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from sklearn.model_selection import train_test_split


# load data from PAF prediction challenge
def load_dataset_PAF(directory, n_records=18000):

    signals = np.empty((n_records, 1280, 2))
    labels = np.empty(n_records)

    record_count = 0
    for filename in sorted(os.listdir(DIRECTORY)):
        
        # only do data integration once per record
        if filename.endswith('.dat'):
            
            # for files named "pXX" label = 1 (AF group)
            if (os.path.basename(filename)[0] == 'p'):
                filename_without_ext = os.path.splitext(filename)[0]

                file_directory = DIRECTORY + os.sep + filename_without_ext
                signals[record_count, :, :] = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1280)[0]
                labels[record_count] = 1

                record_count += 1

            # for files named "pXX" label = 0 (non AF group)
            elif (os.path.basename(filename)[0] == 'n'):
                filename_without_ext = os.path.splitext(filename)[0]

                file_directory = DIRECTORY + os.sep + filename_without_ext
                signals[record_count, :, :] = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1280)[0]
                labels[record_count] = 0

                record_count += 1

            else:
                pass

    return signals, labels




# TESTS

DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"

signals, labels = load_dataset_PAF(DIRECTORY, n_records=100)

train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
    signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=21
    )

print(np.shape(train_data_1))
print(np.shape(test_labels))
print(train_data_2[1,:])



# normalize data
min_val = tf.reduce_min(signals)
max_val = tf.reduce_max(signals)

#train_data = (train_data - min_val) / (max_val - min_val)
#test_data = (test_data - min_val) / (max_val - min_val)



train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))


print(train_data.element_spec)
print(list(train_data.as_numpy_iterator())[0])

#plt.grid()
#plt.plot(np.arange(1280), train_data_1[0])
#plt.title("A Normal ECG")
#plt.show()


BATCH_SIZE = 1
train_data = train_data.batch(BATCH_SIZE)
#test_data = test_data.batch(1)