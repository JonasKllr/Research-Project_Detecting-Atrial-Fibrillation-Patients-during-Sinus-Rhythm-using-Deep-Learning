import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from data_pipeline.filter_butterworth import butter_bandpass_filter, butter_bandpass

from sklearn.model_selection import train_test_split


# load data from PAF prediction challenge
def load_dataset_PAF(DIRECTORY, lowcut, highcut, FREQUENCY_HERTZ, order):

    signals = np.empty((0, 1280, 2))
    labels = np.empty(0)

    for filename in sorted(os.listdir(DIRECTORY)):
        
        # only do data integration once per record
        if filename.endswith('.dat'):
            
            # for files named "pXX" label = 1 (AF group)
            if (os.path.basename(filename)[0] == 'p'):
                filename_without_ext = os.path.splitext(filename)[0]
                file_directory = DIRECTORY + os.sep + filename_without_ext

                signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1800*128)[0]
                signals_temp = butter_bandpass_filter(signals_temp, lowcut, highcut, FREQUENCY_HERTZ, order)
                signals_temp = np.split(signals_temp, indices_or_sections=180)            
                signals = np.append(signals, signals_temp, axis=0)

                lables_temp = np.full(180, 1)   #label
                labels = np.append(labels, lables_temp)


            # for files named "pXX" label = 0 (non AF group)
            elif (os.path.basename(filename)[0] == 'n'):
                filename_without_ext = os.path.splitext(filename)[0]
                file_directory = DIRECTORY + os.sep + filename_without_ext

                signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1800*128)[0]
                signals_temp = butter_bandpass_filter(signals_temp, lowcut, highcut, FREQUENCY_HERTZ, order)
                signals_temp = np.split(signals_temp, indices_or_sections=180)            
                signals = np.append(signals, signals_temp, axis=0)

                lables_temp = np.full(180, 0)   # label
                labels = np.append(labels, lables_temp)


            else:
                pass

    return signals, labels



# TESTS
if __name__ == '__main__':
    
    DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
    lowcut = 10.0
    highcut = 60.0
    FREQUENCY_HERTZ = 128
    order = 5

    signals, labels = load_dataset_PAF(DIRECTORY, lowcut=lowcut, highcut=highcut, FREQUENCY_HERTZ=FREQUENCY_HERTZ, order=order)

    print(np.shape(signals))
    print(signals[0, :, :])
    print(np.shape(labels))
    print(labels[0])


    train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
        signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=21
        )

    print(np.shape(train_data_1))
    print(np.shape(test_labels))
    print(train_data_2[1,:])



    # normalize data
    #min_val = tf.reduce_min(signals)
    #max_val = tf.reduce_max(signals)

    min_val = signals.min()
    max_val = signals.max()

    #train_data = (train_data - min_val) / (max_val - min_val)
    #test_data = (test_data - min_val) / (max_val - min_val)



    train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))


    print(train_data.element_spec)
    print(list(train_data.as_numpy_iterator())[0])

    # from here only for plotting the signal
    RECORD_DURATION_SECONDS = 10
    nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
    t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)


    #x = list(train_data.as_numpy_iterator())[1]
    x = train_data_1[0, :]
    

    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Original signal')


    #y = butter_bandpass_filter(x, lowcut, highcut, FREQUENCY_HERTZ, order)
    #print(np.shape(y))
    #print(type(y))

    #plt.plot(t, y, label='Filtered signal')
    #plt.xlabel('time (seconds)')
    #plt.grid(True)
    #plt.axis('tight')
    #plt.legend(loc='upper left')

    plt.show()


    BATCH_SIZE = 1
    train_data = train_data.batch(BATCH_SIZE)
    #test_data = test_data.batch(1)


