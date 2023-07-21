import numpy as np
import tensorflow as tf
import tensorboard

from datetime import datetime
from sklearn.model_selection import KFold, train_test_split

import my_model
from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline.normalize import normalize_ecg

tf.config.set_visible_devices([], 'GPU')

# load data
FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
signals, labels = load_dataset_PAF(FILE_DIRECTORY)

# data normalization into range [0.0, 1.0] to filter for signal clipping
signals_norm = normalize_ecg(signals)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels = delete_clipped_segments(signals, labels, clipped_segments)
print(np.shape(signals))
print(np.shape(labels))

# Butterworth filter
LOWCUT = 0.3
HIGHCUT = 50
FREQUENCY_HERTZ = 128
ORDER = 5

signals[:,:,0] = butter_bandpass_filter(signals[:,:,0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
signals[:,:,1] = butter_bandpass_filter(signals[:,:,1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

# final normalization
signals = normalize_ecg(signals)

print(np.shape(signals))
print(np.shape(labels))


# set up k-fold cross validation 
#kfold = KFold(n_splits=5, shuffle=True, random_state=42)



# split data into train and test sets randomly
train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
    signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=21
    )


# transform it into tesorflow dataset
train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))
test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), test_labels))

train_data = train_data.batch(100)
test_data = test_data.batch(1)

model = my_model.build_model_without_tuner_2()

# Tensorboard callback
log_dir = r'/media/jonas/SSD_new/CMS/Semester_4/research_project/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

model.fit(train_data,
          epochs = 2,
          validation_data = test_data,
          callbacks=tensorboard_callback
          )

del model