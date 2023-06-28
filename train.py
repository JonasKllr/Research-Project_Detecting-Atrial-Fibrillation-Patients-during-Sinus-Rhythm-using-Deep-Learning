import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from scipy.signal import butter, lfilter, freqz
from sklearn.model_selection import train_test_split

from butterworth_filter import butter_bandpass_filter, butter_bandpass
from load_dataset import load_dataset_PAF


# turn off GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#if gpus:
#  try:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#  except RuntimeError as e:
#    print(e)

# load the data
DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
LOWCUT = 10.0
HIGHCUT = 60.0
FREQUENCY_HERTZ = 128
ORDER = 5

signals, labels = load_dataset_PAF(DIRECTORY, LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)


# split data into train and test sets randomly
train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
    signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=21
    )

# normalize data
min_val = tf.reduce_min(signals)
max_val = tf.reduce_max(signals)

#train_data_1 = (train_data_1 - min_val) / (max_val - min_val)
#train_data_2 = (train_data_2 - min_val) / (max_val - min_val)
#
#test_data_1 = (test_data_1 - min_val) / (max_val - min_val)
#test_data_2 = (test_data_2 - min_val) / (max_val - min_val)


# transform it into tesorflow dataset
train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))
test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), test_labels))


BATCH_SIZE = 10
train_data = train_data.batch(BATCH_SIZE)
test_data = test_data.batch(1)




# build CNN
RECORD_LENGTH = 1280

input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

concatenate = tf.keras.layers.concatenate([input_layer_1, input_layer_2])


x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation='relu')(concatenate)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation='relu')(x)

x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(384, activation='relu')(x)

output_layer = tf.keras.layers.Dense(1)(x)


model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()]
              )

print(model.summary())

model.fit(train_data,
          epochs = 20,
          validation_data = test_data
          )

del model