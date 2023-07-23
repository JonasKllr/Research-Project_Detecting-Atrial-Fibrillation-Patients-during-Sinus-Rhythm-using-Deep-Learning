import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import train_test_split

import my_model
from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline.normalize import normalize_ecg

tf.config.set_visible_devices([], 'GPU')

# load data
FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned_small/"
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

max_trials = 2

tuner = kt.RandomSearch(
    my_model.build_model_tuner,
    objective='val_loss',
    max_trials=max_trials,
    executions_per_trial=2
)




#Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)



# Custom training loop for multiple tries with different data splits
for _ in range(max_trials):
    # Get the next hyperparameter configuration from the tuner
    hyperparameters = tuner.get_state()

    # Build the model with the current hyperparameters
    model = tuner.hypermodel.build(hyperparameters.values)

    # Create multiple data splits and train the model on each split
    for _ in range(tuner.executions_per_trial):

        # split data into train and test sets randomly
        train_data_1, test_data_1, train_data_2, test_data_2, train_labels, test_labels = train_test_split(
        signals[:,:,0], signals[:,:,1], labels, test_size=0.2, random_state=None
        )

        # transform it into tesorflow dataset
        train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))
        test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), test_labels))

        train_data = train_data.batch(32)
        test_data = test_data.batch(100)

        # Tensorboard callback
        log_dir = r'/media/jonas/SSD_new/CMS/Semester_4/research_project/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

        # Train the model using the current data split
        model.fit(train_data,
                  epochs=10,
                  validation_data=test_data,
                  callbacks=[tensorboard_callback, early_stopping])

        # Report the results to the tuner
        tuner.oracle.update_trial(tuner.trial.trial_id, {'val_loss': model.history['val_loss']})


# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.get('kernels'))
print(best_hps.get('pooling'))
print(best_hps.get('learning_rate'))



