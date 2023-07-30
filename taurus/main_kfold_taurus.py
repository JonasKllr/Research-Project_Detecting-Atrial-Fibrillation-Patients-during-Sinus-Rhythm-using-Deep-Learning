import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from datetime import datetime
from sklearn.model_selection import KFold

import my_model_taurus
from data_pipeline_taurus.load_dataset_without_filter import load_dataset_PAF
from data_pipeline_taurus.filter_butterworth import butter_bandpass_filter
from data_pipeline_taurus.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline_taurus.normalize import normalize_ecg

# force tensorflow to use CPU (for my laptop)
#tf.config.set_visible_devices([], 'GPU')

# load data
FILE_DIRECTORY = "/beegfs/ws/0/joke793c-research_project/data_sets/paf/paf_pred_chall/cleaned"
signals, labels = load_dataset_PAF(FILE_DIRECTORY)

# data normalization into range [0.0, 1.0] to filter for signal clipping
signals_norm = normalize_ecg(signals)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels = delete_clipped_segments(signals, labels, clipped_segments)

# Butterworth filter
LOWCUT = 0.3
HIGHCUT = 50
FREQUENCY_HERTZ = 128
ORDER = 5

signals[:,:,0] = butter_bandpass_filter(signals[:,:,0], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
signals[:,:,1] = butter_bandpass_filter(signals[:,:,1], LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)

# final normalization
signals = normalize_ecg(signals)


# Hyperparameters
LEARNING_RATE = [1e-2, 1e-3, 1e-4]
KERNEL_SIZE = [3, 6, 9, 12]
POOLING_LAYER = ['max_pool', 'avg_pool']
MODEL_ARCHITECTURE = ['Model_1', 'Model_2', 'Model_3', 'Model_4']





# k-fold cross validation split
kfold = KFold(n_splits=5, shuffle=True, random_state=21)

fold_number = 1
for train_index, test_index in kfold.split(signals):
    
    # split data into folds
    train_data_1 = signals[train_index,:,0]
    train_data_2 = signals[train_index,:,1]
    train_labels = labels[train_index]

    test_data_1 = signals[test_index,:,0]
    test_data_2 = signals[test_index,:,1]
    test_labels = labels[test_index]

    # transform it into tesorflow dataset
    train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))
    test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), test_labels))

    # shuffle the data
    train_data = train_data.shuffle(buffer_size=train_data.cardinality())
    test_data = test_data.shuffle(buffer_size=test_data.cardinality())
    
    # batch the data
    train_data = train_data.batch(32)
    test_data = test_data.batch(100)
    
    # chose model to train
    model = my_model_taurus.build_model_without_tuner_5()
    print(model.summary())

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_number} ...')

    # Tensorboard callback
    log_dir = '/beegfs/ws/0/joke793c-research_project/history/' + model.name + '/tensorboard/fold_num_' + str(fold_number) +'/'
    os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

    #Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)

    history = model.fit(train_data,
            epochs = 30,
            verbose=2,
            validation_data = test_data,
            callbacks=[tensorboard_callback, early_stopping]
            )

    # save history plots
    HISTORY_DIR = '/beegfs/ws/0/joke793c-research_project/history/' + model.name + '/plots/fold_num_' + str(fold_number) +'/'
    os.makedirs(HISTORY_DIR)

    # Get the training accuracy and validation accuracy
    training_accuracy = history.history['binary_accuracy']
    validation_accuracy = history.history['val_binary_accuracy']

    # Convert accuracy to percentage values
    training_accuracy_percent = [acc * 100 for acc in training_accuracy]
    validation_accuracy_percent = [acc * 100 for acc in validation_accuracy]

    num_epochs = len(history.history['loss'])
    epochs = range(1, num_epochs + 1)

    plt.plot(epochs, training_accuracy_percent)
    plt.plot(epochs, validation_accuracy_percent)
    plt.xticks(epochs)
    plt.ylim(0)
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epoch [-]')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid(True)
    #plt.show()
    plt.savefig(HISTORY_DIR + 'acc.png')
    plt.clf()

    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.xticks(epochs)
    plt.ylim(0)
    plt.ylabel('Loss [-]')
    plt.xlabel('Epoch [-]')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid(True)
    #plt.show()
    plt.savefig(HISTORY_DIR + 'loss.png')
    plt.clf()

    del model
    del history

    fold_number += 1