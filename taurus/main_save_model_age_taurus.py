import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold

import my_model_taurus
from data_pipeline_taurus.filter_butterworth import butter_bandpass_filter
from data_pipeline_taurus.filter_clipped_segments import find_clipped_segments, delete_clipped_segments_CinC
from data_pipeline_taurus.normalize import normalize_ecg
from data_pipeline_taurus.delete_age_zero_CinC import delete_age_zero_CinC

# force tensorflow to use CPU (for my laptop)
tf.config.set_visible_devices([], 'GPU')

# load data
FILE_DIRECTORY = '/beegfs/ws/0/joke793c-research_project/data_sets/cinc/'
signals_temp = np.load(FILE_DIRECTORY + 'CinC_signals.npy', allow_pickle=False)
labels_temp = np.loadtxt(FILE_DIRECTORY + 'age_array.txt')
labels_temp = labels_temp.astype(int)
print(np.shape(signals_temp))
print(np.shape(labels_temp))

signals_temp, labels_temp = delete_age_zero_CinC(signals_temp, labels_temp)
print(np.shape(signals_temp))
print(np.shape(labels_temp))



### create 10% test split ###

# Set a seed for reproducibility
seed = 42  # You can use any integer value as the seed
np.random.seed(seed)

# Calculate the size of the test set (10% of the total dataset size)
test_size = int(0.1 * len(labels_temp))

# Generate random indices for the test set
test_indices = np.random.choice(len(labels_temp), size=test_size, replace=False)

# Create boolean masks for train and test datasets
mask = np.ones(len(labels_temp), dtype=bool)
mask[test_indices] = False

signals = signals_temp[mask]
labels = labels_temp[mask]
#signals_test = signals_temp[~mask]
#labels_test = labels_temp[~mask]

del signals_temp
del labels_temp

### create 10% test split ###



# data normalization into range [0.0, 1.0] to filter for signal clipping
signals_norm = normalize_ecg(signals)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels = delete_clipped_segments_CinC(signals, labels, clipped_segments)

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

# k-fold cross validation split
kfold = KFold(n_splits=5, shuffle=True, random_state=21)


# Hyperparameters
MODEL_ARCHITECTURE = ['age_linear']
KERNEL_SIZE = [6]
POOLING_LAYER = ['max_pool']
LEARNING_RATE = [1e-2, 1e-3, 1e-4]

for model_choice in MODEL_ARCHITECTURE:
    for kernel_choice in KERNEL_SIZE:
        for pooling_choice in POOLING_LAYER:
            for learning_rate_choice in LEARNING_RATE:

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
                    if model_choice == 'Model_1':
                        model = my_model_taurus.build_model_without_tuner_1(LEARNING_RATE=learning_rate_choice, KERNEL_SIZE=kernel_choice, POOLING_LAYER=pooling_choice) 
                    elif model_choice == 'Model_2':
                        model = my_model_taurus.build_model_without_tuner_2(LEARNING_RATE=learning_rate_choice, KERNEL_SIZE=kernel_choice, POOLING_LAYER=pooling_choice)
                    elif model_choice == 'Model_3':
                        model = my_model_taurus.build_model_without_tuner_3(LEARNING_RATE=learning_rate_choice, KERNEL_SIZE=kernel_choice, POOLING_LAYER=pooling_choice)
                    elif model_choice == 'Model_4':
                        model = my_model_taurus.build_model_without_tuner_4(LEARNING_RATE=learning_rate_choice, KERNEL_SIZE=kernel_choice, POOLING_LAYER=pooling_choice)
                    elif model_choice == 'age':
                        model = my_model_taurus.build_model_age_regression(LEARNING_RATE=learning_rate_choice, KERNEL_SIZE=kernel_choice, POOLING_LAYER=pooling_choice)
                    elif model_choice == 'age_linear':
                        model = my_model_taurus.build_model_age_regression_linear(LEARNING_RATE=learning_rate_choice, KERNEL_SIZE=kernel_choice, POOLING_LAYER=pooling_choice)
                    
                    print(model.summary())


                    # Generate a print
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_number} ...')

                    # creating directory for logging
                    LOG_DIR = (f'/beegfs/ws/0/joke793c-research_project/history/' 
                               + model.name + os.path.sep
                               + f'kernel_{kernel_choice}' + os.path.sep
                               + f'pooling_' + pooling_choice + os.path.sep
                               + f'learning_rate_{learning_rate_choice}' + os.path.sep
                               + f'fold_{fold_number}')
                    os.makedirs(LOG_DIR)
                    

                    # Tensorboard callback
                    TB_DIR = LOG_DIR + '/tensorboard/'
                    os.makedirs(TB_DIR)
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TB_DIR, histogram_freq=2)

                    #Early Stopping callback
                    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)

                    # checkpoint callback
                    MODEL_DIR = LOG_DIR + '/model/'
                    os.makedirs(MODEL_DIR)
                    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=MODEL_DIR,
                        save_weights_only=False,
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True)

                    # logging callback
                    log_callback = tf.keras.callbacks.CSVLogger(LOG_DIR + '/history.csv')

                    history = model.fit(train_data,
                            epochs = 50,
                            verbose= 2,
                            validation_data = test_data,
                            callbacks= [log_callback, model_checkpoint_callback]
                            )

                    # save history plots
                    # Get the training accuracy and validation accuracy
                    training_accuracy = history.history['mean_squared_error']
                    validation_accuracy = history.history['val_mean_squared_error']

                    # Convert accuracy to percentage values
                    training_accuracy_percent = [acc * 1 for acc in training_accuracy]
                    validation_accuracy_percent = [acc * 1 for acc in validation_accuracy]

                    num_epochs = len(history.history['loss'])
                    epochs = range(1, num_epochs + 1)

                    plt.plot(epochs, training_accuracy_percent)
                    plt.plot(epochs, validation_accuracy_percent)
                    plt.xticks(epochs)
                    plt.ylim(0)
                    plt.ylabel('Mean Squared Error [-]')
                    plt.xlabel('Epoch [-]')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.grid(True)
                    #plt.show()
                    plt.savefig(LOG_DIR + '/acc.png')
                    plt.clf()

                    plt.plot(epochs, history.history['loss'])
                    plt.plot(epochs, history.history['val_loss'])
                    plt.xticks(epochs)
                    plt.ylim(0)
                    plt.ylabel('Loss (MAE) [-]')
                    plt.xlabel('Epoch [-]')
                    plt.legend(['train', 'val'], loc='upper left')
                    plt.grid(True)
                    #plt.show()
                    plt.savefig(LOG_DIR + '/loss.png')
                    plt.clf()

                    del model
                    del history

                    fold_number += 1