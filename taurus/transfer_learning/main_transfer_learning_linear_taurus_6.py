import sys
sys.path.append("/home/joke793c/research_project/scripts/taurus")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold

from data_pipeline_taurus.load_dataset_without_filter import load_dataset_PAF
from data_pipeline_taurus.filter_butterworth import butter_bandpass_filter
from data_pipeline_taurus.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline_taurus.normalize import normalize_ecg

# force tensorflow to use CPU (for my laptop)
tf.config.set_visible_devices([], 'GPU')

# load data
FILE_DIRECTORY = "/beegfs/ws/0/joke793c-research_project/data_sets/paf/paf_pred_chall/cleaned"
signals, labels, patient_number_array = load_dataset_PAF(FILE_DIRECTORY)

# data normalization into range [0.0, 1.0] to filter for signal clipping
signals_norm = normalize_ecg(signals)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels, patient_number_array = delete_clipped_segments(signals, labels, patient_number_array, clipped_segments)
patient_number_array_unique = np.unique(patient_number_array, return_index=False)

print(np.shape(signals))
print(np.shape(labels))
print(np.shape(patient_number_array))
print(np.shape(patient_number_array_unique))


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

LEARNING_RATE = [1e-2, 1e-3, 1e-4]

for learning_rate_choice in LEARNING_RATE:

    fold_number = 1
    for train_index, test_index in kfold.split(patient_number_array_unique):
        
        # train test split according to patients
        train_split_patient = patient_number_array_unique[train_index]
        test_split_patient = patient_number_array_unique[test_index]
        
        train_signal_index = np.where(np.in1d(patient_number_array, train_split_patient))[0]
        test_signal_index = np.where(np.in1d(patient_number_array, test_split_patient))[0]

        # split data into folds
        train_data_1 = signals[train_signal_index,:,0]
        train_data_2 = signals[train_signal_index,:,1]
        train_labels = labels[train_signal_index]

        test_data_1 = signals[test_signal_index,:,0]
        test_data_2 = signals[test_signal_index,:,1]
        test_labels = labels[test_signal_index]

        # transform it into tesorflow dataset
        train_data = tf.data.Dataset.from_tensor_slices(((train_data_1, train_data_2), train_labels))
        test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), test_labels))

        # shuffle the data
        train_data = train_data.shuffle(buffer_size=train_data.cardinality())
        test_data = test_data.shuffle(buffer_size=test_data.cardinality())

        # batch the data
        train_data = train_data.batch(32)
        test_data = test_data.batch(100)

        
        MODEL_DIR = f'/beegfs/ws/0/joke793c-research_project/model_pretained/model'

        model = tf.keras.models.load_model(MODEL_DIR)
        model.trainable = False

        for layer in model.layers:
            layer._name = layer.name + str("_frozen")

        model.summary()
        
        base_output_1 = model.get_layer('batch_normalization_frozen').output
        base_output_2 = model.get_layer('batch_normalization_4_frozen').output



        x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=6, strides=1, padding='same', activation='relu')(base_output_1)
        x_1 = tf.keras.layers.BatchNormalization()(x_1)
        x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
        x_1 = tf.keras.layers.Dropout(0.25)(x_1)

        x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
        x_1 = tf.keras.layers.BatchNormalization()(x_1)
        x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
        x_1 = tf.keras.layers.BatchNormalization()(x_1)
        x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
        x_1 = tf.keras.layers.Dropout(0.25)(x_1)

        x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
        x_1 = tf.keras.layers.Dense(20, activation='relu')(x_1)
        x_1 = tf.keras.layers.Dense(1, activation='linear')(x_1)


        x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=6, strides=1, padding='same', activation='relu')(base_output_2)
        x_2 = tf.keras.layers.BatchNormalization()(x_2)
        x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
        x_2 = tf.keras.layers.Dropout(0.25)(x_2)

        x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(base_output_2)
        x_2 = tf.keras.layers.BatchNormalization()(x_2)
        x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
        x_2 = tf.keras.layers.BatchNormalization()(x_2)
        x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
        x_2 = tf.keras.layers.Dropout(0.25)(x_2)

        x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
        x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
        x_2 = tf.keras.layers.Dense(1, activation='linear')(x_2)


        # combining both paths
        concatenate = tf.keras.layers.concatenate([x_1, x_2])
        output_layer = tf.keras.layers.Dense(1)(concatenate)

        model = tf.keras.Model(inputs=model.input, outputs=output_layer, name='Model_3_transfer_learning_6')


        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate_choice),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.BinaryAccuracy(),
                                tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
                    )

        model.summary()





        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_number} ...')

        # creating directory for logging
        LOG_DIR = (f'/beegfs/ws/0/joke793c-research_project/history/' 
                    + model.name + os.path.sep
                    + f'learning_rate_{learning_rate_choice}' + os.path.sep
                    + f'fold_{fold_number}')
        os.makedirs(LOG_DIR)
        

        # Tensorboard callback
        TB_DIR = LOG_DIR + '/tensorboard/'
        os.makedirs(TB_DIR)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TB_DIR, histogram_freq=2)

        # checkpoint callback
        MODEL_DIR = LOG_DIR + '/model/'
        os.makedirs(MODEL_DIR)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR,
            save_weights_only=False,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True)

        # logging callback
        log_callback = tf.keras.callbacks.CSVLogger(LOG_DIR + '/history.csv')

        history = model.fit(train_data,
                epochs = 50,
                verbose= 1,
                validation_data = test_data,
                callbacks= [log_callback, model_checkpoint_callback]
                )

        # save history plots
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
        plt.savefig(LOG_DIR + '/acc.png')
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
        plt.savefig(LOG_DIR + '/loss.png')
        plt.clf()

        del model
        del history

        fold_number += 1











