import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_butterworth import butter_bandpass_filter
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline.normalize import normalize_ecg

# force tensorflow to use CPU (for my laptop)
tf.config.set_visible_devices([], 'GPU')

def eval_models_one_by_one(test_data):
    for i in range(5):

        # load the model
        MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/models/best_hyper_params_fourth_try/history/Model_2-blocks_3-layers_per_block_1/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        model = tf.keras.models.load_model(MODEL_DIR)
        #print(model.summary())

        model.evaluate(test_data)


def eval_models_ensemble(test_data):
    
    models = []
    for i in range(5):
        MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/models/best_hyper_params_fourth_try/history/Model_2-blocks_3-layers_per_block_1/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        model = tf.keras.models.load_model(MODEL_DIR)
        models.append(model)

        input_1 = tf.keras.Input(shape=(1280,1), name='input_1')
        input_2 = tf.keras.Input(shape=(1280,1), name='input_2')

        # get output for each model input
        outputs = [model((input_1, input_2)) for model in models]

        x = tf.keras.layers.Average()(outputs)

        ensemble = tf.keras.Model(inputs=[input_1, input_2], outputs=x)

        ensemble.compile(optimizer=tf.keras.optimizers.Adam())

        ensemble.evaluate(test_data)








FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/validation_split"
signals, labels, patient_number_array = load_dataset_PAF(FILE_DIRECTORY)
print(np.shape(signals))

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

# transform data into tf dataset
test_data_1 = signals[:,:,0]
test_data_2 = signals[:,:,1]

test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), labels))
test_data = test_data.batch(100)

#eval_models_one_by_one(test_data)
eval_models_ensemble(test_data)

