import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from data_pipeline.filter_butterworth import butter_bandpass_filter
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments_CinC
from data_pipeline.normalize import normalize_ecg
from data_pipeline.delete_age_zero_CinC import delete_age_zero_CinC

# force tensorflow to use CPU (for my laptop)
tf.config.set_visible_devices([], 'GPU')

def eval_models_one_by_one(test_data):
    for i in range(5):

        # load the model
        #MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        #MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros_100_epochs/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros_250_epochs_2/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        model = tf.keras.models.load_model(MODEL_DIR)
        #print(model.summary())

        model.evaluate(test_data)


def eval_models_ensemble(test_data, labels):
    
    models = []
    for i in range(5):
        #MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        #MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros_100_epochs/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        #MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros_150_epochs/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        #MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros_150_epochs_2/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/correct_no_zeros_250_epochs_2/history/Model_age_regression/kernel_6/pooling_max_pool/learning_rate_0.001/fold_{i+1}/model'
        model = tf.keras.models.load_model(MODEL_DIR)
        models.append(model)


    number_samples = np.shape(labels)[0]
    predictions = np.empty([number_samples, 0])
    for model in models:
        predictions_temp = model.predict(test_data)
        print(predictions_temp)
        predictions = np.hstack((predictions, predictions_temp))

    labels = np.reshape(labels, (-1, 1))
    predictions = np.hstack((predictions, labels))


    return predictions




# load data
FILE_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'
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

#signals = signals_temp[mask]
#labels = labels_temp[mask]
signals_test = signals_temp[~mask]
labels_test = labels_temp[~mask]

del signals_temp
del labels_temp

### create 10% test split ###

print(np.shape(signals_test))
print(np.shape(labels_test))



# data normalization into range [0.0, 1.0] to filter for signal clipping
signals_norm = normalize_ecg(signals_test)

#find rows in signals array with clipped segments
clipped_segments = find_clipped_segments(signals_norm)
del signals_norm

# delete the segments containing signal clipping
signals, labels = delete_clipped_segments_CinC(signals_test, labels_test, clipped_segments)

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

# transform data into tf dataset
test_data_1 = signals[:,:,0]
test_data_2 = signals[:,:,1]

test_data = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), labels))
test_data = test_data.batch(100)

test_data_inference = tf.data.Dataset.from_tensor_slices(((test_data_1, test_data_2), ))
test_data_inference = test_data_inference.batch(100)

#MODEL_DIR = f'/media/jonas/SSD_new/CMS/Semester_4/research_project/models/best_hyper_params_fourth_try/history/Model_2-blocks_3-layers_per_block_1/kernel_6/pooling_max_pool/learning_rate_0.001/fold_1/model'
#model = tf.keras.models.load_model(MODEL_DIR)

#predictions = model.predict(test_data_inference)


eval_models_one_by_one(test_data)
predictions = eval_models_ensemble(test_data_inference, labels)


perfect_df = pd.DataFrame(columns=['pred', 'true'], data=[[0, 0], [100, 100]])

predictions = pd.DataFrame(predictions)
print(predictions.head())
#predictions.rename(columns={0: 'fold_1', 1: 'fold_2', 2: 'fold_3', 3: 'fold_4', 4: 'fold_5', 5: 'true',})
predictions.columns = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'true']
print(predictions.head())

sns.set_style("darkgrid")

plt.figure()
sns.regplot(x='true', y='fold_1', data=predictions, ci=None, line_kws={'color':'red'}, scatter_kws={'edgecolors':'white', 'alpha':1, 'linewidths':0.5})
sns.lineplot(data=perfect_df, x='pred', y='true', color='grey', linestyle='--')
plt.xlim((-5, 100))
plt.ylim((-5, 100))
plt.xlabel('age true [years]')
plt.ylabel('age predicted [years]')
plt.title('Fold 1')
plt.tight_layout()


plt.figure()
plt.xlim((-5, 100))
plt.ylim((-5, 100))
sns.regplot(x='true', y='fold_2', data=predictions, ci=None, line_kws={'color':'red'}, scatter_kws={'edgecolors':'white', 'alpha':1, 'linewidths':0.5})   #scatter_kws={'color':'blue'},
sns.lineplot(data=perfect_df, x='pred', y='true', color='grey', linestyle='--')
plt.xlabel('age true [years]')
plt.ylabel('age predicted [years]')
plt.title('Fold 2')
plt.tight_layout()


plt.figure()
plt.xlim((-5, 100))
plt.ylim((-5, 100))
sns.regplot(x='true', y='fold_3', data=predictions, ci=None, line_kws={'color':'red'}, scatter_kws={'edgecolors':'white', 'alpha':1, 'linewidths':0.5})   #scatter_kws={'color':'blue'},
sns.lineplot(data=perfect_df, x='pred', y='true', color='grey', linestyle='--')
plt.xlabel('age true [years]')
plt.ylabel('age predicted [years]')
plt.title('Fold 3')
plt.tight_layout()

plt.figure()
plt.xlim((-5, 100))
plt.ylim((-5, 100))
sns.regplot(x='true', y='fold_4', data=predictions, ci=None, line_kws={'color':'red'}, scatter_kws={'edgecolors':'white', 'alpha':1, 'linewidths':0.5})   #scatter_kws={'color':'blue'},
sns.lineplot(data=perfect_df, x='pred', y='true', color='grey', linestyle='--')
plt.xlabel('age true [years]')
plt.ylabel('age predicted [years]')
plt.title('Fold 4')
plt.tight_layout()

plt.figure()
plt.xlim((-5, 100))
plt.ylim((-5, 100))
sns.regplot(x='true', y='fold_5', data=predictions, ci=None, line_kws={'color':'red'}, scatter_kws={'edgecolors':'white', 'alpha':1, 'linewidths':0.5})   #scatter_kws={'color':'blue'},
sns.lineplot(data=perfect_df, x='pred', y='true', color='grey', linestyle='--')
plt.xlabel('age true [years]')
plt.ylabel('age predicted [years]')
plt.title('Fold 5')
plt.tight_layout()


plt.show()


