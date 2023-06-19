import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb
import tensorflow as tf


from sklearn.model_selection import train_test_split
#from tensorflow.keras import datasets, layers, models

directory = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/1.0.0/"

signals = np.empty((200, 1280, 2)) # TODO make it dynamic depending on number of records (record_count)
labels = np.empty(200)  # TODO make it dynamic depending on number of records (record_count)

record_count = 0
for filename in sorted(os.listdir(directory)):
    
    # only do data integration once per record
    if filename.endswith('.dat'):
        
        # for files named "pXX" label = 1 (AF group)
        if (os.path.basename(filename)[0] == 'p'):
            filename_without_ext = os.path.splitext(filename)[0]

            file_directory = directory + os.sep + filename_without_ext
            signals[record_count, :, :] = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1280)[0]
            labels[record_count] = 1

            record_count += 1

        # for files named "pXX" label = 0 (non AF group)
        elif (os.path.basename(filename)[0] == 'n'):
            filename_without_ext = os.path.splitext(filename)[0]

            file_directory = directory + os.sep + filename_without_ext
            signals[record_count, :, :] = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=1280)[0]
            labels[record_count] = 0

            record_count += 1

        else:
            pass



        
# split data into train and test sets randomly
train_data, test_data, train_labels, test_labels = train_test_split(
    signals, labels, test_size=0.2, random_state=21
    )



# normalize data
min_val = tf.reduce_min(signals)
max_val = tf.reduce_max(signals)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)



train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

print(train_data.element_spec)
print(list(train_data.as_numpy_iterator())[0])

#plt.grid()
#plt.plot(np.arange(100), train_data[0])
#plt.title("A Normal ECG")
#plt.show()


BATCH_SIZE = 1
train_data = train_data.batch(BATCH_SIZE)
test_data = test_data.batch(1)




# build CNN

RECORD_LENGTH = 1280

input_layer = tf.keras.Input(shape=(RECORD_LENGTH, 2))

x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(input_layer)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=3, activation='relu')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(384, activation='relu')(x)

output_layer = tf.keras.layers.Dense(1, activation='softmax')(x)


model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

print(model.summary())

model.fit(train_data,
          epochs = 20,
          validation_data = test_data
          )

del model