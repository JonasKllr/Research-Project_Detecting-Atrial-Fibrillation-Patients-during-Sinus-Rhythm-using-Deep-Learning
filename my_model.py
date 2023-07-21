import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import tensorflow as tf

def build_model_without_tuner_1():
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=80, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=80, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(80, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=80, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=80, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(80, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])

    # x = tf.keras.layers.GlobalAveragePooling1D()(concatenate)
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
                )

    return model



def build_model_without_tuner_2():
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(40, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(40, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])

    # x = tf.keras.layers.GlobalAveragePooling1D()(concatenate)
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
                )

    return model



if __name__ == '__main__':
    
    model = build_model_without_tuner_2()
    print(model.summary())