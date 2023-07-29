import sys
sys.path.append("/home/joke793c/research_project/scripts")

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


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_2-blocks_3-layers_per_block_2')

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
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

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
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

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


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_3-blocks_3-layers_per_block_1')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
                )

    return model

def build_model_without_tuner_3():
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
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
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])

    # x = tf.keras.layers.GlobalAveragePooling1D()(concatenate)
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_4-blocks_2-layers_per_block_2')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
                )

    return model

def build_model_without_tuner_4():
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(20, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_5-blocks_2-layers_per_block_1')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
                )

    return model



    RECORD_LENGTH = 1280

    # tuner
    hp_kernels = hp.Int('kernels', 3, 15, step=3)

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=hp_kernels, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)

    if hp.Choice('pooling', ['avg', 'max']) == 'max':
      x_1 = tf.keras.layers.MaxPool1D()(x_1)
    else:
      x_1 = tf.keras.layers.AvgPool1D()(x_1)

    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(20, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=hp_kernels, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    
    if hp.Choice('pooling', ['avg', 'max']) == 'max':
      x_2 = tf.keras.layers.MaxPool1D()(x_2)
    else:
      x_2 = tf.keras.layers.AvgPool1D()(x_2)

    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
                )

    return model



if __name__ == '__main__':
    
    model = build_model_without_tuner_2()
    print(model.summary())
    print(model.name)