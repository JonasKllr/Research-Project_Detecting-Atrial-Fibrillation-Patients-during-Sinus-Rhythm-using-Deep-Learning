import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

# import keras_tuner as kt
import tensorflow as tf


def build_model_without_tuner_1(LEARNING_RATE, KERNEL_SIZE, POOLING_LAYER):
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)


    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)


    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(40, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(40, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])

    # x = tf.keras.layers.GlobalAveragePooling1D()(concatenate)
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_1-blocks_3-layers_per_block_2')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
                )

    return model

def build_model_without_tuner_2(LEARNING_RATE, KERNEL_SIZE, POOLING_LAYER):
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(40, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(40, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])

    # x = tf.keras.layers.GlobalAveragePooling1D()(concatenate)
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_2-blocks_3-layers_per_block_1')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
                )

    return model

def build_model_without_tuner_3(LEARNING_RATE, KERNEL_SIZE, POOLING_LAYER):
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(20, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])

    # x = tf.keras.layers.GlobalAveragePooling1D()(concatenate)
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_3-blocks_2-layers_per_block_2')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
                )

    return model

def build_model_without_tuner_4(LEARNING_RATE, KERNEL_SIZE, POOLING_LAYER):
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(20, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_4-blocks_2-layers_per_block_1')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
                )

    return model


def build_model_age_regression(LEARNING_RATE, KERNEL_SIZE, POOLING_LAYER):
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(40, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=40, kernel_size=5, strides=2, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(40, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])
    output_layer = tf.keras.layers.Dense(1)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_age_regression')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=[tf.keras.metrics.MeanSquaredError()]
                )

    return model

def build_model_age_classification(LEARNING_RATE, KERNEL_SIZE, POOLING_LAYER):
    RECORD_LENGTH = 1280

    # input layers
    input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
    input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

    # path 1
    x_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.Conv1D(filters=20, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    if POOLING_LAYER == 'max_pool':
       x_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_1)
    elif POOLING_LAYER == 'avg_pool':
       x_1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_1)
    x_1 = tf.keras.layers.Dropout(0.25)(x_1)

    x_1 = tf.keras.layers.GlobalAveragePooling1D()(x_1)
    x_1 = tf.keras.layers.Dense(20, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(1, activation='relu')(x_1)


    # path 2
    x_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(input_layer_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.Conv1D(filters=20, kernel_size=KERNEL_SIZE, strides=1, padding='same', activation='relu')(x_2)
    x_2 = tf.keras.layers.BatchNormalization()(x_2)
    if POOLING_LAYER == 'max_pool':
       x_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x_2)
    elif POOLING_LAYER == 'avg_pool':
       x_2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x_2)
    x_2 = tf.keras.layers.Dropout(0.25)(x_2)

    x_2 = tf.keras.layers.GlobalAveragePooling1D()(x_2)
    x_2 = tf.keras.layers.Dense(20, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(1, activation='relu')(x_2)


    # combining both paths
    concatenate = tf.keras.layers.concatenate([x_1, x_2])
    output_layer = tf.keras.layers.Dense(5)(concatenate)


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_age_classification')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )

    #tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')

    return model


def build_model_tuner(hp):
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
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.F1Score(name='f1_score')]
                )

    return model



if __name__ == '__main__':
    
    model = build_model_age_regression(LEARNING_RATE=0.001, KERNEL_SIZE=6, POOLING_LAYER='max_pool')
    print(model.summary())
    print(model.name)