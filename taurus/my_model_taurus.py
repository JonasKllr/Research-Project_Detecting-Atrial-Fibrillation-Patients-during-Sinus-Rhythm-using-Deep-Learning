import sys
sys.path.append("/home/joke793c/research_project/scripts")

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


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_2-blocks_3-layers_per_block_2')

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


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_3-blocks_3-layers_per_block_1')

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


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_4-blocks_2-layers_per_block_2')

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


    model = tf.keras.Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='Model_5-blocks_2-layers_per_block_1')

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
                )

    return model



if __name__ == '__main__':
    
    model = build_model_without_tuner_2()
    print(model.summary())
    print(model.name)