import tensorflow as tf

MODEL_DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/history/test/Model_4-blocks_2-layers_per_block_1/kernel_3/pooling_max_pool/learning_rate_0.01/fold_1/model'

model = tf.keras.models.load_model(MODEL_DIR)

print(model.summary())
