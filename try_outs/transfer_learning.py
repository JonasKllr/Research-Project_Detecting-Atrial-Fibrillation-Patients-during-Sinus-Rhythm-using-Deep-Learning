import tensorflow as tf

DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/bachelorarbeit/Netze/Versuch_1_Anomalieerkennung/Struktur_1_siamesisch/Struktur_siamesisch_Datensatz_gesamt/Modell_speicher'

model = tf.keras.saving.load_model(DIR)

# freezw the model
model.trainable = False
#model.summary()
#print(model.layers)

# Freeze the weights of the pre-trained layers
#for layer in model.layers:
#    layer.trainable = False

RECORD_LENGTH = 1280

#input_layer_1 = tf.keras.Input(shape=(RECORD_LENGTH, 1))
#input_layer_2 = tf.keras.Input(shape=(RECORD_LENGTH, 1))

#input_layer_1 = tf.keras.Input(shape=(140, 310, 1))
#input_layer_2 = tf.keras.Input(shape=(140, 310, 1))

#base_output_0 = model.layers['cad_image[0][0]'].output
#base_output_1 = model(input_layer_1, training=False).get_layer('conv2d_1').output
#base_output_2 = model(input_layer_2, training=False).get_layer('conv2d_9').output

base_output_1 = model.get_layer('conv2d_1').output
base_output_2 = model.get_layer('conv2d_9').output

concat = tf.keras.layers.Concatenate()([base_output_1, base_output_2])
flatten = tf.keras.layers.Flatten()(concat)

new_layer = tf.keras.layers.Dense(2, activation='relu')(flatten)


#model_base = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
new_model = tf.keras.Model(inputs=model.input, outputs=new_layer)

#model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
#              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#              metrics=[tf.keras.metrics.BinaryAccuracy()]
#              )


#for layer in (new_model.layers[:2]):
#    layer.trainable = False

#for layer in (new_model.layers[:2])
#
new_model.summary()