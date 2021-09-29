import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import pillow

tf.keras.backend.clear_session()

# path_2_model = "model/final_model_weights.h5"
# model = keras.models.load_model(path_2_model)

with open('../model/final_model.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
#
new_model.load_weights('model/final_model_weights.h5')

print(new_model.summary())

test_data_dir = 'C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\3_test\\3_test'

img_width, img_height = 299, 299
batch_size = 1
nb_test_samples = 1000

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    class_mode='binary')

compilation = new_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

evaluation = new_model.evaluate(
    test_generator,
    steps=nb_test_samples,
    verbose=1)

pred = new_model.predict(
    test_generator,
    steps=nb_test_samples,
    verbose=1)
