from tensorflow import keras
from keras.models import model_from_json
import os




## ruta al directorio del modelo"
#path_2_model = os.path.abspath(os.getcwd())

json_file = open('../model/final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#extract model from file
loaded_model = model_from_json(loaded_model_json)
#model = keras.models.load_model(path_2_model)

# print model info
print(loaded_model.summary())
'''
model_path='final_model.json'

def create_model(model_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

weights_path='final_model_weights.h5'
def load_trained_model(model_path, weights_path):
   model = create_model(model_path)
   model.load_weights(weights_path)
   return model

hello= load_trained_model(model_path,weights_path)

print(hello.summary())
'''

'''

test_data_dir = 'C:\\Users\\shern\\Google Drive\\UPC\\TFM\\10_Images\\3_test\\3_test\\normal'

img_width, img_height = 299, 299
batch_size = 1
nb_test_samples = 1000




test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle= False,
    class_mode='binary')

evaluation = model.evaluate(
    test_generator,
    steps=nb_test_samples,
   verbose=1)


pred = model.predict(
    test_generator,
    steps=nb_test_samples,
    verbose=1)

'''