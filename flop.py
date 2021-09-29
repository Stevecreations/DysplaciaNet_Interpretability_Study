import tensorflow as tf
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras.applications.mobilenet import MobileNet




model_path = 'model/final_model.json'
weigths_path = 'model/final_model_weights.h5'

with open(model_path) as json_file:
    json_config = json_file.read()
dysplacianet_model = keras.models.model_from_json(json_config)

dysplacianet_model.load_weights(weigths_path)
# dysplacianet_model.add(Dense(2, activation='softmax', name='visualized_layer'))

print(dysplacianet_model.summary())



from keras_flops import get_flops



# Calculae FLOPS
flops = get_flops(dysplacianet_model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
# >>> FLOPS: 0.0338 G