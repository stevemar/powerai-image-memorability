import coremltools
import sys
from keras.models import load_model
from keras import backend as K

def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

k_model = load_model(sys.argv[1], custom_objects={'euc_dist_keras': euc_dist_keras})

model = coremltools.converters.keras.convert(k_model, input_names="data", image_input_names="data", image_scale=1./255.)
model.save("lamem.mlmodel")
