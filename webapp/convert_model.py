import sys

import coremltools

# The model was trained in tf.keras, so we need to load the model the same way
from tensorflow.keras.models import load_model

# Since CoreML only supports keras model conversion, we will move the weights from tf.keras to keras
from keras.models import Sequential
from keras.layers import *

# Load the tf.keras model and extract the weights
tf_model = load_model(sys.argv[1])
weights = tf_model.get_weights()

# Define the architecture with keras
model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), activation="relu", input_shape=(227, 227, 3)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 5), activation="relu"))
model.add(ZeroPadding2D((2, 2)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(384, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

# Copy the weights to keras
model.set_weights(weights)

# Export to CoreML & save
coreml_model = coremltools.converters.keras.convert(model, input_names="data", image_input_names="data", image_scale=1./255.)
coreml_model.save("lamem.mlmodel")
