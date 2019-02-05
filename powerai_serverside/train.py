import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import multi_gpu_model

from lamem_generator import lamem_generator, load_split

# Using "Euclidean Distance" loss
def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Define model structure
model = Sequential()
model.add(Conv2D(96, (11, 11), (4, 4), activation="relu", input_shape=(227, 227, 3)))
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

model.summary()

# Load the first train/test split
train_split = load_split("lamem/splits/train_1.txt")
test_split = load_split("lamem/splits/test_1.txt")

# Define the batch size for training + testing (64 * 4 GPUs)
batch_size = 64 * 4

# Create the training & testing data generators
train_gen = lamem_generator(train_split, batch_size=batch_size)
test_gen = lamem_generator(test_split, batch_size=batch_size)

# Distribute the model across all 4 GPUs
model = multi_gpu_model(model, gpus=4)

# Compile the model
model.compile("adam", euclidean_distance_loss)

# Train the model with "fit_generator"
model.fit_generator(train_gen, steps_per_epoch=int(len(train_split) / batch_size), epochs=5, verbose=1, validation_data=test_gen, validation_steps=int(len(test_split) / batch_size))

# Save the model
model.save("memnet_model.h5")
