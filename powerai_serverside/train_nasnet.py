from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.applications.nasnet import NASNetMobile
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from datasequence import DataSequence
import pandas as pd

MODEL_NAME = "nasnet_lamem_model.h5"

# This is a custom loss function (Euclidean Loss) that was used in the MemNet paper, and is specialized for regression.
def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

# Here, we initialize the "NASNetMobile" model type and customize the final feature regressor layer.
# NASNet is a neural network architecture developed by Google.
# This architecture is specialized for transfer learning, and was discovered via Neural Architecture Search.
# NASNetMobile is a smaller version of NASNet.
model = NASNetMobile()
model = Model(model.input, Dense(1, activation='linear', kernel_initializer='normal')(model.layers[-2].output))

# This model will use the "Adam" optimizer.
model.compile("adam", euc_dist_keras)

model.summary()

# Here, we read the label files provided with the LaMem dataset.
train_pd = pd.read_csv("splits/train_1.txt")
test_pd = pd.read_csv("splits/test_1.txt")

# The batch size is set to 32.
batch_size = 32

# This callback will reduce the learning rate if the val_loss isn't decreasing.
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.003)
# This callback will log model stats to Tensorboard.
tb_callback = TensorBoard()
# This callback will checkpoint the best model at every epoch.
mc_callback = ModelCheckpoint(filepath='current_best.hdf5', verbose=1, save_best_only=True)

# This is the train DataSequence.
train_sequence = DataSequence(train_pd, "./images", batch_size=batch_size)
train_steps = len(train_pd) // batch_size

# This is the validation DataSequence.
validation_sequence = DataSequence(test_pd, "./images", batch_size=batch_size)
validation_steps = len(test_pd) // batch_size

# These are the callbacks.
callbacks = [lr_callback, tb_callback, mc_callback]

# This line will train the model.
model.fit_generator(train_sequence, validation_data=validation_sequence, epochs=20, use_multiprocessing=True, workers=80, steps_per_epoch=train_steps, validation_steps=validation_steps, callbacks=callbacks)

# Finally, we save the model.
model.save(MODEL_NAME)
