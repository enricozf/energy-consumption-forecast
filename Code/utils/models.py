import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Dropout, Flatten, Bidirectional,
    TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from utils.metrics import last_timestep_mae, last_timestep_mse

def gen_dense_model_v0(
    input_shape, output_samples: int = 10, print_summary: bool = False):
    
    ip = Input(shape=input_shape)
    x = TimeDistributed(Dense(output_samples))(ip)
    x = TimeDistributed(Dense(output_samples))(x)

    model = Model(ip, x)

    if print_summary:
        print(model.summary())

    return model

def compile_and_fit(
    model: Model, ckpt_filepath: str, train_dataset: tf.data.Dataset, 
    val_dataset: tf.data.Dataset = None, early_stopping: bool=True, **kwargs):
    
    # Configuring model callbacks
    print('Creating callbacks...')
    if early_stopping:
        earlypointer = EarlyStopping(
            monitor='val_loss', min_delta=kwargs.get('early_stop_min_delta', 0.00001),
            patience=kwargs.get('early_stop_patience', 5), verbose=1
            )
        callbacks = [earlypointer]
    else:
        callcacks = []

    ckpt_callback = ModelCheckpoint(
        filepath=ckpt_filepath, verbose=1, monitor='val_loss', save_best_only=True
    )
    callbacks.append(ckpt_callback)

    # log_dir = "..//Results//logs//fit//" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # callbacks.append(tensorboard_callback)

    # Compiling the model
    print('Compiling model...')
    model.compile(
        optimizer=Adam(learning_rate=kwargs.get('lr',0.001)),
        loss='mse',
        metrics=[last_timestep_mae,last_timestep_mse]
    )

    # Fitting the model
    print('Fitting model...')
    hist = model.fit(train_dataset, epochs=10, callbacks=callbacks,
    validation_data=val_dataset, verbose=1)

    return model, hist