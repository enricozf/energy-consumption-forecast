import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Dropout, Flatten, Bidirectional,
    TimeDistributed, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from utils.data_prep import gen_dataset
from utils.metrics import last_timestep_mae, last_timestep_mse

def optuna_dense_model_v0(trial: optuna.trial.Trial):

    fold_json_path = '..//Data//folds.json'
    ckpt_filepath = '..//Results//tmp//best_model.hdf5'
    dic_split, scaler = gen_dataset(fold_json_path=fold_json_path, fold='3',
                                    time_col='index', scale_flg=True,
                                    num_sm_split={
                                        'train':200, 
                                        'val':50, 
                                        'test':50},
                                    boxcox_trnsf_flag=True,
                                    batch_size=512)

    ip = Input(shape=(24,4))

    x = TimeDistributed(Dense(trial.suggest_int('1st_layer',4,20,step=4),
                              activation='relu'))(ip)
    # if num_dense_layers == 3:
    #     x = TimeDistributed(Dense(trial.suggest_int('2nd_layer',4,20,step=4),
    #                               activation='relu'))(ip)

    x = TimeDistributed(Dense(10))(x)

    model = Model(ip,x)

    print(model.summary())

    earlypointer = EarlyStopping(
            monitor='val_loss', min_delta=0.00001,
            patience=5, verbose=1
            )
    ckpt_callback = ModelCheckpoint(
        filepath=ckpt_filepath, verbose=1, monitor='val_loss', save_best_only=True
    )
    callbacks = [earlypointer, ckpt_callback]

    # Compile model
    print('Compiling model...')
    lr = trial.suggest_loguniform('lr', 1e-5,1e-3)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=[last_timestep_mae,last_timestep_mse],
        loss=tf.keras.losses.MeanSquaredError()
    )

    # Fit model
    model.fit(
        dic_split['train'].repeat(), epochs=1000, callbacks=callbacks,
        validation_data=dic_split['val'].repeat(),verbose=2,
        steps_per_epoch=dic_split['train_num_batches']//10,
        validation_steps=dic_split['val_num_batches']*.9//1
    )

    # Load best model weights
    print('Loading best model for validation...')
    best_model = keras.models.load_model(
        ckpt_filepath,
        custom_objects={
            "last_timestep_mae" : last_timestep_mae,
            "last_timestep_mse" : last_timestep_mae
        })
    best_model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=[last_timestep_mae,last_timestep_mse],
        loss=tf.keras.losses.MeanSquaredError()
    )

    # Evaluate and return loss for test dataset
    scores = best_model.evaluate(
        dic_split['test'], steps=dic_split['test_num_batches']*.9//1)
    return scores[0]


def optuna_dense_model_v1(trial: optuna.trial.Trial):

    fold_json_path = '..//Data//folds.json'
    ckpt_filepath = '..//Results//tmp//best_model.hdf5'
    dic_split, scaler = gen_dataset(
        final_data_path='..//Data//acorn_{}_preproc_data.parquet.gzip',
        fold_json_path=fold_json_path, fold='3',
        time_col='index', scale_flg=True,
        num_sm_split={
            'train':200, 
            'val':50, 
            'test':50},
        boxcox_trnsf_flag=True,
        batch_size=256,
        test_gen_dataset_flg=True)

    # All suggested values
    units = trial.suggest_int('1st_layer',4,20,step=4)
    lr = trial.suggest_loguniform('lr', 1e-4,1e-3, step=4.9e-4)

    ip = Input(shape=(24,4))

    x = TimeDistributed(Dense(units, activation='relu'))(ip)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    earlypointer = EarlyStopping(
            monitor='val_loss', min_delta=0.00001,
            patience=5, verbose=1
            )
    ckpt_callback = ModelCheckpoint(
        filepath=ckpt_filepath, verbose=1, monitor='val_loss', save_best_only=True
    )
    callbacks = [earlypointer, ckpt_callback]

    # Compile model
    print('Compiling model...')
    model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=[tf.keras.losses.MeanAbsoluteError()],
        loss=tf.keras.losses.MeanSquaredError()
    )

    # Fit model
    model.fit(
        dic_split['train'].repeat().prefetch(tf.data.AUTOTUNE), 
        epochs=1000, callbacks=callbacks,
        validation_data=dic_split['val'].repeat().prefetch(tf.data.AUTOTUNE),
        verbose=2,
        steps_per_epoch=dic_split['train_num_batches']//10,
        validation_steps=dic_split['val_num_batches']*.9//1
    )

    # Load best model weights
    print('Loading best model for validation...')
    
    model.load_weights(ckpt_filepath)

    # Evaluate and return loss for test dataset
    scores = model.evaluate(
        dic_split['test'], steps=dic_split['test_num_batches']*.9//1)
    return scores[0]
