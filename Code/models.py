#%%
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

import optuna

from utils.metrics import last_timestep_mae, last_timestep_mse
from main import gen_dataset

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
    print('Compiling model noch einmal...')
    model.compile(
        optimizer=Adam(learning_rate=kwargs.get('lr',0.001)),
        metrics=[last_timestep_mae,last_timestep_mse],
        loss=tf.keras.losses.MeanSquaredError()
    )

    # Fitting the model
    print('Fitting model...')
    hist = model.fit(train_dataset, epochs=10, callbacks=callbacks,
                     validation_data=val_dataset, verbose=1, **kwargs)

    return model, hist

def optuna_dense_model_v0(trial: optuna.trial.Trial):

    fold_json_path = '..//Data//folds_test.json'
    dic_split, scaler = gen_dataset(fold_json_path=fold_json_path, fold='3',
                                    time_col='index', scale_flg=True,
                                    boxcox_trnsf_flag=True)

    ip = Input(shape=(24,4))
    x = TimeDistributed(Dense(trial.suggest_int('first_layer',4,20,step=4),
                              activation='relu'))(ip)
    x = TimeDistributed(Dense(10))(x)

    model = Model(ip,x)

    print(model.summary())

    earlypointer = EarlyStopping(
            monitor='val_loss', min_delta=0.00001,
            patience=5, verbose=1
            )
    callbacks = [earlypointer]

    print('Compiling model noch einmal...')
    lr = trial.suggest_loguniform('lr', 1e-5,1e-3)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        metrics=[last_timestep_mae,last_timestep_mse],
        loss=tf.keras.losses.MeanSquaredError()
    )

    model.fit(
        dic_split['train'].repeat(), epochs=100, callbacks=callbacks,
        validation_data=dic_split['val'].repeat(),verbose=1,
        steps_per_epoch=dic_split['train_num_batches']//2,
        validation_steps=dic_split['val_num_batches']//2
    )

    scores = model.evaluate(dic_split['test'])

    return scores[0]

#%%
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_dense_model_v0, n_trials=10, show_progress_bar=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

#TODO: Código para pegar o y_true e o y_pred
# y_true_stacked = None
# y_pred_stacked = None
# for x,y in dic_split['test']:
#     if y_true_stacked is not None:
#         y_true_stacked = np.vstack([y_true_stacked, y])
#         y_pred_stacked = np.vstack([y_pred_stacked,
#                                     model.predict(x)])
#     else:
#         y_true_stacked = y
#         y_pred_stacked = model.predict(x)