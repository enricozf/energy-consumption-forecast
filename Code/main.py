#%%
import joblib
import sys
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Dropout, Flatten, Bidirectional,
    TimeDistributed, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import optuna

# from utils.metrics import last_timestep_mae, last_timestep_mse
# from utils.data_prep import gen_dataset
from utils.optuna_scripts import (
    optuna_dense_many2many_v0, optuna_dense_many2one_v0, 
    optuna_dense_many2one_v1, optuna_mlp_mixer_many2one_v0,
    optuna_lstm_many2one_v0, optuna_lstm_many2one_v1,
    optuna_conv1d_lstm_many2one_v0, optuna_wavenet_many2one_v0,
    optuna_wavenet_many2one_v1, optuna_wavenet_mlp_mixer_many2one_v0,
    optuna_conv1d_lstm_many2one_v1, optuna_wavenet_mlp_mixer_many2one_v1,
    optuna_conv1d_lstm_many2one_v2, optuna_conv1d_lstm_many2one_v3,
    optuna_lstm_many2one_v2,
    wavenet)
from utils.models import Patches, MLPMixerLayer

# def gen_dense_model_v0(
#     input_shape, output_samples: int = 10, print_summary: bool = False):
    
#     ip = Input(shape=input_shape)
#     x = TimeDistributed(Dense(output_samples))(ip)
#     x = TimeDistributed(Dense(output_samples))(x)

#     model = Model(ip, x)

#     if print_summary:
#         print(model.summary())

#     return model


# def gen_dense_model_v1(
#     input_shape, output_samples: int = 10, print_summary: bool = False):
    
#     ip = Input(shape=input_shape)
#     x = TimeDistributed(Dense(output_samples))(ip)
#     # x = TimeDistributed(Dense(output_samples//4))(x)
#     x = Flatten()(x)
#     x = Dense(output_samples)(x)

#     model = Model(ip, x)

#     if print_summary:
#         print(model.summary())

#     return model

# class Patches(tf.keras.layers.Layer):
#     def __init__(self, patch_size=4, num_patches=6, *args, **kwargs):
#         super(Patches, self).__init__(**kwargs)
#         self.patch_size = int(patch_size)
#         self.num_patches = int(num_patches)

#     def call(self, sequences):
#         patches = tf.image.extract_patches(
#             images=tf.expand_dims(sequences, axis=-1),
#             sizes=[1,self.patch_size, self.patch_size,1],
#             strides=[1,self.patch_size,self.patch_size,1],
#             rates=[1,1,1,1],
#             padding='VALID'
#         )
#         patches = tf.squeeze(patches, axis=2)
#         return patches
    
#     def get_config(self):
#         config = super(Patches, self).get_config()
#         config.update(
#             {
#                 "patch_size" : self.patch_size,
#                 "num_patches" : self.num_patches
#             }
#         )
#         return config

# class MLPMixerLayer(tf.keras.layers.Layer):
#     def __init__(self, dropout_rate=0, *args, **kwargs):
#         super(MLPMixerLayer, self).__init__(*args, **kwargs)
#         self.dropout_rate = dropout_rate

#     def build(self, num_patches, hidden_units):
#         self.mlp1 = tf.keras.Sequential(
#             [
#             Dense(units=num_patches, activation='gelu'),
#             Dense(units=num_patches),
#             Dropout(rate=self.dropout_rate)
#             ]
#         )

#         self.mlp2 = tf.keras.Sequential(
#             [
#             Dense(units=num_patches, activation='gelu'),
#             Dense(units=hidden_units),
#             Dropout(rate=self.dropout_rate)
#             ]
#         )
#         self.normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#     def call(self, inputs):
#         # Apply layer normalization.
#         x = self.normalize(inputs)
#         # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
#         x_channels = tf.linalg.matrix_transpose(x)
#         # Apply mlp1 on each channel independently.
#         mlp1_outputs = self.mlp1(x_channels)
#         # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
#         mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
#         # Add skip connection.
#         x = mlp1_outputs + inputs
#         # Apply layer normalization.
#         x_patches = self.normalize(x)
#         # Apply mlp2 on each patch independtenly.
#         mlp2_outputs = self.mlp2(x_patches)
#         # Add skip connection.
#         x = x + mlp2_outputs
#         return x

#     def get_config(self):
#         config = super(MLPMixerLayer, self).get_config()
#         config.update(
#             {
#                 "mlp1" : self.mlp1,
#                 "mlp2" : self.mlp2,
#                 "normalize" : self.normalize
#             }
#         )
#         return config

# def gen_mlp_mixer_model_v0(
#     input_shape: list, patch_size: int, #hidden_units: int, 
#     num_blocks: int = 1, dropout_rate: float = 0, 
#     output_samples: int = 10, print_summary: bool = False):
    
#     num_patches = input_shape[0] / patch_size
#     hidden_units = patch_size**2

#     ip = Input(shape=input_shape)
#     x = Patches(patch_size=patch_size, num_patches=num_patches)(ip)
#     x = keras.Sequential(
#         [MLPMixerLayer(
#             num_patches, hidden_units, dropout_rate) for _ in range(num_blocks)]
#     )(x)
#     x = GlobalAveragePooling1D()(x)
#     x = Dense(output_samples)(x)

#     model = Model(ip, x)

#     if print_summary:
#         print(model.summary())

#     return model

# def compile_and_fit(
#     model: Model, ckpt_filepath: str, train_dataset: tf.data.Dataset, 
#     val_dataset: tf.data.Dataset = None, early_stopping: bool=True,
#     multiple_ts_outputs = True, **kwargs):
    
#     # Configuring model callbacks
#     print('Creating callbacks...')
#     if early_stopping:
#         earlypointer = EarlyStopping(
#             monitor='val_loss', min_delta=kwargs.get('early_stop_min_delta', 0.00001),
#             patience=kwargs.get('early_stop_patience', 5), verbose=1
#             )
#         callbacks = [earlypointer]
#     else:
#         callcacks = []

#     ckpt_callback = ModelCheckpoint(
#         filepath=ckpt_filepath, verbose=1, monitor='val_loss', save_best_only=True
#     )
#     callbacks.append(ckpt_callback)

#     # log_dir = "..//Results//logs//fit//" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     # callbacks.append(tensorboard_callback)

#     # Compiling the model
#     print('Compiling model...')
#     if multiple_ts_outputs:
#         model.compile(
#             optimizer=Adam(learning_rate=kwargs.get('lr',0.001)),
#             metrics=[last_timestep_mae,last_timestep_mse],
#             loss=tf.keras.losses.MeanSquaredError()
#         )
#     else:
#         model.compile(
#             optimizer=Adam(learning_rate=kwargs.get('lr',0.001)),
#             metrics=[tf.keras.losses.MeanAbsoluteError()],
#             loss=tf.keras.losses.MeanSquaredError()
#         )

#     # Fitting the model
#     print('Fitting model...')
#     hist = model.fit(train_dataset, callbacks=callbacks,
#                      validation_data=val_dataset, verbose=1, **kwargs)

#     return model, hist

# def optuna_dense_model_v0(trial: optuna.trial.Trial):

#     fold_json_path = '..//Data//folds.json'
#     ckpt_filepath = '..//Results//tmp//best_model.hdf5'
#     dic_split, scaler = gen_dataset(fold_json_path=fold_json_path, fold='3',
#                                     time_col='index', scale_flg=True,
#                                     num_sm_split={
#                                         'train':NUM_SM_TRAIN, 
#                                         'val':NUM_SM_VAL, 
#                                         'test':NUM_SM_TEST},
#                                     boxcox_trnsf_flag=True,
#                                     batch_size=512)

#     ip = Input(shape=(24,4))

#     x = TimeDistributed(Dense(trial.suggest_int('1st_layer',4,20,step=4),
#                               activation='relu'))(ip)
#     # if num_dense_layers == 3:
#     #     x = TimeDistributed(Dense(trial.suggest_int('2nd_layer',4,20,step=4),
#     #                               activation='relu'))(ip)

#     x = TimeDistributed(Dense(10))(x)

#     model = Model(ip,x)

#     print(model.summary())

#     earlypointer = EarlyStopping(
#             monitor='val_loss', min_delta=0.00001,
#             patience=5, verbose=1
#             )
#     ckpt_callback = ModelCheckpoint(
#         filepath=ckpt_filepath, verbose=1, monitor='val_loss', save_best_only=True
#     )
#     callbacks = [earlypointer, ckpt_callback]

#     # Compile model
#     print('Compiling model...')
#     lr = trial.suggest_loguniform('lr', 1e-5,1e-3)
#     model.compile(
#         optimizer=Adam(learning_rate=lr),
#         metrics=[last_timestep_mae,last_timestep_mse],
#         loss=tf.keras.losses.MeanSquaredError()
#     )

#     # Fit model
#     model.fit(
#         dic_split['train'].repeat(), epochs=1000, callbacks=callbacks,
#         validation_data=dic_split['val'].repeat(),verbose=2,
#         steps_per_epoch=dic_split['train_num_batches']//10,
#         validation_steps=dic_split['val_num_batches']*.9//1
#     )

#     # Load best model weights
#     print('Loading best model for validation...')
#     best_model = keras.models.load_model(
#         ckpt_filepath,
#         custom_objects={
#             "last_timestep_mae" : last_timestep_mae,
#             "last_timestep_mse" : last_timestep_mae
#         })
#     best_model.compile(
#         optimizer=Adam(learning_rate=lr),
#         metrics=[last_timestep_mae,last_timestep_mse],
#         loss=tf.keras.losses.MeanSquaredError()
#     )

#     # Evaluate and return loss for test dataset
#     scores = best_model.evaluate(
#         dic_split['test'], steps=dic_split['test_num_batches']*.9//1)
#     return scores[0]


# def optuna_dense_model_v1(trial: optuna.trial.Trial):

#     fold_json_path = '..//Data//folds.json'
#     ckpt_filepath = '..//Results//tmp//best_model.hdf5'
#     dic_split, scaler = gen_dataset(
#         final_data_path='..//Data//acorn_{}_preproc_data.parquet.gzip',
#         fold_json_path=fold_json_path, fold='3',
#         time_col='index', scale_flg=True,
#         num_sm_split={
#             'train':NUM_SM_TRAIN, 
#             'val':NUM_SM_VAL, 
#             'test':NUM_SM_TEST},
#         boxcox_trnsf_flag=True,
#         batch_size=256,
#         test_gen_dataset_flg=True)

#     # All suggested values
#     units = trial.suggest_int('1st_layer',4,20,step=4)
#     lr = trial.suggest_loguniform('lr', 1e-4,1e-3, 4.9e-4)

#     ip = Input(shape=(24,4))

#     x = TimeDistributed(Dense(units, activation='relu'))(ip)
#     x = Flatten()(x)
#     x = Dense(10)(x)

#     model = Model(ip,x)

#     print(model.summary())

#     earlypointer = EarlyStopping(
#             monitor='val_loss', min_delta=0.00001,
#             patience=5, verbose=1
#             )
#     ckpt_callback = ModelCheckpoint(
#         filepath=ckpt_filepath, verbose=1, monitor='val_loss', save_best_only=True
#     )
#     callbacks = [earlypointer, ckpt_callback]

#     # Compile model
#     print('Compiling model...')
#     model.compile(
#         optimizer=Adam(learning_rate=lr),
#         metrics=[tf.keras.losses.MeanAbsoluteError()],
#         loss=tf.keras.losses.MeanSquaredError()
#     )

#     # Fit model
#     model.fit(
#         dic_split['train'].cache().repeat().prefetch(tf.data.AUTOTUNE), 
#         epochs=1000, callbacks=callbacks,
#         validation_data=dic_split['val'].repeat().prefetch(tf.data.AUTOTUNE),
#         verbose=2,
#         steps_per_epoch=dic_split['train_num_batches']//10,
#         validation_steps=dic_split['val_num_batches']*.9//1
#     )

#     # Load best model weights
#     print('Loading best model for validation...')
    
#     model.load_weights(ckpt_filepath)

#     # Evaluate and return loss for test dataset
#     scores = model.evaluate(
#         dic_split['test'], steps=dic_split['test_num_batches']*.9//1)
#     return scores[0]


def main_optuna(study_name: str, n_trials: int):
    # Select correct model to be tunned 
    dic_models = {
        'dense_many2many_v0' : optuna_dense_many2many_v0,
        'dense_many2one_v0' : optuna_dense_many2one_v0,
        'dense_many2one_v1' : optuna_dense_many2one_v1,
        'mlp_mixer_many2one_v0' : optuna_mlp_mixer_many2one_v0,
        'lstm_many2one_v0' : optuna_lstm_many2one_v0,
        'lstm_many2one_v1' : optuna_lstm_many2one_v1,
        'lstm_many2one_v2' : optuna_lstm_many2one_v2,
        'conv1d_lstm_many2one_v0' : optuna_conv1d_lstm_many2one_v0,
        'conv1d_lstm_many2one_v1' : optuna_conv1d_lstm_many2one_v1,
        'conv1d_lstm_many2one_v2' : optuna_conv1d_lstm_many2one_v2,
        'conv1d_lstm_many2one_v3' : optuna_conv1d_lstm_many2one_v3,
        'wavenet_many2one_v0' : optuna_wavenet_many2one_v0,
        'wavenet_many2one_v1' : optuna_wavenet_many2one_v1,
        'wavenet_mlp_mixer_many2one_v0' : optuna_wavenet_mlp_mixer_many2one_v0,
        'wavenet_mlp_mixer_many2one_v1' : optuna_wavenet_mlp_mixer_many2one_v1
    }

    # Load or create study
    save_file_path = '..//Results//'+study_name+'.pkl'
    if os.path.exists(save_file_path):
        study = joblib.load(save_file_path)
    else:
        study = optuna.create_study(
            direction='minimize', study_name=study_name, load_if_exists=True)

    lefting_trials = n_trials - len(study.trials)
    lefting_trials = 0 if lefting_trials < 0 else lefting_trials
    for _ in range(lefting_trials):
        study.optimize(
            dic_models[study_name], n_trials=1, show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        joblib.dump(study, save_file_path)

# def main_test():
#     fold_json_path = '..//Data//folds.json'
#     dic_split, scaler = gen_dataset(fold_json_path=fold_json_path, fold='3',
#                                     time_col='index', scale_flg=True,
#                                     num_sm_split={
#                                         'train':NUM_SM_TRAIN, 
#                                         'val':NUM_SM_VAL, 
#                                         'test':NUM_SM_TEST},
#                                     boxcox_trnsf_flag=True,
#                                     batch_size=256,
#                                     test_gen_dataset_flg=True)
#     print(dic_split['train_num_batches'])
#     model = gen_mlp_mixer_model_v0(
#         input_shape=(24,4), patch_size=4, print_summary=True)
#     model, hist = compile_and_fit(model, TEST_CKPT_PATH, epochs=1000,
#                                   train_dataset=dic_split['train'].repeat(),
#                                   val_dataset=dic_split['val'].repeat(),
#                                   steps_per_epoch=dic_split['train_num_batches']//10,
#                                   validation_steps=dic_split['val_num_batches']*.9//1,
#                                   multiple_ts_outputs=False)
#     return model, hist

# global NUM_SM_TRAIN, NUM_SM_VAL, NUM_SM_TEST, TEST_CKPT_PATH

# NUM_SM_TRAIN, NUM_SM_VAL, NUM_SM_TEST = 200, 50, 50
# TEST_CKPT_PATH = '..//Results//mlp_mixer_v0.hdf5'

#%%
if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 3, 'Correct arguments passed to python execution.'
    main_optuna(
        study_name=args[1],
        n_trials=int(args[2]))
    # main_test()


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