#%%
import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Dropout, Flatten, Bidirectional,
    TimeDistributed, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.merge import concatenate

from utils.data_prep import gen_dataset
from utils.metrics import last_timestep_mae, last_timestep_mse
from utils.models import (
    Patches, MLPMixerLayer, wavenet_residual_block)

DATASET_PARAMS = {
    'fold_json_path' : '..//Data//folds.json',
    'ckpt_filepath' : '..//Results//tmp//best_model.hdf5',
    'fold' : '3', 'time_col' : 'index', 'scale_flg' : True,
    'num_sm_split' : {
        'train':200, 
        'val':50, 
        'test':50
    },
    'boxcox_trnsf_flag' : True, 'batch_size' : 256
}

def evaluate_model(
    model: Model, dic_split: dict, lr: float, 
    ckpt_filepath: str, epochs: int = 1000):

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
    print('Fitting model...')
    model.fit(
        dic_split['train'].repeat().prefetch(tf.data.AUTOTUNE), 
        epochs=epochs, callbacks=callbacks,
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

def optuna_dense_many2many_v0(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = False
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    units_1st = trial.suggest_int('1st_layer',4,20,step=4)
    lr = trial.suggest_loguniform('lr', 1e-5,1e-3)

    ip = Input(shape=(24,4))

    x = TimeDistributed(Dense(units_1st,
                              activation='relu'))(ip)
    # if num_dense_layers == 3:
    #     x = TimeDistributed(Dense(trial.suggest_int('2nd_layer',4,20,step=4),
    #                               activation='relu'))(ip)

    x = TimeDistributed(Dense(10))(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr,
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_dense_many2one_v0(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    units = trial.suggest_int('1st_layer',4,20,step=4)
    lr = trial.suggest_float('lr', 1e-4,1e-3, step=4.9e-4)

    ip = Input(shape=(24,4))

    x = TimeDistributed(Dense(units, activation='relu'))(ip)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr,
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_dense_many2one_v1(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    units_1st = trial.suggest_int('1st_layer',4,20,step=4)
    units_2nd = trial.suggest_int('2nd_layer',4,20,step=4)
    lr = trial.suggest_float('lr', 1e-4,1e-3, step=4.9e-4)

    ip = Input(shape=(24,4))

    x = TimeDistributed(Dense(units_1st, activation='relu'))(ip)
    x = TimeDistributed(Dense(units_2nd, activation='relu'))(x)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())
    
    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr,
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_mlp_mixer_many2one_v0(trial: optuna.trial.Trial):
    
    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    num_blocks = trial.suggest_int('num_blocks',1,6,step=1)
    dropout_rate = trial.suggest_categorical('dropout_rate',[.0, .3, .5, .8])
    lr = trial.suggest_float('lr', 1e-4,1e-3, step=4.9e-4)
    input_shape = (24,4)
    patch_size = 4
    num_patches = input_shape[0] // patch_size
    hidden_units = patch_size**2

    ip = Input(shape=input_shape)

    x = Patches(patch_size=patch_size, num_patches=num_patches)(ip)
    for block in range(num_blocks):
        x = MLPMixerLayer(
            num_patches=num_patches, hidden_units=hidden_units, 
            dropout_rate=dropout_rate, name=f'MLP_Mixer_Layer_{block}')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr,
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_lstm_many2one_v0(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    units_1st = trial.suggest_int('1st_layer',2,20,step=2)
    lr = trial.suggest_float('lr', 1e-4,1e-3, step=4.9e-4)
    input_shape = (24,4)

    # Create model
    ip = Input(shape=input_shape)

    x = LSTM(units_1st, return_sequences=False)(ip)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr, 
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_lstm_many2one_v1(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    units_1st = trial.suggest_int('1st_layer',2,20,step=2)
    units_2nd = trial.suggest_int('2nd_layer', 2, 20, step=4)
    lr = trial.suggest_float('lr', 1e-4,1e-3, step=4.9e-4)
    input_shape = (24,4)

    # Create model
    ip = Input(shape=input_shape)

    x = LSTM(units_1st, return_sequences=True)(ip)
    x = LSTM(units_2nd, return_sequences=False)(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr, 
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_conv1d_lstm_many2one_v0(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    units_1st = trial.suggest_int('1st_layer',4,16,step=4)
    filters_num = trial.suggest_categorical('filters_num', [4, 8, 16])
    kernel_size = trial.suggest_categorical('kernel_size', [3,5,7])
    stride_size = trial.suggest_categorical('stride_size', [1, 3])
    lr = 7e-4
    input_shape = (24,4)

    # Create model
    ip = Input(shape=input_shape)

    x = Conv1D(
        filters=filters_num, kernel_size=kernel_size, strides=stride_size)(ip)
    x = LSTM(units_1st, return_sequences=False)(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr, 
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )

def optuna_wavenet_many2one_v0(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    wn_num_filters = trial.suggest_categorical('num_filters',[4, 8, 16, 32])
    wn_num_layers = trial.suggest_categorical('num_layers', [2,3,4,5])
    wn_num_blocks = trial.suggest_categorical('num_blocks', [1,2,4,8])
    lr = 5e-4
    input_shape = (24,4)

    # Create model
    ip = Input(shape=input_shape)

    x = Conv1D(wn_num_filters, kernel_size=2, padding='causal')(ip)
    skip_to_last = []
    for dilatation_rate in [2**i for i in range(wn_num_layers)] * wn_num_blocks:
        x, skip = wavenet_residual_block(x, wn_num_filters, dilatation_rate)
        skip = GlobalAveragePooling1D(data_format='channels_last')(skip)
        skip = Reshape((1,-1))(skip)
        skip_to_last.append(skip)
    x = concatenate(skip_to_last, axis=1)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr, 
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )


def optuna_wavenet_many2one_v1(trial: optuna.trial.Trial):

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    dic_split, scaler = gen_dataset(**DATASET_PARAMS)

    # All suggested values
    wn_num_filters = trial.suggest_categorical('num_filters',[4, 8, 16, 32])
    wn_num_layers = trial.suggest_categorical('num_layers', [2,3,4,5])
    wn_num_blocks = trial.suggest_categorical('num_blocks', [1,2,4,8])
    lr = 5e-4
    input_shape = (24,4)

    # Create model
    ip = Input(shape=input_shape)

    x = Conv1D(wn_num_filters, kernel_size=2, padding='causal')(ip)
    skip_to_last = []
    for dilatation_rate in [2**i for i in range(wn_num_layers)] * wn_num_blocks:
        x, skip = wavenet_residual_block(x, wn_num_filters, dilatation_rate)
        skip = GlobalAveragePooling1D(data_format='channels_last')(skip)
        skip = Reshape((-1,1))(skip)
        skip_to_last.append(skip)
    x = concatenate(skip_to_last, axis=2)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())

    return evaluate_model(
        model=model, dic_split=dic_split, lr=lr, 
        ckpt_filepath=DATASET_PARAMS.get('ckpt_filepath')
    )


def wavenet():
    # All suggested values
    wn_num_filters = 32
    wn_num_layers = 3
    wn_num_blocks = 2
    lr = 5e-4
    input_shape = (24,4)

    # Create model
    ip = Input(shape=input_shape)

    x = Conv1D(wn_num_filters, kernel_size=2, padding='causal')(ip)
    skip_to_last = []
    for dilatation_rate in [2**i for i in range(wn_num_layers)] * wn_num_blocks:
        x, skip = wavenet_residual_block(x, wn_num_filters, dilatation_rate)
        skip = GlobalAveragePooling1D(data_format='channels_last')(skip)
        skip = Reshape((1,-1))(skip)
        skip_to_last.append(skip)
    x = concatenate(skip_to_last, axis=1)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(ip,x)

    print(model.summary())
