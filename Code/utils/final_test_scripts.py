import json
from numpy import vstack
from scipy.special import inv_boxcox

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, Dropout, Flatten, Bidirectional,
    TimeDistributed, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from utils.data_prep import gen_dataset

DATASET_PARAMS = {
    'fold_json_path' : '..//Data//folds.json',
    'ckpt_filepath' : '..//Results//tmp//best_model.hdf5',
    'fold' : '3', 'time_col' : 'index', 'scale_flg' : True,
    'num_sm_split' : {
        'train':200, 
        'val':50, 
        'test':50
    },
    'boxcox_trnsf_flag' : True, 'batch_size' : 16
}

def conv1d_lstm_v3_m2o_final_test(
    ckpt_filepath: str = '..//Results//best_con1d_lstm_v3_model.hdf5',
    folds_num : list = ["0", "1", "2", "3"],
    boxcox_lmbdas_per_fold_json_path: str='..//Data//boxcox_lmbdas_per_fold.json',
    result_json_path: str = '..//Results//results_per_fold.json'
):
    
    with open(boxcox_lmbdas_per_fold_json_path, 'r') as f:
        boxcox_lmbdas = json.load(f)

    dic_results = {}

    DATASET_PARAMS['test_gen_dataset_flg'] = True
    DATASET_PARAMS['batch_size'] = 8
    for fold in folds_num:
        DATASET_PARAMS['fold'] = fold
        dic_split, scaler = gen_dataset(**DATASET_PARAMS)

        # All suggested values
        dropout_rate = 0.3
        filters_num = 8
        first_kernel_size = 3
        secnd_kernel_size = 2
        stride_size = 3
        units_1st = 16
        units_2nd = 8
        lr = 5e-4
        input_shape = (24,4)

        # Create model
        ip = Input(shape=input_shape)
        x = Conv1D(
            filters=filters_num, kernel_size=first_kernel_size, 
            strides=stride_size)(ip)
        x = Conv1D(
            filters=filters_num, kernel_size=secnd_kernel_size, strides=1)(x)
        x = LSTM(units_1st, return_sequences=True)(x)
        x = LSTM(units_2nd, return_sequences=False)(x)
        x = Dense(20)(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(10)(x)

        model = Model(ip,x)

        earlypointer = EarlyStopping(
                monitor='val_loss', min_delta=0.00001,
                patience=5, verbose=1
                )
        ckpt_callback = ModelCheckpoint(
            filepath=ckpt_filepath, verbose=1, 
            monitor='val_loss', save_best_only=True
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
            epochs=1000, callbacks=callbacks,
            validation_data=dic_split['val'].repeat().prefetch(tf.data.AUTOTUNE),
            verbose=2,
            steps_per_epoch=dic_split['train_num_batches']//10,
            validation_steps=dic_split['val_num_batches']*.9//1
        )

        # Load best model weights
        print('Loading best model for validation...')
        
        model.load_weights(ckpt_filepath)

        # Make prediction
        y_true_stacked = None
        y_pred_stacked = None
        for x,y in dic_split['test']:
            if y_true_stacked is not None:
                y_true_stacked = vstack([y_true_stacked, y])
                y_pred_stacked = vstack([y_pred_stacked,
                                        model.predict(x)])
            else:
                y_true_stacked = y
                y_pred_stacked = model.predict(x)

        # Inverse transform values
        y_pred_stacked = y_pred_stacked*scaler.scale_ + scaler.mean_
        y_pred_stacked = inv_boxcox(y_pred_stacked, boxcox_lmbdas[fold])
        
        y_true_stacked = y_true_stacked*scaler.scale_ + scaler.mean_
        y_true_stacked = inv_boxcox(y_true_stacked, boxcox_lmbdas[fold])

        # Compare values
        mse = MeanSquaredError()(y_true_stacked, y_pred_stacked).numpy()

        dic_results_fold = {
            'train_mean' : scaler.mean_,
            'train_scale' : scaler.scale_,
            'train_var' : scaler.var_,
            'mse' : mse
        }

        dic_results[fold] = dic_results_fold

    with open(result_json_path, 'w') as f:
        json.dump(dic_results, f, indent=4)
