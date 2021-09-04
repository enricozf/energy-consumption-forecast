#%%
import pandas as pd
from numpy import ndarray
from json import dump, load
from numpy.random import shuffle
from os.path import normpath, join
import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import Assert
from utils.exploratory_data_analysis import (read_acorn_group_blocks, 
                                             split_into_acorns)

def create_df(
    consumption_file_path : str ='..//Data//halfhourly_dataset',
    sm_info_file : str = '..//Data//informations_households.csv',
    desired_acorn_name : str = 'Affluent'):
    
    # Get all SM id from desired acorn and remove ToU SMs
    sm_info_df = pd.read_csv(sm_info_file, header=0)
    sm_info_df = sm_info_df[sm_info_df['stdorToU']=='Std']
    sm_info_df = sm_info_df[sm_info_df['Acorn_grouped']==desired_acorn_name]
    desired_blocks = list(set(sm_info_df['file'].values))
    
    # Read hourly data from desired acorn
    data_df = pd.DataFrame()
    for block_file in desired_blocks:
        print(f'Loading {block_file} file...')
        file_path = normpath(join(consumption_file_path, block_file + '.csv'))
        aux_df = pd.read_csv(file_path, header=0).set_index('LCLid')
        # print('--- Shape: {}'.format(aux_df.shape))
        aux_df = aux_df.loc[sm_info_df[sm_info_df['file']==block_file]\
                            ['LCLid'].values]
        # print('--- Shape: {}'.format(aux_df.shape))
        data_df = pd.concat([data_df, aux_df.reset_index()], axis=0)
    print('All datablock files were read.')

    return data_df

def monday_or_weekend(elem):
    return 1 if elem.weekday() in [0,5,6] else 0

def create_complete_df(
    block_file_name:str = '..//Data//halfhourly_dataset//block_0.csv',
    weather_file_name:str = '..//Data//weather_hourly_darksky_final.csv',
    time_col_data:str = 'tstp',
    time_col_weather:str = 'time',
    desired_weather_cols:list = ['apparentTemperature', 'weekly_temperature'],
    save_df_path:bool = True,
    num_folds:int = 0): #uvIndex
    
    # TODO: Merge + fill NaN
    df = create_df()
    df.head()
    print('Shape: ', df.shape)
    df[time_col_data] = pd.to_datetime(df[time_col_data])
    df['MondayORWE'] = df[time_col_data].apply(monday_or_weekend)
    # df['weekday'] = df[time_col_data].apply(lambda x: x.weekday())
    df.set_index(time_col_data, inplace=True)

    # Create weather df
    weather_df = pd.read_csv(weather_file_name, header=0)
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df.set_index(time_col_weather, inplace=True)

    # Include weather features in main DataFrame
    for w_col in desired_weather_cols:
        weather_sr = pd.Series(
            data=weather_df[w_col].rolling(2).mean()[1:].values,
            index=(weather_df.index[:-1]+pd.Timedelta(30, unit='min')).values
        )
        weather_sr = pd.concat([weather_df[w_col], weather_sr]).sort_index()
        # Insert weather data to main data
        df = pd.merge(df, 
                      pd.DataFrame(weather_sr).rename(columns={0:w_col}),
                      how='inner', left_index=True, right_index=True)
    
    df.reset_index(inplace=True)
    if save_df_path:
        df.to_csv('..//Data//final_data.csv', index=False)


    return df

def separate_sm_in_folds(
    lcl_ids: list, 
    fold_json_path: str, 
    n_folds: int =4):
    fold_step = int(len(lcl_ids)/n_folds//1)
    # print()
    test_step = int(fold_step/5//1)
    shuffle(lcl_ids)

    dic_folds = {str(fold) : lcl_ids[fold*fold_step:(fold+1)*fold_step] \
                 for fold in range(n_folds)}
    plus_test_sms = lcl_ids[fold_step*n_folds:]

    folds_json = {}

    for fold in range(n_folds):
        fold = str(fold)
        folds_json[fold] = {k : [] for k in ['train','val','test']}
        for fold_group in dic_folds.keys():
            if fold == fold_group:
                folds_json[fold]['val'].extend(dic_folds[fold_group][test_step:])
                folds_json[fold]['test'].extend(dic_folds[fold_group][:test_step])
                folds_json[fold]['test'].extend(plus_test_sms)
            else:
                folds_json[fold]['train'].extend(dic_folds[fold_group])

    with open(fold_json_path, 'w') as f:
        dump(folds_json, f)

    # return folds_json

def gen_dataset_obj(
    df_values: ndarray,
    sequence_length: int = 24,
    pred_samples: int = 10,
    batch_size: int = 64):
    
    dataset = tf.data.Dataset.from_tensor_slices(df_values)
    dataset = dataset.window(sequence_length+pred_samples, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length+pred_samples))
    dataset = dataset.batch(batch_size, drop_remainder=True)#.shuffle(10000)
    dataset = dataset.map(lambda window: (window[:, :sequence_length, :], window[:, sequence_length:, 0]))

    return dataset

def gen_dataset(
    df: pd.DataFrame,
    dic_folds: dict,
    fold: str = '0',
    fold_split: str = 'train',
    **kwargs):
    # # Verify sequence_length is in passed arguments
    # sequence_length = kwargs.get('sequence_length')
    # try:
    #     assert sequence_length
    # except AssertionError:
    #     raise ValueError("Missing 'sequence_length' parameter.")

    lst_sms = dic_folds[fold][fold_split].copy()
    print('Tamanho lista de sms: ', len(lst_sms))
    first_sm = lst_sms.pop(0)

    df_sm = df.loc[first_sm].copy()
    df_sm = df_sm.reset_index().drop('LCLid', axis=1).set_index('tstp').values
    dataset = gen_dataset_obj(df_sm, **kwargs)

    # TODO: Nem o dataset nem o dataset_aux sao atualizados neste for. 
    #       Será um problema do MapDataset, por ser uma classe diferente?
    for sm in lst_sms:
        df_sm = df.loc[sm].copy()
        # print('--- Tamanho do DF: ', df_sm.shape)
        df_sm = df_sm.set_index('tstp').sort_index().values
        dataset_aux = gen_dataset_obj(df_sm, **kwargs)
        dataset = dataset.concatenate(dataset_aux)

    return dataset.shuffle(int(5e6))#.map(lambda window: (window[:, :sequence_length, :], window[:, sequence_length:, 0]))


if __name__ == '__main__':
    final_data_path = '..//Data//final_data.csv'
    fold_json_path = '..//Data//folds.json'
    desired_fold = '0'
    # TODO: Splitar os dados em input e target
    df = pd.read_csv(final_data_path)
    # df.drop('Unnamed: 0', axis=1, inplace=True)

    with open(fold_json_path, 'r') as f:
        dic_folds = load(f)

# AQUI
    df.sort_values(['LCLid','tstp'],inplace=True)
    df.set_index('LCLid', inplace=True)
    # df_test = df.loc[dic_folds['0']['train'][0]]
    # df_test_test = df_test.head(100)
    # df_values = df_test_test.reset_index().drop('LCLid',axis=1).set_index('tstp').values

    dataset = gen_dataset(df, dic_folds, fold_split='test')
    # lst_lcl_ids = list(df['LCLid'].unique())
    # separate_sm_in_folds(lst_lcl_ids, fold_json_path)
    # TODO: Splitar os dados em treino teste
    # TODO: Alimentar e treinar modelo

# %%
