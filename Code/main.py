#%%
import pandas as pd
from os.path import normpath, join
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

if __name__ == '__main__':
    block_file_name = '..//Data//halfhourly_dataset//block_0.csv'
    weather_file_name = '..//Data//weather_hourly_darksky_final.csv'
    time_col_data = 'tstp'
    time_col_weather = 'time'
    desired_weather_cols = ['apparentTemperature', 'weekly_temperature'] #uvIndex

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

    # TODO: Splitar os dados em input e target
    # TODO: Splitar os dados em treino teste
    # TODO: Alimentar e treinar modelo
